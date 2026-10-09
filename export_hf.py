"""
Hugging Face Export & Automated Hub Upload Script.
Prepares model weights, configurations, tokenizers, PEFT LoRA adapters, model cards,
and provides automated Hugging Face Hub upload functions.
"""
from src.utils import setup_environment, setup_logger, get_device
setup_environment()

import os
import argparse
import json
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import save_file
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ATC_MODEL_DIR,
    LLM_MODEL_DIR,
    ATC_BASE_MODEL,
    LLM_BASE_MODEL,
    LLM_FALLBACK_MODEL,
    ATC_LEVEL_1_MAP
)

logger = setup_logger("export_hf")

ATC_MODEL_CARD = """---
language:
- es
license: apache-2.0
base_model: dccuchile/bert-base-spanish-wwm-cased
tags:
- text-classification
- spanish
- medical
- pharmacology
- atc-code
- aemps-cima
datasets:
- hsilvosa/aemps-cima
pipeline_tag: text-classification
---

# BETO ATC Level-1 Classifier

Fine-tuned [`dccuchile/bert-base-spanish-wwm-cased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) (BETO) that maps a Spanish drug description (name, dose, dosage form, active ingredients, route) to its ATC level-1 anatomical group (single-label, 14 classes). Levels 2-5 are not modelled. Training code: `src/train_atc_classifier.py`.

## Evaluation

Held-out test set: {num_eval_samples} rows ({num_unique_texts} unique texts), split by active-ingredient set. No ingredient combination in the test set appears in training.

| Metric | Value |
|---|---|
| Top-1 accuracy | {top1_accuracy:.1%} |
| Top-3 accuracy | {top3_accuracy:.1%} |
| Macro F1 | {macro_f1:.3f} |

Limits:
- One split seed was run, so the variation between seeds is unknown.
- Class P (antiparasitics, 143 rows in total) has no rows in the test set and is not measured.
- The dataset has one row per medicine presentation, so many rows repeat. A random row split gives about 99.9% accuracy because almost every test text also appears in training. Do not read that number as generalisation.
- The model predicts the ATC group from the description. It does not replace the official AEMPS classification.

## Intended Use & Disclaimer

Intended only for medical research and pharmaceutical data analytics. Not for clinical prescribing or medical advice.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "hsilvosa/BETO-ATC-Classifier"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)

text = "Medicamento: Omeprazol 20 mg | Forma farmacéutica: COMPRIMIDO | Principios activos: Omeprazol"
inputs = tokenizer(text, return_tensors="pt")
probs = torch.softmax(model(**inputs).logits, dim=-1)
print(probs)
```
"""

LLM_MODEL_CARD = """---
language:
- es
license: other
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
tags:
- lora
- qlora
- spanish
- pharmacology
- aemps-cima
datasets:
- hsilvosa/aemps-cima
pipeline_tag: text-generation
---

# CIMA Spanish Pharmaceutical QA LoRA Adapter (format adapter, not a knowledge source)

QLoRA adapter (r=16, 4-bit NF4) for [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), trained for one epoch on 15,000 templated question-answer pairs built from AEMPS CIMA catalogue fields. Use is subject to the base model licence (Qwen research licence). Training code: `src/train_medical_llm.py`.

## What it does

The three question templates ask about composition, dosage form and route, and excipients of a named medicine. The adapter learns the answer format and common name-to-ingredient associations. It does not learn reliable facts about medicines.

## Evaluation

Held-out set: the first {n_heldout_rows} rows (about 100 medicines) of a 300-medicine split never used in training. The prompt contains only the medicine name. A field counts as correct when the generated answer contains the reference catalogue value ({n_fields} fields).

| Field | Base model | With adapter |
|---|---|---|
| Overall | {base_overall:.1%} | {tuned_overall:.1%} |
| Active ingredient | {base_active_ingredient:.1%} | {tuned_active_ingredient:.1%} |
| Dosage form | {base_form:.1%} | {tuned_form:.1%} |
| Route | {base_route:.1%} | {tuned_route:.1%} |
| Prescription status | {base_prescription:.1%} | {tuned_prescription:.1%} |
| First excipient | {base_first_excipient:.1%} | {tuned_first_excipient:.1%} |

Limits:
- The gain is mostly format: the base model does not state prescription status or the exact catalogue wording.
- 83% of the held-out active-ingredient strings also appear in training under other brands of the same drug, so the ingredient score reflects learned name-to-ingredient associations, not knowledge of unseen drugs.
- Excipient lists and doses for a specific product are often wrong. For example, the adapter invents excipient quantities.
- One training run and one seed. No evaluation of leaflet text, safety, or free-form questions.

## Intended use & disclaimer

Research and education only. Not medical advice. Do not use it as a source of facts about medicines. Factual answers must come from retrieval with citations to official AEMPS documents (see the grounded RAG in the project repository).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="auto")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "hsilvosa/CIMA-Spanish-Medical-Qwen-LoRA")

msgs = [{{"role": "user", "content": "¿Cómo debe administrarse OMEPRAZOL CINFA 20 MG CAPSULAS y cuál es su vía de administración?"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=120)[0], skip_special_tokens=True))
```
"""

ATC_EXPORT_DIRNAME = "hf_export"


def export_trained_atc(atc_dir: Path) -> Path:
    """Copy the trained ATC classifier and a metrics-based model card to a clean staging folder.

    Refuses to export when the trained model or its evaluation report is missing,
    so untrained weights can never be packaged.
    """
    trained = atc_dir / "final_model"
    report_path = Path(PROCESSED_DATA_DIR) / "evaluation_report.json"
    if not trained.exists():
        raise FileNotFoundError(
            f"Trained ATC classifier not found at '{trained}'. Run 'python -m src.train_atc_classifier' first."
        )
    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report '{report_path}' not found. Run 'python -m src.evaluate' first."
        )
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    out = atc_dir / ATC_EXPORT_DIRNAME
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(trained, out, ignore=shutil.ignore_patterns("checkpoint-*", "training_args.bin"))
    shutil.copy(atc_dir / "label_mapping.json", out / "label_mapping.json")
    (out / "README.md").write_text(ATC_MODEL_CARD.format(**report).strip() + "\n", encoding="utf-8")
    logger.info(f"ATC Classifier export complete at '{out}'")
    return out


def export_trained_llm(llm_dir: Path) -> Path:
    """Copy the trained LoRA adapter and a metrics-based model card to a clean staging folder.

    Refuses to export when the trained adapter or its evaluation is missing.
    """
    adapter = llm_dir / "final_adapter"
    eval_path = llm_dir / "eval_final_adapter.json"
    if not adapter.exists():
        raise FileNotFoundError(
            f"Trained adapter not found at '{adapter}'. Run 'python -m src.train_medical_llm' first."
        )
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation '{eval_path}' not found. Run 'python -m src.evaluate_medical_llm' first."
        )
    with open(eval_path, "r", encoding="utf-8") as f:
        ev = json.load(f)
    values = {"n_heldout_rows": ev["n_heldout_rows"], "n_fields": ev["base_model"]["n_fields"]}
    for key in ("overall", "active_ingredient", "form", "route", "prescription", "first_excipient"):
        values[f"base_{key}"] = ev["base_model"][key]
        values[f"tuned_{key}"] = ev["fine_tuned"][key]

    out = llm_dir / ATC_EXPORT_DIRNAME
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(adapter, out, ignore=shutil.ignore_patterns("README.md", "checkpoint-*"))
    (out / "README.md").write_text(LLM_MODEL_CARD.format(**values).strip() + "\n", encoding="utf-8")
    logger.info(f"Medical LLM export complete at '{out}'")
    return out


def prepare_hf_export():
    """Package complete model artifacts (weights, adapters, configs, tokenizers, model cards) for Hugging Face Hub."""
    logger.info("Exporting complete Hugging Face model artifacts...")
    
    # 1. Export BETO ATC Classifier model & artifacts
    atc_dir = Path(ATC_MODEL_DIR)
    atc_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_path = atc_dir / "label_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            label2id = mapping["label2id"]
            id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        unique_levels = sorted(list(ATC_LEVEL_1_MAP.keys()))
        label2id = {lvl: i for i, lvl in enumerate(unique_levels)}
        id2label = {i: lvl for i, lvl in enumerate(unique_levels)}
        mapping = {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "atc_level1_map": ATC_LEVEL_1_MAP
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    export_trained_atc(atc_dir)

    # 2. Export CIMA Medical LLM LoRA adapter
    export_trained_llm(Path(LLM_MODEL_DIR))


def upload_to_huggingface(repo_id: str, model_type: str = "atc", token: str = None):
    """Upload complete model folder to Hugging Face Hub."""
    from huggingface_hub import HfApi
    
    api = HfApi(token=token or os.getenv("HF_TOKEN"))
    folder_path = (Path(ATC_MODEL_DIR) if model_type == "atc" else Path(LLM_MODEL_DIR)) / ATC_EXPORT_DIRNAME
    
    if not folder_path.exists():
        logger.error(f"Model folder '{folder_path}' does not exist.")
        return
        
    logger.info(f"Uploading '{model_type}' model from '{folder_path}' to repository '{repo_id}'...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model"
    )
    logger.info(f"Successfully uploaded model to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and Upload models to Hugging Face Hub")
    parser.add_argument("--upload", action="store_true", help="Upload trained models to HF Hub")
    parser.add_argument("--repo-id", type=str, help="Hugging Face target repository ID (e.g. username/atc-classifier)")
    parser.add_argument("--model-type", type=str, choices=["atc", "llm"], default="atc", help="Model type to upload ('atc' or 'llm')")
    parser.add_argument("--token", type=str, help="Hugging Face access token")
    
    args = parser.parse_args()
    prepare_hf_export()
    
    if args.upload:
        if not args.repo_id:
            logger.error("--repo-id is required when uploading to Hugging Face.")
        else:
            upload_to_huggingface(repo_id=args.repo_id, model_type=args.model_type, token=args.token)
