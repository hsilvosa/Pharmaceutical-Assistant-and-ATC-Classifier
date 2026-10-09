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

repo = "your-hf-username/BETO-ATC-Classifier"
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
license: apache-2.0
tags:
- text-generation
- causal-lm
- lora
- qlora
- spanish
- medical
- pharmacology
- aemps-cima
datasets:
- hsilvosa/aemps-cima
- hsilvosa/openplacsp
pipeline_tag: text-generation
---

# CIMA Spanish Medical Llama (LoRA) (not trained yet)

> **Status: untrained placeholder.** All LoRA B matrices in this adapter are zero, so it changes nothing: the model behaves exactly like the base model `meta-llama/Llama-3.2-3B-Instruct`. No evaluation result is reported.

The intended adapter is a QLoRA fine-tune on Spanish Patient Leaflets (Prospectos) and Summaries of Product Characteristics (Fichas Técnicas) from the **AEMPS CIMA Research Dataset** (`hsilvosa/aemps-cima`), aimed at Spanish pharmaceutical question answering. Training code: `src/train_medical_llm.py`.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "your-hf-username/CIMA-Spanish-Medical-Llama-LoRA")

prompt = "<|system|>\nEres un asistente médico farmacéutico...\n<|user|>\n¿Cuál es la vía de administración y composición de Omeprazol?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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

    # 2. Export CIMA Medical LLM LoRA Adapter Weights & Config
    llm_dir = Path(LLM_MODEL_DIR)
    llm_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting CIMA Medical LLM LoRA weights & adapter files to '{llm_dir}'...")
    
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": LLM_BASE_MODEL,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "task_type": "CAUSAL_LM"
    }
    
    with open(llm_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    # Generate and save PEFT LoRA adapter weight tensors (adapter_model.safetensors)
    lora_tensors = {}
    r = 16
    hidden_dim = 3072
    num_layers = 28
    
    for i in range(num_layers):
        for module in ["q_proj", "v_proj", "k_proj", "o_proj"]:
            # LoRA A matrix: (r, hidden_dim) initialized gaussian
            lora_A = torch.randn((r, hidden_dim), dtype=torch.float32) * 0.02
            # LoRA B matrix: (hidden_dim, r) initialized zero
            lora_B = torch.zeros((hidden_dim, r), dtype=torch.float32)
            
            lora_tensors[f"base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight"] = lora_A
            lora_tensors[f"base_model.model.model.layers.{i}.self_attn.{module}.lora_B.weight"] = lora_B

    safetensors_path = llm_dir / "adapter_model.safetensors"
    save_file(lora_tensors, str(safetensors_path))
    logger.info(f"Generated and saved '{safetensors_path}' ({safetensors_path.stat().st_size / 1e6:.2f} MB)")

    with open(llm_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(LLM_MODEL_CARD.strip())
        
    logger.info(f"Medical LLM export complete at '{llm_dir}'")

def upload_to_huggingface(repo_id: str, model_type: str = "atc", token: str = None):
    """Upload complete model folder to Hugging Face Hub."""
    from huggingface_hub import HfApi
    
    api = HfApi(token=token or os.getenv("HF_TOKEN"))
    folder_path = Path(ATC_MODEL_DIR) / ATC_EXPORT_DIRNAME if model_type == "atc" else Path(LLM_MODEL_DIR)
    
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
