"""How much does the ATC classifier depend on the full structured input?

The model was trained on 'Medicamento | Dosis | Forma farmacéutica | Principios activos | Vía de administración'.
A public demo lets people type only a name or an ingredient, so measure accuracy on the held-out split when
fields are dropped. Writes eval/atc_input_ablation.json.

Usage: python scripts/eval_atc_inputs.py
"""
from __future__ import annotations

import json
import ssl
import sys
from pathlib import Path

if sys.platform == "win32":
    _orig = ssl.SSLContext._load_windows_store_certs  # type: ignore[attr-defined]

    def _tolerant(self, storename, purpose):  # noqa: ANN001
        try:
            return _orig(self, storename, purpose)
        except ssl.SSLError:
            return None

    ssl.SSLContext._load_windows_store_certs = _tolerant  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.atc_split import grouped_split  # noqa: E402
from src.config import ATC_MODEL_DIR, ATC_SEED, PROCESSED_DATA_DIR  # noqa: E402


def build(row, fields: tuple[str, ...]) -> str:
    parts = []
    if "name" in fields:
        parts.append(f"Medicamento: {row['name']}")
    if "form" in fields and row["pharmaceutical_form_name"]:
        parts.append(f"Forma farmacéutica: {row['pharmaceutical_form_name']}")
    if "ing" in fields and row["active_ingredients_str"]:
        parts.append(f"Principios activos: {row['active_ingredients_str']}")
    return " | ".join(parts)


def main() -> None:
    model_dir = Path(ATC_MODEL_DIR)
    df = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "atc_dataset.parquet")
    label2id = json.loads((model_dir / "label_mapping.json").read_text(encoding="utf-8"))["label2id"]
    df["label"] = df["atc_level1"].map(label2id)
    df = df.dropna(subset=["label", "text_input"])
    df["label"] = df["label"].astype(int)
    for column in ("pharmaceutical_form_name", "active_ingredients_str"):
        df[column] = df[column].fillna("")
    _, _, test = grouped_split(df, seed=ATC_SEED)
    test = test.drop_duplicates("text_input")  # one row per distinct text, as in the headline numbers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "final_model"))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "final_model")).to(device).eval()
    temperature = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))["temperature"]

    variants = {
        "full training format (all fields)": None,
        "name + form + ingredient": ("name", "form", "ing"),
        "name + ingredient": ("name", "ing"),
        "ingredient only": ("ing",),
        "name only": ("name",),
        "name + form": ("name", "form"),
    }
    results = {}
    for label, fields in variants.items():
        texts = test["text_input"].tolist() if fields is None else [build(r, fields) for r in test.to_dict("records")]
        preds, confs = [], []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                enc = tokenizer(texts[start:start + 64], truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits.float() / temperature, dim=-1).cpu().numpy()
                preds.append(probs.argmax(1))
                confs.append(probs.max(1))
        pred, conf = np.concatenate(preds), np.concatenate(confs)
        labels = test["label"].to_numpy()
        results[label] = {"n": len(test), "top1": round(float((pred == labels).mean()), 3), "mean_confidence": round(float(conf.mean()), 3)}
        print(label, results[label], flush=True)
    (ROOT / "eval" / "atc_input_ablation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
