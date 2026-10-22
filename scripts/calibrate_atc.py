"""Measure and fix the confidence calibration of the ATC level-1 classifier.

Fits a single temperature on the validation split and reports ECE, NLL and Brier score on the test split
before and after. Both splits are grouped by active ingredient (src/atc_split.py), as in training.

Usage: python scripts/calibrate_atc.py   (writes models/atc_classifier/calibration.json)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.atc_split import grouped_split  # noqa: E402
from src.config import ATC_MODEL_DIR, ATC_SEED, PROCESSED_DATA_DIR  # noqa: E402


def logits_for(model, tokenizer, texts, device, batch=64):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            enc = tokenizer(list(texts[start:start + batch]), truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def report(logits: torch.Tensor, labels: np.ndarray, temperature: float) -> dict:
    probs = torch.softmax(logits / temperature, dim=-1).numpy()
    onehot = np.eye(probs.shape[1])[labels]
    return {
        "accuracy": round(float((probs.argmax(1) == labels).mean()), 4),
        "ece": round(ece(probs, labels), 4),
        "nll": round(float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()), 4),
        "brier": round(float(((probs - onehot) ** 2).sum(1).mean()), 4),
        "mean_confidence": round(float(probs.max(1).mean()), 4),
    }


def main() -> None:
    model_dir = Path(ATC_MODEL_DIR)
    df = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "atc_dataset.parquet")
    mapping = json.loads((model_dir / "label_mapping.json").read_text(encoding="utf-8"))
    label2id = mapping["label2id"]
    df["label"] = df["atc_level1"].map(label2id)
    df = df.dropna(subset=["label", "text_input"])
    df["label"] = df["label"].astype(int)
    _, val_df, test_df = grouped_split(df, seed=ATC_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "final_model"))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "final_model")).to(device)
    val_logits = logits_for(model, tokenizer, val_df["text_input"].tolist(), device)
    test_logits = logits_for(model, tokenizer, test_df["text_input"].tolist(), device)
    val_labels, test_labels = val_df["label"].to_numpy(), test_df["label"].to_numpy()

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    target = torch.tensor(val_labels)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(val_logits / log_t.exp(), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    result = {
        "temperature": round(temperature, 4),
        "fitted_on": f"validation split ({len(val_df)} rows)",
        "evaluated_on": f"test split ({len(test_df)} rows, {test_df['label'].nunique()} classes present)",
        "test_before": report(test_logits, test_labels, 1.0),
        "test_after": report(test_logits, test_labels, temperature),
        "validation_before": report(val_logits, val_labels, 1.0),
        "validation_after": report(val_logits, val_labels, temperature),
        "note": "Single temperature scaling; accuracy and ranking do not change. One seed.",
    }
    (model_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
