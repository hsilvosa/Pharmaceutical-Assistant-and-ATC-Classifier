"""Export the ATC classifier to ONNX (fp32 and int8) for in-browser inference and check parity with PyTorch.

Free Hugging Face Spaces only host static sites, so the demo runs the model in the browser with
Transformers.js. This script writes the files that library expects and measures what quantisation costs.

Usage: python scripts/export_atc_onnx.py [--out static-space/models/atc]
Needs torch, transformers, onnx, onnxruntime (see artifacts/space-venv).
"""
from __future__ import annotations

import argparse
import json
import shutil
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
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "models" / "atc_classifier" / "hf_export"))
    parser.add_argument("--out", default=str(ROOT / "static-space" / "models" / "atc"))
    parser.add_argument("--rows", type=int, default=600)
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(src))
    model = AutoModelForSequenceClassification.from_pretrained(str(src)).eval()
    sample = tokenizer(["Medicamento: A | Principios activos: B"], return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    fp32 = out / "onnx" / "model.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "sequence"} for k in ("input_ids", "attention_mask", "token_type_ids")} | {"logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    int8 = out / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)

    # Parity on held-out rows (same grouped split as training).
    from src.atc_split import grouped_split
    from src.config import ATC_SEED, PROCESSED_DATA_DIR

    label2id = json.loads((src / "label_mapping.json").read_text(encoding="utf-8"))["label2id"]
    df = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "atc_dataset.parquet")
    df["label"] = df["atc_level1"].map(label2id)
    df = df.dropna(subset=["label", "text_input"])
    df["label"] = df["label"].astype(int)
    _, _, test = grouped_split(df, seed=ATC_SEED)
    test = test.drop_duplicates("text_input").sample(n=args.rows, random_state=0)
    texts, labels = test["text_input"].tolist(), test["label"].to_numpy()

    sessions = {"fp32": ort.InferenceSession(str(fp32)), "int8": ort.InferenceSession(str(int8))}
    torch_pred, onnx_pred = [], {k: [] for k in sessions}
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            enc = tokenizer(texts[start:start + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
            torch_pred.append(model(**enc).logits.numpy())
            feed = {k: enc[k].numpy() for k in ("input_ids", "attention_mask", "token_type_ids")}
            for key, session in sessions.items():
                onnx_pred[key].append(session.run(["logits"], feed)[0])
    reference = np.concatenate(torch_pred)
    report = {"rows": len(texts), "torch_top1": round(float((reference.argmax(1) == labels).mean()), 4),
              "size_mb": {"fp32": round(fp32.stat().st_size / 1e6, 1), "int8": round(int8.stat().st_size / 1e6, 1)}}
    temperature = json.loads((ROOT / "space" / "calibration.json").read_text(encoding="utf-8"))["temperature"]

    def ece(logits: np.ndarray, bins: int = 15) -> float:
        z = logits / temperature
        probs = np.exp(z - z.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        confidence, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
        edges, total = np.linspace(0, 1, bins + 1), 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (confidence > low) & (confidence <= high)
            if mask.any():
                total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        return float(total)

    report["torch"] = {"ece_with_temperature": round(ece(reference), 4)}
    for key, parts in onnx_pred.items():
        logits = np.concatenate(parts)
        report[key] = {
            "top1": round(float((logits.argmax(1) == labels).mean()), 4),
            "agreement_with_torch": round(float((logits.argmax(1) == reference.argmax(1)).mean()), 4),
            "max_abs_logit_diff": round(float(np.abs(logits - reference).max()), 4),
            "ece_with_temperature": round(ece(logits), 4),
        }
    fp32.unlink()  # only the int8 file is shipped to the browser
    (ROOT / "eval" / "atc_onnx_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
