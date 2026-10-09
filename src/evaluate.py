"""
Evaluation and Metrics Benchmark Script.
Evaluates:
1. BETO ATC Hierarchical Classifier performance (Top-1/3 Accuracy, Micro/Macro F1, Confusion Matrix)
2. CIMA Spanish Medical Assistant output metrics.
"""
from src.utils import setup_environment, setup_logger, get_device
setup_environment()

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.atc_split import grouped_split
from src.config import (
    PROCESSED_DATA_DIR,
    ATC_MODEL_DIR,
    ATC_LEVEL_1_MAP,
    ATC_SEED
)

logger = setup_logger("evaluate")

def evaluate_atc_classifier():
    """Evaluate BETO ATC Classifier on processed validation dataset."""
    logger.info("Evaluating BETO ATC Classifier...")
    model_dir = Path(ATC_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "final_model"
    mapping_path = model_dir / "label_mapping.json"
    dataset_path = Path(PROCESSED_DATA_DIR) / "atc_dataset.parquet"
    
    if not dataset_path.exists():
        logger.error(f"Dataset file '{dataset_path}' not found.")
        return None

    df = pd.read_parquet(dataset_path)

    if not mapping_path.exists():
        logger.info("Creating default label mapping from dataset...")
        unique_levels = sorted(df["atc_level1"].dropna().unique())
        label2id = {lvl: i for i, lvl in enumerate(unique_levels)}
        id2label = {i: lvl for i, lvl in enumerate(unique_levels)}
        
        mapping_data = {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "atc_level1_map": ATC_LEVEL_1_MAP
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    else:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            label2id = mapping_data["label2id"]
            id2label = {int(k): v for k, v in mapping_data["id2label"].items()}
        
    df["label"] = df["atc_level1"].map(label2id)
    df = df.dropna(subset=["label", "text_input"])
    
    # Held-out test partition: active-ingredient groups never seen during training.
    df["label"] = df["label"].astype(int)
    _, _, eval_df = grouped_split(df, seed=ATC_SEED)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained ATC classifier not found at '{model_path}'. "
            "Run 'python -m src.train_atc_classifier' first. "
            "Evaluating an untrained model would produce misleading results."
        )
    logger.info(f"Loading fine-tuned model from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    device = get_device()
    model.to(device)
    model.eval()

    preds_list = []
    top3_list = []
    labels_list = list(eval_df["label"].astype(int))

    logger.info(f"Running batched evaluation on {len(eval_df)} samples...")
    batch_size = 32
    texts = list(eval_df["text_input"])
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            top1 = torch.argmax(probs, dim=-1).cpu().numpy()
            top3 = torch.topk(probs, k=min(3, probs.shape[-1]), dim=-1).indices.cpu().numpy()
            
            preds_list.extend(top1)
            top3_list.extend(top3)

    top1_acc = np.mean([p == l for p, l in zip(preds_list, labels_list)])
    top3_acc = np.mean([l in t3 for t3, l in zip(top3_list, labels_list)])
    
    precision, recall, micro_f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="micro", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="macro", zero_division=0)
    
    report_data = {
        "split": "held-out test, grouped by active-ingredient set",
        "num_eval_samples": len(eval_df),
        "num_unique_texts": int(eval_df["text_input"].nunique()),
        "top1_accuracy": round(float(top1_acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4)
    }

    report_path = Path(PROCESSED_DATA_DIR) / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation completed. Top-1 Acc: {top1_acc:.4f}, Top-3 Acc: {top3_acc:.4f}, Micro F1: {micro_f1:.4f}")
    logger.info(f"Report saved to '{report_path}'")
    return report_data

if __name__ == "__main__":
    evaluate_atc_classifier()
