"""
Training script for BETO/BiomedBERT ATC Hierarchical Multi-Label Classifier.
Predicts 5-level Anatomical Therapeutic Chemical (ATC) taxonomy from Spanish drug descriptions.
"""
from src.utils import setup_environment, setup_logger, set_seed, get_device
setup_environment()

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from src.config import (
    PROCESSED_DATA_DIR,
    ATC_MODEL_DIR,
    ATC_BASE_MODEL,
    ATC_MAX_LENGTH,
    ATC_BATCH_SIZE,
    ATC_EPOCHS,
    ATC_LR,
    ATC_SEED,
    ATC_LEVEL_1_MAP
)

logger = setup_logger("train_atc_classifier")

class ATCDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        encoding["labels"] = int(label)
        return encoding

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

def train_atc_model():
    set_seed(ATC_SEED)
    device = get_device()
    logger.info(f"Training ATC Classifier on device: {device}")
    
    dataset_file = Path(PROCESSED_DATA_DIR) / "atc_dataset.parquet"
    if not dataset_file.exists():
        logger.error(f"Processed dataset '{dataset_file}' not found. Please run src.data_preprocessing first.")
        return
        
    df = pd.read_parquet(dataset_file)
    logger.info(f"Loaded {len(df)} records from '{dataset_file}'")
    
    # Map Level 1 ATC categories to label IDs
    unique_levels = sorted(df["atc_level1"].dropna().unique())
    label2id = {lvl: i for i, lvl in enumerate(unique_levels)}
    id2label = {i: lvl for i, lvl in enumerate(unique_levels)}
    
    df["label"] = df["atc_level1"].map(label2id)
    df = df.dropna(subset=["label", "text_input"])
    df["label"] = df["label"].astype(int)
    
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=ATC_SEED, stratify=df["label"])
    logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation. Classes: {len(label2id)}")
    
    logger.info(f"Loading tokenizer & model: '{ATC_BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(ATC_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        ATC_BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    train_dataset = ATCDataset(train_df["text_input"], train_df["label"], tokenizer, ATC_MAX_LENGTH)
    val_dataset = ATCDataset(val_df["text_input"], val_df["label"], tokenizer, ATC_MAX_LENGTH)
    
    output_dir = Path(ATC_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label mappings
    mapping_data = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "atc_level1_map": ATC_LEVEL_1_MAP
    }
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=ATC_LR,
        per_device_train_batch_size=ATC_BATCH_SIZE,
        per_device_eval_batch_size=ATC_BATCH_SIZE,
        num_train_epochs=ATC_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting training loop for BETO ATC Classifier...")
    trainer.train()
    
    logger.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation Results: {eval_results}")
    
    # Save final model & tokenizer
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    # Save metrics report
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
        
    logger.info(f"BETO ATC Classifier successfully trained & saved to '{final_model_path}'")

if __name__ == "__main__":
    train_atc_model()
