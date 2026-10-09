"""
QLoRA fine-tuning of a small instruction model on the templated CIMA catalogue QA pairs.

Design notes:
- Medicines are split into train and held-out sets, so evaluation uses unseen medicines.
- The loss covers only the assistant answer (prompt tokens are masked).
- The QA pairs come from three fixed templates over catalogue fields. The model learns the
  answer format and name-to-ingredient patterns. It does not learn reliable facts about
  medicines. Factual answers must come from the grounded RAG, not from this adapter.
"""
from src.utils import setup_environment, setup_logger, set_seed
setup_environment()

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.config import (
    PROCESSED_DATA_DIR,
    LLM_MODEL_DIR,
    LLM_BASE_MODEL,
    LLM_MAX_LENGTH,
    LLM_BATCH_SIZE,
    LLM_GRAD_ACC,
    LLM_EPOCHS,
    LLM_LR,
    LLM_LORA_R,
    LLM_LORA_ALPHA,
    LLM_LORA_DROPOUT,
)
from src.medical_qa import build_messages, load_qa, split_by_medicine

logger = setup_logger("train_medical_llm")

N_HELDOUT_MEDICINES = 300
SEED = 42


class QADataset(torch.utils.data.Dataset):
    """Tokenised chat examples with the prompt part masked out of the loss."""

    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                build_messages(row, with_answer=False), tokenize=False, add_generation_prompt=True
            )
            full = prompt + row["output"] + "<|im_end|>"
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
            self.items.append({"input_ids": ids, "labels": labels[: len(ids)]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    longest = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train_medical_llm(max_samples: int | None = None, max_steps: int = -1, output_name: str = "final_adapter"):
    set_seed(SEED)
    qa_file = Path(PROCESSED_DATA_DIR) / "medical_qa_dataset.jsonl"
    if not qa_file.exists():
        raise FileNotFoundError(f"'{qa_file}' not found. Run src.data_preprocessing first.")

    rows = load_qa(qa_file)
    train_rows, heldout_rows = split_by_medicine(rows, N_HELDOUT_MEDICINES, SEED)
    with open(Path(PROCESSED_DATA_DIR) / "medical_qa_heldout.jsonl", "w", encoding="utf-8") as f:
        for r in heldout_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    random.Random(SEED).shuffle(train_rows)
    if max_samples:
        train_rows = train_rows[:max_samples]
    logger.info(f"{len(train_rows)} train rows, {len(heldout_rows)} held-out rows ({N_HELDOUT_MEDICINES} unseen medicines)")

    tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL)
    train_ds = QADataset(train_rows, tokenizer, LLM_MAX_LENGTH)
    lengths = [len(x["input_ids"]) for x in train_ds.items]
    logger.info(f"Token length mean {sum(lengths)/len(lengths):.0f}, max {max(lengths)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_BASE_MODEL, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LLM_LORA_R,
            lora_alpha=LLM_LORA_ALPHA,
            lora_dropout=LLM_LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    output_dir = Path(LLM_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=LLM_BATCH_SIZE,
        gradient_accumulation_steps=LLM_GRAD_ACC,
        learning_rate=LLM_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=LLM_EPOCHS,
        max_steps=max_steps,
        weight_decay=0.01,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=lambda b: collate(b, pad_id))
    logger.info("Starting QLoRA fine-tuning...")
    result = trainer.train()
    logger.info(f"Training done: {result.metrics}")

    path = output_dir / output_name
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    logger.info(f"LoRA adapter saved to '{path}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--output-name", default="final_adapter")
    a = ap.parse_args()
    train_medical_llm(a.max_samples, a.max_steps, a.output_name)
