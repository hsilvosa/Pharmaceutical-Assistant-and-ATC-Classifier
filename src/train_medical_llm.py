"""
Fine-tuning script for CIMA Spanish Medical Assistant SLM/LLM using LoRA / QLoRA.
Fine-tunes Llama-3.2-3B-Instruct / Phi-3.5-mini on Spanish medical QA & leaflet instruction dataset.
"""
from src.utils import setup_environment, setup_logger, set_seed, get_device
setup_environment()

import os
import json
import torch
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from src.config import (
    PROCESSED_DATA_DIR,
    LLM_MODEL_DIR,
    LLM_BASE_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_MAX_LENGTH,
    LLM_BATCH_SIZE,
    LLM_GRAD_ACC,
    LLM_EPOCHS,
    LLM_LR,
    LLM_LORA_R,
    LLM_LORA_ALPHA,
    LLM_LORA_DROPOUT
)

logger = setup_logger("train_medical_llm")

def format_instruction_prompt(example: dict) -> str:
    """Format single Spanish Medical QA pair into instruction template."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    
    prompt = (
        f"<|system|>\nEres un asistente médico farmacéutico experto en medicamentos de España (AEMPS CIMA). "
        f"Responde de forma rigurosa, clara y precisa basándote en prospectos oficiales.\n"
        f"<|user|>\n{instruction}\n{inp}\n<|assistant|>\n{out}"
    )
    return prompt

def train_medical_llm():
    set_seed(42)
    device = get_device()
    logger.info(f"Training CIMA Spanish Medical LLM on device: {device}")
    
    qa_file = Path(PROCESSED_DATA_DIR) / "medical_qa_dataset.jsonl"
    if not qa_file.exists():
        logger.error(f"Medical QA dataset file '{qa_file}' not found. Run src.data_preprocessing first.")
        return
        
    logger.info(f"Loading QA dataset from '{qa_file}'...")
    raw_dataset = load_dataset("json", data_files=str(qa_file), split="train")
    logger.info(f"Loaded {len(raw_dataset)} instruction samples.")
    
    # Try loading primary model or fallback model
    model_name = LLM_BASE_MODEL
    logger.info(f"Targeting base model: '{model_name}'")
    
    output_dir = Path(LLM_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LLM_LORA_R,
        lora_alpha=LLM_LORA_ALPHA,
        lora_dropout=LLM_LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Check if 4-bit quantization is available
    use_4bit = torch.cuda.is_available()
    bnb_config = None
    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        except Exception as e:
            logger.warning(f"BitsAndBytes 4-bit setup failed: {e}. Falling back to standard fp16/fp32 loading.")
            bnb_config = None
            
    try:
        logger.info(f"Loading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Loading model weights...")
        if bnb_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    except Exception as e:
        logger.warning(f"Primary model '{model_name}' failed to load ({e}). Using fallback SLM model '{LLM_FALLBACK_MODEL}'...")
        model_name = LLM_FALLBACK_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocess dataset text
    def tokenize_function(example):
        formatted_text = format_instruction_prompt(example)
        return tokenizer(
            formatted_text,
            truncation=True,
            max_length=LLM_MAX_LENGTH,
            padding="max_length"
        )
        
    tokenized_ds = raw_dataset.map(tokenize_function, batched=False, remove_columns=raw_dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=LLM_BATCH_SIZE,
        gradient_accumulation_steps=LLM_GRAD_ACC,
        learning_rate=LLM_LR,
        logging_steps=10,
        num_train_epochs=LLM_EPOCHS,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        report_to="none"
    )
    
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    logger.info("Starting LoRA fine-tuning for Spanish Medical Assistant LLM...")
    trainer.train()
    
    final_adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    logger.info(f"LoRA adapter successfully saved to '{final_adapter_path}'")

if __name__ == "__main__":
    train_medical_llm()
