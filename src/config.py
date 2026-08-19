"""
Configuration settings for CIMA Pharmaceutical Assistant & ATC Classifier.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ATC_MODEL_DIR = MODELS_DIR / "atc_classifier"
LLM_MODEL_DIR = MODELS_DIR / "cima_medical_llama"

for d in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ATC_MODEL_DIR, LLM_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Datasets
HF_DATASET_NAME = "hsilvosa/aemps-cima"
LOCAL_DATASET_DIR = "D:/datasets/aemps-cima"

# ATC Classifier Model Settings
ATC_BASE_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
ATC_ALT_BIOMED_MODEL = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
ATC_MAX_LENGTH = 256
ATC_BATCH_SIZE = 32
ATC_EPOCHS = 4
ATC_LR = 3e-5
ATC_SEED = 42

# Spanish Medical SLM/LLM Settings (Llama-3.2-3B / Phi-3.5-mini)
LLM_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LLM_FALLBACK_MODEL = "microsoft/Phi-3.5-mini-instruct"
LLM_MAX_LENGTH = 512
LLM_BATCH_SIZE = 4
LLM_GRAD_ACC = 4
LLM_EPOCHS = 3
LLM_LR = 2e-4
LLM_LORA_R = 16
LLM_LORA_ALPHA = 32
LLM_LORA_DROPOUT = 0.05

# ATC Anatomical Groups (Level 1)
ATC_LEVEL_1_MAP = {
    "A": "Alimentario y Metabolismo",
    "B": "Sangre y Órganos Hematopoyéticos",
    "C": "Sistema Cardiovascular",
    "D": "Dermatológicos",
    "G": "Sistema Genitourinario y Hormonas Sexuales",
    "H": "Preparados Hormonales Sistémicos (Excl. Hormonas Sexuales)",
    "J": "Antiinfecciosos para Uso Sistémico",
    "L": "Agentes Antineoplásicos e Inmunomoduladores",
    "M": "Sistema Musculoesquelético",
    "N": "Sistema Nervioso",
    "P": "Productos Antiparasitarios, Insecticidas y Repelentes",
    "R": "Sistema Respiratorio",
    "S": "Órganos de los Sentidos",
    "V": "Varios"
}
