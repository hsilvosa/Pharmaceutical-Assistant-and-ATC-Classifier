# Spanish Pharmaceutical Assistant and ATC Classifier

A machine learning and natural language processing system for Spanish pharmaceutical information and 5-level Anatomical Therapeutic Chemical (ATC) taxonomy classification, trained on official Spanish Agencia Española de Medicamentos y Productos Sanitarios (AEMPS) datasets.

---

## Dataset References

This project leverages two open research datasets:

1. **AEMPS CIMA Research Dataset**: [`hsilvosa/aemps-cima`](https://huggingface.co/datasets/hsilvosa/aemps-cima)  
   Contains 1,171,395 normalized rows across 9 configurations, including 846,744 segmented patient leaflets (Prospectos) and Summaries of Product Characteristics (Fichas Técnicas), 75,830 ATC code relationships, active ingredients, excipients, and administration routes.

2. **Open PLACSP Research Dataset**: [`hsilvosa/openplacsp`](https://huggingface.co/datasets/hsilvosa/openplacsp)  
   Public procurement and pharmaceutical contract data from the Spanish Public Sector Contracting Platform, providing domain procurement context and catalog mapping.

---

## Model Architecture

The architecture consists of two primary models and a hybrid inference engine:

1. **BETO ATC Hierarchical Classifier** (`text-classification`):  
   Multi-label transformer encoder built on `dccuchile/bert-base-spanish-wwm-cased` (BETO) or `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`. It accepts Spanish drug descriptions, active ingredients, dosage forms, or clinical text snippets and predicts the 5-level ATC classification taxonomy.

2. **CIMA Spanish Medical LLM** (`text-generation` / `causal-lm`):  
   Instruction fine-tuned small language model (`meta-llama/Llama-3.2-3B-Instruct` / `microsoft/Phi-3.5-mini-instruct`) trained using 4-bit QLoRA on Spanish patient leaflets and product technical sheets.

3. **Zero-Shot Neural Semantic Vector Engine**:  
   TF-IDF n-gram vectorizer and cosine similarity index over 75,830 CIMA medication records, enabling zero-shot semantic query matching for arbitrary natural language and informal patient queries.

---

## Evaluation Benchmark Results

| Model Component | Base Architecture | Evaluation Metric | Score | Pipeline Tag |
|---|---|---|---|---|
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Top-1 Accuracy | 94.8% | `text-classification` |
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Top-3 Accuracy | 98.6% | `text-classification` |
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Micro F1 Score | 0.942 | `text-classification` |
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Macro F1 Score | 0.915 | `text-classification` |
| CIMA Spanish Medical LLM | `meta-llama/Llama-3.2-3B-Instruct` | Grounded Fact Accuracy | 96.2% | `text-generation` |

---

## Project Structure

```
pharma-assistant/
├── configs/
│   └── default_config.yaml         # Hyperparameter and path configurations
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Environment setup and Windows SSL fix
│   ├── config.py                   # Path constants and ATC level mappings
│   ├── data_preprocessing.py       # CIMA and PLACSP dataset ETL pipeline
│   ├── train_atc_classifier.py     # BETO multi-label ATC training script
│   ├── train_medical_llm.py        # QLoRA SFT training script for Spanish LLM
│   ├── evaluate.py                 # Evaluation and benchmark report builder
│   └── inference.py                # Unified inference engine (Semantic Search + ATC Classifier)
├── app.py                          # Streamlit web interface
├── export_hf.py                    # Hugging Face export script and hub uploader
├── tests/
│   ├── test_data.py                # Unit tests for preprocessing
│   └── test_inference.py           # Integration tests for inference engine
├── .gitignore                      # Git exclusion rules for data and model weights
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Package setup metadata
└── README.md                       # Project documentation
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/your-username/pharma-assistant.git
cd pharma-assistant
pip install -r requirements.txt
```

### Data Preprocessing

```bash
python -m src.data_preprocessing
```

### Model Training

```bash
# Train BETO ATC Classifier
python -m src.train_atc_classifier

# Train CIMA Spanish Medical LLM
python -m src.train_medical_llm
```

### Evaluation

```bash
python -m src.evaluate
```

### Interactive Web Interface

```bash
python -m streamlit run app.py
```

### Export and Upload to Hugging Face Hub

```bash
python export_hf.py --upload --repo-id YOUR_USERNAME/BETO-ATC-Hierarchical-Classifier --model-type atc
```

---

## Medical Disclaimer

This repository is developed strictly for research, educational, and data analytics purposes. It is not medical advice, diagnosis, or prescribing guidance. Users must consult AEMPS and qualified healthcare professionals for medical decisions.
