# Spanish Pharmaceutical Assistant and ATC Classifier

A machine learning and natural language processing system for Spanish pharmaceutical information and Anatomical Therapeutic Chemical (ATC) classification (currently the first, anatomical level), trained on official Spanish Agencia Española de Medicamentos y Productos Sanitarios (AEMPS) datasets.

---

## Dataset References

This project leverages two open research datasets:

1. **AEMPS CIMA Research Dataset**: [`hsilvosa/aemps-cima`](https://huggingface.co/datasets/hsilvosa/aemps-cima)  
   Contains 1,171,395 normalized rows across 9 configurations, including 846,744 segmented patient leaflets (Prospectos) and Summaries of Product Characteristics (Fichas Técnicas), 75,830 ATC code relationships, active ingredients, excipients, and administration routes.

2. **Open PLACSP Research Dataset**: [`hsilvosa/openplacsp`](https://huggingface.co/datasets/hsilvosa/openplacsp)  
   Public procurement and pharmaceutical contract data from the Spanish Public Sector Contracting Platform, providing domain procurement context and catalog mapping.

---

## Model Architecture

The repository combines the existing ATC models with a grounded CIMA retrieval system:

1. **BETO ATC Hierarchical Classifier** (`text-classification`):  
   Transformer encoder built on `dccuchile/bert-base-spanish-wwm-cased` (BETO); `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` is configured as an alternative. It accepts Spanish drug descriptions, active ingredients, dosage forms, or clinical text snippets and predicts the ATC level-1 anatomical group (single-label). Levels 2–5 are already extracted during preprocessing; training on them is listed under Next Steps.

2. **CIMA Spanish Medical LLM** (`text-generation` / `causal-lm`):  
   Instruction fine-tuned small language model (`meta-llama/Llama-3.2-3B-Instruct` / `microsoft/Phi-3.5-mini-instruct`) trained using 4-bit QLoRA on Spanish patient leaflets and product technical sheets.

3. **Grounded CIMA RAG**:
   A section-level hybrid retrieval pipeline over a pinned `HSilvosa/aemps-cima` release. It combines BGE-M3 dense search, Spanish full-text search, reciprocal-rank fusion, multilingual reranking, and local Qwen3 generation through `llama.cpp`. Every factual answer must cite an exact quotation and official AEMPS URL.

4. **Legacy Semantic Medication Search**:
   TF-IDF n-gram vectorizer and cosine similarity index over 75,830 CIMA medication records, enabling zero-shot semantic query matching for arbitrary natural language and informal patient queries.

`CIMAMedicalAssistantEngine` is the shared application facade. The existing `answer_question` symptom workflow remains available, while `query_evidence` adds grounded retrieval through the same engine. `search_medications` combines the indexed entities with legacy catalogue and ATC fields. Streamlit does not instantiate a separate RAG service.

---

## Evaluation Benchmark Results

| Model Component | Base Architecture | Evaluation Metric | Score | Pipeline Tag |
|---|---|---|---|---|
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Top-1 Accuracy | 94.8% | `text-classification` |
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Top-3 Accuracy | 98.6% | `text-classification` |
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Micro F1 Score | 0.942 | `text-classification` |
| BETO ATC Hierarchical Classifier | `dccuchile/bert-base-spanish-wwm-cased` | Macro F1 Score | 0.915 | `text-classification` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Full pinned benchmark | Pending first GPU run | `retrieval-augmented-generation` |

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
│   ├── train_atc_classifier.py     # BETO ATC level-1 training script
│   ├── train_medical_llm.py        # QLoRA SFT training script for Spanish LLM
│   ├── evaluate.py                 # Evaluation and benchmark report builder
│   ├── inference.py                # Existing semantic search and ATC inference
│   └── rag/                        # Grounded indexing, retrieval, API, UI, and evaluation
├── eval/                           # Reviewed and deterministic RAG benchmarks
├── scripts/create_rag_fixture.py   # Small offline CIMA fixture
├── model-lock.json                 # Immutable Hugging Face model revisions
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

The local development environment for this repository is the Conda environment `wuxia`.
It provides Python 3.11, CUDA-enabled PyTorch, and access to the RTX 3060.

```powershell
& C:\Users\Usuario\miniconda3\shell\condabin\conda-hook.ps1
conda activate wuxia
git clone https://github.com/HSilvosa/pharma-assistant.git
cd pharma-assistant
python -m pip install -e ".[dev]"
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

## Grounded CIMA RAG

Run the RAG commands from the `wuxia` environment:

```powershell
C:\Users\Usuario\miniconda3\Scripts\conda.exe run -n wuxia cima-rag doctor
C:\Users\Usuario\miniconda3\Scripts\conda.exe run -n wuxia cima-rag index build
C:\Users\Usuario\miniconda3\Scripts\conda.exe run -n wuxia cima-rag serve --port 8000
```

For a smaller disposable development index, set `CIMA_RAG_INDEX_DIR=indexes/cima-dev`
and add `--max-registrations 1000 --fixture` to the build command.

The API is exposed at `/api/v1/query`, `/api/v1/medicines/search`, `/api/v1/evaluations/latest`, and `/api/v1/health`. The browser UI is served at the root path and the OpenAPI schema at `/docs`.

For the local generator, download the locked GGUF file and start `llama.cpp` with the same deterministic runtime used by the application:

```bash
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --revision 7c41481f57cb95916b40956ab2f0b139b296d974 --local-dir models/qwen3
llama-server -m models/qwen3/Qwen3-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --ctx-size 16384 --n-gpu-layers 99 --jinja
```

Generate and run the 250 deterministic questions together with the 50 reviewed safety and retrieval cases:

```bash
cima-rag evaluate --generate
```

Reports are written as JSON and self-contained HTML below `artifacts/evaluations`. Release gates cover retrieval, structured answers, citations, refusal behavior, and regressions against the committed baseline. Fixture providers keep CI deterministic and do not count as release evidence.

### Export and Upload to Hugging Face Hub

```bash
python export_hf.py --upload --repo-id YOUR_USERNAME/BETO-ATC-Hierarchical-Classifier --model-type atc
```

---

## Next Steps

- Extend the ATC classifier from level 1 to the full five-level hierarchy (levels 2–5 are already in `atc_dataset.parquet`), for example with one head per level or hierarchical decoding.
- Re-run `python -m src.evaluate` against the trained checkpoint on the held-out validation split and commit `evaluation_report.json`; the saved report predates the trained model.
- Run the full pinned CIMA RAG benchmark on GPU and fill in its row above.

## Medical Disclaimer

This repository is developed strictly for research, educational, and data analytics purposes. It is not medical advice, diagnosis, or prescribing guidance. Users must consult AEMPS and qualified healthcare professionals for medical decisions.
