# Spanish Pharmaceutical Assistant and ATC Classifier

A machine learning and natural language processing system for Spanish pharmaceutical information and Anatomical Therapeutic Chemical (ATC) classification (currently the first, anatomical level), built on official Spanish Agencia Española de Medicamentos y Productos Sanitarios (AEMPS) datasets. **Status:** the grounded CIMA RAG is evaluated and working; the ATC classifier and the medical LLM adapter are *not yet trained* (see [Model status](#model-status)).

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
   Transformer encoder built on `dccuchile/bert-base-spanish-wwm-cased` (BETO); `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` is configured as an alternative. It accepts Spanish drug descriptions, active ingredients, dosage forms, or clinical text snippets and predicts the ATC level-1 anatomical group (single-label). Levels 2–5 are already extracted during preprocessing; training on them is listed under Next Steps. **Not trained yet:** the weights in `models/atc_classifier` are the BETO base model with a randomly initialised classification head (see [Model status](#model-status)).

2. **CIMA Spanish Medical LLM** (`text-generation` / `causal-lm`):  
   Instruction fine-tuned small language model (`meta-llama/Llama-3.2-3B-Instruct` / `microsoft/Phi-3.5-mini-instruct`) intended to be trained with 4-bit QLoRA on Spanish patient leaflets and product technical sheets (`src/train_medical_llm.py`). **Not trained yet:** the adapter in `models/cima_medical_llama` is an untrained placeholder whose LoRA B matrices are all zero, so it behaves exactly like the base model (see [Model status](#model-status)).

3. **Grounded CIMA RAG**:
   A section-level hybrid retrieval pipeline over a pinned `HSilvosa/aemps-cima` release. It combines BGE-M3 dense search, Spanish full-text search, reciprocal-rank fusion, multilingual reranking, and local Qwen3 generation through `llama.cpp`. Every factual answer must cite an exact quotation and official AEMPS URL.

4. **Legacy Semantic Medication Search**:
   TF-IDF n-gram vectorizer and cosine similarity index over 75,830 CIMA medication records, enabling zero-shot semantic query matching for arbitrary natural language and informal patient queries.

`CIMAMedicalAssistantEngine` is the shared application facade. The existing `answer_question` symptom workflow remains available, while `query_evidence` adds grounded retrieval through the same engine. `search_medications` combines the indexed entities with legacy catalogue and ATC fields. Streamlit does not instantiate a separate RAG service.

---

## Evaluation Benchmark Results

| Model Component | Base Architecture | Evaluation Metric | Score | Pipeline Tag |
|---|---|---|---|---|
| BETO ATC classifier (**untrained**, random head) | `dccuchile/bert-base-spanish-wwm-cased` | Top-1 Accuracy (1,000 samples) | 2.0% (chance 7.1%) | `text-classification` |
| BETO ATC classifier (**untrained**, random head) | `dccuchile/bert-base-spanish-wwm-cased` | Top-3 Accuracy (1,000 samples) | 29.0% | `text-classification` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Recall@5 / MRR (300 cases) | 1.000 / 1.000 | `retrieval-augmented-generation` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Citation precision / coverage | 0.990 / 0.997 | `retrieval-augmented-generation` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Answer recall (expected values) | 0.977 | `retrieval-augmented-generation` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Refusal accuracy | 1.000 | `retrieval-augmented-generation` |

The RAG rows come from the full GPU run recorded in `eval/baseline.json` (2026-10-08, 300 cases, all six release gates pass). Read them with these caveats: 250 of the 300 cases are catalogue lookups (ATC, pharmaceutical form, marketed status, prescription, ingredients, routes) and only 11 exercise leaflet or technical-sheet text, so the result says little about free-text question answering. `answer_recall` is the fraction of expected value tokens found in the answer (yes/no read from the first words, an ATC code counts when the answer gives a longer code that starts with it); it replaced a token F1 that scored correct but verbose answers near zero. Catalogue fields are indexed as one citable passage per medicine (`Datos de catalogo AEMPS`). The first run, before these changes, failed four gates (answer F1 0.034, citation precision 0.867, citation coverage 0.870, refusal accuracy 0.800) because the index lacked those fields, Qwen3 reasoning exhausted the token budget (39 invalid generations), and the model omitted inline `[C#]` markers. The same run exposed a safety bug: the personalised-dosing filter missed accented questions such as ¿Cuánto debo tomar…?. Thresholds were never changed.

### Model status

Checked on 2026-10-08:

- **ATC classifier: not trained.** `models/atc_classifier/model.safetensors` was written by `export_hf.py` from the BETO base model with a freshly initialised classification head. `src/train_atc_classifier.py` saves its result to `models/atc_classifier/final_model`, and that folder does not exist. The two rows above are the measured accuracy of those exported weights on 1,000 validation samples (`data/processed/evaluation_report.json` records 10.6% from an earlier run of the same untrained model). Earlier versions of this README and of the generated model card listed 94.8% / 98.6% / 0.942 / 0.915; those numbers did not come from any evaluation and have been removed. Until the model is trained, the ATC tab of the Streamlit app returns predictions from the untrained model and should not be trusted.
- **Medical LLM adapter: not trained.** `export_hf.py` creates `adapter_model.safetensors` with zero LoRA B matrices (112 of 112 are zero), which is a no-op on top of `meta-llama/Llama-3.2-3B-Instruct`. No result is reported for it.
- **Grounded CIMA RAG: evaluated.** Uses pretrained `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` and Qwen3-8B without fine-tuning; results above.

If you published these artifacts to the Hugging Face Hub, their model cards carry the same removed numbers and training claims and need the same correction.

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
llama-server -m models/qwen3/Qwen3-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --ctx-size 8192 --parallel 1 --n-gpu-layers 99 --jinja
```

On a 12 GB GPU, `--parallel 1 --ctx-size 8192` is needed: with the default four slots at 16384 the model, reranker and embedder overflow VRAM and generation drops to about 8 tokens/s.

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
- Train the ATC classifier (`python -m src.train_atc_classifier`, about 25–60 minutes on a 12 GB GPU by estimate), then run `python -m src.evaluate` and commit the real `evaluation_report.json`. `src/evaluate.py` and `src/inference.py` now fail with an explicit error when `models/atc_classifier/final_model` is missing, and the Streamlit ATC tab is disabled until the model exists.
- Train the medical LLM adapter (`python -m src.train_medical_llm`) and evaluate it before reporting any result for it.
- Replace the metrics and training claims in the model cards generated by `export_hf.py` once real results exist.
- Broaden the RAG benchmark with more leaflet and technical-sheet questions (only 11 of 300 cases today) and add a per-section quality review of the generated answers.

## Medical Disclaimer

This repository is developed strictly for research, educational, and data analytics purposes. It is not medical advice, diagnosis, or prescribing guidance. Users must consult AEMPS and qualified healthcare professionals for medical decisions.
