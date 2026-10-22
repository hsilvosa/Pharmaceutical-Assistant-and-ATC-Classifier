# Spanish Pharmaceutical Assistant and ATC Classifier

A machine learning and natural language processing system for Spanish pharmaceutical information and Anatomical Therapeutic Chemical (ATC) classification (currently the first, anatomical level), built on official Spanish Agencia Española de Medicamentos y Productos Sanitarios (AEMPS) datasets. **Status:** the grounded CIMA RAG is evaluated and working; the ATC classifier (level 1) and the medical LLM adapter are trained with limited, honestly measured results (see [Model status](#model-status)).

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
   Transformer encoder built on `dccuchile/bert-base-spanish-wwm-cased` (BETO); `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` is configured as an alternative. It accepts Spanish drug descriptions, active ingredients, dosage forms, or clinical text snippets and predicts the ATC level-1 anatomical group (single-label). Levels 2–5 are already extracted during preprocessing; training on them is listed under Next Steps. **Trained** (4 epochs, `models/atc_classifier/final_model`); held-out accuracy is 79.3% top-1 on unseen active ingredients (see [Model status](#model-status)).

2. **CIMA Spanish Medical LLM** (`text-generation` / `causal-lm`):  
   QLoRA adapter (r=16) on `Qwen/Qwen2.5-3B-Instruct`, trained for one epoch on 15,000 templated QA pairs built from CIMA catalogue fields (`src/train_medical_llm.py`). It is a **format adapter, not a knowledge source**: it learns the answer style and name-to-ingredient associations, not reliable facts about medicines. Factual answers come from the grounded RAG below (see [Model status](#model-status)).

3. **Grounded CIMA RAG**:
   A section-level hybrid retrieval pipeline over a pinned `HSilvosa/aemps-cima` release. It combines BGE-M3 dense search, Spanish full-text search, reciprocal-rank fusion, multilingual reranking, and local Qwen3 generation through `llama.cpp`. Every factual answer must cite an exact quotation and official AEMPS URL.

4. **Legacy Semantic Medication Search**:
   TF-IDF n-gram vectorizer and cosine similarity index over 75,830 CIMA medication records, enabling zero-shot semantic query matching for arbitrary natural language and informal patient queries.

`CIMAMedicalAssistantEngine` is the shared application facade. The existing `answer_question` symptom workflow remains available, while `query_evidence` adds grounded retrieval through the same engine. `search_medications` combines the indexed entities with legacy catalogue and ATC fields. Streamlit does not instantiate a separate RAG service.

---

## Evaluation Benchmark Results

| Model Component | Base Architecture | Evaluation Metric | Score | Pipeline Tag |
|---|---|---|---|---|
| BETO ATC classifier (level 1, 13 of 14 classes present in test) | `dccuchile/bert-base-spanish-wwm-cased` | Top-1 Accuracy (7,996 held-out rows, 2,518 unique texts) | 79.3% (chance 7.1%) | `text-classification` |
| BETO ATC classifier (level 1) | `dccuchile/bert-base-spanish-wwm-cased` | Top-3 Accuracy / macro-F1 (same test set) | 87.7% / 0.691 | `text-classification` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Recall@5 / MRR (300 cases) | 1.000 / 1.000 | `retrieval-augmented-generation` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Citation precision / coverage | 0.990 / 1.000 | `retrieval-augmented-generation` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Answer recall (expected values) | 0.977 | `retrieval-augmented-generation` |
| CIMA RAG | `BAAI/bge-m3` + Qwen3 8B GGUF | Refusal accuracy | 1.000 | `retrieval-augmented-generation` |
| Qwen2.5-3B + LoRA adapter (format adapter) | `Qwen/Qwen2.5-3B-Instruct` | Catalogue fields in answer, held-out medicines, no context (300 rows) | 77.8% (base 27.8%) | `text-generation` |

The RAG rows come from the full GPU run recorded in `eval/baseline.json` (2026-10-08, 300 cases, all six release gates pass). Read them with these caveats: 250 of the 300 cases are catalogue lookups (ATC, pharmaceutical form, marketed status, prescription, ingredients, routes) and only 11 exercise leaflet or technical-sheet text, so the result says little about free-text question answering. `answer_recall` is the fraction of expected value tokens found in the answer (yes/no read from the first words, an ATC code counts when the answer gives a longer code that starts with it); it replaced a token F1 that scored correct but verbose answers near zero. Catalogue fields are indexed as one citable passage per medicine (`Datos de catalogo AEMPS`). The first run, before these changes, failed four gates (answer F1 0.034, citation precision 0.867, citation coverage 0.870, refusal accuracy 0.800) because the index lacked those fields, Qwen3 reasoning exhausted the token budget (39 invalid generations), and the model omitted inline `[C#]` markers. The same run exposed a safety bug: the personalised-dosing filter missed accented questions such as ¿Cuánto debo tomar…?. Thresholds were never changed.

### Model status

Checked on 2026-10-08:

- **ATC classifier: trained, level 1 only.** `python -m src.train_atc_classifier` fine-tunes BETO for 4 epochs (batch 32, lr 3e-5) and writes `models/atc_classifier/final_model`. The data has one row per medicine presentation (about 69% of texts are duplicates), so a random row split leaks almost every validation text into training. A first run with a random 85/15 split reported 99.9% validation accuracy, but 97.9% of its validation texts also appeared in training; that number is not a generalisation estimate and was discarded. The reported run splits by active-ingredient set (`src/atc_split.py`, 61,057 train / 6,777 validation / 7,996 test rows, no ingredient set shared between partitions). Results on the held-out test: top-1 79.3%, top-3 87.7%, macro-F1 0.691 (`data/processed/evaluation_report.json`); validation accuracy 84.1%. Class P (antiparasitics, 143 rows) has no rows in the test split, so it is not measured. Only one split seed was run, so the spread between seeds is unknown.
- **Medical LLM adapter: trained, evaluated, limited use.** Base `Qwen/Qwen2.5-3B-Instruct` (research licence), 4-bit QLoRA, 15,000 samples, 1 epoch, answer-only loss, about 2 h on an RTX 3060 (`models/cima_medical_llama/final_adapter`). Evaluation on 300 rows from medicines held out of training (`python -m src.evaluate_medical_llm`, prompt = medicine name only, no context): fields found in the answer 77.8% with the adapter vs 27.8% for the base model (dosage form 93.5% vs 32.0%, prescription status 92.0% vs 0.0%, route 91.0% vs 83.0%, active ingredient 52.0% vs 20.0%, first excipient 45.0% vs 0.0%). Read it with these caveats: most of the gain is answer format; 83% of held-out ingredient strings appear in training under other brands, so the ingredient score reflects learned name-to-ingredient associations; excipient lists and doses are often invented for a specific product; no leaflet, safety or free-form evaluation; one run, one seed. Do not use the adapter as a source of facts.
- **Grounded CIMA RAG: evaluated.** Uses pretrained `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` and Qwen3-8B without fine-tuning; results above.

The exports in `models/*/hf_export` carry the measured numbers above. If you published earlier artifacts to the Hugging Face Hub, their model cards carry the same removed numbers and training claims and need the same correction.

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

The real index is `indexes/cima-bge`; set `CIMA_RAG_INDEX_DIR=indexes/cima-bge` for evaluation, because the default `indexes/cima-full` is a fixture.

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

### Leaflet and technical-sheet benchmark

`python -m src.rag.cli evaluate --generate-sections` writes `eval/sections.jsonl`: 120 Spanish questions, 10 for each of 12 sections (7 technical-sheet sections and 5 leaflet sections). Each question names a medicine and a section. The expected values are four domain words taken from the first 900 characters of that section. Run it with `--benchmark eval/sections.jsonl --output artifacts/evaluations/sections`. It is kept apart from `eval/baseline.json` and has no release gates.

Result of the run on 2026-10-09 (`artifacts/evaluations/sections`):

- Section retrieval: the right medicine and section is in the top 5 for all 120 questions.
- Generation errors: 0 of 120. A first run had 16. Long answers were cut off at the token limit because the model listed every item. A retry that asks for at most 120 words now fixes this; raising the limit alone did not.
- Citation precision: 100%.
- Evidence-term recall: 0.39 overall, from 0.075 (posology) to 0.85 (contraindications). Do not read this as answer quality. A manual check of posology answers found them correct and cited, but they state doses and do not repeat the generic words used as expected values. A human review is still needed.

### Known issues

- **Relevance filter not verified end to end.** `RagService` now returns `insufficient_evidence` without calling the generator when the best reranker score is below `min_rerank_score` (default 0.5, env `CIMA_RAG_MIN_RERANK_SCORE`). The threshold comes from a sample of 60 benchmark questions: answerable questions scored 0.977 or higher and the 20 unanswerable or injection questions scored 0.15 or lower. The full 300-case and 120-section re-run with the filter was started and stopped before it finished, so `eval/baseline.json` (produced without the filter) does not cover it. Re-run `python -m src.rag.cli evaluate` with `CIMA_RAG_INDEX_DIR=indexes/cima-bge` before relying on it.
- The ATC classifier covers level 1 only, was trained with one seed, and class P has no test rows.
- The LLM adapter is a format adapter and is not evaluated with retrieved context or on safety.
- Improvements not started: per-class ATC metrics, other seeds, ATC levels 2-5, manual review of `eval/sections.jsonl` answers.

### Free-text benchmark

`eval/freetext.jsonl` holds 120 questions (58 dev, 62 test; 100 Spanish, 20 English) written by Qwen3-8B from real leaflet and technical-sheet passages, one per medicine, across 16 section types. Each item carries a verbatim reference excerpt and the gold passage; items were kept only if the reference appears verbatim in the passage and the question names the medicine (`scripts/build_freetext_benchmark.py`). Run with `scripts/run_freetext_eval.py --split dev|test`.

| Metric (Qwen3-8B Q4, judge = same model) | dev (58) | test (62) |
|---|---|---|
| Answered | 100% | 100% |
| Gold section in retrieved context | 93.1% | 93.5% |
| Judge: correct | 82.8% | 82.3% (95% CI 71-90%) |
| Judge: correct or partial | 86.2% | 91.9% |
| Judge: every claim supported by cited passages | 98.3% | 96.8% |
| Negative control (shuffled answers judged correct) | 1.7% | 0.0% |

Caveats: the judge and the generator are the same model; the negative control only shows the judge rejects unrelated answers, not that it is unbiased. A manual review of the 8 dev "incorrect" verdicts found 3 judge false negatives, 3 underspecified questions or poor reference excerpts (for example "which adverse reaction" has many valid answers), and 2 real model errors, so the true accuracy is probably higher than 82%. Confidence intervals are wide at n=62. The first dev run (before the fix below) answered only 90%: six questions were refused with "select the presentation" although every candidate was the same product under several registrations; the resolver now picks the canonical registration for document questions and still asks when the question depends on the registration (marketing status, pack, national code).

### ATC classifier: calibration and input ablation

The classifier was badly overconfident: on the held-out split (7,996 rows, 13 of 14 classes present) it averaged 98.2% confidence at 79.3% accuracy. A single temperature (T = 2.37, fitted on the validation split) brings the expected calibration error from 0.189 to 0.061 and the negative log-likelihood from 1.74 to 0.91, without changing any prediction (`scripts/calibrate_atc.py`, `space/calibration.json`, one seed). Accuracy also depends on how much of the training format the user supplies (2,518 distinct held-out texts, `scripts/eval_atc_inputs.py`):

| Input | Top-1 |
|---|---|
| Full record (name, dose, form, ingredients, route) | 78.8% |
| Name + form + ingredient | 72.2% |
| Name + ingredient | 73.2% |
| Ingredient only | 68.2% |
| Name only | 55.1% |

## Public demo

![La botica: ATC classifier drawers, word highlights and an agent receipt](docs/img/la-botica.png)

Live: **https://huggingface.co/spaces/hsilvosa/la-botica** (English by default, Spanish switch). Free Hugging Face Spaces only host static sites, so the demo is a static page (`static-space/`) and the classifier runs in the visitor's browser: the model is exported to ONNX and quantised to int8 (110 MB) with `scripts/export_atc_onnx.py` and run with Transformers.js, with no server and no data leaving the page. On 600 held-out rows the int8 model agrees with the PyTorch model on 98.2% of predictions (top-1 79.5% vs 78.8%, calibration error 0.061 vs 0.062 with the same temperature; `eval/atc_onnx_parity.json`). The first visit downloads the model (about 30 s on a typical connection); the result appears within a second and the word highlights a few seconds later.

While the model downloads, a progress bar shows the megabytes received and the page stays usable. A list of 2,956 CIMA active ingredients (most common first, searchable, filterable by ATC group) lets a visitor try the classifier without knowing any drug: after a pick, the page shows the group CIMA really files it under and whether the classifier never trained on that ingredient (a fair test) or did (not a fair test). A bilingual glossary explains ATC, active ingredient, calibration and the agent's stamps. The list is built by `scripts/build_space_assets.py`.

The page shows a pharmacy cabinet with 14 drawers (the ATC groups) with calibrated probabilities, word-occlusion highlights and the closest CIMA medicines, next to "receipts" of nine real agent runs recorded by `scripts/build_space_gallery.py` (steps, timings, verbatim quotes, refusals), because the agent needs a GPU. Data files are rebuilt with `scripts/build_space_assets.py`; the model file is not stored in this repository (see `.gitignore`).

`space/` is an alternative Gradio version of the same demo that runs the PyTorch model on a server (`python space/app.py`, set `ATC_MODEL` to `models/atc_classifier/hf_export` to skip the Hub download). Hosting Gradio Spaces now requires a paid Hugging Face plan, so it is not deployed.

## Case study

A discovery-to-handover write-up of this system as a client engagement: [docs/CASE_STUDY.md](docs/CASE_STUDY.md).

## What the evaluations found

Eight places where a first result looked good and was not (a 99.9% accuracy that was leakage, a 98% confidence at 79% accuracy, safety rules that did not generalise, a refusal that was a bug, and more): [docs/FINDINGS.md](docs/FINDINGS.md).

## Agent

`src/agent/` is a bounded tool-using agent on top of the grounded RAG (`POST /agent/run`, `python scripts/run_agent_eval.py`).

- Tools: `search_medications` (navigation only), `query_evidence` (citable AEMPS passages), `classify_atc` (hypothesis, confidence withheld).
- Limits: 5 steps, 5 tool calls, 90 s deadline, one identical call at most, JSONL trace per step (tool, args, latency, tokens).
- Guardrails: personal-advice and prompt-injection requests are refused before any model call; the final answer is rendered from exact quotes of registered evidence, so invented citation ids and non-AEMPS URLs are rejected; a guessed registration number after an ambiguous search becomes `needs_disambiguation`.
- Evaluation: 36 trajectory cases (`eval/agent_cases.jsonl`) checking status, tool use, step count, source relevance and expected values. Latest run with Qwen3-8B Q4 (`eval/agent_baseline.json`): 33/36 passed, median 12.5 s and p95 35 s per question, 2.4 steps on average. The three failures: one leaflet question hit the 90 s budget (long generation in the RAG), and two cases cite a correct medicine but a different section than the benchmark keywords. The rules were tuned on these same cases, so this is a development result, not a held-out one.

### Request guards

Personal-advice and prompt-injection requests are refused before any tool call by two layers: feature rules (a first-person or caregiver marker together with an advice marker) and a zero-shot intent classifier on the same local model (`src/agent/intent.py`). Measured with `scripts/eval_guard.py` on labelled red-team sets in `eval/`:

| Set | Rules only | Rules + model |
|---|---|---|
| Development (39 unsafe, 25 safe) | 39/39 blocked, 25/25 allowed | 39/39, 25/25 |
| Held-out, written after the rules were fixed (32 unsafe, 30 safe) | 13/32 blocked, 29/30 allowed | 32/32, 30/30 |

The first version of the rules, which looked fine on the development set, blocked only 12 of 39 on it before it was rewritten; the held-out number shows the rewritten rules still do not generalise, which is why the model layer exists. One held-out safe question (a pack-size catalogue query) was refused by the rules and fixed afterwards, so that last 30/30 is not a clean held-out figure. Sets are small (64 and 62 items): 32/32 still only supports a lower bound of about 89% at 95% confidence. The agent evaluation is unchanged at 33/36 with the layer on (`eval/agent_with_intent_guard.json`); that run is not comparable on latency (other work ran on the machine). A router outage does not block answers; the rules still apply.

## Serving and observability

Measured on an RTX 3060 12 GB with llama.cpp (`llama-server`) serving Qwen3-8B Q4_K_M (`scripts/loadtest_llm.py`, `scripts/sweep_llama_server.py`, raw results in `eval/serving_llamacpp.json`). Each request has a unique prompt of 1,269 tokens (no prefix cache) and generates exactly 128 tokens; 12 requests per concurrency level, closed loop, no errors in any run. TTFT is time to first token; "tok/s per request" is decode speed.

| Configuration | Concurrency | TTFT p50 (s) | Latency p95 (s) | tok/s per request | Aggregate tok/s | VRAM (MiB) |
|---|---|---|---|---|---|---|
| 1 slot, flash attention off | 1 | 3.61 | 8.14 | 44.3 | 19.3 | 7,497 |
| 1 slot, flash attention off | 4 | 21.9 | 25.5 | 43.5 | 20.4 | 7,783 |
| 1 slot, flash attention on | 1 | 0.87 | 3.66 | 46.2 | 35.4 | 7,414 |
| 1 slot, flash attention on | 4 | 11.7 | 14.5 | 46.2 | 35.4 | 7,472 |
| 4 slots, flash attention on | 1 | 0.87 | 3.69 | 46.2 | 35.3 | 8,629 |
| 4 slots, flash attention on | 4 | 2.69 | 9.92 | 21.1 | 60.2 | 8,656 |
| 4 slots, flash attention + KV cache q8_0 | 1 | 0.90 | 5.73 | 44.4 | 32.7 | 7,537 |
| 4 slots, flash attention + KV cache q8_0 | 4 | 2.71 | 10.2 | 20.4 | 58.7 | 7,557 |

- Flash attention cuts prompt processing about 4x (TTFT 3.6 s to 0.87 s) and raises single-slot throughput by 83%. The "off" row is a forced setting; the server default (`-fa auto`) is probably already on for this GPU, so it is not the configuration used before.
- Several slots raise aggregate throughput by about 70% (35 to 60 tok/s) and cut TTFT at concurrency 4 from 11.7 s to 2.7 s. Each request then decodes slower (46 to 20 tok/s) because they share the GPU, and it saturates at 4 slots.
- A q8_0 KV cache saves about 1.1 GB of VRAM at no measurable speed cost. Its effect on answer quality was not measured separately; the agent evaluation below is the only check.
- Agent evaluation (36 cases, one client): same 33/36 cases passed with `--parallel 2 --ctx-size 16384 -fa on -ctk q8_0 -ctv q8_0` as before; median latency 12.5 s to 10.2 s, whole run 573 s to 480 s, p95 unchanged (36 s, the long-generation tail) (`eval/serving_agent_before_after.json`).
- With the agent API loaded next to the LLM (embedding model, reranker and the 8B model), the card is at about 11.7 of 12 GB. In a manual test of the full stack, latency per question varied from 13 to 44 s, against a 10 s median in the isolated evaluation run. I did not isolate the cause; memory pressure is the main suspect. Treat 12 GB as the minimum for this stack.

Run it:

```bash
llama-server -m models/qwen3/Qwen3-8B-Q4_K_M.gguf --port 8080 --jinja --metrics   --parallel 2 --ctx-size 16384 -fa on -ctk q8_0 -ctv q8_0 --n-gpu-layers 99
CIMA_RAG_INDEX_DIR=indexes/cima-bge python scripts/serve_agent.py --port 8000   # loads models at startup (~50 s)
docker compose -f deploy/docker-compose.yml up -d    # Prometheus :9090, Grafana :3000 (local only)
```

The agent API exposes Prometheus metrics on `/metrics` (runs by status and termination reason, run, model and tool latency histograms, steps per run, tokens, tool errors) and Prometheus also scrapes llama-server's own `/metrics`. Grafana provisions the dashboard "Pharma assistant: agent and LLM serving" (`deploy/grafana/dashboards`, generated by `scripts/build_grafana_dashboard.py`). The whole stack was run once end to end and both Prometheus targets were up; set `GRAFANA_ADMIN_PASSWORD` before using it for anything beyond a local demo. vLLM was not tested: it needs Linux (WSL2 or a GPU container) and a download of several GB.

## Next Steps

- Extend the ATC classifier from level 1 to the full five-level hierarchy (levels 2–5 are already in `atc_dataset.parquet`), for example with one head per level or hierarchical decoding.
- Improve the ATC classifier: report per-class results, try several split seeds, add the missing ingredient information to the input, and extend to levels 2–5.
- Evaluate the medical LLM adapter on leaflet and technical-sheet questions with retrieved context, and add a safety review (refusals, personalised dosing). The current evaluation covers only catalogue fields without context.
- Replace the metrics and training claims in the model cards generated by `export_hf.py` once real results exist.
- Review the generated answers of `eval/sections.jsonl` by hand (the evidence-term metric is weak).
- Verify the relevance filter (see Known issues). It targets the 2 of 20 unanswerable and prompt-injection cases (`curated-016`, `curated-030`) that got an off-topic cited answer instead of a refusal. The evaluator now accepts `refused` as a valid non-answer for these cases and reports `non_answer_accuracy` (18/20 on the last run, not gated).

## Medical Disclaimer

This repository is developed strictly for research, educational, and data analytics purposes. It is not medical advice, diagnosis, or prescribing guidance. Users must consult AEMPS and qualified healthcare professionals for medical decisions.
