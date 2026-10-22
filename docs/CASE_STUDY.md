# Case study: grounded medicine information assistant for a pharmacy chain

A fictional engagement built on real data (AEMPS CIMA) to show a forward-deployed delivery: discovery, design, evidence, risks and handover. Every number below comes from this repository's evaluation files.

## 1. Discovery

**Client problem (assumed).** Pharmacists and call-centre staff look up leaflet and technical-sheet facts by hand in CIMA. Answers vary between people, and a wrong or invented answer is a safety risk.

**Constraints that shaped the design**
- No personal medical advice, ever: the tool informs, it does not diagnose or dose.
- Every statement must be traceable to an official AEMPS document.
- Data stays on the client's hardware (one 12 GB GPU workstation), so no external LLM API.
- Messy source data: 69% of the ATC training rows are duplicate texts, and the same product appears under several registrations (packs, parallel imports).

**Success criteria agreed up front**
1. Zero answers without a verifiable citation.
2. Personal-advice and injection requests are refused before any model call.
3. At least 80% of free-text questions answered correctly on a held-out set, with the measurement caveats stated.
4. Median answer time under 15 s on the client's workstation.

## 2. What was built

| Layer | Component | Evidence |
|---|---|---|
| Retrieval and answers | Hybrid search (BGE-M3 dense + full text), reranker, Qwen3-8B Q4 with cited quotes | `eval/baseline.json`, 300 cases, all release gates pass (250 are catalogue lookups) |
| Free-text quality | 120 questions written from real passages | test split: 82.3% judged correct (95% CI 71-90%), 96.8% faithful to cited passages (`eval/freetext_baseline.json`) |
| Agent | Tool-using loop with step, tool and time budgets, guardrails, JSONL traces | 33/36 trajectory cases (`eval/agent_baseline.json`) |
| Serving | llama.cpp with flash attention, 2 slots, q8 KV cache; Prometheus and Grafana | `eval/serving_llamacpp.json`; median 10.2 s per agent question |
| ATC hint | BETO classifier, level 1 only | 79.3% top-1 on unseen active ingredients (55% with only a name); confidence was 98% before calibration (ECE 0.189), 0.061 after temperature scaling; shown as a hint, never as evidence |

## 3. Decisions worth defending

- **The agent cannot write facts.** The model picks evidence ids and the answer is rendered from exact source quotes. Cost: less fluent answers. Benefit: invented citations and unsupported claims are rejected by code, not by hope.
- **Refuse in code, not in the prompt.** Pattern guards run before the model; a model refusal without a guard match is treated as over-refusal and retried.
- **Ask when it matters, not when it does not.** The first free-text run refused 6 of 58 questions because several registrations shared a name. Fix: collapse same-name registrations for document questions, keep asking for registration-specific ones (marketing status, pack, national code).
- **Report the measurement, not just the score.** The judge is the same model as the generator; a negative control and a manual review of failures are included, and the test split was used once.
- **Local serving first.** Flash attention gave 4x faster prompt processing and extra slots gave about 70% more throughput on the same card.

## 4. Risks and what is not solved

| Risk | State |
|---|---|
| Wrong answers that look cited | 2 of 58 dev items were real model errors (for example a renal dose adjustment stated backwards). A pharmacist must stay in the loop. |
| Long leaflet questions | One question exceeded the 90 s budget in several benchmark runs and finished in 34-43 s when run alone, so the timeout is intermittent; p95 was 36 s. |
| Judge bias | Same model judges its own answers; only partly controlled. |
| VRAM headroom | LLM plus embedding models use about 11.7 of 12 GB; latency was erratic in a full-stack manual test and the cause was not isolated. |
| Guard coverage | Regex guards were tuned on the benchmark cases; novel phrasings of personal advice can slip through. Needs red-team review before any public use. |
| Data freshness | Index is pinned to one CIMA revision; there is no update job. |

## 5. Handover

**Run it**
1. Start `llama-server` and the API as in the README section "Serving and observability".
2. `python scripts/run_agent_eval.py` and `python scripts/run_freetext_eval.py --split dev` after any change; compare with the files in `eval/`.
3. Watch the Grafana dashboard: run rate by status, latency p95, steps per run, tool errors, llama-server queue.

**Alert on**
- Share of `insufficient_evidence` or `timeout` terminations rising (retrieval or latency regression).
- Any `answered` run without citations (should be impossible; alert means a bug).
- llama-server `requests_deferred` above 0 for minutes (capacity).

**Owner tasks for the client**
- Review a weekly sample of 20 answers against the source text.
- Approve the refusal wording and extend the guard patterns from real traffic.
- Decide the index refresh cadence and who runs it.

**Not in scope of this build**: authentication, user management, audit retention, an interface beyond the existing Streamlit demo, and any clinical validation.
