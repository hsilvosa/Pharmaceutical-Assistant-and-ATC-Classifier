from __future__ import annotations

import hashlib
import html
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .data import SourceData
from .models import AnswerStatus, BenchmarkItem, QueryRequest
from .service import GenerationError, RagService

THRESHOLDS = {
    "recall_at_5": 0.85,
    "mrr": 0.75,
    "citation_precision": 0.95,
    "citation_coverage": 0.95,
    "answer_recall": 0.85,
    "refusal_accuracy": 0.90,
}


def _split(registration: str) -> str:
    digest = hashlib.sha256(f"42|{registration}".encode()).digest()
    return "test" if digest[0] % 5 == 0 else "dev"


def _question(
    category: str, language: str, medicine: str, values: list[str]
) -> str:
    questions = {
        "ingredients": {
            "es": f"¿Cuáles son los principios activos de {medicine}?",
            "en": f"What are the active ingredients in {medicine}?",
        },
        "routes": {
            "es": f"¿Qué vías de administración constan para {medicine}?",
            "en": f"Which administration routes are registered for {medicine}?",
        },
        "prescription": {
            "es": f"¿Requiere receta {medicine}?",
            "en": f"Does {medicine} require a prescription?",
        },
        "form": {
            "es": f"¿Cuál es la forma farmacéutica de {medicine}?",
            "en": f"What is the pharmaceutical form of {medicine}?",
        },
        "marketed": {
            "es": f"¿Figura {medicine} como comercializado?",
            "en": f"Is {medicine} listed as marketed?",
        },
        "atc": {
            "es": f"¿Qué clasificación ATC consta para {medicine}?",
            "en": f"Which ATC classification is registered for {medicine}?",
        },
    }
    return questions[category][language]


def generate_benchmark(source: SourceData, count: int = 250) -> list[BenchmarkItem]:
    medications = source.rows("medications")
    names = {str(row["registration_number"]): str(row.get("name") or "") for row in medications}
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in source.rows("active_ingredients"):
        grouped["ingredients"][str(row["registration_number"])].append(
            str(row.get("ingredient_name") or "")
        )
    for row in source.rows("administration_routes"):
        grouped["routes"][str(row["registration_number"])].append(
            str(row.get("route_name") or "")
        )
    for row in source.rows("atc_codes"):
        grouped["atc"][str(row["registration_number"])].append(str(row.get("atc_code") or ""))
    for row in medications:
        registration = str(row["registration_number"])
        grouped["prescription"][registration] = [
            "sí" if row.get("prescription_required") else "no"
        ]
        grouped["form"][registration] = [str(row.get("pharmaceutical_form_name") or "")]
        grouped["marketed"][registration] = ["sí" if row.get("marketed") else "no"]

    candidates = []
    for category in ("ingredients", "routes", "prescription", "form", "marketed", "atc"):
        for registration in sorted(grouped[category]):
            values = sorted({value.strip() for value in grouped[category][registration] if value.strip()})
            if not names.get(registration) or not values:
                continue
            candidates.append((category, registration, values))
    candidates.sort(key=lambda item: hashlib.sha256(f"42|{item}".encode()).hexdigest())
    items = []
    for index, (category, registration, values) in enumerate(candidates[:count]):
        language = "es" if index % 4 else "en"
        items.append(
            BenchmarkItem(
                id=f"generated-{index + 1:03d}",
                split=_split(registration),
                category=category,
                language=language,
                question=_question(category, language, names[registration], values),
                registration_number=registration,
                relevant_registration_numbers=[registration],
                expected_values=values,
                expected_status=AnswerStatus.ANSWERED,
            )
        )
    return items


# (document_type, section) -> (category, Spanish question, English question)
SECTION_SPECS = {
    (1, "4.1"): ("ft_indications", "¿Cuáles son las indicaciones terapéuticas de {m} según su ficha técnica?", "What therapeutic indications does the technical sheet of {m} list?"),
    (1, "4.2"): ("ft_posology", "¿Qué posología y forma de administración recoge la ficha técnica de {m}?", "What posology and method of administration does the technical sheet of {m} give?"),
    (1, "4.3"): ("ft_contraindications", "¿Qué contraindicaciones recoge la ficha técnica de {m}?", "What contraindications does the technical sheet of {m} list?"),
    (1, "4.5"): ("ft_interactions", "¿Qué interacciones con otros medicamentos describe la ficha técnica de {m}?", "What drug interactions does the technical sheet of {m} describe?"),
    (1, "4.8"): ("ft_adverse_reactions", "¿Qué reacciones adversas describe la ficha técnica de {m}?", "What adverse reactions does the technical sheet of {m} describe?"),
    (1, "4.9"): ("ft_overdose", "¿Qué dice la ficha técnica de {m} sobre la sobredosis?", "What does the technical sheet of {m} say about overdose?"),
    (1, "6.4"): ("ft_storage", "¿Qué precauciones de conservación indica la ficha técnica de {m}?", "What storage precautions does the technical sheet of {m} indicate?"),
    (2, "1"): ("leaflet_use", "¿Para qué se utiliza {m} según el prospecto?", "What is {m} used for according to the leaflet?"),
    (2, "2"): ("leaflet_before_taking", "¿Qué debo saber antes de tomar {m} según el prospecto?", "What should I know before taking {m} according to the leaflet?"),
    (2, "3"): ("leaflet_how_to_take", "¿Cómo se toma {m} según el prospecto?", "How is {m} taken according to the leaflet?"),
    (2, "4"): ("leaflet_side_effects", "¿Qué efectos adversos menciona el prospecto de {m}?", "What side effects does the leaflet of {m} mention?"),
    (2, "5"): ("leaflet_storage", "¿Cómo debe conservarse {m} según el prospecto?", "How should {m} be stored according to the leaflet?"),
}
_MIN_SECTION_CHARS = 250
_LEAD_CHARS = 900
_MIN_TERM_FREQUENCY = 3


def generate_section_benchmark(source: SourceData, count: int = 120, terms: int = 4) -> list[BenchmarkItem]:
    """Build leaflet and technical-sheet questions with checkable evidence terms.

    For each sampled medicine and section, the expected values are the rarest words of the
    first part of that section (the part most likely to be retrieved first). They are not
    in the question or the medicine name. A good grounded answer should repeat some of them,
    so ``answer_recall`` measures how much of the source section reaches the answer.
    """
    medications = source.rows("medications")
    names = {str(r["registration_number"]): str(r.get("name") or "") for r in medications}
    registrations = sorted(r for r, n in names.items() if n)
    pool = set(
        sorted(registrations, key=lambda r: hashlib.sha256(f"42|sec|{r}".encode()).hexdigest())[:1500]
    )
    sections: dict[tuple[str, int, str], str] = {}
    for row in source.iter_rows("documents"):
        key = (int(row.get("document_type") or 0), str(row.get("section") or ""))
        registration = str(row["registration_number"])
        if key not in SECTION_SPECS or registration not in pool:
            continue
        text = str(row.get("content_text") or "").strip()
        if len(text) >= _MIN_SECTION_CHARS:
            sections.setdefault((registration, *key), text)

    document_frequency: dict[tuple[int, str, str], int] = defaultdict(int)
    token_cache: dict[tuple[str, int, str], list[str]] = {}
    for key, text in sections.items():
        tokens = _tokens(text[:_LEAD_CHARS])
        token_cache[key] = tokens
        for token in set(tokens):
            document_frequency[(key[1], key[2], token)] += 1

    candidates = []
    for (registration, doc_type, section), tokens in token_cache.items():
        name_tokens = set(_tokens(names[registration]))
        seen, picked = set(), []
        # Words used in at least 3 sections of the same kind are real domain terms.
        # Words used only once are often typos or product-specific noise.
        def frequency(token: str) -> int:
            return document_frequency[(doc_type, section, token)]

        for token in sorted(tokens, key=lambda t: (frequency(t), t)):
            if (
                token in seen
                or token in name_tokens
                or len(token) < 7
                or not token.isalpha()
                or frequency(token) < _MIN_TERM_FREQUENCY
            ):
                continue
            seen.add(token)
            picked.append(token)
            if len(picked) == terms:
                break
        if len(picked) == terms:
            candidates.append((registration, doc_type, section, picked))
    candidates.sort(key=lambda c: hashlib.sha256(f"42|{c[0]}|{c[1]}|{c[2]}".encode()).hexdigest())

    per_category: dict[str, int] = defaultdict(int)
    cap = max(1, math.ceil(count / len(SECTION_SPECS)))
    items = []
    for registration, doc_type, section, picked in candidates:
        category, es, en = SECTION_SPECS[(doc_type, section)]
        if per_category[category] >= cap or len(items) >= count:
            continue
        per_category[category] += 1
        # Evidence terms come from the Spanish source text, so questions are Spanish.
        # Bilingual behaviour is covered by the curated benchmark.
        language = "es"
        medicine = names[registration]
        items.append(
            BenchmarkItem(
                id=f"section-{len(items) + 1:03d}",
                split=_split(registration),
                category=category,
                language=language,
                question=es.format(m=medicine),
                registration_number=registration,
                relevant_registration_numbers=[registration],
                expected_values=picked,
                expected_status=AnswerStatus.ANSWERED,
                metadata={"document_type": doc_type, "section": section},
            )
        )
    return items


def load_benchmark(paths: Sequence[Path]) -> list[BenchmarkItem]:
    items = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(BenchmarkItem.model_validate_json(line))
    return items


def write_benchmark(path: Path, items: Iterable[BenchmarkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8"
    )


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", _fold(value))


_NEGATIVE = re.compile(
    r"\b(no|not|sin|without)\b|\bdoes not\b|\bisn t\b|\bno requiere\b|\bno figura\b"
)
_AFFIRMATIVE = re.compile(
    r"\b(si|yes)\b|\brequiere\b|\brequires?\b|\bcomercializado\b|\bmarketed\b|\bcon receta\b"
)


def _verdict(answer_tokens: list[str]) -> str | None:
    """Read a yes/no verdict from the first words of an answer, in Spanish or English."""
    lead = " ".join(answer_tokens[:10])
    negative = _NEGATIVE.search(lead)
    affirmative = _AFFIRMATIVE.search(lead)
    if negative and (not affirmative or negative.start() <= affirmative.start()):
        return "no"
    if affirmative:
        return "si"
    return None


_YES_NO = {"si": "si", "no": "no"}
_ATC_CODE = re.compile(r"^[a-z]\d{2}[a-z]{0,2}\d{0,2}$")


def _answer_recall(answer: str, expected: Sequence[str]) -> float | None:
    """Fraction of expected value tokens present in the answer.

    Unlike an F1 score this does not penalize long answers. Yes/no values are read from
    the first yes/no word within the first eight words of the answer, and an ATC code counts when the answer states a
    code that starts with it (a level-5 code implies its parent levels).
    """
    if not expected:
        return None
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    answer_set = set(answer_tokens)
    verdict = _verdict(answer_tokens)
    expected_tokens = []
    for value in expected:
        expected_tokens.extend(_tokens(value))
    if not expected_tokens:
        return None
    found = 0
    for token in expected_tokens:
        if token in _YES_NO:
            found += verdict == _YES_NO[token]
        elif token in answer_set:
            found += 1
        elif _ATC_CODE.match(token):
            found += any(_ATC_CODE.match(item) and item.startswith(token) for item in answer_set)
    return found / len(expected_tokens)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _section_rank(item: BenchmarkItem, hits: Sequence) -> int | None:
    """Rank of the first hit from the expected medicine, document type and section."""
    wanted = item.metadata.get("section")
    if wanted is None:
        return None
    relevant = set(item.relevant_registration_numbers)
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        if (
            chunk.registration_number in relevant
            and chunk.document_type == item.metadata.get("document_type")
            and chunk.section.split(".")[:2] == str(wanted).split(".")[:2]
        ):
            return rank
    return None


def _generation_error_case(item: BenchmarkItem, hits: Sequence) -> dict:
    relevant = set(item.relevant_registration_numbers)
    ranks = [
        rank
        for rank, hit in enumerate(hits, start=1)
        if hit.chunk.registration_number in relevant
    ]
    return {
        "id": item.id,
        "split": item.split,
        "category": item.category,
        "question": item.question,
        "expected_status": item.expected_status.value,
        "actual_status": "generation_error",
        "answer": "",
        "citation_ids": [],
        "retrieved": [hit.chunk.chunk_id for hit in hits],
        "first_relevant_rank": min(ranks) if ranks else None,
        "section_rank": _section_rank(item, hits),
        "answer_recall": 0.0 if item.expected_values else None,
        "citation_precision": 0.0,
        "citation_coverage": 0.0,
        "status_correct": False,
        "latency_ms": 0.0,
    }


def evaluate(service: RagService, items: Sequence[BenchmarkItem]) -> dict:
    started = time.perf_counter()
    cases = []
    peak_gpu_mb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None  # type: ignore[assignment]

    for item in items:
        try:
            response, hits = service.query_with_trace(
                QueryRequest(
                    question=item.question,
                    registration_number=item.registration_number,
                    language=item.language,
                )
            )
        except GenerationError as exc:
            # A generation failure is a failed case, not a reason to abort the whole run.
            cases.append(_generation_error_case(item, exc.hits))
            continue
        relevant = set(item.relevant_registration_numbers)
        ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.chunk.registration_number in relevant
        ]
        first_rank = min(ranks) if ranks else None
        section_rank = _section_rank(item, hits)
        citation_correct = [
            citation.registration_number in relevant for citation in response.citations
        ]
        cases.append(
            {
                "id": item.id,
                "split": item.split,
                "category": item.category,
                "question": item.question,
                "expected_status": item.expected_status.value,
                "actual_status": response.status.value,
                "answer": response.answer,
                "citation_ids": [citation.chunk_id for citation in response.citations],
                "retrieved": [hit.chunk.chunk_id for hit in hits],
                "first_relevant_rank": first_rank,
                "section_rank": section_rank,
                "answer_recall": _answer_recall(response.answer, item.expected_values),
                "citation_precision": _mean([float(value) for value in citation_correct])
                if citation_correct
                else (1.0 if response.status != AnswerStatus.ANSWERED else 0.0),
                "citation_coverage": float(
                    response.status != AnswerStatus.ANSWERED or bool(response.citations)
                ),
                "status_correct": response.status == item.expected_status,
                "latency_ms": response.latency_ms.get("total", 0.0),
            }
        )

    elapsed = time.perf_counter() - started
    if torch is not None and torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    retrieval_cases = [case for case in cases if case["first_relevant_rank"] is not None]
    expected_retrieval = [case for case in cases if case["expected_status"] == "answered"]
    answer_scores = [case["answer_recall"] for case in cases if case["answer_recall"] is not None]
    refusal_cases = [case for case in cases if case["expected_status"] == "refused"]
    ambiguity_cases = [
        case for case in cases if case["expected_status"] == "needs_disambiguation"
    ]
    metrics = {
        "recall_at_5": _mean(
            [float(case["first_relevant_rank"] is not None and case["first_relevant_rank"] <= 5) for case in expected_retrieval]
        ),
        "recall_at_10": _mean(
            [float(case["first_relevant_rank"] is not None and case["first_relevant_rank"] <= 10) for case in expected_retrieval]
        ),
        "mrr": _mean([1.0 / case["first_relevant_rank"] for case in retrieval_cases]),
        "ndcg_at_10": _mean(
            [
                1.0 / math.log2(case["first_relevant_rank"] + 1)
                if case["first_relevant_rank"] and case["first_relevant_rank"] <= 10
                else 0.0
                for case in expected_retrieval
            ]
        ),
        "answer_recall": _mean(answer_scores),
        "citation_precision": _mean([case["citation_precision"] for case in cases]),
        "citation_recall": _mean(
            [float(bool(case["citation_ids"])) for case in expected_retrieval]
        ),
        "citation_coverage": _mean([case["citation_coverage"] for case in cases]),
        "refusal_accuracy": _mean([float(case["status_correct"]) for case in refusal_cases]),
        "ambiguity_accuracy": _mean([float(case["status_correct"]) for case in ambiguity_cases]),
        "generation_errors": float(
            sum(case["actual_status"] == "generation_error" for case in cases)
        ),
        "mean_latency_ms": _mean([case["latency_ms"] for case in cases]),
        "throughput_qps": len(cases) / elapsed if elapsed else 0.0,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }
    gates = {name: metrics[name] >= threshold for name, threshold in THRESHOLDS.items()}
    by_category = {}
    for category in sorted({case["category"] for case in cases}):
        group = [case for case in cases if case["category"] == category]
        scores = [case["answer_recall"] for case in group if case["answer_recall"] is not None]
        by_category[category] = {
            "cases": len(group),
            "status_accuracy": round(_mean([float(case["status_correct"]) for case in group]), 4),
            "answer_recall": round(_mean(scores), 4) if scores else None,
            "citation_precision": round(_mean([case["citation_precision"] for case in group]), 4),
        }
        if any("section_rank" in case for case in group):
            by_category[category]["section_recall_at_5"] = round(
                _mean(
                    [
                        float(case.get("section_rank") is not None and case["section_rank"] <= 5)
                        for case in group
                    ]
                ),
                4,
            )
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": service.source_revision,
        "question_count": len(cases),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "gates": gates,
        "by_category": by_category,
        "cases": cases,
    }


def compare_baseline(report: dict, baseline: dict, tolerance: float = 0.03) -> dict[str, bool]:
    baseline_metrics = baseline.get("metrics", {})
    return {
        name: report["metrics"].get(name, 0.0) + tolerance >= baseline_metrics[name]
        for name in THRESHOLDS
        if name in baseline_metrics
    }


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = "".join(
        f"<tr><td>{html.escape(case['id'])}</td><td>{html.escape(case['category'])}</td>"
        f"<td>{html.escape(case['actual_status'])}</td><td>{case['first_relevant_rank'] or ''}</td>"
        f"<td>{case['latency_ms']:.1f}</td></tr>"
        for case in report["cases"]
    )
    metric_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td>"
        f"<td>{report['thresholds'].get(name, '')}</td></tr>"
        for name, value in report["metrics"].items()
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>CIMA RAG evaluation</title><style>body{{font:14px Arial;margin:32px;color:#17211d}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #ccd7d1;padding:8px;text-align:left}}
th{{background:#edf3f0}}.status{{font-size:22px;font-weight:bold}}</style></head><body>
<h1>CIMA RAG evaluation</h1><p class="status">{html.escape(report['status'].upper())}</p>
<p>Source revision: {html.escape(report['source_revision'])}; questions: {report['question_count']}</p>
<h2>Metrics</h2><table><tr><th>Metric</th><th>Value</th><th>Gate</th></tr>{metric_rows}</table>
<h2>Cases</h2><table><tr><th>ID</th><th>Category</th><th>Status</th><th>Rank</th><th>ms</th></tr>{rows}</table>
</body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")

