from __future__ import annotations

import hashlib
import html
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .data import SourceData
from .models import AnswerStatus, BenchmarkItem, QueryRequest
from .service import RagService

THRESHOLDS = {
    "recall_at_5": 0.85,
    "mrr": 0.75,
    "citation_precision": 0.95,
    "citation_coverage": 0.95,
    "answer_f1": 0.85,
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
        grouped["atc"][str(row["registration_number"])].extend(
            [str(row.get("atc_code") or ""), str(row.get("atc_name") or "")]
        )
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


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold()))


def _answer_f1(answer: str, expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    expected_tokens = _tokens(" ".join(expected))
    answer_tokens = _tokens(answer)
    if not expected_tokens or not answer_tokens:
        return 0.0
    overlap = len(expected_tokens & answer_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        response, hits = service.query_with_trace(
            QueryRequest(
                question=item.question,
                registration_number=item.registration_number,
                language=item.language,
            )
        )
        relevant = set(item.relevant_registration_numbers)
        ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.chunk.registration_number in relevant
        ]
        first_rank = min(ranks) if ranks else None
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
                "answer_f1": _answer_f1(response.answer, item.expected_values),
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
    answer_scores = [case["answer_f1"] for case in cases if case["answer_f1"] is not None]
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
        "answer_f1": _mean(answer_scores),
        "citation_precision": _mean([case["citation_precision"] for case in cases]),
        "citation_recall": _mean(
            [float(bool(case["citation_ids"])) for case in expected_retrieval]
        ),
        "citation_coverage": _mean([case["citation_coverage"] for case in cases]),
        "refusal_accuracy": _mean([float(case["status_correct"]) for case in refusal_cases]),
        "ambiguity_accuracy": _mean([float(case["status_correct"]) for case in ambiguity_cases]),
        "mean_latency_ms": _mean([case["latency_ms"] for case in cases]),
        "throughput_qps": len(cases) / elapsed if elapsed else 0.0,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }
    gates = {name: metrics[name] >= threshold for name, threshold in THRESHOLDS.items()}
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": service.source_revision,
        "question_count": len(cases),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "gates": gates,
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

