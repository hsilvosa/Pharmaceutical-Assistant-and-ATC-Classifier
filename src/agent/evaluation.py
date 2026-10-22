from __future__ import annotations

import json
import statistics
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import AgentRequest, AgentResult


class AgentCase(BaseModel):
    id: str
    category: str
    question: str
    registration_number: str | None = None
    expected_status: str
    accepted_statuses: list[str] = Field(default_factory=list)
    relevant_registration_numbers: list[str] = Field(default_factory=list)
    expected_values: list[str] = Field(default_factory=list)
    expect_no_tools: bool = False
    max_steps: int = 5


def load_cases(path: str | Path) -> list[AgentCase]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [AgentCase.model_validate_json(line) for line in lines if line.strip()]


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def score_case(case: AgentCase, result: AgentResult) -> dict[str, Any]:
    status_ok = result.status in {case.expected_status, *case.accepted_statuses}
    checks: dict[str, bool] = {"status": status_ok, "steps": result.steps <= case.max_steps}
    if case.expect_no_tools:
        checks["no_tools"] = not result.tool_sequence
    if case.expected_status == "answered":
        checks["has_citation"] = bool(result.citations)
        if case.relevant_registration_numbers:
            wanted = set(case.relevant_registration_numbers)
            # Sibling presentations share a medicine name, so a name match in the question also counts.
            folded_question = _fold(case.question)
            checks["relevant_source"] = bool(result.citations) and all(
                c.registration_number in wanted or _fold(c.medicine_name) in folded_question
                for c in result.citations
            )
        if case.expected_values:
            haystack = _fold(result.answer)
            hits = sum(1 for value in case.expected_values if _fold(value) in haystack)
            checks["values"] = hits / len(case.expected_values) >= 0.5
    else:
        # Non-answers must not carry citations.
        checks["no_unsupported_citation"] = not result.citations
    return {
        "id": case.id,
        "category": case.category,
        "passed": all(checks.values()),
        "checks": checks,
        "status": result.status,
        "expected_status": case.expected_status,
        "termination_reason": result.termination_reason,
        "steps": result.steps,
        "tool_sequence": result.tool_sequence,
        "elapsed_ms": result.elapsed_ms,
        "answer": result.answer[:700],
        "cited_registrations": sorted({c.registration_number for c in result.citations}),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["passed"])
    latencies = sorted(row["elapsed_ms"] for row in rows)
    check_totals: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for name, ok in row["checks"].items():
            check_totals[name].append(ok)
    return {
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "pass_rate": round(sum(row["passed"] for row in rows) / max(1, len(rows)), 3),
        "by_category": {k: {"passed": sum(v), "total": len(v)} for k, v in sorted(by_category.items())},
        "by_check": {k: {"passed": sum(v), "total": len(v)} for k, v in sorted(check_totals.items())},
        "mean_steps": round(statistics.mean(row["steps"] for row in rows), 2) if rows else 0,
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else 0,
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
        "termination": {
            reason: sum(1 for row in rows if row["termination_reason"] == reason)
            for reason in sorted({row["termination_reason"] for row in rows})
        },
    }


def run_cases(
    cases: Iterable[AgentCase], run: Callable[[AgentRequest], AgentResult]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for case in cases:
        request = AgentRequest(question=case.question, registration_number=case.registration_number)
        try:
            rows.append(score_case(case, run(request)))
        except Exception as exc:  # noqa: BLE001  # a crash is a failed case, not an aborted evaluation
            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "passed": False,
                    "checks": {"no_crash": False},
                    "status": "crash",
                    "expected_status": case.expected_status,
                    "termination_reason": f"crash:{type(exc).__name__}",
                    "steps": 0,
                    "tool_sequence": [],
                    "elapsed_ms": 0.0,
                }
            )
    return rows, summarize(rows)


def write_report(path: str | Path, rows: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
