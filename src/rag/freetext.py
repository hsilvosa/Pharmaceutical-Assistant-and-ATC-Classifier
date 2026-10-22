"""Free-text RAG evaluation: retrieval of the gold passage, reference recall and an LLM judge.

The judge is the same local model family that writes the answers, so it can be biased towards them.
Two controls bound that bias: a negative control (answers shuffled across questions must be judged
incorrect) and reporting of the raw verdict counts next to every rate.
"""
from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from .models import AnswerStatus, SearchHit


def load_items(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    return [i for i in items if split in (None, "all", i["split"])]


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return re.sub(r"\s+", " ", "".join(c for c in decomposed if not unicodedata.combining(c))).strip()


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", fold(text)) if len(t) > 3}


def reference_recall(answer: str, reference: str) -> float:
    wanted = content_tokens(reference)
    return len(wanted & content_tokens(answer)) / len(wanted) if wanted else 0.0


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3)]


def retrieval_ranks(item: dict[str, Any], hits: Sequence[SearchHit]) -> tuple[int | None, int | None]:
    """Rank (1-based) of the gold chunk, and of any chunk from the gold section of the gold medicine."""
    chunk_rank = section_rank = None
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        if chunk_rank is None and chunk.chunk_id == item["gold_chunk_id"]:
            chunk_rank = rank
        if (
            section_rank is None
            and chunk.registration_number == item["gold_registration_number"]
            and chunk.document_type == item["gold_document_type"]
            and chunk.section == item["gold_section"]
        ):
            section_rank = rank
    return chunk_rank, section_rank


class LlamaJudge:
    def __init__(self, base_url: str, model: str, seed: int = 7) -> None:
        self.base_url, self.model, self.seed = base_url.rstrip("/"), model, seed

    def _ask(self, prompt: str) -> dict[str, Any] | None:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "seed": self.seed,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=180)
        response.raise_for_status()
        try:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError):
            return None

    def correctness(self, question: str, reference: str, answer: str) -> str | None:
        data = self._ask(
            "Judge a candidate answer against a reference excerpt from the official document.\n"
            f"Question: {question}\nReference excerpt: {reference}\nCandidate answer: {answer}\n\n"
            "correct = the candidate states the fact in the reference (extra detail is fine); "
            "partial = it gives part of the fact or only related information; "
            "incorrect = it misses the fact, contradicts it, or does not answer.\n"
            'Return only JSON: {"verdict": "correct" | "partial" | "incorrect"}'
        )
        verdict = str((data or {}).get("verdict", "")).lower()
        return verdict if verdict in {"correct", "partial", "incorrect"} else None

    def faithful(self, answer: str, passages: Sequence[str]) -> bool | None:
        sources = "\n\n".join(f"[{i}] {p[:2500]}" for i, p in enumerate(passages, start=1))
        data = self._ask(
            "Check whether every factual statement in the answer is supported by the source passages.\n"
            f"Source passages:\n{sources}\n\nAnswer: {answer}\n\n"
            "supported = true only if each factual statement can be found in the passages.\n"
            'Return only JSON: {"supported": true | false}'
        )
        value = (data or {}).get("supported")
        return value if isinstance(value, bool) else None


def evaluate_items(
    items: Sequence[dict[str, Any]],
    answer: Callable[[dict[str, Any]], tuple[Any, list[SearchHit]]],
    judge: LlamaJudge | None,
    seed: int = 13,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        try:
            response, hits = answer(item)
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": item["id"], "category": item["category"], "split": item["split"], "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        chunk_rank, section_rank = retrieval_ranks(item, hits)
        cited = {c.chunk_id for c in response.citations}
        passages = [h.chunk.text for h in hits if h.chunk.chunk_id in cited]
        row: dict[str, Any] = {
            "id": item["id"],
            "category": item["category"],
            "split": item["split"],
            "status": str(response.status.value if hasattr(response.status, "value") else response.status),
            "chunk_rank": chunk_rank,
            "section_rank": section_rank,
            "answer": response.answer,
            "cited_gold_chunk": item["gold_chunk_id"] in cited,
            "reference_recall": round(reference_recall(response.answer, item["reference"]), 3),
        }
        if judge is not None and response.status == AnswerStatus.ANSWERED:
            row["correctness"] = judge.correctness(item["question"], item["reference"], response.answer)
            row["faithful"] = judge.faithful(response.answer, passages) if passages else None
        rows.append(row)

    control: list[str | None] = []
    answered_rows = [r for r in rows if r.get("status") == "answered" and r.get("answer")]
    if judge is not None and len(answered_rows) > 3:
        rng = random.Random(seed)
        by_id = {i["id"]: i for i in items}
        for row in answered_rows:
            other = rng.choice([r for r in answered_rows if r["id"] != row["id"]])
            source = by_id[row["id"]]
            control.append(judge.correctness(source["question"], source["reference"], other["answer"]))
    return {"rows": rows, "summary": summarize(rows, control)}


def _rate(flags: Sequence[bool]) -> dict[str, Any]:
    n, k = len(flags), sum(flags)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def summarize(rows: Sequence[dict[str, Any]], control: Sequence[str | None]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r["status"] == "answered"]
    graded = [r for r in answered if r.get("correctness")]
    faithful = [r for r in answered if r.get("faithful") is not None]
    summary: dict[str, Any] = {
        "items": len(rows),
        "errors": len(rows) - len(ok),
        "answered": _rate([r["status"] == "answered" for r in ok]),
        "gold_chunk_in_context": _rate([r["chunk_rank"] is not None for r in ok]),
        "gold_section_in_context": _rate([r["section_rank"] is not None for r in ok]),
        "gold_chunk_cited": _rate([r["cited_gold_chunk"] for r in ok]),
        "mean_reference_recall": round(sum(r["reference_recall"] for r in answered) / len(answered), 3) if answered else None,
        "mrr_chunk": round(sum(1 / r["chunk_rank"] for r in ok if r["chunk_rank"]) / len(ok), 3) if ok else None,
    }
    if graded:
        summary["judge_correct"] = _rate([r["correctness"] == "correct" for r in graded])
        summary["judge_correct_or_partial"] = _rate([r["correctness"] in {"correct", "partial"} for r in graded])
        summary["judge_verdicts"] = {v: sum(r["correctness"] == v for r in graded) for v in ("correct", "partial", "incorrect")}
    if faithful:
        summary["judge_faithful"] = _rate([bool(r["faithful"]) for r in faithful])
    valid_control = [c for c in control if c]
    if valid_control:
        summary["negative_control_false_positive"] = _rate([c != "incorrect" for c in valid_control])
    categories: dict[str, list[dict[str, Any]]] = {}
    for r in ok:
        categories.setdefault(r["category"], []).append(r)
    summary["by_category"] = {
        name: {
            "n": len(group),
            "gold_section_in_context": round(sum(g["section_rank"] is not None for g in group) / len(group), 2),
            "judge_correct": round(sum(g.get("correctness") == "correct" for g in group) / len(group), 2),
        }
        for name, group in sorted(categories.items())
    }
    return summary
