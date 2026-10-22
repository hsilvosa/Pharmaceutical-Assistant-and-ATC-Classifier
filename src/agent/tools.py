from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from .guards import blocked_request
from .models import EvidenceItem, Observation, ToolCall

MAX_TEXT_CHARS = 600
MAX_PASSAGE_CHARS = 6000


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _fn(
        "search_medications",
        "Find medicines in the AEMPS catalogue by name or active ingredient. Returns candidates "
        "with registration numbers. Navigation only: results are NOT citable evidence.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _fn(
        "query_evidence",
        "Retrieve official AEMPS passages (leaflet, technical sheet, catalogue) that answer a factual "
        "question. Returns citable evidence items with ids. Pass registration_number when known.",
        {"question": {"type": "string"}, "registration_number": {"type": "string"}},
        ["question"],
    ),
    _fn(
        "classify_atc",
        "Predict the ATC anatomical group (level 1) from a medicine description. A hypothesis only: "
        "verify with query_evidence before stating it.",
        {"text": {"type": "string"}},
        ["text"],
    ),
]


class EngineTools:
    """Allowlisted adapters over CIMAMedicalAssistantEngine and ATCClassifierEngine."""

    def __init__(self, engine: Any, atc_factory: Callable[[], Any] | None = None, workers: int = 2) -> None:
        self._engine = engine
        self._atc_factory = atc_factory
        self._atc: Any = None
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-tool")

    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(self, call: ToolCall, language: str, remaining_seconds: float) -> Observation:
        handler = {
            "search_medications": self._search,
            "query_evidence": self._evidence,
            "classify_atc": self._classify,
        }.get(call.name)
        if handler is None:
            return Observation(status="error", error=f"unknown_tool:{call.name}")
        started = time.monotonic()
        future = self._pool.submit(handler, call.arguments, language)
        try:
            observation = future.result(timeout=max(0.1, remaining_seconds))
        except FutureTimeout:
            return Observation(status="error", error="tool_timeout")
        except Exception as exc:  # noqa: BLE001  # tool failures are observations, never crashes
            return Observation(status="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        observation.elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        return observation

    @staticmethod
    def _text(arguments: dict[str, Any], key: str) -> str | None:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()[:MAX_TEXT_CHARS]

    def _search(self, arguments: dict[str, Any], language: str) -> Observation:
        query = self._text(arguments, "query")
        if query is None:
            return Observation(status="error", error="invalid_arguments:query")
        rows = self._engine.search_medications(query, limit=5)
        candidates = [
            {
                "registration_number": str(row.get("registration_number")),
                "name": row.get("name"),
                "active_ingredients": row.get("active_ingredients"),
                "atc_code": row.get("atc_code"),
            }
            for row in rows
        ]
        return Observation(status="ok", payload={"candidates": candidates}, candidates=candidates)

    def _evidence(self, arguments: dict[str, Any], language: str) -> Observation:
        question = self._text(arguments, "question")
        if question is None:
            return Observation(status="error", error="invalid_arguments:question")
        if blocked_request(question) is not None:
            return Observation(status="refused", payload={"reason": "personalized_medical_request"})
        registration = arguments.get("registration_number")
        if registration is not None:
            registration = str(registration).strip()
            if not registration.isdigit() or len(registration) > 12:
                return Observation(status="error", error="invalid_arguments:registration_number")
        if hasattr(self._engine, "query_evidence_trace"):
            response, hits = self._engine.query_evidence_trace(question, registration or None, language)
            # Citations carry a 360-character quote; the full passage lets the renderer pick the right span.
            passages = {hit.chunk.chunk_id: hit.chunk.text for hit in hits}
        else:
            response = self._engine.query_evidence(question, registration or None, language)
            passages = {}
        evidence = [
            EvidenceItem(
                evidence_id=citation.citation_id,
                registration_number=citation.registration_number,
                medicine_name=citation.medicine_name,
                section=citation.section,
                quote=passages.get(citation.chunk_id, citation.quote)[:MAX_PASSAGE_CHARS],
                url=citation.url,
            )
            for citation in response.citations
        ]
        candidates = [
            {"registration_number": c.registration_number, "name": c.name, "presentation": c.presentation}
            for c in response.candidates
        ]
        status = str(response.status.value if hasattr(response.status, "value") else response.status)
        payload: dict[str, Any] = {"retrieval_status": status}
        if candidates:
            payload["candidates"] = candidates
        return Observation(status=status, payload=payload, evidence=evidence, candidates=candidates)

    def _classify(self, arguments: dict[str, Any], language: str) -> Observation:
        text = self._text(arguments, "text")
        if text is None:
            return Observation(status="error", error="invalid_arguments:text")
        if self._atc is None:
            if self._atc_factory is None:
                return Observation(status="error", error="classifier_unavailable")
            self._atc = self._atc_factory()
        result = self._atc.predict(text, top_k=3)
        if "error" in result:
            return Observation(status="error", error=str(result["error"])[:100])
        groups = [
            {"code": p["atc_level1_code"], "name": p["atc_level1_name"]} for p in result["all_predictions"]
        ]
        # Confidence is deliberately withheld: softmax calibration is unverified.
        return Observation(
            status="ok",
            payload={"hypothesis": groups, "note": "Unverified model prediction, not evidence."},
        )
