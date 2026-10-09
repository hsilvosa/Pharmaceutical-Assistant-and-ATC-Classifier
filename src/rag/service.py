from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Sequence

from pydantic import ValidationError

from .models import (
    AnswerStatus,
    Citation,
    GeneratedAnswer,
    QueryRequest,
    QueryResponse,
    SearchHit,
)
from .providers import Generator
from .retrieval import EntityResolver, HybridRetriever

PERSONALIZED_PATTERNS = (
    r"\b(que|qué) (medicamento|dosis) (debo|tengo que)\b",
    r"\bcuanto debo tomar\b",
    r"\bdiagnost[ií]ca(me|rme)?\b",
    r"\b(puedo|podria|podría|deberia|debería) (tomar|usar|dejar)(lo|la|los|las)?\b",
    r"\b(que|qué) (me )?recomiendas\b",
    r"\brecomiendame\b",
    r"\bshould i (take|stop|increase|reduce)\b",
    r"\bcan i (take|use|stop)\b",
    r"\bwhat dose should i\b",
    r"\bdiagnose me\b",
)


def detect_language(question: str, requested: str) -> str:
    if requested in {"es", "en"}:
        return requested
    lowered = question.casefold()
    spanish_markers = (" qué ", " cuál ", " medicamento", "según", "puedo", "tiene")
    padded = f" {lowered} "
    return "es" if any(marker in padded for marker in spanish_markers) else "en"


def personalized_medical_request(question: str) -> bool:
    decomposed = unicodedata.normalize("NFKD", question.casefold())
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    normalized = re.sub(r"\s+", " ", unaccented)
    return any(re.search(pattern, normalized) for pattern in PERSONALIZED_PATTERNS)


def _quote(text: str, limit: int = 360) -> str:
    quote = text.strip()
    if len(quote) <= limit:
        return quote
    boundary = quote.rfind(" ", 0, limit)
    return quote[: boundary if boundary > 80 else limit]


RETRY_BREVITY_NOTE = (
    "\n\nIMPORTANT: the previous answer was too long and was cut off. Answer in at most 120 words. "
    "For long lists, name the most important items and state that the list is not exhaustive."
)


class GenerationError(ValueError):
    """The generator never produced valid structured output; keeps the retrieved hits."""

    def __init__(self, message: str, hits: Sequence[SearchHit]) -> None:
        super().__init__(message)
        self.hits = list(hits)


class RagService:
    def __init__(
        self,
        retriever: HybridRetriever,
        resolver: EntityResolver,
        generator: Generator,
        source_revision: str,
        min_rerank_score: float = 0.0,
    ) -> None:
        self.min_rerank_score = min_rerank_score
        self.retriever = retriever
        self.resolver = resolver
        self.generator = generator
        self.source_revision = source_revision

    def query(self, request: QueryRequest) -> QueryResponse:
        response, _ = self.query_with_trace(request)
        return response

    def query_with_trace(self, request: QueryRequest) -> tuple[QueryResponse, list[SearchHit]]:
        started = time.perf_counter()
        language = detect_language(request.question, request.language)
        if personalized_medical_request(request.question):
            answer = (
                "No puedo recomendar un tratamiento o una dosis personal. Consulta la ficha oficial "
                "y a un profesional sanitario."
                if language == "es"
                else "I cannot recommend personal treatment or dosing. Consult the official document "
                "and a healthcare professional."
            )
            return (
                QueryResponse(
                    status=AnswerStatus.REFUSED,
                    answer=answer,
                    source_revision=self.source_revision,
                    language=language,
                    latency_ms={"total": (time.perf_counter() - started) * 1000},
                ),
                [],
            )

        if request.registration_number and not self.resolver.has_registration(
            request.registration_number
        ):
            answer = (
                "El número de registro indicado no existe en el índice."
                if language == "es"
                else "The supplied registration number is not present in the index."
            )
            return (
                QueryResponse(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    answer=answer,
                    source_revision=self.source_revision,
                    language=language,
                    latency_ms={"total": (time.perf_counter() - started) * 1000},
                ),
                [],
            )

        registration, candidates = self.resolver.resolve(
            request.question, request.registration_number
        )
        if candidates:
            return (
                QueryResponse(
                    status=AnswerStatus.NEEDS_DISAMBIGUATION,
                    answer=(
                        "Selecciona la presentación correcta."
                        if language == "es"
                        else "Select the intended presentation."
                    ),
                    candidates=candidates,
                    source_revision=self.source_revision,
                    language=language,
                    latency_ms={"total": (time.perf_counter() - started) * 1000},
                ),
                [],
            )

        retrieval_started = time.perf_counter()
        hits = self.retriever.retrieve(request.question, registration)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        if not hits or hits[0].score < self.min_rerank_score:
            return (
                QueryResponse(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    answer=(
                        "No encontré evidencia suficiente en la documentación indexada."
                        if language == "es"
                        else "I found insufficient evidence in the indexed documentation."
                    ),
                    source_revision=self.source_revision,
                    language=language,
                    latency_ms={"retrieval": retrieval_ms, "total": retrieval_ms},
                ),
                hits,
            )

        generation_started = time.perf_counter()
        try:
            generated = self._generate_validated(request.question, language, hits)
        except ValueError as exc:
            raise GenerationError(str(exc), hits) from exc
        generation_ms = (time.perf_counter() - generation_started) * 1000
        citations = self._citations(generated, hits)
        status = AnswerStatus(generated.status)
        answer = generated.answer
        if status == AnswerStatus.ANSWERED and not citations:
            status = AnswerStatus.INSUFFICIENT_EVIDENCE
            answer = (
                "La respuesta generada no incluía citas verificables."
                if language == "es"
                else "The generated answer did not contain verifiable citations."
            )
        return (
            QueryResponse(
                status=status,
                answer=answer,
                citations=citations,
                source_revision=self.source_revision,
                language=language,
                latency_ms={
                    "retrieval": retrieval_ms,
                    "generation": generation_ms,
                    "total": (time.perf_counter() - started) * 1000,
                },
            ),
            hits,
        )

    def _generate_validated(
        self, question: str, language: str, hits: Sequence[SearchHit]
    ) -> GeneratedAnswer:
        last_error: Exception | None = None
        for attempt in range(2):
            # A plain retry repeats the same output (the seed is fixed). A truncated JSON answer
            # usually means the model listed every item, so the retry asks for a short answer.
            prompt = question if attempt == 0 else question + RETRY_BREVITY_NOTE
            raw = self.generator.generate(prompt, language, hits)
            try:
                payload = json.loads(raw)
                return GeneratedAnswer.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        raise ValueError("Generator returned invalid structured output twice") from last_error

    @staticmethod
    def _citations(generated: GeneratedAnswer, hits: Sequence[SearchHit]) -> list[Citation]:
        available = {f"C{index}": hit.chunk for index, hit in enumerate(hits, start=1)}
        result = []
        seen: set[str] = set()
        for citation_id in generated.citation_ids:
            if citation_id in seen or citation_id not in available:
                continue
            if f"[{citation_id}]" not in generated.answer:
                continue
            seen.add(citation_id)
            chunk = available[citation_id]
            result.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    registration_number=chunk.registration_number,
                    medicine_name=chunk.medicine_name,
                    document_type=chunk.document_type,
                    section=chunk.section,
                    title=chunk.title,
                    quote=_quote(chunk.text),
                    url=chunk.source_url,
                )
            )
        return result
