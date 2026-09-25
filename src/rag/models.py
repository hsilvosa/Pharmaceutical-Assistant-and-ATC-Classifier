from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSED = "refused"
    NEEDS_DISAMBIGUATION = "needs_disambiguation"


class Chunk(BaseModel):
    chunk_id: str
    registration_number: str
    medicine_name: str
    document_type: int
    section: str
    title: str
    source_order: int
    char_start: int
    char_end: int
    text: str
    embedding_text: str
    source_url: str
    photo_url: str = ""
    content_hash: str


class MedicineCandidate(BaseModel):
    registration_number: str
    name: str
    presentation: str = ""
    national_code: str = ""
    photo_url: str = ""


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    registration_number: str
    medicine_name: str
    document_type: int
    section: str
    title: str
    quote: str
    url: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    registration_number: str | None = None
    language: Literal["auto", "es", "en"] = "auto"


class GeneratedAnswer(BaseModel):
    status: Literal["answered", "insufficient_evidence", "refused"]
    answer: str
    citation_ids: list[str] = Field(default_factory=list)
    refusal_reason: str = ""


class QueryResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    candidates: list[MedicineCandidate] = Field(default_factory=list)
    source_revision: str
    language: Literal["es", "en"]
    latency_ms: dict[str, float] = Field(default_factory=dict)


class IndexManifest(BaseModel):
    schema_version: int = 1
    dataset_id: str
    dataset_revision: str
    source_checksums: dict[str, str]
    model_revisions: dict[str, str]
    chunking: dict[str, int]
    counts: dict[str, int]
    rejected: dict[str, int]
    content_checksum: str
    created_at: str


class BenchmarkItem(BaseModel):
    id: str
    split: Literal["dev", "test"]
    category: str
    language: Literal["es", "en"]
    question: str
    registration_number: str | None = None
    relevant_registration_numbers: list[str] = Field(default_factory=list)
    expected_values: list[str] = Field(default_factory=list)
    expected_status: AnswerStatus = AnswerStatus.ANSWERED
    metadata: dict[str, Any] = Field(default_factory=dict)


