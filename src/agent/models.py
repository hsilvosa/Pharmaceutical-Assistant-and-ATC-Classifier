from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 5
    timeout_seconds: float = 90.0
    max_tool_calls: int = 5
    max_repair_attempts: int = 2
    max_quote_chars: int = 800
    allowed_source_hosts: tuple[str, ...] = ("aemps.es",)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    registration_number: str | None = Field(default=None, pattern=r"^\d{1,12}$")
    language: Literal["auto", "es", "en"] = "auto"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = "call_0"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """One model decision: exactly one tool call, or no call (a protocol error)."""

    tool_call: ToolCall | None = None
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class EvidenceItem(BaseModel):
    evidence_id: str
    registration_number: str
    medicine_name: str
    section: str
    quote: str
    url: str


class Observation(BaseModel):
    """Result of one tool execution. `payload` goes to the model, `evidence` to the registry."""

    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str = ""


class AgentResult(BaseModel):
    run_id: str
    status: Literal["answered", "insufficient_evidence", "refused", "needs_disambiguation"]
    answer: str
    citations: list[EvidenceItem] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    termination_reason: str
    steps: int = 0
    tool_sequence: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class TraceEvent:
    run_id: str
    step: int
    kind: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: str = ""
