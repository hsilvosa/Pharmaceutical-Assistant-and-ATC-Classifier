from __future__ import annotations

from typing import Any, Protocol

from .models import Observation, ToolCall, TraceEvent, Turn


class LLM(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], remaining_seconds: float
    ) -> Turn: ...


class ToolExecutor(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...

    def execute(self, call: ToolCall, language: str, remaining_seconds: float) -> Observation: ...


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...
