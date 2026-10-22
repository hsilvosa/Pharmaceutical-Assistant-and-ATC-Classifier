from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

from .models import TraceEvent
from .ports import TraceSink

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 45, 90)


class MetricsSink:
    """Trace sink that turns agent events into Prometheus metrics (labels stay low-cardinality)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.runs = Counter(
            "agent_runs_total", "Finished agent runs", ["status", "termination"], registry=self.registry
        )
        self.run_seconds = Histogram(
            "agent_run_seconds", "End-to-end run time", buckets=LATENCY_BUCKETS, registry=self.registry
        )
        self.steps = Histogram(
            "agent_run_steps", "Model decisions per run", buckets=(1, 2, 3, 4, 5, 6), registry=self.registry
        )
        self.llm_seconds = Histogram(
            "agent_llm_seconds", "One model decision", buckets=LATENCY_BUCKETS, registry=self.registry
        )
        self.tool_seconds = Histogram(
            "agent_tool_seconds", "One tool execution", ["tool"], buckets=LATENCY_BUCKETS, registry=self.registry
        )
        self.tool_calls = Counter(
            "agent_tool_calls_total", "Tool executions", ["tool", "status"], registry=self.registry
        )
        self.tokens = Counter("agent_llm_tokens_total", "Model tokens", ["kind"], registry=self.registry)
        self.llm_errors = Counter("agent_llm_errors_total", "Failed model calls", registry=self.registry)

    def emit(self, event: TraceEvent) -> None:
        if event.kind == "decision":
            self.llm_seconds.observe(event.latency_ms / 1000)
            if event.prompt_tokens:
                self.tokens.labels("prompt").inc(event.prompt_tokens)
            if event.completion_tokens:
                self.tokens.labels("completion").inc(event.completion_tokens)
        elif event.kind == "tool" and event.tool:
            self.tool_seconds.labels(event.tool).observe(event.latency_ms / 1000)
            self.tool_calls.labels(event.tool, event.status or "unknown").inc()
        elif event.kind == "llm_error":
            self.llm_errors.inc()
        elif event.kind == "final":
            reason = event.summary.split(":")[0] if event.summary.startswith(("input_guard", "intent_guard")) else event.summary
            self.runs.labels(event.status, reason).inc()
            self.run_seconds.observe(event.latency_ms / 1000)
            self.steps.observe(max(1, event.step))


class FanoutSink:
    def __init__(self, *sinks: TraceSink) -> None:
        self.sinks = sinks

    def emit(self, event: TraceEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)
