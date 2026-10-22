from __future__ import annotations

from pathlib import Path

from .intent import LlamaIntentClassifier
from .llm import LlamaServerLLM
from .metrics import FanoutSink
from .models import AgentConfig
from .ports import TraceSink
from .runner import AgentRunner
from .tools import EngineTools
from .trace import JsonlTraceSink


def build_agent(
    *,
    trace_path: str | Path = "artifacts/agent/traces.jsonl",
    config: AgentConfig | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    extra_sink: TraceSink | None = None,
    use_intent_guard: bool = True,
) -> AgentRunner:
    """Wire the real engine, the llama-server client and the JSONL trace sink."""
    from src.inference import ATCClassifierEngine, CIMAMedicalAssistantEngine
    from src.rag.config import Settings

    settings = Settings()
    engine = CIMAMedicalAssistantEngine()
    engine._get_rag_service()  # load embedding model, reranker and index now, not on the first request
    llm = LlamaServerLLM(llm_base_url or settings.llama_base_url, llm_model or settings.llama_model)
    tools = EngineTools(engine, atc_factory=ATCClassifierEngine)
    sink: TraceSink = JsonlTraceSink(trace_path)
    if extra_sink is not None:
        sink = FanoutSink(sink, extra_sink)
    intent = LlamaIntentClassifier(llm.base_url, llm.model) if use_intent_guard else None
    return AgentRunner(llm, tools, sink, config, intent=intent)
