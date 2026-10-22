from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from .models import ToolCall, Turn


class LlamaServerLLM:
    """OpenAI-compatible chat client for llama-server started with --jinja."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3",
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], remaining_seconds: float) -> Turn:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            # Qwen3 otherwise spends the token budget on reasoning before the tool call.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=max(1.0, remaining_seconds)
        )
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        usage = body.get("usage") or {}
        calls = message.get("tool_calls") or []
        call = None
        if len(calls) == 1:  # several calls in one turn count as a protocol error
            function = calls[0].get("function", {})
            try:
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if isinstance(arguments, dict) and function.get("name"):
                    call = ToolCall(id=calls[0].get("id") or "call_0", name=function["name"], arguments=arguments)
            except (json.JSONDecodeError, ValueError):
                call = None
        return Turn(
            tool_call=call,
            text=message.get("content") or "",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class ScriptedLLM:
    """Deterministic stand-in for tests: returns a fixed turn per step or computes one."""

    def __init__(self, script: Sequence[Turn | ToolCall | None] | Callable[[list[dict[str, Any]]], Turn]) -> None:
        self._script = script
        self._index = 0
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], remaining_seconds: float) -> Turn:
        self.calls += 1
        if callable(self._script):
            return self._script(messages)
        item = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        if isinstance(item, Turn):
            return item
        if item is None:
            return Turn(text="I will think about it.")
        return Turn(tool_call=item)
