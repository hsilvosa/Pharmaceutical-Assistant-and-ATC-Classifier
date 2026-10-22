"""Closed-loop load test for an OpenAI-compatible chat endpoint (llama-server or vLLM).

For each concurrency level, N workers send requests back to back for a fixed number of requests.
Reports time to first token (TTFT), per-request decode speed, end-to-end latency percentiles and
aggregate token throughput. Prompts are unique per request and prompt caching is disabled, so the
numbers reflect cold prefill, as in a RAG workload where every question retrieves different passages.

Usage: python scripts/loadtest_llm.py --levels 1,2,4 --requests 12 --prompt-tokens 1500 --max-tokens 128
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

FILLER = (
    "El principio activo se absorbe de forma rápida tras la administración oral y alcanza la concentración "
    "máxima en plasma en dos a tres horas. La eliminación es principalmente renal y la semivida es de seis "
    "horas. No se han observado interacciones relevantes con alimentos. "
)


def make_prompt(index: int, target_tokens: int) -> str:
    # About 1.6 tokens per Spanish word with the Qwen tokenizer; a unique header defeats prefix caching.
    words = int(target_tokens / 1.6)
    body = (FILLER * (words // len(FILLER.split()) + 1)).split()[:words]
    return (
        f"Consulta número {index}. Lee el texto y resume en una frase qué dice sobre la eliminación.\n\n"
        + " ".join(body)
    )


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
        "ignore_eos": True,  # fixed output length so runs are comparable
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    first = None
    chunks = 0
    usage: dict = {}
    try:
        async with client.stream("POST", f"{url}/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    chunks += 1
                    if first is None:
                        first = time.perf_counter()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    end = time.perf_counter()
    completion = usage.get("completion_tokens") or chunks
    decode = (end - first) if first else 0.0
    return {
        "ttft": (first - started) if first else None,
        "latency": end - started,
        "completion_tokens": completion,
        "prompt_tokens": usage.get("prompt_tokens"),
        "decode_tps": (completion - 1) / decode if decode > 0 and completion > 1 else None,
    }


async def run_level(url: str, model: str, concurrency: int, requests: int, prompt_tokens: int, max_tokens: int) -> dict:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(requests):
        queue.put_nowait(index)
    results: list[dict] = []
    limits = httpx.Limits(max_connections=concurrency + 2)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        async def worker() -> None:
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                prompt = make_prompt(index + concurrency * 1000, prompt_tokens)
                results.append(await one_request(client, url, model, prompt, max_tokens))

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started
    ok = [r for r in results if "error" not in r]

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)

    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    latency = [r["latency"] for r in ok]
    tps = [r["decode_tps"] for r in ok if r["decode_tps"]]
    tokens = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "errors": len(results) - len(ok),
        "wall_s": round(wall, 1),
        "ttft_p50_s": pct(ttft, 0.5),
        "ttft_p95_s": pct(ttft, 0.95),
        "latency_p50_s": pct(latency, 0.5),
        "latency_p95_s": pct(latency, 0.95),
        "decode_tps_per_request": round(statistics.mean(tps), 1) if tps else None,
        "throughput_tok_s": round(tokens / wall, 1) if wall else None,
        "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
    }


def gpu_memory_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


async def main_async(args: argparse.Namespace) -> dict:
    warm = await run_level(args.url, args.model, 1, 1, 200, 16)  # load weights, compile kernels
    levels = []
    for concurrency in [int(x) for x in args.levels.split(",")]:
        result = await run_level(args.url, args.model, concurrency, args.requests, args.prompt_tokens, args.max_tokens)
        result["gpu_mem_mib"] = gpu_memory_mib()
        levels.append(result)
        print(json.dumps(result), flush=True)
    return {"label": args.label, "url": args.url, "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
            "warmup_errors": warm["errors"], "levels": levels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
