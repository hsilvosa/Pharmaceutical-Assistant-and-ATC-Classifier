"""Run the agent trajectory evaluation against the real engine and llama-server.

Usage: python scripts/run_agent_eval.py [--limit N] [--category C] [--out artifacts/agent/eval.json]
Requires llama-server on CIMA_RAG_LLAMA_BASE_URL and CIMA_RAG_INDEX_DIR=indexes/cima-bge.
"""
from __future__ import annotations

import argparse
import ssl
import sys
import time
from pathlib import Path

# A broken certificate in the Windows store makes ssl.create_default_context() fail on import.
_original = ssl.SSLContext._load_windows_store_certs  # type: ignore[attr-defined]


def _tolerant(self, storename, purpose):  # noqa: ANN001
    try:
        return _original(self, storename, purpose)
    except ssl.SSLError:
        return None


if sys.platform == "win32":
    ssl.SSLContext._load_windows_store_certs = _tolerant  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.evaluation import load_cases, run_cases, write_report  # noqa: E402
from src.agent.factory import build_agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval/agent_cases.jsonl")
    parser.add_argument("--out", default="artifacts/agent/eval.json")
    parser.add_argument("--traces", default="artifacts/agent/traces.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", default="")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]
    runner = build_agent(trace_path=args.traces)
    started = time.time()
    rows, summary = run_cases(cases, runner.run)
    for row in rows:
        mark = "ok  " if row["passed"] else "FAIL"
        print(f"{mark} {row['id']:<44} {row['status']:<22} {row['termination_reason']:<22} {row['tool_sequence']}")
    print(summary)
    write_report(args.out, rows, summary, {"cases_file": args.cases, "wall_seconds": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
