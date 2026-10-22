"""Measure the request guards on a labelled file ({"q": ..., "label": "block"|"allow"} per line).

Usage: python scripts/eval_guard.py eval/guard_redteam_heldout.jsonl [--llm]
--llm adds the model-based intent layer (needs llama-server on CIMA_RAG_LLAMA_BASE_URL).
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path

if sys.platform == "win32":
    _orig = ssl.SSLContext._load_windows_store_certs  # type: ignore[attr-defined]

    def _tolerant(self, storename, purpose):  # noqa: ANN001
        try:
            return _orig(self, storename, purpose)
        except ssl.SSLError:
            return None

    ssl.SSLContext._load_windows_store_certs = _tolerant  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.guards import blocked_request  # noqa: E402
from src.agent.intent import LlamaIntentClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.path).read_text(encoding="utf-8").splitlines() if line.strip()]
    classifier = LlamaIntentClassifier(args.base_url, args.model) if args.llm else None
    missed, over = [], []
    for row in rows:
        reason = blocked_request(row["q"])
        if reason is None and classifier is not None:
            label = classifier.classify(row["q"])
            if label in {"personal_advice", "prompt_injection"}:
                reason = f"intent:{label}"
        blocked = reason is not None
        if row["label"] == "block" and not blocked:
            missed.append(row["q"])
        if row["label"] == "allow" and blocked:
            over.append((row["q"], reason))
    block_total = sum(r["label"] == "block" for r in rows)
    allow_total = len(rows) - block_total
    print(f"{args.path} llm={args.llm}: blocked {block_total - len(missed)}/{block_total}, allowed {allow_total - len(over)}/{allow_total}")
    if args.verbose or True:
        for q in missed:
            print("  MISSED:", q)
        for q, reason in over:
            print("  OVER-BLOCKED:", q, reason)


if __name__ == "__main__":
    main()
