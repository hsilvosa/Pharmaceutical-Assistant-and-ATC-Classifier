"""Evaluate the grounded RAG on eval/freetext.jsonl.

Usage: python scripts/run_freetext_eval.py --split dev [--limit N] [--no-judge] [--out artifacts/freetext/dev.json]
Requires llama-server and CIMA_RAG_INDEX_DIR=indexes/cima-bge.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
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

from src.rag.api import load_service  # noqa: E402
from src.rag.config import Settings  # noqa: E402
from src.rag.freetext import LlamaJudge, evaluate_items, load_items  # noqa: E402
from src.rag.models import QueryRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="eval/freetext.jsonl")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    settings = Settings()
    service = load_service(settings)
    items = load_items(args.items, args.split)
    if args.limit:
        items = items[: args.limit]
    judge = None if args.no_judge else LlamaJudge(settings.llama_base_url, settings.llama_model)

    def answer(item):
        request = QueryRequest(question=item["question"], language=item["language"])
        return service.query_with_trace(request)

    started = time.time()
    result = evaluate_items(items, answer, judge)
    result["meta"] = {"split": args.split, "items_file": args.items, "wall_seconds": round(time.time() - started, 1)}
    out = Path(args.out or f"artifacts/freetext/{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = result["summary"]
    for key, value in summary.items():
        if key != "by_category":
            print(f"{key}: {value}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
