"""Serve the agent API (uvicorn) with the Windows certificate-store workaround.

Usage: python scripts/serve_agent.py [--port 8000]
Requires llama-server (CIMA_RAG_LLAMA_BASE_URL) and CIMA_RAG_INDEX_DIR=indexes/cima-bge.
Metrics for Prometheus are exposed on /metrics.
"""
from __future__ import annotations

import argparse
import ssl
import sys
from pathlib import Path

if sys.platform == "win32":  # a broken certificate in the Windows store breaks ssl.create_default_context()
    _orig = ssl.SSLContext._load_windows_store_certs  # type: ignore[attr-defined]

    def _tolerant(self, storename, purpose):  # noqa: ANN001
        try:
            return _orig(self, storename, purpose)
        except ssl.SSLError:
            return None

    ssl.SSLContext._load_windows_store_certs = _tolerant  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.agent.api:app", host=args.host, port=args.port, log_level="info")
