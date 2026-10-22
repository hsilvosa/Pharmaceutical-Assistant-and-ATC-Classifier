"""Start llama-server with several configurations, load test each one and write a comparison.

Usage: python scripts/sweep_llama_server.py [--requests 8] [--only A,B]
Each configuration gets a fresh server so VRAM and slots are not shared between runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "qwen3" / "Qwen3-8B-Q4_K_M.gguf"
PORT = 8081

CONFIGS = {
    "A_baseline_np1": ["--parallel", "1", "--ctx-size", "8192", "-fa", "off"],
    "B_flashattn_np1": ["--parallel", "1", "--ctx-size", "8192", "-fa", "on"],
    "C_np4_fa": ["--parallel", "4", "--ctx-size", "16384", "-fa", "on"],
    "D_np4_fa_kvq8": ["--parallel", "4", "--ctx-size", "16384", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"],
}


def start(extra: list[str]) -> subprocess.Popen:
    command = [
        "llama-server", "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--n-gpu-layers", "99", "--jinja", "--metrics", *extra,
    ]
    log = open(ROOT / "artifacts" / "tmp" / "sweep_server.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def wait_healthy(timeout: float = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/PID", str(process.pid)], capture_output=True)
    time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default="eval/serving_llamacpp.json")
    args = parser.parse_args()

    (ROOT / "artifacts" / "tmp").mkdir(parents=True, exist_ok=True)
    names = [n for n in CONFIGS if not args.only or n.split("_")[0] in args.only.split(",")]
    report = {}
    for name in names:
        print(f"== {name}", flush=True)
        server = start(CONFIGS[name])
        try:
            if not wait_healthy():
                report[name] = {"error": "server did not become healthy"}
                continue
            output = ROOT / "artifacts" / "tmp" / f"load_{name}.json"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "loadtest_llm.py"), "--url", f"http://127.0.0.1:{PORT}/v1",
                 "--levels", args.levels, "--requests", str(args.requests), "--label", name, "--out", str(output)],
                check=False,
            )
            report[name] = {"flags": CONFIGS[name], **json.loads(output.read_text(encoding="utf-8"))}
        finally:
            stop(server)
    Path(ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("written", args.out)


if __name__ == "__main__":
    main()
