from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .api import load_service
from .chunking import RegexOffsetTokenizer
from .config import Settings
from .data import download_hf_source, open_local_source
from .evaluation import (
    compare_baseline,
    evaluate,
    generate_benchmark,
    load_benchmark,
    write_benchmark,
    write_report,
)
from .indexing import build_index, production_tokenizer
from .providers import HashEmbedder, SentenceTransformerEmbedder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded local RAG over AEMPS CIMA")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    build = index_commands.add_parser("build")
    build.add_argument("--local-data", type=Path)
    build.add_argument("--max-registrations", type=int)
    build.add_argument("--fixture", action="store_true")
    build.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted build from the chunks already written to the index",
    )

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--benchmark", type=Path, action="append")
    evaluation.add_argument("--generate", action="store_true")
    evaluation.add_argument("--local-data", type=Path)
    evaluation.add_argument("--output", type=Path)

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--write-lock", action="store_true")
    return parser


def _settings(development: bool = False, **overrides) -> Settings:
    if development:
        overrides["development_mode"] = True
    return Settings(**overrides)


def _source(settings: Settings, local_data: Path | None):
    if local_data or settings.local_data_dir:
        return open_local_source(local_data or settings.local_data_dir, revision="local")
    return download_hf_source(
        settings.dataset_id, settings.dataset_revision, Path("data/huggingface-cache")
    )


def _build(args: argparse.Namespace) -> int:
    settings = _settings(development=args.fixture or bool(args.local_data), local_data_dir=args.local_data)
    lock = settings.load_model_lock()
    source = _source(settings, args.local_data)
    if args.fixture:
        tokenizer = RegexOffsetTokenizer()
        embedder = HashEmbedder()
        model_revisions = {"embedding": "fixture", "reranker": "fixture", "generator": "fixture"}
    else:
        tokenizer = production_tokenizer(lock)
        details = lock["embedding"]
        embedder = SentenceTransformerEmbedder(
            details["repo_id"], details["revision"], settings.device
        )
        model_revisions = {name: str(details["revision"]) for name, details in lock.items()}
    if settings.index_dir.exists() and not args.resume:
        shutil.rmtree(settings.index_dir)
    manifest = build_index(
        source=source,
        output=settings.index_dir,
        tokenizer=tokenizer,
        embedder=embedder,
        dataset_id=settings.dataset_id,
        model_revisions=model_revisions,
        batch_size=settings.embedding_batch_size,
        max_registrations=args.max_registrations,
        resume=args.resume,
    )
    print(manifest.model_dump_json(indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.rag.api:app", host=args.host, port=args.port, reload=False)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    settings = _settings(development=bool(args.local_data), local_data_dir=args.local_data)
    paths = args.benchmark or [Path("eval/generated.jsonl"), Path("eval/curated.jsonl")]
    if args.generate:
        source = _source(settings, args.local_data)
        write_benchmark(Path("eval/generated.jsonl"), generate_benchmark(source))
    items = load_benchmark(paths)
    if not items:
        raise SystemExit("No benchmark questions found")
    report = evaluate(load_service(settings), items)
    if settings.baseline_path.exists():
        baseline = json.loads(settings.baseline_path.read_text(encoding="utf-8"))
        report["regression_gates"] = compare_baseline(report, baseline)
        if report["regression_gates"] and not all(report["regression_gates"].values()):
            report["status"] = "fail"
    output = args.output or settings.evaluations_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    write_report(output, report)
    print(json.dumps({key: report[key] for key in ("status", "metrics", "gates")}, indent=2))
    return 0 if report["status"] == "pass" else 2


def _doctor(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    settings = _settings(development=True)
    lock = settings.load_model_lock()
    api = HfApi()
    resolved = {}
    for name, details in lock.items():
        info = api.model_info(details["repo_id"], revision=None)
        resolved[name] = {**details, "revision": info.sha}
    dataset = api.dataset_info(settings.dataset_id, revision=None)
    result = {"dataset_revision": dataset.sha, "models": resolved}
    if args.write_lock:
        settings.model_lock.write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    try:
        import torch

        result["cuda"] = bool(torch.cuda.is_available())
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except ImportError:
        result["cuda"] = False
        result["gpu"] = "torch not installed"
    result["llama_server"] = shutil.which("llama-server") or "not found on PATH"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "index":
        return _build(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "evaluate":
        return _evaluate(args)
    return _doctor(args)


if __name__ == "__main__":
    raise SystemExit(main())

