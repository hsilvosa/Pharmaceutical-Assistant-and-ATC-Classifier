from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .models import IndexManifest, QueryRequest, QueryResponse
from .providers import (
    CrossEncoderReranker,
    ExtractiveFixtureGenerator,
    HashEmbedder,
    LlamaCppGenerator,
    SentenceTransformerEmbedder,
    TokenOverlapReranker,
)
from .retrieval import EntityResolver, HybridRetriever
from .service import RagService
from .store import LanceStore

WEB_DIR = Path(__file__).with_name("web")


def load_service(settings: Settings) -> RagService:
    manifest_path = settings.index_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Index manifest not found: {manifest_path}")
    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    lock = settings.load_model_lock()
    if not settings.development_mode:
        expected = {name: str(details["revision"]) for name, details in lock.items()}
        if any(manifest.model_revisions.get(name) != revision for name, revision in expected.items()):
            raise RuntimeError("Index model revisions do not match model-lock.json")
    store = LanceStore(settings.index_dir)
    if settings.fixture_mode:
        retriever = HybridRetriever(
            store,
            HashEmbedder(),
            TokenOverlapReranker(),
            dense_candidates=settings.dense_candidates,
            lexical_candidates=settings.lexical_candidates,
            rerank_candidates=settings.rerank_candidates,
            final_passages=settings.context_passages,
        )
        return RagService(
            retriever,
            EntityResolver(store.medicines()),
            ExtractiveFixtureGenerator(),
            manifest.dataset_revision,
        )
    embedding = lock["embedding"]
    reranker = lock["reranker"]
    retriever = HybridRetriever(
        store,
        SentenceTransformerEmbedder(
            embedding["repo_id"], embedding["revision"], device=settings.device
        ),
        CrossEncoderReranker(
            reranker["repo_id"], reranker["revision"], device=settings.device
        ),
        dense_candidates=settings.dense_candidates,
        lexical_candidates=settings.lexical_candidates,
        rerank_candidates=settings.rerank_candidates,
        final_passages=settings.context_passages,
    )
    return RagService(
        retriever,
        EntityResolver(store.medicines()),
        LlamaCppGenerator(settings.llama_base_url, settings.llama_model),
        manifest.dataset_revision,
    )


def create_app(
    service: RagService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(
        title="CIMA Evidence",
        version="0.1.0",
        description="Grounded retrieval over official AEMPS CIMA documentation",
    )
    active_settings = settings
    active_service = service
    startup_error: str | None = None
    if active_service is None:
        try:
            active_settings = active_settings or get_settings()
            active_service = load_service(active_settings)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            startup_error = str(exc)

    def service_dependency() -> RagService:
        if active_service is None:
            raise HTTPException(status_code=503, detail=startup_error or "Service unavailable")
        return active_service

    @app.get("/api/v1/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok" if active_service is not None else "degraded",
            "ready": active_service is not None,
            "detail": startup_error or "",
        }

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query_endpoint(
        request: QueryRequest,
        rag: RagService = Depends(service_dependency),  # noqa: B008
    ) -> QueryResponse:
        try:
            return rag.query(request)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Local model unavailable: {exc}") from exc

    @app.get("/api/v1/medicines/search")
    def medicine_search(
        q: str = Query(min_length=2, max_length=200),
        rag: RagService = Depends(service_dependency),  # noqa: B008
    ) -> list[dict]:
        return [item.model_dump() for item in rag.resolver.search(q)]

    @app.get("/api/v1/evaluations/latest")
    def latest_evaluation() -> dict:
        if active_settings is None:
            return {"status": "not_run", "metrics": {}}
        reports = sorted(
            active_settings.evaluations_dir.glob("*/report.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if reports:
            return json.loads(reports[0].read_text(encoding="utf-8"))
        if active_settings.baseline_path.exists():
            return json.loads(active_settings.baseline_path.read_text(encoding="utf-8"))
        return {"status": "not_run", "metrics": {}}

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()

