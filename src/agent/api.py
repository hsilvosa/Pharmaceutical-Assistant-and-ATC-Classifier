from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import MetricsSink
from .models import AgentRequest, AgentResult
from .runner import AgentRunner


def create_app(
    runner_factory: Callable[[], AgentRunner] | None = None,
    metrics: MetricsSink | None = None,
    eager: bool = False,
) -> FastAPI:
    """Thin HTTP layer. The runner is built lazily so importing the app stays cheap."""
    state: dict[str, AgentRunner] = {}
    metrics = metrics or MetricsSink()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if eager:  # load the models at startup so the first request does not pay about a minute
            runner()
        yield

    app = FastAPI(title="CIMA pharma agent", version="0.1", lifespan=lifespan)

    def runner() -> AgentRunner:
        if "runner" not in state:
            if runner_factory is None:
                from .factory import build_agent

                state["runner"] = build_agent(extra_sink=metrics)
            else:
                state["runner"] = runner_factory()
        return state["runner"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/agent/run", response_model=AgentResult)
    def run(request: AgentRequest) -> AgentResult:
        try:
            return runner().run(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)[:200]) from exc

    return app


app = create_app(eager=True)
