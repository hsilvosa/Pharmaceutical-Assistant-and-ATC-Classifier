from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CIMA_RAG_", env_file=".env", extra="ignore"
    )

    dataset_id: str = "HSilvosa/aemps-cima"
    dataset_revision: str = "05da2ddf387a47a274384084068462ad8e2ff908"
    local_data_dir: Path | None = None
    index_dir: Path = Path("indexes/cima-full")
    model_lock: Path = Path("model-lock.json")
    evaluations_dir: Path = Path("artifacts/evaluations")
    baseline_path: Path = Path("eval/baseline.json")
    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_model: str = "Qwen3-8B-Q4_K_M.gguf"
    development_mode: bool = False
    fixture_mode: bool = False
    device: str = "cuda"
    embedding_batch_size: int = Field(default=16, ge=1)
    dense_candidates: int = Field(default=50, ge=1)
    lexical_candidates: int = Field(default=50, ge=1)
    rerank_candidates: int = Field(default=30, ge=1)
    context_passages: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def require_pinned_dataset(self) -> Settings:
        if (
            not self.development_mode
            and not self.local_data_dir
            and (
                len(self.dataset_revision) < 7
                or self.dataset_revision.lower() in {"main", "latest"}
            )
        ):
            raise ValueError(
                "CIMA_RAG_DATASET_REVISION must be a commit SHA outside development mode"
            )
        return self

    def load_model_lock(self) -> dict[str, Any]:
        payload = json.loads(self.model_lock.read_text(encoding="utf-8"))
        if not self.development_mode:
            for name, details in payload.items():
                revision = str(details.get("revision", ""))
                if len(revision) < 7 or revision.startswith("PIN_"):
                    raise ValueError(f"Model {name} does not have a pinned revision")
        return payload


@lru_cache
def get_settings() -> Settings:
    return Settings()
