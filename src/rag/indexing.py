from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .chunking import HuggingFaceOffsetTokenizer, OffsetTokenizer
from .data import SourceData, iter_batches, prepare_records
from .models import IndexManifest
from .providers import Embedder
from .store import LanceStore


def build_index(
    *,
    source: SourceData,
    output: Path,
    tokenizer: OffsetTokenizer,
    embedder: Embedder,
    dataset_id: str,
    model_revisions: dict[str, str],
    batch_size: int,
    max_registrations: int | None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> IndexManifest:
    chunks, medicines, rejected = prepare_records(
        source,
        tokenizer,
        max_registrations=max_registrations,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    vectors: list[list[float]] = []
    for batch in iter_batches(chunks, batch_size):
        vectors.extend(embedder.encode([chunk.embedding_text for chunk in batch]))
    LanceStore.build(output, chunks, vectors, medicines)
    checksum = hashlib.sha256()
    for chunk, vector in zip(chunks, vectors, strict=True):
        checksum.update(chunk.chunk_id.encode("ascii"))
        checksum.update(json.dumps(vector, separators=(",", ":")).encode("ascii"))
    manifest = IndexManifest(
        dataset_id=dataset_id,
        dataset_revision=source.revision,
        source_checksums=source.checksums,
        model_revisions=model_revisions,
        chunking={"max_tokens": max_tokens, "overlap_tokens": overlap_tokens},
        counts={"chunks": len(chunks), "medicines": len(medicines)},
        rejected=rejected,
        content_checksum=checksum.hexdigest(),
        created_at=datetime.now(UTC).isoformat(),
    )
    (output / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def production_tokenizer(model_lock: dict) -> HuggingFaceOffsetTokenizer:
    details = model_lock["embedding"]
    return HuggingFaceOffsetTokenizer(details["repo_id"], details["revision"])


