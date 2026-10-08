from __future__ import annotations

import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunking import HuggingFaceOffsetTokenizer, OffsetTokenizer
from .data import SourceData, iter_batches, stream_records
from .models import IndexManifest
from .providers import Embedder
from .store import LanceStoreWriter

EMBED_CALL_BATCHES = 64


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
    resume: bool = False,
) -> IndexManifest:
    chunk_stream, medicines, rejected = stream_records(
        source,
        tokenizer,
        max_registrations=max_registrations,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    # Large calls let the embedder sort texts by length internally, which cuts padding waste.
    call_size = batch_size * EMBED_CALL_BATCHES
    writer = LanceStoreWriter(output, resume=resume)
    checksum = hashlib.sha256()
    done = 0
    if writer.count:
        # Re-walk the deterministic chunk stream past what is already persisted, verifying
        # every id, so the resumed index is identical to an uninterrupted build.
        print(f"resuming after {writer.count} persisted chunks", file=sys.stderr, flush=True)
        for (stored_id, vector), chunk in zip(writer.iter_persisted(), chunk_stream, strict=False):
            if chunk.chunk_id != stored_id:
                raise RuntimeError(
                    f"Persisted index diverges from source at chunk {done}; rebuild without resume"
                )
            checksum.update(stored_id.encode("ascii"))
            checksum.update(vector.tobytes())
            done += 1
        if done != writer.count:
            raise RuntimeError("Source ended before the persisted chunks were matched")
        print(f"verified {done} persisted chunks", file=sys.stderr, flush=True)
    resumed = done
    calls = 0
    started = time.monotonic()
    for batch in iter_batches(chunk_stream, call_size):
        vectors = np.asarray(
            embedder.encode([chunk.embedding_text for chunk in batch]), dtype=np.float32
        )
        writer.add(batch, vectors)
        for chunk, vector in zip(batch, vectors, strict=True):
            checksum.update(chunk.chunk_id.encode("ascii"))
            checksum.update(vector.tobytes())
        done += len(batch)
        calls += 1
        if calls % 20 == 0:
            rate = (done - resumed) / (time.monotonic() - started)
            print(f"embedded {done} chunks ({rate:.1f}/s)", file=sys.stderr, flush=True)
    print(f"embedded {done} chunks; building indexes", file=sys.stderr, flush=True)
    writer.finish(medicines)
    manifest = IndexManifest(
        dataset_id=dataset_id,
        dataset_revision=source.revision,
        source_checksums=source.checksums,
        model_revisions=model_revisions,
        chunking={"max_tokens": max_tokens, "overlap_tokens": overlap_tokens},
        counts={"chunks": done, "medicines": len(medicines)},
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


