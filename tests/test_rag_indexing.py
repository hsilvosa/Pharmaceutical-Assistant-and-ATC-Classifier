from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.rag import store
from src.rag.chunking import RegexOffsetTokenizer
from src.rag.data import open_local_source
from src.rag.indexing import build_index
from src.rag.providers import HashEmbedder


def _write(root, table, rows):
    directory = root / table
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part.parquet")


def _source(tmp_path):
    registrations = [str(number) for number in range(1, 41)]
    _write(
        tmp_path,
        "medications",
        [{"registration_number": number, "name": f"Medicine {number}"} for number in registrations],
    )
    _write(tmp_path, "presentations", [])
    _write(
        tmp_path,
        "documents",
        [
            {
                "registration_number": number,
                "document_type": 1,
                "section": "4.1",
                "title": "Use",
                "source_order": 1,
                "content_text": " ".join(f"palabra{number}x{word}" for word in range(120)),
            }
            for number in registrations
        ],
    )
    _write(
        tmp_path,
        "document_links",
        [
            {"registration_number": number, "document_type": 1, "html_url": "https://e.test/", "url": ""}
            for number in registrations
        ],
    )
    _write(tmp_path, "photos", [])
    return open_local_source(tmp_path, revision="fixture")


class _FlakyEmbedder(HashEmbedder):
    def __init__(self, fail_after_calls: int) -> None:
        super().__init__()
        self.remaining = fail_after_calls

    def encode(self, texts):
        if self.remaining == 0:
            raise KeyboardInterrupt("simulated interruption")
        self.remaining -= 1
        return super().encode(texts)


def _build(source, output, embedder, **extra):
    return build_index(
        source=source,
        output=output,
        tokenizer=RegexOffsetTokenizer(),
        embedder=embedder,
        dataset_id="fixture",
        model_revisions={"embedding": "fixture", "reranker": "fixture", "generator": "fixture"},
        batch_size=1,
        max_registrations=None,
        max_tokens=20,
        overlap_tokens=4,
        **extra,
    )


def test_resumed_build_matches_uninterrupted_build(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(store, "WRITE_BATCH_ROWS", 64)
    source = _source(tmp_path / "source")
    clean = _build(source, tmp_path / "clean", HashEmbedder())
    assert clean.counts["chunks"] > 64 * 4

    with pytest.raises(KeyboardInterrupt):
        _build(source, tmp_path / "resumed", _FlakyEmbedder(fail_after_calls=3))
    persisted = store.LanceStoreWriter(tmp_path / "resumed", resume=True).count
    assert 0 < persisted < clean.counts["chunks"]

    resumed = _build(source, tmp_path / "resumed", HashEmbedder(), resume=True)
    assert resumed.counts == clean.counts
    assert resumed.rejected == clean.rejected
    assert resumed.content_checksum == clean.content_checksum
    assert store.LanceStore(tmp_path / "resumed").table.count_rows() == clean.counts["chunks"]


def test_resume_rejects_index_from_different_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(store, "WRITE_BATCH_ROWS", 64)
    source = _source(tmp_path / "source")
    with pytest.raises(KeyboardInterrupt):
        _build(source, tmp_path / "index", _FlakyEmbedder(fail_after_calls=3))
    changed = _source(tmp_path / "other")
    # Different chunk size changes chunk boundaries, so persisted ids must not match.
    with pytest.raises(RuntimeError, match="diverges"):
        build_index(
            source=changed,
            output=tmp_path / "index",
            tokenizer=RegexOffsetTokenizer(),
            embedder=HashEmbedder(),
            dataset_id="fixture",
            model_revisions={},
            batch_size=1,
            max_registrations=None,
            max_tokens=30,
            overlap_tokens=4,
            resume=True,
        )
