from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Chunk, MedicineCandidate, SearchHit


WRITE_BATCH_ROWS = 10_000


class SearchStore(Protocol):
    def dense_search(
        self, vector: Sequence[float], limit: int, registration_number: str | None = None
    ) -> list[SearchHit]: ...

    def lexical_search(
        self, query: str, limit: int, registration_number: str | None = None
    ) -> list[SearchHit]: ...

    def medicines(self) -> list[MedicineCandidate]: ...


def _where_registration(registration_number: str | None) -> str | None:
    if registration_number is None:
        return None
    safe = registration_number.replace("'", "''")
    return f"registration_number = '{safe}'"


class LanceStoreWriter:
    """Incrementally writes chunks and vectors so large indexes never sit fully in memory."""

    def __init__(self, path: Path, *, resume: bool = False) -> None:
        import lancedb

        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = lancedb.connect(path)
        self.table = None
        self.count = 0
        self._rows: list[dict] = []
        if resume and "chunks" in self.db.list_tables().tables:
            self.table = self.db.open_table("chunks")
            self.count = self.table.count_rows()

    def iter_persisted(self, batch_size: int = 10_000):
        """Yield (chunk_id, vector) for rows already written, in insertion order."""
        if self.table is None:
            return
        for offset in range(0, self.count, batch_size):
            page = (
                self.table.search()
                .select(["chunk_id", "vector"])
                .limit(batch_size)
                .offset(offset)
                .to_arrow()
            )
            ids = page.column("chunk_id").to_pylist()
            vectors = (
                page.column("vector")
                .combine_chunks()
                .values.to_numpy(zero_copy_only=False)
                .astype(np.float32, copy=False)
                .reshape(len(ids), -1)
            )
            yield from zip(ids, vectors, strict=True)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            row = chunk.model_dump()
            row["vector"] = np.asarray(vector, dtype=np.float32).tolist()
            self._rows.append(row)
        if len(self._rows) >= WRITE_BATCH_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        if self.table is None:
            self.table = self.db.create_table("chunks", data=self._rows, mode="overwrite")
        else:
            self.table.add(self._rows)
        self.count += len(self._rows)
        self._rows = []

    def finish(self, medicines: Sequence[MedicineCandidate]) -> LanceStore:
        from lancedb.index import FTS

        self._flush()
        if self.table is None:
            raise ValueError("Cannot build an empty index")
        if self.count >= 256:
            self.table.create_index(vector_column_name="vector", metric="cosine", replace=True)
        self.table.create_index(
            "embedding_text",
            config=FTS(language="Spanish", lower_case=True, ascii_folding=True),
            replace=True,
        )
        (self.path / "entities.json").write_text(
            json.dumps([item.model_dump() for item in medicines], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return LanceStore(self.path)


class LanceStore:
    def __init__(self, path: Path) -> None:
        import lancedb

        self.path = path
        self.db = lancedb.connect(path)
        self.table = self.db.open_table("chunks")
        self._medicines = [
            MedicineCandidate.model_validate(item)
            for item in json.loads((path / "entities.json").read_text(encoding="utf-8"))
        ]

    @classmethod
    def build(
        cls,
        path: Path,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        medicines: Sequence[MedicineCandidate],
    ) -> LanceStore:
        writer = LanceStoreWriter(path)
        writer.add(chunks, vectors)
        return writer.finish(medicines)

    @staticmethod
    def _chunk(row: dict) -> Chunk:
        row.pop("vector", None)
        row.pop("_distance", None)
        row.pop("_score", None)
        row.pop("_relevance_score", None)
        return Chunk.model_validate(row)

    def dense_search(
        self, vector: Sequence[float], limit: int, registration_number: str | None = None
    ) -> list[SearchHit]:
        query = self.table.search(list(vector), query_type="vector").limit(limit)
        where = _where_registration(registration_number)
        if where:
            query = query.where(where, prefilter=True)
        rows = query.to_list()
        return [
            SearchHit(
                chunk=self._chunk(dict(row)),
                score=1.0 - float(row.get("_distance", 1.0)),
                dense_rank=index,
            )
            for index, row in enumerate(rows, start=1)
        ]

    def lexical_search(
        self, query: str, limit: int, registration_number: str | None = None
    ) -> list[SearchHit]:
        builder = self.table.search(query, query_type="fts", fts_columns="embedding_text").limit(
            limit
        )
        where = _where_registration(registration_number)
        if where:
            builder = builder.where(where)
        rows = builder.to_list()
        return [
            SearchHit(
                chunk=self._chunk(dict(row)),
                score=float(row.get("_score", row.get("_relevance_score", 0.0))),
                lexical_rank=index,
            )
            for index, row in enumerate(rows, start=1)
        ]

    def medicines(self) -> list[MedicineCandidate]:
        return list(self._medicines)


class InMemoryStore:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        medicines: Sequence[MedicineCandidate],
    ) -> None:
        self.chunks = list(chunks)
        self.vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        self._medicines = list(medicines)

    def dense_search(
        self, vector: Sequence[float], limit: int, registration_number: str | None = None
    ) -> list[SearchHit]:
        query = np.asarray(vector, dtype=np.float32)
        scored = []
        for chunk, candidate in zip(self.chunks, self.vectors, strict=True):
            if registration_number and chunk.registration_number != registration_number:
                continue
            denom = float(np.linalg.norm(query) * np.linalg.norm(candidate)) or 1.0
            scored.append((float(np.dot(query, candidate) / denom), chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchHit(chunk=chunk, score=score, dense_rank=index)
            for index, (score, chunk) in enumerate(scored[:limit], start=1)
        ]

    def lexical_search(
        self, query: str, limit: int, registration_number: str | None = None
    ) -> list[SearchHit]:
        query_tokens = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for chunk in self.chunks:
            if registration_number and chunk.registration_number != registration_number:
                continue
            tokens = set(re.findall(r"\w+", chunk.embedding_text.casefold()))
            overlap = len(query_tokens & tokens)
            score = overlap / math.sqrt(max(len(query_tokens) * len(tokens), 1))
            if overlap:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchHit(chunk=chunk, score=score, lexical_rank=index)
            for index, (score, chunk) in enumerate(scored[:limit], start=1)
        ]

    def medicines(self) -> list[MedicineCandidate]:
        return list(self._medicines)

