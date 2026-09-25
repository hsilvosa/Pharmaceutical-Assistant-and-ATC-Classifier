from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Chunk, MedicineCandidate, SearchHit


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
        import lancedb
        from lancedb.index import FTS

        if not chunks:
            raise ValueError("Cannot build an empty index")
        path.mkdir(parents=True, exist_ok=True)
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            row = chunk.model_dump()
            row["vector"] = np.asarray(vector, dtype=np.float32).tolist()
            rows.append(row)
        db = lancedb.connect(path)
        table = db.create_table("chunks", data=rows, mode="overwrite")
        if len(rows) >= 256:
            table.create_index(vector_column_name="vector", metric="cosine", replace=True)
        table.create_index(
            "embedding_text",
            config=FTS(language="Spanish", lower_case=True, ascii_folding=True),
            replace=True,
        )
        (path / "entities.json").write_text(
            json.dumps([item.model_dump() for item in medicines], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cls(path)

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

