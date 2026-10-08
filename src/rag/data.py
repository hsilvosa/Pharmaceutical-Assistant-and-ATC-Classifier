from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .chunking import OffsetTokenizer, chunk_document
from .models import Chunk, MedicineCandidate

TABLES = ("medications", "presentations", "documents", "document_links", "photos")


@dataclass
class SourceData:
    root: Path
    revision: str
    checksums: dict[str, str]

    def _table_path(self, table: str) -> Path | None:
        candidates = [self.root / table, self.root / "data" / table]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            matches = list(self.root.rglob(f"{table}/*.parquet"))
            if not matches:
                return None
            path = matches[0].parent
        return path

    def rows(self, table: str) -> list[dict[str, Any]]:
        path = self._table_path(table)
        if path is None:
            return []
        return ds.dataset(path, format="parquet").to_table().to_pylist()

    def iter_rows(self, table: str, batch_size: int = 512) -> Iterable[dict[str, Any]]:
        """Stream a table row by row without materializing it in memory."""
        path = self._table_path(table)
        if path is None:
            return
        # ParquetFile keeps only the current batch resident; dataset scanners prefetch GBs.
        for file in sorted(path.glob("*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_local_source(root: Path, revision: str = "local") -> SourceData:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below {root}")
    checksums = {str(path.relative_to(root)).replace(chr(92), "/"): _sha256(path) for path in files}
    return SourceData(root=root, revision=revision, checksums=checksums)


def download_hf_source(repo_id: str, revision: str, cache_dir: Path) -> SourceData:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["data/**/*.parquet", "*.parquet", "artifacts/checksums.json"],
        )
    )
    return open_local_source(root, revision=revision)


def prepare_records(
    source: SourceData,
    tokenizer: OffsetTokenizer,
    *,
    max_registrations: int | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> tuple[list[Chunk], list[MedicineCandidate], dict[str, int]]:
    chunk_stream, entities, rejected = stream_records(
        source,
        tokenizer,
        max_registrations=max_registrations,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunks = list(chunk_stream)
    return chunks, entities, rejected


def stream_records(
    source: SourceData,
    tokenizer: OffsetTokenizer,
    *,
    max_registrations: int | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> tuple[Iterator[Chunk], list[MedicineCandidate], dict[str, int]]:
    """Return a lazy chunk iterator plus medicines and a rejection counter.

    The counter is final only once the iterator has been exhausted.
    """
    medications = source.rows("medications")
    if max_registrations is not None:
        allowed = {
            str(row["registration_number"])
            for row in sorted(medications, key=lambda row: str(row["registration_number"]))[
                :max_registrations
            ]
        }
        medications = [row for row in medications if str(row["registration_number"]) in allowed]
    else:
        allowed = {str(row["registration_number"]) for row in medications}

    medicine_names = {
        str(row["registration_number"]): str(row.get("name") or "") for row in medications
    }
    presentations = [
        row
        for row in source.rows("presentations")
        if str(row["registration_number"]) in allowed
    ]
    links = [
        row for row in source.rows("document_links") if str(row["registration_number"]) in allowed
    ]
    photos = [row for row in source.rows("photos") if str(row["registration_number"]) in allowed]

    url_by_key: dict[tuple[str, int], str] = {}
    for row in links:
        key = (str(row["registration_number"]), int(row.get("document_type") or 0))
        url_by_key.setdefault(key, str(row.get("html_url") or row.get("url") or ""))
    photo_by_registration: dict[str, str] = {}
    for row in photos:
        photo_by_registration.setdefault(
            str(row["registration_number"]), str(row.get("url") or "")
        )

    entities: dict[tuple[str, str, str], MedicineCandidate] = {}
    for row in medications:
        registration = str(row["registration_number"])
        candidate = MedicineCandidate(
            registration_number=registration,
            name=str(row.get("name") or ""),
            photo_url=photo_by_registration.get(registration, ""),
        )
        entities[(registration, "", "")] = candidate
    for row in presentations:
        registration = str(row["registration_number"])
        candidate = MedicineCandidate(
            registration_number=registration,
            name=medicine_names.get(registration, ""),
            presentation=str(row.get("name") or ""),
            national_code=str(row.get("national_code") or ""),
            photo_url=photo_by_registration.get(registration, ""),
        )
        entities[(registration, candidate.presentation, candidate.national_code)] = candidate

    rejected = {"empty": 0, "duplicate": 0, "missing_medicine": 0, "missing_url": 0}

    def generate() -> Iterator[Chunk]:
        seen: set[tuple[str, int, str, bytes]] = set()
        for row in source.iter_rows("documents"):
            registration = str(row["registration_number"])
            if registration not in allowed:
                continue
            medicine_name = medicine_names.get(registration, "")
            if not medicine_name:
                rejected["missing_medicine"] += 1
                continue
            text = str(row.get("content_text") or "").strip()
            if not text:
                rejected["empty"] += 1
                continue
            document_type = int(row.get("document_type") or 0)
            dedupe_key = (
                registration,
                document_type,
                str(row.get("section") or ""),
                hashlib.sha256(text.encode("utf-8")).digest(),
            )
            if dedupe_key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(dedupe_key)
            source_url = url_by_key.get((registration, document_type), "")
            if not source_url:
                rejected["missing_url"] += 1
            yield from chunk_document(
                dataset_revision=source.revision,
                registration_number=registration,
                medicine_name=medicine_name,
                document_type=document_type,
                section=str(row.get("section") or ""),
                title=str(row.get("title") or ""),
                source_order=int(row.get("source_order") or 0),
                text=text,
                source_url=source_url,
                photo_url=photo_by_registration.get(registration, ""),
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

    return generate(), list(entities.values()), rejected


def iter_batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch
