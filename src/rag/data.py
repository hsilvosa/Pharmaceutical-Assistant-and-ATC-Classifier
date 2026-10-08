from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .chunking import OffsetTokenizer, _digest, chunk_document
from .models import Chunk, MedicineCandidate

CATALOG_TITLE = "Datos de catalogo AEMPS"
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

    def generate_all() -> Iterator[Chunk]:
        # Catalogue passages come last so an existing document-only index can be extended
        # with --resume instead of being rebuilt.
        yield from generate()
        yield from catalog_chunks(source, medications, photo_by_registration)

    return generate_all(), list(entities.values()), rejected


def _by_registration(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["registration_number"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row.get("position") or 0))
    return grouped


def catalog_chunks(
    source: SourceData,
    medications: Iterable[dict[str, Any]],
    photo_by_registration: dict[str, str],
) -> Iterator[Chunk]:
    """One citable passage per medicine with its structured AEMPS catalogue fields."""
    ingredients = _by_registration(source.iter_rows("active_ingredients"))
    routes = _by_registration(source.iter_rows("administration_routes"))
    atc = _by_registration(source.iter_rows("atc_codes"))
    for row in medications:
        registration = str(row["registration_number"])
        name = str(row.get("name") or "")
        if not name:
            continue
        form = str(row.get("pharmaceutical_form_name") or "").strip()
        conditions = str(row.get("prescription_conditions") or "").strip()
        ingredient_text = "; ".join(
            " ".join(
                part
                for part in (
                    str(item.get("ingredient_name") or "").strip(),
                    str(item.get("quantity") or "").strip(),
                    str(item.get("unit") or "").strip(),
                )
                if part
            )
            for item in ingredients.get(registration, [])
        )
        route_text = "; ".join(
            str(item.get("route_name") or "").strip() for item in routes.get(registration, [])
        )
        atc_text = "; ".join(
            f"{item.get('atc_code') or ''} ({item.get('atc_name') or ''})"
            for item in atc.get(registration, [])
        )
        lines = [
            f"Datos de catalogo AEMPS de {name} (registro {registration}).",
            f"Forma farmaceutica (pharmaceutical form): {form or 'no consta'}.",
            "Comercializacion (marketed): "
            + ("si, comercializado (yes, marketed)." if row.get("marketed") else "no comercializado (no, not marketed)."),
            "Receta (prescription required): "
            + ("si, requiere receta (yes)" if row.get("prescription_required") else "no requiere receta (no)")
            + (f"; condiciones de prescripcion: {conditions}." if conditions else "."),
            f"Principios activos (active ingredients): {ingredient_text or 'no constan'}.",
            f"Vias de administracion (administration routes): {route_text or 'no constan'}.",
            f"Clasificacion ATC (ATC classification): {atc_text or 'no consta'}.",
        ]
        text = "\n".join(lines)
        content_hash = _digest(text)
        yield Chunk(
            chunk_id=_digest("|".join([source.revision, registration, "catalog", content_hash])),
            registration_number=registration,
            medicine_name=name,
            document_type=0,
            section="catalogo",
            title=CATALOG_TITLE,
            source_order=0,
            char_start=0,
            char_end=len(text),
            text=text,
            embedding_text=f"{name} | {CATALOG_TITLE}\n{text}",
            source_url=f"https://cima.aemps.es/cima/publico/detalle.html?nregistro={registration}",
            photo_url=photo_by_registration.get(registration, ""),
            content_hash=content_hash,
        )


def iter_batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch
