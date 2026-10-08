from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from src.rag.chunking import RegexOffsetTokenizer
from src.rag.data import open_local_source, prepare_records


def _write(root, table, rows):
    directory = root / table
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part.parquet")


def test_local_source_joins_and_deduplicates(tmp_path) -> None:
    _write(tmp_path, "medications", [{"registration_number": "1", "name": "Example"}])
    _write(tmp_path, "presentations", [])
    _write(
        tmp_path,
        "documents",
        [
            {
                "registration_number": "1",
                "document_type": 1,
                "section": "4.1",
                "title": "Use",
                "source_order": 1,
                "content_text": "No procede.",
            },
            {
                "registration_number": "1",
                "document_type": 1,
                "section": "4.1",
                "title": "Use",
                "source_order": 1,
                "content_text": "No procede.",
            },
        ],
    )
    _write(
        tmp_path,
        "document_links",
        [
            {
                "registration_number": "1",
                "document_type": 1,
                "html_url": "https://example.test/1",
                "url": "",
            }
        ],
    )
    _write(tmp_path, "photos", [{"registration_number": "1", "url": "image.jpg"}])
    source = open_local_source(tmp_path, revision="fixture")
    chunks, _, rejected = prepare_records(source, RegexOffsetTokenizer())
    document_chunks = [chunk for chunk in chunks if chunk.document_type != 0]
    assert len(document_chunks) == 1
    assert document_chunks[0].source_url == "https://example.test/1"
    assert document_chunks[0].photo_url == "image.jpg"
    assert rejected["duplicate"] == 1
    assert chunks[-1].section == "catalogo"


def test_catalog_chunk_exposes_structured_fields(tmp_path) -> None:
    _write(
        tmp_path,
        "medications",
        [
            {
                "registration_number": "7",
                "name": "Ejemplo 10 mg",
                "marketed": False,
                "prescription_required": True,
                "pharmaceutical_form_name": "COMPRIMIDO",
                "prescription_conditions": "Sujeto a prescripcion",
            }
        ],
    )
    _write(
        tmp_path,
        "active_ingredients",
        [
            {"registration_number": "7", "position": 1, "ingredient_name": "B", "quantity": "5", "unit": "mg"},
            {"registration_number": "7", "position": 0, "ingredient_name": "A", "quantity": "10", "unit": "mg"},
        ],
    )
    _write(
        tmp_path,
        "administration_routes",
        [{"registration_number": "7", "position": 0, "route_name": "VIA ORAL"}],
    )
    _write(
        tmp_path,
        "atc_codes",
        [{"registration_number": "7", "position": 0, "atc_code": "N02BE01", "atc_name": "Paracetamol"}],
    )
    _write(tmp_path, "documents", [])
    source = open_local_source(tmp_path, revision="fixture")
    chunks, _, _ = prepare_records(source, RegexOffsetTokenizer())
    assert len(chunks) == 1
    catalog = chunks[0]
    assert catalog.registration_number == "7"
    assert catalog.source_url.endswith("nregistro=7")
    assert "COMPRIMIDO" in catalog.text
    assert "no comercializado" in catalog.text
    assert "requiere receta (yes)" in catalog.text
    assert catalog.text.index("A 10 mg") < catalog.text.index("B 5 mg")
    assert "VIA ORAL" in catalog.text
    assert "N02BE01 (Paracetamol)" in catalog.text


