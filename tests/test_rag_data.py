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
    assert len(chunks) == 1
    assert chunks[0].source_url == "https://example.test/1"
    assert chunks[0].photo_url == "image.jpg"
    assert rejected["duplicate"] == 1


