from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_table(root: Path, name: str, rows: list[dict]) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-00000.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic CIMA RAG fixture")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    medicines = [
        {
            "registration_number": "10001",
            "name": "Ácido Ejemplo 100 mg",
            "prescription_required": True,
            "marketed": True,
            "pharmaceutical_form_name": "Comprimido",
        },
        {
            "registration_number": "10002",
            "name": "Marca Común 10 mg",
            "prescription_required": False,
            "marketed": True,
            "pharmaceutical_form_name": "Solución oral",
        },
        {
            "registration_number": "10003",
            "name": "Marca Común 20 mg",
            "prescription_required": True,
            "marketed": False,
            "pharmaceutical_form_name": "Cápsula",
        },
    ]
    write_table(root, "medications", medicines)
    write_table(
        root,
        "presentations",
        [
            {
                "registration_number": "10001",
                "national_code": "700001",
                "name": "Ácido Ejemplo 100 mg, 20 comprimidos",
            },
            {
                "registration_number": "10002",
                "national_code": "700002",
                "name": "Marca Común, frasco",
            },
            {
                "registration_number": "10003",
                "national_code": "700003",
                "name": "Marca Común, 30 cápsulas",
            },
        ],
    )
    documents = [
        {
            "registration_number": "10001",
            "document_type": 1,
            "section": "4.1",
            "title": "Indicaciones terapéuticas",
            "source_order": 1,
            "content_text": "Está indicado para el alivio temporal del dolor leve.",
        },
        {
            "registration_number": "10001",
            "document_type": 1,
            "section": "4.3",
            "title": "Contraindicaciones",
            "source_order": 3,
            "content_text": "No debe utilizarse en caso de alergia al principio activo.",
        },
        {
            "registration_number": "10002",
            "document_type": 1,
            "section": "2",
            "title": "Composición",
            "source_order": 2,
            "content_text": "Cada mililitro contiene 10 mg del principio activo de ejemplo.",
        },
        {
            "registration_number": "10003",
            "document_type": 1,
            "section": "2",
            "title": "Composición",
            "source_order": 2,
            "content_text": "Cada cápsula contiene 20 mg del principio activo de ejemplo.",
        },
    ]
    write_table(root, "documents", documents)
    write_table(
        root,
        "document_links",
        [
            {
                "registration_number": registration,
                "position": 0,
                "document_type": 1,
                "url": f"https://example.test/{registration}.pdf",
                "html_url": f"https://example.test/{registration}",
            }
            for registration in ("10001", "10002", "10003")
        ],
    )
    write_table(
        root,
        "photos",
        [
            {
                "registration_number": registration,
                "position": 0,
                "url": f"https://placehold.co/160x160/ffffff/176b4a?text={registration}",
            }
            for registration in ("10001", "10002", "10003")
        ],
    )


if __name__ == "__main__":
    main()


