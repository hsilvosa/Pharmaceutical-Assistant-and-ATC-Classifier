"""Build eval/freetext.jsonl: diverse questions written from real leaflet and technical-sheet passages.

Unlike the templated section benchmark, each question targets one fact in one passage and carries a
verbatim reference excerpt. The question writer is a local LLM (llama-server); every item is checked
mechanically: the reference must appear verbatim in the passage and the question must name the medicine.

Usage: python scripts/build_freetext_benchmark.py [--count 120] [--seed 42]
Requires llama-server on CIMA_RAG_LLAMA_BASE_URL and the index at CIMA_RAG_INDEX_DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":  # broken certificate in the Windows store breaks ssl on import
    _orig = ssl.SSLContext._load_windows_store_certs  # type: ignore[attr-defined]

    def _tolerant(self, storename, purpose):  # noqa: ANN001
        try:
            return _orig(self, storename, purpose)
        except ssl.SSLError:
            return None

    ssl.SSLContext._load_windows_store_certs = _tolerant  # type: ignore[attr-defined]

import httpx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [
    (1, "4.1"), (1, "4.2"), (1, "4.3"), (1, "4.4"), (1, "4.5"), (1, "4.6"), (1, "4.8"), (1, "4.9"),
    (1, "5.1"), (1, "5.2"), (1, "6.4"), (2, "1"), (2, "2"), (2, "3"), (2, "4"), (2, "5"),
]
FORBIDDEN = re.compile(r"\b(texto|pasaje|fragmento|extracto|passage|excerpt|text above|el documento|the document)\b", re.I)

PROMPT = (
    "You write evaluation questions for a retrieval system over official Spanish medicine documents.\n"
    "Medicine: {name}\nSection: {title}\n\nPassage:\n{passage}\n\n"
    "Write ONE specific question that a pharmacist or patient could ask about this medicine and that is "
    "answered by a single fact in the passage. Rules: write the question in {language}; include the exact "
    "medicine name '{name}'; never mention 'passage', 'text' or 'document'; the question must make sense "
    "without seeing the passage; do not ask for personal dosing advice. Also copy VERBATIM from the passage "
    "the shortest excerpt (at most 200 characters) that answers the question.\n"
    'Return only JSON: {{"question": "...", "reference": "..."}}'
)


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return re.sub(r"\s+", " ", "".join(c for c in decomposed if not unicodedata.combining(c))).strip()


def digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def sample_passages(index_dir: Path, seed: int, pool: int):
    import lancedb

    table = lancedb.connect(index_dir).open_table("chunks")
    entities = json.loads((index_dir / "entities.json").read_text(encoding="utf-8"))
    regs = sorted({e["registration_number"] for e in entities}, key=lambda r: digest(str(seed), "ft", r))[:pool]
    in_list = ",".join(f"'{r}'" for r in regs)
    rows = (
        table.search()
        .where(f"registration_number IN ({in_list}) AND document_type IN (1,2)")
        .select(["chunk_id", "registration_number", "medicine_name", "document_type", "section", "title", "text"])
        .limit(200_000)
        .to_list()
    )
    by_section: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (int(row["document_type"]), str(row["section"]))
        if key in SECTIONS and 500 <= len(row["text"]) <= 1800:
            by_section[key].append(row)
    for items in by_section.values():
        items.sort(key=lambda r: digest(str(seed), r["chunk_id"]))
    # Round robin over sections so every kind of section is represented; one chunk per medicine.
    used: set[str] = set()
    while any(by_section.values()):
        for key in SECTIONS:
            while by_section[key]:
                row = by_section[key].pop(0)
                if row["registration_number"] not in used:
                    used.add(row["registration_number"])
                    yield row
                    break


def ask(base_url: str, model: str, row: dict, language: str, seed: int) -> dict | None:
    prompt = PROMPT.format(
        name=row["medicine_name"], title=row["title"], passage=row["text"], language=language
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "seed": seed,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=120)
    response.raise_for_status()
    try:
        data = json.loads(response.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return None
    question, reference = str(data.get("question", "")).strip(), str(data.get("reference", "")).strip()
    if not (15 <= len(question) <= 300 and 10 <= len(reference) <= 220):
        return None
    if FORBIDDEN.search(question) or fold(reference) not in fold(row["text"]):
        return None
    name_tokens = [t for t in re.findall(r"[a-z0-9]+", fold(row["medicine_name"])) if len(t) > 2][:2]
    if not all(t in fold(question) for t in name_tokens):
        return None
    return {"question": question, "reference": reference}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-dir", default="indexes/cima-bge")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--out", default="eval/freetext.jsonl")
    args = parser.parse_args()

    items, tried, rejected = [], 0, 0
    for row in sample_passages(ROOT / args.index_dir, args.seed, pool=600):
        if len(items) >= args.count:
            break
        tried += 1
        language = "English" if tried % 6 == 0 else "Spanish"
        result = ask(args.base_url, args.model, row, language, args.seed + tried)
        if result is None:
            rejected += 1
            continue
        items.append(
            {
                "id": f"freetext-{len(items) + 1:03d}",
                "split": "dev" if int(digest("split", row["registration_number"]), 16) % 2 == 0 else "test",
                "category": f"{'ft' if row['document_type'] == 1 else 'leaflet'}_{row['section']}",
                "language": "en" if language == "English" else "es",
                "question": result["question"],
                "reference": result["reference"],
                "gold_registration_number": row["registration_number"],
                "gold_chunk_id": row["chunk_id"],
                "gold_document_type": int(row["document_type"]),
                "gold_section": row["section"],
                "medicine_name": row["medicine_name"],
            }
        )
        print(f"{len(items):3d} {row['medicine_name'][:40]:<40} {result['question'][:90]}", flush=True)
    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    print(f"{len(items)} items from {tried} tries ({rejected} rejected) -> {out}")


if __name__ == "__main__":
    main()
