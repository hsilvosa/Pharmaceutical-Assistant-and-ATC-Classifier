from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

from .models import Chunk


class OffsetTokenizer(Protocol):
    def offsets(self, text: str) -> Sequence[tuple[int, int]]: ...


class RegexOffsetTokenizer:
    """Deterministic fixture tokenizer; production uses the embedding model tokenizer."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


class HuggingFaceOffsetTokenizer:
    def __init__(self, model_id: str, revision: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunk_document(
    *,
    dataset_revision: str,
    registration_number: str,
    medicine_name: str,
    document_type: int,
    section: str,
    title: str,
    source_order: int,
    text: str,
    source_url: str,
    photo_url: str,
    tokenizer: OffsetTokenizer,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    if max_tokens <= overlap_tokens:
        raise ValueError("max_tokens must be greater than overlap_tokens")
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    offsets = list(tokenizer.offsets(cleaned))
    if not offsets:
        return []
    chunks: list[Chunk] = []
    step = max_tokens - overlap_tokens
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + max_tokens, len(offsets))
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        chunk_text = cleaned[char_start:char_end]
        content_hash = _digest(chunk_text)
        identity = "|".join(
            [
                dataset_revision,
                registration_number,
                str(document_type),
                section,
                str(char_start),
                str(char_end),
                content_hash,
            ]
        )
        heading = " | ".join(
            value for value in (medicine_name, f"seccion {section}", title) if value
        )
        chunks.append(
            Chunk(
                chunk_id=_digest(identity),
                registration_number=registration_number,
                medicine_name=medicine_name,
                document_type=document_type,
                section=section,
                title=title,
                source_order=source_order,
                char_start=char_start,
                char_end=char_end,
                text=chunk_text,
                embedding_text=f"{heading}\n{chunk_text}",
                source_url=source_url,
                photo_url=photo_url,
                content_hash=content_hash,
            )
        )
        if token_end == len(offsets):
            break
    return chunks


