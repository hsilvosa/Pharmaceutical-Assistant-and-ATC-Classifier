from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

from .models import MedicineCandidate, SearchHit
from .providers import Embedder, Reranker
from .store import SearchStore


def normalize_term(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


class EntityResolver:
    def __init__(self, candidates: Sequence[MedicineCandidate]) -> None:
        self.candidates = list(candidates)
        self.by_registration: dict[str, list[MedicineCandidate]] = defaultdict(list)
        self.aliases: dict[str, set[str]] = defaultdict(set)
        for candidate in self.candidates:
            self.by_registration[candidate.registration_number].append(candidate)
            values = (
                candidate.registration_number,
                candidate.national_code,
                candidate.name,
                candidate.presentation,
            )
            for value in values:
                alias = normalize_term(value)
                if len(alias) >= 4:
                    self.aliases[alias].add(candidate.registration_number)
            name_alias = normalize_term(candidate.name)
            brand_alias = re.sub(
                r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|ml|por ciento)\b.*$",
                "",
                name_alias,
            ).strip()
            if len(brand_alias) >= 4 and brand_alias != name_alias:
                self.aliases[brand_alias].add(candidate.registration_number)

    def resolve(
        self, question: str, explicit_registration: str | None = None
    ) -> tuple[str | None, list[MedicineCandidate]]:
        if explicit_registration:
            matches = self.by_registration.get(explicit_registration, [])
            return (explicit_registration, []) if matches else (None, [])
        normalized = f" {normalize_term(question)} "
        registrations: set[str] = set()
        for alias in sorted(self.aliases, key=len, reverse=True):
            if f" {alias} " in normalized:
                registrations.update(self.aliases[alias])
                if registrations:
                    break
        if len(registrations) == 1:
            return next(iter(registrations)), []
        if len(registrations) > 1:
            candidates = []
            for registration in sorted(registrations):
                candidates.append(self.by_registration[registration][0])
            return None, candidates[:12]
        return None, []

    def has_registration(self, registration_number: str) -> bool:
        return registration_number in self.by_registration

    def search(self, query: str, limit: int = 12) -> list[MedicineCandidate]:
        normalized = normalize_term(query)
        if len(normalized) < 2:
            return []
        matches = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in self.candidates:
            haystack = normalize_term(
                f"{candidate.registration_number} {candidate.national_code} "
                f"{candidate.name} {candidate.presentation}"
            )
            key = (
                candidate.registration_number,
                candidate.presentation,
                candidate.national_code,
            )
            if normalized in haystack and key not in seen:
                seen.add(key)
                matches.append(candidate)
        return matches[:limit]


class HybridRetriever:
    def __init__(
        self,
        store: SearchStore,
        embedder: Embedder,
        reranker: Reranker,
        *,
        dense_candidates: int = 50,
        lexical_candidates: int = 50,
        rerank_candidates: int = 30,
        final_passages: int = 8,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.dense_candidates = dense_candidates
        self.lexical_candidates = lexical_candidates
        self.rerank_candidates = rerank_candidates
        self.final_passages = final_passages

    def retrieve(self, query: str, registration_number: str | None = None) -> list[SearchHit]:
        vector = self.embedder.encode([query])[0]
        dense = self.store.dense_search(vector, self.dense_candidates, registration_number)
        lexical = self.store.lexical_search(query, self.lexical_candidates, registration_number)
        fused: dict[str, SearchHit] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        for hits, field in ((dense, "dense_rank"), (lexical, "lexical_rank")):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.chunk.chunk_id] += 1.0 / (60 + rank)
                existing = fused.get(hit.chunk.chunk_id)
                if existing is None:
                    fused[hit.chunk.chunk_id] = hit.model_copy(deep=True)
                else:
                    setattr(existing, field, rank)
        candidates = sorted(
            fused.values(), key=lambda hit: (-scores[hit.chunk.chunk_id], hit.chunk.chunk_id)
        )[: self.rerank_candidates]
        rerank_scores = self.reranker.score(query, [hit.chunk.embedding_text for hit in candidates])
        for hit, score in zip(candidates, rerank_scores, strict=True):
            hit.rerank_score = score
            hit.score = score
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return candidates[: self.final_passages]
