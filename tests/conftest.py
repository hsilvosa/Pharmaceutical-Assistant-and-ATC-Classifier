from __future__ import annotations

import pytest

from src.rag.chunking import RegexOffsetTokenizer, chunk_document
from src.rag.models import MedicineCandidate
from src.rag.providers import ExtractiveFixtureGenerator, HashEmbedder, TokenOverlapReranker
from src.rag.retrieval import EntityResolver, HybridRetriever
from src.rag.service import RagService
from src.rag.store import InMemoryStore


@pytest.fixture
def fixture_components():
    tokenizer = RegexOffsetTokenizer()
    chunks = []
    chunks.extend(
        chunk_document(
            dataset_revision="fixture-revision",
            registration_number="10001",
            medicine_name="Ácido Ejemplo 100 mg",
            document_type=1,
            section="4.3",
            title="Contraindicaciones",
            source_order=4,
            text="No debe utilizarse en caso de alergia al principio activo.",
            source_url="https://example.test/10001",
            photo_url="https://example.test/10001.jpg",
            tokenizer=tokenizer,
        )
    )
    chunks.extend(
        chunk_document(
            dataset_revision="fixture-revision",
            registration_number="10002",
            medicine_name="Ácido Ejemplo Forte",
            document_type=1,
            section="4.1",
            title="Indicaciones",
            source_order=3,
            text="Está indicado para el alivio temporal del dolor leve.",
            source_url="https://example.test/10002",
            photo_url="",
            tokenizer=tokenizer,
        )
    )
    medicines = [
        MedicineCandidate(
            registration_number="10001",
            name="Ácido Ejemplo 100 mg",
            presentation="Ácido Ejemplo, 20 comprimidos",
            national_code="700001",
        ),
        MedicineCandidate(
            registration_number="10002",
            name="Ácido Ejemplo Forte",
            presentation="Ácido Ejemplo, solución",
            national_code="700002",
        ),
    ]
    embedder = HashEmbedder()
    vectors = embedder.encode([chunk.embedding_text for chunk in chunks])
    store = InMemoryStore(chunks, vectors, medicines)
    retriever = HybridRetriever(
        store,
        embedder,
        TokenOverlapReranker(),
        dense_candidates=10,
        lexical_candidates=10,
        rerank_candidates=10,
        final_passages=4,
    )
    service = RagService(
        retriever,
        EntityResolver(medicines),
        ExtractiveFixtureGenerator(),
        "fixture-revision",
    )
    return chunks, medicines, store, service


