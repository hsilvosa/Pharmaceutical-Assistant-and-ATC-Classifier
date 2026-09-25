from __future__ import annotations

import pytest

from src.rag.models import AnswerStatus, QueryRequest
from src.rag.providers import Generator
from src.rag.retrieval import EntityResolver, HybridRetriever
from src.rag.service import RagService
from src.rag.store import InMemoryStore


def test_answer_contains_verified_citation(fixture_components) -> None:
    _, _, _, service = fixture_components
    response = service.query(
        QueryRequest(
            question="¿Qué contraindicaciones tiene Ácido Ejemplo 100 mg?", language="es"
        )
    )
    assert response.status == AnswerStatus.ANSWERED
    assert response.citations[0].citation_id == "C1"
    assert response.citations[0].quote in fixture_components[0][0].text


def test_personalized_request_is_refused_before_retrieval(fixture_components) -> None:
    _, _, _, service = fixture_components
    response = service.query(QueryRequest(question="¿Qué dosis debo tomar hoy?", language="es"))
    assert response.status == AnswerStatus.REFUSED
    assert response.citations == []


@pytest.mark.parametrize(
    "question",
    [
        "¿Puedo tomar Ácido Ejemplo para mi dolor?",
        "¿Qué me recomiendas para la fiebre?",
        "Can I take Example Acid for my symptoms?",
    ],
)
def test_treatment_selection_requests_are_refused(fixture_components, question: str) -> None:
    _, _, _, service = fixture_components
    response = service.query(QueryRequest(question=question, language="auto"))
    assert response.status == AnswerStatus.REFUSED
    assert response.citations == []


def test_unknown_explicit_registration_does_not_search_other_medicines(
    fixture_components,
) -> None:
    _, _, _, service = fixture_components
    response = service.query(
        QueryRequest(
            question="¿Qué contraindicaciones constan?",
            registration_number="99999",
            language="es",
        )
    )
    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []


def test_empty_index_returns_insufficient_evidence(fixture_components) -> None:
    _, medicines, _, service = fixture_components
    empty_store = InMemoryStore([], [], medicines)
    empty_service = RagService(
        HybridRetriever(empty_store, service.retriever.embedder, service.retriever.reranker),
        EntityResolver(medicines),
        service.generator,
        "fixture-revision",
    )
    response = empty_service.query(QueryRequest(question="What is documented?", language="en"))
    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE


class BrokenGenerator(Generator):
    def generate(self, question, language, hits):
        return "not json"


def test_invalid_generator_output_is_retried_and_rejected(fixture_components) -> None:
    _, medicines, _, service = fixture_components
    broken = RagService(
        service.retriever,
        EntityResolver(medicines),
        BrokenGenerator(),
        "fixture-revision",
    )
    with pytest.raises(ValueError, match="invalid structured output twice"):
        broken.query(QueryRequest(question="What are the contraindications?", language="en"))

