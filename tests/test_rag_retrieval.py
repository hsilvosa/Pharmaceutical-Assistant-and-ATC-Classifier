from src.rag.models import MedicineCandidate
from src.rag.providers import TokenOverlapReranker
from src.rag.retrieval import EntityResolver


def test_resolver_handles_accents_codes_and_ambiguity(fixture_components) -> None:
    _, medicines, _, _ = fixture_components
    resolver = EntityResolver(medicines)
    registration, candidates = resolver.resolve("Contraindicaciones de Acido Ejemplo 100 mg")
    assert registration == "10001"
    assert candidates == []
    registration, _ = resolver.resolve("Información del código 700002")
    assert registration == "10002"
    dose_ambiguous = EntityResolver(
        [
            MedicineCandidate(registration_number="10002", name="Marca Común 10 mg"),
            MedicineCandidate(registration_number="10003", name="Marca Común 20 mg"),
        ]
    )
    registration, candidates = dose_ambiguous.resolve("Composición de Marca Común")
    assert registration is None
    assert {item.registration_number for item in candidates} == {"10002", "10003"}

    ambiguous = EntityResolver(
        [
            MedicineCandidate(registration_number="1", name="Marca Común"),
            MedicineCandidate(registration_number="2", name="Marca Común"),
        ]
    )
    registration, candidates = ambiguous.resolve("Información sobre Marca Comun")
    assert registration is None
    assert {item.registration_number for item in candidates} == {"1", "2"}


def test_hybrid_retrieval_returns_relevant_section(fixture_components) -> None:
    _, _, _, service = fixture_components
    hits = service.retriever.retrieve("alergia principio activo", "10001")
    assert hits[0].chunk.registration_number == "10001"
    assert hits[0].chunk.section == "4.3"


def test_fixture_reranker_prioritizes_requested_section_heading() -> None:
    reranker = TokenOverlapReranker()
    passages = [
        "MEDICAMENTO | seccion 6 | Uso del envase\nSiga las indicaciones del médico.",
        "MEDICAMENTO | seccion 4.1 | Indicaciones terapéuticas\nTratamiento documentado.",
    ]

    scores = reranker.score("¿Qué indicaciones recoge la ficha técnica?", passages)

    assert scores[1] > scores[0]
