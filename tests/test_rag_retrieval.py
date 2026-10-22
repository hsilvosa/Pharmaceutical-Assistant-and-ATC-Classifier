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
    # Same name under two registrations stays ambiguous when the question depends on the registration.
    registration, candidates = ambiguous.resolve("¿Está comercializada Marca Comun?")
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


def test_same_name_registrations_resolve_to_the_canonical_one_unless_the_question_is_registration_specific():
    from src.rag.models import MedicineCandidate
    from src.rag.retrieval import EntityResolver

    def candidate(registration: str, name: str) -> MedicineCandidate:
        return MedicineCandidate(registration_number=registration, name=name)

    resolver = EntityResolver(
        [
            candidate("66514", "INEGY 10 mg/40 mg COMPRIMIDOS"),
            candidate("BE286334IP1", "INEGY 10 MG/40 MG COMPRIMIDOS"),
            candidate("1", "A.A.S. 100 mg COMPRIMIDOS"),
            candidate("2", "A.A.S. 300 mg COMPRIMIDOS"),
        ]
    )
    assert resolver.resolve("¿Qué efectos adversos tiene INEGY 10 mg/40 mg COMPRIMIDOS?") == ("66514", [])
    registration, options = resolver.resolve("¿Está comercializado INEGY 10 mg/40 mg COMPRIMIDOS?")
    assert registration is None and len(options) == 2
    registration, options = resolver.resolve("¿Qué contraindicaciones tiene A.A.S.?")
    assert registration is None and len(options) == 2
