from src.inference import CIMAMedicalAssistantEngine


def integrated_engine(service) -> CIMAMedicalAssistantEngine:
    engine = CIMAMedicalAssistantEngine.__new__(CIMAMedicalAssistantEngine)
    engine.dataset_path = None
    engine.df_meds = None
    engine.retriever = None
    engine._rag_service = service
    engine._rag_error = None
    return engine


def test_existing_assistant_answer_api_uses_grounded_rag(fixture_components) -> None:
    _, _, _, service = fixture_components
    engine = integrated_engine(service)

    response = engine.answer_with_evidence(
        "¿Qué contraindicaciones tiene Ácido Ejemplo 100 mg?",
        registration_number="10001",
        language="es",
    )

    assert response["status"] == "answered"
    assert response["citations"][0]["registration_number"] == "10001"
    assert response["citations"][0]["quote"] in response["answer"]


def test_existing_medication_search_uses_rag_entities(fixture_components) -> None:
    _, _, _, service = fixture_components
    engine = integrated_engine(service)

    results = engine.search_medications("700001")

    assert results[0]["registration_number"] == "10001"
    assert results[0]["national_code"] == "700001"


def test_existing_assistant_preserves_safe_rag_refusals(fixture_components) -> None:
    _, _, _, service = fixture_components
    engine = integrated_engine(service)

    response = engine.answer_with_evidence("¿Puedo tomarlo para mi dolor?", language="es")

    assert response["status"] == "refused"
    assert response["citations"] == []


def test_existing_symptom_workflow_remains_available(fixture_components) -> None:
    _, _, _, service = fixture_components
    engine = integrated_engine(service)

    response = engine.answer_question("tengo dolor al orinar")

    assert response["symptom_data"]["condition"].startswith("Cistitis")
    assert "Orientación Terapéutica AEMPS" in response["answer"]


def test_symptom_workflow_accepts_rag_only_medication_metadata(fixture_components) -> None:
    _, _, _, service = fixture_components
    engine = integrated_engine(service)
    engine.search_medications = lambda query, limit=5: [
        {
            "registration_number": "rag-only",
            "name": "Medicamento documental",
            "presentation": "20 comprimidos",
        }
    ]

    response = engine.answer_question("dolor muscular por golpe")

    assert response["symptom_data"]["condition"].startswith("Dolor Osteomuscular")
    assert "Consultar ficha oficial" in response["answer"]
