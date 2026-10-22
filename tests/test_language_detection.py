from __future__ import annotations

import pytest

from src.rag.service import detect_language


@pytest.mark.parametrize(
    "question",
    [
        "¿Qué dosis debo tomar de A.A.S. si me duele el pecho?",
        "¿Para qué se utiliza el omeprazol?",
        "Dime las contraindicaciones de Nolotil",
        "efectos adversos del ibuprofeno según el prospecto",
    ],
)
def test_spanish_questions_are_detected(question):
    assert detect_language(question, "auto") == "es"


@pytest.mark.parametrize(
    "question",
    [
        "Which ATC classification is registered for SUNITINIB VIVANTA 25 MG CÁPSULAS DURAS EFG?",
        "What contraindications does A.A.S. have?",
        "How should ILOPROST RAFARM POLVO PARA SOLUCION be stored according to the leaflet?",
    ],
)
def test_english_questions_with_spanish_product_names_stay_english(question):
    assert detect_language(question, "auto") == "en"


def test_explicit_language_wins():
    assert detect_language("¿Qué es esto?", "en") == "en"
