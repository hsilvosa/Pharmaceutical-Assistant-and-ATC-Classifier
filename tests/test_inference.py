"""
Integration tests for inference engines.
"""
import pytest
from src.inference import ATCClassifierEngine, CIMAMedicalAssistantEngine

def test_atc_classifier_engine():
    engine = ATCClassifierEngine()
    res = engine.predict("Omeprazol 20mg capsulas orales para reflujo", top_k=3)
    assert "top_prediction" in res
    assert "all_predictions" in res
    assert len(res["all_predictions"]) == 3

def test_cima_medical_assistant_engine():
    engine = CIMAMedicalAssistantEngine()
    res = engine.answer_question("Omeprazol")
    assert "answer" in res
