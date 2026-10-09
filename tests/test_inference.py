"""
Integration tests for inference engines.
"""
import pytest
from pathlib import Path

from src.config import ATC_MODEL_DIR
from src.inference import ATCClassifierEngine, ATCModelNotTrainedError, CIMAMedicalAssistantEngine

def test_atc_engine_refuses_untrained_model(tmp_path):
    with pytest.raises(ATCModelNotTrainedError):
        ATCClassifierEngine(model_dir=str(tmp_path))


@pytest.mark.skipif(
    not (Path(ATC_MODEL_DIR) / "final_model").exists(),
    reason="ATC classifier is not trained yet",
)
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
