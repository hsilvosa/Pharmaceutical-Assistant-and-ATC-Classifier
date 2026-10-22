from __future__ import annotations

import json
from pathlib import Path

SPACE = Path("space")


def test_gallery_entries_are_complete_and_real_runs():
    gallery = json.loads((SPACE / "gallery.json").read_text(encoding="utf-8"))
    entries = gallery["entries"]
    assert len(entries) >= 8 and len({e["id"] for e in entries}) == len(entries)
    for entry in entries:
        assert entry["question"] and entry["answer"]
        assert entry["status"] in {"answered", "refused", "insufficient_evidence", "needs_disambiguation"}
        assert entry["total_ms"] >= 0
        if entry["status"] == "answered":
            assert entry["citations"], entry["id"]
            assert all(c["url"].startswith("https://") for c in entry["citations"])
        else:
            assert entry["citations"] == [], entry["id"]
    statuses = {e["status"] for e in entries}
    assert {"answered", "refused", "needs_disambiguation", "insufficient_evidence"} <= statuses


def test_calibration_and_ablation_files_support_the_claims_in_the_ui():
    calibration = json.loads((SPACE / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["temperature"] > 1.0
    assert calibration["test_after"]["ece"] < calibration["test_before"]["ece"]
    assert calibration["test_after"]["accuracy"] == calibration["test_before"]["accuracy"]
    ablation = json.loads((SPACE / "data" / "atc_input_ablation.json").read_text(encoding="utf-8"))
    assert ablation["name only"]["top1"] < ablation["name + ingredient"]["top1"] < ablation["full training format (all fields)"]["top1"]


def test_space_readme_has_hub_front_matter_and_requirements_pin_gradio():
    readme = (SPACE / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---") and "sdk: gradio" in readme and "app_file: app.py" in readme
    assert "gradio==" in (SPACE / "requirements.txt").read_text(encoding="utf-8")


def test_static_space_ships_the_same_data_as_the_gradio_app_and_the_quantised_model_is_checked():
    static = Path("static-space")
    assert (static / "index.html").exists() and (static / "app.js").exists()
    gallery = json.loads((static / "data" / "gallery.json").read_text(encoding="utf-8"))
    assert gallery == json.loads((SPACE / "gallery.json").read_text(encoding="utf-8"))
    readme = (static / "README.md").read_text(encoding="utf-8")
    assert "sdk: static" in readme and "app_file: index.html" in readme
    parity = json.loads(Path("eval/atc_onnx_parity.json").read_text(encoding="utf-8"))
    assert parity["int8"]["agreement_with_torch"] >= 0.97
    assert abs(parity["int8"]["ece_with_temperature"] - parity["torch"]["ece_with_temperature"]) < 0.02
