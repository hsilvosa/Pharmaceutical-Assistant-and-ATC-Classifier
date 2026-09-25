from pathlib import Path

from src.rag.evaluation import load_benchmark


def test_committed_benchmarks_have_expected_size_and_no_registration_leakage() -> None:
    generated = load_benchmark([Path("eval/generated.jsonl")])
    curated = load_benchmark([Path("eval/curated.jsonl")])
    assert len(generated) == 250
    assert len(curated) == 50
    dev = {item.registration_number for item in generated if item.split == "dev"}
    test = {item.registration_number for item in generated if item.split == "test"}
    assert not dev & test

