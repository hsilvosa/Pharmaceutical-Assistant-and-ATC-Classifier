from __future__ import annotations

from types import SimpleNamespace as NS

from src.rag.freetext import evaluate_items, reference_recall, retrieval_ranks, summarize, wilson
from src.rag.models import AnswerStatus


def hit(chunk_id: str, registration: str = "1", doc_type: int = 1, section: str = "4.3") -> NS:
    return NS(chunk=NS(chunk_id=chunk_id, registration_number=registration, document_type=doc_type, section=section, text=f"text {chunk_id}"))


ITEM = {
    "id": "f1", "split": "dev", "category": "ft_4.3", "question": "¿Contraindicaciones de X?",
    "reference": "No debe utilizarse en caso de alergia", "gold_chunk_id": "k2",
    "gold_registration_number": "1", "gold_document_type": 1, "gold_section": "4.3",
}


def test_wilson_interval_is_wide_for_small_samples_and_bounded():
    low, high = wilson(9, 10)
    assert 0.5 < low < 0.9 < high <= 1.0
    assert wilson(0, 0) == [0.0, 0.0]
    assert wilson(60, 60)[1] == 1.0


def test_reference_recall_ignores_accents_case_and_short_words():
    assert reference_recall("Se usa en caso de ALERGIA; no debe utilizarse.", ITEM["reference"]) == 1.0
    assert reference_recall("Nada que ver", ITEM["reference"]) == 0.0


def test_retrieval_ranks_distinguish_chunk_from_section():
    hits = [hit("k0", registration="2"), hit("k1"), hit("k2")]
    assert retrieval_ranks(ITEM, hits) == (3, 2)
    assert retrieval_ranks(ITEM, [hit("k9", registration="9")]) == (None, None)


class FakeJudge:
    def correctness(self, question, reference, answer):
        return "correct" if "alergia" in answer else "incorrect"

    def faithful(self, answer, passages):
        return bool(passages)


def test_evaluate_items_scores_rows_and_negative_control():
    items = [dict(ITEM, id=f"f{i}") for i in range(5)]

    def answer(item):
        response = NS(
            status=AnswerStatus.ANSWERED,
            answer="No debe usarse en caso de alergia [C1]",
            citations=[NS(chunk_id="k2")],
        )
        return response, [hit("k2")]

    result = evaluate_items(items, answer, FakeJudge())
    summary = result["summary"]
    assert summary["gold_chunk_in_context"]["rate"] == 1.0
    assert summary["judge_correct"]["k"] == 5 and summary["judge_faithful"]["k"] == 5
    # the fake judge accepts any answer containing "alergia", so the control must flag it (rate 1.0)
    assert summary["negative_control_false_positive"]["rate"] == 1.0


def test_errors_are_counted_not_raised():
    def boom(item):
        raise RuntimeError("server down")

    summary = evaluate_items([ITEM], boom, None)["summary"]
    assert summary["errors"] == 1 and summary["items"] == 1


def test_summarize_handles_refusals_without_judge_fields():
    rows = [{"id": "a", "category": "c", "split": "dev", "status": "refused", "chunk_rank": None, "section_rank": None,
             "cited_gold_chunk": False, "reference_recall": 0.0}]
    summary = summarize(rows, [])
    assert summary["answered"]["rate"] == 0.0 and "judge_correct" not in summary
