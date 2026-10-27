"""Record-only metric reproduction tests."""
from __future__ import annotations

from eval.compare_results import abstention_metrics, recompute


def _record(question_id: str, answerable: bool, prediction: str, confidence: float):
    return {
        "config": "E: full",
        "question_id": question_id,
        "doc_id": f"doc-{question_id}",
        "answerable": answerable,
        "prediction": prediction,
        "raw_prediction": prediction,
        "gold": "first reference" if answerable else "",
        "gold_answers": ["first reference", "correct answer"] if answerable else [],
        "gold_evidence": ["correct answer evidence"] if answerable else [],
        "citations": [],
        "retrieved": ["correct answer evidence"],
        "retrieved_chunks": [
            {
                "chunk_id": f"chunk-{question_id}",
                "text": "correct answer evidence",
            }
        ],
        "abstained": confidence < 0.5,
        "scores": {"confidence": confidence},
        "latency_s": 0.1,
    }


def test_clean_records_reproduce_best_over_all_gold_answers():
    records = [
        _record("a", True, "correct answer", 0.9),
        _record("u", False, "wrong", 0.1),
    ]
    result = recompute(records)
    assert result["gold_reference_mode"] == "all_gold_answers"
    assert result["f1"] == 1.0
    assert result["em"] == 1.0
    assert result["retrieval_recall"] == 0.5
    assert result["duplicate_retrieved_slots"] == 0


def test_abstention_metrics_are_derived_from_records():
    records = [
        _record("a", True, "correct answer", 0.9),
        _record("u", False, "wrong", 0.1),
    ]
    result = abstention_metrics(records)
    assert result["auc"] == 1.0
    assert result["peak_balanced_accuracy"]["balanced_accuracy"] == 1.0
