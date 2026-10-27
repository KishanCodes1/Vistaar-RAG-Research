"""Tests for document-disjoint threshold selection."""
from __future__ import annotations

from eval.threshold_protocol import (
    document_holdout_split,
    evaluate_threshold,
    grouped_cross_validation,
    select_threshold,
)


def _record(doc: str, question: str, answerable: bool, confidence: float, f1: bool = True):
    return {
        "doc_id": doc,
        "question_id": question,
        "answerable": answerable,
        "prediction": "correct" if f1 else "wrong",
        "raw_prediction": "correct" if f1 else "wrong",
        "gold": "correct" if answerable else "",
        "scores": {"confidence": confidence},
    }


def _records():
    rows = []
    for index in range(10):
        doc = f"doc-{index}"
        rows.append(_record(doc, f"{doc}-a", True, 0.8 if index < 5 else 0.3))
        if index % 2 == 0:
            rows.append(_record(doc, f"{doc}-u", False, 0.2))
    return rows


def test_holdout_is_deterministic_and_document_disjoint():
    records = _records()
    first_cal, first_test, first_split = document_holdout_split(records, 0.6, 17)
    second_cal, second_test, second_split = document_holdout_split(records, 0.6, 17)

    assert first_split == second_split
    assert {row["doc_id"] for row in first_cal}.isdisjoint(
        {row["doc_id"] for row in first_test}
    )
    assert {row["question_id"] for row in first_cal + first_test} == {
        row["question_id"] for row in records
    }
    assert first_cal == second_cal
    assert first_test == second_test


def test_threshold_is_selected_from_calibration_only():
    calibration = [
        _record("c1", "ca", True, 0.8),
        _record("c2", "cu", False, 0.2),
    ]
    # Test labels favour another threshold, but cannot affect selection.
    test = [
        _record("t1", "ta", True, 0.1),
        _record("t2", "tu", False, 0.9),
    ]
    threshold, _ = select_threshold(calibration, "balanced_accuracy")
    locked = evaluate_threshold(test, threshold)

    assert threshold == 0.25
    assert locked["balanced_accuracy"] == 0.0


def test_covered_f1_penalises_answered_unanswerable_questions():
    records = [
        _record("d1", "a", True, 0.9),
        _record("d2", "u", False, 0.9),
    ]
    metrics = evaluate_threshold(records, 0.5)
    assert metrics["conditional_answerable_f1"] == 1.0
    assert metrics["covered_f1"] == 0.5
    assert metrics["answered_unanswerable"] == 1


def test_grouped_cross_validation_covers_every_question():
    records = _records()
    result = grouped_cross_validation(
        records,
        objective="balanced_accuracy",
        folds=5,
        repeats=3,
        seed=23,
    )
    assert len(result["runs"]) == 3
    assert all(run["metrics"]["n"] == len(records) for run in result["runs"])
