"""Tests for document-cluster full-run analysis helpers."""
from __future__ import annotations

from eval.full_qasper_analysis import (
    compare_overlap,
    document_cluster_intervals,
    paired_document_delta,
)


def _record(config: str, doc: str, question: str, confidence: float, answer: str):
    return {
        "config": config,
        "doc_id": doc,
        "question_id": question,
        "question": question,
        "answerable": True,
        "prediction": answer,
        "raw_prediction": answer,
        "gold": "correct",
        "gold_answers": ["correct"],
        "gold_evidence": ["correct evidence"],
        "citations": [],
        "retrieved": ["correct evidence"],
        "retrieved_chunks": [
            {"chunk_id": f"{doc}:0", "score": 1.0, "text": "correct evidence", "source": "dense"}
        ],
        "abstained": confidence < 0.5,
        "scores": {"confidence": confidence},
        "latency_s": 0.1,
        "_cache": {"run_fingerprint": "run"},
    }


def test_cluster_intervals_are_deterministic_and_bracket_point():
    records = [
        _record("A", "d1", "q1", 0.9, "correct"),
        _record("A", "d2", "q2", 0.1, "wrong"),
    ]
    first = document_cluster_intervals(records, resamples=100, seed=7)
    second = document_cluster_intervals(records, resamples=100, seed=7)
    assert first == second
    assert first["f1"]["lower"] <= first["f1"]["point"] <= first["f1"]["upper"]


def test_paired_document_delta_detects_improvement():
    left = [_record("A", "d1", "q1", 0.9, "wrong")]
    right = [_record("C", "d1", "q1", 0.9, "correct")]
    result = paired_document_delta(left, right, "f1", resamples=20, seed=3)
    assert result["point_delta"] == 1.0
    assert result["lower"] == 1.0
    assert result["upper"] == 1.0


def test_overlap_ignores_only_runtime_and_cache_metadata():
    old = _record("A", "d1", "q1", 0.9, "correct")
    new = dict(old)
    new["latency_s"] = 99.0
    new["_cache"] = {"run_fingerprint": "other"}
    result = compare_overlap([new], [old])
    assert result == {
        "overlapping_records": 1,
        "scientifically_identical": 1,
        "different": 0,
    }
