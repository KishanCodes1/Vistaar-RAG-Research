"""Deterministic tests for the frozen Milestone 5B protocol."""
from __future__ import annotations

import pytest

from eval.shadow_protocol import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CV_FOLDS,
    CV_REPEATS,
    CV_SEED,
    LOGISTIC_SETTINGS,
    MATCHED_COVERAGE_TARGET,
    RISK_TARGET,
    answerability_aurc,
    d_anchor_select,
    frozen_protocol_metadata,
    grouped_cv_assignments,
    paired_aurc_bootstrap,
    select_matched_coverage_threshold,
    select_risk_target_threshold,
    validate_partition_role,
)


def _candidate(chunk_id: str, score: float) -> dict:
    return {"chunk_id": chunk_id, "score": score}


def test_partition_role_is_explicit_and_filename_consistent(tmp_path):
    train = tmp_path / "qasper_train.parquet"
    validation = tmp_path / "qasper_validation.parquet"
    test = tmp_path / "qasper_test.parquet"

    assert validate_partition_role(train, "train") == "train"
    assert validate_partition_role(validation, "validation") == "validation"
    assert validate_partition_role(test, "test") == "test"
    with pytest.raises(ValueError, match="filename indicates partition"):
        validate_partition_role(test, "validation")
    with pytest.raises(ValueError, match="partition role"):
        validate_partition_role(test, "unknown")


def test_grouped_cv_assignments_are_deterministic_and_paper_level():
    papers = {f"paper-{index}": index % 3 == 0 for index in range(20)}

    first = grouped_cv_assignments(papers, repeat=0)
    repeated = grouped_cv_assignments(dict(reversed(list(papers.items()))), repeat=0)
    next_repeat = grouped_cv_assignments(papers, repeat=1)

    assert first == repeated
    assert set(first) == set(papers)
    assert set(first.values()) == set(range(CV_FOLDS))
    assert first != next_repeat


def test_answerability_aurc_uses_full_tie_groups_and_trapezoids():
    area, curve = answerability_aurc(
        [0.9, 0.8, 0.8, 0.1],
        [True, False, True, False],
        coverage_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    assert curve[1] == {
        "target_coverage": 0.25,
        "actual_coverage": 0.25,
        "risk": 0.0,
        "threshold": 0.9,
    }
    assert curve[2]["actual_coverage"] == 0.75
    assert curve[2]["risk"] == pytest.approx(1 / 3)
    assert area == pytest.approx(11 / 48)


def test_threshold_rules_are_deterministic_and_report_fallback():
    matched = select_matched_coverage_threshold(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [True, True, False, True, False, True, False, True, False, True],
    )
    fallback = select_matched_coverage_threshold(
        [0.9, 0.8, 0.2, 0.1], [True, False, True, False]
    )
    risk_target = select_risk_target_threshold(
        [0.9, 0.8, 0.7, 0.1], [True, True, False, False]
    )

    assert matched["coverage"] == pytest.approx(0.3)
    assert matched["fallback_used"] is False
    assert fallback["coverage"] == pytest.approx(0.25)
    assert fallback["fallback_used"] is True
    assert risk_target["risk"] <= RISK_TARGET
    assert risk_target["coverage"] == pytest.approx(0.5)


def test_paired_paper_bootstrap_is_seeded_and_clustered():
    records = [
        {
            "doc_id": "paper-a",
            "answerable": True,
            "baseline_score": 0.8,
            "candidate_score": 0.9,
        },
        {
            "doc_id": "paper-a",
            "answerable": False,
            "baseline_score": 0.7,
            "candidate_score": 0.1,
        },
        {
            "doc_id": "paper-b",
            "answerable": True,
            "baseline_score": 0.6,
            "candidate_score": 0.8,
        },
        {
            "doc_id": "paper-b",
            "answerable": False,
            "baseline_score": 0.5,
            "candidate_score": 0.2,
        },
    ]

    first = paired_aurc_bootstrap(records, resamples=50, seed=123)
    second = paired_aurc_bootstrap(records, resamples=50, seed=123)

    assert first == second
    assert first["papers"] == 2
    assert first["resamples"] == 50
    assert first["difference"] < 0


def test_d_anchor_ties_and_anchor_positions_are_deterministic():
    rrf = [
        _candidate("a", 0.04),
        _candidate("b", 0.03),
        _candidate("c", 0.02),
        _candidate("d", 0.01),
    ]
    reranked = [
        _candidate("d", 0.9),
        _candidate("c", 0.9),
        _candidate("b", 0.1),
        _candidate("a", 0.0),
    ]

    selected = d_anchor_select(rrf, reranked)

    assert [row["chunk_id"] for row in selected] == ["a", "b", "c", "d"]


def test_d_anchor_handles_short_anchor_and_candidate_pools_without_padding():
    one_anchor = d_anchor_select(
        [_candidate("a", 0.1)],
        [_candidate("c", 0.8), _candidate("b", 0.8), _candidate("a", 0.2)],
    )
    two_only = d_anchor_select(
        [_candidate("a", 0.2), _candidate("b", 0.1)],
        [_candidate("b", 0.9), _candidate("a", 0.8)],
    )

    assert [row["chunk_id"] for row in one_anchor] == ["a", "b", "c"]
    assert [row["chunk_id"] for row in two_only] == ["a", "b"]


def test_d_anchor_with_zero_rrf_anchors_uses_deterministic_reranker_order():
    selected = d_anchor_select(
        [],
        [_candidate("c", 0.8), _candidate("b", 0.8), _candidate("a", 0.7)],
    )

    assert [row["chunk_id"] for row in selected] == ["b", "c", "a"]
    assert len({row["chunk_id"] for row in selected}) == len(selected)


def test_d_anchor_short_reranker_pool_falls_back_to_unused_rrf_order():
    selected = d_anchor_select(
        [
            _candidate("a", 0.4),
            _candidate("b", 0.3),
            _candidate("c", 0.2),
            _candidate("d", 0.1),
        ],
        [_candidate("d", 0.9), _candidate("a", 0.8)],
    )

    assert [row["chunk_id"] for row in selected] == ["a", "b", "d", "c"]
    assert len({row["chunk_id"] for row in selected}) == len(selected)


def test_frozen_protocol_metadata_contains_all_predeclared_choices():
    metadata = frozen_protocol_metadata()

    assert metadata["grouped_cross_validation"] == {
        "folds": CV_FOLDS,
        "repeats": CV_REPEATS,
        "seed": CV_SEED,
        "split_algorithm": (
            "stratify papers by presence of any unanswerable question; "
            "SHA-256 order within stratum using seed+repeat; round-robin folds"
        ),
        "selection_metric": "mean_answerability_aurc",
        "selection_rule": "minimum mean AURC; tie selects smaller C",
        "refit": "refit preprocessing and selected C on all official train papers",
    }
    assert metadata["bootstrap"]["resamples"] == BOOTSTRAP_RESAMPLES
    assert metadata["bootstrap"]["seed"] == BOOTSTRAP_SEED
    assert metadata["matched_coverage_threshold"]["target"] == MATCHED_COVERAGE_TARGET
    assert metadata["risk_target_threshold"]["target"] == RISK_TARGET
    assert LOGISTIC_SETTINGS["dependency"] == "scikit-learn==1.9.1"
    assert metadata["reranker_arm"]["calibrator_combination"] == (
        "prohibited_in_primary_comparison"
    )
