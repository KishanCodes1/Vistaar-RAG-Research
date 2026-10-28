"""Frozen, outcome-independent methods for the Milestone 5B shadow protocol."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_SCHEMA = "vistaar-shadow-protocol-v1"
PARTITION_ROLES = ("train", "validation", "test", "development-fixture")
CV_FOLDS = 5
CV_REPEATS = 20
CV_SEED = 20261028
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20261029
COVERAGE_GRID = tuple(index / 100 for index in range(101))
MATCHED_COVERAGE_TARGET = 0.28
MATCHED_COVERAGE_TOLERANCE = 0.02
RISK_TARGET = 0.05
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_SETTINGS = {
    "dependency": "scikit-learn==1.9.1",
    "estimator": "sklearn.linear_model.LogisticRegression",
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "max_iter": 2000,
    "tol": 1e-8,
    "random_state": CV_SEED,
}


def frozen_protocol_metadata() -> dict:
    """Return all choices that must be fingerprinted before outcomes are read."""
    return {
        "schema": PROTOCOL_SCHEMA,
        "partitions": {
            "roles": list(PARTITION_ROLES),
            "group_unit": "qasper_paper_id",
            "confirmatory": {
                "train": "official_qasper_train",
                "calibration": "official_qasper_validation",
                "evaluation": "official_qasper_test",
            },
        },
        "grouped_cross_validation": {
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
        },
        "answerability_aurc": {
            "risk": "covered_unanswerable_count / covered_question_count",
            "zero_coverage_risk": 0.0,
            "coverage_grid": list(COVERAGE_GRID),
            "integration": "trapezoid_on_target_coverage_grid",
            "ties": "include_entire_score_tie_group_at_cutoff",
            "score_direction": "higher_is_more_answerable",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "qasper_paper_id",
            "procedure": (
                "sample N papers with replacement; include all questions for "
                "each sampled paper; use identical sample for paired arms"
            ),
            "interval": "percentile_95_numpy_linear",
        },
        "matched_coverage_threshold": {
            "target": MATCHED_COVERAGE_TARGET,
            "tolerance": MATCHED_COVERAGE_TOLERANCE,
            "candidate_thresholds": "unique calibration scores plus all-abstain",
            "objective": "maximum_balanced_accuracy",
            "selection_data": "official_qasper_validation_only",
            "tie_rule": (
                "closest target coverage, then fewer answered-unanswerable, "
                "then higher threshold"
            ),
            "fallback": (
                "if no candidate is in tolerance, choose globally closest "
                "coverage with the same tie rule and report fallback"
            ),
        },
        "risk_target_threshold": {
            "target": RISK_TARGET,
            "justification": "policy ceiling: at most one unanswerable per 20 answers",
            "selection_data": "official_qasper_validation_only",
            "objective": "maximum_coverage_subject_to_empirical_risk_at_most_target",
            "tie_rule": "higher balanced accuracy, then higher threshold",
            "fallback": "all-abstain, with zero coverage and zero empirical risk",
        },
        "preprocessing": {
            "counts": "log1p",
            "continuous": "training-fold median imputation then standardization",
            "binary": "0_or_1",
            "categorical": "fixed one-hot categories with explicit unknown",
            "zero_variance_scale": 1.0,
            "missingness": "indicator plus training-fold median; never zero-fill",
            "score_families": "standardize each stage family separately",
            "fit_scope": "inside each grouped-CV training fold",
        },
        "logistic_regression": {
            "c_grid": list(LOGISTIC_C_GRID),
            **LOGISTIC_SETTINGS,
        },
        "reranker_arm": {
            "name": "D-anchor",
            "anchors": 2,
            "output_size": 4,
            "anchor_source": "first_unique_rrf_candidates",
            "anchor_positions": [1, 2],
            "fill_order": (
                "cross-encoder score descending; tie by lower RRF rank; "
                "then chunk ID"
            ),
            "short_pool": "return every available unique candidate without padding",
            "primary_comparison": "D-anchor_minus_D_Ret-R@4",
            "calibrator_combination": "prohibited_in_primary_comparison",
        },
    }


def validate_partition_role(dataset_path: str | Path, role: str) -> str:
    """Validate an explicit role and reject recognizable filename mismatches."""
    if role not in PARTITION_ROLES:
        raise ValueError(
            f"dataset partition role must be one of {PARTITION_ROLES}, got {role!r}"
        )
    normalized = Path(dataset_path).stem.lower().replace("-", "_")
    tokens = set(normalized.split("_"))
    recognized = next(
        (candidate for candidate in ("train", "validation", "test") if candidate in tokens),
        None,
    )
    if recognized is not None and role != recognized:
        raise ValueError(
            f"dataset filename indicates partition {recognized!r}, not {role!r}"
        )
    return role


def grouped_cv_assignments(
    paper_has_unanswerable: Mapping[str, bool], repeat: int
) -> dict[str, int]:
    """Assign each paper to one deterministic, label-stratified fold."""
    if repeat < 0 or repeat >= CV_REPEATS:
        raise ValueError(f"repeat must be in [0, {CV_REPEATS})")
    if len(paper_has_unanswerable) < CV_FOLDS:
        raise ValueError("paper count is smaller than the frozen fold count")
    assignments: dict[str, int] = {}
    for stratum in (False, True):
        papers = [
            paper
            for paper, has_unanswerable in paper_has_unanswerable.items()
            if bool(has_unanswerable) is stratum
        ]
        papers.sort(
            key=lambda paper: hashlib.sha256(
                f"{CV_SEED + repeat}:{int(stratum)}:{paper}".encode()
            ).hexdigest()
        )
        for index, paper in enumerate(papers):
            assignments[paper] = index % CV_FOLDS
    return assignments


def answerability_aurc(
    scores: Sequence[float],
    answerable: Sequence[bool],
    coverage_grid: Sequence[float] = COVERAGE_GRID,
) -> tuple[float, list[dict]]:
    """Compute tie-grouped answerability risk over a fixed coverage grid."""
    if len(scores) != len(answerable) or not scores:
        raise ValueError("scores and labels must be nonempty and equal length")
    numeric_scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in numeric_scores):
        raise ValueError("scores must be finite")
    grid = [float(value) for value in coverage_grid]
    if not grid or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("coverage grid must start at 0 and end at 1")
    if any(left >= right for left, right in zip(grid, grid[1:])):
        raise ValueError("coverage grid must be strictly increasing")

    order = sorted(range(len(scores)), key=lambda index: numeric_scores[index], reverse=True)
    curve = []
    for target in grid:
        if target == 0.0:
            curve.append(
                {
                    "target_coverage": target,
                    "actual_coverage": 0.0,
                    "risk": 0.0,
                    "threshold": None,
                }
            )
            continue
        rank = max(1, math.ceil(target * len(order)))
        threshold = numeric_scores[order[rank - 1]]
        covered = [index for index, score in enumerate(numeric_scores) if score >= threshold]
        risk = sum(not bool(answerable[index]) for index in covered) / len(covered)
        curve.append(
            {
                "target_coverage": target,
                "actual_coverage": len(covered) / len(order),
                "risk": risk,
                "threshold": threshold,
            }
        )

    area = sum(
        (right["target_coverage"] - left["target_coverage"])
        * (left["risk"] + right["risk"])
        / 2
        for left, right in zip(curve, curve[1:])
    )
    return float(area), curve


def _threshold_rows(scores: Sequence[float], answerable: Sequence[bool]) -> list[dict]:
    if len(scores) != len(answerable) or not scores:
        raise ValueError("scores and labels must be nonempty and equal length")
    labels = [bool(label) for label in answerable]
    if all(labels) or not any(labels):
        raise ValueError("threshold selection requires both answerability classes")
    numeric_scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in numeric_scores):
        raise ValueError("scores must be finite")
    rows = []
    for threshold in [math.inf, *sorted(set(numeric_scores), reverse=True)]:
        decisions = [score >= threshold for score in numeric_scores]
        covered = sum(decisions)
        answered_answerable = sum(
            decision and label for decision, label in zip(decisions, labels)
        )
        answered_unanswerable = sum(
            decision and not label for decision, label in zip(decisions, labels)
        )
        answerable_count = sum(labels)
        unanswerable_count = len(labels) - answerable_count
        tpr = answered_answerable / answerable_count
        tnr = (unanswerable_count - answered_unanswerable) / unanswerable_count
        rows.append(
            {
                "threshold": threshold,
                "coverage": covered / len(labels),
                "balanced_accuracy": (tpr + tnr) / 2,
                "answered_unanswerable": answered_unanswerable,
                "risk": answered_unanswerable / covered if covered else 0.0,
            }
        )
    return rows


def select_matched_coverage_threshold(
    scores: Sequence[float], answerable: Sequence[bool]
) -> dict:
    """Select the frozen calibration-only matched-coverage operating point."""
    rows = _threshold_rows(scores, answerable)
    eligible = [
        row
        for row in rows
        if abs(row["coverage"] - MATCHED_COVERAGE_TARGET)
        <= MATCHED_COVERAGE_TOLERANCE
    ]
    fallback = not eligible
    pool = eligible or rows
    if fallback:
        closest = min(
            abs(row["coverage"] - MATCHED_COVERAGE_TARGET) for row in pool
        )
        pool = [
            row
            for row in pool
            if abs(row["coverage"] - MATCHED_COVERAGE_TARGET) == closest
        ]
    selected = max(
        pool,
        key=lambda row: (
            row["balanced_accuracy"],
            -abs(row["coverage"] - MATCHED_COVERAGE_TARGET),
            -row["answered_unanswerable"],
            row["threshold"],
        ),
    )
    return {**selected, "fallback_used": fallback}


def select_risk_target_threshold(
    scores: Sequence[float], answerable: Sequence[bool]
) -> dict:
    """Select maximum calibration coverage under the frozen empirical risk cap."""
    eligible = [row for row in _threshold_rows(scores, answerable) if row["risk"] <= RISK_TARGET]
    selected = max(
        eligible,
        key=lambda row: (
            row["coverage"],
            row["balanced_accuracy"],
            row["threshold"],
        ),
    )
    return selected


def paired_aurc_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired paper-cluster percentile interval for candidate-minus-baseline AURC."""
    by_paper: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_paper[str(record["doc_id"])].append(record)
    papers = sorted(by_paper)
    if not papers or resamples < 1:
        raise ValueError("bootstrap requires papers and at least one resample")

    def difference(rows: Iterable[Mapping[str, Any]]) -> float:
        values = list(rows)
        labels = [bool(row["answerable"]) for row in values]
        baseline, _ = answerability_aurc(
            [float(row["baseline_score"]) for row in values], labels
        )
        candidate, _ = answerability_aurc(
            [float(row["candidate_score"]) for row in values], labels
        )
        return candidate - baseline

    point = difference(record for paper in papers for record in by_paper[paper])
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        sampled = [rng.choice(papers) for _ in papers]
        differences.append(
            difference(record for paper in sampled for record in by_paper[paper])
        )
    lower, upper = np.percentile(differences, [2.5, 97.5], method="linear")
    return {
        "difference": point,
        "lower": float(lower),
        "upper": float(upper),
        "resamples": resamples,
        "seed": seed,
        "papers": len(papers),
    }


def _chunk_id(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return str(candidate["chunk_id"])
    return str(candidate.chunk.chunk_id)


def _score(candidate: Any) -> float:
    if isinstance(candidate, Mapping):
        return float(candidate["score"])
    return float(candidate.score)


def d_anchor_select(
    rrf_candidates: Sequence[Any],
    reranked_candidates: Sequence[Any],
    *,
    anchor_count: int = 2,
    output_size: int = 4,
) -> list[Any]:
    """Select RRF anchors followed by deterministically tied reranker candidates."""
    if anchor_count < 0 or output_size < 1 or anchor_count > output_size:
        raise ValueError("invalid anchor/output size")
    unique_rrf = []
    seen = set()
    for candidate in rrf_candidates:
        chunk_id = _chunk_id(candidate)
        if chunk_id not in seen:
            unique_rrf.append(candidate)
            seen.add(chunk_id)
    anchors = unique_rrf[:anchor_count]
    rrf_ranks = {
        _chunk_id(candidate): rank for rank, candidate in enumerate(unique_rrf, start=1)
    }
    ordered_reranked = sorted(
        reranked_candidates,
        key=lambda candidate: (
            -_score(candidate),
            rrf_ranks.get(_chunk_id(candidate), math.inf),
            _chunk_id(candidate),
        ),
    )
    selected = list(anchors)
    selected_ids = {_chunk_id(candidate) for candidate in selected}
    for candidate in ordered_reranked:
        chunk_id = _chunk_id(candidate)
        if chunk_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(chunk_id)
        if len(selected) == output_size:
            return selected
    for candidate in unique_rrf[anchor_count:]:
        chunk_id = _chunk_id(candidate)
        if chunk_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(chunk_id)
        if len(selected) == output_size:
            break
    return selected
