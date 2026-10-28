"""Synthetic-only, leakage-safe mechanics for the future confidence calibrator.

This module deliberately has no dataset loader, evaluation CLI, cache default, or
held-out-test scoring entry point.  Public fitting functions accept only validated
``origin="synthetic"`` observations in the explicit ``train`` role.  The code
exercises preprocessing, grouped-CV, refit, threshold, and artifact mechanics; it
does not authorize or support fitting on project evaluation records.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .provenance import hash_file, stable_fingerprint
from .shadow_protocol import (
    CV_FOLDS,
    CV_REPEATS,
    CV_SEED,
    LOGISTIC_C_GRID,
    LOGISTIC_SETTINGS,
    PROTOCOL_SCHEMA,
    answerability_aurc,
    grouped_cv_assignments,
    select_matched_coverage_threshold,
    select_risk_target_threshold,
)


OBSERVATION_SCHEMA = "vistaar-synthetic-calibrator-observation-v1"
FEATURE_SCHEMA = "vistaar-calibrator-features-v1"
LABEL_SCHEMA = "vistaar-answerability-label-v1"
PARTITION_SCHEMA = "vistaar-synthetic-partition-v1"
PREPROCESSOR_SCHEMA = "vistaar-calibrator-preprocessor-v1"
ESTIMATOR_SCHEMA = "vistaar-calibrator-estimator-v1"
CV_SCHEMA = "vistaar-synthetic-grouped-cv-v1"
ARTIFACT_SCHEMA = "vistaar-synthetic-calibrator-artifact-v2"
THRESHOLD_SCHEMA = "vistaar-synthetic-threshold-selection-v1"
IMPLEMENTATION_SCHEMA = "vistaar-calibrator-mechanics-implementation-v2"
FINGERPRINT_LIMIT = (
    "integrity and compatibility only; not proof of data authenticity, "
    "split correctness, or scientific validity"
)

SYNTHETIC_ORIGIN = "synthetic"
SYNTHETIC_PARTITION_ROLES = ("train", "validation")

__all__ = (
    "fit_synthetic_calibrator",
    "load_synthetic_artifact",
    "predict_synthetic_scores",
    "run_grouped_cv",
    "save_synthetic_artifact",
    "select_synthetic_threshold",
    "validate_artifact",
    "validate_observations",
    "validate_partition_pair",
)

QUESTION_FORM_CATEGORIES = (
    "what",
    "how",
    "which",
    "why",
    "when",
    "where",
    "who",
    "yes_no_auxiliary",
    "other",
    "unknown",
)
AVAILABILITY_CATEGORIES = (
    "available",
    "not_requested",
    "empty_namespace",
    "model_unavailable",
    "externally_supplied_hits",
    "no_retrieved_passages",
    "unavailable",
    "unknown",
)

COUNT_FEATURES = (
    "question_length_tokens",
    "answer_length_tokens",
    "reader_window_count",
    "reader_candidate_start_count",
    "reader_candidate_span_count",
    "answer_passage_rank",
    "dense_candidate_count",
    "bm25_candidate_count",
    "rrf_candidate_count",
    "reranker_candidate_count",
    "selected_context_candidate_count",
    "dense_bm25_overlap_count",
)

_SUMMARY_FIELDS = (
    "top_score",
    "top1_top2_margin",
    "mean",
    "std",
    "min",
    "max",
    "range",
)
_STAGE_FAMILIES = ("dense", "bm25", "rrf", "reranker", "selected_context")

CONTINUOUS_FEATURES = (
    "reader_confidence",
    "reader_span_null_margin",
    "reader_best_second_span_gap",
    "reader_selected_null_score",
    "reader_null_score_min",
    "reader_null_score_max",
    "reader_null_score_range",
    *tuple(
        f"{family}_{summary}"
        for family in _STAGE_FAMILIES
        for summary in _SUMMARY_FIELDS
    ),
)

BINARY_FEATURES = ("reader_found",)

CATEGORICAL_FEATURES = (
    "question_form",
    "dense_status",
    "bm25_status",
    "rrf_status",
    "reranker_status",
    "selected_context_status",
    "reader_status",
)

RAW_FEATURE_ORDER = (
    *COUNT_FEATURES,
    *CONTINUOUS_FEATURES,
    *BINARY_FEATURES,
    *CATEGORICAL_FEATURES,
)
ALLOWED_FEATURES = frozenset(RAW_FEATURE_ORDER)

_STAGE_STATUS_FIELDS = {
    family: f"{family}_status" for family in _STAGE_FAMILIES
}
_STAGE_COUNT_FIELDS = {
    family: f"{family}_candidate_count" for family in _STAGE_FAMILIES
}
_UNAVAILABLE_STATUSES = {
    "not_requested",
    "model_unavailable",
    "no_retrieved_passages",
    "unavailable",
    "unknown",
}
_READER_DIAGNOSTIC_FIELDS = {
    "reader_window_count",
    "reader_candidate_start_count",
    "reader_candidate_span_count",
    "answer_passage_rank",
    "reader_span_null_margin",
    "reader_best_second_span_gap",
    "reader_selected_null_score",
    "reader_null_score_min",
    "reader_null_score_max",
    "reader_null_score_range",
}

PROHIBITED_FEATURES = frozenset(
    {
        "answerable",
        "label",
        "paper_id",
        "doc_id",
        "question_id",
        "partition",
        "partition_role",
        "question",
        "answer",
        "raw_answer",
        "prediction",
        "gold",
        "gold_answers",
        "gold_evidence",
        "citations",
        "citation_correctness",
        "retrieval_recall",
        "f1",
        "exact_match",
    }
)

_OBSERVATION_KEYS = {
    "schema",
    "origin",
    "paper_id",
    "question_id",
    "partition",
    "features",
    "label",
}
_PARTITION_KEYS = {"schema", "role"}
_FEATURE_KEYS = {"schema", "values"}
_LABEL_KEYS = {"schema", "answerable"}


@dataclass(frozen=True)
class ValidatedBatch:
    """Strict, deterministically ordered synthetic observations."""

    X: np.ndarray
    y: np.ndarray
    paper_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    role: str
    content_digest: str


@dataclass(frozen=True)
class CVSelection:
    """Deterministic grouped-CV selection metadata, with no fitted fold models."""

    selected_c: float
    mean_aurc_by_c: tuple[tuple[float, float], ...]
    fold_aurc_by_c: tuple[tuple[float, tuple[float, ...]], ...]
    assignments: tuple[dict[str, Any], ...]
    assignments_fingerprint: str


@dataclass(frozen=True)
class SyntheticCalibratorArtifact:
    """In-memory synthetic artifact plus its validated JSON-safe metadata."""

    pipeline: Pipeline
    manifest: dict[str, Any]
    fitted_state: dict[str, Any]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, unknown={unknown}")


def _require_nonempty_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{name} must be Unicode NFC normalized")
    return value


def _validate_numeric(value: Any, name: str, *, count: bool) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"feature {name!r} must not be boolean")
    if count:
        if not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"count feature {name!r} must be a nonnegative integer or null")
        return int(value)
    if not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"numeric feature {name!r} must be finite or null")
    return float(value)


def _validate_feature_values(values: Mapping[str, Any]) -> tuple[Any, ...]:
    actual = set(values)
    if actual != ALLOWED_FEATURES:
        prohibited = sorted(actual & PROHIBITED_FEATURES)
        missing = sorted(ALLOWED_FEATURES - actual)
        unknown = sorted(actual - ALLOWED_FEATURES)
        raise ValueError(
            "feature allowlist mismatch; "
            f"prohibited={prohibited}, missing={missing}, unknown={unknown}"
        )

    normalized: dict[str, Any] = {}
    for name in COUNT_FEATURES:
        normalized[name] = _validate_numeric(values[name], name, count=True)
    for name in CONTINUOUS_FEATURES:
        normalized[name] = _validate_numeric(values[name], name, count=False)
    for name in BINARY_FEATURES:
        value = values[name]
        if not isinstance(value, bool):
            raise ValueError(f"binary feature {name!r} must be a bool")
        normalized[name] = value

    categories = {
        "question_form": QUESTION_FORM_CATEGORIES,
        **{
            name: AVAILABILITY_CATEGORIES
            for name in CATEGORICAL_FEATURES
            if name != "question_form"
        },
    }
    for name, allowed in categories.items():
        value = values[name]
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                f"categorical feature {name!r} must be one of {allowed}; got {value!r}"
            )
        normalized[name] = value
    _validate_cross_feature_invariants(normalized)
    return tuple(normalized[name] for name in RAW_FEATURE_ORDER)


def _validate_cross_feature_invariants(values: Mapping[str, Any]) -> None:
    """Reject contradictions established by the frozen diagnostics contract.

    This checks only relationships defined by the source schema. It cannot prove
    that an allowed value was genuinely produced at inference time.
    """
    for family in _STAGE_FAMILIES:
        status = values[_STAGE_STATUS_FIELDS[family]]
        count_name = _STAGE_COUNT_FIELDS[family]
        count = values[count_name]
        summary_names = [f"{family}_{field}" for field in _SUMMARY_FIELDS]
        summaries = {name: values[name] for name in summary_names}
        if status in _UNAVAILABLE_STATUSES:
            contradictory = [
                name
                for name in (count_name, *summary_names)
                if values[name] is not None
            ]
            if contradictory:
                raise ValueError(
                    f"stage {family!r} is unavailable but has numeric values: "
                    f"{contradictory}"
                )
        elif status == "empty_namespace":
            if count != 0 or any(value is not None for value in summaries.values()):
                raise ValueError(
                    f"empty stage {family!r} requires zero candidates and null summaries"
                )
        elif status == "available":
            if count is None:
                raise ValueError(f"available stage {family!r} requires candidate count")
            required = [
                f"{family}_{field}"
                for field in ("top_score", "mean", "std", "min", "max", "range")
            ]
            if count == 0:
                if any(value is not None for value in summaries.values()):
                    raise ValueError(
                        f"zero-candidate stage {family!r} requires null summaries"
                    )
            else:
                if any(values[name] is None for name in required):
                    raise ValueError(
                        f"available stage {family!r} requires complete score summaries"
                    )
                margin = values[f"{family}_top1_top2_margin"]
                if (count == 1 and margin is not None) or (count > 1 and margin is None):
                    raise ValueError(
                        f"stage {family!r} margin is inconsistent with candidate count"
                    )
                minimum = values[f"{family}_min"]
                maximum = values[f"{family}_max"]
                mean = values[f"{family}_mean"]
                top_score = values[f"{family}_top_score"]
                score_range = values[f"{family}_range"]
                standard_deviation = values[f"{family}_std"]
                if minimum > maximum or standard_deviation < 0:
                    raise ValueError(f"stage {family!r} score summary is inconsistent")
                if not math.isclose(
                    score_range, maximum - minimum, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"stage {family!r} score range is inconsistent")
                tolerance = 1e-12
                if not (
                    minimum - tolerance <= mean <= maximum + tolerance
                ):
                    raise ValueError(
                        f"stage {family!r} mean is outside the score range"
                    )
                if not (minimum <= top_score <= maximum):
                    raise ValueError(
                        f"stage {family!r} top score is outside the score range"
                    )
                if margin is not None:
                    if margin < 0:
                        raise ValueError(f"stage {family!r} top margin is negative")
                    if margin > score_range + tolerance:
                        raise ValueError(
                            f"stage {family!r} top margin exceeds the score range"
                        )
                    implied_second = top_score - margin
                    if not (
                        minimum - tolerance
                        <= implied_second
                        <= maximum + tolerance
                    ):
                        raise ValueError(
                            f"stage {family!r} implied second score is outside "
                            "the score range"
                        )
        # ``externally_supplied_hits`` is intentionally not normalized here: the
        # frozen production feature mapping does not define which upstream stage
        # values remain semantically available in that case.

    dense_status = values["dense_status"]
    bm25_status = values["bm25_status"]
    overlap = values["dense_bm25_overlap_count"]
    if dense_status in _UNAVAILABLE_STATUSES or bm25_status in _UNAVAILABLE_STATUSES:
        if overlap is not None:
            raise ValueError("dense/BM25 overlap must be null when either stage is unavailable")
    elif dense_status == "available" and bm25_status == "available":
        if overlap is None:
            raise ValueError("available dense/BM25 stages require overlap count")
        if overlap > min(
            values["dense_candidate_count"], values["bm25_candidate_count"]
        ):
            raise ValueError("dense/BM25 overlap exceeds a candidate count")
    elif "empty_namespace" in {dense_status, bm25_status} and overlap != 0:
        raise ValueError("empty dense/BM25 stage requires zero overlap")

    reader_status = values["reader_status"]
    if reader_status != "available" and reader_status != "externally_supplied_hits":
        contradictory = [
            name for name in _READER_DIAGNOSTIC_FIELDS if values[name] is not None
        ]
        if contradictory:
            raise ValueError(
                f"reader is unavailable but has reader diagnostics: {sorted(contradictory)}"
            )
        if values["reader_found"] or values["reader_confidence"] != 0.0:
            raise ValueError(
                "unavailable reader requires found=false and confidence=0"
            )
    null_values = (
        values["reader_null_score_min"],
        values["reader_null_score_max"],
        values["reader_null_score_range"],
    )
    if all(value is not None for value in null_values):
        minimum, maximum, score_range = null_values
        if minimum > maximum or not math.isclose(
            score_range, maximum - minimum, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("reader null-score range is inconsistent")


def validate_observations(
    observations: Sequence[Mapping[str, Any]], *, expected_role: str
) -> ValidatedBatch:
    """Validate and extract an exact synthetic feature matrix.

    Input order is not trusted. Rows are sorted by ``(paper_id, question_id)`` so
    training-data fingerprints and fitted behavior are reproducible.
    """
    if expected_role not in SYNTHETIC_PARTITION_ROLES:
        raise ValueError(
            f"synthetic partition role must be one of {SYNTHETIC_PARTITION_ROLES}"
        )
    if isinstance(observations, (str, bytes)) or not observations:
        raise ValueError("observations must be a nonempty sequence")

    rows = []
    seen_questions: dict[str, str] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ValueError(f"observation {index} must be a mapping")
        _require_exact_keys(observation, _OBSERVATION_KEYS, f"observation {index}")
        if observation["schema"] != OBSERVATION_SCHEMA:
            raise ValueError(f"observation {index} has incompatible schema")
        if observation["origin"] != SYNTHETIC_ORIGIN:
            raise ValueError("calibrator mechanics accept synthetic observations only")
        paper_id = _require_nonempty_identifier(observation["paper_id"], "paper_id")
        question_id = _require_nonempty_identifier(
            observation["question_id"], "question_id"
        )
        if question_id in seen_questions:
            raise ValueError(
                f"duplicate question_id {question_id!r} in papers "
                f"{seen_questions[question_id]!r} and {paper_id!r}"
            )
        seen_questions[question_id] = paper_id

        partition = observation["partition"]
        if not isinstance(partition, Mapping):
            raise ValueError("partition must be a mapping")
        _require_exact_keys(partition, _PARTITION_KEYS, "partition")
        if partition["schema"] != PARTITION_SCHEMA:
            raise ValueError("partition has incompatible schema")
        role = partition["role"]
        if role not in SYNTHETIC_PARTITION_ROLES:
            raise ValueError(
                "synthetic mechanics have no test role; role must be train or validation"
            )
        if role != expected_role:
            raise ValueError(
                f"partition role {role!r} is inconsistent with expected {expected_role!r}"
            )

        features = observation["features"]
        if not isinstance(features, Mapping):
            raise ValueError("features must be a mapping")
        _require_exact_keys(features, _FEATURE_KEYS, "features")
        if features["schema"] != FEATURE_SCHEMA:
            raise ValueError("features have incompatible schema")
        values = features["values"]
        if not isinstance(values, Mapping):
            raise ValueError("feature values must be a mapping")
        feature_row = _validate_feature_values(values)

        label = observation["label"]
        if not isinstance(label, Mapping):
            raise ValueError("label must be a mapping")
        _require_exact_keys(label, _LABEL_KEYS, "label")
        if label["schema"] != LABEL_SCHEMA:
            raise ValueError("label has incompatible schema")
        answerable = label["answerable"]
        if not isinstance(answerable, bool):
            raise ValueError("answerable label must be a bool; coercion is prohibited")
        rows.append((paper_id, question_id, feature_row, answerable))

    rows.sort(key=lambda row: (row[0], row[1]))
    content_payload = [
        {
            "paper_id": paper_id,
            "question_id": question_id,
            "features": dict(zip(RAW_FEATURE_ORDER, feature_row)),
            "answerable": answerable,
            "partition_role": expected_role,
        }
        for paper_id, question_id, feature_row, answerable in rows
    ]
    X = np.asarray([row[2] for row in rows], dtype=object)
    y = np.asarray([row[3] for row in rows], dtype=bool)
    X.setflags(write=False)
    y.setflags(write=False)
    return ValidatedBatch(
        X=X,
        y=y,
        paper_ids=tuple(row[0] for row in rows),
        question_ids=tuple(row[1] for row in rows),
        role=expected_role,
        content_digest=stable_fingerprint(content_payload),
    )


def validate_partition_pair(
    train_observations: Sequence[Mapping[str, Any]],
    validation_observations: Sequence[Mapping[str, Any]],
) -> tuple[ValidatedBatch, ValidatedBatch]:
    """Validate synthetic train/validation roles and reject identity overlap."""
    train = validate_observations(train_observations, expected_role="train")
    validation = validate_observations(
        validation_observations, expected_role="validation"
    )
    question_overlap = set(train.question_ids) & set(validation.question_ids)
    if question_overlap:
        raise ValueError(f"question identities cross partitions: {sorted(question_overlap)}")
    paper_overlap = set(train.paper_ids) & set(validation.paper_ids)
    if paper_overlap:
        raise ValueError(f"paper identities cross partitions: {sorted(paper_overlap)}")
    return train, validation


class _LeakageSafePreprocessor(TransformerMixin, BaseEstimator):
    """Fixed-schema, fold-fitted transformer for the frozen feature semantics.

    Numeric ``None`` values receive a training-fold median and an always-present
    missingness indicator. Entirely missing training columns fail closed because
    the protocol supplies no defensible median. Count values are imputed on their
    raw scale and then transformed with ``log1p``. Continuous values are imputed,
    standardized independently, and use scale one for zero variance. Binary
    fields remain 0/1. Categories use a fixed one-hot vocabulary including
    explicit ``unknown``; unsupported values are rejected before this class.
    """

    def __init__(self, schema: str = PREPROCESSOR_SCHEMA):
        self.schema = schema

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if self.schema != PREPROCESSOR_SCHEMA:
            raise ValueError("incompatible preprocessor schema")
        values = self._validate_shape(X)
        count_values = self._numeric_block(values, 0, len(COUNT_FEATURES))
        continuous_start = len(COUNT_FEATURES)
        continuous_values = self._numeric_block(
            values, continuous_start, continuous_start + len(CONTINUOUS_FEATURES)
        )
        self.count_medians_ = self._medians(count_values, COUNT_FEATURES)
        self.continuous_medians_ = self._medians(
            continuous_values, CONTINUOUS_FEATURES
        )
        imputed_continuous = np.where(
            np.isnan(continuous_values), self.continuous_medians_, continuous_values
        )
        self.continuous_means_ = imputed_continuous.mean(axis=0)
        scales = imputed_continuous.std(axis=0)
        self.continuous_scales_ = np.where(scales == 0.0, 1.0, scales)
        self.n_features_in_ = len(RAW_FEATURE_ORDER)
        self.feature_names_out_ = np.asarray(_transformed_feature_names(), dtype=object)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "count_medians_",
                "continuous_medians_",
                "continuous_means_",
                "continuous_scales_",
                "feature_names_out_",
            ),
        )
        values = self._validate_shape(X)
        count_end = len(COUNT_FEATURES)
        continuous_end = count_end + len(CONTINUOUS_FEATURES)
        binary_end = continuous_end + len(BINARY_FEATURES)

        counts = self._numeric_block(values, 0, count_end)
        count_missing = np.isnan(counts).astype(float)
        counts = np.where(np.isnan(counts), self.count_medians_, counts)
        counts = np.log1p(counts)

        continuous = self._numeric_block(values, count_end, continuous_end)
        continuous_missing = np.isnan(continuous).astype(float)
        continuous = np.where(
            np.isnan(continuous), self.continuous_medians_, continuous
        )
        continuous = (
            continuous - self.continuous_means_
        ) / self.continuous_scales_

        binary = values[:, continuous_end:binary_end].astype(float)
        categorical = values[:, binary_end:]
        encoded = []
        category_sets = (QUESTION_FORM_CATEGORIES,) + (AVAILABILITY_CATEGORIES,) * (
            len(CATEGORICAL_FEATURES) - 1
        )
        for column, categories in enumerate(category_sets):
            encoded.append(
                np.asarray(
                    [
                        [1.0 if value == category else 0.0 for category in categories]
                        for value in categorical[:, column]
                    ],
                    dtype=float,
                )
            )
        result = np.hstack(
            [
                counts,
                count_missing,
                continuous,
                continuous_missing,
                binary,
                *encoded,
            ]
        )
        if result.shape[1] != len(self.feature_names_out_):
            raise RuntimeError("transformed feature ordering is inconsistent")
        if not np.isfinite(result).all():
            raise ValueError("preprocessing produced a non-finite value")
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and tuple(input_features) != RAW_FEATURE_ORDER:
            raise ValueError("input feature ordering does not match the frozen schema")
        return self.feature_names_out_.copy()

    @staticmethod
    def _validate_shape(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != len(RAW_FEATURE_ORDER):
            raise ValueError(
                f"feature matrix must have {len(RAW_FEATURE_ORDER)} ordered columns"
            )
        if values.shape[0] == 0:
            raise ValueError("feature matrix must contain at least one row")
        normalized = [
            _validate_feature_values(dict(zip(RAW_FEATURE_ORDER, row)))
            for row in values
        ]
        return np.asarray(normalized, dtype=object)

    @staticmethod
    def _numeric_block(values: np.ndarray, start: int, end: int) -> np.ndarray:
        return np.asarray(
            [
                [np.nan if value is None else float(value) for value in row[start:end]]
                for row in values
            ],
            dtype=float,
        )

    @staticmethod
    def _medians(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
        all_missing = np.all(np.isnan(values), axis=0)
        if all_missing.any():
            missing_names = [name for name, missing in zip(names, all_missing) if missing]
            raise ValueError(
                "training fold has entirely missing numeric features with no median: "
                f"{missing_names}"
            )
        return np.nanmedian(values, axis=0)


def _transformed_feature_names() -> tuple[str, ...]:
    category_names = []
    for name in CATEGORICAL_FEATURES:
        categories = (
            QUESTION_FORM_CATEGORIES
            if name == "question_form"
            else AVAILABILITY_CATEGORIES
        )
        category_names.extend(f"{name}={category}" for category in categories)
    return (
        *COUNT_FEATURES,
        *(f"{name}__missing" for name in COUNT_FEATURES),
        *CONTINUOUS_FEATURES,
        *(f"{name}__missing" for name in CONTINUOUS_FEATURES),
        *BINARY_FEATURES,
        *category_names,
    )


def preprocessing_configuration() -> dict[str, Any]:
    return {
        "schema": PREPROCESSOR_SCHEMA,
        "ordered_raw_features": list(RAW_FEATURE_ORDER),
        "count_features": list(COUNT_FEATURES),
        "continuous_features": list(CONTINUOUS_FEATURES),
        "binary_features": list(BINARY_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "count_transform": "training-fold raw median, then log1p",
        "continuous_transform": (
            "training-fold median, then training-fold standardization per column"
        ),
        "missing_indicators": "fixed indicator for every numeric feature",
        "entirely_missing_training_feature": "fail_closed",
        "zero_variance_scale": 1.0,
        "question_form_categories": list(QUESTION_FORM_CATEGORIES),
        "availability_categories": list(AVAILABILITY_CATEGORIES),
        "unknown_category_policy": "explicit_unknown_only; reject other categories",
    }


def estimator_configuration() -> dict[str, Any]:
    return {
        "schema": ESTIMATOR_SCHEMA,
        "c_grid": list(LOGISTIC_C_GRID),
        **LOGISTIC_SETTINGS,
    }


def cv_configuration() -> dict[str, Any]:
    return {
        "schema": CV_SCHEMA,
        "folds": CV_FOLDS,
        "repeats": CV_REPEATS,
        "seed": CV_SEED,
        "group": "synthetic_paper_id",
        "stratum": "paper_has_any_unanswerable_question",
        "assignment": (
            "SHA-256 order within stratum using seed+repeat; round-robin folds"
        ),
        "selection_metric": "mean_answerability_aurc_over_100_validation_folds",
        "tie_rule": "smaller_C",
    }


def label_configuration() -> dict[str, Any]:
    return {
        "schema": LABEL_SCHEMA,
        "field": "answerable",
        "type": "bool",
        "positive_class": True,
        "semantics": "synthetic fixture answerability only",
        "coercion": "prohibited",
    }


def _dependency_versions() -> dict[str, str]:
    return {
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
    }


def _require_pinned_dependencies() -> dict[str, str]:
    versions = _dependency_versions()
    expected = {"numpy": "2.5.4", "scikit-learn": "1.9.1"}
    pinned = {name: versions[name] for name in expected}
    if pinned != expected:
        raise RuntimeError(
            f"synthetic calibrator mechanics require pinned dependencies {expected}; "
            f"found {versions}"
        )
    return versions


def _implementation_identity() -> dict[str, Any]:
    module_directory = Path(__file__).resolve().parent
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "calibrator_mechanics_sha256": hash_file(Path(__file__)),
        "shadow_protocol_schema": PROTOCOL_SCHEMA,
        "shadow_protocol_sha256": hash_file(module_directory / "shadow_protocol.py"),
        "provenance_sha256": hash_file(module_directory / "provenance.py"),
    }


def _runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def mechanics_configuration() -> dict[str, Any]:
    """Return the complete non-data configuration covered by compatibility checks."""
    return {
        "schemas": {
            "observation": OBSERVATION_SCHEMA,
            "features": FEATURE_SCHEMA,
            "label": LABEL_SCHEMA,
            "partition": PARTITION_SCHEMA,
            "preprocessor": PREPROCESSOR_SCHEMA,
            "estimator": ESTIMATOR_SCHEMA,
            "cv": CV_SCHEMA,
            "artifact": ARTIFACT_SCHEMA,
            "threshold": THRESHOLD_SCHEMA,
        },
        "origin": SYNTHETIC_ORIGIN,
        "partition_roles": list(SYNTHETIC_PARTITION_ROLES),
        "preprocessing": preprocessing_configuration(),
        "estimator": estimator_configuration(),
        "cv": cv_configuration(),
        "label": label_configuration(),
        "dependencies": _dependency_versions(),
        "implementation": _implementation_identity(),
        "runtime": _runtime_identity(),
    }


def mechanics_configuration_fingerprint() -> str:
    return stable_fingerprint(mechanics_configuration())


def _build_calibrator_pipeline(c_value: float) -> Pipeline:
    """Return an unsupported internal pipeline.

    Underscore naming and ``__all__`` exclusion are conventions, not access control.
    """
    if (
        isinstance(c_value, bool)
        or not isinstance(c_value, Real)
        or float(c_value) not in LOGISTIC_C_GRID
    ):
        raise ValueError(f"C must be one of {LOGISTIC_C_GRID}")
    _require_pinned_dependencies()
    settings = {
        key: value
        for key, value in LOGISTIC_SETTINGS.items()
        if key not in {"dependency", "estimator"}
    }
    return Pipeline(
        steps=[
            ("preprocess", _LeakageSafePreprocessor()),
            ("estimator", LogisticRegression(C=float(c_value), **settings)),
        ]
    )


def build_grouped_fold_assignments(batch: ValidatedBatch) -> tuple[dict[str, Any], ...]:
    """Build and adversarially validate the exact frozen grouped assignments."""
    if batch.role != "train":
        raise ValueError("grouped CV accepts the synthetic train role only")
    paper_has_unanswerable: dict[str, bool] = {}
    for paper_id, answerable in zip(batch.paper_ids, batch.y):
        paper_has_unanswerable[paper_id] = paper_has_unanswerable.get(
            paper_id, False
        ) or not bool(answerable)

    plans = []
    papers = set(batch.paper_ids)
    for repeat in range(CV_REPEATS):
        assignments = grouped_cv_assignments(paper_has_unanswerable, repeat)
        if set(assignments) != papers:
            raise ValueError("fold assignment is incomplete")
        fold_values = set(assignments.values())
        if fold_values != set(range(CV_FOLDS)):
            raise ValueError(
                f"repeat {repeat} has empty folds under the frozen assignment: "
                f"{sorted(fold_values)}"
            )
        for fold in range(CV_FOLDS):
            validation_papers = {
                paper for paper, assigned in assignments.items() if assigned == fold
            }
            training_papers = papers - validation_papers
            if not validation_papers or not training_papers:
                raise ValueError(f"repeat {repeat} fold {fold} has an empty partition")
            if training_papers & validation_papers:
                raise ValueError("paper identity crosses a CV boundary")
            train_labels = [
                bool(label)
                for paper, label in zip(batch.paper_ids, batch.y)
                if paper in training_papers
            ]
            validation_labels = [
                bool(label)
                for paper, label in zip(batch.paper_ids, batch.y)
                if paper in validation_papers
            ]
            if len(set(train_labels)) != 2 or len(set(validation_labels)) != 2:
                raise ValueError(
                    f"repeat {repeat} fold {fold} lacks both answerability classes"
                )
        plans.append(
            {
                "repeat": repeat,
                "paper_to_validation_fold": {
                    paper: assignments[paper] for paper in sorted(assignments)
                },
            }
        )
    return tuple(plans)


def select_c_from_fold_scores(
    fold_scores: Mapping[float, Sequence[float]],
) -> tuple[float, tuple[tuple[float, float], ...]]:
    """Select minimum mean AURC; exact ties select smaller C."""
    if any(isinstance(key, bool) or not isinstance(key, Real) for key in fold_scores):
        raise ValueError("fold score C keys must be numeric without coercion")
    if set(float(key) for key in fold_scores) != set(LOGISTIC_C_GRID):
        raise ValueError("fold scores must cover the complete frozen C grid")
    means = []
    for c_value in LOGISTIC_C_GRID:
        raw_scores = fold_scores[c_value]
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in raw_scores
        ):
            raise ValueError("fold AURC values must be numeric without coercion")
        scores = [float(value) for value in raw_scores]
        if len(scores) != CV_FOLDS * CV_REPEATS:
            raise ValueError("each C must have exactly 100 held-out-fold scores")
        if any(not math.isfinite(value) for value in scores):
            raise ValueError("fold AURC values must be finite")
        means.append((float(c_value), float(np.mean(scores))))
    selected = min(means, key=lambda item: (item[1], item[0]))[0]
    return selected, tuple(means)


def run_grouped_cv(
    observations: Sequence[Mapping[str, Any]],
) -> CVSelection:
    """Run the frozen 5x20 mechanics on synthetic train observations only."""
    batch = validate_observations(observations, expected_role="train")
    return _run_grouped_cv_batch(batch)


def _run_grouped_cv_batch(batch: ValidatedBatch) -> CVSelection:
    """Run grouped CV over one already validated, immutable snapshot."""
    if batch.role != "train":
        raise ValueError("grouped CV accepts the synthetic train role only")
    plans = build_grouped_fold_assignments(batch)
    fold_scores: dict[float, list[float]] = {
        float(c_value): [] for c_value in LOGISTIC_C_GRID
    }
    papers_array = np.asarray(batch.paper_ids, dtype=object)

    for plan in plans:
        assignments = plan["paper_to_validation_fold"]
        for fold in range(CV_FOLDS):
            validation_mask = np.asarray(
                [assignments[paper] == fold for paper in batch.paper_ids], dtype=bool
            )
            training_mask = ~validation_mask
            if set(papers_array[training_mask]) & set(papers_array[validation_mask]):
                raise RuntimeError("paper leakage detected after fold construction")
            for c_value in LOGISTIC_C_GRID:
                pipeline = _build_calibrator_pipeline(c_value)
                pipeline.fit(batch.X[training_mask], batch.y[training_mask])
                classes = list(pipeline.named_steps["estimator"].classes_)
                positive_index = classes.index(True)
                scores = pipeline.predict_proba(batch.X[validation_mask])[
                    :, positive_index
                ]
                aurc, _ = answerability_aurc(
                    scores.tolist(), batch.y[validation_mask].tolist()
                )
                fold_scores[float(c_value)].append(aurc)

    selected_c, means = select_c_from_fold_scores(fold_scores)
    assignments_fingerprint = stable_fingerprint(list(plans))
    return CVSelection(
        selected_c=selected_c,
        mean_aurc_by_c=means,
        fold_aurc_by_c=tuple(
            (c_value, tuple(fold_scores[c_value])) for c_value in LOGISTIC_C_GRID
        ),
        assignments=plans,
        assignments_fingerprint=assignments_fingerprint,
    )


def _pipeline_state(pipeline: Pipeline) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["estimator"]
    check_is_fitted(estimator)
    return {
        "preprocessor": {
            "schema": preprocessor.schema,
            "count_medians": preprocessor.count_medians_.tolist(),
            "continuous_medians": preprocessor.continuous_medians_.tolist(),
            "continuous_means": preprocessor.continuous_means_.tolist(),
            "continuous_scales": preprocessor.continuous_scales_.tolist(),
            "transformed_feature_order": preprocessor.get_feature_names_out().tolist(),
        },
        "estimator": {
            "schema": ESTIMATOR_SCHEMA,
            "classes": estimator.classes_.tolist(),
            "coef": estimator.coef_.tolist(),
            "intercept": estimator.intercept_.tolist(),
            "n_iter": estimator.n_iter_.tolist(),
        },
    }


def fit_synthetic_calibrator(
    observations: Sequence[Mapping[str, Any]],
) -> SyntheticCalibratorArtifact:
    """Select and refit a calibrator on an explicit synthetic train partition."""
    batch = validate_observations(observations, expected_role="train")
    selection = _run_grouped_cv_batch(batch)
    pipeline = _build_calibrator_pipeline(selection.selected_c)
    pipeline.fit(batch.X, batch.y)
    fitted_state = _pipeline_state(pipeline)
    fitted_state_fingerprint = stable_fingerprint(fitted_state)
    configuration = mechanics_configuration()
    configuration_fingerprint = stable_fingerprint(configuration)
    manifest_without_fingerprint = {
        "schema": ARTIFACT_SCHEMA,
        "purpose": "synthetic-only-calibrator-mechanics",
        "configuration": configuration,
        "configuration_fingerprint": configuration_fingerprint,
        "schemas": configuration["schemas"],
        "ordered_raw_features": list(RAW_FEATURE_ORDER),
        "ordered_transformed_features": pipeline.named_steps[
            "preprocess"
        ].get_feature_names_out().tolist(),
        "label": label_configuration(),
        "preprocessing": preprocessing_configuration(),
        "estimator": estimator_configuration(),
        "cv": {
            **cv_configuration(),
            "assignments": list(selection.assignments),
            "assignments_fingerprint": selection.assignments_fingerprint,
            "mean_aurc_by_c": [list(item) for item in selection.mean_aurc_by_c],
        },
        "dependencies": configuration["dependencies"],
        "training_partition_role": "train",
        "training_data_digest": batch.content_digest,
        "training_rows": len(batch.question_ids),
        "training_papers": len(set(batch.paper_ids)),
        "selected_c": selection.selected_c,
        "fitted_state_fingerprint": fitted_state_fingerprint,
        "fingerprint_limit": FINGERPRINT_LIMIT,
    }
    manifest = {
        **manifest_without_fingerprint,
        "artifact_fingerprint": stable_fingerprint(manifest_without_fingerprint),
    }
    return SyntheticCalibratorArtifact(
        pipeline=pipeline, manifest=manifest, fitted_state=fitted_state
    )


def predict_synthetic_scores(
    artifact: SyntheticCalibratorArtifact,
    observations: Sequence[Mapping[str, Any]],
    *,
    expected_role: str,
) -> dict[str, float]:
    """Score an explicitly named synthetic train or validation partition."""
    validate_artifact(artifact)
    batch = validate_observations(observations, expected_role=expected_role)
    return _predict_validated_batch(artifact, batch)


def _predict_validated_batch(
    artifact: SyntheticCalibratorArtifact, batch: ValidatedBatch
) -> dict[str, float]:
    """Score one already validated snapshot without reopening caller input."""
    classes = list(artifact.pipeline.named_steps["estimator"].classes_)
    positive_index = classes.index(True)
    scores = artifact.pipeline.predict_proba(batch.X)[:, positive_index]
    return {
        question_id: float(score)
        for question_id, score in zip(batch.question_ids, scores)
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"artifact {name} has incompatible shape")
    if any(
        isinstance(item, bool)
        or not isinstance(item, Real)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"artifact {name} must contain finite numeric values")
    return [float(item) for item in value]


def _validate_fitted_state_structure(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("artifact fitted state must be a mapping")
    _require_exact_keys(state, {"preprocessor", "estimator"}, "fitted state")
    preprocessor = state["preprocessor"]
    estimator = state["estimator"]
    if not isinstance(preprocessor, Mapping) or not isinstance(estimator, Mapping):
        raise ValueError("artifact fitted components must be mappings")
    _require_exact_keys(
        preprocessor,
        {
            "schema",
            "count_medians",
            "continuous_medians",
            "continuous_means",
            "continuous_scales",
            "transformed_feature_order",
        },
        "fitted preprocessor",
    )
    _require_exact_keys(
        estimator,
        {"schema", "classes", "coef", "intercept", "n_iter"},
        "fitted estimator",
    )
    if preprocessor["schema"] != PREPROCESSOR_SCHEMA:
        raise ValueError("fitted preprocessor schema is incompatible")
    if estimator["schema"] != ESTIMATOR_SCHEMA:
        raise ValueError("fitted estimator schema is incompatible")
    _finite_vector(
        preprocessor["count_medians"], len(COUNT_FEATURES), "count medians"
    )
    _finite_vector(
        preprocessor["continuous_medians"],
        len(CONTINUOUS_FEATURES),
        "continuous medians",
    )
    _finite_vector(
        preprocessor["continuous_means"],
        len(CONTINUOUS_FEATURES),
        "continuous means",
    )
    scales = _finite_vector(
        preprocessor["continuous_scales"],
        len(CONTINUOUS_FEATURES),
        "continuous scales",
    )
    if any(scale <= 0 for scale in scales):
        raise ValueError("artifact continuous scales must be positive")
    if preprocessor["transformed_feature_order"] != list(
        _transformed_feature_names()
    ):
        raise ValueError("fitted preprocessor feature ordering is incompatible")
    if estimator["classes"] != [False, True]:
        raise ValueError("fitted estimator classes are incompatible")
    coefficient = estimator["coef"]
    if not isinstance(coefficient, list) or len(coefficient) != 1:
        raise ValueError("artifact estimator coefficient shape is incompatible")
    _finite_vector(
        coefficient[0], len(_transformed_feature_names()), "estimator coefficients"
    )
    _finite_vector(estimator["intercept"], 1, "estimator intercept")
    n_iter = estimator["n_iter"]
    if (
        not isinstance(n_iter, list)
        or len(n_iter) != 1
        or isinstance(n_iter[0], bool)
        or not isinstance(n_iter[0], Integral)
        or int(n_iter[0]) < 1
    ):
        raise ValueError("artifact estimator iteration count is incompatible")


def validate_artifact(
    artifact: SyntheticCalibratorArtifact,
    *,
    expected_configuration_fingerprint: str | None = None,
) -> None:
    """Fail closed on incompatible metadata or fitted state."""
    if not isinstance(artifact, SyntheticCalibratorArtifact):
        raise ValueError("artifact has the wrong type")
    manifest = artifact.manifest
    expected_keys = {
        "schema",
        "purpose",
        "configuration",
        "configuration_fingerprint",
        "schemas",
        "ordered_raw_features",
        "ordered_transformed_features",
        "label",
        "preprocessing",
        "estimator",
        "cv",
        "dependencies",
        "training_partition_role",
        "training_data_digest",
        "training_rows",
        "training_papers",
        "selected_c",
        "fitted_state_fingerprint",
        "fingerprint_limit",
        "artifact_fingerprint",
    }
    _require_exact_keys(manifest, expected_keys, "artifact manifest")
    if manifest["schema"] != ARTIFACT_SCHEMA:
        raise ValueError("artifact schema is incompatible")
    if manifest["purpose"] != "synthetic-only-calibrator-mechanics":
        raise ValueError("artifact purpose is incompatible")
    if manifest["training_partition_role"] != "train":
        raise ValueError("artifact was not fitted on the synthetic train role")
    if manifest["fingerprint_limit"] != FINGERPRINT_LIMIT:
        raise ValueError("artifact fingerprint limitation metadata is incompatible")
    if (
        isinstance(manifest["selected_c"], bool)
        or not isinstance(manifest["selected_c"], Real)
        or float(manifest["selected_c"]) not in LOGISTIC_C_GRID
    ):
        raise ValueError("artifact selected C is incompatible")
    loaded_configuration = manifest["configuration"]
    if not isinstance(loaded_configuration, Mapping):
        raise ValueError("artifact configuration must be a mapping")
    calculated_loaded_configuration = stable_fingerprint(loaded_configuration)
    current_configuration = mechanics_configuration()
    calculated_current_configuration = stable_fingerprint(current_configuration)
    if manifest["configuration_fingerprint"] != calculated_loaded_configuration:
        raise ValueError("artifact declared configuration fingerprint is invalid")
    if calculated_loaded_configuration != calculated_current_configuration:
        raise ValueError("artifact configuration fingerprint is incompatible")
    if expected_configuration_fingerprint is not None:
        if not _is_sha256(expected_configuration_fingerprint):
            raise ValueError("expected configuration fingerprint is malformed")
        if expected_configuration_fingerprint != calculated_current_configuration:
            raise ValueError("expected configuration fingerprint is incompatible")
    if manifest["dependencies"] != _require_pinned_dependencies():
        raise ValueError("artifact dependency versions are incompatible")
    current_metadata = current_configuration
    if dict(loaded_configuration) != current_metadata:
        raise ValueError("artifact loaded configuration metadata is incompatible")
    if manifest["schemas"] != current_metadata["schemas"]:
        raise ValueError("artifact schema metadata is incompatible")
    if manifest["label"] != label_configuration():
        raise ValueError("artifact label semantics are incompatible")
    if manifest["preprocessing"] != preprocessing_configuration():
        raise ValueError("artifact preprocessing configuration is incompatible")
    if manifest["estimator"] != estimator_configuration():
        raise ValueError("artifact estimator configuration is incompatible")
    if manifest["ordered_raw_features"] != list(RAW_FEATURE_ORDER):
        raise ValueError("artifact raw feature ordering is incompatible")
    if manifest["ordered_transformed_features"] != list(
        _transformed_feature_names()
    ):
        raise ValueError("artifact transformed feature ordering is incompatible")
    if stable_fingerprint(artifact.fitted_state) != manifest[
        "fitted_state_fingerprint"
    ]:
        raise ValueError("artifact fitted state fingerprint is invalid")
    if not _is_sha256(manifest["training_data_digest"]):
        raise ValueError("artifact training data digest is invalid")
    if not _is_sha256(manifest["configuration_fingerprint"]):
        raise ValueError("artifact configuration fingerprint is invalid")
    if not _is_sha256(manifest["fitted_state_fingerprint"]):
        raise ValueError("artifact fitted state fingerprint is invalid")
    if not _is_sha256(manifest["artifact_fingerprint"]):
        raise ValueError("artifact fingerprint is invalid")
    if (
        isinstance(manifest["training_rows"], bool)
        or not isinstance(manifest["training_rows"], Integral)
        or manifest["training_rows"] < 1
        or isinstance(manifest["training_papers"], bool)
        or not isinstance(manifest["training_papers"], Integral)
        or manifest["training_papers"] < CV_FOLDS
    ):
        raise ValueError("artifact training counts are incompatible")
    cv_metadata = manifest["cv"]
    if not isinstance(cv_metadata, Mapping):
        raise ValueError("artifact CV metadata must be a mapping")
    _require_exact_keys(
        cv_metadata,
        {
            *cv_configuration().keys(),
            "assignments",
            "assignments_fingerprint",
            "mean_aurc_by_c",
        },
        "artifact CV metadata",
    )
    for key, value in cv_configuration().items():
        if cv_metadata.get(key) != value:
            raise ValueError(f"artifact CV configuration {key!r} is incompatible")
    assignments = cv_metadata.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != CV_REPEATS:
        raise ValueError("artifact CV assignments are incomplete")
    paper_set: set[str] | None = None
    for repeat, plan in enumerate(assignments):
        if not isinstance(plan, Mapping):
            raise ValueError("artifact CV assignment plan must be a mapping")
        _require_exact_keys(
            plan, {"repeat", "paper_to_validation_fold"}, "CV assignment plan"
        )
        if plan["repeat"] != repeat or not isinstance(
            plan["paper_to_validation_fold"], Mapping
        ):
            raise ValueError("artifact CV assignment repeat is incompatible")
        mapping = plan["paper_to_validation_fold"]
        if not mapping or any(
            not isinstance(paper, str)
            or isinstance(fold, bool)
            or not isinstance(fold, Integral)
            or fold not in range(CV_FOLDS)
            for paper, fold in mapping.items()
        ):
            raise ValueError("artifact CV paper assignments are incompatible")
        for paper in mapping:
            _require_nonempty_identifier(paper, "artifact CV paper identity")
        if set(mapping.values()) != set(range(CV_FOLDS)):
            raise ValueError("artifact CV paper assignments contain empty folds")
        if paper_set is None:
            paper_set = set(mapping)
        elif set(mapping) != paper_set:
            raise ValueError("artifact CV paper identity set changes across repeats")
    if paper_set is None:
        raise ValueError("artifact CV assignments are incomplete")
    if manifest["training_papers"] != len(paper_set):
        raise ValueError(
            "artifact training paper count disagrees with CV assignments"
        )
    if manifest["training_rows"] < manifest["training_papers"]:
        raise ValueError(
            "artifact training row count is smaller than its paper count"
        )
    if stable_fingerprint(assignments) != cv_metadata.get("assignments_fingerprint"):
        raise ValueError("artifact CV assignment fingerprint is invalid")
    means = cv_metadata.get("mean_aurc_by_c")
    if not isinstance(means, list) or len(means) != len(LOGISTIC_C_GRID):
        raise ValueError("artifact CV mean scores are incomplete")
    for expected_c, row in zip(LOGISTIC_C_GRID, means):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or row[0] != expected_c
            or isinstance(row[1], bool)
            or not isinstance(row[1], Real)
            or not math.isfinite(float(row[1]))
        ):
            raise ValueError("artifact CV mean scores are incompatible")
    selected_from_means = min(
        ((float(row[1]), float(row[0])) for row in means),
        key=lambda item: (item[0], item[1]),
    )[1]
    if float(manifest["selected_c"]) != selected_from_means:
        raise ValueError("artifact selected C disagrees with CV mean scores")
    _validate_fitted_state_structure(artifact.fitted_state)
    if stable_fingerprint(_pipeline_state(artifact.pipeline)) != manifest[
        "fitted_state_fingerprint"
    ]:
        raise ValueError("artifact in-memory pipeline state is incompatible")
    without_fingerprint = {
        key: value for key, value in manifest.items() if key != "artifact_fingerprint"
    }
    if stable_fingerprint(without_fingerprint) != manifest["artifact_fingerprint"]:
        raise ValueError("artifact manifest fingerprint is invalid")


def save_synthetic_artifact(
    path: str | Path, artifact: SyntheticCalibratorArtifact
) -> None:
    """Write a JSON-only synthetic artifact, refusing overwrite."""
    validate_artifact(artifact)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"manifest": artifact.manifest, "fitted_state": artifact.fitted_state},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _restore_pipeline(manifest: Mapping[str, Any], state: Mapping[str, Any]) -> Pipeline:
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("artifact schema is incompatible")
    selected_c = manifest.get("selected_c")
    if isinstance(selected_c, bool) or not isinstance(selected_c, Real):
        raise ValueError("artifact selected C is incompatible")
    _validate_fitted_state_structure(state)
    pipeline = _build_calibrator_pipeline(float(selected_c))
    preprocessor = pipeline.named_steps["preprocess"]
    pre_state = state["preprocessor"]
    preprocessor.count_medians_ = np.asarray(pre_state["count_medians"], dtype=float)
    preprocessor.continuous_medians_ = np.asarray(
        pre_state["continuous_medians"], dtype=float
    )
    preprocessor.continuous_means_ = np.asarray(
        pre_state["continuous_means"], dtype=float
    )
    preprocessor.continuous_scales_ = np.asarray(
        pre_state["continuous_scales"], dtype=float
    )
    preprocessor.feature_names_out_ = np.asarray(
        pre_state["transformed_feature_order"], dtype=object
    )
    preprocessor.n_features_in_ = len(RAW_FEATURE_ORDER)

    estimator = pipeline.named_steps["estimator"]
    estimator_state = state["estimator"]
    estimator.classes_ = np.asarray(estimator_state["classes"], dtype=bool)
    estimator.coef_ = np.asarray(estimator_state["coef"], dtype=float)
    estimator.intercept_ = np.asarray(estimator_state["intercept"], dtype=float)
    estimator.n_iter_ = np.asarray(estimator_state["n_iter"], dtype=np.int32)
    estimator.n_features_in_ = estimator.coef_.shape[1]
    return pipeline


def load_synthetic_artifact(
    path: str | Path,
    *,
    expected_configuration_fingerprint: str | None = None,
) -> SyntheticCalibratorArtifact:
    """Load and validate the JSON-only representation; never unpickle code."""
    payload = _strict_json_loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be a mapping")
    _require_exact_keys(payload, {"manifest", "fitted_state"}, "artifact payload")
    if not isinstance(payload["manifest"], dict) or not isinstance(
        payload["fitted_state"], dict
    ):
        raise ValueError("artifact manifest and state must be mappings")
    pipeline = _restore_pipeline(payload["manifest"], payload["fitted_state"])
    artifact = SyntheticCalibratorArtifact(
        pipeline=pipeline,
        manifest=payload["manifest"],
        fitted_state=payload["fitted_state"],
    )
    validate_artifact(
        artifact,
        expected_configuration_fingerprint=expected_configuration_fingerprint,
    )
    return artifact


def _strict_json_loads(text: str) -> Any:
    """Parse standards-compliant JSON while rejecting duplicate object keys."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON numeric constant: {value}")

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonstandard_constant,
    )


def select_synthetic_threshold(
    observations: Sequence[Mapping[str, Any]],
    *,
    method: str,
    artifact: SyntheticCalibratorArtifact,
) -> dict[str, Any]:
    """Apply frozen threshold arithmetic to a synthetic validation role only."""
    batch = validate_observations(observations, expected_role="validation")
    validate_artifact(artifact)
    scores_by_question = _predict_validated_batch(artifact, batch)
    if set(scores_by_question) != set(batch.question_ids):
        raise RuntimeError("synthetic score identities do not match validation rows")
    numeric_scores = [scores_by_question[question_id] for question_id in batch.question_ids]
    parent_artifact_fingerprint = artifact.manifest["artifact_fingerprint"]
    if method == "matched_coverage":
        result = select_matched_coverage_threshold(numeric_scores, batch.y.tolist())
    elif method == "risk_target":
        result = select_risk_target_threshold(numeric_scores, batch.y.tolist())
    else:
        raise ValueError("threshold method must be matched_coverage or risk_target")
    serializable_result = {
        key: ("infinity" if value == math.inf else value)
        for key, value in result.items()
    }
    payload_without_fingerprint = {
        "schema": THRESHOLD_SCHEMA,
        "purpose": "synthetic-validation-threshold-mechanics",
        "partition_role": "validation",
        "method": method,
        "parent_artifact_fingerprint": parent_artifact_fingerprint,
        "validation_data_digest": batch.content_digest,
        "score_digest": stable_fingerprint(
            [
                {"question_id": question_id, "score": score}
                for question_id, score in zip(batch.question_ids, numeric_scores)
            ]
        ),
        "result": serializable_result,
    }
    return {
        **payload_without_fingerprint,
        "threshold_fingerprint": stable_fingerprint(payload_without_fingerprint),
    }
