"""Synthetic-only adversarial tests for Milestone 5D calibrator mechanics."""
from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

import eval.calibrator_mechanics as mechanics
from eval.calibrator_mechanics import (
    ALLOWED_FEATURES,
    ARTIFACT_SCHEMA,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    COUNT_FEATURES,
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    LOGISTIC_C_GRID,
    OBSERVATION_SCHEMA,
    PARTITION_SCHEMA,
    QUESTION_FORM_CATEGORIES,
    RAW_FEATURE_ORDER,
    build_grouped_fold_assignments,
    fit_synthetic_calibrator,
    load_synthetic_artifact,
    mechanics_configuration,
    mechanics_configuration_fingerprint,
    predict_synthetic_scores,
    run_grouped_cv,
    save_synthetic_artifact,
    select_c_from_fold_scores,
    select_synthetic_threshold,
    validate_artifact,
    validate_observations,
    validate_partition_pair,
)
from eval.provenance import stable_fingerprint
from eval.shadow_protocol import CV_FOLDS, CV_REPEATS


pytestmark = pytest.mark.filterwarnings(
    "ignore:'penalty' was deprecated in version 1.8:FutureWarning"
)


def _feature_values(seed: int) -> dict:
    values = {
        name: seed + index + 1 for index, name in enumerate(COUNT_FEATURES)
    }
    values.update(
        {
            name: (seed + 1) * 0.1 + index * 0.01
            for index, name in enumerate(CONTINUOUS_FEATURES)
        }
    )
    values.update({name: seed % 2 == 0 for name in BINARY_FEATURES})
    values["question_form"] = QUESTION_FORM_CATEGORIES[
        seed % (len(QUESTION_FORM_CATEGORIES) - 1)
    ]
    stage_status = ("available", "empty_namespace", "not_requested")[seed % 3]
    for family in ("dense", "bm25", "rrf", "reranker", "selected_context"):
        values[f"{family}_status"] = stage_status
        summary_names = [
            f"{family}_{name}"
            for name in (
                "top_score",
                "top1_top2_margin",
                "mean",
                "std",
                "min",
                "max",
                "range",
            )
        ]
        if stage_status == "available":
            top = (seed + 1) * 0.1
            values[f"{family}_candidate_count"] = 2
            values.update(
                {
                    summary_names[0]: top,
                    summary_names[1]: 0.25,
                    summary_names[2]: top - 0.5,
                    summary_names[3]: 0.5,
                    summary_names[4]: top - 1.0,
                    summary_names[5]: top,
                    summary_names[6]: 1.0,
                }
            )
        elif stage_status == "empty_namespace":
            values[f"{family}_candidate_count"] = 0
            values.update({name: None for name in summary_names})
        else:
            values[f"{family}_candidate_count"] = None
            values.update({name: None for name in summary_names})
    values["dense_bm25_overlap_count"] = (
        1 if stage_status == "available" else 0 if stage_status == "empty_namespace" else None
    )
    if seed % 2 == 0:
        values.update(
            {
                "reader_status": "available",
                "reader_found": True,
                "reader_confidence": 0.75,
                "reader_window_count": 2,
                "reader_candidate_start_count": 3,
                "reader_candidate_span_count": 4,
                "answer_passage_rank": 1,
                "reader_span_null_margin": 0.5,
                "reader_best_second_span_gap": 0.25,
                "reader_selected_null_score": 0.1,
                "reader_null_score_min": 0.1,
                "reader_null_score_max": 0.4,
                "reader_null_score_range": 0.3,
            }
        )
    else:
        values.update(
            {
                "reader_status": "no_retrieved_passages",
                "reader_found": False,
                "reader_confidence": 0.0,
                **{
                    name: None
                    for name in (
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
                    )
                },
            }
        )
    assert set(values) == ALLOWED_FEATURES
    return values


def _observation(
    *,
    paper_id: str,
    question_id: str,
    role: str,
    answerable: bool,
    seed: int,
) -> dict:
    return {
        "schema": OBSERVATION_SCHEMA,
        "origin": "synthetic",
        "paper_id": paper_id,
        "question_id": question_id,
        "partition": {"schema": PARTITION_SCHEMA, "role": role},
        "features": {"schema": FEATURE_SCHEMA, "values": _feature_values(seed)},
        "label": {"schema": LABEL_SCHEMA, "answerable": answerable},
    }


def _observations(role: str, prefix: str = "synthetic", papers: int = 10) -> list[dict]:
    rows = []
    for paper in range(papers):
        for question in range(2):
            rows.append(
                _observation(
                    paper_id=f"{prefix}-paper-{paper:02d}",
                    question_id=f"{prefix}-question-{paper:02d}-{question}",
                    role=role,
                    answerable=not (paper < 5 and question == 1),
                    seed=paper * 2 + question,
                )
            )
    return rows


@pytest.fixture(scope="module")
def synthetic_train() -> list[dict]:
    return _observations("train")


@pytest.fixture(scope="module")
def fitted_artifact(synthetic_train):
    return fit_synthetic_calibrator(synthetic_train)


def _fold_expectations(batch):
    plans = build_grouped_fold_assignments(batch)
    expectations = []
    for plan in plans:
        assignments = plan["paper_to_validation_fold"]
        for fold in range(CV_FOLDS):
            validation_mask = np.asarray(
                [assignments[paper] == fold for paper in batch.paper_ids], dtype=bool
            )
            for c_value in LOGISTIC_C_GRID:
                expectations.append((~validation_mask, validation_mask, c_value))
    return expectations


def _independent_numeric_state(X):
    count_end = len(COUNT_FEATURES)
    continuous_end = count_end + len(CONTINUOUS_FEATURES)

    def numeric_block(start, end):
        return np.asarray(
            [
                [np.nan if value is None else float(value) for value in row[start:end]]
                for row in X
            ],
            dtype=float,
        )

    counts = numeric_block(0, count_end)
    continuous = numeric_block(count_end, continuous_end)
    count_medians = np.nanmedian(counts, axis=0)
    continuous_medians = np.nanmedian(continuous, axis=0)
    imputed = np.where(np.isnan(continuous), continuous_medians, continuous)
    means = imputed.mean(axis=0)
    raw_scales = imputed.std(axis=0)
    scales = np.where(raw_scales == 0.0, 1.0, raw_scales)
    return count_medians, continuous_medians, means, scales


def _tracking_pipeline_factory(original_factory, records, *, replacement_batch=None):
    def factory(c_value):
        pipeline = original_factory(c_value)
        original_fit = pipeline.fit

        def tracked_fit(X, y, **fit_params):
            actual_X = replacement_batch.X if replacement_batch is not None else X
            actual_y = replacement_batch.y if replacement_batch is not None else y
            fitted = original_fit(actual_X, actual_y, **fit_params)
            preprocessor = fitted.named_steps["preprocess"]
            records.append(
                {
                    "c": float(c_value),
                    "X": np.asarray(actual_X, dtype=object).copy(),
                    "y": np.asarray(actual_y, dtype=bool).copy(),
                    "pipeline": fitted,
                    "preprocessor": preprocessor,
                    "pipeline_id": id(fitted),
                    "preprocessor_id": id(preprocessor),
                    "count_medians": preprocessor.count_medians_.copy(),
                    "continuous_medians": preprocessor.continuous_medians_.copy(),
                    "continuous_means": preprocessor.continuous_means_.copy(),
                    "continuous_scales": preprocessor.continuous_scales_.copy(),
                    "feature_names": tuple(preprocessor.get_feature_names_out()),
                }
            )
            return fitted

        pipeline.fit = tracked_fit
        return pipeline

    return factory


@pytest.mark.parametrize(
    "field",
    [
        "answerable",
        "paper_id",
        "question_id",
        "gold_answers",
        "gold_evidence",
        "citations",
        "retrieval_recall",
        "f1",
    ],
)
def test_feature_allowlist_rejects_labels_identity_and_outcomes(field):
    row = _observations("train")[0]
    row["features"]["values"][field] = 1

    with pytest.raises(ValueError, match="feature allowlist mismatch"):
        validate_observations([row], expected_role="train")


def test_feature_allowlist_rejects_unknown_and_missing_fields():
    unknown = _observations("train")[0]
    unknown["features"]["values"]["future_signal"] = 1.0
    missing = _observations("train")[0]
    del missing["features"]["values"][RAW_FEATURE_ORDER[0]]

    with pytest.raises(ValueError, match="unknown=.*future_signal"):
        validate_observations([unknown], expected_role="train")
    with pytest.raises(ValueError, match="missing="):
        validate_observations([missing], expected_role="train")


def test_labels_are_strict_booleans_and_not_coerced():
    row = _observations("train")[0]
    row["label"]["answerable"] = 1

    with pytest.raises(ValueError, match="must be a bool"):
        validate_observations([row], expected_role="train")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), "1"])
def test_invalid_numeric_values_fail_closed(invalid):
    row = _observations("train")[0]
    row["features"]["values"][CONTINUOUS_FEATURES[0]] = invalid

    with pytest.raises(ValueError, match="must be finite or null"):
        validate_observations([row], expected_role="train")


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_invalid_count_values_fail_closed(invalid):
    row = _observations("train")[0]
    row["features"]["values"][COUNT_FEATURES[0]] = invalid

    with pytest.raises(ValueError, match="count feature|must not be boolean"):
        validate_observations([row], expected_role="train")


def test_unsupported_category_requires_explicit_unknown():
    row = _observations("train")[0]
    row["features"]["values"]["reader_status"] = "new-secret-status"

    with pytest.raises(ValueError, match="must be one of"):
        validate_observations([row], expected_role="train")


def test_duplicate_question_identity_is_rejected():
    rows = _observations("train")[:2]
    rows[1]["question_id"] = rows[0]["question_id"]

    with pytest.raises(ValueError, match="duplicate question_id"):
        validate_observations(rows, expected_role="train")


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("paper_id", " paper"),
        ("paper_id", "paper "),
        ("question_id", " question"),
        ("question_id", "question "),
        ("paper_id", ""),
        ("question_id", ""),
        ("paper_id", "cafe\u0301"),
        ("question_id", "question-e\u0301"),
    ],
)
def test_noncanonical_identity_aliases_are_rejected(field, invalid):
    row = _observations("train")[0]
    row[field] = invalid

    with pytest.raises(ValueError, match="nonempty|whitespace|NFC"):
        validate_observations([row], expected_role="train")


def test_identity_policy_does_not_implicitly_case_fold():
    rows = _observations("train")[:2]
    rows[0]["paper_id"] = "Paper-A"
    rows[1]["paper_id"] = "paper-a"
    rows[0]["question_id"] = "Question-A"
    rows[1]["question_id"] = "question-a"

    batch = validate_observations(rows, expected_role="train")

    assert set(batch.paper_ids) == {"Paper-A", "paper-a"}
    assert set(batch.question_ids) == {"Question-A", "question-a"}


def test_missing_and_contradictory_partition_roles_fail_closed():
    missing = _observations("train")[0]
    del missing["partition"]["role"]
    contradictory = _observations("validation")[0]
    test_role = _observations("validation")[0]
    test_role["partition"]["role"] = "test"

    with pytest.raises(ValueError, match="partition keys mismatch"):
        validate_observations([missing], expected_role="train")
    with pytest.raises(ValueError, match="inconsistent"):
        validate_observations([contradictory], expected_role="train")
    with pytest.raises(ValueError, match="no test role"):
        validate_observations([test_role], expected_role="validation")


def test_non_synthetic_origin_is_rejected():
    row = _observations("train")[0]
    row["origin"] = "qasper"

    with pytest.raises(ValueError, match="synthetic observations only"):
        validate_observations([row], expected_role="train")


def test_unavailable_stage_rejects_numeric_diagnostics():
    row = _observations("train")[2]
    assert row["features"]["values"]["dense_status"] == "not_requested"
    row["features"]["values"]["dense_top_score"] = 1.0

    with pytest.raises(ValueError, match="unavailable but has numeric values"):
        validate_observations([row], expected_role="train")


@pytest.mark.parametrize(
    ("field", "value"),
    [("dense_candidate_count", 1), ("dense_top_score", 0.5)],
)
def test_empty_stage_rejects_candidates_or_summaries(field, value):
    row = _observations("train")[1]
    assert row["features"]["values"]["dense_status"] == "empty_namespace"
    row["features"]["values"][field] = value

    with pytest.raises(ValueError, match="empty stage"):
        validate_observations([row], expected_role="train")


def test_available_stage_rejects_inconsistent_count_and_summary():
    row = _observations("train")[0]
    assert row["features"]["values"]["dense_status"] == "available"
    row["features"]["values"]["dense_candidate_count"] = 0

    with pytest.raises(ValueError, match="zero-candidate stage"):
        validate_observations([row], expected_role="train")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"dense_mean": 1.0}, "mean is outside"),
        ({"dense_mean": -2.0}, "mean is outside"),
        ({"dense_top_score": 2.0}, "top score is outside"),
        ({"dense_top_score": -2.0}, "top score is outside"),
        ({"dense_top1_top2_margin": -0.25}, "top margin is negative"),
        ({"dense_top1_top2_margin": 2.0}, "top margin exceeds"),
        (
            {
                "dense_min": 0.0,
                "dense_max": 10.0,
                "dense_range": 10.0,
                "dense_mean": 5.0,
                "dense_std": 1.0,
                "dense_top_score": 1.0,
                "dense_top1_top2_margin": 5.0,
            },
            "implied second score is outside",
        ),
    ],
)
def test_available_stage_rejects_algebraic_summary_contradictions(changes, match):
    row = _observations("train")[0]
    values = row["features"]["values"]
    assert values["dense_status"] == "available"
    values.update(changes)

    with pytest.raises(ValueError, match=match):
        validate_observations([row], expected_role="train")


def test_unavailable_reader_rejects_reader_diagnostics():
    row = _observations("train")[1]
    assert row["features"]["values"]["reader_status"] == "no_retrieved_passages"
    row["features"]["values"]["reader_span_null_margin"] = 0.5

    with pytest.raises(ValueError, match="reader is unavailable"):
        validate_observations([row], expected_role="train")


def test_private_preprocessor_rejects_direct_unsupported_category(synthetic_train):
    batch = validate_observations(synthetic_train, expected_role="train")
    direct = batch.X.copy()
    direct[0, RAW_FEATURE_ORDER.index("question_form")] = "unsupported"

    with pytest.raises(ValueError, match="categorical feature"):
        mechanics._LeakageSafePreprocessor().fit(direct, batch.y)

    assert "_build_calibrator_pipeline" not in mechanics.__all__
    assert "_LeakageSafePreprocessor" not in mechanics.__all__


def test_partition_pair_rejects_paper_and_question_leakage():
    train = _observations("train", prefix="shared")
    validation = _observations("validation", prefix="shared")
    for row in validation:
        row["question_id"] = f"validation-{row['question_id']}"

    with pytest.raises(ValueError, match="paper identities cross partitions"):
        validate_partition_pair(train, validation)

    validation = _observations("validation", prefix="separate")
    validation[0]["question_id"] = train[0]["question_id"]
    with pytest.raises(ValueError, match="question identities cross partitions"):
        validate_partition_pair(train, validation)


def test_partition_swaps_are_rejected_by_operation_contracts(fitted_artifact):
    validation = _observations("validation", prefix="validation")
    train = _observations("train", prefix="train")

    with pytest.raises(ValueError, match="inconsistent"):
        run_grouped_cv(validation)
    with pytest.raises(ValueError, match="inconsistent"):
        select_synthetic_threshold(
            train,
            method="matched_coverage",
            artifact=fitted_artifact,
        )


def test_feature_matrix_order_is_deterministic_and_excludes_identity(synthetic_train):
    first = validate_observations(synthetic_train, expected_role="train")
    second = validate_observations(
        list(reversed(synthetic_train)), expected_role="train"
    )

    assert first.question_ids == second.question_ids
    assert first.content_digest == second.content_digest
    assert np.array_equal(first.X, second.X)
    assert not ({"paper_id", "question_id", "answerable"} & set(RAW_FEATURE_ORDER))


def test_validated_snapshot_is_defensive_and_read_only(synthetic_train):
    rows = copy.deepcopy(synthetic_train)
    batch = validate_observations(rows, expected_role="train")
    original_X = batch.X.copy()
    original_y = batch.y.copy()
    original_digest = batch.content_digest

    rows[0]["features"]["values"]["question_length_tokens"] += 10_000
    rows[0]["label"]["answerable"] = not rows[0]["label"]["answerable"]

    assert np.array_equal(batch.X, original_X)
    assert np.array_equal(batch.y, original_y)
    assert batch.content_digest == original_digest
    with pytest.raises(ValueError, match="read-only"):
        batch.X[0, 0] = 999
    with pytest.raises(ValueError, match="read-only"):
        batch.y[0] = not batch.y[0]


def test_final_fit_reuses_one_snapshot_despite_caller_mutation(
    monkeypatch, synthetic_train, fitted_artifact
):
    rows = copy.deepcopy(synthetic_train)
    expected_digest = validate_observations(
        copy.deepcopy(rows), expected_role="train"
    ).content_digest
    original_validate = mechanics.validate_observations
    calls = 0

    def validate_then_mutate(observations, *, expected_role):
        nonlocal calls
        calls += 1
        batch = original_validate(observations, expected_role=expected_role)
        observations[0]["features"]["values"]["question_length_tokens"] += 99_999
        observations[0]["label"]["answerable"] = not observations[0]["label"][
            "answerable"
        ]
        return batch

    monkeypatch.setattr(mechanics, "validate_observations", validate_then_mutate)

    repaired = fit_synthetic_calibrator(rows)

    assert calls == 1
    assert repaired.manifest["training_data_digest"] == expected_digest
    assert repaired.manifest["artifact_fingerprint"] == fitted_artifact.manifest[
        "artifact_fingerprint"
    ]
    assert repaired.fitted_state == fitted_artifact.fitted_state


def test_threshold_reuses_one_validation_snapshot_despite_label_mutation(
    monkeypatch, fitted_artifact
):
    baseline_rows = _observations("validation", prefix="snapshot-threshold")
    baseline = select_synthetic_threshold(
        copy.deepcopy(baseline_rows),
        method="matched_coverage",
        artifact=fitted_artifact,
    )
    rows = copy.deepcopy(baseline_rows)
    original_validate = mechanics.validate_observations
    calls = 0

    def validate_then_mutate(observations, *, expected_role):
        nonlocal calls
        calls += 1
        batch = original_validate(observations, expected_role=expected_role)
        for observation in observations:
            observation["label"]["answerable"] = not observation["label"][
                "answerable"
            ]
        return batch

    monkeypatch.setattr(mechanics, "validate_observations", validate_then_mutate)

    selected = select_synthetic_threshold(
        rows, method="matched_coverage", artifact=fitted_artifact
    )

    assert calls == 1
    assert selected == baseline


def test_missing_values_have_fixed_indicators_and_training_medians(synthetic_train):
    rows = copy.deepcopy(synthetic_train)
    missing_name = CONTINUOUS_FEATURES[0]
    rows[0]["features"]["values"][missing_name] = None
    batch = validate_observations(rows, expected_role="train")
    preprocessor = mechanics._LeakageSafePreprocessor().fit(batch.X, batch.y)
    transformed = preprocessor.transform(batch.X)
    names = preprocessor.get_feature_names_out().tolist()

    indicator = names.index(f"{missing_name}__missing")
    missing_row = batch.question_ids.index(rows[0]["question_id"])
    assert transformed[missing_row, indicator] == 1.0
    assert transformed[:, indicator].sum() == 1.0


def test_entirely_missing_training_feature_fails_closed(synthetic_train):
    rows = copy.deepcopy(synthetic_train)
    missing_name = "reader_best_second_span_gap"
    for row in rows:
        row["features"]["values"][missing_name] = None
    batch = validate_observations(rows, expected_role="train")

    with pytest.raises(ValueError, match="entirely missing"):
        mechanics._LeakageSafePreprocessor().fit(batch.X, batch.y)


def test_absent_training_category_has_fixed_validation_column(synthetic_train):
    train = copy.deepcopy(synthetic_train)
    for row in train:
        row["features"]["values"]["question_form"] = "what"
    validation = _observations("validation", prefix="validation")
    validation[0]["features"]["values"]["question_form"] = "why"
    train_batch = validate_observations(train, expected_role="train")
    validation_batch = validate_observations(validation, expected_role="validation")
    preprocessor = mechanics._LeakageSafePreprocessor().fit(train_batch.X, train_batch.y)
    transformed = preprocessor.transform(validation_batch.X)
    names = preprocessor.get_feature_names_out().tolist()

    why_column = names.index("question_form=why")
    target_row = validation_batch.question_ids.index(validation[0]["question_id"])
    assert transformed[target_row, why_column] == 1.0


def test_validation_sentinels_do_not_change_training_preprocessing(synthetic_train):
    train_batch = validate_observations(synthetic_train, expected_role="train")
    validation = _observations("validation", prefix="sentinel")
    validation[0]["features"]["values"][CONTINUOUS_FEATURES[0]] = 1e100
    validation[0]["features"]["values"][COUNT_FEATURES[0]] = 10**12
    validation[0]["features"]["values"]["question_form"] = "unknown"
    validation_batch = validate_observations(validation, expected_role="validation")
    preprocessor = mechanics._LeakageSafePreprocessor().fit(train_batch.X, train_batch.y)
    before = (
        preprocessor.count_medians_.copy(),
        preprocessor.continuous_medians_.copy(),
        preprocessor.continuous_means_.copy(),
        preprocessor.continuous_scales_.copy(),
    )

    preprocessor.transform(validation_batch.X)
    after = (
        preprocessor.count_medians_,
        preprocessor.continuous_medians_,
        preprocessor.continuous_means_,
        preprocessor.continuous_scales_,
    )

    for old, new in zip(before, after):
        assert np.array_equal(old, new)
    assert preprocessor.continuous_means_[0] < 1e50


def test_validation_values_are_not_used_when_fitting_fold_pipeline(synthetic_train):
    batch = validate_observations(synthetic_train, expected_role="train")
    plans = build_grouped_fold_assignments(batch)
    assignments = plans[0]["paper_to_validation_fold"]
    validation_mask = np.asarray(
        [assignments[paper] == 0 for paper in batch.paper_ids], dtype=bool
    )
    training_mask = ~validation_mask
    sentinel_X = batch.X.copy()
    confidence_column = len(COUNT_FEATURES) + CONTINUOUS_FEATURES.index(
        "reader_confidence"
    )
    reader_status_column = RAW_FEATURE_ORDER.index("reader_status")
    available_validation = validation_mask & (
        sentinel_X[:, reader_status_column] == "available"
    )
    assert available_validation.any()
    sentinel_X[available_validation, confidence_column] = 1e100
    pipeline = mechanics._build_calibrator_pipeline(LOGISTIC_C_GRID[0])
    pipeline.fit(sentinel_X[training_mask], batch.y[training_mask])
    fitted_mean = pipeline.named_steps["preprocess"].continuous_means_[0]
    polluted = mechanics._LeakageSafePreprocessor().fit(sentinel_X, batch.y)

    assert fitted_mean < 1e50
    assert polluted.continuous_means_[0] > 1e90


def test_actual_cv_coordinator_fits_all_400_pipelines_on_training_rows_only(
    monkeypatch, synthetic_train
):
    rows = copy.deepcopy(synthetic_train)
    for index, row in enumerate(rows):
        row["features"]["values"]["question_length_tokens"] = index + 1
        if row["features"]["values"]["reader_status"] == "available":
            row["features"]["values"]["reader_confidence"] = 10_000.0 + index
    batch = validate_observations(rows, expected_role="train")
    expectations = _fold_expectations(batch)
    records = []
    original_factory = mechanics._build_calibrator_pipeline
    monkeypatch.setattr(
        mechanics,
        "_build_calibrator_pipeline",
        _tracking_pipeline_factory(original_factory, records),
    )

    run_grouped_cv(rows)

    assert len(records) == CV_FOLDS * CV_REPEATS * len(LOGISTIC_C_GRID) == 400
    assert len({record["pipeline_id"] for record in records}) == 400
    assert len({record["preprocessor_id"] for record in records}) == 400
    marker_column = RAW_FEATURE_ORDER.index("question_length_tokens")
    for record, (training_mask, validation_mask, expected_c) in zip(
        records, expectations
    ):
        expected_X = batch.X[training_mask]
        expected_y = batch.y[training_mask]
        validation_markers = set(
            int(value) for value in batch.X[validation_mask, marker_column]
        )
        fit_markers = set(int(value) for value in record["X"][:, marker_column])
        assert record["c"] == expected_c
        assert np.array_equal(record["X"], expected_X)
        assert np.array_equal(record["y"], expected_y)
        assert not (validation_markers & fit_markers)
        expected_state = _independent_numeric_state(expected_X)
        for actual, expected in zip(
            (
                record["count_medians"],
                record["continuous_medians"],
                record["continuous_means"],
                record["continuous_scales"],
            ),
            expected_state,
        ):
            assert np.array_equal(actual, expected)
        assert record["feature_names"] == tuple(
            mechanics._transformed_feature_names()
        )


def test_coordinator_leakage_oracle_rejects_an_intentionally_global_fit(
    monkeypatch, synthetic_train
):
    rows = copy.deepcopy(synthetic_train)
    for index, row in enumerate(rows):
        row["features"]["values"]["question_length_tokens"] = index + 1
    batch = validate_observations(rows, expected_role="train")
    expected_training_mask = _fold_expectations(batch)[0][0]
    expected_markers = set(
        int(value)
        for value in batch.X[
            expected_training_mask, RAW_FEATURE_ORDER.index("question_length_tokens")
        ]
    )
    original_factory = mechanics._build_calibrator_pipeline

    def intentionally_leaking_factory(c_value):
        pipeline = original_factory(c_value)
        original_fit = pipeline.fit

        def leaked_fit(X, y, **fit_params):
            actual_markers = set(
                int(value)
                for value in batch.X[
                    :, RAW_FEATURE_ORDER.index("question_length_tokens")
                ]
            )
            assert actual_markers == expected_markers, (
                "validation-row sentinel reached fit through global preprocessing"
            )
            return original_fit(batch.X, batch.y, **fit_params)

        pipeline.fit = leaked_fit
        return pipeline

    monkeypatch.setattr(
        mechanics, "_build_calibrator_pipeline", intentionally_leaking_factory
    )

    with pytest.raises(AssertionError, match="validation-row sentinel reached fit"):
        run_grouped_cv(rows)


def test_fold_construction_is_deterministic_complete_and_paper_isolated(
    synthetic_train,
):
    batch = validate_observations(synthetic_train, expected_role="train")
    first = build_grouped_fold_assignments(batch)
    second = build_grouped_fold_assignments(batch)

    assert first == second
    assert len(first) == CV_REPEATS
    papers = set(batch.paper_ids)
    for plan in first:
        assignments = plan["paper_to_validation_fold"]
        assert set(assignments) == papers
        assert set(assignments.values()) == set(range(CV_FOLDS))
        for fold in range(CV_FOLDS):
            validation_papers = {
                paper for paper, assigned in assignments.items() if assigned == fold
            }
            training_papers = papers - validation_papers
            assert validation_papers
            assert not (training_papers & validation_papers)


def test_frozen_assignment_with_insufficient_strata_fails_closed():
    rows = _observations("train", papers=5)
    for row in rows:
        row["label"]["answerable"] = True
    batch = validate_observations(rows, expected_role="train")

    with pytest.raises(ValueError, match="lacks both answerability classes"):
        build_grouped_fold_assignments(batch)


def test_exact_tie_selects_smaller_c():
    scores = {c_value: [0.25] * (CV_FOLDS * CV_REPEATS) for c_value in LOGISTIC_C_GRID}

    selected, means = select_c_from_fold_scores(scores)

    assert selected == min(LOGISTIC_C_GRID)
    assert len(means) == len(LOGISTIC_C_GRID)
    with pytest.raises(ValueError, match="without coercion"):
        select_c_from_fold_scores(
            {str(key): value for key, value in scores.items()}
        )
    with pytest.raises(ValueError, match="C must be one of"):
        mechanics._build_calibrator_pipeline("0.1")


def test_grouped_cv_runs_all_100_folds_and_is_reproducible(synthetic_train):
    first = run_grouped_cv(synthetic_train)
    second = run_grouped_cv(list(reversed(synthetic_train)))

    assert first.selected_c in LOGISTIC_C_GRID
    assert first.mean_aurc_by_c == second.mean_aurc_by_c
    assert first.assignments_fingerprint == second.assignments_fingerprint
    assert all(len(scores) == CV_FOLDS * CV_REPEATS for _, scores in first.fold_aurc_by_c)


def test_artifact_records_frozen_configuration_and_fitted_state(fitted_artifact):
    validate_artifact(fitted_artifact)
    manifest = fitted_artifact.manifest

    assert manifest["schema"] == ARTIFACT_SCHEMA
    assert manifest["training_partition_role"] == "train"
    assert manifest["selected_c"] in LOGISTIC_C_GRID
    assert manifest["configuration_fingerprint"] == mechanics_configuration_fingerprint()
    assert manifest["fitted_state_fingerprint"] == stable_fingerprint(
        fitted_artifact.fitted_state
    )
    assert manifest["artifact_fingerprint"] == stable_fingerprint(
        {key: value for key, value in manifest.items() if key != "artifact_fingerprint"}
    )


def test_two_independent_fits_have_identical_artifacts_and_predictions(
    synthetic_train, fitted_artifact
):
    second = fit_synthetic_calibrator(copy.deepcopy(synthetic_train))
    validation = _observations("validation", prefix="reproducibility")

    assert second.manifest == fitted_artifact.manifest
    assert second.fitted_state == fitted_artifact.fitted_state
    assert predict_synthetic_scores(
        second, validation, expected_role="validation"
    ) == predict_synthetic_scores(
        fitted_artifact, validation, expected_role="validation"
    )


def test_external_validation_labels_do_not_change_predictions_and_cannot_be_fit(
    fitted_artifact,
):
    validation = _observations("validation", prefix="heldout")
    before = predict_synthetic_scores(
        fitted_artifact, validation, expected_role="validation"
    )
    permuted = copy.deepcopy(validation)
    labels = [row["label"]["answerable"] for row in reversed(permuted)]
    assert labels != [row["label"]["answerable"] for row in permuted]
    for row, label in zip(permuted, labels):
        row["label"]["answerable"] = label
    after = predict_synthetic_scores(
        fitted_artifact, permuted, expected_role="validation"
    )

    assert before == after
    with pytest.raises(ValueError, match="inconsistent"):
        fit_synthetic_calibrator(validation)


def test_configuration_and_training_changes_change_covered_fingerprints(
    synthetic_train, fitted_artifact
):
    changed_configuration = mechanics_configuration()
    changed_configuration["schemas"]["features"] = "incompatible-v2"
    assert stable_fingerprint(changed_configuration) != mechanics_configuration_fingerprint()

    changed_train = copy.deepcopy(synthetic_train)
    changed_train[0]["features"]["values"][CONTINUOUS_FEATURES[0]] += 1.0
    changed_digest = validate_observations(
        changed_train, expected_role="train"
    ).content_digest
    assert changed_digest != fitted_artifact.manifest["training_data_digest"]
    changed_manifest = {
        key: value
        for key, value in fitted_artifact.manifest.items()
        if key != "artifact_fingerprint"
    }
    changed_manifest["training_data_digest"] = changed_digest
    assert stable_fingerprint(changed_manifest) != fitted_artifact.manifest[
        "artifact_fingerprint"
    ]


def test_json_artifact_round_trip_and_incompatible_load_rejection(
    tmp_path, fitted_artifact
):
    path = tmp_path / "synthetic-artifact.json"
    save_synthetic_artifact(path, fitted_artifact)
    restored = load_synthetic_artifact(path)
    explicitly_restored = load_synthetic_artifact(
        path,
        expected_configuration_fingerprint=mechanics_configuration_fingerprint(),
    )
    validation = _observations("validation", prefix="roundtrip")

    original_scores = predict_synthetic_scores(
        fitted_artifact, validation, expected_role="validation"
    )
    restored_scores = predict_synthetic_scores(
        restored, validation, expected_role="validation"
    )
    assert original_scores.keys() == restored_scores.keys()
    assert np.allclose(list(original_scores.values()), list(restored_scores.values()))
    assert explicitly_restored.manifest == restored.manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_synthetic_artifact(path, fitted_artifact)
    with pytest.raises(ValueError, match="configuration fingerprint"):
        load_synthetic_artifact(
            path, expected_configuration_fingerprint="0" * 64
        )
    with pytest.raises(ValueError, match="expected configuration fingerprint is malformed"):
        load_synthetic_artifact(
            path, expected_configuration_fingerprint="not-a-sha256"
        )

    payload = json.loads(path.read_text())
    payload["manifest"]["schema"] = "incompatible"
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema is incompatible"):
        load_synthetic_artifact(incompatible)


def test_configuration_fingerprint_override_attack_is_rejected(
    tmp_path, fitted_artifact
):
    attacker_fingerprint = "a" * 64
    payload = {
        "manifest": copy.deepcopy(fitted_artifact.manifest),
        "fitted_state": copy.deepcopy(fitted_artifact.fitted_state),
    }
    payload["manifest"]["configuration_fingerprint"] = attacker_fingerprint
    without_outer = {
        key: value
        for key, value in payload["manifest"].items()
        if key != "artifact_fingerprint"
    }
    payload["manifest"]["artifact_fingerprint"] = stable_fingerprint(without_outer)
    path = tmp_path / "configuration-override-attack.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="declared configuration fingerprint"):
        load_synthetic_artifact(
            path, expected_configuration_fingerprint=attacker_fingerprint
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_artifact_json_rejects_nonstandard_numeric_constants(tmp_path, constant):
    path = tmp_path / f"nonstandard-{constant.replace('-', 'minus')}.json"
    path.write_text(f'{{"manifest": {constant}, "fitted_state": {{}}}}')

    with pytest.raises(ValueError, match="nonstandard JSON numeric constant"):
        load_synthetic_artifact(path)


def test_artifact_json_rejects_duplicate_object_keys(tmp_path):
    path = tmp_path / "duplicate-key.json"
    path.write_text('{"manifest": {}, "manifest": {}, "fitted_state": {}}')

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_synthetic_artifact(path)


@pytest.mark.parametrize("root", ["[]", "null", '"string"'])
def test_artifact_json_rejects_invalid_root_types(tmp_path, root):
    path = tmp_path / "invalid-root.json"
    path.write_text(root)

    with pytest.raises(ValueError, match="payload must be a mapping"):
        load_synthetic_artifact(path)


def test_artifact_load_rejects_modified_fitted_state(tmp_path, fitted_artifact):
    path = tmp_path / "synthetic-artifact.json"
    save_synthetic_artifact(path, fitted_artifact)
    payload = json.loads(path.read_text())
    payload["fitted_state"]["estimator"]["coef"][0][0] += 1.0
    modified = tmp_path / "modified-state.json"
    modified.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fitted state fingerprint"):
        load_synthetic_artifact(modified)


def test_artifact_rejects_training_digest_and_fold_metadata_tampering(
    tmp_path, fitted_artifact
):
    base = {
        "manifest": copy.deepcopy(fitted_artifact.manifest),
        "fitted_state": copy.deepcopy(fitted_artifact.fitted_state),
    }
    training_tamper = copy.deepcopy(base)
    training_tamper["manifest"]["training_data_digest"] = "0" * 64
    training_path = tmp_path / "training-digest-tamper.json"
    training_path.write_text(json.dumps(training_tamper))
    with pytest.raises(ValueError, match="manifest fingerprint"):
        load_synthetic_artifact(training_path)

    fold_tamper = copy.deepcopy(base)
    assignment = fold_tamper["manifest"]["cv"]["assignments"][0][
        "paper_to_validation_fold"
    ]
    paper = next(iter(assignment))
    assignment[paper] = (assignment[paper] + 1) % CV_FOLDS
    fold_path = tmp_path / "fold-tamper.json"
    fold_path.write_text(json.dumps(fold_tamper))
    with pytest.raises(ValueError, match="assignment fingerprint"):
        load_synthetic_artifact(fold_path)


def test_artifact_rejects_noncanonical_fold_id_even_with_recomputed_hashes(
    tmp_path, fitted_artifact
):
    payload = {
        "manifest": copy.deepcopy(fitted_artifact.manifest),
        "fitted_state": copy.deepcopy(fitted_artifact.fitted_state),
    }
    assignments = payload["manifest"]["cv"]["assignments"]
    original_paper = next(iter(assignments[0]["paper_to_validation_fold"]))
    for plan in assignments:
        mapping = plan["paper_to_validation_fold"]
        mapping[f"{original_paper} "] = mapping.pop(original_paper)
    payload["manifest"]["cv"]["assignments_fingerprint"] = stable_fingerprint(
        assignments
    )
    without_outer = {
        key: value
        for key, value in payload["manifest"].items()
        if key != "artifact_fingerprint"
    }
    payload["manifest"]["artifact_fingerprint"] = stable_fingerprint(without_outer)
    path = tmp_path / "noncanonical-fold-id.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        load_synthetic_artifact(path)


@pytest.mark.parametrize(
    ("field", "value_factory", "match"),
    [
        (
            "training_papers",
            lambda manifest: manifest["training_papers"] + 1,
            "training paper count disagrees",
        ),
        (
            "training_rows",
            lambda manifest: manifest["training_papers"] - 1,
            "training row count is smaller",
        ),
    ],
)
def test_artifact_rejects_training_counts_contradicting_assignments(
    tmp_path, fitted_artifact, field, value_factory, match
):
    payload = {
        "manifest": copy.deepcopy(fitted_artifact.manifest),
        "fitted_state": copy.deepcopy(fitted_artifact.fitted_state),
    }
    payload["manifest"][field] = value_factory(payload["manifest"])
    without_outer = {
        key: value
        for key, value in payload["manifest"].items()
        if key != "artifact_fingerprint"
    }
    payload["manifest"]["artifact_fingerprint"] = stable_fingerprint(without_outer)
    path = tmp_path / f"{field}-contradiction.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        load_synthetic_artifact(path)


def test_artifact_rejects_inconsistent_cv_means_even_with_new_outer_hash(
    tmp_path, fitted_artifact
):
    payload = {
        "manifest": copy.deepcopy(fitted_artifact.manifest),
        "fitted_state": copy.deepcopy(fitted_artifact.fitted_state),
    }
    selected_c = payload["manifest"]["selected_c"]
    alternative = next(c_value for c_value in LOGISTIC_C_GRID if c_value != selected_c)
    for row in payload["manifest"]["cv"]["mean_aurc_by_c"]:
        row[1] = 0.0 if row[0] == alternative else 1.0
    without_outer = {
        key: value
        for key, value in payload["manifest"].items()
        if key != "artifact_fingerprint"
    }
    payload["manifest"]["artifact_fingerprint"] = stable_fingerprint(without_outer)
    path = tmp_path / "cv-mean-tamper.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="selected C disagrees"):
        load_synthetic_artifact(path)


def test_changed_implementation_identity_invalidates_artifact(
    tmp_path, fitted_artifact
):
    payload = {
        "manifest": copy.deepcopy(fitted_artifact.manifest),
        "fitted_state": copy.deepcopy(fitted_artifact.fitted_state),
    }
    configuration = payload["manifest"]["configuration"]
    configuration["implementation"]["calibrator_mechanics_sha256"] = "f" * 64
    payload["manifest"]["configuration_fingerprint"] = stable_fingerprint(
        configuration
    )
    without_outer = {
        key: value
        for key, value in payload["manifest"].items()
        if key != "artifact_fingerprint"
    }
    payload["manifest"]["artifact_fingerprint"] = stable_fingerprint(without_outer)
    path = tmp_path / "implementation-identity-tamper.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="configuration fingerprint is incompatible"):
        load_synthetic_artifact(path)


@pytest.mark.parametrize("method", ["matched_coverage", "risk_target"])
def test_threshold_selection_accepts_synthetic_validation_only(
    method, fitted_artifact
):
    validation = _observations("validation", prefix=f"threshold-{method}")

    threshold = select_synthetic_threshold(
        validation,
        method=method,
        artifact=fitted_artifact,
    )

    assert threshold["partition_role"] == "validation"
    assert threshold["method"] == method
    assert threshold["threshold_fingerprint"] == stable_fingerprint(
        {
            key: value
            for key, value in threshold.items()
            if key != "threshold_fingerprint"
        }
    )


def test_synthetic_score_output_is_keyed_by_exact_validation_identities(
    fitted_artifact,
):
    validation = _observations("validation", prefix="score-identity")
    scores = predict_synthetic_scores(
        fitted_artifact, validation, expected_role="validation"
    )

    assert set(scores) == {row["question_id"] for row in validation}
    assert all(np.isfinite(score) for score in scores.values())


def test_module_has_no_dataset_loader_cache_default_or_test_scoring_entrypoint():
    source = inspect.getsource(mechanics)

    assert "data/raw" not in source
    assert "results/" not in source
    assert "load_qasper" not in source
    assert not hasattr(mechanics, "score_test")
