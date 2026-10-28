"""Evaluation cache isolation and fingerprint invalidation tests."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from eval.provenance import EvaluationCache, stable_fingerprint
from eval.qasper import EvalItem
from eval.run_eval import DEFAULT_CACHE_ROOT, build_run_payload, pending_items
from eval.shadow import (
    SHADOW_CACHE_ROOT,
    SHADOW_RUN_SCHEMA,
    diagnostic_record_is_complete,
)
from eval.shadow_protocol import frozen_protocol_metadata
from vistaar.config import Settings
from vistaar.diagnostics import CONFIDENCE_DIAGNOSTICS_SCHEMA
from vistaar.pipeline import RetrievalConfig


@pytest.fixture
def run_payload() -> dict:
    return {
        "dataset": {
            "name": "qasper-validation",
            "sha256": "dataset-a",
            "selection": {"papers": 8, "negative_enriched": False},
        },
        "documents": [{"doc_id": "doc-1", "sha256": "content-a"}],
        "chunking": {"size": 800, "overlap": 150, "code": "chunker-a"},
        "embedding": {"model": "embed", "revision": "rev-a", "normalise": True},
        "retrieval": [{"name": "E", "top_k": 8, "rerank_top_n": 4}],
        "reader": {"model": "reader", "revision": "rev-r", "threshold": 0.5},
        "prompt_sha256": "prompt-a",
        "implementation": {"pipeline.py": "source-a"},
    }


def test_fingerprint_is_stable_for_identical_canonical_content(run_payload):
    reordered = dict(reversed(list(run_payload.items())))
    assert stable_fingerprint(run_payload) == stable_fingerprint(reordered)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("dataset", "sha256", "dataset-b"),
        ("chunking", "size", 900),
        ("embedding", "revision", "rev-b"),
        ("reader", "threshold", 0.2),
        ("implementation", "pipeline.py", "source-b"),
    ],
)
def test_relevant_input_change_invalidates_run_fingerprint(
    run_payload, section, key, value
):
    changed = copy.deepcopy(run_payload)
    changed[section][key] = value
    assert stable_fingerprint(changed) != stable_fingerprint(run_payload)


def test_document_or_retrieval_change_invalidates_run_fingerprint(run_payload):
    document_changed = copy.deepcopy(run_payload)
    document_changed["documents"][0]["sha256"] = "content-b"
    retrieval_changed = copy.deepcopy(run_payload)
    retrieval_changed["retrieval"][0]["top_k"] = 12

    original = stable_fingerprint(run_payload)
    assert stable_fingerprint(document_changed) != original
    assert stable_fingerprint(retrieval_changed) != original


def test_identical_inputs_reuse_cached_record(tmp_path, run_payload):
    run_fp = stable_fingerprint(run_payload)
    cache = EvaluationCache(tmp_path, run_fp)
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q", "document": "a"})
    record = {"config": "E", "question_id": "q1", "prediction": "answer"}
    cached = cache.append("E", config_fp, item_fp, record)

    loaded = cache.load("E", config_fp, {"q1": item_fp})

    assert loaded.accepted == 1
    assert loaded.rejected == 0
    assert loaded.records["q1"] == cached


def test_changed_item_or_config_invalidates_cached_record(tmp_path, run_payload):
    run_fp = stable_fingerprint(run_payload)
    cache = EvaluationCache(tmp_path, run_fp)
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q", "document": "a"})
    cache.append("E", config_fp, item_fp, {"config": "E", "question_id": "q1"})

    changed_item = cache.load(
        "E", config_fp, {"q1": stable_fingerprint({"question": "changed"})}
    )
    changed_config = cache.load(
        "E", stable_fingerprint({"config": "D"}), {"q1": item_fp}
    )

    assert changed_item.records == {}
    assert changed_item.rejected == 1
    assert changed_config.records == {}


def test_different_runs_have_isolated_paths_and_reject_mixed_rows(tmp_path, run_payload):
    first_fp = stable_fingerprint(run_payload)
    changed = copy.deepcopy(run_payload)
    changed["dataset"]["selection"]["papers"] = 40
    second_fp = stable_fingerprint(changed)
    first = EvaluationCache(tmp_path, first_fp)
    second = EvaluationCache(tmp_path, second_fp)
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q1"})
    first.append("E", config_fp, item_fp, {"config": "E", "question_id": "q1"})

    assert first.path_for("E", config_fp) != second.path_for("E", config_fp)
    assert second.load("E", config_fp, {"q1": item_fp}).records == {}

    mixed_path = second.path_for("E", config_fp)
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_path.write_text(first.path_for("E", config_fp).read_text())
    mixed = second.load("E", config_fp, {"q1": item_fp})
    assert mixed.records == {}
    assert mixed.rejected == 1


def test_malformed_cache_rows_are_ignored(tmp_path, run_payload):
    run_fp = stable_fingerprint(run_payload)
    cache = EvaluationCache(tmp_path, run_fp)
    config_fp = stable_fingerprint({"config": "E"})
    path = cache.path_for("E", config_fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json\n" + json.dumps({"question_id": "q1"}) + "\n")

    loaded = cache.load("E", config_fp, {"q1": "item"})

    assert loaded.records == {}
    assert loaded.malformed == 2


def test_real_run_payload_fingerprints_evaluation_inputs(tmp_path, monkeypatch):
    dataset = tmp_path / "validation.parquet"
    dataset.write_bytes(b"fixed dataset bytes")
    item = EvalItem(
        doc_id="paper-1",
        question_id="question-1",
        question="What is reported?",
        doc_text="The paper reports a deterministic result.",
        doc_title="Paper",
        gold_answers=["a deterministic result"],
        gold_evidence=["The paper reports a deterministic result."],
        answerable=True,
    )
    settings = Settings(api_key="")
    config = RetrievalConfig(name="E")

    def payload(**overrides):
        values = {
            "dataset_path": dataset,
            "items": [item],
            "settings": settings,
            "configs": [config],
            "papers": 1,
            "limit": 0,
            "negative_enriched": False,
            "workers": 2,
            "dataset_partition": "validation",
        }
        values.update(overrides)
        return build_run_payload(**values)

    original_payload = payload()
    original = stable_fingerprint(original_payload)
    assert original_payload["diagnostics"] == {
        "schema_version": CONFIDENCE_DIAGNOSTICS_SCHEMA,
        "enabled_for_records": True,
        "inference_time_only": True,
    }
    assert "vistaar/diagnostics.py" in original_payload["implementation"]
    assert "eval/shadow.py" in original_payload["implementation"]
    assert "eval/shadow_protocol.py" in original_payload["implementation"]
    changed_item = replace(item, doc_text="Changed document content.")
    changed_chunking = replace(settings, chunk_size=900)
    changed_model = replace(settings, embedding_revision="different-revision")
    changed_reader = replace(settings, reader_threshold=0.2)
    changed_retrieval = replace(config, top_k=12)

    assert stable_fingerprint(payload(items=[changed_item])) != original
    assert stable_fingerprint(payload(settings=changed_chunking)) != original
    assert stable_fingerprint(payload(settings=changed_model)) != original
    assert stable_fingerprint(payload(settings=changed_reader)) != original
    assert stable_fingerprint(payload(configs=[changed_retrieval])) != original
    assert stable_fingerprint(payload(negative_enriched=True)) != original
    assert stable_fingerprint(payload(workers=3)) != original
    changed_diagnostics = copy.deepcopy(original_payload)
    changed_diagnostics["diagnostics"]["schema_version"] = "changed-schema"
    assert stable_fingerprint(changed_diagnostics) != original

    monkeypatch.setattr("eval.run_eval.SYSTEM_PROMPT", "changed prompt")
    assert stable_fingerprint(payload()) != original


def _diagnostic_record() -> dict:
    stages = {
        name: {
            "requested": name in {"dense", "selected_context"},
            "available": name in {"dense", "selected_context"},
            "reason": None if name in {"dense", "selected_context"} else "not_requested",
            "candidates": [] if name in {"dense", "selected_context"} else None,
            "score_summary": None,
        }
        for name in ("dense", "bm25", "rrf", "reranker", "selected_context")
    }
    return {
        "config": "E",
        "question_id": "q1",
        "diagnostics": {
            "schema_version": CONFIDENCE_DIAGNOSTICS_SCHEMA,
            "retrieval": {
                "namespace": "paper",
                "config": {},
                "limits": {},
                "stages": stages,
                "features": {"selected_chunk_count": 0},
            },
            "reader": {"available": False, "reason": "no_retrieved_passages"},
            "decision": {
                "backend": "local",
                "abstention_enabled": True,
                "configured_threshold": 0.5,
                "found": False,
                "confidence": 0.0,
                "abstained": True,
                "triggered_reasons": ["no_retrieved_passages"],
            },
        },
    }


def _available_diagnostic_record() -> dict:
    record = _diagnostic_record()
    candidate = {
        "rank": 1,
        "chunk_id": "paper:0001",
        "doc_id": "paper",
        "page": 1,
        "section": "page 1",
        "source": "dense",
        "score": 0.75,
        "dense_rank": 1,
        "bm25_rank": None,
        "fused_rank": None,
    }
    summary = {
        "count": 1,
        "top_score": 0.75,
        "top1_top2_margin": None,
        "mean": 0.75,
        "std": 0.0,
        "min": 0.75,
        "max": 0.75,
        "range": 0.0,
    }
    for name in ("dense", "selected_context"):
        record["diagnostics"]["retrieval"]["stages"][name]["candidates"] = [
            dict(candidate)
        ]
        record["diagnostics"]["retrieval"]["stages"][name]["score_summary"] = dict(
            summary
        )
    record["diagnostics"]["reader"] = {
        "available": True,
        "found": True,
        "passage_count": 1,
        "context_characters": 100,
        "window_count": 1,
        "candidate_start_count": 2,
        "candidate_span_count": 3,
        "best_span_score": 5.0,
        "second_best_span_score": 4.0,
        "selected_window": 0,
        "selected_start_token": 3,
        "selected_end_token": 4,
        "selected_null_score": 2.0,
        "null_score_min": 2.0,
        "null_score_max": 2.0,
        "span_null_margin": 3.0,
        "selected_passage": 1,
        "no_span_reason": None,
    }
    record["diagnostics"]["decision"].update(
        {"found": True, "confidence": 0.95, "abstained": False, "triggered_reasons": []}
    )
    return record


def _two_candidate_diagnostic_record() -> dict:
    record = _available_diagnostic_record()
    for name in ("dense", "selected_context"):
        stage = record["diagnostics"]["retrieval"]["stages"][name]
        second = dict(stage["candidates"][0])
        second.update({"rank": 2, "chunk_id": "paper:0002", "score": 0.5})
        stage["candidates"].append(second)
        stage["score_summary"] = {
            "count": 2,
            "top_score": 0.75,
            "top1_top2_margin": 0.25,
            "mean": 0.625,
            "std": 0.125,
            "min": 0.5,
            "max": 0.75,
            "range": 0.25,
        }
    return record


def _load_with_shadow_validator(tmp_path, record: dict):
    run_fp = stable_fingerprint({"workflow": "shadow"})
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q1"})
    EvaluationCache(tmp_path, run_fp).append("E", config_fp, item_fp, record)
    shadow_cache = EvaluationCache(
        tmp_path,
        run_fp,
        record_validator=diagnostic_record_is_complete,
    )
    return shadow_cache.load("E", config_fp, {"q1": item_fp})


def test_shadow_cache_rejects_diagnostic_free_and_wrong_schema_rows(tmp_path):
    run_fp = stable_fingerprint({"workflow": "shadow"})
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q1"})
    ordinary_cache = EvaluationCache(tmp_path, run_fp)
    ordinary_cache.append(
        "E",
        config_fp,
        item_fp,
        {"config": "E", "question_id": "q1", "prediction": "legacy"},
    )
    shadow_cache = EvaluationCache(
        tmp_path,
        run_fp,
        record_validator=diagnostic_record_is_complete,
    )

    loaded = shadow_cache.load("E", config_fp, {"q1": item_fp})

    assert loaded.records == {}
    assert loaded.rejected == 1
    wrong_schema = _diagnostic_record()
    wrong_schema["diagnostics"]["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="schema contract"):
        shadow_cache.append("E", config_fp, item_fp, wrong_schema)
    incomplete_reader = _diagnostic_record()
    incomplete_reader["diagnostics"]["reader"] = {
        "available": True,
        "found": True,
    }
    with pytest.raises(ValueError, match="schema contract"):
        shadow_cache.append("E", config_fp, item_fp, incomplete_reader)


def test_shadow_cache_accepts_explicit_unavailable_values(tmp_path):
    run_fp = stable_fingerprint({"workflow": "shadow"})
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q1"})
    cache = EvaluationCache(
        tmp_path,
        run_fp,
        record_validator=diagnostic_record_is_complete,
    )
    record = _diagnostic_record()

    cached = cache.append("E", config_fp, item_fp, record)
    loaded = cache.load("E", config_fp, {"q1": item_fp})

    assert loaded.accepted == 1
    assert loaded.records["q1"] == cached
    assert loaded.records["q1"]["diagnostics"]["reader"]["available"] is False
    assert "candidate_start_count" not in loaded.records["q1"]["diagnostics"]["reader"]


def test_shadow_cache_accepts_complete_available_diagnostics(tmp_path):
    run_fp = stable_fingerprint({"workflow": "shadow"})
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q1"})
    cache = EvaluationCache(
        tmp_path,
        run_fp,
        record_validator=diagnostic_record_is_complete,
    )
    record = _available_diagnostic_record()

    cached = cache.append("E", config_fp, item_fp, record)

    assert cache.load("E", config_fp, {"q1": item_fp}).records["q1"] == cached


def test_shadow_cache_rejects_unavailable_stage_with_score_summary(tmp_path):
    record = _diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["bm25"]["score_summary"] = {
        "count": 1
    }

    assert diagnostic_record_is_complete(record) is False


def test_shadow_cache_load_fails_closed_on_contradictory_cached_row(tmp_path):
    run_fp = stable_fingerprint({"workflow": "shadow"})
    config_fp = stable_fingerprint({"config": "E"})
    item_fp = stable_fingerprint({"question": "q1"})
    record = _diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["bm25"]["score_summary"] = {
        "count": 1
    }
    EvaluationCache(tmp_path, run_fp).append("E", config_fp, item_fp, record)
    shadow_cache = EvaluationCache(
        tmp_path,
        run_fp,
        record_validator=diagnostic_record_is_complete,
    )

    loaded = shadow_cache.load("E", config_fp, {"q1": item_fp})

    assert loaded.records == {}
    assert loaded.rejected == 1


def test_shadow_cache_rejects_invalid_candidate_and_score_summary_shapes():
    missing_candidate_field = _available_diagnostic_record()
    del missing_candidate_field["diagnostics"]["retrieval"]["stages"]["dense"][
        "candidates"
    ][0]["chunk_id"]
    wrong_summary_count = _available_diagnostic_record()
    wrong_summary_count["diagnostics"]["retrieval"]["stages"]["dense"][
        "score_summary"
    ]["count"] = 2
    missing_summary_field = _available_diagnostic_record()
    del missing_summary_field["diagnostics"]["retrieval"]["stages"]["dense"][
        "score_summary"
    ]["std"]

    assert diagnostic_record_is_complete(missing_candidate_field) is False
    assert diagnostic_record_is_complete(wrong_summary_count) is False
    assert diagnostic_record_is_complete(missing_summary_field) is False


@pytest.mark.parametrize("rank", [True, "1", 1.0, 0, -1])
def test_shadow_cache_rejects_non_integer_or_out_of_range_candidate_rank(rank):
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["candidates"][0][
        "rank"
    ] = rank

    assert diagnostic_record_is_complete(record) is False


def test_shadow_cache_accepts_sequential_positive_candidate_rank_boundaries():
    record = _two_candidate_diagnostic_record()
    stage = record["diagnostics"]["retrieval"]["stages"]["dense"]

    assert [candidate["rank"] for candidate in stage["candidates"]] == [1, 2]
    assert diagnostic_record_is_complete(record) is True


@pytest.mark.parametrize("page", [True, "1", 1.0, 0, -1, -7])
def test_shadow_cache_rejects_non_integer_or_nonpositive_candidate_page(page):
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["candidates"][0][
        "page"
    ] = page

    assert diagnostic_record_is_complete(record) is False


@pytest.mark.parametrize("page", [1, 999])
def test_shadow_cache_accepts_positive_one_based_candidate_pages(page):
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["candidates"][0][
        "page"
    ] = page

    assert diagnostic_record_is_complete(record) is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_score", -1.0),
        ("mean", 0.5),
        ("std", 0.25),
        ("range", 1.0),
        ("count", 2),
    ],
)
def test_shadow_cache_rejects_summary_values_inconsistent_with_candidates(
    field, value
):
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["score_summary"][
        field
    ] = value

    assert diagnostic_record_is_complete(record) is False


def test_shadow_cache_rejects_invalid_summary_margin_semantics():
    singleton = _available_diagnostic_record()
    singleton["diagnostics"]["retrieval"]["stages"]["dense"]["score_summary"][
        "top1_top2_margin"
    ] = 0.0
    missing_pair_margin = _two_candidate_diagnostic_record()
    missing_pair_margin["diagnostics"]["retrieval"]["stages"]["dense"][
        "score_summary"
    ]["top1_top2_margin"] = None
    wrong_pair_margin = _two_candidate_diagnostic_record()
    wrong_pair_margin["diagnostics"]["retrieval"]["stages"]["dense"][
        "score_summary"
    ]["top1_top2_margin"] = 999.0

    assert diagnostic_record_is_complete(singleton) is False
    assert diagnostic_record_is_complete(missing_pair_margin) is False
    assert diagnostic_record_is_complete(wrong_pair_margin) is False


def test_shadow_cache_rejects_summary_with_min_greater_than_max():
    record = _available_diagnostic_record()
    summary = record["diagnostics"]["retrieval"]["stages"]["dense"][
        "score_summary"
    ]
    summary.update({"min": 1.0, "max": 0.5})

    assert diagnostic_record_is_complete(record) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("top_score", float("inf")), ("mean", float("nan")), ("std", float("-inf"))],
)
def test_shadow_cache_rejects_nonfinite_summary_values(field, value):
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["score_summary"][
        field
    ] = value

    assert diagnostic_record_is_complete(record) is False


def test_shadow_cache_rejects_exact_audit_contradictory_summary():
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["score_summary"] = {
        "count": 1,
        "top_score": -999.0,
        "top1_top2_margin": 123.0,
        "mean": 42.0,
        "std": 0.0,
        "min": 100.0,
        "max": -100.0,
        "range": 0.0,
    }

    assert diagnostic_record_is_complete(record) is False


def test_corrupted_summary_cache_row_is_rejected_and_scheduled_for_rerun(tmp_path):
    record = _available_diagnostic_record()
    record["diagnostics"]["retrieval"]["stages"]["dense"]["score_summary"][
        "top_score"
    ] = -999.0
    loaded = _load_with_shadow_validator(tmp_path, record)
    item = EvalItem(
        doc_id="paper",
        question_id="q1",
        question="question",
        doc_text="document text",
        doc_title="Paper",
        gold_answers=["answer"],
        gold_evidence=["evidence"],
        answerable=True,
    )

    todo = pending_items([item], loaded.records)

    assert loaded.records == {}
    assert loaded.rejected == 1
    assert [pending.question_id for pending in todo] == ["q1"]


def test_combined_boolean_rank_negative_page_regression_rejects_and_reruns(tmp_path):
    record = _available_diagnostic_record()
    candidate = record["diagnostics"]["retrieval"]["stages"]["dense"][
        "candidates"
    ][0]
    candidate.update({"rank": True, "page": -7})

    assert diagnostic_record_is_complete(record) is False

    loaded = _load_with_shadow_validator(tmp_path, record)
    item = EvalItem(
        doc_id="paper",
        question_id="q1",
        question="question",
        doc_text="document text",
        doc_title="Paper",
        gold_answers=["answer"],
        gold_evidence=["evidence"],
        answerable=True,
    )

    assert loaded.records == {}
    assert loaded.rejected == 1
    assert [pending.question_id for pending in pending_items([item], loaded.records)] == [
        "q1"
    ]


def test_shadow_cache_rejects_fabricated_unavailable_reader_values():
    record = _diagnostic_record()
    record["diagnostics"]["reader"]["candidate_start_count"] = 99

    assert diagnostic_record_is_complete(record) is False


def test_shadow_payload_has_separate_schema_and_fingerprint(tmp_path):
    dataset = tmp_path / "validation.parquet"
    dataset.write_bytes(b"fixed dataset bytes")
    item = EvalItem(
        doc_id="paper-1",
        question_id="question-1",
        question="What is reported?",
        doc_text="The paper reports a deterministic result.",
        doc_title="Paper",
        gold_answers=["a deterministic result"],
        gold_evidence=["The paper reports a deterministic result."],
        answerable=True,
    )
    values = {
        "dataset_path": dataset,
        "items": [item],
        "settings": Settings(api_key=""),
        "configs": [RetrievalConfig(name="E")],
        "papers": 1,
        "limit": 0,
        "negative_enriched": False,
        "workers": 1,
        "dataset_partition": "validation",
    }

    ordinary = build_run_payload(**values)
    shadow = build_run_payload(**values, shadow=True)

    assert shadow["schema"] == SHADOW_RUN_SCHEMA
    assert shadow["shadow"]["diagnostics_required_for_every_record"] is True
    assert shadow["shadow"]["changes_inference_decisions"] is False
    assert shadow["shadow"]["frozen_protocol"] == frozen_protocol_metadata()
    assert shadow["dataset"] == {
        **ordinary["dataset"],
        "partition_role": "validation",
    }
    assert stable_fingerprint(shadow) != stable_fingerprint(ordinary)
    assert SHADOW_CACHE_ROOT != DEFAULT_CACHE_ROOT


def test_partition_role_is_manifested_and_fingerprinted(tmp_path):
    dataset = tmp_path / "qasper.parquet"
    dataset.write_bytes(b"fixed dataset bytes")
    item = EvalItem(
        doc_id="paper-1",
        question_id="question-1",
        question="What is reported?",
        doc_text="The paper reports a deterministic result.",
        doc_title="Paper",
        gold_answers=["a deterministic result"],
        gold_evidence=["The paper reports a deterministic result."],
        answerable=True,
    )
    values = {
        "dataset_path": dataset,
        "items": [item],
        "settings": Settings(api_key=""),
        "configs": [RetrievalConfig(name="E")],
        "papers": 1,
        "limit": 0,
        "negative_enriched": False,
        "workers": 1,
    }

    train = build_run_payload(**values, dataset_partition="train", shadow=True)
    test = build_run_payload(**values, dataset_partition="test", shadow=True)

    assert train["dataset"]["name"] == "qasper"
    assert train["dataset"]["partition_role"] == "train"
    assert test["dataset"]["partition_role"] == "test"
    assert stable_fingerprint(train) != stable_fingerprint(test)
