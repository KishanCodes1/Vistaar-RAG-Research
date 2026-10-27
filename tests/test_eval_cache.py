"""Evaluation cache isolation and fingerprint invalidation tests."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from eval.provenance import EvaluationCache, stable_fingerprint
from eval.qasper import EvalItem
from eval.run_eval import build_run_payload
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

    monkeypatch.setattr("eval.run_eval.SYSTEM_PROMPT", "changed prompt")
    assert stable_fingerprint(payload()) != original
