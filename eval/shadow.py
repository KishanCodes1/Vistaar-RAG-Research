"""Contracts for the local-only diagnostic shadow evaluation workflow."""
from __future__ import annotations

import math
from pathlib import Path

from vistaar.diagnostics import CONFIDENCE_DIAGNOSTICS_SCHEMA

from .shadow_protocol import frozen_protocol_metadata


SHADOW_RUN_SCHEMA = "vistaar-shadow-instrumentation-run-v1"
SHADOW_CACHE_ROOT = Path("results/cache/shadow-instrumentation")
SHADOW_OUTPUT_ROOT = Path("results/shadow-runs")

_LOCAL_READER_FIELDS = {
    "available",
    "found",
    "passage_count",
    "context_characters",
    "window_count",
    "candidate_start_count",
    "candidate_span_count",
    "best_span_score",
    "second_best_span_score",
    "selected_window",
    "selected_start_token",
    "selected_end_token",
    "selected_null_score",
    "null_score_min",
    "null_score_max",
    "span_null_margin",
    "selected_passage",
    "no_span_reason",
}
_DECISION_FIELDS = {
    "backend",
    "abstention_enabled",
    "configured_threshold",
    "found",
    "confidence",
    "abstained",
    "triggered_reasons",
}


def shadow_run_metadata() -> dict:
    """Return deterministic metadata that separates shadow runs and caches."""
    return {
        "schema": SHADOW_RUN_SCHEMA,
        "mode": "local-only-shadow-instrumentation",
        "diagnostics_schema_version": CONFIDENCE_DIAGNOSTICS_SCHEMA,
        "diagnostics_required_for_every_record": True,
        "changes_inference_decisions": False,
        "cache_policy": "dedicated-root-and-run-fingerprint",
        "frozen_protocol": frozen_protocol_metadata(),
    }


def path_is_within(path: str | Path, protected_root: str | Path) -> bool:
    """Return whether a resolved path equals or descends from a protected root."""
    resolved_path = Path(path).resolve()
    resolved_root = Path(protected_root).resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def validate_shadow_cache_root(
    cache_root: str | Path, ordinary_cache_root: str | Path
) -> Path:
    """Reject the ordinary cache tree, including paths entering through symlinks."""
    resolved = Path(cache_root).resolve()
    if path_is_within(resolved, ordinary_cache_root):
        raise ValueError(
            "shadow cache root must be outside the ordinary evaluation cache tree"
        )
    return resolved


def diagnostic_record_is_complete(record: dict) -> bool:
    """Fail closed unless a final record has the required diagnostic structure.

    Optional signals may still be explicitly unavailable inside the schema. This
    validator checks structure and availability metadata; it never substitutes a
    numeric value for an unavailable signal.
    """
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    if diagnostics.get("schema_version") != CONFIDENCE_DIAGNOSTICS_SCHEMA:
        return False

    retrieval = diagnostics.get("retrieval")
    reader = diagnostics.get("reader")
    decision = diagnostics.get("decision")
    if not all(isinstance(value, dict) for value in (retrieval, reader, decision)):
        return False
    if decision.get("backend") != "local":
        return False
    if not {"namespace", "config", "limits", "stages", "features"} <= set(
        retrieval
    ):
        return False
    if not isinstance(retrieval["namespace"], str) or not retrieval["namespace"]:
        return False
    if not all(
        isinstance(retrieval[field], dict)
        for field in ("config", "limits", "features")
    ):
        return False

    stages = retrieval.get("stages")
    if not isinstance(stages, dict):
        return False
    for name in ("dense", "bm25", "rrf", "reranker", "selected_context"):
        stage = stages.get(name)
        if not isinstance(stage, dict):
            return False
        if not isinstance(stage.get("requested"), bool):
            return False
        if not isinstance(stage.get("available"), bool):
            return False
        if not stage["requested"] and stage["available"]:
            return False
        if not {"reason", "candidates", "score_summary"} <= set(stage):
            return False
        if stage["reason"] is not None and not isinstance(stage["reason"], str):
            return False
        if not stage["available"] and not stage["reason"]:
            return False
        candidates = stage["candidates"]
        if stage["available"] and not isinstance(candidates, list):
            return False
        if not stage["available"] and candidates is not None:
            return False
        if not stage["available"] and stage["score_summary"] is not None:
            return False
        if candidates and not isinstance(stage["score_summary"], dict):
            return False
        if not candidates and stage["score_summary"] is not None:
            return False
        if candidates:
            if not _valid_candidates(candidates):
                return False
            if not _valid_score_summary(stage["score_summary"], candidates):
                return False

    if not isinstance(reader.get("available"), bool):
        return False
    if reader["available"]:
        if set(reader) != _LOCAL_READER_FIELDS:
            return False
        if not _valid_available_reader(reader):
            return False
    elif not isinstance(reader.get("reason"), str):
        return False
    elif set(reader) != {"available", "reason"} or not reader["reason"]:
        return False
    if not _DECISION_FIELDS <= set(decision):
        return False
    if not isinstance(decision["abstention_enabled"], bool):
        return False
    if not _is_number(decision["configured_threshold"]):
        return False
    if not isinstance(decision["found"], bool):
        return False
    if not _is_number(decision["confidence"]):
        return False
    if not isinstance(decision["abstained"], bool):
        return False
    if not _is_string_list(decision["triggered_reasons"]):
        return False
    if "counterfactual_conditions" in decision and not _is_string_list(
        decision["counterfactual_conditions"]
    ):
        return False
    return True


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_optional_number(value: object) -> bool:
    return value is None or _is_number(value)


def _is_optional_rank(value: object) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
    )


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_candidates(candidates: list) -> bool:
    required = {
        "rank",
        "chunk_id",
        "doc_id",
        "page",
        "section",
        "source",
        "score",
        "dense_rank",
        "bm25_rank",
        "fused_rank",
    }
    for expected_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or not required <= set(candidate):
            return False
        rank = candidate["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or rank != expected_rank
        ):
            return False
        if not all(
            isinstance(candidate[field], str) and candidate[field]
            for field in ("chunk_id", "doc_id", "section", "source")
        ):
            return False
        page = candidate["page"]
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
        ):
            return False
        if not _is_number(candidate["score"]):
            return False
        if not all(
            _is_optional_rank(candidate[field])
            for field in ("dense_rank", "bm25_rank", "fused_rank")
        ):
            return False
    return True


def _valid_score_summary(summary: object, candidates: list[dict]) -> bool:
    required = {
        "count",
        "top_score",
        "top1_top2_margin",
        "mean",
        "std",
        "min",
        "max",
        "range",
    }
    if not isinstance(summary, dict) or not required <= set(summary):
        return False
    if (
        not isinstance(summary["count"], int)
        or isinstance(summary["count"], bool)
        or summary["count"] != len(candidates)
    ):
        return False
    if not all(
        _is_number(summary[field])
        for field in ("top_score", "mean", "std", "min", "max", "range")
    ):
        return False
    scores = [float(candidate["score"]) for candidate in candidates]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    expected_margin = scores[0] - scores[1] if len(scores) > 1 else None
    if expected_margin is None:
        if summary["top1_top2_margin"] is not None:
            return False
    elif (
        not _is_number(summary["top1_top2_margin"])
        or float(summary["top1_top2_margin"]) != expected_margin
    ):
        return False
    expected = {
        "top_score": scores[0],
        "mean": mean,
        "std": variance**0.5,
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores),
    }
    # The producer performs these same Python-float operations over the same
    # ordered scores. JSON's float representation round-trips exactly, so an
    # approximation tolerance would only permit corrupted cached summaries.
    return all(float(summary[field]) == value for field, value in expected.items())


def _valid_available_reader(reader: dict) -> bool:
    integer_fields = (
        "passage_count",
        "context_characters",
        "window_count",
        "candidate_start_count",
        "candidate_span_count",
    )
    if not isinstance(reader["found"], bool):
        return False
    if not all(
        isinstance(reader[field], int)
        and not isinstance(reader[field], bool)
        and reader[field] >= 0
        for field in integer_fields
    ):
        return False
    optional_numbers = (
        "best_span_score",
        "second_best_span_score",
        "selected_null_score",
        "null_score_min",
        "null_score_max",
        "span_null_margin",
    )
    if not all(_is_optional_number(reader[field]) for field in optional_numbers):
        return False
    optional_indices = (
        "selected_window",
        "selected_start_token",
        "selected_end_token",
    )
    if not all(
        value is None
        or (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )
        for value in (reader[field] for field in optional_indices)
    ):
        return False
    selected_passage = reader["selected_passage"]
    if selected_passage is not None and (
        not isinstance(selected_passage, int)
        or isinstance(selected_passage, bool)
        or selected_passage < 1
    ):
        return False
    return reader["no_span_reason"] is None or isinstance(
        reader["no_span_reason"], str
    )
