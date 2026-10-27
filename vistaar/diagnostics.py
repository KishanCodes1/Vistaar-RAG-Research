"""Inference-time confidence diagnostics with no effect on model decisions."""
from __future__ import annotations

from typing import Any


CONFIDENCE_DIAGNOSTICS_SCHEMA = "vistaar-confidence-diagnostics-v1"


def snapshot_hits(hits: list[Any]) -> list[Any]:
    """Copy a candidate list only when diagnostic collection is enabled."""
    return list(hits)


def score_summary(hits: list[Any]) -> dict | None:
    """Summarize an already-ranked hit list without changing it."""
    if not hits:
        return None
    scores = [float(hit.score) for hit in hits]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "count": len(scores),
        "top_score": scores[0],
        "top1_top2_margin": scores[0] - scores[1] if len(scores) > 1 else None,
        "mean": mean,
        "std": variance**0.5,
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores),
    }


def ranked_hits(
    hits: list[Any],
    *,
    dense_ranks: dict[str, int] | None = None,
    bm25_ranks: dict[str, int] | None = None,
    fused_ranks: dict[str, int] | None = None,
) -> list[dict]:
    """Serialize hit provenance and any known upstream ranks."""
    rows = []
    for rank, hit in enumerate(hits, start=1):
        chunk_id = hit.chunk.chunk_id
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "doc_id": hit.chunk.doc_id,
                "page": hit.chunk.page,
                "section": hit.chunk.section,
                "source": hit.source,
                "score": float(hit.score),
                "dense_rank": dense_ranks.get(chunk_id) if dense_ranks else None,
                "bm25_rank": bm25_ranks.get(chunk_id) if bm25_ranks else None,
                "fused_rank": fused_ranks.get(chunk_id) if fused_ranks else None,
            }
        )
    return rows


def stage(
    *,
    requested: bool,
    available: bool,
    hits: list[Any] | None,
    reason: str | None = None,
    dense_ranks: dict[str, int] | None = None,
    bm25_ranks: dict[str, int] | None = None,
    fused_ranks: dict[str, int] | None = None,
) -> dict:
    """Represent stage availability explicitly; never fabricate missing scores."""
    if not requested or not available:
        return {
            "requested": requested,
            "available": available,
            "reason": reason or ("not_requested" if not requested else "unavailable"),
            "candidates": None,
            "score_summary": None,
        }
    values = hits or []
    return {
        "requested": True,
        "available": True,
        "reason": reason,
        "candidates": ranked_hits(
            values,
            dense_ranks=dense_ranks,
            bm25_ranks=bm25_ranks,
            fused_ranks=fused_ranks,
        ),
        "score_summary": score_summary(values),
    }
