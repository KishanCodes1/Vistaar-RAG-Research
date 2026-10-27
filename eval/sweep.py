"""Offline abstention threshold sweep.

The reader records a raw confidence for every question, so the abstention
threshold can be swept after the fact. That yields a risk-coverage curve
(El-Yaniv & Wiener, 2010) and an AUC for confidence-based abstention without
re-running the model.

Usage:
    python -m eval.sweep --config E --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import hash_file


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """AUC of `scores` at predicting label == 1. Rank-based, no sklearn needed."""
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    n_pos = int(labels_arr.sum())
    n_neg = int(len(labels_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores_arr)
    ranks = np.empty(len(scores_arr), dtype=float)
    # average ranks for ties
    sorted_scores = scores_arr[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum = ranks[labels_arr == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1(pred: str, gold: str) -> float:
    from .metrics import token_f1

    return token_f1(pred, gold)


def sweep(records: list[dict], thresholds: list[float] | None = None) -> dict:
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.0, 1.001, 0.05)]

    scored = []
    for r in records:
        conf = float(r.get("scores", {}).get("confidence", 0.0))
        raw_prediction = (
            r["raw_prediction"] if "raw_prediction" in r else r["prediction"]
        )
        scored.append(
            {
                "confidence": conf,
                "answerable": bool(r["answerable"]),
                # Use the reader's raw span, not the abstention message, so that
        # selective F1 measures answer quality rather than being zeroed out by
        # the very decision being swept.
        "f1": f1(raw_prediction, r["gold"])
        if r["answerable"] and r["gold"]
        else 0.0,
            }
        )

    labels = [1 if s["answerable"] else 0 for s in scored]
    confidences = [s["confidence"] for s in scored]
    auc = roc_auc(confidences, labels)

    curve = []
    for threshold in thresholds:
        answered = [s for s in scored if s["confidence"] >= threshold]
        coverage = len(answered) / len(scored) if scored else 0.0
        answerable = [s for s in answered if s["answerable"]]
        unanswerable = [s for s in answered if not s["answerable"]]
        selective_f1 = (
            float(np.mean([s["f1"] for s in answerable])) if answerable else 0.0
        )
        abstention_acc = (
            float(
                np.mean(
                    [
                        1.0 if (s["confidence"] >= threshold) == s["answerable"] else 0.0
                        for s in scored
                    ]
                )
            )
            if scored
            else 0.0
        )
        # Balanced accuracy over the two answerability classes. Plain
        # accuracy is dominated by the answerable base rate (~96% here), so it
        # would always select "answer everything".
        tpr = (len(answerable) / len([s for s in scored if s["answerable"]])) if any(
            s["answerable"] for s in scored
        ) else 0.0
        tnr = (len([s for s in scored if not s["answerable"] and s["confidence"] < threshold])
               / len([s for s in scored if not s["answerable"]])) if any(
            not s["answerable"] for s in scored
        ) else 0.0
        curve.append(
            {
                "threshold": threshold,
                "coverage": round(coverage, 4),
                "selective_f1": round(selective_f1, 4),
                "abstention_accuracy": round(abstention_acc, 4),
                "balanced_accuracy": round((tpr + tnr) / 2, 4),
                "answered_unanswerable": len(unanswerable),
                "answered_answerable": len(answerable),
            }
        )

    # Operating point: maximise selective F1 (quality of answers actually
    # given), tie-break toward coverage. Balanced accuracy is reported
    # alongside for the abstention decision itself.
    best = max(curve, key=lambda row: (row["selective_f1"], row["coverage"]))
    return {"auc": round(auc, 4), "n": len(scored), "curve": curve, "best": best}


def to_markdown(result: dict, config: str) -> str:
    lines = [
        f"### Abstention sweep — {config}",
        "",
        f"Confidence AUC (answerable vs unanswerable): **{result['auc']:.4f}** (n={result['n']})",
        "",
        "| Threshold | Coverage | Selective F1 | Balanced Acc | Answered unanswerable |",
        "|---|---|---|---|---|",
    ]
    for row in result["curve"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['coverage']:.3f} | {row['selective_f1']:.3f} "
            f"| {row['balanced_accuracy']:.3f} | {row['answered_unanswerable']} |"
        )
    lines.append("")
    lines.append(
        f"**Best threshold:** {result['best']['threshold']:.2f} "
        f"(coverage {result['best']['coverage']:.3f}, "
        f"selective F1 {result['best']['selective_f1']:.3f}, "
        f"balanced accuracy {result['best']['balanced_accuracy']:.3f})"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="results/records.jsonl")
    parser.add_argument("--config", default="E")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing sweep artifacts",
    )
    args = parser.parse_args()

    records = load_records(Path(args.records))
    records = [r for r in records if r["config"].startswith(args.config)]
    if not records:
        raise SystemExit(f"No records for config prefix {args.config!r}")

    result = sweep(records)
    result["source"] = {
        "records": str(args.records),
        "records_sha256": hash_file(args.records),
        "config_prefix": args.config,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"sweep_{args.config}.json"
    markdown_path = out_dir / f"sweep_{args.config}.md"
    if not args.overwrite and (json_path.exists() or markdown_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing sweep in {out_dir}; "
            "choose a new --out path or pass --overwrite explicitly"
        )
    json_path.write_text(json.dumps(result, indent=2))
    text = to_markdown(result, args.config)
    markdown_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
