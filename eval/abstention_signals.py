"""Offline comparison of abstention signals.

The reader records both its confidence and the top retrieval score for every
question, so alternative abstention signals can be scored without re-running
the model. This script reproduces the three-signal comparison reported in the
paper:

    confidence                 (reader logit margin)
    retrieval                  (top retrieval score)
    confidence + z(retrieval)  (simple standardised combination)

Usage:
    python -m eval.abstention_signals \
        --records results/balanced/records.jsonl \
        --config E --out results/balanced/signals.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .sweep import load_records, roc_auc


def signals(records: list[dict]) -> dict:
    confidence = np.asarray(
        [float(r.get("scores", {}).get("confidence", 0.0)) for r in records],
        dtype=float,
    )
    retrieval = np.asarray(
        [float(r.get("scores", {}).get("retrieval", 0.0)) for r in records],
        dtype=float,
    )
    labels = np.asarray([1 if r["answerable"] else 0 for r in records], dtype=int)

    # Population z-score (ddof=0); the scale is arbitrary for a rank metric but
    # is fixed here so the script is deterministic and reproducible.
    retrieval_z = (retrieval - retrieval.mean()) / retrieval.std()
    combined = confidence + retrieval_z

    auc_conf = roc_auc(confidence.tolist(), labels.tolist())
    auc_ret = roc_auc(retrieval.tolist(), labels.tolist())
    auc_comb = roc_auc(combined.tolist(), labels.tolist())

    return {
        "n": len(records),
        "answerable": int(labels.sum()),
        "unanswerable": int(len(labels) - labels.sum()),
        "retrieval_mean": round(float(retrieval.mean()), 5),
        "retrieval_std": round(float(retrieval.std()), 5),
        "auc": {
            "confidence": round(auc_conf, 4),
            "retrieval": round(auc_ret, 4),
            "confidence_plus_z_retrieval": round(auc_comb, 4),
        },
        "delta_vs_confidence": {
            "retrieval": round(auc_ret - auc_conf, 4),
            "confidence_plus_z_retrieval": round(auc_comb - auc_conf, 4),
        },
    }


def to_markdown(result: dict) -> str:
    lines = [
        "### Abstention signal comparison",
        "",
        f"n={result['n']} ({result['answerable']} answerable, "
        f"{result['unanswerable']} unanswerable)",
        "",
        "| Signal | AUC |",
        "|---|---|",
        f"| Reader confidence (logit margin) | **{result['auc']['confidence']:.4f}** |",
        f"| Retrieval score alone | {result['auc']['retrieval']:.4f} |",
        "| Confidence + z-scored retrieval score | "
        f"{result['auc']['confidence_plus_z_retrieval']:.4f} |",
        "",
        "Deltas vs confidence: retrieval "
        f"{result['delta_vs_confidence']['retrieval']:+.4f}, combined "
        f"{result['delta_vs_confidence']['confidence_plus_z_retrieval']:+.4f}.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="results/balanced/records.jsonl")
    parser.add_argument("--config", default="E")
    parser.add_argument("--out", default="results/balanced/signals.json")
    args = parser.parse_args()

    records = load_records(Path(args.records))
    records = [r for r in records if r["config"].startswith(args.config)]
    if not records:
        raise SystemExit(f"No records for config prefix {args.config!r}")

    result = signals(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
