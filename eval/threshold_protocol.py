"""Document-disjoint threshold selection for selective abstention.

The final test partition is never consulted when selecting a threshold. The
default objective is balanced answerability accuracy on calibration documents.
Questions are grouped by source paper for both holdout and cross-validation.

Usage:
    python -m eval.threshold_protocol \
        --records results/milestone2/balanced-clean-v1/records.jsonl \
        --config E --out results/milestone2/balanced-clean-v1/threshold_protocol
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .metrics import token_f1
from .provenance import hash_file, stable_fingerprint, write_json_atomic
from .sweep import load_records, roc_auc


PROTOCOL_SCHEMA = "vistaar-threshold-protocol-v1"
DEFAULT_THRESHOLDS = [round(value, 2) for value in np.arange(0.0, 1.001, 0.05)]


def _confidence(record: dict) -> float:
    return float(record.get("scores", {}).get("confidence", 0.0))


def _raw_prediction(record: dict) -> str:
    return record["raw_prediction"] if "raw_prediction" in record else record["prediction"]


def _answer_f1(record: dict) -> float:
    if not record["answerable"] or not record.get("gold"):
        return 0.0
    return token_f1(_raw_prediction(record), record["gold"])


def evaluate_decisions(records: list[dict], answered: list[bool]) -> dict:
    if len(records) != len(answered):
        raise ValueError("records and decisions must have the same length")
    answerable = [i for i, record in enumerate(records) if record["answerable"]]
    unanswerable = [i for i, record in enumerate(records) if not record["answerable"]]
    answered_indices = [i for i, decision in enumerate(answered) if decision]
    answered_answerable = [i for i in answerable if answered[i]]
    answered_unanswerable = [i for i in unanswerable if answered[i]]
    f1_values = [_answer_f1(record) for record in records]

    tpr = (
        len(answered_answerable) / len(answerable) if answerable else float("nan")
    )
    tnr = (
        sum(not answered[i] for i in unanswerable) / len(unanswerable)
        if unanswerable
        else float("nan")
    )
    balanced_accuracy = (
        (tpr + tnr) / 2 if not math.isnan(tpr) and not math.isnan(tnr) else float("nan")
    )
    conditional_f1 = (
        mean(f1_values[i] for i in answered_answerable)
        if answered_answerable
        else 0.0
    )
    covered_f1 = (
        mean(f1_values[i] for i in answered_indices) if answered_indices else 0.0
    )
    return {
        "n": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "coverage": len(answered_indices) / len(records) if records else 0.0,
        "conditional_answerable_f1": conditional_f1,
        "covered_f1": covered_f1,
        "balanced_accuracy": balanced_accuracy,
        "abstention_accuracy": (
            mean(answered[i] == bool(records[i]["answerable"]) for i in range(len(records)))
            if records
            else 0.0
        ),
        "answered_answerable": len(answered_answerable),
        "answered_unanswerable": len(answered_unanswerable),
        "answering_precision": (
            len(answered_answerable) / len(answered_indices) if answered_indices else 0.0
        ),
    }


def evaluate_threshold(records: list[dict], threshold: float) -> dict:
    result = evaluate_decisions(
        records, [bool(_confidence(record) >= threshold) for record in records]
    )
    result["threshold"] = threshold
    return result


def select_threshold(
    calibration: list[dict],
    objective: str = "balanced_accuracy",
    thresholds: list[float] | None = None,
) -> tuple[float, list[dict]]:
    """Select only from calibration records, tie-breaking toward coverage."""
    if objective not in {"balanced_accuracy", "covered_f1"}:
        raise ValueError(f"unsupported threshold objective: {objective}")
    thresholds = thresholds or DEFAULT_THRESHOLDS
    curve = [evaluate_threshold(calibration, threshold) for threshold in thresholds]
    valid = [row for row in curve if not math.isnan(row[objective])]
    if not valid:
        raise ValueError("calibration partition lacks both answerability classes")
    best = max(valid, key=lambda row: (row[objective], row["coverage"]))
    return float(best["threshold"]), curve


def _hash_order(doc_id: str, seed: int, salt: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{doc_id}".encode()).hexdigest()


def document_holdout_split(
    records: list[dict], calibration_fraction: float, seed: int
) -> tuple[list[dict], list[dict], dict]:
    """Deterministically split negative-containing and positive-only documents."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_doc[record["doc_id"]].append(record)
    if len(by_doc) < 2:
        raise ValueError("at least two documents are required for a holdout")

    strata = {
        "contains_unanswerable": [
            doc_id
            for doc_id, rows in by_doc.items()
            if any(not row["answerable"] for row in rows)
        ],
        "answerable_only": [
            doc_id
            for doc_id, rows in by_doc.items()
            if all(row["answerable"] for row in rows)
        ],
    }
    calibration_docs: set[str] = set()
    for name, docs in strata.items():
        if not docs:
            continue
        ordered = sorted(docs, key=lambda doc_id: _hash_order(doc_id, seed, name))
        if len(ordered) == 1:
            count = 1 if not calibration_docs else 0
        else:
            count = min(
                len(ordered) - 1,
                max(1, round(len(ordered) * calibration_fraction)),
            )
        calibration_docs.update(ordered[:count])

    all_docs = set(by_doc)
    test_docs = all_docs - calibration_docs
    if not calibration_docs or not test_docs:
        raise ValueError("holdout split produced an empty document partition")
    calibration = [row for row in records if row["doc_id"] in calibration_docs]
    test = [row for row in records if row["doc_id"] in test_docs]
    split = {
        "seed": seed,
        "calibration_fraction_target": calibration_fraction,
        "calibration_documents": sorted(calibration_docs),
        "test_documents": sorted(test_docs),
        "calibration_question_ids": [row["question_id"] for row in calibration],
        "test_question_ids": [row["question_id"] for row in test],
        "strata": {name: len(docs) for name, docs in strata.items()},
    }
    return calibration, test, split


def _partition_summary(records: list[dict]) -> dict:
    docs = {record["doc_id"] for record in records}
    labels = [int(record["answerable"]) for record in records]
    scores = [_confidence(record) for record in records]
    return {
        "documents": len(docs),
        "questions": len(records),
        "answerable": sum(labels),
        "unanswerable": len(labels) - sum(labels),
        "auc": roc_auc(scores, labels),
    }


def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def cluster_bootstrap(
    records: list[dict], threshold: float, resamples: int = 2000, seed: int = 13
) -> dict:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_doc[record["doc_id"]].append(record)
    docs = sorted(by_doc)
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    metrics = (
        "balanced_accuracy",
        "coverage",
        "conditional_answerable_f1",
        "covered_f1",
        "abstention_accuracy",
        "answered_unanswerable",
    )
    auc_values = []
    for _ in range(resamples):
        sample = []
        for _ in docs:
            sample.extend(by_doc[rng.choice(docs)])
        result = evaluate_threshold(sample, threshold)
        for metric in metrics:
            value = float(result[metric])
            if not math.isnan(value):
                values[metric].append(value)
        labels = [int(record["answerable"]) for record in sample]
        auc = roc_auc([_confidence(record) for record in sample], labels)
        if not math.isnan(auc):
            auc_values.append(auc)

    output = {}
    for metric in metrics:
        metric_values = values[metric]
        output[metric] = {
            "lower": float(np.percentile(metric_values, 2.5)),
            "upper": float(np.percentile(metric_values, 97.5)),
            "valid_resamples": len(metric_values),
        }
    output["auc"] = {
        "lower": float(np.percentile(auc_values, 2.5)) if auc_values else None,
        "upper": float(np.percentile(auc_values, 97.5)) if auc_values else None,
        "valid_resamples": len(auc_values),
    }
    return output


def _folds_by_document(records: list[dict], folds: int, seed: int) -> list[set[str]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_doc[record["doc_id"]].append(record)
    if len(by_doc) < folds:
        raise ValueError("number of folds exceeds document count")
    fold_docs = [set() for _ in range(folds)]
    strata = [
        [doc for doc, rows in by_doc.items() if any(not row["answerable"] for row in rows)],
        [doc for doc, rows in by_doc.items() if all(row["answerable"] for row in rows)],
    ]
    for stratum_index, docs in enumerate(strata):
        ordered = sorted(
            docs, key=lambda doc: _hash_order(doc, seed, f"fold-{stratum_index}")
        )
        for index, doc in enumerate(ordered):
            fold_docs[index % folds].add(doc)
    return fold_docs


def grouped_cross_validation(
    records: list[dict],
    objective: str,
    folds: int,
    repeats: int,
    seed: int,
) -> dict:
    runs = []
    threshold_counts: Counter[float] = Counter()
    for repeat in range(repeats):
        decisions: dict[str, bool] = {}
        selected = []
        for test_docs in _folds_by_document(records, folds, seed + repeat):
            calibration = [row for row in records if row["doc_id"] not in test_docs]
            test = [row for row in records if row["doc_id"] in test_docs]
            threshold, _ = select_threshold(calibration, objective)
            selected.append(threshold)
            threshold_counts[threshold] += 1
            for record in test:
                decisions[record["question_id"]] = bool(
                    _confidence(record) >= threshold
                )
        if len(decisions) != len(records):
            raise RuntimeError("cross-validation did not produce one decision per question")
        metrics = evaluate_decisions(
            records, [decisions[record["question_id"]] for record in records]
        )
        runs.append({"repeat": repeat, "thresholds": selected, "metrics": metrics})

    summary = {}
    for metric in (
        "balanced_accuracy",
        "coverage",
        "conditional_answerable_f1",
        "covered_f1",
        "abstention_accuracy",
        "answered_unanswerable",
    ):
        values = [float(run["metrics"][metric]) for run in runs]
        values = [value for value in values if not math.isnan(value)]
        summary[metric] = {
            "mean": mean(values),
            "split_p2_5": float(np.percentile(values, 2.5)),
            "split_p97_5": float(np.percentile(values, 97.5)),
        }
    return {
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "threshold_counts": {str(key): value for key, value in sorted(threshold_counts.items())},
        "summary": summary,
        "runs": runs,
    }


def run_protocol(
    records: list[dict],
    *,
    objective: str = "balanced_accuracy",
    calibration_fraction: float = 0.6,
    seed: int = 20261028,
    bootstrap_resamples: int = 2000,
    cv_folds: int = 5,
    cv_repeats: int = 20,
) -> dict:
    calibration, test, split = document_holdout_split(
        records, calibration_fraction, seed
    )
    threshold, calibration_curve = select_threshold(calibration, objective)
    calibration_metrics = evaluate_threshold(calibration, threshold)
    test_metrics = evaluate_threshold(test, threshold)
    labels = [int(record["answerable"]) for record in test]
    test_auc = roc_auc([_confidence(record) for record in test], labels)
    baselines = {
        "always_answer": evaluate_threshold(test, 0.0),
        "always_abstain": evaluate_threshold(test, 1.000001),
        "default_0_5": evaluate_threshold(test, 0.5),
    }
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "objective": objective,
        "threshold_grid": DEFAULT_THRESHOLDS,
        "selected_threshold": threshold,
        "selection_rule": "maximise calibration objective; tie-break toward coverage",
        "split": split,
        "calibration": {
            **_partition_summary(calibration),
            "selected_threshold_metrics": calibration_metrics,
            "curve": calibration_curve,
        },
        "test": {
            **_partition_summary(test),
            "auc": test_auc,
            "locked_threshold_metrics": test_metrics,
            "baselines": baselines,
            "cluster_bootstrap_95": cluster_bootstrap(
                test, threshold, bootstrap_resamples, seed + 1
            ),
        },
        "grouped_cross_validation": grouped_cross_validation(
            records, objective, cv_folds, cv_repeats, seed + 2
        ),
    }
    payload["protocol_fingerprint"] = stable_fingerprint(payload)
    return _json_safe(payload)


def to_markdown(result: dict, config: str) -> str:
    calibration = result["calibration"]
    test = result["test"]
    metrics = test["locked_threshold_metrics"]
    baseline = test["baselines"]["always_answer"]
    ci = test["cluster_bootstrap_95"]
    return "\n".join(
        [
            f"# Document-grouped threshold protocol — {config}",
            "",
            f"Objective: `{result['objective']}` on calibration documents only.",
            f"Locked threshold: **{result['selected_threshold']:.2f}**.",
            "",
            "| Partition | Documents | Questions | Answerable | Unanswerable |",
            "|---|---:|---:|---:|---:|",
            f"| Calibration | {calibration['documents']} | {calibration['questions']} | {calibration['answerable']} | {calibration['unanswerable']} |",
            f"| Held-out test | {test['documents']} | {test['questions']} | {test['answerable']} | {test['unanswerable']} |",
            "",
            "| Held-out metric | Locked threshold | Always answer |",
            "|---|---:|---:|",
            f"| Coverage | {metrics['coverage']:.4f} | {baseline['coverage']:.4f} |",
            f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} | {baseline['balanced_accuracy']:.4f} |",
            f"| Conditional answerable F1 | {metrics['conditional_answerable_f1']:.4f} | {baseline['conditional_answerable_f1']:.4f} |",
            f"| Covered F1 | {metrics['covered_f1']:.4f} | {baseline['covered_f1']:.4f} |",
            f"| Answered unanswerable | {metrics['answered_unanswerable']} | {baseline['answered_unanswerable']} |",
            f"| AUC | {test['auc']:.4f} | — |",
            "",
            "Cluster-bootstrap 95% intervals (fixed locked threshold):",
            f"balanced accuracy [{ci['balanced_accuracy']['lower']:.4f}, {ci['balanced_accuracy']['upper']:.4f}], "
            f"coverage [{ci['coverage']['lower']:.4f}, {ci['coverage']['upper']:.4f}], "
            f"AUC [{ci['auc']['lower']:.4f}, {ci['auc']['upper']:.4f}].",
            "",
            "Cross-validation percentiles describe split sensitivity, not a confidence interval.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--config", default="E")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--objective",
        choices=("balanced_accuracy", "covered_f1"),
        default="balanced_accuracy",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=20)
    args = parser.parse_args()

    records = [
        record
        for record in load_records(Path(args.records))
        if record["config"].startswith(args.config)
    ]
    if not records:
        raise SystemExit(f"No records found for config prefix {args.config!r}")
    if len({record["question_id"] for record in records}) != len(records):
        raise SystemExit("Records contain duplicate question IDs")

    result = run_protocol(
        records,
        objective=args.objective,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
    )
    result.pop("protocol_fingerprint", None)
    result["source"] = {
        "records": str(args.records),
        "records_sha256": hash_file(args.records),
        "config_prefix": args.config,
        "record_count": len(records),
        "implementation_sha256": hash_file(Path(__file__)),
    }
    result["protocol_fingerprint"] = stable_fingerprint(result)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty protocol directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "protocol.json", result)
    markdown = to_markdown(result, args.config)
    (out / "protocol.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
