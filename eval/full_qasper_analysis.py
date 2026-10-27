"""Independent record-only analysis for the full QASPER evaluation.

This module does not participate in inference or cache fingerprints. It verifies
saved records, computes document-cluster uncertainty, compares the full run with
the clean 40-paper run, and extracts held-out failure cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .compare_results import load_run
from .metrics import best_f1, citation_scores, exact_match, retrieval_recall
from .provenance import hash_file, write_json_atomic
from .sweep import load_records, sweep


SCHEMA = "vistaar-full-qasper-analysis-v1"


def _raw(record: dict) -> str:
    return record["raw_prediction"] if "raw_prediction" in record else record["prediction"]


def _references(record: dict) -> list[str]:
    return record.get("gold_answers") or ([record["gold"]] if record.get("gold") else [])


def _record_values(record: dict) -> dict[str, float]:
    answerable = bool(record["answerable"])
    refs = _references(record)
    f1 = best_f1(_raw(record), refs) if answerable else 0.0
    em = exact_match(_raw(record), refs) if answerable else 0.0
    cite_p, cite_r, _ = citation_scores(record["citations"], record["gold_evidence"])
    return {
        "answerable": float(answerable),
        "unanswerable": float(not answerable),
        "f1": f1,
        "em": em,
        "citation_precision": cite_p,
        "citation_recall": cite_r,
        "retrieval_recall": retrieval_recall(
            record["retrieved"][:4], record["gold_evidence"]
        ),
        "abstention_accuracy": float(record["abstained"] == (not answerable)),
        "over_abstention": float(answerable and record["abstained"]),
        "missed_unanswerable": float((not answerable) and (not record["abstained"])),
    }


def _aggregate(values: list[dict[str, float]]) -> dict[str, float]:
    answerable = [value for value in values if value["answerable"]]
    unanswerable = [value for value in values if value["unanswerable"]]
    return {
        "f1": mean(value["f1"] for value in answerable) if answerable else float("nan"),
        "em": mean(value["em"] for value in answerable) if answerable else float("nan"),
        "citation_precision": mean(value["citation_precision"] for value in values),
        "citation_recall": mean(value["citation_recall"] for value in values),
        "retrieval_recall": mean(value["retrieval_recall"] for value in values),
        "abstention_accuracy": mean(value["abstention_accuracy"] for value in values),
        "over_abstention": (
            sum(value["over_abstention"] for value in answerable) / len(answerable)
            if answerable
            else float("nan")
        ),
        "missed_unanswerable": (
            sum(value["missed_unanswerable"] for value in unanswerable)
            / len(unanswerable)
            if unanswerable
            else float("nan")
        ),
    }


def document_cluster_intervals(
    records: list[dict], resamples: int = 2000, seed: int = 20261028
) -> dict:
    by_doc: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        by_doc[record["doc_id"]].append(_record_values(record))
    docs = sorted(by_doc)
    point = _aggregate([value for doc in docs for value in by_doc[doc]])
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sample = []
        for _ in docs:
            sample.extend(by_doc[rng.choice(docs)])
        aggregate = _aggregate(sample)
        for metric, value in aggregate.items():
            if not math.isnan(value):
                distributions[metric].append(value)
    output = {}
    for metric, point_value in point.items():
        values = distributions[metric]
        output[metric] = {
            "point": None if math.isnan(point_value) else point_value,
            "lower": float(np.percentile(values, 2.5)) if values else None,
            "upper": float(np.percentile(values, 97.5)) if values else None,
            "valid_resamples": len(values),
        }
    return output


def paired_document_delta(
    left: list[dict], right: list[dict], metric: str, resamples: int = 5000, seed: int = 13
) -> dict:
    left_by_id = {record["question_id"]: record for record in left}
    right_by_id = {record["question_id"]: record for record in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired configurations have different question IDs")
    by_doc: dict[str, list[float]] = defaultdict(list)
    for question_id, left_record in left_by_id.items():
        right_record = right_by_id[question_id]
        if metric == "retrieval_recall":
            left_value = retrieval_recall(
                left_record["retrieved"][:4], left_record["gold_evidence"]
            )
            right_value = retrieval_recall(
                right_record["retrieved"][:4], right_record["gold_evidence"]
            )
        elif metric == "f1":
            if not left_record["answerable"]:
                continue
            left_value = best_f1(_raw(left_record), _references(left_record))
            right_value = best_f1(_raw(right_record), _references(right_record))
        else:
            raise ValueError(f"unsupported paired metric: {metric}")
        by_doc[left_record["doc_id"]].append(right_value - left_value)
    docs = sorted(by_doc)
    observed = mean(value for doc in docs for value in by_doc[doc])
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        values = []
        for _ in docs:
            values.extend(by_doc[rng.choice(docs)])
        samples.append(mean(values))
    return {
        "metric": metric,
        "documents": len(docs),
        "point_delta": observed,
        "lower": float(np.percentile(samples, 2.5)),
        "upper": float(np.percentile(samples, 97.5)),
        "resamples": resamples,
    }


def compare_overlap(full_records: list[dict], previous_records: list[dict]) -> dict:
    def scientific(record: dict) -> dict:
        excluded = {"latency_s", "_cache"}
        return {key: value for key, value in record.items() if key not in excluded}

    full = {(record["config"], record["question_id"]): record for record in full_records}
    previous = {
        (record["config"], record["question_id"]): record for record in previous_records
    }
    common = sorted(full.keys() & previous.keys())
    exact = sum(scientific(full[key]) == scientific(previous[key]) for key in common)
    return {
        "overlapping_records": len(common),
        "scientifically_identical": exact,
        "different": len(common) - exact,
    }


def verify_run(run_dir: Path, protocol: dict, protocol_plan: dict) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    records = load_records(run_dir / "records.jsonl")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["config"]].append(record)
    loaded = load_run(run_dir)
    expected_records = summary["questions"] * len(summary["results"])
    implementation_matches = all(
        hash_file(Path(relative)) == expected
        for relative, expected in manifest["fingerprinted_inputs"]["implementation"].items()
    )
    rerank_valid = all(
        all(hit["source"] == "rerank" for hit in record["retrieved_chunks"])
        for record in records
        if "rerank" in record["config"]
    )
    cache_valid = all(
        record.get("_cache", {}).get("run_fingerprint") == summary["run_fingerprint"]
        for record in records
    )
    unique_per_config = all(
        len(rows) == len({row["question_id"] for row in rows})
        for rows in grouped.values()
    )
    no_retrieval_duplicates = all(
        len(record["retrieved_chunks"])
        == len({hit["chunk_id"] for hit in record["retrieved_chunks"]})
        for record in records
    )
    max_summary_difference = max(
        (
            difference
            for config in loaded["configs"].values()
            for difference in config["summary_abs_difference"].values()
        ),
        default=0.0,
    )
    calibration_docs = set(protocol["split"]["calibration_documents"])
    test_docs = set(protocol["split"]["test_documents"])
    record_docs = {record["doc_id"] for record in records}
    namespace_to_document = {
        row["namespace"]: row["document_id"] for row in manifest["index"]["per_document"]
    }
    chunk_provenance_valid = all(
        all(hit["doc_id"] == namespace_to_document.get(record["doc_id"]) for hit in record["retrieved_chunks"])
        for record in records
    )
    e_records = next(rows for config, rows in grouped.items() if config.startswith("E"))
    saved_sweep_path = run_dir / "sweep_E.json"
    sweep_matches = False
    if saved_sweep_path.exists():
        saved_sweep = json.loads(saved_sweep_path.read_text())
        generated_sweep = sweep(e_records)
        sweep_matches = all(
            saved_sweep.get(key) == generated_sweep.get(key)
            for key in ("auc", "n", "curve", "best")
        )
    protocol_source_valid = (
        protocol.get("source", {}).get("records_sha256")
        == hash_file(run_dir / "records.jsonl")
    )
    protocol_matches_plan = (
        protocol["split"]["calibration_documents"]
        == protocol_plan["split"]["calibration_documents"]
        and protocol["split"]["test_documents"]
        == protocol_plan["split"]["test_documents"]
        and protocol["split"]["calibration_question_ids"]
        == protocol_plan["split"]["calibration_question_ids"]
        and protocol["split"]["test_question_ids"]
        == protocol_plan["split"]["test_question_ids"]
        and protocol["objective"] == protocol_plan["protocol"]["objective"]
        and protocol["threshold_grid"] == protocol_plan["protocol"]["threshold_grid"]
    )
    checks = {
        "manifest_complete": manifest.get("status") == "complete",
        "record_count": len(records),
        "expected_record_count": expected_records,
        "unique_question_ids_per_config": unique_per_config,
        "index_chunks": summary["index"]["chunks"],
        "index_unique_chunks": summary["index"]["unique_chunks"],
        "index_duplicate_chunks": summary["index"]["duplicate_chunks"],
        "duplicate_retrieval_lists": 0 if no_retrieval_duplicates else 1,
        "cache_metadata_matches_run": cache_valid,
        "implementation_hashes_match": implementation_matches,
        "rerank_sources_valid": rerank_valid,
        "chunk_provenance_valid": chunk_provenance_valid,
        "max_summary_rounding_difference": max_summary_difference,
        "sweep_matches_records": sweep_matches,
        "protocol_source_matches_records": protocol_source_valid,
        "protocol_matches_predeclared_plan": protocol_matches_plan,
        "threshold_documents_disjoint": calibration_docs.isdisjoint(test_docs),
        "threshold_all_documents_covered": calibration_docs | test_docs == record_docs,
        "errors": sum(row["errors"] for row in summary["results"]),
    }
    checks["all_checks_passed"] = (
        checks["manifest_complete"]
        and checks["record_count"] == checks["expected_record_count"]
        and checks["unique_question_ids_per_config"]
        and checks["index_chunks"] == checks["index_unique_chunks"]
        and checks["index_duplicate_chunks"] == 0
        and checks["duplicate_retrieval_lists"] == 0
        and checks["cache_metadata_matches_run"]
        and checks["implementation_hashes_match"]
        and checks["rerank_sources_valid"]
        and checks["chunk_provenance_valid"]
        and checks["max_summary_rounding_difference"] == 0
        and checks["sweep_matches_records"]
        and checks["protocol_source_matches_records"]
        and checks["protocol_matches_predeclared_plan"]
        and checks["threshold_documents_disjoint"]
        and checks["threshold_all_documents_covered"]
        and checks["errors"] == 0
    )
    return checks


def failure_cases(records: list[dict], protocol: dict, limit: int = 10) -> dict:
    test_ids = set(protocol["split"]["test_question_ids"])
    threshold = float(protocol["selected_threshold"])
    test = [record for record in records if record["question_id"] in test_ids]

    def compact(record: dict) -> dict:
        return {
            "question_id": record["question_id"],
            "doc_id": record["doc_id"],
            "question": record["question"],
            "answerable": record["answerable"],
            "confidence": float(record["scores"]["confidence"]),
            "raw_prediction": _raw(record),
            "gold_answers": record.get("gold_answers") or [record.get("gold", "")],
            "answer_f1": (
                best_f1(_raw(record), _references(record)) if record["answerable"] else 0.0
            ),
            "retrieval_recall": retrieval_recall(
                record["retrieved"][:4], record["gold_evidence"]
            ),
            "top_chunks": [
                {
                    "chunk_id": hit["chunk_id"],
                    "score": hit["score"],
                    "text_preview": hit["text"][:300],
                }
                for hit in record["retrieved_chunks"][:2]
            ],
        }

    answered_unanswerable = sorted(
        [
            record
            for record in test
            if not record["answerable"]
            and float(record["scores"]["confidence"]) >= threshold
        ],
        key=lambda record: float(record["scores"]["confidence"]),
        reverse=True,
    )
    abstained_answerable = [
        record
        for record in test
        if record["answerable"]
        and float(record["scores"]["confidence"]) < threshold
    ]
    missed_good_answers = sorted(
        abstained_answerable,
        key=lambda record: best_f1(_raw(record), _references(record)),
        reverse=True,
    )
    retrieval_failures = sorted(
        [record for record in test if record["answerable"]],
        key=lambda record: (
            retrieval_recall(record["retrieved"][:4], record["gold_evidence"]),
            -best_f1(_raw(record), _references(record)),
        ),
    )
    return {
        "threshold": threshold,
        "test_questions": len(test),
        "answered_unanswerable_count": len(answered_unanswerable),
        "abstained_answerable_count": len(abstained_answerable),
        "highest_confidence_answered_unanswerable": [
            compact(record) for record in answered_unanswerable[:limit]
        ],
        "highest_f1_abstained_answerable": [
            compact(record) for record in missed_good_answers[:limit]
        ],
        "lowest_retrieval_recall_answerable": [
            compact(record) for record in retrieval_failures[:limit]
        ],
    }


def _fmt_interval(value: dict) -> str:
    return f"{value['point']:.4f} [{value['lower']:.4f}, {value['upper']:.4f}]"


def to_markdown(analysis: dict) -> str:
    summary = analysis["summary"]
    protocol = analysis["threshold_protocol"]
    test = protocol["test"]
    locked = test["locked_threshold_metrics"]
    always = test["baselines"]["always_answer"]
    lines = [
        "# Full-QASPER record analysis",
        "",
        f"Run fingerprint: `{summary['run_fingerprint']}`",
        f"Dataset: {summary['documents']} papers, {summary['questions']} questions "
        f"({summary['answerable']} answerable / {summary['unanswerable']} unanswerable).",
        f"Index: {summary['index']['chunks']} stored / {summary['index']['unique_chunks']} unique chunks.",
        "",
        "## Aggregate metrics with document-cluster 95% intervals",
        "",
        "| Config | F1 | EM | Cite-P | Cite-R | Ret-R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config, intervals in analysis["cluster_intervals"].items():
        lines.append(
            f"| {config} | {_fmt_interval(intervals['f1'])} | "
            f"{_fmt_interval(intervals['em'])} | "
            f"{_fmt_interval(intervals['citation_precision'])} | "
            f"{_fmt_interval(intervals['citation_recall'])} | "
            f"{_fmt_interval(intervals['retrieval_recall'])} |"
        )
    novel = analysis["new_document_subset"]
    lines.extend(
        [
            "",
            f"A separate expansion subset excludes the {novel['excluded_previous_documents']} "
            f"papers in the clean Milestone 2 natural run: {novel['documents']} papers, "
            f"{novel['questions_per_config']} questions per config "
            f"({novel['answerable_per_config']} answerable / {novel['unanswerable_per_config']} unanswerable).",
            "",
            "## Locked held-out abstention",
            "",
            f"Threshold `{protocol['selected_threshold']:.2f}` was selected only on "
            f"{protocol['calibration']['documents']} calibration documents and applied to "
            f"{test['documents']} held-out documents.",
            "",
            "| Metric | Locked threshold | Always answer |",
            "|---|---:|---:|",
            f"| AUC | {test['auc']:.4f} | — |",
            f"| Balanced accuracy | {locked['balanced_accuracy']:.4f} | {always['balanced_accuracy']:.4f} |",
            f"| Coverage | {locked['coverage']:.4f} | {always['coverage']:.4f} |",
            f"| Conditional answerable F1 | {locked['conditional_answerable_f1']:.4f} | {always['conditional_answerable_f1']:.4f} |",
            f"| Covered F1 | {locked['covered_f1']:.4f} | {always['covered_f1']:.4f} |",
            f"| Answered unanswerable | {locked['answered_unanswerable']} | {always['answered_unanswerable']} |",
            "",
            "## Integrity and continuity",
            "",
            f"All checks passed: **{analysis['verification']['all_checks_passed']}**. "
            f"The clean 40-paper overlap has {analysis['previous_overlap']['scientifically_identical']}/"
            f"{analysis['previous_overlap']['overlapping_records']} scientifically identical records.",
            "",
            "## Failure counts on held-out documents",
            "",
            f"Answered unanswerable: {analysis['failure_cases']['answered_unanswerable_count']}; "
            f"abstained answerable: {analysis['failure_cases']['abstained_answerable_count']}.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--previous-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty analysis directory {out}")
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(run_dir / "records.jsonl")
    previous = load_records(Path(args.previous_dir) / "records.jsonl")
    protocol = json.loads(Path(args.protocol).read_text())
    protocol_plan = json.loads(Path(args.protocol_plan).read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["config"]].append(record)
    config_by_prefix = {
        prefix: next(name for name in grouped if name.startswith(prefix))
        for prefix in ("A", "C", "D", "E")
    }
    intervals = {
        config: document_cluster_intervals(
            config_records, args.bootstrap_resamples, seed=20261028
        )
        for config, config_records in grouped.items()
    }
    previous_docs = {record["doc_id"] for record in previous}
    new_document_records = [
        record for record in records if record["doc_id"] not in previous_docs
    ]
    new_document_grouped: dict[str, list[dict]] = defaultdict(list)
    for record in new_document_records:
        new_document_grouped[record["config"]].append(record)
    new_document_intervals = {
        config: document_cluster_intervals(
            config_records, args.bootstrap_resamples, seed=20261029
        )
        for config, config_records in new_document_grouped.items()
    }
    paired = {
        "A_to_C_retrieval": paired_document_delta(
            grouped[config_by_prefix["A"]], grouped[config_by_prefix["C"]], "retrieval_recall"
        ),
        "C_to_D_retrieval": paired_document_delta(
            grouped[config_by_prefix["C"]], grouped[config_by_prefix["D"]], "retrieval_recall"
        ),
        "A_to_E_retrieval": paired_document_delta(
            grouped[config_by_prefix["A"]], grouped[config_by_prefix["E"]], "retrieval_recall"
        ),
        "A_to_C_f1": paired_document_delta(
            grouped[config_by_prefix["A"]], grouped[config_by_prefix["C"]], "f1"
        ),
        "C_to_D_f1": paired_document_delta(
            grouped[config_by_prefix["C"]], grouped[config_by_prefix["D"]], "f1"
        ),
    }
    e_records = grouped[config_by_prefix["E"]]
    analysis = {
        "schema": SCHEMA,
        "records_sha256": hash_file(run_dir / "records.jsonl"),
        "summary": json.loads((run_dir / "summary.json").read_text()),
        "cluster_intervals": intervals,
        "new_document_subset": {
            "excluded_previous_documents": len(previous_docs),
            "documents": len({record["doc_id"] for record in new_document_records}),
            "questions_per_config": len(new_document_records) // len(grouped),
            "answerable_per_config": sum(
                record["answerable"]
                for record in next(iter(new_document_grouped.values()))
            ),
            "unanswerable_per_config": sum(
                not record["answerable"]
                for record in next(iter(new_document_grouped.values()))
            ),
            "cluster_intervals": new_document_intervals,
        },
        "paired_document_deltas": paired,
        "descriptive_sweep": sweep(e_records),
        "threshold_protocol": protocol,
        "previous_overlap": compare_overlap(records, previous),
        "verification": verify_run(run_dir, protocol, protocol_plan),
        "failure_cases": failure_cases(e_records, protocol),
    }
    analysis["analysis_fingerprint"] = hashlib.sha256(
        json.dumps(analysis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json_atomic(out / "analysis.json", analysis)
    write_json_atomic(out / "failure_cases.json", analysis["failure_cases"])
    write_json_atomic(out / "verification.json", analysis["verification"])
    markdown = to_markdown(analysis)
    (out / "analysis.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
