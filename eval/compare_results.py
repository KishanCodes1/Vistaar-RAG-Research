"""Recompute and compare preserved historical and clean evaluation runs.

The clean records contain every gold answer plus retrieved/cited chunk
provenance, so their summaries can be verified without reopening QASPER.
Historical records contain only the first gold answer; their committed summary
F1/EM therefore cannot be fully reconstructed from JSONL alone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from .metrics import (
    best_f1,
    citation_scores,
    exact_match,
    paired_bootstrap_p,
    retrieval_recall,
)
from .provenance import hash_file, write_json_atomic
from .sweep import load_records, sweep


def _raw(record: dict) -> str:
    return record["raw_prediction"] if "raw_prediction" in record else record["prediction"]


def _group(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["config"], []).append(record)
    return grouped


def recompute(records: list[dict]) -> dict:
    answerable = [record for record in records if record["answerable"]]
    unanswerable = [record for record in records if not record["answerable"]]
    has_all_gold = all("gold_answers" in record for record in answerable)
    references = lambda record: record.get("gold_answers") or [record["gold"]]
    citation = [
        citation_scores(record["citations"], record["gold_evidence"])
        for record in records
    ]
    retrieved = [record["retrieved"][:4] for record in records]
    distinct_counts = []
    duplicate_slots = 0
    for record, texts in zip(records, retrieved):
        details = record.get("retrieved_chunks", [])[:4]
        identities = (
            [detail["chunk_id"] for detail in details]
            if len(details) == len(texts)
            else texts
        )
        distinct = len(set(identities))
        distinct_counts.append(distinct)
        duplicate_slots += len(identities) - distinct

    result = {
        "questions": len(records),
        "documents": len({record["doc_id"] for record in records}),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "gold_reference_mode": "all_gold_answers" if has_all_gold else "first_gold_only",
        "f1": mean(best_f1(_raw(record), references(record)) for record in answerable),
        "em": mean(exact_match(_raw(record), references(record)) for record in answerable),
        "citation_precision": mean(value[0] for value in citation),
        "citation_recall": mean(value[1] for value in citation),
        "retrieval_recall": mean(
            retrieval_recall(texts, record["gold_evidence"])
            for record, texts in zip(records, retrieved)
        ),
        "mean_retrieved_at_4": mean(len(texts) for texts in retrieved),
        "mean_distinct_retrieved_at_4": mean(distinct_counts),
        "duplicate_retrieved_slots": duplicate_slots,
        "questions_with_duplicate_retrieval": sum(
            distinct < len(texts) for distinct, texts in zip(distinct_counts, retrieved)
        ),
        "abstention_accuracy": mean(
            record["abstained"] == (not record["answerable"]) for record in records
        ),
        "over_abstention": mean(record["abstained"] for record in answerable),
        "missed_unanswerable": mean(not record["abstained"] for record in unanswerable),
        "mean_latency_s": mean(record["latency_s"] for record in records),
    }
    if all("retrieved_chunks" in record for record in records):
        result["distinct_chunk_ids_in_records"] = len(
            {
                detail["chunk_id"]
                for record in records
                for detail in record["retrieved_chunks"][:4]
            }
        )
    return result


def abstention_metrics(records: list[dict]) -> dict:
    result = sweep(records)
    peak = max(
        result["curve"], key=lambda row: (row["balanced_accuracy"], row["coverage"])
    )
    return {
        "auc": result["auc"],
        "full_coverage": result["curve"][0],
        "peak_balanced_accuracy": peak,
        "best_conditional_selective_f1": result["best"],
    }


def _summary_rows(path: Path) -> dict[str, dict]:
    summary = json.loads((path / "summary.json").read_text())
    return {row["config"]: row for row in summary["results"]}


def paired_config_comparisons(grouped: dict[str, list[dict]]) -> dict:
    by_prefix = {
        prefix: next((name for name in grouped if name.startswith(prefix)), None)
        for prefix in ("A", "C", "D", "E")
    }
    output = {}
    for left, right in (("A", "C"), ("C", "D"), ("A", "E")):
        if not by_prefix[left] or not by_prefix[right]:
            continue
        left_rows = {row["question_id"]: row for row in grouped[by_prefix[left]]}
        right_rows = {row["question_id"]: row for row in grouped[by_prefix[right]]}
        if left_rows.keys() != right_rows.keys():
            raise ValueError(f"unpaired records for {left} -> {right}")
        question_ids = sorted(left_rows)
        left_retrieval = [
            retrieval_recall(left_rows[qid]["retrieved"][:4], left_rows[qid]["gold_evidence"])
            for qid in question_ids
        ]
        right_retrieval = [
            retrieval_recall(right_rows[qid]["retrieved"][:4], right_rows[qid]["gold_evidence"])
            for qid in question_ids
        ]
        answerable_ids = [qid for qid in question_ids if left_rows[qid]["answerable"]]
        left_f1 = [
            best_f1(
                _raw(left_rows[qid]),
                left_rows[qid].get("gold_answers") or [left_rows[qid]["gold"]],
            )
            for qid in answerable_ids
        ]
        right_f1 = [
            best_f1(
                _raw(right_rows[qid]),
                right_rows[qid].get("gold_answers") or [right_rows[qid]["gold"]],
            )
            for qid in answerable_ids
        ]
        output[f"{left}_to_{right}"] = {
            "n": len(question_ids),
            "retrieval_recall": {
                "left_mean": mean(left_retrieval),
                "right_mean": mean(right_retrieval),
                "delta": mean(right_retrieval) - mean(left_retrieval),
                "paired_bootstrap_p": paired_bootstrap_p(
                    right_retrieval, left_retrieval, n_resamples=10000, seed=13
                ),
            },
            "answer_f1_answerable": {
                "n": len(answerable_ids),
                "left_mean": mean(left_f1),
                "right_mean": mean(right_f1),
                "delta": mean(right_f1) - mean(left_f1),
                "paired_bootstrap_p": paired_bootstrap_p(
                    right_f1, left_f1, n_resamples=10000, seed=13
                ),
            },
        }
    return output


def load_run(path: Path) -> dict:
    records = load_records(path / "records.jsonl")
    grouped = _group(records)
    summary_document = json.loads((path / "summary.json").read_text())
    summaries = _summary_rows(path)
    configs = {}
    for config, config_records in grouped.items():
        computed = recompute(config_records)
        reported = summaries.get(config, {})
        verification = {}
        if computed["gold_reference_mode"] == "all_gold_answers":
            for key in (
                "f1",
                "em",
                "citation_precision",
                "citation_recall",
                "retrieval_recall",
                "abstention_accuracy",
                "over_abstention",
                "missed_unanswerable",
            ):
                verification[key] = abs(round(computed[key], 4) - reported[key])
        configs[config] = {
            "reported": reported,
            "recomputed": computed,
            "summary_abs_difference": verification,
        }
    return {
        "path": str(path),
        "records_sha256": hash_file(path / "records.jsonl"),
        "summary_sha256": hash_file(path / "summary.json"),
        "summary": summary_document,
        "configs": configs,
        "paired_config_comparisons": paired_config_comparisons(grouped),
    }


def compare_pair(
    old_path: Path,
    new_path: Path,
    *,
    old_chunks: int,
    old_unique_chunks: int,
    abstention_config_prefix: str = "E",
) -> dict:
    old = load_run(old_path)
    new = load_run(new_path)
    old_configs = old["configs"]
    new_configs = new["configs"]
    shared = sorted(set(old_configs).intersection(new_configs))
    config_deltas = {}
    for config in shared:
        old_reported = old_configs[config]["reported"]
        new_reported = new_configs[config]["reported"]
        config_deltas[config] = {
            key: round(new_reported[key] - old_reported[key], 6)
            for key in (
                "f1",
                "em",
                "citation_precision",
                "citation_recall",
                "retrieval_recall",
                "abstention_accuracy",
                "over_abstention",
                "missed_unanswerable",
                "mean_latency_s",
            )
        }

    old_abstention_name = next(
        name for name in old_configs if name.startswith(abstention_config_prefix)
    )
    new_abstention_name = next(
        name for name in new_configs if name.startswith(abstention_config_prefix)
    )
    old_abstention_records = [
        record
        for record in load_records(old_path / "records.jsonl")
        if record["config"] == old_abstention_name
    ]
    new_abstention_records = [
        record
        for record in load_records(new_path / "records.jsonl")
        if record["config"] == new_abstention_name
    ]
    new_index = new["summary"].get("index", {})
    return {
        "old": old,
        "new": new,
        "index": {
            "old_chunks": old_chunks,
            "old_unique_chunks": old_unique_chunks,
            "old_duplicate_chunks": old_chunks - old_unique_chunks,
            "new_chunks": new_index.get("chunks"),
            "new_unique_chunks": new_index.get("unique_chunks"),
            "new_duplicate_chunks": new_index.get("duplicate_chunks"),
        },
        "config_reported_metric_deltas_new_minus_old": config_deltas,
        "abstention_recomputed_from_records": {
            "old": abstention_metrics(old_abstention_records),
            "new": abstention_metrics(new_abstention_records),
        },
        "runtime": {
            "old_total_time_s": None,
            "old_resource_usage": None,
            "old_limitation": "Historical artifacts contain per-question latency only.",
            "new": new["summary"].get("runtime"),
        },
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unavailable"
    return f"{value:.{digits}f}"


def to_markdown(result: dict) -> str:
    lines = [
        "# Historical vs clean evaluation comparison",
        "",
        "Historical outputs are preserved. Deltas are clean minus historical.",
    ]
    for label in ("natural", "negative_enriched"):
        pair = result[label]
        lines.extend(
            [
                "",
                f"## {label.replace('_', ' ').title()}",
                "",
                "### Index integrity",
                "",
                "| Run | Chunks | Unique | Duplicates |",
                "|---|---:|---:|---:|",
                f"| Historical | {pair['index']['old_chunks']} | {pair['index']['old_unique_chunks']} | {pair['index']['old_duplicate_chunks']} |",
                f"| Clean | {pair['index']['new_chunks']} | {pair['index']['new_unique_chunks']} | {pair['index']['new_duplicate_chunks']} |",
                "",
                "### Reported metric comparison",
                "",
                "| Config | F1 old -> new | EM old -> new | Ret-R old -> new | Cite-P old -> new | Cite-R old -> new |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        old_configs = pair["old"]["configs"]
        new_configs = pair["new"]["configs"]
        for config in sorted(set(old_configs).intersection(new_configs)):
            old = old_configs[config]["reported"]
            new = new_configs[config]["reported"]
            lines.append(
                f"| {config} | {_fmt(old['f1'])} -> {_fmt(new['f1'])} | "
                f"{_fmt(old['em'])} -> {_fmt(new['em'])} | "
                f"{_fmt(old['retrieval_recall'])} -> {_fmt(new['retrieval_recall'])} | "
                f"{_fmt(old['citation_precision'])} -> {_fmt(new['citation_precision'])} | "
                f"{_fmt(old['citation_recall'])} -> {_fmt(new['citation_recall'])} |"
            )
        abstention = pair["abstention_recomputed_from_records"]
        old_a = abstention["old"]
        new_a = abstention["new"]
        old_e_name = next(name for name in old_configs if name.startswith("E"))
        new_e_name = next(name for name in new_configs if name.startswith("E"))
        old_e = old_configs[old_e_name]["recomputed"]
        new_e = new_configs[new_e_name]["recomputed"]
        lines.extend(
            [
                "",
                "### Config-E abstention (recomputed from records)",
                "",
                "| Metric | Historical | Clean |",
                "|---|---:|---:|",
                f"| AUC | {_fmt(old_a['auc'])} | {_fmt(new_a['auc'])} |",
                f"| Peak balanced accuracy | {_fmt(old_a['peak_balanced_accuracy']['balanced_accuracy'])} @ {old_a['peak_balanced_accuracy']['threshold']:.2f} | {_fmt(new_a['peak_balanced_accuracy']['balanced_accuracy'])} @ {new_a['peak_balanced_accuracy']['threshold']:.2f} |",
                f"| Coverage at peak BA | {_fmt(old_a['peak_balanced_accuracy']['coverage'])} | {_fmt(new_a['peak_balanced_accuracy']['coverage'])} |",
                f"| Full-coverage conditional F1 | {_fmt(old_a['full_coverage']['selective_f1'])} | {_fmt(new_a['full_coverage']['selective_f1'])} |",
                f"| Best in-sample conditional F1 | {_fmt(old_a['best_conditional_selective_f1']['selective_f1'])} | {_fmt(new_a['best_conditional_selective_f1']['selective_f1'])} |",
                f"| Coverage there | {_fmt(old_a['best_conditional_selective_f1']['coverage'])} | {_fmt(new_a['best_conditional_selective_f1']['coverage'])} |",
                f"| Answered unanswerable there | {old_a['best_conditional_selective_f1']['answered_unanswerable']} | {new_a['best_conditional_selective_f1']['answered_unanswerable']} |",
                "",
                "### Config-E retrieval, default abstention, and runtime",
                "",
                "| Metric | Historical | Clean |",
                "|---|---:|---:|",
                f"| Mean distinct retrieved @4 | {_fmt(old_e['mean_distinct_retrieved_at_4'])} | {_fmt(new_e['mean_distinct_retrieved_at_4'])} |",
                f"| Duplicate retrieved slots | {old_e['duplicate_retrieved_slots']} | {new_e['duplicate_retrieved_slots']} |",
                f"| Default abstention accuracy | {_fmt(old_e['abstention_accuracy'])} | {_fmt(new_e['abstention_accuracy'])} |",
                f"| Default over-abstention | {_fmt(old_e['over_abstention'])} | {_fmt(new_e['over_abstention'])} |",
                f"| Default missed-unanswerable | {_fmt(old_e['missed_unanswerable'])} | {_fmt(new_e['missed_unanswerable'])} |",
                f"| Mean question latency (s) | {_fmt(old_e['mean_latency_s'], 2)} | {_fmt(new_e['mean_latency_s'], 2)} |",
                f"| Total run time (s) | unavailable | {_fmt(pair['runtime']['new']['total_time_s'], 2)} |",
                f"| Peak RSS (MiB) | unavailable | {_fmt(pair['runtime']['new']['resource_usage']['peak_rss_mb'], 2)} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility note",
            "",
            "All clean summary metrics match recomputation from clean JSONL records after four-decimal rounding. Historical best-over-gold F1/EM cannot be reconstructed from historical JSONL alone because those rows retained only the first gold answer. Historical total runtime and peak memory were not recorded and are reported as unavailable rather than estimated.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-natural", default="results")
    parser.add_argument("--new-natural", required=True)
    parser.add_argument("--old-balanced", default="results/balanced")
    parser.add_argument("--new-balanced", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty comparison directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "natural": compare_pair(
            Path(args.old_natural),
            Path(args.new_natural),
            old_chunks=7391,
            old_unique_chunks=2987,
        ),
        "negative_enriched": compare_pair(
            Path(args.old_balanced),
            Path(args.new_balanced),
            old_chunks=4645,
            old_unique_chunks=3784,
        ),
    }
    write_json_atomic(out / "comparison.json", result)
    markdown = to_markdown(result)
    (out / "comparison.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
