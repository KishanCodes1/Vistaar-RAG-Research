"""Run the evaluation matrix and emit results.

Usage:
    python -m eval.run_eval --papers 40 --limit 0
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from vistaar.config import Settings
from vistaar.diagnostics import CONFIDENCE_DIAGNOSTICS_SCHEMA
from vistaar.generate import SYSTEM_PROMPT
from vistaar.pipeline import RAGPipeline, RetrievalConfig
from vistaar.reader import MAX_ANSWER_TOKENS, MAX_LENGTH, STRIDE

from .metrics import (
    ConfigResult,
    best_f1,
    citation_scores,
    exact_match,
    retrieval_recall,
    summarize,
    to_markdown,
)
from .qasper import EvalItem, load_qasper
from .provenance import (
    EvaluationCache,
    config_fingerprint,
    evaluation_item_fingerprint,
    hash_file,
    stable_fingerprint,
    write_json_atomic,
)

CONFIGS = [
    RetrievalConfig(name="A: dense", use_bm25=False, use_rerank=False, use_abstention=False),
    RetrievalConfig(name="B: dense+abstain", use_bm25=False, use_rerank=False, use_abstention=True),
    RetrievalConfig(name="C: hybrid", use_bm25=True, use_rerank=False, use_abstention=False),
    RetrievalConfig(name="D: hybrid+rerank", use_bm25=True, use_rerank=True, use_abstention=False),
    RetrievalConfig(name="E: full (hybrid+rerank+abstain)", use_bm25=True, use_rerank=True, use_abstention=True),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path("data/raw/qasper_validation.parquet")
DEFAULT_CACHE_ROOT = Path("results/cache/eval")
RUN_SCHEMA = "vistaar-eval-run-v1"

# The reader always receives this many passages, regardless of the retriever's
# top_k. Retrieval metrics are therefore computed at the same depth for every
# configuration, so the ablation isolates *which* passages are chosen.
READER_PASSAGES = 4


IMPLEMENTATION_FILES = (
    "eval/qasper.py",
    "eval/run_eval.py",
    "eval/provenance.py",
    "vistaar/config.py",
    "vistaar/diagnostics.py",
    "vistaar/generate.py",
    "vistaar/index.py",
    "vistaar/pipeline.py",
    "vistaar/reader.py",
    "vistaar/text.py",
)


def _package_versions() -> dict[str, str]:
    packages = ("faiss-cpu", "numpy", "sentence-transformers", "torch", "transformers")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _resource_usage() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports KiB.
    peak_rss_mb = (
        usage.ru_maxrss / (1024 * 1024)
        if sys.platform == "darwin"
        else usage.ru_maxrss / 1024
    )
    return {
        "user_cpu_s": round(usage.ru_utime, 3),
        "system_cpu_s": round(usage.ru_stime, 3),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }


def build_run_payload(
    *,
    dataset_path: Path,
    items: list[EvalItem],
    settings: Settings,
    configs: list[RetrievalConfig],
    papers: int,
    limit: int,
    negative_enriched: bool,
    workers: int,
) -> dict:
    """Return every deterministic input that can affect records or metrics."""
    item_fingerprints = {
        item.question_id: evaluation_item_fingerprint(item) for item in items
    }
    docs: dict[str, dict] = {}
    for item in items:
        entry = docs.setdefault(
            item.doc_id,
            {
                "doc_id": item.doc_id,
                "content_sha256": hashlib.sha256(item.doc_text.encode()).hexdigest(),
                "question_ids": [],
                "answerable": 0,
                "unanswerable": 0,
            },
        )
        entry["question_ids"].append(item.question_id)
        entry["answerable" if item.answerable else "unanswerable"] += 1

    implementation = {
        relative: hash_file(REPO_ROOT / relative) for relative in IMPLEMENTATION_FILES
    }
    return {
        "schema": RUN_SCHEMA,
        "dataset": {
            "name": "qasper-validation",
            "filename": dataset_path.name,
            "sha256": hash_file(dataset_path),
            "selection": {
                "max_papers": papers,
                "limit": limit,
                "mode": "negative_enriched" if negative_enriched else "natural_file_order",
                "question_ids_in_order": [item.question_id for item in items],
                "answerable": sum(item.answerable for item in items),
                "unanswerable": sum(not item.answerable for item in items),
            },
        },
        "documents": [docs[key] for key in sorted(docs)],
        "items": [
            {"question_id": item.question_id, "fingerprint": item_fingerprints[item.question_id]}
            for item in items
        ],
        "chunking": {
            "size": settings.chunk_size,
            "overlap": settings.chunk_overlap,
            "implementation_sha256": implementation["vistaar/text.py"],
        },
        "embedding": {
            "model": settings.embedding_model,
            "revision": settings.embedding_revision,
            "normalise_embeddings": True,
        },
        "retrieval": [
            {**asdict(config), "fingerprint": config_fingerprint(config)}
            for config in configs
        ],
        "reranker": {
            "model": settings.reranker_model,
            "revision": settings.reranker_revision,
        },
        "reader": {
            "backend": settings.reader,
            "model": settings.reader_model,
            "revision": settings.reader_revision,
            "threshold": settings.reader_threshold,
            "max_length": MAX_LENGTH,
            "stride": STRIDE,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
        },
        "generation": {
            "model": settings.model,
            "fallback_models": list(settings.fallback_models),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "temperature": 0.0,
            "thinking_budget": 0,
            "max_output_tokens": 1024,
        },
        "execution": {"workers": workers, "reader_passages": READER_PASSAGES},
        "diagnostics": {
            "schema_version": CONFIDENCE_DIAGNOSTICS_SCHEMA,
            "enabled_for_records": True,
            "inference_time_only": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "implementation": implementation,
        "packages": _package_versions(),
    }


def _new_output_directory(requested: str, run_fingerprint: str) -> Path:
    if requested:
        output = Path(requested)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results/runs") / f"{stamp}-{run_fingerprint[:12]}"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory {output}. "
            "Choose a new --out path."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_items(items: list[EvalItem]) -> tuple[list[EvalItem], int]:
    """Keep items we can score: unanswerable ones, and answerable ones with gold text."""
    kept, dropped = [], 0
    for item in items:
        if not item.answerable:
            kept.append(item)
        elif item.gold_answers:
            kept.append(item)
        else:
            dropped += 1
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--papers", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="cap questions for a smoke run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out",
        default="",
        help="new output directory; defaults to a timestamped results/runs directory",
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--only", default="", help="run only configs whose label starts with this prefix")
    parser.add_argument("--balanced", action="store_true",
                        help="negative-enriched stress slice: documents with an unanswerable question")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    dataset_path = Path(args.dataset)
    settings = Settings.from_env()
    items = load_qasper(dataset_path, max_papers=args.papers)
    items, dropped = prepare_items(items)

    if args.balanced:
        # Label-conditioned stress slice. This is negative-enriched, not a
        # representative or exactly class-balanced deployment sample.
        docs_with_negatives = {i.doc_id for i in items if not i.answerable}
        items = [i for i in items if i.doc_id in docs_with_negatives]
    if args.limit:
        # keep a mix of answerable and unanswerable
        unans = [i for i in items if not i.answerable]
        ans = [i for i in items if i.answerable]
        half = max(args.limit // 2, 1)
        items = ans[:half] + unans[: args.limit - half]

    if not items:
        raise SystemExit("Dataset selection produced no evaluable questions")
    configs = [c for c in CONFIGS if c.name.startswith(args.only)] if args.only else CONFIGS
    if not configs:
        raise SystemExit(f"No configurations start with {args.only!r}")

    n_unans = sum(not item.answerable for item in items)
    print(f"papers={args.papers} questions={len(items)} "
          f"(answerable={len(items)-n_unans}, unanswerable={n_unans}, dropped_no_gold={dropped})")

    run_payload = build_run_payload(
        dataset_path=dataset_path,
        items=items,
        settings=settings,
        configs=configs,
        papers=args.papers,
        limit=args.limit,
        negative_enriched=args.balanced,
        workers=args.workers,
    )
    run_fingerprint = stable_fingerprint(run_payload)
    cache = EvaluationCache(args.cache_root, run_fingerprint)
    base_manifest = {
        "schema": RUN_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprinted_inputs": run_payload,
    }
    cached_manifest_path = cache.run_dir / "run_manifest.json"
    if cached_manifest_path.exists():
        cached_manifest = json.loads(cached_manifest_path.read_text())
        if cached_manifest.get("fingerprinted_inputs") != run_payload:
            raise RuntimeError("run fingerprint collision or corrupt cached manifest")
    else:
        write_json_atomic(cached_manifest_path, base_manifest)

    out_dir = _new_output_directory(args.out, run_fingerprint)
    write_json_atomic(out_dir / "run_manifest.json", {**base_manifest, "status": "running"})
    isolated_settings = replace(settings, data_dir=cache.run_dir / "index")
    pipeline = RAGPipeline(isolated_settings)
    run_started = time.monotonic()

    # Index each unique document once, shared across all configs.
    docs: dict[str, str] = {}
    for item in items:
        docs.setdefault(item.doc_id, item.doc_text)
    print(f"indexing {len(docs)} documents ...")
    started = time.monotonic()
    document_index_stats = []
    for doc_id, text in docs.items():
        stats = pipeline.index_document(f"{doc_id}.txt", text.encode(), namespace=doc_id)
        namespace = pipeline.store.get(doc_id)
        chunk_ids = [chunk.chunk_id for chunk in namespace.all_chunks()]
        if namespace.size != stats["chunks"] or len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError(
                f"index integrity failure for {doc_id}: expected {stats['chunks']} fresh "
                f"chunks, found {namespace.size} ({len(set(chunk_ids))} unique)"
            )
        document_index_stats.append(
            {
                "namespace": doc_id,
                "document_id": stats["doc_id"],
                "chunks": stats["chunks"],
                "unique_chunks": len(set(chunk_ids)),
                "replaced_chunks": stats["replaced_chunks"],
            }
        )
    total_chunks = sum(pipeline.store.get(d).size for d in docs)
    unique_chunks = sum(
        len({chunk.chunk_id for chunk in pipeline.store.get(doc_id).all_chunks()})
        for doc_id in docs
    )
    index_time = time.monotonic() - started
    if total_chunks != unique_chunks:
        raise RuntimeError(
            f"index integrity failure: {total_chunks} chunks but {unique_chunks} unique IDs"
        )
    index_report = {
        "documents": len(docs),
        "chunks": total_chunks,
        "unique_chunks": unique_chunks,
        "duplicate_chunks": total_chunks - unique_chunks,
        "index_time_s": round(index_time, 3),
        "data_dir": str(isolated_settings.data_dir),
        "per_document": document_index_stats,
    }
    print(f"indexed {total_chunks} unique chunks in {index_time:.1f}s")

    results: dict[str, ConfigResult] = {c.name: ConfigResult(name=c.name) for c in configs}
    raw_records: list[dict] = []
    item_fingerprints = {
        item.question_id: evaluation_item_fingerprint(item) for item in items
    }
    cache_report = {}
    evaluation_started = time.monotonic()

    for config in configs:
        config_fp = config_fingerprint(config)
        loaded = cache.load(config.name, config_fp, item_fingerprints)
        records_by_id = dict(loaded.records)
        todo = [item for item in items if item.question_id not in records_by_id]
        print(
            f"\n[{config.name}] cached={len(items)-len(todo)} todo={len(todo)} "
            f"rejected={loaded.rejected} malformed={loaded.malformed}"
        )

        def work(item: EvalItem) -> dict:
            diagnostics: dict = {}
            hits = pipeline.retrieve(
                item.question,
                namespace=item.doc_id,
                config=config,
                diagnostics=diagnostics,
            )
            answer = pipeline.answer(
                item.question,
                namespace=item.doc_id,
                config=config,
                hits=hits,
                diagnostics=diagnostics,
            )
            return {
                "config": config.name,
                "question_id": item.question_id,
                "doc_id": item.doc_id,
                "question": item.question,
                "gold": item.gold,
                "gold_answers": list(item.gold_answers),
                "answerable": item.answerable,
                "prediction": answer.answer,
                "raw_prediction": (
                    answer.answer if answer.raw_answer is None else answer.raw_answer
                ),
                "abstained": answer.abstained,
                "citations": [c["text"] for c in answer.citations],
                "citation_details": list(answer.citations),
                "retrieved": [h.chunk.text for h in hits],
                "retrieved_chunks": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "doc_id": hit.chunk.doc_id,
                        "page": hit.chunk.page,
                        "section": hit.chunk.section,
                        "text": hit.chunk.text,
                        "score": hit.score,
                        "source": hit.source,
                    }
                    for hit in hits
                ],
                "scores": answer.scores,
                "latency_s": answer.latency_s,
                "model": answer.model,
                "gold_evidence": item.gold_evidence,
                "diagnostics": answer.diagnostics,
            }

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, i): i for i in todo}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    results[config.name].errors += 1
                    print(f"  ERROR {item.question_id}: {exc}")
                    continue
                record = cache.append(
                    config.name,
                    config_fp,
                    item_fingerprints[item.question_id],
                    record,
                )
                records_by_id[item.question_id] = record
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(todo)}")

        cache_report[config.name] = {
            "config_fingerprint": config_fp,
            "hits": loaded.accepted,
            "misses": len(todo),
            "rejected": loaded.rejected,
            "malformed": loaded.malformed,
            "cache_file": str(cache.path_for(config.name, config_fp)),
        }

        for item in items:
            record = records_by_id.get(item.question_id)
            if record is None:
                continue
            raw_records.append(record)
            r = results[config.name]

            if item.answerable:
                raw_prediction = (
                    record["raw_prediction"]
                    if "raw_prediction" in record
                    else record["prediction"]
                )
                r.f1.append(best_f1(raw_prediction, item.gold_answers))
                r.em.append(exact_match(raw_prediction, item.gold_answers))
                r.abstain_on_answerable.append(1.0 if record["abstained"] else 0.0)
            else:
                r.answer_on_unanswerable.append(0.0 if record["abstained"] else 1.0)

            r.abstention_correct.append(
                1.0 if record["abstained"] == (not item.answerable) else 0.0
            )
            precision, recall, _ = citation_scores(record["citations"], item.gold_evidence)
            r.citation_precision.append(precision)
            r.citation_recall.append(recall)
            r.retrieval_recall.append(
                retrieval_recall(record["retrieved"][:READER_PASSAGES], item.gold_evidence)
            )
            r.latencies.append(record["latency_s"])

    rows = summarize(results)
    evaluation_time = time.monotonic() - evaluation_started
    runtime_report = {
        "workers": args.workers,
        "index_time_s": round(index_time, 3),
        "evaluation_time_s": round(evaluation_time, 3),
        "total_time_s": round(time.monotonic() - run_started, 3),
        "resource_usage": _resource_usage(),
    }
    summary = {
        "run_fingerprint": run_fingerprint,
        "papers_requested": args.papers,
        "documents": len(docs),
        "questions": len(items),
        "answerable": len(items) - n_unans,
        "unanswerable": n_unans,
        "dropped_no_gold": dropped,
        "selection": "negative_enriched" if args.balanced else "natural_file_order",
        "index": {key: value for key, value in index_report.items() if key != "per_document"},
        "cache": cache_report,
        "runtime": runtime_report,
        "results": rows,
    }
    write_json_atomic(out_dir / "summary.json", summary)
    (out_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in raw_records)
    )
    table = to_markdown(rows)
    (out_dir / "table.md").write_text(table)
    write_json_atomic(
        out_dir / "run_manifest.json",
        {
            **base_manifest,
            "status": "complete",
            "index": index_report,
            "cache": cache_report,
            "runtime": runtime_report,
            "record_count": len(raw_records),
        },
    )

    print("\n" + table)
    print(f"\nrun fingerprint: {run_fingerprint}")
    print(f"wrote {out_dir}/run_manifest.json, summary.json, table.md, records.jsonl")


if __name__ == "__main__":
    main()
