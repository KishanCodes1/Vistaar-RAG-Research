"""Run the evaluation matrix and emit results.

Usage:
    python -m eval.run_eval --papers 40 --limit 0
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vistaar.config import Settings
from vistaar.pipeline import RAGPipeline, RetrievalConfig

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

CONFIGS = [
    RetrievalConfig(name="A: dense", use_bm25=False, use_rerank=False, use_abstention=False),
    RetrievalConfig(name="B: dense+abstain", use_bm25=False, use_rerank=False, use_abstention=True),
    RetrievalConfig(name="C: hybrid", use_bm25=True, use_rerank=False, use_abstention=False),
    RetrievalConfig(name="D: hybrid+rerank", use_bm25=True, use_rerank=True, use_abstention=False),
    RetrievalConfig(name="E: full (hybrid+rerank+abstain)", use_bm25=True, use_rerank=True, use_abstention=True),
]

CACHE_DIR = Path("results/cache")

# The reader always receives this many passages, regardless of the retriever's
# top_k. Retrieval metrics are therefore computed at the same depth for every
# configuration, so the ablation isolates *which* passages are chosen.
READER_PASSAGES = 4


def _cache_path(config_name: str) -> Path:
    safe = config_name.split(":")[0].strip()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{safe}.jsonl"


def load_cache(config_name: str) -> dict[str, dict]:
    path = _cache_path(config_name)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            out[record["question_id"]] = record
        except json.JSONDecodeError:
            continue
    return out


def append_cache(config_name: str, record: dict) -> None:
    with open(_cache_path(config_name), "a") as fh:
        fh.write(json.dumps(record) + "\n")


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
    parser.add_argument("--papers", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="cap questions for a smoke run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results")
    parser.add_argument("--only", default="", help="run only configs whose label starts with this prefix")
    parser.add_argument("--balanced", action="store_true",
                        help="keep only documents that contain at least one unanswerable question")
    args = parser.parse_args()

    settings = Settings.from_env()
    items = load_qasper("data/raw/qasper_validation.parquet", max_papers=args.papers)
    items, dropped = prepare_items(items)

    if args.balanced:
        # QASPER validation is ~96% answerable, so abstention accuracy is
        # dominated by the base rate. Restricting to documents that contain at
        # least one unanswerable question gives a usable class balance.
        docs_with_negatives = {i.doc_id for i in items if not i.answerable}
        items = [i for i in items if i.doc_id in docs_with_negatives]
    if args.limit:
        # keep a mix of answerable and unanswerable
        unans = [i for i in items if not i.answerable]
        ans = [i for i in items if i.answerable]
        half = max(args.limit // 2, 1)
        items = ans[:half] + unans[: args.limit - half]

    n_unans = sum(1 for i in items if not i.answerable)
    print(f"papers={args.papers} questions={len(items)} "
          f"(answerable={len(items)-n_unans}, unanswerable={n_unans}, dropped_no_gold={dropped})")

    pipeline = RAGPipeline(settings)

    # Index each unique document once, shared across all configs.
    docs: dict[str, str] = {}
    for item in items:
        docs.setdefault(item.doc_id, item.doc_text)
    print(f"indexing {len(docs)} documents ...")
    started = time.monotonic()
    for doc_id, text in docs.items():
        pipeline.index_document(f"{doc_id}.txt", text.encode(), namespace=doc_id)
    total_chunks = sum(pipeline.store.get(d).size for d in docs)
    print(f"indexed {total_chunks} chunks in {time.monotonic() - started:.1f}s")

    configs = [c for c in CONFIGS if c.name.startswith(args.only)] if args.only else CONFIGS
    results: dict[str, ConfigResult] = {c.name: ConfigResult(name=c.name) for c in configs}
    raw_records: list[dict] = []

    for config in configs:
        cache = load_cache(config.name)
        todo = [i for i in items if i.question_id not in cache]
        print(f"\n[{config.name}] cached={len(items)-len(todo)} todo={len(todo)}")

        def work(item: EvalItem) -> dict:
            hits = pipeline.retrieve(item.question, namespace=item.doc_id, config=config)
            answer = pipeline.answer(item.question, namespace=item.doc_id, config=config, hits=hits)
            return {
                "config": config.name,
                "question_id": item.question_id,
                "doc_id": item.doc_id,
                "question": item.question,
                "gold": item.gold,
                "answerable": item.answerable,
                "prediction": answer.answer,
                "raw_prediction": answer.raw_answer or answer.answer,
                "abstained": answer.abstained,
                "citations": [c["text"] for c in answer.citations],
                "retrieved": [h.chunk.text for h in hits],
                "scores": answer.scores,
                "latency_s": answer.latency_s,
                "model": answer.model,
                "gold_evidence": item.gold_evidence,
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
                append_cache(config.name, record)
                cache[item.question_id] = record
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(todo)}")

        for item in items:
            record = cache.get(item.question_id)
            if record is None:
                continue
            raw_records.append(record)
            r = results[config.name]

            if item.answerable:
                r.f1.append(best_f1(record.get("raw_prediction") or record["prediction"], item.gold_answers))
                r.em.append(exact_match(record.get("raw_prediction") or record["prediction"], item.gold_answers))
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"papers": args.papers, "questions": len(items), "results": rows}, indent=2))
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in raw_records))
    table = to_markdown(rows)
    (out_dir / "table.md").write_text(table)

    print("\n" + table)
    print(f"\nwrote {out_dir}/summary.json, {out_dir}/table.md, {out_dir}/records.jsonl")


if __name__ == "__main__":
    main()
