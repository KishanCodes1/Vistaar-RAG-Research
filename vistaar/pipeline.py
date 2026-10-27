"""End-to-end RAG pipeline: ingest -> index -> retrieve -> read -> cite."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings
from .diagnostics import (
    CONFIDENCE_DIAGNOSTICS_SCHEMA,
    ranked_hits,
    score_summary,
    snapshot_hits,
    stage,
)
from .generate import GeminiClient, parse_answer
from .index import Embedder, Hit, Reranker, Store, reciprocal_rank_fusion
from .text import chunk_document, extract_document

ABSTAIN_MESSAGE = "I could not find this in the document."


@dataclass(frozen=True)
class RetrievalConfig:
    """A named retrieval strategy. Ablations are expressed as instances."""

    name: str = "hybrid+rerank"
    use_bm25: bool = True
    use_rerank: bool = True
    use_abstention: bool = True
    top_k: int = 8
    rerank_top_n: int = 4


@dataclass
class Answer:
    answer: str
    abstained: bool
    citations: list[dict]
    scores: dict
    latency_s: float
    model: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw_answer: str | None = None
    diagnostics: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "answer": self.answer,
            "raw_answer": self.answer if self.raw_answer is None else self.raw_answer,
            "abstained": self.abstained,
            "citations": self.citations,
            "scores": self.scores,
            "latency_s": round(self.latency_s, 3),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        return payload


class RAGPipeline:
    def __init__(self, settings: Settings, config: RetrievalConfig | None = None):
        self.settings = settings
        self.config = config or RetrievalConfig()
        self.embedder = Embedder.get(
            settings.embedding_model, settings.embedding_revision
        )
        self.store = Store(self.embedder.dimension, settings.data_dir)
        self._reranker: Reranker | None = None
        self._reader = None
        self._llm: GeminiClient | None = None

    # -- lazily constructed backends -----------------------------------
    @property
    def reader(self):
        if self._reader is None:
            from .reader import LocalReader

            self._reader = LocalReader.get(
                self.settings.reader_model, self.settings.reader_revision
            )
        return self._reader

    @property
    def llm(self) -> GeminiClient:
        if self._llm is None:
            self._llm = GeminiClient(
                api_key=self.settings.api_key,
                model=self.settings.model,
                fallback_models=self.settings.fallback_models,
                timeout=self.settings.request_timeout,
            )
        return self._llm

    def _get_reranker(self) -> Reranker | None:
        if self._reranker is None:
            self._reranker = Reranker.get(
                self.settings.reranker_model, self.settings.reranker_revision
            )
        return self._reranker

    # -- ingestion -----------------------------------------------------
    def index_document(self, filename: str, content: bytes, namespace: str = "default") -> dict:
        """Index or replace one deterministic document within ``namespace``.

        Re-indexing identical filename/content is idempotent. A replacement is
        scoped to the extracted document ID and never clears other documents.
        """
        document = extract_document(filename, content)
        chunks = chunk_document(
            document, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("Document produced no usable chunks.")

        started = time.monotonic()
        vectors = self.embedder.encode([c.text for c in chunks])
        target = self.store.get(namespace)
        replaced_chunks = target.upsert_document(chunks, vectors)
        self.store.persist(namespace)

        return {
            "doc_id": document.doc_id,
            "filename": filename,
            "pages": len(document.pages),
            "chunks": len(chunks),
            "namespace_chunks": target.size,
            "replaced_chunks": replaced_chunks,
            "namespace": namespace,
            "index_time_s": round(time.monotonic() - started, 3),
        }

    # -- retrieval -----------------------------------------------------
    def retrieve(
        self,
        question: str,
        namespace: str = "default",
        config: RetrievalConfig | None = None,
        diagnostics: dict | None = None,
    ) -> list[Hit]:
        config = config or self.config
        ns = self.store.get(namespace)
        if ns.size == 0:
            if diagnostics is not None:
                self._record_retrieval_diagnostics(
                    diagnostics,
                    namespace=namespace,
                    config=config,
                    dense=[],
                    sparse=[],
                    fused=[],
                    reranked=None,
                    selected=[],
                    reranker_available=None,
                    empty_reason="empty_namespace",
                )
            return []

        vector = self.embedder.encode([question])[0]
        dense = ns.dense(vector, config.top_k * 3)
        sparse: list[Hit] = []

        if config.use_bm25:
            sparse = ns.bm25(question, config.top_k * 3)
            fused = reciprocal_rank_fusion([dense, sparse], limit=config.top_k * 2)
        else:
            fused = dense

        fused_candidates = snapshot_hits(fused) if diagnostics is not None else None
        reranked_candidates: list[Hit] | None = None
        reranker_available: bool | None = None
        if config.use_rerank:
            reranker = self._get_reranker()
            if reranker is not None:
                reranker_available = bool(getattr(reranker, "available", False))
                scored = [] if diagnostics is not None else None
                fused = reranker.rerank(
                    question,
                    fused,
                    config.rerank_top_n,
                    scored_candidates=scored,
                )
                reranked_candidates = scored
            else:
                reranker_available = False
                fused = fused[: config.rerank_top_n]
        else:
            fused = fused[: config.top_k]

        if diagnostics is not None:
            self._record_retrieval_diagnostics(
                diagnostics,
                namespace=namespace,
                config=config,
                dense=dense,
                sparse=sparse,
                fused=fused_candidates or [],
                reranked=reranked_candidates,
                selected=fused,
                reranker_available=reranker_available,
            )
        return fused

    def _record_retrieval_diagnostics(
        self,
        diagnostics: dict,
        *,
        namespace: str,
        config: RetrievalConfig,
        dense: list[Hit],
        sparse: list[Hit],
        fused: list[Hit],
        reranked: list[Hit] | None,
        selected: list[Hit],
        reranker_available: bool | None,
        empty_reason: str | None = None,
        upstream_available: bool = True,
    ) -> None:
        """Populate a side channel without changing any candidate list."""
        dense_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(dense, 1)}
        bm25_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(sparse, 1)}
        fused_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(fused, 1)}
        diagnostics.clear()
        diagnostics.update(
            {
                "schema_version": CONFIDENCE_DIAGNOSTICS_SCHEMA,
                "retrieval": {
                    "namespace": namespace,
                    "config": {
                        "name": config.name,
                        "use_bm25": config.use_bm25,
                        "use_rerank": config.use_rerank,
                        "use_abstention": config.use_abstention,
                        "top_k": config.top_k,
                        "rerank_top_n": config.rerank_top_n,
                    },
                    "limits": {
                        "dense": config.top_k * 3,
                        "bm25": config.top_k * 3,
                        "rrf": config.top_k * 2,
                        "reranker_output": config.rerank_top_n,
                    },
                    "stages": {
                        "dense": stage(
                            requested=True,
                            available=upstream_available,
                            hits=dense,
                            reason=empty_reason,
                        ),
                        "bm25": stage(
                            requested=config.use_bm25,
                            available=config.use_bm25 and upstream_available,
                            hits=sparse,
                            reason=empty_reason if config.use_bm25 else "not_requested",
                        ),
                        "rrf": stage(
                            requested=config.use_bm25,
                            available=config.use_bm25 and upstream_available,
                            hits=fused,
                            reason=empty_reason if config.use_bm25 else "not_requested",
                            dense_ranks=dense_ranks,
                            bm25_ranks=bm25_ranks,
                        ),
                        "reranker": stage(
                            requested=config.use_rerank,
                            available=bool(reranker_available),
                            hits=reranked,
                            reason=(
                                empty_reason
                                if empty_reason and config.use_rerank
                                else (
                                    "model_unavailable"
                                    if config.use_rerank and not reranker_available
                                    else "not_requested"
                                )
                            ),
                            dense_ranks=dense_ranks,
                            bm25_ranks=bm25_ranks,
                            fused_ranks=fused_ranks,
                        ),
                        "selected_context": {
                            "requested": True,
                            "available": True,
                            "reason": empty_reason,
                            "candidates": ranked_hits(
                                selected,
                                dense_ranks=dense_ranks,
                                bm25_ranks=bm25_ranks,
                                fused_ranks=fused_ranks,
                            ),
                            "score_summary": score_summary(selected),
                        },
                    },
                    "features": {
                        "dense_bm25_overlap_count": (
                            len(set(dense_ranks) & set(bm25_ranks))
                            if config.use_bm25 and upstream_available
                            else None
                        ),
                        "selected_chunk_count": len(selected),
                    },
                },
                "reader": None,
                "decision": None,
            }
        )

    # -- answering -----------------------------------------------------
    def answer(
        self,
        question: str,
        namespace: str = "default",
        config: RetrievalConfig | None = None,
        hits: list[Hit] | None = None,
        diagnostics: dict | None = None,
    ) -> Answer:
        config = config or self.config
        started = time.monotonic()
        if hits is None:
            hits = self.retrieve(question, namespace, config, diagnostics=diagnostics)
        elif diagnostics is not None and "schema_version" not in diagnostics:
            self._record_retrieval_diagnostics(
                diagnostics,
                namespace=namespace,
                config=config,
                dense=[],
                sparse=[],
                fused=[],
                reranked=None,
                selected=hits,
                reranker_available=None,
                empty_reason="externally_supplied_hits",
                upstream_available=False,
            )

        if not hits:
            if diagnostics is not None:
                diagnostics["reader"] = {
                    "available": False,
                    "reason": "no_retrieved_passages",
                }
                diagnostics["decision"] = {
                    "backend": self.settings.reader,
                    "abstention_enabled": bool(config.use_abstention),
                    "configured_threshold": (
                        self.settings.reader_threshold
                        if self.settings.reader == "local"
                        else None
                    ),
                    "found": False,
                    "confidence": 0.0,
                    "abstained": True,
                    "triggered_reasons": ["no_retrieved_passages"],
                }
            return Answer(
                answer=ABSTAIN_MESSAGE,
                raw_answer="",
                abstained=True,
                citations=[],
                scores={"confidence": 0.0, "retrieval": 0.0, "passages": 0},
                latency_s=time.monotonic() - started,
                diagnostics=diagnostics,
            )

        if self.settings.reader == "local":
            return self._answer_local(question, hits, started, config, diagnostics)
        return self._answer_gemini(question, hits, started, config, diagnostics)

    def _answer_local(
        self,
        question: str,
        hits: list[Hit],
        started: float,
        config: RetrievalConfig,
        diagnostics: dict | None = None,
    ) -> Answer:
        if diagnostics is None:
            output = self.reader.answer(
                question, hits, max_passages=config.rerank_top_n
            )
        else:
            output = self.reader.answer(
                question,
                hits,
                max_passages=config.rerank_top_n,
                collect_diagnostics=True,
            )
        retrieval_score = max((h.score for h in hits), default=0.0)

        citations = []
        for number in output.passage_ids:
            index = number - 1
            if 0 <= index < len(hits):
                hit = hits[index]
                citations.append(
                    {
                        "id": f"P{number}",
                        "chunk_id": hit.chunk.chunk_id,
                        "page": hit.chunk.page,
                        "text": hit.chunk.text,
                    }
                )

        abstained = bool(config.use_abstention) and (
            (not output.found) or output.confidence < self.settings.reader_threshold
        )
        if diagnostics is not None:
            diagnostics["reader"] = (
                output.diagnostics.to_dict()
                if output.diagnostics is not None
                else {
                    "available": False,
                    "reason": "reader_diagnostics_unavailable",
                }
            )
            conditions = []
            if not output.found:
                conditions.append("reader_no_span")
            if output.confidence < self.settings.reader_threshold:
                conditions.append("below_configured_threshold")
            diagnostics["decision"] = {
                "backend": "local",
                "abstention_enabled": bool(config.use_abstention),
                "configured_threshold": self.settings.reader_threshold,
                "found": output.found,
                "confidence": round(output.confidence, 5),
                "abstained": abstained,
                "triggered_reasons": conditions if abstained else [],
                "counterfactual_conditions": conditions,
            }
        return Answer(
            answer=ABSTAIN_MESSAGE if abstained else output.answer,
            raw_answer=output.answer,
            abstained=abstained,
            citations=citations,
            scores={
                # Raw reader confidence is recorded so the abstention
                # threshold can be swept offline for risk-coverage curves.
                "confidence": round(output.confidence, 5),
                "retrieval": round(retrieval_score, 5),
                "passages": len(hits),
            },
            latency_s=time.monotonic() - started,
            model=self.settings.reader_model,
            diagnostics=diagnostics,
        )

    def _answer_gemini(
        self,
        question: str,
        hits: list[Hit],
        started: float,
        config: RetrievalConfig,
        diagnostics: dict | None = None,
    ) -> Answer:
        best_dense = max((h.score for h in hits if h.source == "dense"), default=0.0)
        best_any = max((h.score for h in hits), default=0.0)

        if config.use_abstention and hits[0].source == "dense" and best_dense < self.settings.min_retrieval_score:
            if diagnostics is not None:
                diagnostics["reader"] = {
                    "available": False,
                    "reason": "hosted_reader_skipped_by_dense_threshold",
                }
                diagnostics["decision"] = {
                    "backend": "gemini",
                    "abstention_enabled": True,
                    "configured_threshold": self.settings.min_retrieval_score,
                    "found": None,
                    "confidence": None,
                    "abstained": True,
                    "triggered_reasons": ["dense_retrieval_below_threshold"],
                }
            return Answer(
                answer=ABSTAIN_MESSAGE,
                raw_answer="",
                abstained=True,
                citations=[],
                scores={"confidence": 0.0, "retrieval": round(best_dense, 5), "passages": len(hits)},
                latency_s=time.monotonic() - started,
                diagnostics=diagnostics,
            )

        labelled = [f"[P{i}] (page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)]
        prompt = (
            f"Context passages:\n\n" + "\n\n".join(labelled)
            + f"\n\nQuestion: {question}\n\nAnswer with JSON only."
        )

        completion = self.llm.complete(prompt)
        parsed = parse_answer(completion.text)

        by_label = {f"P{i}": h for i, h in enumerate(hits, start=1)}
        citations = []
        for ref in parsed["citations"]:
            key = ref.strip().upper().lstrip("[").rstrip("]")
            hit = by_label.get(key)
            if hit:
                citations.append(
                    {
                        "id": key,
                        "chunk_id": hit.chunk.chunk_id,
                        "page": hit.chunk.page,
                        "text": hit.chunk.text,
                    }
                )

        abstained = not parsed["found"] and bool(config.use_abstention)
        if diagnostics is not None:
            diagnostics["reader"] = {
                "available": False,
                "reason": "hosted_reader_has_no_span_null_diagnostics",
            }
            diagnostics["decision"] = {
                "backend": "gemini",
                "abstention_enabled": bool(config.use_abstention),
                "configured_threshold": None,
                "found": parsed["found"],
                "confidence": 1.0 if parsed["found"] else 0.0,
                "abstained": abstained,
                "triggered_reasons": (
                    ["model_reported_not_found"] if abstained else []
                ),
            }
        return Answer(
            answer=ABSTAIN_MESSAGE if abstained else parsed["answer"],
            raw_answer=parsed["answer"],
            abstained=abstained,
            citations=citations,
            scores={
                "confidence": 1.0 if parsed["found"] else 0.0,
                "retrieval": round(best_any, 5),
                "dense": round(best_dense, 5),
                "passages": len(hits),
            },
            latency_s=time.monotonic() - started,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            output_tokens=completion.output_tokens,
            diagnostics=diagnostics,
        )
