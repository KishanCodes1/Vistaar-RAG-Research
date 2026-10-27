"""End-to-end RAG pipeline: ingest -> index -> retrieve -> read -> cite."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings
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
    raw_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "raw_answer": self.raw_answer or self.answer,
            "abstained": self.abstained,
            "citations": self.citations,
            "scores": self.scores,
            "latency_s": round(self.latency_s, 3),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


class RAGPipeline:
    def __init__(self, settings: Settings, config: RetrievalConfig | None = None):
        self.settings = settings
        self.config = config or RetrievalConfig()
        self.embedder = Embedder.get(settings.embedding_model)
        self.store = Store(self.embedder.dimension, settings.data_dir)
        self._reranker: Reranker | None = None
        self._reader = None
        self._llm: GeminiClient | None = None

    # -- lazily constructed backends -----------------------------------
    @property
    def reader(self):
        if self._reader is None:
            from .reader import LocalReader

            self._reader = LocalReader.get(self.settings.reader_model)
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
            self._reranker = Reranker.get(self.settings.reranker_model)
        return self._reranker

    # -- ingestion -----------------------------------------------------
    def index_document(self, filename: str, content: bytes, namespace: str = "default") -> dict:
        document = extract_document(filename, content)
        chunks = chunk_document(
            document, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("Document produced no usable chunks.")

        started = time.monotonic()
        vectors = self.embedder.encode([c.text for c in chunks])
        self.store.get(namespace).add(chunks, vectors)
        try:
            self.store.persist(namespace)
        except Exception:
            pass

        return {
            "doc_id": document.doc_id,
            "filename": filename,
            "pages": len(document.pages),
            "chunks": len(chunks),
            "namespace": namespace,
            "index_time_s": round(time.monotonic() - started, 3),
        }

    # -- retrieval -----------------------------------------------------
    def retrieve(self, question: str, namespace: str = "default", config: RetrievalConfig | None = None) -> list[Hit]:
        config = config or self.config
        ns = self.store.get(namespace)
        if ns.size == 0:
            return []

        vector = self.embedder.encode([question])[0]
        dense = ns.dense(vector, config.top_k * 3)

        if config.use_bm25:
            sparse = ns.bm25(question, config.top_k * 3)
            fused = reciprocal_rank_fusion([dense, sparse], limit=config.top_k * 2)
        else:
            fused = dense

        if config.use_rerank:
            reranker = self._get_reranker()
            if reranker is not None:
                fused = reranker.rerank(question, fused, config.rerank_top_n)
            else:
                fused = fused[: config.rerank_top_n]
        else:
            fused = fused[: config.top_k]

        return fused

    # -- answering -----------------------------------------------------
    def answer(self, question: str, namespace: str = "default", config: RetrievalConfig | None = None, hits: list[Hit] | None = None) -> Answer:
        config = config or self.config
        started = time.monotonic()
        if hits is None:
            hits = self.retrieve(question, namespace, config)

        if not hits:
            return Answer(
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                citations=[],
                scores={"confidence": 0.0, "retrieval": 0.0, "passages": 0},
                latency_s=time.monotonic() - started,
            )

        if self.settings.reader == "local":
            return self._answer_local(question, hits, started, config)
        return self._answer_gemini(question, hits, started, config)

    def _answer_local(self, question: str, hits: list[Hit], started: float, config: RetrievalConfig) -> Answer:
        output = self.reader.answer(question, hits, max_passages=config.rerank_top_n)
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
        )

    def _answer_gemini(self, question: str, hits: list[Hit], started: float, config: RetrievalConfig) -> Answer:
        best_dense = max((h.score for h in hits if h.source == "dense"), default=0.0)
        best_any = max((h.score for h in hits), default=0.0)

        if config.use_abstention and hits[0].source == "dense" and best_dense < self.settings.min_retrieval_score:
            return Answer(
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                citations=[],
                scores={"confidence": 0.0, "retrieval": round(best_dense, 5), "passages": len(hits)},
                latency_s=time.monotonic() - started,
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
        return Answer(
            answer=ABSTAIN_MESSAGE if abstained else parsed["answer"],
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
        )
