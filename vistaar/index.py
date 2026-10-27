"""Embedding, indexing and hybrid retrieval.

Namespaced so that one tenant's documents are never visible to another,
which was the most serious defect in the original single global index.
"""
from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .text import Chunk


class Embedder:
    """Lazy singleton around a sentence-transformers model."""

    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = self.model.get_embedding_dimension()

    @classmethod
    def get(cls, model_name: str) -> "Embedder":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name)
            return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # inner product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str  # "dense" | "bm25" | "hybrid" | "rerank"


class Namespace:
    """A single isolated document collection."""

    def __init__(self, name: str, dimension: int):
        import faiss
        from rank_bm25 import BM25Okapi

        self.name = name
        self.dimension = dimension
        self._faiss = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
        self._lock = threading.RLock()

    # -- writes -------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from rank_bm25 import BM25Okapi

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self._lock:
            self._faiss.add(vectors)
            self._chunks.extend(chunks)
            self._bm25_corpus = [_tokenise(c.text) for c in self._chunks]
            self._bm25 = BM25Okapi(self._bm25_corpus)

    # -- reads --------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._chunks)

    def dense(self, vector: np.ndarray, k: int) -> list[Hit]:
        with self._lock:
            if self.size == 0:
                return []
            k = min(k, self.size)
            scores, indices = self._faiss.search(np.asarray([vector], dtype="float32"), k)
            return [
                Hit(self._chunks[i], float(s), "dense")
                for s, i in zip(scores[0], indices[0])
                if 0 <= i < len(self._chunks)
            ]

    def bm25(self, query: str, k: int) -> list[Hit]:
        with self._lock:
            if self.size == 0 or self._bm25 is None:
                return []
            scores = self._bm25.get_scores(_tokenise(query))
            order = np.argsort(scores)[::-1][: min(k, self.size)]
            return [
                Hit(self._chunks[i], float(scores[i]), "bm25")
                for i in order
                if scores[i] > 0
            ]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    # -- persistence ---------------------------------------------------
    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._faiss, str(directory / f"{self.name}.faiss"))
            with open(directory / f"{self.name}.chunks.pkl", "wb") as fh:
                pickle.dump(self._chunks, fh)

    @classmethod
    def load(cls, name: str, dimension: int, directory: Path) -> "Namespace | None":
        import faiss

        index_path = directory / f"{name}.faiss"
        chunks_path = directory / f"{name}.chunks.pkl"
        if not index_path.exists() or not chunks_path.exists():
            return None
        ns = cls(name, dimension)
        ns._faiss = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as fh:
            ns._chunks = pickle.load(fh)
        if ns._chunks:
            from rank_bm25 import BM25Okapi

            ns._bm25_corpus = [_tokenise(c.text) for c in ns._chunks]
            ns._bm25 = BM25Okapi(ns._bm25_corpus)
        return ns


def _tokenise(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


class Store:
    """Thread-safe registry of namespaces."""

    def __init__(self, dimension: int, data_dir: Path | None = None):
        self.dimension = dimension
        self.data_dir = data_dir
        self._namespaces: dict[str, Namespace] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> Namespace:
        with self._lock:
            if name not in self._namespaces:
                if self.data_dir:
                    loaded = Namespace.load(name, self.dimension, self.data_dir)
                    if loaded is not None:
                        self._namespaces[name] = loaded
                        return loaded
                self._namespaces[name] = Namespace(name, self.dimension)
            return self._namespaces[name]

    def namespaces(self) -> list[str]:
        with self._lock:
            return sorted(self._namespaces)

    def persist(self, name: str) -> None:
        if self.data_dir:
            self.get(name).save(self.data_dir)

    def reset(self) -> None:
        with self._lock:
            self._namespaces.clear()

    def reset_namespace(self, name: str) -> None:
        with self._lock:
            self._namespaces.pop(name, None)
            if self.data_dir:
                for suffix in (".faiss", ".chunks.pkl"):
                    path = self.data_dir / f"{name}{suffix}"
                    if path.exists():
                        path.unlink()


def reciprocal_rank_fusion(
    result_lists: list[list[Hit]], k: int = 60, limit: int = 10
) -> list[Hit]:
    """Fuse ranked lists without needing comparable score scales."""
    scores: dict[str, float] = {}
    lookup: dict[str, Chunk] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            lookup.setdefault(key, hit.chunk)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [Hit(lookup[key], score, "hybrid") for key, score in ranked]


class Reranker:
    """Optional cross-encoder reranker; degrades to a no-op if unavailable."""

    _instance: "Reranker | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")
        self.available = True

    @classmethod
    def get(cls, model_name: str) -> "Reranker | None":
        with cls._lock:
            if cls._instance is None:
                try:
                    cls._instance = cls(model_name)
                except Exception:
                    cls._instance = cls.__new__(cls)
                    cls._instance.model = None
                    cls._instance.available = False
            return cls._instance

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return hits
        if not getattr(self, "available", False):
            return hits[:top_n]
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self.model.predict(pairs)
        rescored = [
            Hit(h.chunk, float(s), "rerank") for h, s in zip(hits, scores)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_n]
