"""Embedding, indexing and hybrid retrieval.

Namespaced so that one tenant's documents are never visible to another,
which was the most serious defect in the original single global index.
"""
from __future__ import annotations

import os
import pickle
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .text import Chunk


class Embedder:
    """Lazy singleton around a sentence-transformers model."""

    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str, revision: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision, device="cpu")
        self.dimension = self.model.get_embedding_dimension()

    @classmethod
    def get(cls, model_name: str, revision: str | None = None) -> "Embedder":
        with cls._lock:
            if (
                cls._instance is None
                or cls._instance.model_name != model_name
                or cls._instance.revision != revision
            ):
                cls._instance = cls(model_name, revision)
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
    """A single isolated document collection.

    ``add`` is a strict append operation and rejects duplicate chunk IDs.
    ``upsert_document`` atomically replaces chunks for one deterministic
    document ID while retaining every other document in the namespace.
    """

    def __init__(self, name: str, dimension: int):
        import faiss
        from rank_bm25 import BM25Okapi

        self.name = name
        self.dimension = dimension
        self._faiss = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, dimension), dtype="float32")
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
        self._lock = threading.RLock()

    # -- writes -------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Append new chunks, failing closed if any chunk ID already exists."""
        vectors = self._validate_vectors(chunks, vectors)
        incoming_ids = [chunk.chunk_id for chunk in chunks]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("duplicate chunk IDs in incoming batch")
        with self._lock:
            existing_ids = {chunk.chunk_id for chunk in self._chunks}
            duplicates = existing_ids.intersection(incoming_ids)
            if duplicates:
                sample = sorted(duplicates)[0]
                raise ValueError(f"chunk ID already exists: {sample}")
            self._set_contents([*self._chunks, *chunks], np.vstack([self._vectors, vectors]))

    def upsert_document(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Replace one document's chunks and return the number removed.

        Every incoming chunk must have the same document ID. Repeating an
        identical indexing operation therefore leaves the namespace size and
        chunk-ID set unchanged. Other documents are never removed.
        """
        vectors = self._validate_vectors(chunks, vectors)
        if not chunks:
            raise ValueError("cannot upsert an empty document")
        doc_ids = {chunk.doc_id for chunk in chunks}
        if len(doc_ids) != 1:
            raise ValueError("document upsert requires exactly one document ID")
        incoming_ids = [chunk.chunk_id for chunk in chunks]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("duplicate chunk IDs in incoming document")

        doc_id = next(iter(doc_ids))
        with self._lock:
            keep = [i for i, chunk in enumerate(self._chunks) if chunk.doc_id != doc_id]
            retained_chunks = [self._chunks[i] for i in keep]
            retained_vectors = (
                self._vectors[keep]
                if keep
                else np.zeros((0, self.dimension), dtype="float32")
            )
            retained_ids = {chunk.chunk_id for chunk in retained_chunks}
            overlap = retained_ids.intersection(incoming_ids)
            if overlap:
                sample = sorted(overlap)[0]
                raise ValueError(f"chunk ID collides with another document: {sample}")
            removed = len(self._chunks) - len(retained_chunks)
            self._set_contents(
                [*retained_chunks, *chunks],
                np.vstack([retained_vectors, vectors]),
            )
            return removed

    def _validate_vectors(self, chunks: list[Chunk], vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected vectors with shape (n, {self.dimension}), got {vectors.shape}"
            )
        return vectors

    def _set_contents(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Replace the in-memory snapshot. Caller must hold ``_lock``."""
        import faiss
        from rank_bm25 import BM25Okapi

        self._faiss = faiss.IndexFlatIP(self.dimension)
        self._vectors = np.asarray(vectors, dtype="float32")
        if len(self._vectors):
            self._faiss.add(self._vectors)
        self._chunks = list(chunks)
        self._bm25_corpus = [_tokenise(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus) if self._bm25_corpus else None

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
        """Persist a complete snapshot, replacing files without appending."""
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            token = uuid.uuid4().hex
            index_path = directory / f"{self.name}.faiss"
            chunks_path = directory / f"{self.name}.chunks.pkl"
            tmp_index = directory / f".{self.name}.{token}.faiss.tmp"
            tmp_chunks = directory / f".{self.name}.{token}.chunks.tmp"
            faiss.write_index(self._faiss, str(tmp_index))
            with open(tmp_chunks, "wb") as fh:
                pickle.dump(self._chunks, fh)
            os.replace(tmp_chunks, chunks_path)
            os.replace(tmp_index, index_path)

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
        if ns._faiss.ntotal != len(ns._chunks):
            raise ValueError(
                f"namespace {name!r} is inconsistent: "
                f"{ns._faiss.ntotal} vectors for {len(ns._chunks)} chunks"
            )
        chunk_ids = [chunk.chunk_id for chunk in ns._chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"namespace {name!r} contains duplicate chunk IDs")
        ns._vectors = (
            np.asarray(ns._faiss.reconstruct_n(0, ns._faiss.ntotal), dtype="float32")
            if ns._faiss.ntotal
            else np.zeros((0, dimension), dtype="float32")
        )
        if ns._chunks:
            from rank_bm25 import BM25Okapi

            ns._bm25_corpus = [_tokenise(c.text) for c in ns._chunks]
            ns._bm25 = BM25Okapi(ns._bm25_corpus)
        return ns


def _tokenise(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


class Store:
    """Thread-safe registry of persisted namespace snapshots."""

    def __init__(self, dimension: int, data_dir: Path | None = None):
        self.dimension = dimension
        self.data_dir = data_dir
        self._namespaces: dict[str, Namespace] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> Namespace:
        """Load an existing namespace or create an empty one; never replace it."""
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
        """Replace the persisted snapshot with the current in-memory state."""
        if self.data_dir:
            self.get(name).save(self.data_dir)

    def reset(self) -> None:
        """Unload all in-memory namespaces without deleting persisted data."""
        with self._lock:
            self._namespaces.clear()

    def reset_namespace(self, name: str) -> None:
        """Destructively remove one namespace from memory and persistence."""
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

    def __init__(self, model_name: str, revision: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.revision = revision
        self.model = CrossEncoder(model_name, revision=revision, device="cpu")
        self.available = True

    @classmethod
    def get(cls, model_name: str, revision: str | None = None) -> "Reranker | None":
        with cls._lock:
            if (
                cls._instance is None
                or cls._instance.model_name != model_name
                or cls._instance.revision != revision
            ):
                try:
                    cls._instance = cls(model_name, revision)
                except Exception:
                    cls._instance = cls.__new__(cls)
                    cls._instance.model_name = model_name
                    cls._instance.revision = revision
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
