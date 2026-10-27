"""Regression tests for deterministic namespace replacement and persistence."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vistaar.config import Settings
from vistaar.index import Namespace
from vistaar.pipeline import RAGPipeline
from vistaar.text import Chunk


def _chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}:{ordinal:04d}",
        text=text,
        page=1,
        section="page 1",
        ordinal=ordinal,
    )


def _vectors(count: int, dimension: int = 4) -> np.ndarray:
    values = np.arange(1, count * dimension + 1, dtype="float32")
    return values.reshape(count, dimension)


class FakeEmbedder:
    dimension = 4

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            row = np.frombuffer(digest[: self.dimension * 4], dtype=np.uint32)
            row = row.astype("float32")
            row /= max(float(np.linalg.norm(row)), 1.0)
            rows.append(row)
        return np.asarray(rows, dtype="float32")


def test_strict_add_rejects_duplicate_chunk_ids_without_mutation():
    namespace = Namespace("strict", 4)
    chunks = [_chunk("doc", 0, "first"), _chunk("doc", 1, "second")]
    namespace.add(chunks, _vectors(2))

    with pytest.raises(ValueError, match="already exists"):
        namespace.add([chunks[0]], _vectors(1))

    assert namespace.size == 2
    assert [chunk.chunk_id for chunk in namespace.all_chunks()] == [
        "doc:0000",
        "doc:0001",
    ]


def test_document_upsert_replaces_only_target_document():
    namespace = Namespace("upsert", 4)
    namespace.add([_chunk("other", 0, "retained")], _vectors(1))
    namespace.upsert_document(
        [_chunk("target", 0, "old one"), _chunk("target", 1, "old two")],
        _vectors(2),
    )

    removed = namespace.upsert_document(
        [_chunk("target", 0, "replacement")], _vectors(1)
    )

    chunks = namespace.all_chunks()
    assert removed == 2
    assert namespace.size == 2
    assert {(chunk.doc_id, chunk.text) for chunk in chunks} == {
        ("other", "retained"),
        ("target", "replacement"),
    }
    assert len({chunk.chunk_id for chunk in chunks}) == namespace.size


def test_repeated_pipeline_indexing_is_idempotent_across_reload(tmp_path, monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(
        "vistaar.pipeline.Embedder.get",
        classmethod(lambda cls, model_name, revision=None: fake),
    )
    settings = Settings(api_key="", data_dir=tmp_path)
    content = ("A deterministic sentence about evaluation integrity. " * 80).encode()

    first_pipeline = RAGPipeline(settings)
    first = first_pipeline.index_document("paper.txt", content, namespace="evaluation")
    first_size = first_pipeline.store.get("evaluation").size
    second = first_pipeline.index_document("paper.txt", content, namespace="evaluation")

    assert first_size == first["chunks"]
    assert second["namespace_chunks"] == first_size
    assert second["replaced_chunks"] == first_size
    assert len(
        {chunk.chunk_id for chunk in first_pipeline.store.get("evaluation").all_chunks()}
    ) == first_size

    reloaded_pipeline = RAGPipeline(settings)
    third = reloaded_pipeline.index_document("paper.txt", content, namespace="evaluation")
    reloaded = reloaded_pipeline.store.get("evaluation")

    assert third["namespace_chunks"] == first_size
    assert third["replaced_chunks"] == first_size
    assert reloaded.size == first_size
    assert len({chunk.chunk_id for chunk in reloaded.all_chunks()}) == first_size


def test_pipeline_keeps_other_documents_in_namespace(tmp_path, monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(
        "vistaar.pipeline.Embedder.get",
        classmethod(lambda cls, model_name, revision=None: fake),
    )
    pipeline = RAGPipeline(Settings(api_key="", data_dir=tmp_path))
    first = ("First document has enough useful text for indexing. " * 40).encode()
    second = ("Second document also has enough useful text for indexing. " * 40).encode()

    first_result = pipeline.index_document("first.txt", first, namespace="shared")
    second_result = pipeline.index_document("second.txt", second, namespace="shared")
    before_repeat = pipeline.store.get("shared").size
    repeated = pipeline.index_document("first.txt", first, namespace="shared")

    assert before_repeat == first_result["chunks"] + second_result["chunks"]
    assert repeated["namespace_chunks"] == before_repeat
    assert {chunk.doc_id for chunk in pipeline.store.get("shared").all_chunks()} == {
        first_result["doc_id"],
        second_result["doc_id"],
    }
