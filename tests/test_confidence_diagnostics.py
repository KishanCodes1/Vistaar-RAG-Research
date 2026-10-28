"""Focused tests for additive confidence diagnostics and behavior equivalence."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from eval.qasper import EvalItem
from eval.run_eval import build_evaluation_record
from eval.shadow import diagnostic_record_is_complete
from vistaar.config import Settings
from vistaar.diagnostics import CONFIDENCE_DIAGNOSTICS_SCHEMA
from vistaar.index import Hit, Reranker
from vistaar.pipeline import RAGPipeline, RetrievalConfig
from vistaar.reader import LocalReader, ReaderDiagnostics, ReaderOutput
from vistaar.text import Chunk


class FakeEmbedder:
    dimension = 4

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for index, _ in enumerate(texts):
            row = np.zeros(self.dimension, dtype="float32")
            row[index % self.dimension] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype="float32")


class FakeReranker:
    available = True

    def rerank(self, query, hits, top_n, scored_candidates=None):
        rescored = [
            Hit(hit.chunk, float(len(hits) - rank), "rerank")
            for rank, hit in enumerate(reversed(hits))
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        if scored_candidates is not None:
            scored_candidates.extend(rescored)
        return rescored[:top_n]


class FakeReader:
    def __init__(self, confidence: float = 0.4, found: bool = True):
        self.confidence = confidence
        self.found = found
        self.collect_flags = []

    def answer(self, question, hits, max_passages=4, collect_diagnostics=False):
        self.collect_flags.append(collect_diagnostics)
        return ReaderOutput(
            answer="diagnostic answer" if self.found else "",
            confidence=self.confidence,
            passage_ids=[1] if self.found else [],
            found=self.found,
            diagnostics=(
                ReaderDiagnostics(
                    available=True,
                    found=self.found,
                    passage_count=min(len(hits), max_passages),
                    context_characters=42,
                    window_count=2,
                    candidate_start_count=3,
                    candidate_span_count=7,
                    best_span_score=5.0 if self.found else None,
                    second_best_span_score=4.0 if self.found else None,
                    selected_window=1 if self.found else None,
                    selected_null_score=5.4 if self.found else None,
                    null_score_min=2.0,
                    null_score_max=5.4,
                    span_null_margin=-0.4 if self.found else None,
                    selected_passage=1 if self.found else None,
                    no_span_reason=None if self.found else "no_candidate_span",
                )
                if collect_diagnostics
                else None
            ),
        )


def _chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(
        doc_id="document",
        chunk_id=f"document:{ordinal:04d}",
        text=text,
        page=1,
        section="page 1",
        ordinal=ordinal,
    )


def _pipeline(monkeypatch) -> RAGPipeline:
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "vistaar.pipeline.Embedder.get",
        classmethod(lambda cls, model_name, revision=None: fake_embedder),
    )
    pipeline = RAGPipeline(Settings(api_key="", data_dir=None, reader_threshold=0.5))
    chunks = [
        _chunk(index, f"diagnostic evidence passage {index}") for index in range(6)
    ]
    vectors = fake_embedder.encode([chunk.text for chunk in chunks])
    pipeline.store.get("paper").add(chunks, vectors)
    pipeline._reranker = FakeReranker()
    pipeline._reader = FakeReader()
    return pipeline


def _business_answer(answer):
    return {
        "answer": answer.answer,
        "raw_answer": answer.raw_answer,
        "abstained": answer.abstained,
        "citations": answer.citations,
        "scores": answer.scores,
        "model": answer.model,
    }


def test_instrumentation_preserves_ranking_answer_and_decision(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    config = RetrievalConfig(
        name="instrumented",
        use_bm25=True,
        use_rerank=True,
        use_abstention=True,
        top_k=2,
        rerank_top_n=2,
    )

    plain_hits = pipeline.retrieve("diagnostic evidence", "paper", config)
    plain_answer = pipeline.answer(
        "diagnostic evidence", "paper", config, hits=plain_hits
    )

    diagnostics = {}
    instrumented_hits = pipeline.retrieve(
        "diagnostic evidence", "paper", config, diagnostics=diagnostics
    )
    instrumented_answer = pipeline.answer(
        "diagnostic evidence",
        "paper",
        config,
        hits=instrumented_hits,
        diagnostics=diagnostics,
    )

    assert [
        (hit.chunk.chunk_id, hit.score, hit.source) for hit in instrumented_hits
    ] == [(hit.chunk.chunk_id, hit.score, hit.source) for hit in plain_hits]
    assert _business_answer(instrumented_answer) == _business_answer(plain_answer)
    assert pipeline._reader.collect_flags == [False, True]
    assert instrumented_answer.abstained is True
    assert "diagnostics" not in plain_answer.to_dict()
    assert instrumented_answer.to_dict()["diagnostics"] == diagnostics

    assert diagnostics["schema_version"] == CONFIDENCE_DIAGNOSTICS_SCHEMA
    retrieval = diagnostics["retrieval"]
    assert retrieval["stages"]["dense"]["available"] is True
    assert retrieval["stages"]["bm25"]["available"] is True
    assert retrieval["stages"]["rrf"]["available"] is True
    assert retrieval["stages"]["reranker"]["available"] is True
    assert retrieval["stages"]["reranker"]["score_summary"]["count"] == 4
    assert all(
        candidate["fused_rank"] is not None
        for candidate in retrieval["stages"]["reranker"]["candidates"]
    )
    assert all(
        candidate["chunk_id"].startswith("document:")
        and candidate["doc_id"] == "document"
        and candidate["page"] == 1
        for candidate in retrieval["stages"]["selected_context"]["candidates"]
    )
    assert retrieval["features"]["selected_chunk_count"] == 2
    assert "gold" not in str(diagnostics).lower()
    assert diagnostics["reader"]["span_null_margin"] == -0.4
    assert diagnostics["decision"]["abstained"] is True
    assert diagnostics["decision"]["triggered_reasons"] == [
        "below_configured_threshold"
    ]


def test_shadow_record_is_additive_and_preserves_legacy_fields(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    config = RetrievalConfig(
        name="E: shadow fixture",
        use_bm25=True,
        use_rerank=True,
        use_abstention=True,
        top_k=2,
        rerank_top_n=2,
    )
    item = EvalItem(
        doc_id="paper",
        question_id="fixture-question",
        question="diagnostic evidence",
        doc_text="fixture text",
        doc_title="Fixture",
        gold_answers=["diagnostic answer"],
        gold_evidence=["diagnostic evidence passage"],
        answerable=True,
    )

    plain = build_evaluation_record(
        pipeline, item, config, collect_diagnostics=False
    )
    shadow = build_evaluation_record(
        pipeline, item, config, collect_diagnostics=True
    )
    plain_business = {key: value for key, value in plain.items() if key != "latency_s"}
    shadow_business = {
        key: value
        for key, value in shadow.items()
        if key not in {"latency_s", "diagnostics"}
    }

    assert "diagnostics" not in plain
    assert diagnostic_record_is_complete(shadow)
    assert shadow_business == plain_business
    assert pipeline._reader.collect_flags == [False, True]


def test_diagnostics_are_deterministic_and_mark_unavailable_stages(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    pipeline._reader = FakeReader(confidence=0.8)
    config = RetrievalConfig(
        name="dense-only",
        use_bm25=False,
        use_rerank=False,
        use_abstention=True,
        top_k=2,
        rerank_top_n=2,
    )

    outputs = []
    for _ in range(2):
        diagnostics = {}
        hits = pipeline.retrieve("diagnostic evidence", "paper", config, diagnostics)
        answer = pipeline.answer(
            "diagnostic evidence", "paper", config, hits, diagnostics
        )
        outputs.append((diagnostics, _business_answer(answer)))

    assert outputs[0] == outputs[1]
    diagnostics = outputs[0][0]
    for name in ("bm25", "rrf", "reranker"):
        assert diagnostics["retrieval"]["stages"][name] == {
            "requested": False,
            "available": False,
            "reason": "not_requested",
            "candidates": None,
            "score_summary": None,
        }
    assert diagnostics["decision"]["abstained"] is False


def test_retrieval_disabled_path_skips_diagnostic_candidate_copy(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    config = RetrievalConfig(name="no-diagnostics", top_k=2, rerank_top_n=2)

    def fail_if_called(hits):
        raise AssertionError("diagnostic candidate copy ran while disabled")

    monkeypatch.setattr("vistaar.pipeline.snapshot_hits", fail_if_called)
    hits = pipeline.retrieve("diagnostic evidence", "paper", config)

    assert len(hits) == 2


def test_empty_namespace_records_missing_signals_without_fabrication(monkeypatch):
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "vistaar.pipeline.Embedder.get",
        classmethod(lambda cls, model_name, revision=None: fake_embedder),
    )
    pipeline = RAGPipeline(Settings(api_key="", data_dir=None))
    diagnostics = {}
    answer = pipeline.answer("question", "empty", diagnostics=diagnostics)

    assert answer.abstained is True
    assert diagnostics["reader"] == {
        "available": False,
        "reason": "no_retrieved_passages",
    }
    assert diagnostics["retrieval"]["stages"]["dense"]["candidates"] == []
    assert diagnostics["retrieval"]["stages"]["dense"]["score_summary"] is None
    assert diagnostics["decision"]["triggered_reasons"] == [
        "no_retrieved_passages"
    ]
    assert diagnostics["decision"]["configured_threshold"] == 0.5


def test_hosted_no_hits_has_no_fabricated_reader_threshold(monkeypatch):
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "vistaar.pipeline.Embedder.get",
        classmethod(lambda cls, model_name, revision=None: fake_embedder),
    )
    pipeline = RAGPipeline(
        Settings(api_key="unused", reader="gemini", data_dir=None)
    )
    diagnostics = {}

    answer = pipeline.answer("question", "empty", diagnostics=diagnostics)

    assert answer.abstained is True
    assert diagnostics["decision"]["backend"] == "gemini"
    assert diagnostics["decision"]["configured_threshold"] is None


def test_reranker_exposes_all_scores_without_changing_selected_hits():
    chunks = [_chunk(index, f"passage {index}") for index in range(3)]
    hits = [Hit(chunk, float(index), "hybrid") for index, chunk in enumerate(chunks)]
    reranker = Reranker.__new__(Reranker)
    reranker.available = True
    reranker.model = SimpleNamespace(
        predict=lambda pairs: np.asarray([0.2, 0.9, -0.1], dtype="float32")
    )

    plain = reranker.rerank("question", hits, 2)
    scored = []
    instrumented = reranker.rerank(
        "question", hits, 2, scored_candidates=scored
    )

    assert [(hit.chunk.chunk_id, hit.score) for hit in instrumented] == [
        (hit.chunk.chunk_id, hit.score) for hit in plain
    ]
    assert [hit.chunk.chunk_id for hit in scored] == [
        "document:0001",
        "document:0000",
        "document:0002",
    ]
    assert [hit.score for hit in scored] == pytest.approx([0.9, 0.2, -0.1])


def test_reader_disabled_path_skips_diagnostic_state_and_preserves_answer():
    # Torch and FAISS abort if loaded in the wrong order on this macOS runtime.
    # Exercise the real reader loop in a clean process with Torch loaded first.
    script = textwrap.dedent(
        """
        import json
        from types import SimpleNamespace
        import torch
        from vistaar.index import Hit
        from vistaar.reader import LocalReader
        from vistaar.text import Chunk

        class Encoding(dict):
            def sequence_ids(self, window):
                return [None, 0, 1, 1]

        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return Encoding(
                    input_ids=torch.tensor([[0, 1, 2, 3]]),
                    offset_mapping=torch.tensor(
                        [[[0, 0], [0, 0], [5, 11], [12, 16]]]
                    ),
                )

        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(
                    start_logits=torch.tensor([[1.0, -10.0, 4.0, -10.0]]),
                    end_logits=torch.tensor([[1.0, -10.0, 3.0, -10.0]]),
                )

        reader = LocalReader.__new__(LocalReader)
        reader.torch = torch
        reader.tokenizer = Tokenizer()
        reader.model = Model()
        chunk = Chunk("document", "document:0000", "answer text", 1, "page 1", 0)
        original_state = __import__("vistaar.reader", fromlist=["x"])._ReaderDiagnosticState
        original_diagnostics = __import__("vistaar.reader", fromlist=["x"]).ReaderDiagnostics
        reader_module = __import__("vistaar.reader", fromlist=["x"])
        def fail(*args, **kwargs):
            raise AssertionError("diagnostic-only state constructed while disabled")
        reader_module._ReaderDiagnosticState = fail
        reader_module.ReaderDiagnostics = fail
        disabled = reader.answer(
            "question", [Hit(chunk, 1.0, "dense")], 1, collect_diagnostics=False
        )
        reader_module._ReaderDiagnosticState = original_state
        reader_module.ReaderDiagnostics = original_diagnostics
        output = reader.answer(
            "question", [Hit(chunk, 1.0, "dense")], 1, collect_diagnostics=True
        )
        print(json.dumps({
            "answer": output.answer,
            "found": output.found,
            "confidence": output.confidence,
            "diagnostics": output.diagnostics.to_dict(),
            "disabled_answer": disabled.answer,
            "disabled_confidence": disabled.confidence,
            "disabled_diagnostics": disabled.diagnostics,
        }))
        """
    )
    result = json.loads(subprocess.check_output([sys.executable, "-c", script]))
    diagnostics = result["diagnostics"]

    assert result["answer"] == "answer"
    assert result["disabled_answer"] == result["answer"]
    assert result["disabled_confidence"] == result["confidence"]
    assert result["disabled_diagnostics"] is None
    assert result["found"] is True
    assert result["confidence"] == pytest.approx(1 / (1 + np.exp(-5.0)))
    assert diagnostics["best_span_score"] == 7.0
    assert diagnostics["found"] is True
    assert diagnostics["selected_null_score"] == 2.0
    assert diagnostics["span_null_margin"] == 5.0
    assert diagnostics["selected_passage"] == 1
    assert diagnostics["window_count"] == 1
    assert diagnostics["candidate_start_count"] == 1
    assert diagnostics["candidate_span_count"] == 2
