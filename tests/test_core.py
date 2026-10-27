"""Unit tests for the parts that must never silently regress.

Deliberately does not load embedding or reader models: these run in ~1s.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import (
    best_f1,
    bootstrap_ci,
    citation_scores,
    exact_match,
    token_f1,
)
from eval.qasper import normalize
from eval.sweep import sweep
from vistaar.config import Settings, load_env
from vistaar.index import Namespace, reciprocal_rank_fusion
from vistaar.pipeline import ABSTAIN_MESSAGE, Answer
from vistaar.text import chunk_document, Document, Page, extract_document


def test_leading_whitespace_in_env_is_stripped(tmp_path, monkeypatch):
    """Regression: the original .env had 'KEY= <value>' which produced 403s."""
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY= abc123\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    load_env(env)
    settings = Settings.from_env(reader="gemini")
    assert settings.api_key == "abc123"


def test_local_reader_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("VISTAAR_READER", "local")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings.from_env(reader="local")
    assert settings.reader == "local"


def test_chunking_respects_size_and_overlap():
    text = "Sentence number %d lives here. " * 200
    text = " ".join(text.split())
    doc = Document(doc_id="d", filename="d.txt", pages=(Page(number=1, text=text),))
    chunks = chunk_document(doc, size=400, overlap=80)
    assert chunks
    assert all(len(c.text) <= 400 + 80 for c in chunks)
    assert chunks[0].page == 1
    assert chunks[0].doc_id == "d"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_unsupported_file_type_rejected():
    with pytest.raises(ValueError):
        extract_document("virus.exe", b"MZ\x90\x00")


def test_empty_text_document_rejected():
    with pytest.raises(ValueError):
        extract_document("blank.txt", b"   ")


def test_normalize_and_exact_match():
    assert normalize("The  Model, is Fast!") == "model is fast"
    assert exact_match("The model", ["the model"]) == 1.0
    assert exact_match("wrong", ["the model"]) == 0.0


def test_token_f1_known_values():
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert token_f1("xyz", "abc") == 0.0
    assert 0 < token_f1("the cat", "the cat sat") < 1


def test_best_f1_picks_best_reference():
    refs = ["completely unrelated text here", "the answer is 42"]
    assert best_f1("the answer is 42", refs) == pytest.approx(1.0)


def test_empty_raw_answer_is_not_replaced_by_abstention_message():
    answer = Answer(
        answer=ABSTAIN_MESSAGE,
        raw_answer="",
        abstained=True,
        citations=[],
        scores={"confidence": 0.0},
        latency_s=0.0,
    )
    assert answer.to_dict()["raw_answer"] == ""

    result = sweep(
        [
            {
                "answerable": True,
                "prediction": ABSTAIN_MESSAGE,
                "raw_prediction": "",
                "gold": "could find document",
                "scores": {"confidence": 0.0},
            }
        ]
    )
    assert result["curve"][0]["selective_f1"] == 0.0


def test_citation_scores_reward_correct_passage():
    gold = ["the model reaches 42.5 f1 on the development split"]
    good, _, _ = citation_scores(["the model reaches 42.5 f1 on the development split"], gold)
    bad, _, _ = citation_scores(["bananas are yellow and grow in the tropics"], gold)
    assert good == 1.0
    assert bad == 0.0


def test_bootstrap_ci_brackets_mean():
    values = [0.1, 0.4, 0.6, 0.9, 0.5, 0.3, 0.7]
    mean, low, high = bootstrap_ci(values, n_resamples=300)
    assert low <= mean <= high


def test_namespace_isolation():
    """The original code shared one global index between all users."""
    a = Namespace("a", 4)
    b = Namespace("b", 4)
    from vistaar.text import Chunk

    chunk = Chunk("docA", "docA:0", "secret alpha text", 1, "page 1", 0)
    a.add([chunk], np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    assert a.size == 1
    assert b.size == 0
    assert b.dense(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), 5) == []


def test_rrf_fuses_and_dedupes():
    from vistaar.index import Hit
    from vistaar.text import Chunk

    c1 = Chunk("d", "d:0", "one", 1, "p", 0)
    c2 = Chunk("d", "d:1", "two", 1, "p", 1)
    fused = reciprocal_rank_fusion(
        [[Hit(c1, 0.9, "dense"), Hit(c2, 0.5, "dense")],
         [Hit(c2, 12.0, "bm25"), Hit(c1, 3.0, "bm25")]],
        limit=10,
    )
    assert {h.chunk.chunk_id for h in fused} == {"d:0", "d:1"}
