"""Local extractive reader.

Retriever-reader RAG (Lewis et al., 2020) using a SQuAD2-trained span model.
The SQuAD2 objective has an explicit no-answer candidate, so we can derive a
proper confidence score for abstention instead of relying on prose heuristics.

Implemented directly against AutoModelForQuestionAnswering because
transformers v5 removed the `question-answering` pipeline.

Runs on CPU: the MPS backend aborts on this workload (Metal command-buffer
assertion), and a base-size encoder is fast enough on Apple silicon anyway.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from .index import Hit

MAX_LENGTH = 384
STRIDE = 128
MAX_ANSWER_TOKENS = 48


@dataclass
class ReaderOutput:
    answer: str
    confidence: float
    passage_ids: list[int]
    found: bool


class LocalReader:
    _instance: "LocalReader | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @classmethod
    def get(cls, model_name: str) -> "LocalReader":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name)
            return cls._instance

    def answer(self, question: str, hits: list[Hit], max_passages: int = 4) -> ReaderOutput:
        if not hits:
            return ReaderOutput("", 0.0, [], False)

        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []  # (char_start, char_end, passage_number)
        cursor = 0
        for number, hit in enumerate(hits[:max_passages], start=1):
            header = f"[P{number}]\n"
            block = header + hit.chunk.text + "\n\n"
            start = cursor + len(header)
            spans.append((start, start + len(hit.chunk.text), number))
            parts.append(block)
            cursor += len(block)
        context = "".join(parts)

        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")

        with self.torch.no_grad():
            logits = self.model(**encoded)

        best_span: tuple[float, int, int, int] | None = None  # (score, window, start_tok, end_tok)
        null_by_window: dict[int, float] = {}

        for window in range(logits.start_logits.shape[0]):
            start_logits = logits.start_logits[window]
            end_logits = logits.end_logits[window]
            sequence_ids = encoded.sequence_ids(window)
            context_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
            if not context_positions:
                continue

            null_by_window[window] = float(start_logits[0] + end_logits[0])
            start_probs = self.torch.softmax(start_logits, dim=-1)

            for i in context_positions:
                if float(start_probs[i]) < 0.02:
                    continue
                for j in context_positions:
                    if j < i or j - i > MAX_ANSWER_TOKENS:
                        continue
                    score = float(start_logits[i] + end_logits[j])
                    if best_span is None or score > best_span[0]:
                        best_span = (score, window, i, j)

        if best_span is None:
            return ReaderOutput("", 0.0, [], False)

        score, window, start_tok, end_tok = best_span
        char_start = offsets[window][start_tok][0].item()
        char_end = offsets[window][end_tok][1].item()
        answer = context[char_start:char_end].strip()

        if not answer:
            return ReaderOutput("", 0.0, [], False)

        # Confidence is the probability that the answer span beats the
        # SQuAD2 null hypothesis, i.e. sigmoid(span_logit - null_logit).
        # Using the raw logit margin is necessary: a joint softmax over all
        # positions is spread thin and makes every span look improbable, which
        # would drive the abstention threshold to meaningless values.
        margin = score - null_by_window.get(window, float("-inf"))
        confidence = 1.0 / (1.0 + self._exp(-margin))

        cited = [number for s, e, number in spans if s <= char_start < e]
        return ReaderOutput(answer, confidence, cited, True)

    @staticmethod
    def _exp(x: float) -> float:
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")
