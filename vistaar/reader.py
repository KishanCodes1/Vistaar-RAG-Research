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
from dataclasses import asdict, dataclass

from .index import Hit

MAX_LENGTH = 384
STRIDE = 128
MAX_ANSWER_TOKENS = 48


@dataclass
class ReaderDiagnostics:
    """Signals already available while selecting a local extractive span."""

    available: bool
    found: bool
    passage_count: int
    context_characters: int
    window_count: int
    candidate_start_count: int
    candidate_span_count: int
    best_span_score: float | None = None
    second_best_span_score: float | None = None
    selected_window: int | None = None
    selected_start_token: int | None = None
    selected_end_token: int | None = None
    selected_null_score: float | None = None
    null_score_min: float | None = None
    null_score_max: float | None = None
    span_null_margin: float | None = None
    selected_passage: int | None = None
    no_span_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ReaderDiagnosticState:
    """Diagnostic-only search state; instantiated only when explicitly enabled."""

    candidate_start_count: int = 0
    candidate_span_count: int = 0
    second_best_span: tuple[float, int, int, int] | None = None

    def record_start(self) -> None:
        self.candidate_start_count += 1

    def record_span(self) -> None:
        self.candidate_span_count += 1

    def replace_best(self, previous_best: tuple[float, int, int, int] | None) -> None:
        if previous_best is not None and (
            self.second_best_span is None
            or previous_best[0] > self.second_best_span[0]
        ):
            self.second_best_span = previous_best

    def consider_non_best(self, candidate: tuple[float, int, int, int]) -> None:
        if self.second_best_span is None or candidate[0] > self.second_best_span[0]:
            self.second_best_span = candidate


@dataclass
class ReaderOutput:
    answer: str
    confidence: float
    passage_ids: list[int]
    found: bool
    diagnostics: ReaderDiagnostics | None = None


class LocalReader:
    _instance: "LocalReader | None" = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_name = model_name
        self.revision = revision
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, revision=revision
        )
        self.model.eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @classmethod
    def get(cls, model_name: str, revision: str | None = None) -> "LocalReader":
        with cls._lock:
            if (
                cls._instance is None
                or cls._instance.model_name != model_name
                or cls._instance.revision != revision
            ):
                cls._instance = cls(model_name, revision)
            return cls._instance

    def answer(
        self,
        question: str,
        hits: list[Hit],
        max_passages: int = 4,
        collect_diagnostics: bool = False,
    ) -> ReaderOutput:
        if not hits:
            return ReaderOutput(
                "",
                0.0,
                [],
                False,
                (
                    ReaderDiagnostics(
                        available=False,
                        found=False,
                        passage_count=0,
                        context_characters=0,
                        window_count=0,
                        candidate_start_count=0,
                        candidate_span_count=0,
                        no_span_reason="no_retrieved_passages",
                    )
                    if collect_diagnostics
                    else None
                ),
            )

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
        diagnostic_state = _ReaderDiagnosticState() if collect_diagnostics else None

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
                if diagnostic_state is not None:
                    diagnostic_state.record_start()
                for j in context_positions:
                    if j < i or j - i > MAX_ANSWER_TOKENS:
                        continue
                    if diagnostic_state is not None:
                        diagnostic_state.record_span()
                    score = float(start_logits[i] + end_logits[j])
                    if best_span is None or score > best_span[0]:
                        if diagnostic_state is not None:
                            diagnostic_state.replace_best(best_span)
                        best_span = (score, window, i, j)
                    elif diagnostic_state is not None:
                        diagnostic_state.consider_non_best((score, window, i, j))

        if best_span is None:
            diagnostics = None
            if diagnostic_state is not None:
                null_scores = list(null_by_window.values())
                diagnostics = ReaderDiagnostics(
                    available=True,
                    found=False,
                    passage_count=min(len(hits), max_passages),
                    context_characters=len(context),
                    window_count=int(encoded["input_ids"].shape[0]),
                    candidate_start_count=diagnostic_state.candidate_start_count,
                    candidate_span_count=diagnostic_state.candidate_span_count,
                    null_score_min=min(null_scores) if null_scores else None,
                    null_score_max=max(null_scores) if null_scores else None,
                    no_span_reason="no_candidate_span",
                )
            return ReaderOutput(
                "",
                0.0,
                [],
                False,
                diagnostics,
            )

        score, window, start_tok, end_tok = best_span
        char_start = offsets[window][start_tok][0].item()
        char_end = offsets[window][end_tok][1].item()
        answer = context[char_start:char_end].strip()

        if not answer:
            diagnostics = None
            if diagnostic_state is not None:
                null_scores = list(null_by_window.values())
                diagnostics = ReaderDiagnostics(
                    available=True,
                    found=False,
                    passage_count=min(len(hits), max_passages),
                    context_characters=len(context),
                    window_count=int(encoded["input_ids"].shape[0]),
                    candidate_start_count=diagnostic_state.candidate_start_count,
                    candidate_span_count=diagnostic_state.candidate_span_count,
                    best_span_score=score,
                    second_best_span_score=(
                        diagnostic_state.second_best_span[0]
                        if diagnostic_state.second_best_span
                        else None
                    ),
                    selected_window=window,
                    selected_start_token=start_tok,
                    selected_end_token=end_tok,
                    selected_null_score=null_by_window.get(window),
                    null_score_min=min(null_scores) if null_scores else None,
                    null_score_max=max(null_scores) if null_scores else None,
                    no_span_reason="empty_decoded_span",
                )
            return ReaderOutput(
                "",
                0.0,
                [],
                False,
                diagnostics,
            )

        # Confidence is the probability that the answer span beats the
        # SQuAD2 null hypothesis, i.e. sigmoid(span_logit - null_logit).
        # Using the raw logit margin is necessary: a joint softmax over all
        # positions is spread thin and makes every span look improbable, which
        # would drive the abstention threshold to meaningless values.
        margin = score - null_by_window.get(window, float("-inf"))
        confidence = 1.0 / (1.0 + self._exp(-margin))

        cited = [number for s, e, number in spans if s <= char_start < e]
        diagnostics = None
        if diagnostic_state is not None:
            null_scores = list(null_by_window.values())
            diagnostics = ReaderDiagnostics(
                available=True,
                found=True,
                passage_count=min(len(hits), max_passages),
                context_characters=len(context),
                window_count=int(encoded["input_ids"].shape[0]),
                candidate_start_count=diagnostic_state.candidate_start_count,
                candidate_span_count=diagnostic_state.candidate_span_count,
                best_span_score=score,
                second_best_span_score=(
                    diagnostic_state.second_best_span[0]
                    if diagnostic_state.second_best_span
                    else None
                ),
                selected_window=window,
                selected_start_token=start_tok,
                selected_end_token=end_tok,
                selected_null_score=null_by_window.get(window),
                null_score_min=min(null_scores) if null_scores else None,
                null_score_max=max(null_scores) if null_scores else None,
                span_null_margin=margin,
                selected_passage=cited[0] if cited else None,
            )
        return ReaderOutput(
            answer,
            confidence,
            cited,
            True,
            diagnostics,
        )

    @staticmethod
    def _exp(x: float) -> float:
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")
