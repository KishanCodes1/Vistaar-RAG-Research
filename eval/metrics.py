"""Evaluation metrics for citation-grounded QA.

Deliberately dependency-free (no bert-score) so results reproduce anywhere.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .qasper import normalize


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(prediction: str, references: list[str]) -> float:
    p = normalize(prediction)
    return float(any(p == normalize(r) for r in references))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(reference)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(token_f1(prediction, r) for r in references)


def _tokenset(text: str) -> set[str]:
    return set(_tokens(text))


def evidence_coverage(texts: list[str], gold_evidence: list[str]) -> float:
    """Fraction of gold-evidence vocabulary present in the passage set.

    Uses the UNION of passages rather than per-passage maximum, because a
    single evidence paragraph is routinely split across chunk boundaries;
    scoring per passage would penalise a correct retrieval for an artifact
    of chunking.
    """
    if not gold_evidence or not texts:
        return 0.0
    gold = _tokenset(" ".join(gold_evidence))
    if not gold:
        return 0.0
    retrieved: set[str] = set()
    for text in texts:
        retrieved |= _tokenset(text)
    return len(gold & retrieved) / len(gold)


def citation_scores(
    cited_texts: list[str], gold_evidence: list[str], containment: float = 0.5
) -> tuple[float, float, int]:
    """Return (citation precision, citation recall, n_correct).

    Precision: a cited passage is correct if at least `containment` of its
    vocabulary appears in the gold evidence (i.e. the passage is largely
    drawn from the evidence paragraph rather than elsewhere in the paper).
    Recall: share of gold-evidence vocabulary covered by the cited passages.
    """
    if not cited_texts:
        return 0.0, 0.0, 0

    gold = _tokenset(" ".join(gold_evidence)) if gold_evidence else set()
    if not gold:
        return 0.0, 0.0, 0

    correct = 0
    for cited in cited_texts:
        tokens = _tokenset(cited)
        if not tokens:
            continue
        if len(tokens & gold) / len(tokens) >= containment:
            correct += 1

    precision = correct / len(cited_texts)
    recall = evidence_coverage(cited_texts, gold_evidence)
    return precision, recall, correct


def retrieval_recall(chunk_texts: list[str], gold_evidence: list[str]) -> float:
    return evidence_coverage(chunk_texts, gold_evidence)


def bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 13) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower = means[int((alpha / 2) * n_resamples)]
    upper = means[int((1 - alpha / 2) * n_resamples) - 1]
    return sum(values) / n, lower, upper


def paired_bootstrap_p(a: list[float], b: list[float], n_resamples: int = 1000, seed: int = 13) -> float:
    """Two-sided p-value for the difference in means of paired samples."""
    if len(a) != len(b) or not a:
        return 1.0
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    observed = sum(diffs) / n

    # Resample under the null hypothesis: the paired differences must be
    # centred at zero, otherwise the test compares the observed mean against
    # itself and returns ~0.5 by construction.
    centred = [d - observed for d in diffs]
    count = 0
    for _ in range(n_resamples):
        resample = [centred[rng.randrange(n)] for _ in range(n)]
        if abs(sum(resample) / n) >= abs(observed):
            count += 1
    return count / n_resamples


@dataclass
class ConfigResult:
    name: str
    f1: list[float] = field(default_factory=list)
    em: list[float] = field(default_factory=list)
    citation_precision: list[float] = field(default_factory=list)
    citation_recall: list[float] = field(default_factory=list)
    abstention_correct: list[float] = field(default_factory=list)
    retrieval_recall: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    abstain_on_answerable: list[float] = field(default_factory=list)
    answer_on_unanswerable: list[float] = field(default_factory=list)
    errors: int = 0

    def add(self, **kwargs) -> None:
        for key, value in kwargs.items():
            getattr(self, key).append(value)


def summarize(results: dict[str, ConfigResult]) -> list[dict]:
    rows = []
    for name, r in results.items():
        f1_mean, f1_lo, f1_hi = bootstrap_ci(r.f1)
        em_mean, em_lo, em_hi = bootstrap_ci(r.em)
        cp_mean, _, _ = bootstrap_ci(r.citation_precision)
        cr_mean, _, _ = bootstrap_ci(r.citation_recall)
        rr_mean, _, _ = bootstrap_ci(r.retrieval_recall)
        ab_mean, _, _ = bootstrap_ci(r.abstention_correct)
        lat = sum(r.latencies) / len(r.latencies) if r.latencies else 0.0
        rows.append(
            {
                "config": name,
                "n": len(r.f1),
                "f1": round(f1_mean, 4),
                "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
                "em": round(em_mean, 4),
                "citation_precision": round(cp_mean, 4),
                "citation_recall": round(cr_mean, 4),
                "retrieval_recall": round(rr_mean, 4),
                "abstention_accuracy": round(ab_mean, 4),
                "over_abstention": round(
                    sum(r.abstain_on_answerable) / len(r.abstain_on_answerable), 4
                )
                if r.abstain_on_answerable
                else 0.0,
                "missed_unanswerable": round(
                    sum(r.answer_on_unanswerable) / len(r.answer_on_unanswerable), 4
                )
                if r.answer_on_unanswerable
                else 0.0,
                "mean_latency_s": round(lat, 2),
                "errors": r.errors,
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    header = (
        "| Config | n | F1 | 95% CI | EM | Cite-P | Cite-R | Ret-R | Abst-Acc | Over-abst | Missed-UN | Lat(s) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['n']} | {r['f1']:.4f} | [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}] "
            f"| {r['em']:.4f} | {r['citation_precision']:.3f} | {r['citation_recall']:.3f} "
            f"| {r['retrieval_recall']:.3f} | {r['abstention_accuracy']:.3f} "
            f"| {r['over_abstention']:.3f} | {r['missed_unanswerable']:.3f} | {r['mean_latency_s']:.2f} |"
        )
    return header + "\n".join(lines)
