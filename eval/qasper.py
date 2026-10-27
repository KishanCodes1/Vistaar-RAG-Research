"""QASPER loader.

QASPER (Dasigi et al., 2021) provides 5,049 questions over 1,585 NLP papers,
with gold evidence annotations and explicit `unanswerable` flags. That makes it
usable for citation and abstention evaluation without any new annotation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

# QASPER extractive spans sometimes contain only pointers such as "BIBREF19"
# or "TABREF3" instead of real answer text. Scoring those as answers would
# inflate or deflate F1 for reasons unrelated to system quality.
_JUNK_SPAN = re.compile(r"^(BI|TA|FI)G?B?REF\d+[A-Z]?$", re.IGNORECASE)


def _is_usable(span: str) -> bool:
    span = span.strip()
    if len(span) < 2:
        return False
    return not _JUNK_SPAN.match(span)


@dataclass
class EvalItem:
    doc_id: str
    question_id: str
    question: str
    doc_text: str
    doc_title: str
    gold_answers: list[str] = field(default_factory=list)
    gold_evidence: list[str] = field(default_factory=list)
    answerable: bool = True

    @property
    def gold(self) -> str:
        return self.gold_answers[0] if self.gold_answers else ""


def _paper_text(row: dict) -> str:
    parts = [row.get("title", "").strip(), row.get("abstract", "").strip()]
    full = row.get("full_text") or {}
    names = full.get("section_name") or []
    paragraphs = full.get("paragraphs") or []
    for i, name in enumerate(names):
        parts.append(f"\n## {name}")
        section_paras = paragraphs[i] if i < len(paragraphs) else []
        for para in section_paras:
            text = " ".join(s for s in para if s)
            if text.strip():
                parts.append(text.strip())
    return "\n\n".join(p for p in parts if p)


def _answer_texts(annotation: dict) -> tuple[list[str], bool]:
    """Return (answer strings, is_unanswerable) for one annotation.

    `annotation` is a single answer record with keys: unanswerable,
    extractive_spans, yes_no, free_form_answer, evidence.
    """
    if annotation.get("unanswerable"):
        return [], True

    spans = [s for s in (annotation.get("extractive_spans") or []) if s and _is_usable(s)]
    if spans:
        return [s.strip() for s in spans], False

    free = (annotation.get("free_form_answer") or "").strip()
    if free:
        return [free], False

    if annotation.get("yes_no") is not None:
        return ["yes" if annotation["yes_no"] else "no"], False

    return [], False


def _evidence_texts(annotations: list[dict]) -> list[str]:
    evidence: list[str] = []
    for ans in annotations:
        for item in ans.get("evidence") or []:
            if isinstance(item, str) and item.strip():
                evidence.append(item.strip())
        for item in ans.get("highlighted_evidence") or []:
            if isinstance(item, str) and item.strip():
                evidence.append(item.strip())
    # de-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for item in evidence:
        key = item[:120].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def load_qasper(
    path: str | Path,
    max_papers: int | None = None,
    max_questions_per_paper: int | None = None,
    require_evidence: bool = False,
) -> list[EvalItem]:
    reader = pq.ParquetFile(path)
    items: list[EvalItem] = []

    for row in reader.read().to_pylist():
        doc_id = row["id"]
        doc_text = _paper_text(row)
        if not doc_text.strip():
            continue

        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        qids = qas.get("question_id") or []
        all_answers = qas.get("answers") or []

        taken = 0
        for idx, question in enumerate(questions):
            container = all_answers[idx] if idx < len(all_answers) else {}
            annotations = container.get("answer", []) if isinstance(container, dict) else []
            gold: list[str] = []
            any_answerable = False
            for annotation in annotations:
                texts, unanswerable = _answer_texts(annotation)
                if not unanswerable:
                    any_answerable = True
                gold.extend(texts)

            evidence = _evidence_texts(annotations)
            if require_evidence and not evidence:
                continue

            items.append(
                EvalItem(
                    doc_id=doc_id,
                    question_id=qids[idx] if idx < len(qids) else f"{doc_id}-{idx}",
                    question=question.strip(),
                    doc_text=doc_text,
                    doc_title=row.get("title", ""),
                    gold_answers=_dedupe(gold),
                    gold_evidence=evidence,
                    answerable=any_answerable,
                )
            )
            taken += 1
            if max_questions_per_paper and taken >= max_questions_per_paper:
                break

        if max_papers and len({i.doc_id for i in items}) >= max_papers:
            break

    return items


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for v in values:
        key = normalize(v)
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
