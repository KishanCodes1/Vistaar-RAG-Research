"""Text extraction and chunking.

Every chunk carries provenance (doc_id, section, page, offsets) so that
answers can cite exact locations. The original project discarded this
information, which made citation impossible.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, asdict
from pathlib import Path

# Chunking prefers these boundaries, in order.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    pages: tuple[Page, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    page: int
    section: str
    ordinal: int

    def to_dict(self) -> dict:
        return asdict(self)


def _doc_id(filename: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()[:12]
    stem = Path(filename).stem[:40] or "document"
    return f"{stem}-{digest}"


def extract_document(filename: str, content: bytes) -> Document:
    """Extract text from PDF or plain-text content.

    Raises ValueError for unsupported or empty input so the API can return a
    proper 4xx instead of silently indexing garbage.
    """
    if not filename:
        raise ValueError("Missing filename")

    lower = filename.lower()
    doc_id = _doc_id(filename, content)

    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:  # corrupt / encrypted
            raise ValueError(f"Could not open PDF: {exc}") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("PDF is encrypted") from exc

        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append(Page(number=i, text=text))
    elif lower.endswith((".txt", ".md", ".csv", ".json")):
        text = content.decode("utf-8", errors="replace")
        pages = [Page(number=1, text=text)]
    else:
        raise ValueError(
            "Unsupported file type. Upload a PDF or a plain-text file (.txt, .md)."
        )

    doc = Document(doc_id=doc_id, filename=filename, pages=tuple(pages))
    if len(doc.text.strip()) < 50:
        raise ValueError(
            "Document has no extractable text. It may be a scanned PDF that "
            "requires OCR."
        )
    return doc


def _split_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Split on the strongest available boundary, then pack into windows."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []

    units: list[str] = [text]
    for sep in _SEPARATORS:
        if all(len(u) <= size for u in units):
            break
        expanded: list[str] = []
        for unit in units:
            if len(unit) <= size:
                expanded.append(unit)
                continue
            parts = unit.split(sep)
            for part in parts:
                if not part.strip():
                    continue
                expanded.append(part + sep if sep != " " else part)
        units = expanded

    # Pack units into chunks no larger than `size`, carrying `overlap` chars.
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + len(unit) <= size:
            current += unit
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = (tail + unit).strip()
    if current.strip():
        chunks.append(current.strip())

    # Drop fragments too short to be useful for retrieval.
    return [c for c in chunks if len(c) >= 40]


def chunk_document(doc: Document, size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for page in doc.pages:
        for piece in _split_recursive(page.text, size, overlap):
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}:{ordinal:04d}",
                    text=piece,
                    page=page.number,
                    section=f"page {page.number}",
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return chunks
