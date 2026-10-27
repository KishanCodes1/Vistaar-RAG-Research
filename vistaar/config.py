"""Configuration loading.

Environment values are normalised because .env files in the wild frequently
contain leading/trailing whitespace around values (this project's original
.env had a leading space, which silently produced a 403 from the API).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_SEARCH = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'").strip()
    return value or None


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader that does not overwrite existing environment vars."""
    candidates = [path] if path else _ENV_SEARCH
    for candidate in candidates:
        if candidate and candidate.exists():
            for raw in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = value.strip()
            return


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Immutable so it can be passed across threads safely."""

    api_key: str
    model: str = "gemini-3.5-flash"
    fallback_models: tuple[str, ...] = ("gemini-3.8-flash",)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reader_model: str = "deepset/roberta-base-squad2"

    # "local" = offline extractive reader (default, reproducible)
    # "gemini" = hosted generative model (needs API quota)
    reader: str = "local"
    reader_threshold: float = 0.5

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 8
    rerank_top_n: int = 4

    # Retrieval score below which we decline to answer.
    min_retrieval_score: float = 0.30

    max_upload_bytes: int = 20 * 1024 * 1024
    request_timeout: int = 60
    max_concurrency: int = 4
    data_dir: Path = field(default_factory=lambda: Path("data/store"))

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_env()
        reader = _clean(os.getenv("VISTAAR_READER")) or cls.reader
        key = _clean(os.getenv("GEMINI_API_KEY"))
        if reader == "gemini" and not key:
            raise RuntimeError(
                "GEMINI_API_KEY is required when VISTAAR_READER=gemini. "
                "Copy .env.example to .env and add your key, or use the local reader."
            )
        defaults = dict(
            api_key=key or "",
            reader=reader,
            model=_clean(os.getenv("VISTAAR_MODEL")) or cls.model,
            chunk_size=int(os.getenv("VISTAAR_CHUNK_SIZE", cls.chunk_size)),
            top_k=int(os.getenv("VISTAAR_TOP_K", cls.top_k)),
        )
        defaults.update(overrides)
        return cls(**defaults)

    def as_safe_dict(self) -> dict:
        """Settings without secrets, for logging and /health output."""
        data = {k: v for k, v in self.__dict__.items()}
        data["api_key"] = "***redacted***"
        return data
