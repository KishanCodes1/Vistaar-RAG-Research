"""Stable evaluation fingerprints and isolated record caches.

Fingerprints are SHA-256 hashes of canonical JSON (UTF-8, sorted keys, compact
separators). A run fingerprint covers the selected dataset records, document
content, code/prompt hashes, chunking, model revisions, inference settings, and
retrieval configurations. Each cached row additionally carries fingerprints
for its run, retrieval config, and evaluation item. A mismatch is a cache miss,
never a best-effort reuse.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vistaar-eval-cache-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_item_fingerprint(item: Any) -> str:
    """Fingerprint every item field copied into an evaluation record."""
    return stable_fingerprint(
        {
            "doc_id": item.doc_id,
            "question_id": item.question_id,
            "question": item.question,
            "doc_text_sha256": hashlib.sha256(item.doc_text.encode()).hexdigest(),
            "doc_title": item.doc_title,
            "gold_answers": list(item.gold_answers),
            "gold_evidence": list(item.gold_evidence),
            "answerable": bool(item.answerable),
        }
    )


def config_fingerprint(config: Any) -> str:
    return stable_fingerprint(
        {
            "name": config.name,
            "use_bm25": bool(config.use_bm25),
            "use_rerank": bool(config.use_rerank),
            "use_abstention": bool(config.use_abstention),
            "top_k": int(config.top_k),
            "rerank_top_n": int(config.rerank_top_n),
        }
    )


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return cleaned.strip("-") or "cache"


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class CacheLoad:
    records: dict[str, dict]
    accepted: int = 0
    rejected: int = 0
    malformed: int = 0


class EvaluationCache:
    """Run-scoped JSONL cache that fails closed on any fingerprint mismatch."""

    def __init__(self, root: str | Path, run_fingerprint: str):
        self.root = Path(root)
        self.run_fingerprint = run_fingerprint

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_fingerprint

    def path_for(self, config_name: str, config_fp: str) -> Path:
        return self.run_dir / "records" / f"{safe_name(config_name)}-{config_fp[:16]}.jsonl"

    def load(
        self,
        config_name: str,
        config_fp: str,
        item_fingerprints: dict[str, str],
    ) -> CacheLoad:
        path = self.path_for(config_name, config_fp)
        if not path.exists():
            return CacheLoad(records={})

        records: dict[str, dict] = {}
        rejected = 0
        malformed = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question_id = record["question_id"]
                metadata = record["_cache"]
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed += 1
                continue
            expected_item = item_fingerprints.get(question_id)
            valid = (
                metadata.get("schema") == SCHEMA_VERSION
                and metadata.get("run_fingerprint") == self.run_fingerprint
                and metadata.get("config_fingerprint") == config_fp
                and record.get("config") == config_name
                and expected_item is not None
                and metadata.get("item_fingerprint") == expected_item
            )
            if not valid:
                rejected += 1
                continue
            records[question_id] = record
        return CacheLoad(
            records=records,
            accepted=len(records),
            rejected=rejected,
            malformed=malformed,
        )

    def append(
        self,
        config_name: str,
        config_fp: str,
        item_fingerprint: str,
        record: dict,
    ) -> dict:
        cached = dict(record)
        cached["_cache"] = {
            "schema": SCHEMA_VERSION,
            "run_fingerprint": self.run_fingerprint,
            "config_fingerprint": config_fp,
            "item_fingerprint": item_fingerprint,
        }
        path = self.path_for(config_name, config_fp)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(cached, ensure_ascii=False) + "\n")
        return cached
