"""Safety checks for the local-only shadow instrumentation workflow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eval.run_eval import DEFAULT_CACHE_ROOT, _new_output_directory, main
from eval.shadow import validate_shadow_cache_root
from vistaar.config import Settings


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shadow_mode_rejects_hosted_reader_before_dataset_access(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval",
            "--shadow",
            "--dataset",
            "does-not-exist.parquet",
            "--dataset-partition",
            "development-fixture",
        ],
    )
    monkeypatch.setattr(
        "eval.run_eval.Settings.from_env",
        lambda: Settings(api_key="unused", reader="gemini", data_dir=None),
    )

    with pytest.raises(SystemExit, match="local-only"):
        main()


def test_shadow_mode_rejects_ordinary_cache_root(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval",
            "--shadow",
            "--dataset",
            "does-not-exist.parquet",
            "--dataset-partition",
            "development-fixture",
            "--cache-root",
            str(DEFAULT_CACHE_ROOT),
        ],
    )
    monkeypatch.setattr(
        "eval.run_eval.Settings.from_env",
        lambda: Settings(api_key="", reader="local", data_dir=None),
    )

    with pytest.raises(SystemExit, match="outside the ordinary evaluation cache"):
        main()


def test_shadow_cache_rejects_exact_root_and_descendant(tmp_path):
    ordinary = tmp_path / "eval"
    outside = tmp_path / "shadow"

    with pytest.raises(ValueError, match="outside the ordinary evaluation cache"):
        validate_shadow_cache_root(ordinary, ordinary)
    with pytest.raises(ValueError, match="outside the ordinary evaluation cache"):
        validate_shadow_cache_root(ordinary / "child", ordinary)
    assert validate_shadow_cache_root(outside, ordinary) == outside.resolve()


def test_shadow_cache_rejects_symlink_into_ordinary_tree(tmp_path):
    ordinary = tmp_path / "eval"
    ordinary.mkdir()
    link = tmp_path / "linked-eval"
    try:
        link.symlink_to(ordinary, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="outside the ordinary evaluation cache"):
        validate_shadow_cache_root(link / "child", ordinary)


def test_shadow_output_directory_is_new_and_non_overwriting(tmp_path):
    root = tmp_path / "shadow-runs"
    output = _new_output_directory("", "a" * 64, root)

    assert output.parent == root
    assert output.name.endswith("-aaaaaaaaaaaa")
    (output / "record.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        _new_output_directory(str(output), "a" * 64, root)


@pytest.mark.parametrize(
    ("relative_path", "expected_commands"),
    [
        ("Makefile", 1),
        (".github/workflows/ci.yml", 1),
        ("paper/PAPER.md", 2),
    ],
)
def test_maintained_eval_entrypoints_declare_validation_partition(
    relative_path, expected_commands
):
    text = (REPO_ROOT / relative_path).read_text()
    commands = [line.strip() for line in text.splitlines() if "eval.run_eval" in line]

    assert len(commands) == expected_commands
    assert all("--dataset-partition validation" in command for command in commands)
