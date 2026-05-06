"""Tests for cell-matrix generator."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.moe.generate_cells import MATRIX, generate_cells


def test_matrix_includes_all_headline_cells() -> None:
    cell_ids = {(c["tier"], c["model"], c["preset"]) for c in MATRIX}
    # Article reproduction (6GB / qwen3-30b-a3b / optimized-moe) must be present
    assert ("6gb", "qwen3-30b-a3b", "optimized-moe") in cell_ids
    # OLMoE vs TinyLlama at 2GB
    assert ("2gb", "olmoe-1b-7b", "optimized-moe") in cell_ids
    assert ("2gb", "tinyllama-1.1b", "default") in cell_ids
    # Qwen3-Next-80B-A3B at 8GB (RAM-unlock test)
    assert ("8gb-64gbram", "qwen3-next-80b-a3b", "optimized-moe") in cell_ids


def test_each_cell_has_run_id_pattern() -> None:
    for c in MATRIX:
        assert c["run_id"].startswith(c["tier"]) or c["tier"] in c["run_id"]


def test_generate_cells_writes_yaml_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "cells"

    written = generate_cells(out_dir)

    assert len(written) == len(MATRIX)
    for path in written:
        assert path.exists()
        assert path.suffix == ".yaml"
