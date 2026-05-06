"""Tests for per-cell result storage."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.moe.result_storage import CellPaths, ensure_cell_dir, write_run_config


def test_ensure_cell_dir_creates_layout(tmp_results_dir: Path) -> None:
    paths = ensure_cell_dir(
        results_root=tmp_results_dir,
        tier="6gb",
        model="qwen3-30b-a3b",
        preset="optimized-moe",
    )

    assert isinstance(paths, CellPaths)
    assert paths.dir.exists()
    assert paths.dir == tmp_results_dir / "6gb" / "qwen3-30b-a3b" / "optimized-moe"
    assert paths.config_yaml == paths.dir / "config.yaml"
    assert paths.lm_eval_json == paths.dir / "lm_eval.json"
    assert paths.niah_json == paths.dir / "niah.json"
    assert paths.llama_bench_json == paths.dir / "llama_bench.json"
    assert paths.resources_json == paths.dir / "resources.json"
    assert paths.stability_json == paths.dir / "stability.json"
    assert paths.run_log == paths.dir / "run.log"


def test_write_run_config_serializes_full_state(tmp_results_dir: Path) -> None:
    paths = ensure_cell_dir(tmp_results_dir, "6gb", "qwen3-30b-a3b", "optimized-moe")

    write_run_config(
        paths=paths,
        run_id="6gb-1060-qwen3-30b-a3b-optimized-moe",
        host={"motherboard": "X99M-A", "ram_gb": 32},
        gpu={"model": "GTX 1060", "vram_gb": 6},
        container={"image": "ghcr.io/ggml-org/llama.cpp:server-cuda-deadbeef", "ipc_lock": True},
        model={"name": "qwen3-30b-a3b", "sha256": "abc"},
        preset_name="optimized-moe",
        preset_flags={"ngl": 99, "n_cpu_moe": 36, "ctx_size": 262144},
    )

    text = paths.config_yaml.read_text()
    assert "X99M-A" in text
    assert "deadbeef" in text
    assert "n_cpu_moe: 36" in text
    assert "run_id: 6gb-1060-qwen3-30b-a3b-optimized-moe" in text
