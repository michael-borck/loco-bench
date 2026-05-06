"""Shared pytest fixtures for MoE harness tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Create a temp results/moe-budget root for a test."""
    results = tmp_path / "results" / "moe-budget"
    results.mkdir(parents=True)
    return results


@pytest.fixture
def sample_preset_yaml() -> dict[str, Any]:
    """Minimal optimized-moe preset for tests."""
    return {
        "name": "optimized-moe",
        "flags": {
            "ngl": 99,
            "n_cpu_moe": 35,
            "no_mmap": True,
            "mlock": True,
            "cache_type_k": "q4_0",
            "cache_type_v": "q3_0",
            "ctx_size": 262144,
        },
    }


@pytest.fixture
def sample_model_entry() -> dict[str, Any]:
    """Minimal model registry entry for tests."""
    return {
        "name": "qwen3-30b-a3b",
        "hf_repo": "Qwen/Qwen3-30B-A3B-GGUF",
        "filename": "qwen3-30b-a3b-q4_k_m.gguf",
        "sha256": "0" * 64,
        "tokenizer": "Qwen/Qwen3-30B-A3B",
        "params_total_b": 30,
        "params_active_b": 3,
    }
