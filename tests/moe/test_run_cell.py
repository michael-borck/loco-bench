"""Tests for the cell driver's wiring (not actual benchmark execution)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.moe.run_cell import resolve_run_inputs


def test_resolve_run_inputs_merges_preset_and_overrides(tmp_path: Path) -> None:
    presets_yaml = tmp_path / "presets.yaml"
    presets_yaml.write_text("""
presets:
  optimized-moe:
    flags:
      ngl: 99
      n_cpu_moe: 99
      no_mmap: true
      mlock: true
      cache_type_k: q4_0
      cache_type_v: q3_0
      ctx_size: 262144
""")
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text("""
models:
  qwen3-30b-a3b:
    hf_repo: Qwen/Qwen3-30B-A3B-GGUF
    filename: qwen3-30b-a3b-q4_k_m.gguf
    sha256: abc
    tokenizer: Qwen/Qwen3-30B-A3B
    params_total_b: 30
    params_active_b: 3
""")
    cell_yaml = tmp_path / "cell.yaml"
    cell_yaml.write_text("""
cell:
  run_id: test-run
  tier: 6gb
  model: qwen3-30b-a3b
  preset: optimized-moe
  preset_overrides:
    n_cpu_moe: 35
    ctx_size: 65536
""")

    inputs = resolve_run_inputs(
        cell_path=cell_yaml,
        presets_path=presets_yaml,
        models_path=models_yaml,
    )

    assert inputs.cell.run_id == "test-run"
    assert inputs.model_entry.name == "qwen3-30b-a3b"
    assert inputs.effective_preset.flags["n_cpu_moe"] == 35
    assert inputs.effective_preset.flags["ctx_size"] == 65536
    assert inputs.effective_preset.flags["mlock"] is True
