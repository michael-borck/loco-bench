"""Tests for config preset loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.moe.config import (
    Preset,
    load_presets,
    preset_to_llama_cpp_args,
)


def test_load_presets_returns_named_dict(tmp_path: Path) -> None:
    yaml_text = """
presets:
  default:
    flags:
      ngl: 99
  optimized-moe:
    flags:
      ngl: 99
      n_cpu_moe: 35
      no_mmap: true
      mlock: true
      cache_type_k: q4_0
      cache_type_v: q3_0
      ctx_size: 262144
"""
    cfg_path = tmp_path / "presets.yaml"
    cfg_path.write_text(yaml_text)

    presets = load_presets(cfg_path)

    assert set(presets.keys()) == {"default", "optimized-moe"}
    assert isinstance(presets["default"], Preset)
    assert presets["optimized-moe"].flags["n_cpu_moe"] == 35


def test_preset_to_llama_cpp_args_emits_dashes(sample_preset_yaml: dict) -> None:
    preset = Preset(name=sample_preset_yaml["name"], flags=sample_preset_yaml["flags"])

    args = preset_to_llama_cpp_args(preset)

    assert "-ngl" in args
    assert "99" in args
    assert "--n-cpu-moe" in args
    assert "35" in args
    assert "--no-mmap" in args
    assert "--mlock" in args
    assert "--cache-type-k" in args
    assert "q4_0" in args
    assert "--cache-type-v" in args
    assert "q3_0" in args
    assert "-c" in args
    assert "262144" in args


def test_preset_omits_false_boolean_flags() -> None:
    preset = Preset(name="t", flags={"ngl": 99, "no_mmap": False, "mlock": False})

    args = preset_to_llama_cpp_args(preset)

    assert "--no-mmap" not in args
    assert "--mlock" not in args


from scripts.moe.config import ModelEntry, load_models


def test_load_models_returns_dict(tmp_path: Path) -> None:
    yaml_text = """
models:
  qwen3-30b-a3b:
    hf_repo: Qwen/Qwen3-30B-A3B-GGUF
    filename: qwen3-30b-a3b-q4_k_m.gguf
    sha256: abc123
    tokenizer: Qwen/Qwen3-30B-A3B
    params_total_b: 30
    params_active_b: 3
"""
    cfg_path = tmp_path / "models.yaml"
    cfg_path.write_text(yaml_text)

    models = load_models(cfg_path)

    assert "qwen3-30b-a3b" in models
    entry = models["qwen3-30b-a3b"]
    assert isinstance(entry, ModelEntry)
    assert entry.params_active_b == 3
    assert entry.sha256 == "abc123"
