# LocoBench MoE-on-a-Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark harness that boots `llama.cpp server` with named config presets, runs quality + speed + memory + stability + long-context probes, and produces reproducible results across 2–8GB consumer GPU tiers — then execute it across the spec's matrix to answer the five headline questions.

**Architecture:** New `scripts/moe/` Python package wraps `llama.cpp server` (OpenAI-compatible API) and feeds it to `lm-eval-harness` via `--model openai-completions`. Per-cell results land in `results/moe-budget/<tier>/<model>/<preset>/` with a load-bearing `config.yaml` capturing exact flags + GGUF SHA-256 + hardware fingerprint. Existing dense-model scripts (`scripts/benchmark_speed.py`, `scripts/benchmark_quality.py`) are not modified — they continue to run their own dense matrix. New code is isolated in `scripts/moe/` and `tests/moe/`.

**Tech Stack:** Python 3.10+, `llama.cpp` (Docker container, GHCR), `lm-evaluation-harness`, `requests`, `PyYAML`, `pytest`. Existing `pyproject.toml` is extended with a `moe` optional dependency group.

**Spec reference:** `docs/superpowers/specs/2026-05-06-locobench-moe-budget-design.md`

---

## File structure

### New files (all under `loco-bench/`)

| Path | Responsibility |
|---|---|
| `scripts/moe/__init__.py` | package marker |
| `scripts/moe/config.py` | load/validate `config.yaml` preset files |
| `scripts/moe/server.py` | boot `llama.cpp server` with a config preset, wait `/health`, return URL + handle |
| `scripts/moe/resource_probe.py` | poll `nvidia-smi` + `/proc/<pid>/status`, write `resources.json` |
| `scripts/moe/result_storage.py` | write per-cell directory layout (config.yaml + JSON outputs) |
| `scripts/moe/quality.py` | run `lm_eval --model openai-completions` against a running server, parse output |
| `scripts/moe/niah.py` | needle-in-a-haystack long-context probe at 1K/8K/64K/256K |
| `scripts/moe/llama_bench.py` | wrap `llama-bench` for tok/s + TTFT at multiple context lengths |
| `scripts/moe/stability.py` | warmup → idle interval → re-measure, log degradation |
| `scripts/moe/flag_sweep.py` | the article's 6-step incremental flag-attribution sweep |
| `scripts/moe/run_cell.py` | top-level driver: takes (tier, model, preset) → runs all benchmarks → writes outputs |
| `scripts/moe/hardware_fingerprint.py` | capture host CPU/RAM/GPU/kernel into `config.yaml` host block |
| `scripts/moe/aggregate.py` | walk `results/moe-budget/` and produce summary tables/plots |
| `configs/moe-budget/presets.yaml` | named flag presets (default, optimized-moe, optimized-dense) |
| `configs/moe-budget/models.yaml` | model registry: name → HF repo + filename + SHA-256 + tokenizer |
| `configs/moe-budget/cells/*.yaml` | per-cell run configs (tier × model × preset) |
| `tests/moe/__init__.py` | test package marker |
| `tests/moe/conftest.py` | pytest fixtures (mock server, temp dirs) |
| `tests/moe/test_config.py` | config loading and validation tests |
| `tests/moe/test_server.py` | server wrapper tests (mocked subprocess) |
| `tests/moe/test_resource_probe.py` | resource probe tests (mocked /proc, nvidia-smi) |
| `tests/moe/test_result_storage.py` | output layout tests |
| `tests/moe/test_niah.py` | NIAH prompt construction + scoring tests |
| `tests/moe/test_stability.py` | stability runner tests (mocked time) |
| `tests/moe/test_flag_sweep.py` | flag sweep ordering and config generation tests |
| `docs/moe-on-a-budget.md` | the primary narrative writeup |
| `docs/the-five-flags.md` | standalone copy-paste recipe doc |
| `results/moe-budget/.gitkeep` | reserve directory in git (per-run outputs are committed) |

### Modified files

| Path | Change |
|---|---|
| `pyproject.toml` | add `moe` optional-dependency group with `requests`, `PyYAML`, `pytest`, `huggingface-hub` |
| `README.md` | add "Best MoE that fits" column to tier table; add link to `docs/moe-on-a-budget.md` |
| `mkdocs.yml` | add nav entries for `moe-on-a-budget.md` and `the-five-flags.md` |

### Existing files NOT touched

`scripts/benchmark_speed.py`, `scripts/benchmark_quality.py`, `scripts/download_models.sh` (extended in a new task, not modified — see Task 11), `scripts/generate_chart_data.py`, `scripts/convert_and_quantize.sh`, `scripts/gpu-onboard.sh`. The dense-model matrix continues to run as today.

---

## Phase 0: Project bootstrap

### Task 0: Add `moe` optional dependency group and create skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/moe/__init__.py`
- Create: `tests/moe/__init__.py`
- Create: `tests/moe/conftest.py`
- Create: `results/moe-budget/.gitkeep`

- [ ] **Step 1: Add dependency group to pyproject.toml**

Open `pyproject.toml` and add to `[project.optional-dependencies]` (the section already exists with `eval` and `docs` groups):

```toml
moe = [
    "requests>=2.31",
    "PyYAML>=6.0",
    "pytest>=7.4",
    "huggingface-hub>=0.20",
    "lm-eval[openai]",
]
```

- [ ] **Step 2: Create scripts/moe/__init__.py**

Write to `scripts/moe/__init__.py`:

```python
"""LocoBench MoE-on-a-budget harness.

Boots llama.cpp server with named config presets and runs the spec's
benchmark suite (quality, speed, memory, stability, long-context).
See docs/superpowers/specs/2026-05-06-locobench-moe-budget-design.md.
"""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create tests/moe/__init__.py**

Write to `tests/moe/__init__.py`:

```python
```

(empty file — package marker)

- [ ] **Step 4: Create tests/moe/conftest.py**

Write to `tests/moe/conftest.py`:

```python
"""Shared pytest fixtures for MoE harness tests."""
from __future__ import annotations

import json
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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))
```

- [ ] **Step 5: Create results/moe-budget/.gitkeep**

```bash
touch results/moe-budget/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml scripts/moe/ tests/moe/ results/moe-budget/.gitkeep
git commit -m "Add MoE harness scaffolding and dependency group"
```

---

## Phase 1: Config loading

### Task 1: Config presets file and loader

**Files:**
- Create: `configs/moe-budget/presets.yaml`
- Create: `scripts/moe/config.py`
- Test: `tests/moe/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/moe/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd loco-bench
python -m pytest tests/moe/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.moe.config'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/moe/config.py`:

```python
"""Load and translate config presets to llama.cpp server CLI args."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Preset:
    """A named bundle of llama.cpp server flags."""
    name: str
    flags: dict[str, Any]


# Maps internal flag names (snake_case) to llama.cpp CLI flag names
FLAG_TRANSLATION: dict[str, str] = {
    "ngl": "-ngl",
    "n_cpu_moe": "--n-cpu-moe",
    "no_mmap": "--no-mmap",
    "mlock": "--mlock",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "ctx_size": "-c",
    "threads": "-t",
    "batch_size": "-b",
    "ubatch_size": "-ub",
}


def load_presets(path: Path) -> dict[str, Preset]:
    """Parse a presets YAML file. Returns name -> Preset."""
    raw = yaml.safe_load(path.read_text())
    presets_raw = raw.get("presets", {})
    return {
        name: Preset(name=name, flags=dict(body.get("flags", {})))
        for name, body in presets_raw.items()
    }


def preset_to_llama_cpp_args(preset: Preset) -> list[str]:
    """Translate a Preset to a list of CLI args for llama.cpp server."""
    args: list[str] = []
    for key, value in preset.flags.items():
        cli = FLAG_TRANSLATION.get(key)
        if cli is None:
            raise ValueError(f"Unknown flag in preset {preset.name!r}: {key!r}")
        if isinstance(value, bool):
            if value:
                args.append(cli)
        else:
            args.extend([cli, str(value)])
    return args
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/moe/test_config.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Create the actual presets.yaml**

Write to `configs/moe-budget/presets.yaml`:

```yaml
# Named flag bundles for the llama.cpp server. Each cell config references one of these by name.
# See docs/superpowers/specs/2026-05-06-locobench-moe-budget-design.md §5.

presets:
  default:
    description: "Stock llama.cpp server: auto -ngl, mmap on, no mlock, KV cache Q8 (lossless)"
    flags:
      ngl: 99               # llama.cpp will silently cap at actual layer count if too high
      ctx_size: 8192        # default modest context

  optimized-moe:
    description: "The article's 5 flags applied as a bundle"
    flags:
      ngl: 99
      n_cpu_moe: 35         # tuned per tier — overridden in cell config
      no_mmap: true
      mlock: true
      cache_type_k: q4_0    # Turbo Quant K
      cache_type_v: q3_0    # Turbo Quant V
      ctx_size: 262144      # 256K target

  optimized-dense:
    description: "no-mmap + mlock only. KV Turbo Quant excluded for dense models."
    flags:
      ngl: 99
      no_mmap: true
      mlock: true
      ctx_size: 8192
```

- [ ] **Step 6: Commit**

```bash
git add scripts/moe/config.py tests/moe/test_config.py configs/moe-budget/presets.yaml
git commit -m "Add config preset loader with llama.cpp arg translation"
```

---

### Task 2: Model registry loader

**Files:**
- Create: `configs/moe-budget/models.yaml`
- Modify: `scripts/moe/config.py` (add model loading)
- Modify: `tests/moe/test_config.py` (add model loader tests)

- [ ] **Step 1: Add failing test for model loading**

Append to `tests/moe/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_config.py::test_load_models_returns_dict -v
```

Expected: FAIL with `ImportError: cannot import name 'ModelEntry'`

- [ ] **Step 3: Add ModelEntry and load_models to config.py**

Append to `scripts/moe/config.py`:

```python
@dataclass
class ModelEntry:
    """Registry entry for one GGUF model."""
    name: str
    hf_repo: str
    filename: str
    sha256: str
    tokenizer: str
    params_total_b: float
    params_active_b: float


def load_models(path: Path) -> dict[str, ModelEntry]:
    """Parse a models YAML file. Returns name -> ModelEntry."""
    raw = yaml.safe_load(path.read_text())
    models_raw = raw.get("models", {})
    return {
        name: ModelEntry(
            name=name,
            hf_repo=body["hf_repo"],
            filename=body["filename"],
            sha256=body["sha256"],
            tokenizer=body["tokenizer"],
            params_total_b=float(body["params_total_b"]),
            params_active_b=float(body["params_active_b"]),
        )
        for name, body in models_raw.items()
    }
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/moe/test_config.py -v
```

Expected: PASS (4 tests total)

- [ ] **Step 5: Create the actual models.yaml**

Write to `configs/moe-budget/models.yaml` (SHA-256 values are placeholders to be filled in Task 11 after download — flagged with `TBD-FILL-IN-TASK-17` so a grep can find them):

```yaml
# Model registry for the MoE-on-a-budget study.
# SHA-256 values are populated in Task 17 (download + verify) and committed
# alongside the downloaded models. The string "TBD-FILL-IN-TASK-17" is a
# load-bearing sentinel — Task 17 fails fast if any remain.

models:
  # MoE candidates
  olmoe-1b-7b:
    hf_repo: allenai/OLMoE-1B-7B-0924-Instruct-GGUF
    filename: olmoe-1b-7b-0924-instruct-q4_k_m.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: allenai/OLMoE-1B-7B-0924-Instruct
    params_total_b: 6.9
    params_active_b: 1.0

  qwen1.5-moe-a2.7b:
    hf_repo: Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF
    filename: qwen1.5-moe-a2.7b-chat-q4_k_m.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: Qwen/Qwen1.5-MoE-A2.7B-Chat
    params_total_b: 14.3
    params_active_b: 2.7

  deepseek-v2-lite:
    hf_repo: bartowski/DeepSeek-V2-Lite-Chat-GGUF
    filename: DeepSeek-V2-Lite-Chat-Q4_K_M.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: deepseek-ai/DeepSeek-V2-Lite-Chat
    params_total_b: 15.7
    params_active_b: 2.4

  qwen3-30b-a3b:
    hf_repo: Qwen/Qwen3-30B-A3B-GGUF
    filename: qwen3-30b-a3b-q4_k_m.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: Qwen/Qwen3-30B-A3B
    params_total_b: 30.0
    params_active_b: 3.0

  qwen3-next-80b-a3b:
    hf_repo: Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF
    filename: qwen3-next-80b-a3b-instruct-q4_k_m.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: Qwen/Qwen3-Next-80B-A3B-Instruct
    params_total_b: 80.0
    params_active_b: 3.0

  # Dense baselines
  tinyllama-1.1b:
    hf_repo: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
    filename: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    params_total_b: 1.1
    params_active_b: 1.1

  qwen3-4b-instruct:
    hf_repo: Qwen/Qwen3-4B-Instruct-GGUF
    filename: qwen3-4b-instruct-q4_k_m.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: Qwen/Qwen3-4B-Instruct
    params_total_b: 4.0
    params_active_b: 4.0

  phi-4-mini-instruct:
    hf_repo: bartowski/Phi-4-mini-instruct-GGUF
    filename: Phi-4-mini-instruct-Q4_K_M.gguf
    sha256: TBD-FILL-IN-TASK-17
    tokenizer: microsoft/Phi-4-mini-instruct
    params_total_b: 3.8
    params_active_b: 3.8
```

> **Note for the implementer:** HF repo paths and exact filenames must be verified before downloading. If a listed repo does not exist or the GGUF filename differs (common — repos use varied casing/naming), update `models.yaml` and re-commit before Task 11. Do not silently fall back to a different file.

- [ ] **Step 6: Commit**

```bash
git add scripts/moe/config.py tests/moe/test_config.py configs/moe-budget/models.yaml
git commit -m "Add model registry loader and models.yaml"
```

---

### Task 3: Cell config loader

**Files:**
- Modify: `scripts/moe/config.py` (add cell loading)
- Modify: `tests/moe/test_config.py` (add cell tests)
- Create: `configs/moe-budget/cells/example.yaml` (one example; full set generated in Task 4)

- [ ] **Step 1: Add failing test**

Append to `tests/moe/test_config.py`:

```python
from scripts.moe.config import CellConfig, load_cell_config


def test_load_cell_config(tmp_path: Path) -> None:
    yaml_text = """
cell:
  run_id: 2026-05-08-1060-qwen3-30b-a3b-optimized-moe
  tier: 6gb
  model: qwen3-30b-a3b
  preset: optimized-moe
  preset_overrides:
    n_cpu_moe: 35
    ctx_size: 65536
"""
    cfg_path = tmp_path / "cell.yaml"
    cfg_path.write_text(yaml_text)

    cell = load_cell_config(cfg_path)

    assert isinstance(cell, CellConfig)
    assert cell.tier == "6gb"
    assert cell.model == "qwen3-30b-a3b"
    assert cell.preset == "optimized-moe"
    assert cell.preset_overrides == {"n_cpu_moe": 35, "ctx_size": 65536}
    assert cell.run_id == "2026-05-08-1060-qwen3-30b-a3b-optimized-moe"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_config.py::test_load_cell_config -v
```

Expected: FAIL with `ImportError: cannot import name 'CellConfig'`

- [ ] **Step 3: Add CellConfig and loader**

Append to `scripts/moe/config.py`:

```python
@dataclass
class CellConfig:
    """One cell of the (tier × model × preset) result matrix."""
    run_id: str
    tier: str
    model: str
    preset: str
    preset_overrides: dict[str, Any]


def load_cell_config(path: Path) -> CellConfig:
    """Parse a cell config YAML file."""
    raw = yaml.safe_load(path.read_text())
    body = raw["cell"]
    return CellConfig(
        run_id=body["run_id"],
        tier=body["tier"],
        model=body["model"],
        preset=body["preset"],
        preset_overrides=dict(body.get("preset_overrides", {})),
    )


def resolve_preset(preset: Preset, overrides: dict[str, Any]) -> Preset:
    """Apply cell-level overrides on top of a base preset."""
    merged = dict(preset.flags)
    merged.update(overrides)
    return Preset(name=preset.name, flags=merged)
```

- [ ] **Step 4: Add a test for resolve_preset**

Append to `tests/moe/test_config.py`:

```python
from scripts.moe.config import resolve_preset


def test_resolve_preset_overrides_base(sample_preset_yaml: dict) -> None:
    base = Preset(name=sample_preset_yaml["name"], flags=sample_preset_yaml["flags"])

    merged = resolve_preset(base, {"n_cpu_moe": 36, "ctx_size": 65536})

    assert merged.flags["n_cpu_moe"] == 36
    assert merged.flags["ctx_size"] == 65536
    assert merged.flags["mlock"] is True  # untouched by override
```

- [ ] **Step 5: Run test**

```bash
python -m pytest tests/moe/test_config.py -v
```

Expected: PASS (6 tests total)

- [ ] **Step 6: Create example cell config**

Write to `configs/moe-budget/cells/example.yaml`:

```yaml
# Example cell config. Real cell configs are generated by Task 4 across the matrix.

cell:
  run_id: 2026-05-08-1060-qwen3-30b-a3b-optimized-moe
  tier: 6gb
  model: qwen3-30b-a3b
  preset: optimized-moe
  preset_overrides:
    n_cpu_moe: 35    # tuned for 1060 6GB per article
    ctx_size: 262144 # full Turbo Quant context
```

- [ ] **Step 7: Commit**

```bash
git add scripts/moe/config.py tests/moe/test_config.py configs/moe-budget/cells/example.yaml
git commit -m "Add cell config loader with preset overrides"
```

---

### Task 4: Generate the full cell matrix from spec

**Files:**
- Create: `scripts/moe/generate_cells.py`
- Test: `tests/moe/test_generate_cells.py`
- Create: `configs/moe-budget/cells/*.yaml` (output of running the generator)

- [ ] **Step 1: Write the failing test**

Create `tests/moe/test_generate_cells.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_generate_cells.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the generator**

Create `scripts/moe/generate_cells.py`:

```python
"""Generate the full (tier × model × preset) cell matrix from spec.

The MATRIX constant is the single source of truth for which cells get run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


# Per-tier `n_cpu_moe` defaults — tuned per VRAM budget. Confirmed empirically in calibration tasks.
N_CPU_MOE_PER_TIER: dict[str, int] = {
    "2gb": 99,            # near-all on CPU; only attention head + small embeddings on tiny VRAM
    "4gb": 40,
    "6gb": 35,            # article's value for 1060 6GB at 64K context
    "6gb-256k": 36,       # article's value for 1060 6GB at 256K context with Turbo Quant
    "8gb-32gbram": 30,
    "8gb-64gbram": 30,    # same n_cpu_moe; total RAM is the variable, not VRAM
}


# The cell matrix. Each entry produces one cell config YAML.
MATRIX: list[dict[str, Any]] = [
    # 2GB tier — OLMoE vs TinyLlama
    {
        "run_id": "2gb-950-olmoe-1b-7b-default",
        "tier": "2gb", "model": "olmoe-1b-7b", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },
    {
        "run_id": "2gb-950-olmoe-1b-7b-optimized-moe",
        "tier": "2gb", "model": "olmoe-1b-7b", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["2gb"], "ctx_size": 4096},
    },
    {
        "run_id": "2gb-950-tinyllama-default",
        "tier": "2gb", "model": "tinyllama-1.1b", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },

    # 4GB tier — Qwen3-30B-A3B floor test, Qwen1.5-MoE, DeepSeek-V2-Lite, Qwen3-4B baseline
    {
        "run_id": "4gb-1050ti-qwen3-30b-a3b-default",
        "tier": "4gb", "model": "qwen3-30b-a3b", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },
    {
        "run_id": "4gb-1050ti-qwen3-30b-a3b-optimized-moe",
        "tier": "4gb", "model": "qwen3-30b-a3b", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["4gb"], "ctx_size": 65536},
    },
    {
        "run_id": "4gb-1050ti-qwen1.5-moe-default",
        "tier": "4gb", "model": "qwen1.5-moe-a2.7b", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },
    {
        "run_id": "4gb-1050ti-qwen1.5-moe-optimized-moe",
        "tier": "4gb", "model": "qwen1.5-moe-a2.7b", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["4gb"], "ctx_size": 32768},
    },
    {
        "run_id": "4gb-1050ti-deepseek-v2-lite-default",
        "tier": "4gb", "model": "deepseek-v2-lite", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },
    {
        "run_id": "4gb-1050ti-deepseek-v2-lite-optimized-moe",
        "tier": "4gb", "model": "deepseek-v2-lite", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["4gb"], "ctx_size": 32768},
    },
    {
        "run_id": "4gb-1050ti-qwen3-4b-default",
        "tier": "4gb", "model": "qwen3-4b-instruct", "preset": "default",
        "preset_overrides": {"ctx_size": 8192},
    },

    # 6GB tier — article reproduction
    {
        "run_id": "6gb-1060-qwen3-30b-a3b-default",
        "tier": "6gb", "model": "qwen3-30b-a3b", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },
    {
        "run_id": "6gb-1060-qwen3-30b-a3b-optimized-moe",
        "tier": "6gb", "model": "qwen3-30b-a3b", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["6gb-256k"], "ctx_size": 262144},
    },
    {
        "run_id": "6gb-1060-qwen3-4b-default",
        "tier": "6gb", "model": "qwen3-4b-instruct", "preset": "default",
        "preset_overrides": {"ctx_size": 8192},
    },

    # 8GB tier — 32GB RAM (initial config)
    {
        "run_id": "8gb-32gbram-2060super-qwen3-30b-a3b-default",
        "tier": "8gb-32gbram", "model": "qwen3-30b-a3b", "preset": "default",
        "preset_overrides": {"ctx_size": 4096},
    },
    {
        "run_id": "8gb-32gbram-2060super-qwen3-30b-a3b-optimized-moe",
        "tier": "8gb-32gbram", "model": "qwen3-30b-a3b", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["8gb-32gbram"], "ctx_size": 262144},
    },
    {
        "run_id": "8gb-32gbram-2060super-phi-4-mini-default",
        "tier": "8gb-32gbram", "model": "phi-4-mini-instruct", "preset": "default",
        "preset_overrides": {"ctx_size": 8192},
    },

    # 8GB tier — 64GB RAM (the unlock test)
    {
        "run_id": "8gb-64gbram-2060super-qwen3-next-80b-a3b-optimized-moe",
        "tier": "8gb-64gbram", "model": "qwen3-next-80b-a3b", "preset": "optimized-moe",
        "preset_overrides": {"n_cpu_moe": N_CPU_MOE_PER_TIER["8gb-64gbram"], "ctx_size": 65536},
    },
]


def generate_cells(out_dir: Path) -> list[Path]:
    """Write one YAML file per cell. Returns list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cell in MATRIX:
        path = out_dir / f"{cell['run_id']}.yaml"
        body = {"cell": cell}
        path.write_text(yaml.safe_dump(body, sort_keys=False))
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-cell config YAMLs.")
    parser.add_argument(
        "--out", type=Path, default=Path("configs/moe-budget/cells"),
        help="Output directory for cell YAMLs.",
    )
    args = parser.parse_args()
    written = generate_cells(args.out)
    print(f"Wrote {len(written)} cell configs to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/moe/test_generate_cells.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Generate the actual cell files**

```bash
python scripts/moe/generate_cells.py
```

Expected output: `Wrote 17 cell configs to configs/moe-budget/cells`

Verify with `ls configs/moe-budget/cells/ | wc -l` → 17 (plus the `example.yaml` from Task 3 = 18).

- [ ] **Step 6: Commit**

```bash
git add scripts/moe/generate_cells.py tests/moe/test_generate_cells.py configs/moe-budget/cells/
git commit -m "Generate full cell matrix (17 cells across 4 tiers)"
```

---

## Phase 2: llama.cpp server wrapper

### Task 5: llama.cpp server boot wrapper

**Files:**
- Create: `scripts/moe/server.py`
- Test: `tests/moe/test_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_server.py`:

```python
"""Tests for llama.cpp server wrapper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.moe.config import Preset
from scripts.moe.server import (
    LlamaCppServer,
    ServerConfig,
    build_docker_command,
)


def test_build_docker_command_includes_preset_args() -> None:
    preset = Preset(
        name="optimized-moe",
        flags={"ngl": 99, "n_cpu_moe": 35, "no_mmap": True, "mlock": True, "ctx_size": 65536},
    )
    cfg = ServerConfig(
        image="ghcr.io/ggml-org/llama.cpp:server-cuda-test",
        gguf_path=Path("/models/qwen3-30b-a3b-q4_k_m.gguf"),
        host_port=8080,
        ipc_lock=True,
    )

    cmd = build_docker_command(cfg, preset)

    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--rm" in cmd
    assert "--gpus" in cmd and "all" in cmd
    assert "--cap-add" in cmd and "IPC_LOCK" in cmd
    assert "-p" in cmd and "8080:8080" in cmd
    assert "ghcr.io/ggml-org/llama.cpp:server-cuda-test" in cmd
    # Model arg
    assert "-m" in cmd
    # Preset args translated through
    assert "--n-cpu-moe" in cmd
    assert "35" in cmd
    assert "--no-mmap" in cmd
    assert "--mlock" in cmd


def test_build_docker_command_omits_ipc_lock_when_disabled() -> None:
    preset = Preset(name="default", flags={"ngl": 99, "ctx_size": 8192})
    cfg = ServerConfig(
        image="test:image",
        gguf_path=Path("/models/x.gguf"),
        host_port=8080,
        ipc_lock=False,
    )

    cmd = build_docker_command(cfg, preset)

    assert "IPC_LOCK" not in cmd


@patch("scripts.moe.server.requests.get")
@patch("scripts.moe.server.subprocess.Popen")
def test_server_start_polls_health(mock_popen, mock_get) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    mock_popen.return_value = mock_proc

    # Simulate /health returning 200 on second poll
    mock_get.side_effect = [
        MagicMock(status_code=503),
        MagicMock(status_code=200, json=lambda: {"status": "ok"}),
    ]

    preset = Preset(name="default", flags={"ngl": 99, "ctx_size": 8192})
    cfg = ServerConfig(
        image="test:image", gguf_path=Path("/models/x.gguf"),
        host_port=8080, ipc_lock=False,
    )
    server = LlamaCppServer(cfg, preset, health_poll_interval_s=0.0, health_timeout_s=10)

    server.start()

    assert server.url == "http://localhost:8080"
    assert mock_get.call_count == 2
    server.stop()


@patch("scripts.moe.server.requests.get")
@patch("scripts.moe.server.subprocess.Popen")
def test_server_start_raises_on_health_timeout(mock_popen, mock_get) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_get.return_value = MagicMock(status_code=503)

    preset = Preset(name="default", flags={"ngl": 99, "ctx_size": 8192})
    cfg = ServerConfig(
        image="test:image", gguf_path=Path("/models/x.gguf"),
        host_port=8080, ipc_lock=False,
    )
    server = LlamaCppServer(cfg, preset, health_poll_interval_s=0.0, health_timeout_s=0.05)

    with pytest.raises(TimeoutError):
        server.start()
    server.stop()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_server.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.moe.server'`

- [ ] **Step 3: Write the server wrapper**

Create `scripts/moe/server.py`:

```python
"""Boot a llama.cpp server in Docker, wait for /health, return URL."""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from scripts.moe.config import Preset, preset_to_llama_cpp_args


@dataclass
class ServerConfig:
    """Container + host config for one llama.cpp server boot."""
    image: str
    gguf_path: Path
    host_port: int = 8080
    ipc_lock: bool = True
    container_name: str | None = None


def build_docker_command(cfg: ServerConfig, preset: Preset) -> list[str]:
    """Compose the `docker run ...` command line for a server boot."""
    cmd: list[str] = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-p", f"{cfg.host_port}:8080",
        "-v", f"{cfg.gguf_path.parent}:/models:ro",
    ]
    if cfg.ipc_lock:
        cmd += ["--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1"]
    if cfg.container_name:
        cmd += ["--name", cfg.container_name]

    cmd += [
        cfg.image,
        "--server",
        "--host", "0.0.0.0",
        "--port", "8080",
        "-m", f"/models/{cfg.gguf_path.name}",
    ]
    cmd += preset_to_llama_cpp_args(preset)
    return cmd


class LlamaCppServer:
    """Lifecycle wrapper for a llama.cpp Docker server."""

    def __init__(
        self,
        cfg: ServerConfig,
        preset: Preset,
        health_poll_interval_s: float = 2.0,
        health_timeout_s: float = 600.0,
    ) -> None:
        self.cfg = cfg
        self.preset = preset
        self._health_interval = health_poll_interval_s
        self._health_timeout = health_timeout_s
        self._proc: subprocess.Popen | None = None
        self.url = f"http://localhost:{cfg.host_port}"

    def start(self) -> None:
        cmd = build_docker_command(self.cfg, self.preset)
        print(f"[server] launching: {' '.join(shlex.quote(c) for c in cmd)}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_for_health()

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._health_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp server exited early with code {self._proc.returncode}"
                )
            try:
                r = requests.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(self._health_interval)
        raise TimeoutError(f"llama.cpp server did not become healthy within {self._health_timeout}s")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def __enter__(self) -> "LlamaCppServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/moe/test_server.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/server.py tests/moe/test_server.py
git commit -m "Add llama.cpp server wrapper with Docker + health polling"
```

---

## Phase 3: Resource probe and storage

### Task 6: Resource probe (VRAM / RAM / mlocked)

**Files:**
- Create: `scripts/moe/resource_probe.py`
- Test: `tests/moe/test_resource_probe.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_resource_probe.py`:

```python
"""Tests for resource probe."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.moe.resource_probe import (
    ResourceSnapshot,
    parse_nvidia_smi_output,
    parse_proc_status,
    snapshot,
)


def test_parse_nvidia_smi_output_csv() -> None:
    raw = "5824, 6144\n"  # used, total in MB

    used_mb, total_mb = parse_nvidia_smi_output(raw)

    assert used_mb == 5824
    assert total_mb == 6144


def test_parse_proc_status_extracts_rss_lck_hwm() -> None:
    raw = """
Name:   llama-server
VmHWM:    24112844 kB
VmRSS:    24108200 kB
VmLck:    16776832 kB
VmPeak:   24500000 kB
"""

    parsed = parse_proc_status(raw)

    assert parsed["VmRSS_kb"] == 24108200
    assert parsed["VmLck_kb"] == 16776832
    assert parsed["VmHWM_kb"] == 24112844


@patch("scripts.moe.resource_probe._read_proc_status")
@patch("scripts.moe.resource_probe._run_nvidia_smi")
def test_snapshot_combines_sources(mock_smi, mock_proc) -> None:
    mock_smi.return_value = "5824, 6144\n"
    mock_proc.return_value = "VmRSS:    24108200 kB\nVmLck:    16776832 kB\nVmHWM:    24112844 kB\n"

    snap = snapshot(pid=12345)

    assert isinstance(snap, ResourceSnapshot)
    assert snap.vram_used_mb == 5824
    assert snap.vram_total_mb == 6144
    assert snap.rss_kb == 24108200
    assert snap.mlocked_kb == 16776832
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_resource_probe.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write resource_probe.py**

Create `scripts/moe/resource_probe.py`:

```python
"""Probe GPU and process memory state for benchmark snapshots."""
from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResourceSnapshot:
    """One point-in-time snapshot of memory usage."""
    timestamp: float
    vram_used_mb: int
    vram_total_mb: int
    rss_kb: int
    mlocked_kb: int
    hwm_kb: int


def parse_nvidia_smi_output(raw: str) -> tuple[int, int]:
    """Parse `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader` output."""
    line = raw.strip().splitlines()[0]
    used, total = (int(s.strip()) for s in line.split(","))
    return used, total


def parse_proc_status(raw: str) -> dict[str, int]:
    """Parse /proc/<pid>/status into a dict keyed by 'VmRSS_kb', 'VmLck_kb', 'VmHWM_kb'."""
    out: dict[str, int] = {}
    keys = {"VmRSS": "VmRSS_kb", "VmLck": "VmLck_kb", "VmHWM": "VmHWM_kb"}
    for line in raw.splitlines():
        for prefix, dest in keys.items():
            if line.startswith(prefix + ":"):
                value = line.split(":", 1)[1].strip().split()[0]
                out[dest] = int(value)
    return out


def _run_nvidia_smi() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def _read_proc_status(pid: int) -> str:
    return Path(f"/proc/{pid}/status").read_text()


def snapshot(pid: int) -> ResourceSnapshot:
    """Take a single snapshot of GPU + process memory."""
    smi = parse_nvidia_smi_output(_run_nvidia_smi())
    proc = parse_proc_status(_read_proc_status(pid))
    return ResourceSnapshot(
        timestamp=time.time(),
        vram_used_mb=smi[0],
        vram_total_mb=smi[1],
        rss_kb=proc.get("VmRSS_kb", 0),
        mlocked_kb=proc.get("VmLck_kb", 0),
        hwm_kb=proc.get("VmHWM_kb", 0),
    )


def poll_loop(pid: int, duration_s: float, interval_s: float = 1.0) -> list[ResourceSnapshot]:
    """Poll repeatedly for duration_s, return list of snapshots."""
    snaps: list[ResourceSnapshot] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        try:
            snaps.append(snapshot(pid))
        except Exception as e:
            # Process may have exited; record and bail
            print(f"[probe] snapshot failed: {e}")
            break
        time.sleep(interval_s)
    return snaps


def snapshots_to_summary(snaps: list[ResourceSnapshot]) -> dict[str, Any]:
    """Reduce a list of snapshots to a peak summary dict."""
    if not snaps:
        return {"samples": 0}
    return {
        "samples": len(snaps),
        "peak_vram_used_mb": max(s.vram_used_mb for s in snaps),
        "vram_total_mb": snaps[-1].vram_total_mb,
        "peak_rss_kb": max(s.rss_kb for s in snaps),
        "peak_mlocked_kb": max(s.mlocked_kb for s in snaps),
        "peak_hwm_kb": max(s.hwm_kb for s in snaps),
        "snapshots": [asdict(s) for s in snaps],
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_resource_probe.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/resource_probe.py tests/moe/test_resource_probe.py
git commit -m "Add resource probe for VRAM/RAM/mlocked snapshots"
```

---

### Task 7: Result storage layout

**Files:**
- Create: `scripts/moe/result_storage.py`
- Test: `tests/moe/test_result_storage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_result_storage.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_result_storage.py -v
```

Expected: FAIL

- [ ] **Step 3: Write result_storage.py**

Create `scripts/moe/result_storage.py`:

```python
"""Per-cell result directory layout and config.yaml writer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CellPaths:
    """All output paths for one cell of the result matrix."""
    dir: Path
    config_yaml: Path
    lm_eval_json: Path
    niah_json: Path
    llama_bench_json: Path
    resources_json: Path
    stability_json: Path
    run_log: Path


def ensure_cell_dir(
    results_root: Path, tier: str, model: str, preset: str
) -> CellPaths:
    """Create results/moe-budget/<tier>/<model>/<preset>/ and return all output paths."""
    cell_dir = results_root / tier / model / preset
    cell_dir.mkdir(parents=True, exist_ok=True)
    return CellPaths(
        dir=cell_dir,
        config_yaml=cell_dir / "config.yaml",
        lm_eval_json=cell_dir / "lm_eval.json",
        niah_json=cell_dir / "niah.json",
        llama_bench_json=cell_dir / "llama_bench.json",
        resources_json=cell_dir / "resources.json",
        stability_json=cell_dir / "stability.json",
        run_log=cell_dir / "run.log",
    )


def write_run_config(
    *,
    paths: CellPaths,
    run_id: str,
    host: dict[str, Any],
    gpu: dict[str, Any],
    container: dict[str, Any],
    model: dict[str, Any],
    preset_name: str,
    preset_flags: dict[str, Any],
) -> None:
    """Write the load-bearing config.yaml for this cell."""
    body = {
        "run_id": run_id,
        "host": host,
        "gpu": gpu,
        "container": container,
        "model": model,
        "preset": preset_name,
        "flags": preset_flags,
    }
    paths.config_yaml.write_text(yaml.safe_dump(body, sort_keys=False))
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_result_storage.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/result_storage.py tests/moe/test_result_storage.py
git commit -m "Add per-cell result directory layout and config writer"
```

---

## Phase 4: Quality, speed, NIAH, stability

### Task 8: Quality eval driver (lm-eval against llama.cpp server)

**Files:**
- Create: `scripts/moe/quality.py`
- Test: `tests/moe/test_quality.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_quality.py`:

```python
"""Tests for lm-eval quality driver."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.moe.quality import build_lm_eval_command, parse_lm_eval_results


def test_build_lm_eval_command_targets_openai_completions() -> None:
    cmd = build_lm_eval_command(
        server_url="http://localhost:8080",
        tokenizer="Qwen/Qwen3-30B-A3B",
        tasks="mmlu,gsm8k,hellaswag",
        output_path=Path("/tmp/out"),
    )

    assert "lm_eval" in cmd[0] or cmd[0] == "lm_eval"
    assert "--model" in cmd and "openai-completions" in cmd
    joined = " ".join(cmd)
    assert "base_url=http://localhost:8080/v1" in joined
    assert "tokenizer=Qwen/Qwen3-30B-A3B" in joined
    assert "--tasks" in cmd and "mmlu,gsm8k,hellaswag" in cmd


def test_parse_lm_eval_results_extracts_acc(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text("""
{
  "results": {
    "mmlu": {"acc,none": 0.567, "acc_stderr,none": 0.012},
    "gsm8k": {"exact_match,strict-match": 0.234},
    "hellaswag": {"acc_norm,none": 0.612}
  }
}
""")

    parsed = parse_lm_eval_results(results_path)

    assert parsed["mmlu"]["acc"] == pytest.approx(0.567)
    assert parsed["gsm8k"]["exact_match"] == pytest.approx(0.234)
    assert parsed["hellaswag"]["acc_norm"] == pytest.approx(0.612)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_quality.py -v
```

Expected: FAIL

- [ ] **Step 3: Write quality.py**

> **Implementer note on the `--model` value:** lm-evaluation-harness 0.4.x exposes both `openai-completions` (for the actual OpenAI service) and `local-completions` (for OpenAI-compatible local servers like llama.cpp). The exact name varies by lm-eval version. The plan uses `openai-completions` to match the spec narrative; if Task 18 (smoke test) fails with `ValueError: Model 'openai-completions' not found`, change `LM_EVAL_MODEL_NAME` below to `local-completions` and re-run the test.

Create `scripts/moe/quality.py`:

```python
"""Run lm-evaluation-harness against a running llama.cpp server."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_TASKS = "mmlu,gsm8k,hellaswag,truthfulqa_mc2,arc_challenge"

# lm-eval model backend name. Switch to "local-completions" if your lm-eval
# version doesn't accept "openai-completions" for non-OpenAI base_url values.
LM_EVAL_MODEL_NAME = "openai-completions"


def build_lm_eval_command(
    server_url: str,
    tokenizer: str,
    tasks: str,
    output_path: Path,
    apply_chat_template: bool = False,
    num_fewshot: int | None = None,
) -> list[str]:
    """Compose the `lm_eval --model <backend> ...` command."""
    model_args = f"base_url={server_url}/v1,tokenizer={tokenizer}"
    cmd: list[str] = [
        "lm_eval",
        "--model", LM_EVAL_MODEL_NAME,
        "--model_args", model_args,
        "--tasks", tasks,
        "--output_path", str(output_path),
    ]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    return cmd


def run_quality(
    server_url: str,
    tokenizer: str,
    tasks: str,
    output_dir: Path,
    apply_chat_template: bool = False,
) -> Path:
    """Invoke lm_eval and return path to the results JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_lm_eval_command(server_url, tokenizer, tasks, output_dir, apply_chat_template)
    print(f"[quality] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    candidates = list(output_dir.glob("**/results*.json"))
    if not candidates:
        raise RuntimeError(f"lm_eval produced no results JSON in {output_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


# Maps lm_eval canonical metric names to a friendlier short key
RESULT_METRIC_MAP: dict[str, str] = {
    "acc,none": "acc",
    "acc_norm,none": "acc_norm",
    "acc_stderr,none": "acc_stderr",
    "exact_match,strict-match": "exact_match",
    "exact_match,flexible-extract": "exact_match_flexible",
    "mc2,none": "mc2",
}


def parse_lm_eval_results(results_path: Path) -> dict[str, dict[str, float]]:
    """Reduce lm_eval results JSON to {task: {metric_short: value}}."""
    raw = json.loads(results_path.read_text())
    out: dict[str, dict[str, float]] = {}
    for task, metrics in raw.get("results", {}).items():
        out[task] = {}
        for key, value in metrics.items():
            short = RESULT_METRIC_MAP.get(key)
            if short is None:
                continue
            try:
                out[task][short] = float(value)
            except (TypeError, ValueError):
                continue
    return out
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_quality.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/quality.py tests/moe/test_quality.py
git commit -m "Add lm-eval quality driver targeting llama.cpp OpenAI API"
```

---

### Task 9: NIAH long-context probe

**Files:**
- Create: `scripts/moe/niah.py`
- Test: `tests/moe/test_niah.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_niah.py`:

```python
"""Tests for needle-in-a-haystack probe."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.moe.niah import (
    NEEDLE_TEMPLATE,
    build_haystack,
    grade_response,
    score_at_context_lengths,
)


def test_build_haystack_inserts_needle_at_depth() -> None:
    needle = "The magic number is 7421."
    haystack, needle_position = build_haystack(
        needle=needle,
        target_tokens=1000,
        depth_fraction=0.5,
        filler_text="The quick brown fox jumps over the lazy dog. ",
        approx_chars_per_token=4,
    )

    # Needle present
    assert needle in haystack
    # Inserted near the middle
    assert 0.3 < needle_position < 0.7
    # Roughly the right size (within 30% of target)
    estimated_tokens = len(haystack) / 4
    assert 700 < estimated_tokens < 1300


def test_grade_response_correct_extraction() -> None:
    needle = "The magic number is 7421."
    response = "Based on the text, the magic number is 7421."

    assert grade_response(response, needle, expected_value="7421") == 1.0


def test_grade_response_wrong_value() -> None:
    needle = "The magic number is 7421."
    response = "The magic number is 9999."

    assert grade_response(response, needle, expected_value="7421") == 0.0


def test_grade_response_partial_credit() -> None:
    # Response references the needle indirectly
    response = "It looks like 7421 is mentioned somewhere."

    assert grade_response(response, needle="x", expected_value="7421") == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_niah.py -v
```

Expected: FAIL

- [ ] **Step 3: Write niah.py**

Create `scripts/moe/niah.py`:

```python
"""Needle-in-a-haystack long-context retrieval probe.

Inserts a known fact into a long filler document at a controlled depth, asks
the model to retrieve it, grades whether the response contains the expected
value. Run at multiple target context lengths to test KV-cache fidelity.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


NEEDLE_TEMPLATE = "The magic number for project {project} is {value}."

QUESTION_TEMPLATE = (
    "What is the magic number for project {project}? "
    "Answer with just the number."
)


@dataclass
class NiahResult:
    target_tokens: int
    depth_fraction: float
    needle_value: str
    response: str
    score: float


def build_haystack(
    *,
    needle: str,
    target_tokens: int,
    depth_fraction: float,
    filler_text: str,
    approx_chars_per_token: int = 4,
) -> tuple[str, float]:
    """Construct a (haystack, needle_position) pair.

    `depth_fraction` is the target relative position (0=start, 1=end). The actual
    returned position is the fraction of `len(haystack)` where the needle sits.
    """
    target_chars = target_tokens * approx_chars_per_token
    repeat = max(1, target_chars // len(filler_text) + 1)
    body = filler_text * repeat
    body = body[:target_chars]

    insert_at = int(len(body) * depth_fraction)
    haystack = body[:insert_at] + needle + " " + body[insert_at:]
    actual_position = insert_at / len(haystack)
    return haystack, actual_position


def build_filler(seed: int = 0) -> str:
    """Deterministic filler text (no semantic content that could leak the needle)."""
    rng = random.Random(seed)
    sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "Pack my box with five dozen liquor jugs. ",
        "Sphinx of black quartz, judge my vow. ",
        "How vexingly quick daft zebras jump. ",
    ]
    rng.shuffle(sentences)
    return "".join(sentences)


def grade_response(response: str, needle: str, expected_value: str) -> float:
    """Score 1.0 if expected_value appears in response, 0.0 otherwise."""
    return 1.0 if expected_value in response else 0.0


def query_server(
    server_url: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0
) -> str:
    """Send a completion request to a llama.cpp OpenAI-compatible server."""
    r = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": "local",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def run_one_probe(
    *,
    server_url: str,
    target_tokens: int,
    depth_fraction: float,
    project: str,
    value: str,
    filler_text: str,
) -> NiahResult:
    needle = NEEDLE_TEMPLATE.format(project=project, value=value)
    haystack, _ = build_haystack(
        needle=needle,
        target_tokens=target_tokens,
        depth_fraction=depth_fraction,
        filler_text=filler_text,
    )
    question = QUESTION_TEMPLATE.format(project=project)
    prompt = haystack + "\n\n" + question + "\n"
    response = query_server(server_url, prompt)
    score = grade_response(response, needle, expected_value=value)
    return NiahResult(
        target_tokens=target_tokens,
        depth_fraction=depth_fraction,
        needle_value=value,
        response=response.strip(),
        score=score,
    )


def score_at_context_lengths(
    server_url: str,
    target_token_lengths: list[int] = [1000, 8000, 64000, 256000],
    depths: list[float] = [0.1, 0.5, 0.9],
) -> dict[str, Any]:
    """Run NIAH across target lengths × depths grid. Returns summary dict."""
    filler = build_filler()
    rng = random.Random(42)
    runs: list[NiahResult] = []
    for tok in target_token_lengths:
        for depth in depths:
            value = str(rng.randint(1000, 9999))
            project = f"alpha{tok}d{int(depth * 100)}"
            runs.append(
                run_one_probe(
                    server_url=server_url,
                    target_tokens=tok,
                    depth_fraction=depth,
                    project=project,
                    value=value,
                    filler_text=filler,
                )
            )

    by_length: dict[int, list[float]] = {}
    for r in runs:
        by_length.setdefault(r.target_tokens, []).append(r.score)

    return {
        "scores_by_context": {
            tok: sum(scores) / len(scores) for tok, scores in by_length.items()
        },
        "raw_runs": [asdict(r) for r in runs],
    }


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.write_text(json.dumps(results, indent=2))
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_niah.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/niah.py tests/moe/test_niah.py
git commit -m "Add needle-in-a-haystack long-context probe"
```

---

### Task 10: llama-bench wrapper

**Files:**
- Create: `scripts/moe/llama_bench.py`
- Test: `tests/moe/test_llama_bench.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_llama_bench.py`:

```python
"""Tests for llama-bench wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.moe.llama_bench import (
    SpeedSample,
    build_llama_bench_command,
    parse_llama_bench_json,
)


def test_build_command_includes_context_sizes() -> None:
    cmd = build_llama_bench_command(
        gguf_path=Path("/models/m.gguf"),
        ngl=99,
        ctx_sizes=[1024, 8192, 65536],
        gen_tokens=128,
        repetitions=3,
        threads=4,
    )

    assert "llama-bench" in cmd[0] or cmd[0] == "llama-bench"
    assert "-m" in cmd and "/models/m.gguf" in cmd
    assert "-ngl" in cmd and "99" in cmd
    assert "-p" in cmd
    # context sizes go through as -p values
    joined = " ".join(cmd)
    assert "1024" in joined and "8192" in joined and "65536" in joined
    assert "-n" in cmd and "128" in cmd
    assert "-r" in cmd and "3" in cmd
    assert "-o" in cmd and "json" in cmd


def test_parse_llama_bench_json_extracts_pp_and_tg() -> None:
    raw = [
        {"test": "pp1024", "n_prompt": 1024, "avg_ts": 250.5, "stddev_ts": 5.2},
        {"test": "tg128", "n_gen": 128, "avg_ts": 17.3, "stddev_ts": 0.4},
    ]

    samples = parse_llama_bench_json(raw)

    pp = [s for s in samples if s.kind == "pp"]
    tg = [s for s in samples if s.kind == "tg"]
    assert len(pp) == 1 and pp[0].avg_ts == 250.5
    assert len(tg) == 1 and tg[0].avg_ts == 17.3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_llama_bench.py -v
```

Expected: FAIL

- [ ] **Step 3: Write llama_bench.py**

Create `scripts/moe/llama_bench.py`:

```python
"""Wrap llama-bench for speed measurements at multiple context lengths."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SpeedSample:
    kind: str  # "pp" or "tg"
    n_tokens: int
    avg_ts: float
    stddev_ts: float


def build_llama_bench_command(
    *,
    gguf_path: Path,
    ngl: int,
    ctx_sizes: list[int],
    gen_tokens: int,
    repetitions: int,
    threads: int,
) -> list[str]:
    """Compose llama-bench CLI."""
    cmd: list[str] = [
        "llama-bench",
        "-m", str(gguf_path),
        "-ngl", str(ngl),
        "-p", ",".join(str(s) for s in ctx_sizes),
        "-n", str(gen_tokens),
        "-r", str(repetitions),
        "-t", str(threads),
        "-o", "json",
    ]
    return cmd


def parse_llama_bench_json(raw: list[dict[str, Any]]) -> list[SpeedSample]:
    """Convert llama-bench JSON output into SpeedSample list."""
    out: list[SpeedSample] = []
    for entry in raw:
        test = entry.get("test", "")
        kind = "pp" if test.startswith("pp") else "tg" if test.startswith("tg") else "?"
        n_tokens = int(entry.get("n_prompt") or entry.get("n_gen") or 0)
        out.append(SpeedSample(
            kind=kind,
            n_tokens=n_tokens,
            avg_ts=float(entry.get("avg_ts", 0.0)),
            stddev_ts=float(entry.get("stddev_ts", 0.0)),
        ))
    return out


def run_llama_bench(
    *,
    gguf_path: Path,
    ngl: int,
    ctx_sizes: list[int],
    gen_tokens: int = 128,
    repetitions: int = 3,
    threads: int = 4,
    output_json_path: Path | None = None,
) -> list[SpeedSample]:
    """Run llama-bench and return parsed samples. Optionally writes raw JSON."""
    cmd = build_llama_bench_command(
        gguf_path=gguf_path, ngl=ngl, ctx_sizes=ctx_sizes,
        gen_tokens=gen_tokens, repetitions=repetitions, threads=threads,
    )
    print(f"[llama-bench] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(result.stdout)
    if output_json_path:
        output_json_path.write_text(json.dumps(raw, indent=2))
    return parse_llama_bench_json(raw)


def samples_to_summary(samples: list[SpeedSample]) -> dict[str, Any]:
    """Summarize per-context-length tok/s for a results.json."""
    return {
        "samples": [asdict(s) for s in samples],
        "tok_per_sec_by_ctx": {
            f"ctx_{s.n_tokens}_{s.kind}": s.avg_ts for s in samples
        },
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_llama_bench.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/llama_bench.py tests/moe/test_llama_bench.py
git commit -m "Add llama-bench wrapper for multi-context speed sampling"
```

---

### Task 11: Stability runner

**Files:**
- Create: `scripts/moe/stability.py`
- Test: `tests/moe/test_stability.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_stability.py`:

```python
"""Tests for stability runner."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.moe.stability import StabilitySample, summarize_drift


def test_summarize_drift_computes_relative_change() -> None:
    samples = [
        StabilitySample(elapsed_s=0, tok_per_sec=17.0, vram_used_mb=5800),
        StabilitySample(elapsed_s=3600, tok_per_sec=16.9, vram_used_mb=5800),
        StabilitySample(elapsed_s=86400, tok_per_sec=16.7, vram_used_mb=5800),
        StabilitySample(elapsed_s=259200, tok_per_sec=16.5, vram_used_mb=5800),
    ]

    summary = summarize_drift(samples)

    assert summary["t0_tok_per_sec"] == pytest.approx(17.0)
    assert summary["max_relative_drop"] == pytest.approx((17.0 - 16.5) / 17.0, rel=1e-3)
    assert summary["passed_5pct_gate"] is True


def test_summarize_drift_fails_when_drop_exceeds_threshold() -> None:
    samples = [
        StabilitySample(elapsed_s=0, tok_per_sec=17.0, vram_used_mb=5800),
        StabilitySample(elapsed_s=259200, tok_per_sec=10.0, vram_used_mb=5800),
    ]

    summary = summarize_drift(samples)

    assert summary["passed_5pct_gate"] is False
    assert summary["max_relative_drop"] > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_stability.py -v
```

Expected: FAIL

- [ ] **Step 3: Write stability.py**

Create `scripts/moe/stability.py`:

```python
"""Run stability test: warmup → idle interval → re-measure tok/s.

Tests the article's "day 3 slowdown" claim. Pass criterion: <5% relative drop
from t=0 to t=72h with --mlock; expected to fail without it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_INTERVALS_S: list[int] = [0, 3600, 86400, 259200]  # t0, +1h, +24h, +72h


@dataclass
class StabilitySample:
    elapsed_s: int
    tok_per_sec: float
    vram_used_mb: int


def summarize_drift(samples: list[StabilitySample], threshold: float = 0.05) -> dict[str, Any]:
    """Reduce stability samples to pass/fail summary."""
    if not samples:
        return {"samples": 0, "passed_5pct_gate": False}
    t0 = samples[0].tok_per_sec
    drops = [(t0 - s.tok_per_sec) / t0 for s in samples]
    max_drop = max(drops)
    return {
        "samples": [asdict(s) for s in samples],
        "t0_tok_per_sec": t0,
        "max_relative_drop": max_drop,
        "passed_5pct_gate": max_drop < threshold,
    }


def run_stability(
    *,
    measure_fn: Callable[[], tuple[float, int]],
    intervals_s: list[int] = DEFAULT_INTERVALS_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Take samples at each elapsed time. measure_fn returns (tok_per_sec, vram_used_mb)."""
    samples: list[StabilitySample] = []
    elapsed = 0
    for target in intervals_s:
        if target > elapsed:
            sleep_fn(target - elapsed)
            elapsed = target
        tok_s, vram_mb = measure_fn()
        samples.append(StabilitySample(elapsed_s=elapsed, tok_per_sec=tok_s, vram_used_mb=vram_mb))
        print(f"[stability] t={elapsed}s tok/s={tok_s:.2f} vram={vram_mb}MB")

    summary = summarize_drift(samples)
    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_stability.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/stability.py tests/moe/test_stability.py
git commit -m "Add stability runner for 72h drift test"
```

---

### Task 12: Per-flag attribution sweep

**Files:**
- Create: `scripts/moe/flag_sweep.py`
- Test: `tests/moe/test_flag_sweep.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/moe/test_flag_sweep.py`:

```python
"""Tests for the article's incremental flag attribution sweep."""
from __future__ import annotations

import pytest

from scripts.moe.config import Preset
from scripts.moe.flag_sweep import (
    SWEEP_STEPS,
    build_sweep_presets,
)


def test_sweep_has_six_steps_matching_article() -> None:
    assert len(SWEEP_STEPS) == 6
    assert SWEEP_STEPS[0]["name"] == "baseline-ngl-only"
    assert SWEEP_STEPS[-1]["name"] == "all-flags-with-mlock"


def test_build_sweep_presets_each_step_is_valid_preset() -> None:
    presets = build_sweep_presets()

    assert len(presets) == 6
    for p in presets:
        assert isinstance(p, Preset)

    # Step 1: baseline has only -ngl, no n_cpu_moe
    assert "n_cpu_moe" not in presets[0].flags

    # Step 2: adds n_cpu_moe 41
    assert presets[1].flags["n_cpu_moe"] == 41

    # Step 3: adds no_mmap
    assert presets[2].flags["no_mmap"] is True

    # Step 4: rebalances n_cpu_moe to 35
    assert presets[3].flags["n_cpu_moe"] == 35

    # Step 5: adds Turbo Quant + bumps to 36 for 256K context
    assert presets[4].flags["cache_type_k"] == "q4_0"
    assert presets[4].flags["cache_type_v"] == "q3_0"
    assert presets[4].flags["n_cpu_moe"] == 36
    assert presets[4].flags["ctx_size"] == 262144

    # Step 6: adds mlock
    assert presets[5].flags["mlock"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_flag_sweep.py -v
```

Expected: FAIL

- [ ] **Step 3: Write flag_sweep.py**

Create `scripts/moe/flag_sweep.py`:

```python
"""The article's per-flag attribution sweep, run only on the 6GB-1060 / Qwen3-30B-A3B cell.

Six incremental presets, each adding one change. Lets readers see per-flag value.
"""
from __future__ import annotations

from typing import Any

from scripts.moe.config import Preset


# Each step adds incrementally to the previous. Order matches the article.
SWEEP_STEPS: list[dict[str, Any]] = [
    {
        "name": "baseline-ngl-only",
        "description": "Article's 'dumb' baseline: -ngl 20 split, mmap on, no mlock, KV Q8.",
        "flags": {"ngl": 20, "ctx_size": 4096},
    },
    {
        "name": "add-n-cpu-moe-41",
        "description": "Pin all 41 layers' experts to CPU; -ngl 99.",
        "flags": {"ngl": 99, "n_cpu_moe": 41, "ctx_size": 4096},
    },
    {
        "name": "add-no-mmap",
        "description": "Force whole model into RAM up front.",
        "flags": {"ngl": 99, "n_cpu_moe": 41, "no_mmap": True, "ctx_size": 4096},
    },
    {
        "name": "rebalance-to-35",
        "description": "Pull 6 expert layers back to GPU (n_cpu_moe 41→35); context must drop to 64K.",
        "flags": {"ngl": 99, "n_cpu_moe": 35, "no_mmap": True, "ctx_size": 65536},
    },
    {
        "name": "add-turbo-quant-kv",
        "description": "Add Q4/Q3 KV; bump n_cpu_moe to 36 to fit 256K KV cache.",
        "flags": {
            "ngl": 99, "n_cpu_moe": 36, "no_mmap": True,
            "cache_type_k": "q4_0", "cache_type_v": "q3_0",
            "ctx_size": 262144,
        },
    },
    {
        "name": "all-flags-with-mlock",
        "description": "Add --mlock for production stability (no further speed change expected).",
        "flags": {
            "ngl": 99, "n_cpu_moe": 36, "no_mmap": True, "mlock": True,
            "cache_type_k": "q4_0", "cache_type_v": "q3_0",
            "ctx_size": 262144,
        },
    },
]


def build_sweep_presets() -> list[Preset]:
    """Materialize the six sweep steps as Preset objects."""
    return [Preset(name=step["name"], flags=dict(step["flags"])) for step in SWEEP_STEPS]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_flag_sweep.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/flag_sweep.py tests/moe/test_flag_sweep.py
git commit -m "Add per-flag attribution sweep (6 incremental steps)"
```

---

### Task 13: Hardware fingerprint capture

**Files:**
- Create: `scripts/moe/hardware_fingerprint.py`
- Test: `tests/moe/test_hardware_fingerprint.py`

- [ ] **Step 1: Write the failing test**

Create `tests/moe/test_hardware_fingerprint.py`:

```python
"""Tests for hardware fingerprint capture."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.moe.hardware_fingerprint import (
    parse_dmidecode_memory,
    parse_lscpu_output,
)


def test_parse_lscpu_extracts_model_name() -> None:
    raw = """Architecture:                    x86_64
CPU(s):                          8
Model name:                      Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
Thread(s) per core:              2
Core(s) per socket:              4
Socket(s):                       1
"""

    parsed = parse_lscpu_output(raw)

    assert parsed["model"] == "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz"
    assert parsed["cores_total"] == 8
    assert parsed["sockets"] == 1


def test_parse_dmidecode_memory_lists_dimms() -> None:
    raw = """
Memory Device
        Size: 8192 MB
        Type: DDR4
        Speed: 2400 MT/s
        Locator: DIMM_A1

Memory Device
        Size: 8192 MB
        Type: DDR4
        Speed: 2400 MT/s
        Locator: DIMM_A2

Memory Device
        Size: No Module Installed
        Locator: DIMM_B1
"""

    dimms = parse_dmidecode_memory(raw)

    assert len(dimms) == 2
    assert dimms[0]["size_mb"] == 8192
    assert dimms[0]["speed_mts"] == 2400
    assert dimms[0]["locator"] == "DIMM_A1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_hardware_fingerprint.py -v
```

Expected: FAIL

- [ ] **Step 3: Write hardware_fingerprint.py**

Create `scripts/moe/hardware_fingerprint.py`:

```python
"""Capture host hardware fingerprint into a config.yaml host block."""
from __future__ import annotations

import platform
import re
import subprocess
from typing import Any


def parse_lscpu_output(raw: str) -> dict[str, Any]:
    """Extract model, core count, socket count from `lscpu` output."""
    out: dict[str, Any] = {}
    for line in raw.splitlines():
        if line.startswith("Model name:"):
            out["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):"):
            out["cores_total"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Socket(s):"):
            out["sockets"] = int(line.split(":", 1)[1].strip())
    return out


def parse_dmidecode_memory(raw: str) -> list[dict[str, Any]]:
    """Extract installed DIMMs from `dmidecode -t memory` output."""
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    in_device = False
    for line in raw.splitlines():
        s = line.strip()
        if s == "Memory Device":
            if current and current.get("size_mb"):
                devices.append(current)
            current = {}
            in_device = True
            continue
        if not in_device:
            continue
        if s.startswith("Size:"):
            val = s.split(":", 1)[1].strip()
            if "No Module" in val:
                continue
            m = re.match(r"(\d+)\s+(MB|GB)", val)
            if m:
                size = int(m.group(1))
                if m.group(2) == "GB":
                    size *= 1024
                current["size_mb"] = size
        elif s.startswith("Speed:"):
            m = re.search(r"(\d+)\s*MT/s", s)
            if m:
                current["speed_mts"] = int(m.group(1))
        elif s.startswith("Locator:"):
            current["locator"] = s.split(":", 1)[1].strip()
        elif s.startswith("Type:"):
            current["type"] = s.split(":", 1)[1].strip()
    if current and current.get("size_mb"):
        devices.append(current)
    return devices


def fingerprint_host() -> dict[str, Any]:
    """Run host introspection commands and return a host dict."""
    cpu_info: dict[str, Any] = {}
    dimms: list[dict[str, Any]] = []
    try:
        cpu_info = parse_lscpu_output(subprocess.check_output(["lscpu"], text=True))
    except Exception as e:
        cpu_info = {"error": str(e)}
    try:
        dimms = parse_dmidecode_memory(
            subprocess.check_output(["sudo", "-n", "dmidecode", "-t", "memory"], text=True)
        )
    except Exception as e:
        dimms = [{"error": str(e)}]
    total_ram_mb = sum(d.get("size_mb", 0) for d in dimms if "error" not in d)
    speeds = [d.get("speed_mts") for d in dimms if "speed_mts" in d]

    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu": cpu_info,
        "ram_gb": total_ram_mb // 1024,
        "ram_speed_mts": speeds[0] if speeds else None,
        "ram_dimms": dimms,
    }


def fingerprint_gpu() -> dict[str, Any]:
    """Run nvidia-smi --query-gpu and return one GPU's info."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()[0]
        name, mem, drv, pci = (s.strip() for s in raw.split(","))
        return {"model": name, "vram_total": mem, "driver": drv, "pci_bus": pci}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_hardware_fingerprint.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/hardware_fingerprint.py tests/moe/test_hardware_fingerprint.py
git commit -m "Add hardware fingerprint capture (CPU, RAM DIMMs, GPU)"
```

---

## Phase 5: Top-level driver

### Task 14: run_cell.py — orchestrate one cell of the matrix

**Files:**
- Create: `scripts/moe/run_cell.py`
- Test: `tests/moe/test_run_cell.py`

- [ ] **Step 1: Write the failing test**

Create `tests/moe/test_run_cell.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_run_cell.py -v
```

Expected: FAIL

- [ ] **Step 3: Write run_cell.py**

Create `scripts/moe/run_cell.py`:

```python
"""Top-level driver: take one cell config, run all benchmarks, write outputs.

Usage:
    python -m scripts.moe.run_cell \\
        --cell configs/moe-budget/cells/6gb-1060-qwen3-30b-a3b-optimized-moe.yaml \\
        --models-dir ./models \\
        --container ghcr.io/ggml-org/llama.cpp:server-cuda-<digest>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.moe.config import (
    CellConfig,
    ModelEntry,
    Preset,
    load_cell_config,
    load_models,
    load_presets,
    resolve_preset,
)
from scripts.moe.hardware_fingerprint import fingerprint_gpu, fingerprint_host
from scripts.moe.llama_bench import run_llama_bench, samples_to_summary
from scripts.moe.niah import score_at_context_lengths, write_results as write_niah
from scripts.moe.quality import run_quality, parse_lm_eval_results
from scripts.moe.resource_probe import poll_loop, snapshots_to_summary
from scripts.moe.result_storage import (
    CellPaths,
    ensure_cell_dir,
    write_run_config,
)
from scripts.moe.server import LlamaCppServer, ServerConfig


@dataclass
class RunInputs:
    cell: CellConfig
    model_entry: ModelEntry
    effective_preset: Preset


def resolve_run_inputs(
    *, cell_path: Path, presets_path: Path, models_path: Path
) -> RunInputs:
    cell = load_cell_config(cell_path)
    presets = load_presets(presets_path)
    models = load_models(models_path)
    base = presets[cell.preset]
    effective = resolve_preset(base, cell.preset_overrides)
    return RunInputs(
        cell=cell,
        model_entry=models[cell.model],
        effective_preset=effective,
    )


def run_cell(
    *,
    cell_path: Path,
    presets_path: Path,
    models_path: Path,
    models_dir: Path,
    container_image: str,
    results_root: Path,
    skip_stability: bool = True,
    quality_tasks: str = "mmlu,gsm8k,hellaswag,truthfulqa_mc2,arc_challenge",
    niah_lengths: list[int] | None = None,
) -> None:
    """Execute one cell of the matrix end-to-end."""
    inputs = resolve_run_inputs(
        cell_path=cell_path, presets_path=presets_path, models_path=models_path
    )

    paths = ensure_cell_dir(
        results_root=results_root,
        tier=inputs.cell.tier,
        model=inputs.cell.model,
        preset=inputs.cell.preset,
    )

    gguf_path = models_dir / inputs.model_entry.filename
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF missing: {gguf_path}")

    host = fingerprint_host()
    gpu = fingerprint_gpu()

    write_run_config(
        paths=paths,
        run_id=inputs.cell.run_id,
        host=host,
        gpu=gpu,
        container={"image": container_image, "ipc_lock": True},
        model={
            "name": inputs.model_entry.name,
            "hf_repo": inputs.model_entry.hf_repo,
            "filename": inputs.model_entry.filename,
            "sha256": inputs.model_entry.sha256,
            "tokenizer": inputs.model_entry.tokenizer,
            "params_total_b": inputs.model_entry.params_total_b,
            "params_active_b": inputs.model_entry.params_active_b,
        },
        preset_name=inputs.cell.preset,
        preset_flags=inputs.effective_preset.flags,
    )

    server_cfg = ServerConfig(
        image=container_image,
        gguf_path=gguf_path,
        host_port=8080,
        ipc_lock=True,
    )

    with LlamaCppServer(server_cfg, inputs.effective_preset) as server:
        # Resource baseline
        if server.pid() is not None:
            snaps = poll_loop(server.pid(), duration_s=10, interval_s=1.0)
            paths.resources_json.write_text(json.dumps(snapshots_to_summary(snaps), indent=2))

        # Quality eval
        run_quality(
            server_url=server.url,
            tokenizer=inputs.model_entry.tokenizer,
            tasks=quality_tasks,
            output_dir=paths.dir / "lm_eval_raw",
        )
        results_files = list((paths.dir / "lm_eval_raw").glob("**/results*.json"))
        if results_files:
            parsed = parse_lm_eval_results(sorted(results_files, key=lambda p: p.stat().st_mtime)[-1])
            paths.lm_eval_json.write_text(json.dumps(parsed, indent=2))

        # NIAH
        lengths = niah_lengths or [1000, 8000, 64000, 256000]
        # Cap NIAH lengths at preset's ctx_size
        ctx = int(inputs.effective_preset.flags.get("ctx_size", 4096))
        lengths = [L for L in lengths if L <= ctx]
        if lengths:
            niah_summary = score_at_context_lengths(
                server_url=server.url, target_token_lengths=lengths
            )
            write_niah(paths.niah_json, niah_summary)

    # Speed via llama-bench (separate process, no server)
    speed_samples = run_llama_bench(
        gguf_path=gguf_path,
        ngl=int(inputs.effective_preset.flags.get("ngl", 99)),
        ctx_sizes=[1024, 8192, min(65536, ctx)],
        output_json_path=paths.dir / "llama_bench_raw.json",
    )
    paths.llama_bench_json.write_text(json.dumps(samples_to_summary(speed_samples), indent=2))

    print(f"[run_cell] cell complete: {inputs.cell.run_id}")
    print(f"[run_cell] outputs: {paths.dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one cell of the MoE-budget matrix.")
    parser.add_argument("--cell", required=True, type=Path)
    parser.add_argument("--presets", default=Path("configs/moe-budget/presets.yaml"), type=Path)
    parser.add_argument("--models-yaml", default=Path("configs/moe-budget/models.yaml"), type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--container", required=True, help="llama.cpp Docker image (digest-pinned)")
    parser.add_argument("--results-root", default=Path("results/moe-budget"), type=Path)
    args = parser.parse_args()

    run_cell(
        cell_path=args.cell,
        presets_path=args.presets,
        models_path=args.models_yaml,
        models_dir=args.models_dir,
        container_image=args.container,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/moe/test_run_cell.py -v
```

Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/run_cell.py tests/moe/test_run_cell.py
git commit -m "Add top-level cell driver wiring all benchmarks together"
```

---

## Phase 6: Calibration

### Task 15: Pin llama.cpp Docker image digest and verify

**Files:**
- Create: `configs/moe-budget/container.yaml`

- [ ] **Step 1: Pull a current llama.cpp server-cuda image and capture its digest**

```bash
# Try the current canonical registry first; fall back to legacy if it fails
REGISTRY="ghcr.io/ggml-org/llama.cpp"
if ! docker pull "${REGISTRY}:server-cuda" 2>&1 | tee /tmp/pull.log; then
  echo "Primary registry failed, trying legacy..."
  REGISTRY="ghcr.io/ggerganov/llama.cpp"
  docker pull "${REGISTRY}:server-cuda"
fi
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "${REGISTRY}:server-cuda")
echo "Captured: ${DIGEST}"
echo "REGISTRY=${REGISTRY}" > /tmp/llamacpp_image.env
echo "DIGEST=${DIGEST}" >> /tmp/llamacpp_image.env
```

Expected: `/tmp/llamacpp_image.env` exists, `DIGEST` looks like `ghcr.io/<org>/llama.cpp@sha256:<64hex>`.

- [ ] **Step 2: Smoke test the image with `--help`**

```bash
source /tmp/llamacpp_image.env
docker run --rm "${DIGEST}" --server --help 2>&1 | grep -E '(--n-cpu-moe|--mlock|--no-mmap|--cache-type-k|--cache-type-v)'
```

Expected: at least 5 matching lines (one per required flag). If any are missing, the image is too old — pull a newer tag (e.g., a recent dated tag like `:server-cuda-b<NNNN>`) and re-run step 1.

- [ ] **Step 3: Write the pinned digest to container.yaml using the captured value**

```bash
source /tmp/llamacpp_image.env
mkdir -p configs/moe-budget
TODAY=$(date +%Y-%m-%d)
SHA_ONLY="${DIGEST##*@}"

cat > configs/moe-budget/container.yaml <<EOF
# Pinned llama.cpp Docker image digest. ALL runs in this study use this exact image.
# To update: pull new image, capture digest with \`docker inspect --format='{{index .RepoDigests 0}}'\`,
# update this file, commit. Reason for pinning: any change in expert-routing kernels,
# CUDA kernels, or KV-cache implementation could shift results in non-obvious ways.

container:
  registry: ${REGISTRY}
  tag_at_pin: server-cuda
  digest: ${SHA_ONLY}
  full_reference: ${DIGEST}
  pinned_at: ${TODAY}
  pinned_reason: "Initial study baseline. See spec §5."
EOF

cat configs/moe-budget/container.yaml
```

Expected: the file is written with concrete digest, no `<FILL-IN>` text. Verify with:

```bash
grep -c '<FILL-IN' configs/moe-budget/container.yaml || echo "OK: no placeholders"
```

- [ ] **Step 4: Commit**

```bash
git add configs/moe-budget/container.yaml
git commit -m "Pin llama.cpp Docker image digest for MoE-budget runs"
```

---

### Task 16: Resolve article model identity (Qwen3-30B-A3B vs Qwen3-Next-30B-A3B)

This is the spec's Open Question #1. Resolve before downloading the wrong GGUF.

**Files:**
- Modify: `configs/moe-budget/models.yaml` (only if identity changes)
- Create: `docs/superpowers/notes/2026-05-09-article-model-identity.md`

- [ ] **Step 1: Inspect article evidence**

The article transcript states "256 specialists, 8 wake up per token" — this matches **Qwen3-30B-A3B** (`128 experts × 2 layers = 256` per the model card; top-8 routing). It also claims "30 of 40 layers are SSM," which would point to a Qwen3-Next variant. These are inconsistent.

- [ ] **Step 2: Check model cards on HuggingFace for both candidates**

```bash
# Inspect Qwen3-30B-A3B
curl -s https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B | python -m json.tool | head -80

# Inspect Qwen3-Next-80B-A3B (closest Next variant)
curl -s https://huggingface.co/api/models/Qwen/Qwen3-Next-80B-A3B-Instruct | python -m json.tool | head -80
```

Look for: `num_experts`, `num_experts_per_tok`, `num_hidden_layers`, mention of `state_space` or `mamba`.

Expected outcome: Qwen3-30B-A3B has 128 experts × top-8, all attention layers (no SSM). Qwen3-Next-80B-A3B has hybrid SSM-attention.

- [ ] **Step 3: Decide which model the article ran**

Decision rule: model size (~20GB after Q4 → 18GB on 24GB-RAM rig) is the dominant evidence. Qwen3-30B-A3B Q4 is ~18GB. Qwen3-Next-80B-A3B Q4 is ~45GB and would NOT fit 24GB RAM at all. **Therefore the article ran Qwen3-30B-A3B**, and the SSM claim in the transcript is an auto-caption error.

- [ ] **Step 4: Document the decision**

Create `docs/superpowers/notes/2026-05-09-article-model-identity.md`:

```markdown
# Article Model Identity Resolution

## Resolution

The video creator ran **Qwen3-30B-A3B**, not a Qwen3-Next variant.

## Evidence

1. **Model size constraint:** the article's rig has 24GB RAM. Qwen3-30B-A3B at Q4_K_M is ~18GB
   (fits with breathing room). Qwen3-Next-80B-A3B at Q4_K_M is ~45GB (would not fit at all).
2. **Expert count match:** the transcript's "256 specialists, 8 wake up per token" matches
   Qwen3-30B-A3B's expert configuration (128 experts × 2 sub-layers, top-8 routing in some
   model-card phrasings, or 128 experts and top-8 directly).
3. **SSM claim refuted:** the transcript's "30 of 40 layers are SSM" is inconsistent with
   Qwen3-30B-A3B (no SSM) and with the 256-expert count of vanilla Qwen3 MoE. The most
   parsimonious explanation is an auto-caption error.

## Implication for the spec

- `models.yaml` already pins `qwen3-30b-a3b` as the article-reproduction target — correct.
- Speculative-decoding negative-result documentation should be framed as "MoE memory-thrash"
  rather than "SSM serialization." Update `docs/moe-on-a-budget.md` writeup accordingly.
- The Qwen3-Next-80B-A3B unlock test on 8GB+64GB is a SEPARATE finding, not the article repro.

## Open follow-up (does not block the spec)

If the article author can be reached, confirm directly. Until then this resolution stands.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/
git commit -m "Resolve article model identity: Qwen3-30B-A3B (not Qwen3-Next)"
```

---

### Task 17: Download GGUFs and populate SHA-256 hashes

**Files:**
- Modify: `configs/moe-budget/models.yaml` (replace TBD-FILL-IN-TASK-17 sentinels with real hashes)
- Create: `scripts/moe/download_models.py`

- [ ] **Step 1: Write a download helper that verifies and records hashes**

Create `scripts/moe/download_models.py`:

```python
"""Download GGUFs from HuggingFace and update models.yaml with their SHA-256 hashes.

Reads configs/moe-budget/models.yaml, downloads each model into the local --models-dir,
computes SHA-256, and writes the hash back to the YAML.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from scripts.moe.config import load_models


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def update_yaml_hashes(yaml_path: Path, name_to_hash: dict[str, str]) -> None:
    raw = yaml.safe_load(yaml_path.read_text())
    for name, body in raw["models"].items():
        if name in name_to_hash:
            body["sha256"] = name_to_hash[name]
    yaml_path.write_text(yaml.safe_dump(raw, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models and record hashes.")
    parser.add_argument("--models-yaml", type=Path, default=Path("configs/moe-budget/models.yaml"))
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated names to download")
    args = parser.parse_args()

    models = load_models(args.models_yaml)
    target_names = set(args.only.split(",")) if args.only else set(models.keys())

    args.models_dir.mkdir(parents=True, exist_ok=True)
    name_to_hash: dict[str, str] = {}
    for name, entry in models.items():
        if name not in target_names:
            continue
        local = args.models_dir / entry.filename
        if not local.exists():
            print(f"[download] {name}: pulling from {entry.hf_repo}/{entry.filename}")
            downloaded = hf_hub_download(
                repo_id=entry.hf_repo,
                filename=entry.filename,
                local_dir=str(args.models_dir),
                local_dir_use_symlinks=False,
            )
            local = Path(downloaded)
        else:
            print(f"[download] {name}: already present at {local}")
        digest = sha256_file(local)
        print(f"[download] {name}: sha256={digest}")
        name_to_hash[name] = digest

    update_yaml_hashes(args.models_yaml, name_to_hash)
    print(f"Updated {args.models_yaml} with {len(name_to_hash)} hashes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify HF repo paths exist before downloading anything**

```bash
for repo in \
  "Qwen/Qwen3-30B-A3B-GGUF" \
  "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF" \
  "allenai/OLMoE-1B-7B-0924-Instruct-GGUF" \
  "Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF" \
  "bartowski/DeepSeek-V2-Lite-Chat-GGUF" \
  "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  "Qwen/Qwen3-4B-Instruct-GGUF" \
  "bartowski/Phi-4-mini-instruct-GGUF" \
; do
  echo "Checking $repo"
  curl -sf "https://huggingface.co/api/models/$repo" > /dev/null && echo "  OK" || echo "  MISSING — investigate"
done
```

Expected: every repo resolves. Any MISSING means update `models.yaml` with the correct repo path before continuing — Bartowski/TheBloke/Unsloth often re-pack the same model under different repos.

- [ ] **Step 3: Download Q4_K_M variants of all 8 models**

```bash
mkdir -p models
python -m scripts.moe.download_models --models-dir ./models
```

Expected: `~80-100 GB` total download. Each model logs a SHA-256 hash.

- [ ] **Step 4: Verify no TBD sentinels remain**

```bash
grep -c "TBD-FILL-IN-TASK-17" configs/moe-budget/models.yaml
```

Expected: `0` (zero remaining sentinels). If non-zero, re-run download for the missing models.

- [ ] **Step 5: Commit (do NOT commit the GGUFs themselves — just hashes)**

```bash
echo "models/" >> .gitignore
git add scripts/moe/download_models.py configs/moe-budget/models.yaml .gitignore
git commit -m "Download GGUFs and record SHA-256 hashes in models.yaml"
```

---

### Task 18: Smoke test the end-to-end harness on one cell

This is a sanity check before doing real benchmark runs. Run the smallest cell to verify all components wire together.

**Files:** none modified — this is a runtime validation.

- [ ] **Step 1: Pick the smoke-test cell**

The smallest cell is **2GB / TinyLlama-1.1B / default** — fast to download, fast to load, fast to evaluate. We are NOT yet on the GTX 950 rig — run on whatever GPU is currently in the X99M-A box for this smoke test only.

- [ ] **Step 2: Run it**

```bash
python -m scripts.moe.run_cell \
  --cell configs/moe-budget/cells/2gb-950-tinyllama-default.yaml \
  --models-dir ./models \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
```

Expected duration: ~10-30 minutes (mostly lm-eval running MMLU). Outputs land in `results/moe-budget/2gb/tinyllama-1.1b/default/`.

- [ ] **Step 3: Verify all expected output files exist**

```bash
ls results/moe-budget/2gb/tinyllama-1.1b/default/
```

Expected files:
- `config.yaml` — non-empty, contains real GPU/host fingerprint, no `<placeholder>` text
- `lm_eval.json` — non-empty, contains scores for at least mmlu/gsm8k/hellaswag
- `niah.json` — non-empty (lengths capped at 4096)
- `llama_bench.json` — non-empty
- `resources.json` — non-empty
- `lm_eval_raw/` — directory of raw lm_eval outputs

If any file is missing or empty → debug the corresponding component before proceeding.

- [ ] **Step 4: Sanity-check tok/s and quality numbers**

Open `llama_bench.json`. TinyLlama-1.1B Q4 should generate at hundreds of tok/s on any modern GPU. If it's <50, GPU offload isn't working — check `nvidia-smi` during the run.

Open `lm_eval.json`. TinyLlama MMLU should be ~25-30%, GSM8K ~2-5%. If wildly off, the lm-eval connection or chat template is wrong.

- [ ] **Step 5: Commit the smoke-test results**

```bash
git add results/moe-budget/2gb/tinyllama-1.1b/default/
git commit -m "Smoke-test cell: 2GB / TinyLlama-1.1B / default"
```

---

## Phase 7: Validation gate — 6GB tier (article reproduction)

This is the methodology gate. If we cannot reproduce 17 tok/s on a GTX 1060 with the optimized-moe preset, the methodology has a bug — fix it before continuing.

### Task 19: Install the GTX 1060 6GB into the rig

**Files:** none — physical hardware swap.

- [ ] **Step 1: Power down the X99M-A**
- [ ] **Step 2: Remove current GPU; install GTX 1060 6GB**
- [ ] **Step 3: Boot, verify with `nvidia-smi`**

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Expected: `GeForce GTX 1060 6GB, 6144 MiB, <driver>`. If driver version doesn't support llama.cpp's CUDA build, update.

---

### Task 20: Run default-preset cell on 1060 6GB

**Files:** writes to `results/moe-budget/6gb/qwen3-30b-a3b/default/`

- [ ] **Step 1: Run the default cell**

```bash
python -m scripts.moe.run_cell \
  --cell configs/moe-budget/cells/6gb-1060-qwen3-30b-a3b-default.yaml \
  --models-dir ./models \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
```

Expected: SLOW. Article reports ~3 tok/s for this configuration. Cell completes in hours (lm-eval over MMLU at 3 tok/s is painful). Acceptable — this baseline is the "satellite phone" floor.

- [ ] **Step 2: Verify outputs and tok/s**

Open `results/moe-budget/6gb/qwen3-30b-a3b/default/llama_bench.json`. Generation tok/s at small context should be in the 2-5 range.

- [ ] **Step 3: Commit**

```bash
git add results/moe-budget/6gb/qwen3-30b-a3b/default/
git commit -m "6GB / Qwen3-30B-A3B / default — baseline (~3 tok/s expected)"
```

---

### Task 21: Run optimized-moe cell on 1060 6GB

**Files:** writes to `results/moe-budget/6gb/qwen3-30b-a3b/optimized-moe/`

- [ ] **Step 1: Run the optimized cell**

```bash
python -m scripts.moe.run_cell \
  --cell configs/moe-budget/cells/6gb-1060-qwen3-30b-a3b-optimized-moe.yaml \
  --models-dir ./models \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
```

Expected: cell completes in 1-3 hours (much faster than baseline due to ~17 tok/s).

- [ ] **Step 2: Validation gate — does it reproduce 17 tok/s?**

Open `results/moe-budget/6gb/qwen3-30b-a3b/optimized-moe/llama_bench.json`. Look at `tok_per_sec_by_ctx`. The `tg128` (text generation, 128 tokens) entry should be close to **17 tok/s**.

| Observed tok/s | Decision |
|---|---|
| **15-19 tok/s** | PASS. Methodology validated. Proceed. |
| 10-15 tok/s | INVESTIGATE. Possibly RAM speed difference, n_cpu_moe tuning, or driver. Do not proceed until resolved. |
| <10 tok/s | FAIL. Stop. Diff your config.yaml against the article's setup. Common culprits: (a) wrong llama.cpp version, (b) `--no-mmap` not actually applied (check `run.log` for the docker command), (c) RAM not at full bandwidth (verify with `dmidecode`), (d) GPU not actually connected x16 Gen3 (`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv`). |
| >19 tok/s | INVESTIGATE — possibly faster RAM than the article. Document. |

- [ ] **Step 3: Verify VRAM utilization**

Open `resources.json`. Peak VRAM should be ~5.9/6.0 GB (per article). If <5GB you have headroom and could run a different `n_cpu_moe`; if 6.0 GB and unstable, lower `n_cpu_moe` value (move more experts back to CPU).

- [ ] **Step 4: Commit**

```bash
git add results/moe-budget/6gb/qwen3-30b-a3b/optimized-moe/
git commit -m "6GB / Qwen3-30B-A3B / optimized-moe — article reproduction"
```

---

### Task 22: Run per-flag attribution sweep on 1060 6GB

**Files:**
- Create: `scripts/moe/run_flag_sweep.py`
- Output: `results/moe-budget/6gb/qwen3-30b-a3b/flag-sweep/`

- [ ] **Step 1: Write run_flag_sweep.py driver**

Create `scripts/moe/run_flag_sweep.py`:

```python
"""Run the article's incremental flag attribution sweep.

For each step in SWEEP_STEPS, boot a fresh server with the step's preset, run
llama-bench at small context (tok/s only — no quality eval needed), capture
peak VRAM, write per-step JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.moe.flag_sweep import build_sweep_presets
from scripts.moe.llama_bench import run_llama_bench, samples_to_summary
from scripts.moe.resource_probe import poll_loop, snapshots_to_summary
from scripts.moe.server import LlamaCppServer, ServerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    presets = build_sweep_presets()
    summary = []

    for preset in presets:
        step_dir = args.out_dir / preset.name
        step_dir.mkdir(exist_ok=True)
        cfg = ServerConfig(image=args.container, gguf_path=args.gguf, host_port=8080, ipc_lock=True)

        with LlamaCppServer(cfg, preset) as server:
            # 30-second resource baseline at idle
            snaps = poll_loop(server.pid(), duration_s=30, interval_s=2.0)
            (step_dir / "resources.json").write_text(json.dumps(snapshots_to_summary(snaps), indent=2))

        # Speed: separate llama-bench run (server stopped between)
        speed = run_llama_bench(
            gguf_path=args.gguf,
            ngl=int(preset.flags.get("ngl", 99)),
            ctx_sizes=[1024, min(8192, int(preset.flags.get("ctx_size", 4096)))],
            output_json_path=step_dir / "llama_bench_raw.json",
        )
        speed_summary = samples_to_summary(speed)
        (step_dir / "llama_bench.json").write_text(json.dumps(speed_summary, indent=2))

        tg_entry = next((s for s in speed if s.kind == "tg"), None)
        summary.append({
            "step": preset.name,
            "tg_tok_per_sec": tg_entry.avg_ts if tg_entry else None,
            "flags": preset.flags,
        })

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[flag-sweep] complete. {len(presets)} steps. Summary at {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sweep**

```bash
python -m scripts.moe.run_flag_sweep \
  --gguf models/qwen3-30b-a3b-q4_k_m.gguf \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')" \
  --out-dir results/moe-budget/6gb/qwen3-30b-a3b/flag-sweep
```

Expected duration: ~30-90 minutes (6 boots, each with ~5 min benchmark).

- [ ] **Step 3: Verify the article's curve**

Open `results/moe-budget/6gb/qwen3-30b-a3b/flag-sweep/summary.json`. Expected per-step tok/s pattern (approximately):

| Step | Expected tok/s |
|---|---|
| baseline-ngl-only | 3 |
| add-n-cpu-moe-41 | 10 |
| add-no-mmap | 13.5 |
| rebalance-to-35 | 17 |
| add-turbo-quant-kv | 17 |
| all-flags-with-mlock | 17 |

Document any deviation in `docs/moe-on-a-budget.md` later.

- [ ] **Step 4: Commit**

```bash
git add scripts/moe/run_flag_sweep.py results/moe-budget/6gb/qwen3-30b-a3b/flag-sweep/
git commit -m "6GB flag-attribution sweep results (article reproduction)"
```

---

### Task 23: Run 72h stability test on 1060 6GB / optimized-moe

**Files:** writes to `results/moe-budget/6gb/qwen3-30b-a3b/optimized-moe/stability.json`

- [ ] **Step 1: Write a stability driver**

Create `scripts/moe/run_stability.py`:

```python
"""72h stability test: boot server with optimized-moe preset, measure tok/s at t=0,
+1h, +24h, +72h. Pass criterion: <5% relative drop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.moe.config import load_cell_config, load_models, load_presets, resolve_preset
from scripts.moe.llama_bench import run_llama_bench
from scripts.moe.resource_probe import snapshot
from scripts.moe.server import LlamaCppServer, ServerConfig
from scripts.moe.stability import DEFAULT_INTERVALS_S, StabilitySample, summarize_drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervals", type=str, default=",".join(str(s) for s in DEFAULT_INTERVALS_S))
    args = parser.parse_args()

    intervals = [int(s) for s in args.intervals.split(",")]
    cell = load_cell_config(args.cell)
    presets = load_presets(Path("configs/moe-budget/presets.yaml"))
    base = presets[cell.preset]
    effective = resolve_preset(base, cell.preset_overrides)

    cfg = ServerConfig(image=args.container, gguf_path=args.gguf, host_port=8080, ipc_lock=True)
    server = LlamaCppServer(cfg, effective, health_timeout_s=900)
    server.start()

    samples: list[StabilitySample] = []
    elapsed = 0
    try:
        for target in intervals:
            if target > elapsed:
                time.sleep(target - elapsed)
                elapsed = target
            speed = run_llama_bench(
                gguf_path=args.gguf,
                ngl=int(effective.flags.get("ngl", 99)),
                ctx_sizes=[1024],
                gen_tokens=128,
                repetitions=2,
            )
            tg = next((s for s in speed if s.kind == "tg"), None)
            tok_s = tg.avg_ts if tg else 0.0
            snap = snapshot(server.pid())
            samples.append(StabilitySample(elapsed_s=elapsed, tok_per_sec=tok_s, vram_used_mb=snap.vram_used_mb))
            print(f"[stability] t={elapsed}s tok/s={tok_s:.2f} vram={snap.vram_used_mb}MB")
    finally:
        server.stop()

    summary = summarize_drift(samples)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[stability] passed_5pct_gate={summary['passed_5pct_gate']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Schedule the 72h run**

This blocks the rig for 72 hours. User has confirmed this is acceptable.

```bash
python -m scripts.moe.run_stability \
  --cell configs/moe-budget/cells/6gb-1060-qwen3-30b-a3b-optimized-moe.yaml \
  --gguf models/qwen3-30b-a3b-q4_k_m.gguf \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')" \
  --output results/moe-budget/6gb/qwen3-30b-a3b/optimized-moe/stability.json
```

For initial dev/iteration, override intervals: `--intervals 0,60,120` (2-minute test) to validate the script. Re-run with the real intervals once the script works.

- [ ] **Step 3: Verify pass/fail**

Open `stability.json`. `passed_5pct_gate` should be `true`. If `false`, document in the writeup as the expected mlock-not-working case worth investigating; do NOT block subsequent tiers on stability.

- [ ] **Step 4: Commit**

```bash
git add scripts/moe/run_stability.py results/moe-budget/6gb/qwen3-30b-a3b/optimized-moe/stability.json
git commit -m "6GB / Qwen3-30B-A3B / optimized-moe — 72h stability test"
```

---

## Phase 8: Other tier runs

These run after the validation gate passes. Each tier is a sequence of GPU swap → run cells → commit results. The harness is unchanged from Phase 7.

### Task 24: 4GB tier — install GTX 1050 Ti, run all 4GB cells

**Files:** writes to `results/moe-budget/4gb/...`

- [ ] **Step 1: Swap GPU to GTX 1050 Ti 4GB; verify with `nvidia-smi`**

- [ ] **Step 2: Run each 4GB cell**

```bash
for cell in configs/moe-budget/cells/4gb-*.yaml; do
  python -m scripts.moe.run_cell \
    --cell "$cell" \
    --models-dir ./models \
    --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
done
```

Expected: 6 cells × ~1-3h each = ~10-20h total. Run overnight.

- [ ] **Step 3: Commit results**

```bash
git add results/moe-budget/4gb/
git commit -m "4GB tier results — Qwen3-30B-A3B floor + Qwen1.5-MoE + DeepSeek-V2-Lite + Qwen3-4B baseline"
```

---

### Task 25: 2GB tier — install GTX 950, run all 2GB cells

**Files:** writes to `results/moe-budget/2gb/...`

- [ ] **Step 1: Swap GPU to GTX 950 2GB; verify with `nvidia-smi`**

- [ ] **Step 2: Smoke-test that Q4_K_M MoE inference works on Maxwell**

This addresses spec Open Question #4. Run a quick sanity check before full sweep:

```bash
python -m scripts.moe.run_cell \
  --cell configs/moe-budget/cells/2gb-950-tinyllama-default.yaml \
  --models-dir ./models \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
```

If this fails with a CUDA kernel error, the GTX 950 cannot run the current llama.cpp build. **Fall-back:** drop the 2GB tier from primary scope, document in writeup, proceed with 4GB+ tiers.

- [ ] **Step 3: Run remaining 2GB cells**

```bash
for cell in configs/moe-budget/cells/2gb-*.yaml; do
  python -m scripts.moe.run_cell \
    --cell "$cell" --models-dir ./models \
    --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
done
```

- [ ] **Step 4: Commit**

```bash
git add results/moe-budget/2gb/
git commit -m "2GB tier results — OLMoE-1B-7B vs TinyLlama-1.1B"
```

---

### Task 26: 8GB / 32GB-RAM tier — install RTX 2060 Super, run cells at current 32GB RAM

**Files:** writes to `results/moe-budget/8gb-32gbram/...`

- [ ] **Step 1: Swap GPU to RTX 2060 Super 8GB; verify with `nvidia-smi`**

- [ ] **Step 2: Confirm RAM is still 32GB (4×8GB)**

```bash
sudo dmidecode -t memory | grep -E "Size:|Speed:" | head -16
```

- [ ] **Step 3: Run all 8gb-32gbram cells**

```bash
for cell in configs/moe-budget/cells/8gb-32gbram-*.yaml; do
  python -m scripts.moe.run_cell \
    --cell "$cell" --models-dir ./models \
    --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
done
```

- [ ] **Step 4: Commit**

```bash
git add results/moe-budget/8gb-32gbram/
git commit -m "8GB/32GB-RAM tier — Qwen3-30B-A3B headroom + Phi-4-Mini baseline"
```

---

### Task 27: Source 4×16GB DDR4, swap to 64GB

This is a hardware procurement step, not a code step.

**Files:** none — physical RAM swap.

- [ ] **Step 1: Source 4×16GB DDR4 ECC RDIMM**

Used DDR4-2400 ECC RDIMM 16GB sticks: ~$25-40 each on eBay or used-server-parts vendors. Buy a matched-speed kit if possible.

- [ ] **Step 2: Power down, install 4×16GB, remove existing 8GB sticks**

- [ ] **Step 3: Boot, verify `dmidecode` shows 4×16GB at expected speed**

```bash
sudo dmidecode -t memory | grep -E "Size:|Speed:" | head -16
```

Expected: four `Size: 16384 MB` and four `Speed: 2400 MT/s` (or similar).

```bash
free -g
```

Expected: `total: 62-64 GB`.

---

### Task 28: 8GB / 64GB-RAM tier — Qwen3-Next-80B-A3B unlock test

**Files:** writes to `results/moe-budget/8gb-64gbram/...`

- [ ] **Step 1: Verify Qwen3-Next-80B-A3B GGUF exists locally**

```bash
ls -lh models/qwen3-next-80b-a3b-instruct-q4_k_m.gguf
```

Expected: file size ~40-50 GB. If missing, re-run `download_models.py --only qwen3-next-80b-a3b`.

- [ ] **Step 2: Run the unlock cell**

```bash
python -m scripts.moe.run_cell \
  --cell configs/moe-budget/cells/8gb-64gbram-2060super-qwen3-next-80b-a3b-optimized-moe.yaml \
  --models-dir ./models \
  --container "$(grep full_reference configs/moe-budget/container.yaml | cut -d: -f2- | tr -d ' ')"
```

Expected: load time ~3-5 minutes (47GB read into RAM via `--no-mmap`). Generation tok/s in the 5-15 range — slower than 30B-A3B due to larger total model and proportionally more bus traffic per token.

If load fails with OOM despite 64GB:
- Check `ctx_size` — KV cache for 80B-class model at 256K is large; spec preset uses `ctx_size: 65536` for this cell already
- Check `vm.swappiness` and disable swap to confirm not silently paging
- Check `dmesg | tail` for OOM kills

- [ ] **Step 3: Verify the unlock**

If tok/s ≥ 5 and quality outputs sane: **the RAM-as-second-axis story works**. Document it.

If tok/s < 5 but model loads: marginal win, document with measured numbers.

If model fails to load at all: this cell is `OOM` — record the failure, document the threshold.

- [ ] **Step 4: Commit**

```bash
git add results/moe-budget/8gb-64gbram/
git commit -m "8GB/64GB-RAM unlock — Qwen3-Next-80B-A3B (or OOM record)"
```

---

### Task 29: Decision check on 128GB stretch

**Files:** none — decision point.

- [ ] **Step 1: Read 80B-A3B unlock results**

If 80B-A3B at 64GB RAM gave tok/s ≥ 8 and quality holds up, **proceed to 128GB stretch**. The next cohort of MoEs to test:
- Phi-3.5-MoE-instruct (60B/6.6B active) — 6.6B active fits 8GB? Tight; verify
- GRIN-MoE (60B/6.6B active) — same constraint
- Larger Qwen MoEs if released

If 80B-A3B at 64GB RAM was < 8 tok/s or unstable: **stop here**. The 128GB upgrade does not buy enough for the cost. Document the decision and skip Tasks 30-31.

- [ ] **Step 2: If proceeding — source 4×32GB DDR4 ECC RDIMM (~$240) and swap**

Same hardware swap process as Task 27. Verify with `free -g` showing ~128GB total.

- [ ] **Step 3: Add 128GB cells to the matrix**

Edit `scripts/moe/generate_cells.py` `MATRIX` list to add e.g.:

```python
{
    "run_id": "8gb-128gbram-2060super-phi-3.5-moe-optimized-moe",
    "tier": "8gb-128gbram", "model": "phi-3.5-moe", "preset": "optimized-moe",
    "preset_overrides": {"n_cpu_moe": 30, "ctx_size": 32768},
},
```

(Plus a `models.yaml` entry for `phi-3.5-moe` and a download.)

Re-run `generate_cells.py`, run the new cells, commit.

This is a stretch task — keep it tightly scoped.

---

## Phase 9: Analysis and docs

### Task 30: Aggregate results across all cells

**Files:**
- Create: `scripts/moe/aggregate.py`
- Test: `tests/moe/test_aggregate.py`
- Output: `results/moe-budget/summary.json`, `results/moe-budget/summary.csv`

- [ ] **Step 1: Write the failing test**

Create `tests/moe/test_aggregate.py`:

```python
"""Tests for results aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.moe.aggregate import aggregate_results


def test_aggregate_walks_results_tree(tmp_path: Path) -> None:
    cell = tmp_path / "6gb" / "qwen3-30b-a3b" / "optimized-moe"
    cell.mkdir(parents=True)
    (cell / "config.yaml").write_text("run_id: test\n")
    (cell / "lm_eval.json").write_text(json.dumps({"mmlu": {"acc": 0.55}}))
    (cell / "llama_bench.json").write_text(json.dumps({
        "tok_per_sec_by_ctx": {"ctx_1024_tg": 17.2}
    }))

    rows = aggregate_results(tmp_path)

    assert len(rows) == 1
    assert rows[0]["tier"] == "6gb"
    assert rows[0]["model"] == "qwen3-30b-a3b"
    assert rows[0]["preset"] == "optimized-moe"
    assert rows[0]["mmlu_acc"] == pytest.approx(0.55)
    assert rows[0]["tg_1024"] == pytest.approx(17.2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/moe/test_aggregate.py -v
```

Expected: FAIL

- [ ] **Step 3: Write aggregate.py**

Create `scripts/moe/aggregate.py`:

```python
"""Walk results/moe-budget/ and produce a flat summary table (JSON + CSV)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def aggregate_results(results_root: Path) -> list[dict[str, Any]]:
    """Walk <root>/<tier>/<model>/<preset>/ and produce one row per cell."""
    rows: list[dict[str, Any]] = []
    for tier_dir in sorted(p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for model_dir in sorted(p for p in tier_dir.iterdir() if p.is_dir()):
            for preset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                row: dict[str, Any] = {
                    "tier": tier_dir.name,
                    "model": model_dir.name,
                    "preset": preset_dir.name,
                }
                lm_eval = preset_dir / "lm_eval.json"
                if lm_eval.exists():
                    data = json.loads(lm_eval.read_text())
                    for task, metrics in data.items():
                        for metric, value in metrics.items():
                            row[f"{task}_{metric}"] = value
                bench = preset_dir / "llama_bench.json"
                if bench.exists():
                    data = json.loads(bench.read_text())
                    by_ctx = data.get("tok_per_sec_by_ctx", {})
                    for k, v in by_ctx.items():
                        # ctx_1024_tg → tg_1024
                        parts = k.split("_")
                        if len(parts) == 3 and parts[0] == "ctx":
                            row[f"{parts[2]}_{parts[1]}"] = v
                resources = preset_dir / "resources.json"
                if resources.exists():
                    data = json.loads(resources.read_text())
                    row["peak_vram_mb"] = data.get("peak_vram_used_mb")
                    row["peak_rss_kb"] = data.get("peak_rss_kb")
                    row["peak_mlocked_kb"] = data.get("peak_mlocked_kb")
                stability = preset_dir / "stability.json"
                if stability.exists():
                    data = json.loads(stability.read_text())
                    row["stability_passed"] = data.get("passed_5pct_gate")
                    row["stability_max_drop"] = data.get("max_relative_drop")
                rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in rows:
                w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/moe-budget"))
    args = parser.parse_args()
    rows = aggregate_results(args.results_root)
    write_outputs(rows, args.results_root)
    print(f"Aggregated {len(rows)} cells. Outputs: {args.results_root}/summary.json, summary.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + aggregation**

```bash
python -m pytest tests/moe/test_aggregate.py -v
python -m scripts.moe.aggregate
```

Expected: PASS, then summary files written.

- [ ] **Step 5: Commit**

```bash
git add scripts/moe/aggregate.py tests/moe/test_aggregate.py results/moe-budget/summary.json results/moe-budget/summary.csv
git commit -m "Add results aggregator and write summary tables"
```

---

### Task 31: Write `docs/moe-on-a-budget.md`

**Files:** Create `docs/moe-on-a-budget.md`

- [ ] **Step 1: Draft the writeup**

Write `docs/moe-on-a-budget.md` with these sections (use **actual measured numbers from `results/moe-budget/summary.json`** — placeholders are NOT acceptable):

1. **The thesis** — VRAM holds active; RAM holds total
2. **The five flags (recipe summary, link to `the-five-flags.md`)**
3. **2GB tier walkthrough** — table of OLMoE vs TinyLlama on MMLU/GSM8K/HellaSwag/TruthfulQA/ARC, plus tok/s, plus a one-paragraph finding
4. **4GB tier walkthrough** — does 30B-A3B fit a 1050 Ti? table + finding
5. **6GB tier walkthrough** — article reproduction confirmation, plus per-flag attribution table from flag-sweep
6. **8GB tier walkthrough** — sleeper card story, plus 24GB→64GB RAM unlock
7. **Cross-tier findings** — speedup-per-flag chart, RAM-as-second-axis chart
8. **What didn't work** — speculative decoding (note from spec/article)
9. **Transfer to V100** — placeholder section (filled if Task 33 happens)
10. **AI-last frame** — closing message

- [ ] **Step 2: Verify all numbers come from real result files**

```bash
grep -nE 'TBD|TODO|<placeholder>' docs/moe-on-a-budget.md
```

Expected: empty.

- [ ] **Step 3: Commit**

```bash
git add docs/moe-on-a-budget.md
git commit -m "Write moe-on-a-budget docs page with measured findings"
```

---

### Task 32: Write `docs/the-five-flags.md`

**Files:** Create `docs/the-five-flags.md`

- [ ] **Step 1: Write the standalone recipe doc**

Write `docs/the-five-flags.md` with structure:

1. **TL;DR** — one Docker command containing all five flags, copy-pasteable
2. **Flag 1: `--n-cpu-moe N`** — what it does, when to use, how to tune
3. **Flag 2: `--no-mmap`** — what, when, RAM cost
4. **Flag 3: VRAM tuning of `--n-cpu-moe`** — how to find the sweet spot
5. **Flag 4: Turbo Quant KV (`--cache-type-k q4_0 --cache-type-v q3_0`)** — what, why GQA matters, when to skip
6. **Flag 5: `--mlock`** — what, container/Docker prerequisites, what it prevents
7. **Per-flag measured impact table** — pulled from 6GB sweep
8. **What we tried that didn't work** — speculative decoding regression, brief

- [ ] **Step 2: Commit**

```bash
git add docs/the-five-flags.md
git commit -m "Write the-five-flags standalone recipe doc"
```

---

### Task 33: Update README.md tier table and mkdocs nav

**Files:**
- Modify: `README.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Update README.md tier table**

Open `README.md`. Find the existing tier table. Add a new column **"Best MoE that fits"** with measured top MoE per tier (e.g., 2GB → OLMoE-1B-7B with MMLU score; 6GB → Qwen3-30B-A3B with article-reproduction tok/s).

Add after the tier table:

```markdown
**Going further:** see [MoE on a Budget](docs/moe-on-a-budget.md) for how to run 30B-parameter
Mixture-of-Experts models on 4–8GB GPUs by spending system RAM instead of VRAM, and
[The Five Flags](docs/the-five-flags.md) for the standalone recipe.
```

- [ ] **Step 2: Update mkdocs.yml nav**

Open `mkdocs.yml`. Add to the nav entries:

```yaml
- "MoE on a Budget": moe-on-a-budget.md
- "The Five Flags": the-five-flags.md
```

- [ ] **Step 3: Build the docs locally and verify**

```bash
mkdocs build --strict
```

Expected: no errors. New pages appear in nav.

- [ ] **Step 4: Commit**

```bash
git add README.md mkdocs.yml
git commit -m "Surface MoE-budget findings in README tier table and docs nav"
```

---

## Phase 10: Final integrity sweep

### Task 34: Repository integrity check

**Files:** none modified — verification only.

- [ ] **Step 1: Verify no placeholder text in committed deliverables**

```bash
grep -rnE 'TBD-FILL-IN|<FILL-IN|TODO|FIXME' \
  configs/moe-budget/ docs/moe-on-a-budget.md docs/the-five-flags.md \
  results/moe-budget/summary.json
```

Expected: empty.

- [ ] **Step 2: Verify every cell has a complete output set**

```bash
for cell in results/moe-budget/*/*/*/; do
  for required in config.yaml lm_eval.json llama_bench.json resources.json; do
    if [ ! -s "$cell/$required" ]; then
      echo "MISSING/EMPTY: $cell/$required"
    fi
  done
done
```

Expected: no MISSING/EMPTY lines.

- [ ] **Step 3: Verify aggregate summary covers all expected cells**

Count cells in `summary.json` and compare to expected matrix size (~17 cells; some may be OOM/skipped — document each skip in the writeup):

```bash
python -c "import json; print(len(json.load(open('results/moe-budget/summary.json'))))"
```

- [ ] **Step 4: Verify docs build clean**

```bash
mkdocs build --strict
```

Expected: no errors, no warnings.

- [ ] **Step 5: Commit any final fixes**

```bash
git status
# If anything modified during cleanup, commit it
git add -A
git commit -m "Final integrity sweep — moe-budget study deliverables complete"
```

---

## Out of scope for this plan (parking lot)

- **V100 32GB transfer experiment** — see spec §7. Triggered only after consumer-tier results are validated. Will be its own follow-up plan.
- **NVMe-as-RAM-extension** — for models exceeding RAM. See spec §7 future work.
- **MoE inference engine comparison** (vLLM, SGLang, TensorRT-LLM) — separate study.
- **Asymmetric expert-weight quantization** — separate study.

These are documented in the spec's §11 follow-up list. Do not attempt within this plan.

---

## Appendix: How to use this plan

- **Each task is a unit of work.** Mark `[x]` as you complete each step.
- **TDD discipline:** never commit production code before its test passes.
- **Frequent commits:** commit at the end of every task, not less often. The plan's commit messages are starter copy — refine if useful.
- **When a task fails:** investigate. Do not delete the failing test or skip the step. The plan's value comes from honest negative-result reporting.
- **Long-running benchmarks** can be kicked off and walked away from — the rig is dedicated. Use `tmux` or `screen` for headless runs.
- **If a hardware swap fails** (e.g., GTX 950 doesn't run Q4_K_M MoE) — document the failure in the writeup, drop that tier from primary scope, continue with the rest.

