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
