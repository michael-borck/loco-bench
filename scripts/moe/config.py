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
