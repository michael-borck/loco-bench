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
