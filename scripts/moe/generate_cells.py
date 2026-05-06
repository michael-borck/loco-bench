"""Generate the full (tier × model × preset) cell matrix from spec.

The MATRIX constant is the single source of truth for which cells get run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


# Per-tier `n_cpu_moe` defaults. The 6gb and 6gb-256k values come from the article (Qwen3-30B-A3B
# on GTX 1060 6GB). The 2gb, 4gb, 8gb-32gbram, and 8gb-64gbram values are PROVISIONAL — they will
# be refined empirically during the calibration task (Plan Task 18 smoke test and Phase 7 runs).
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
