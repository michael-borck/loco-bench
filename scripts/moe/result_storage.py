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
    lm_eval_raw_dir: Path
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
    lm_eval_raw_dir = cell_dir / "lm_eval_raw"
    lm_eval_raw_dir.mkdir(parents=True, exist_ok=True)
    return CellPaths(
        dir=cell_dir,
        config_yaml=cell_dir / "config.yaml",
        lm_eval_json=cell_dir / "lm_eval.json",
        lm_eval_raw_dir=lm_eval_raw_dir,
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
