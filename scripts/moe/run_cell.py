"""Top-level driver: take one cell config, run all benchmarks, write outputs.

Usage:
    python -m scripts.moe.run_cell \
        --cell configs/moe-budget/cells/6gb-1060-qwen3-30b-a3b-optimized-moe.yaml \
        --models-dir ./models \
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
    preset_to_llama_cpp_args,
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

    # Effective context size — caps both NIAH probe lengths and llama-bench upper context.
    # Both must stay within what the server was booted to handle.
    ctx = int(inputs.effective_preset.flags.get("ctx_size", 4096))

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
            output_dir=paths.lm_eval_raw_dir,
        )
        results_files = list(paths.lm_eval_raw_dir.glob("**/results*.json"))
        if results_files:
            parsed = parse_lm_eval_results(sorted(results_files, key=lambda p: p.stat().st_mtime)[-1])
            paths.lm_eval_json.write_text(json.dumps(parsed, indent=2))

        # NIAH (ctx already defined above)
        lengths = niah_lengths or [1000, 8000, 64000, 256000]
        lengths = [L for L in lengths if L <= ctx]
        if lengths:
            niah_summary = score_at_context_lengths(
                server_url=server.url, target_token_lengths=lengths
            )
            write_niah(paths.niah_json, niah_summary)

    # Speed via llama-bench (separate process, no server)
    # llama-bench needs the same MoE flags the server got, minus ngl/ctx_size which are explicit
    bench_extra_flags = {
        k: v for k, v in inputs.effective_preset.flags.items()
        if k not in ("ngl", "ctx_size")
    }
    bench_preset = Preset(name=f"{inputs.cell.preset}-llama-bench", flags=bench_extra_flags)
    bench_extra_args = preset_to_llama_cpp_args(bench_preset)
    speed_samples = run_llama_bench(
        gguf_path=gguf_path,
        ngl=int(inputs.effective_preset.flags.get("ngl", 99)),
        ctx_sizes=[1024, 8192, min(65536, ctx)],
        output_json_path=paths.dir / "llama_bench_raw.json",
        extra_args=bench_extra_args,
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
