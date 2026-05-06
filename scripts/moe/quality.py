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
