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
