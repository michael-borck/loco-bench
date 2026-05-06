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


def test_extra_args_appended() -> None:
    cmd = build_llama_bench_command(
        gguf_path=Path("/m.gguf"),
        ngl=99,
        ctx_sizes=[1024],
        gen_tokens=128,
        repetitions=3,
        threads=4,
        extra_args=["--n-cpu-moe", "35", "--no-mmap"],
    )
    assert "--n-cpu-moe" in cmd and "35" in cmd
    assert "--no-mmap" in cmd


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
