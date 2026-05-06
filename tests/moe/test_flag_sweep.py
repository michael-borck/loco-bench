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
