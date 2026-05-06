"""Tests for stability runner."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.moe.stability import StabilitySample, summarize_drift


def test_summarize_drift_computes_relative_change() -> None:
    samples = [
        StabilitySample(elapsed_s=0, tok_per_sec=17.0, vram_used_mb=5800),
        StabilitySample(elapsed_s=3600, tok_per_sec=16.9, vram_used_mb=5800),
        StabilitySample(elapsed_s=86400, tok_per_sec=16.7, vram_used_mb=5800),
        StabilitySample(elapsed_s=259200, tok_per_sec=16.5, vram_used_mb=5800),
    ]

    summary = summarize_drift(samples)

    assert summary["t0_tok_per_sec"] == pytest.approx(17.0)
    assert summary["max_relative_drop"] == pytest.approx((17.0 - 16.5) / 17.0, rel=1e-3)
    assert summary["passed_5pct_gate"] is True


def test_summarize_drift_fails_when_drop_exceeds_threshold() -> None:
    samples = [
        StabilitySample(elapsed_s=0, tok_per_sec=17.0, vram_used_mb=5800),
        StabilitySample(elapsed_s=259200, tok_per_sec=10.0, vram_used_mb=5800),
    ]

    summary = summarize_drift(samples)

    assert summary["passed_5pct_gate"] is False
    assert summary["max_relative_drop"] > 0.05
