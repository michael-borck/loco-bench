"""Run stability test: warmup → idle interval → re-measure tok/s.

Tests the article's "day 3 slowdown" claim. Pass criterion: <5% relative drop
from t=0 to t=72h with --mlock; expected to fail without it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_INTERVALS_S: list[int] = [0, 3600, 86400, 259200]  # t0, +1h, +24h, +72h


@dataclass
class StabilitySample:
    elapsed_s: int
    tok_per_sec: float
    vram_used_mb: int


def summarize_drift(samples: list[StabilitySample], threshold: float = 0.05) -> dict[str, Any]:
    """Reduce stability samples to pass/fail summary."""
    if not samples:
        return {"samples": 0, "passed_5pct_gate": False}
    t0 = samples[0].tok_per_sec
    drops = [(t0 - s.tok_per_sec) / t0 for s in samples]
    max_drop = max(drops)
    return {
        "samples": [asdict(s) for s in samples],
        "t0_tok_per_sec": t0,
        "max_relative_drop": max_drop,
        "passed_5pct_gate": max_drop < threshold,
    }


def run_stability(
    *,
    measure_fn: Callable[[], tuple[float, int]],
    intervals_s: list[int] = DEFAULT_INTERVALS_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Take samples at each elapsed time. measure_fn returns (tok_per_sec, vram_used_mb)."""
    samples: list[StabilitySample] = []
    elapsed = 0
    for target in intervals_s:
        if target > elapsed:
            sleep_fn(target - elapsed)
            elapsed = target
        tok_s, vram_mb = measure_fn()
        samples.append(StabilitySample(elapsed_s=elapsed, tok_per_sec=tok_s, vram_used_mb=vram_mb))
        print(f"[stability] t={elapsed}s tok/s={tok_s:.2f} vram={vram_mb}MB")

    summary = summarize_drift(samples)
    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2))
    return summary
