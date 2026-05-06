"""Wrap llama-bench for speed measurements at multiple context lengths."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SpeedSample:
    kind: str  # "pp" or "tg"
    n_tokens: int
    avg_ts: float
    stddev_ts: float


def build_llama_bench_command(
    *,
    gguf_path: Path,
    ngl: int,
    ctx_sizes: list[int],
    gen_tokens: int,
    repetitions: int,
    threads: int,
) -> list[str]:
    """Compose llama-bench CLI."""
    cmd: list[str] = [
        "llama-bench",
        "-m", str(gguf_path),
        "-ngl", str(ngl),
        "-p", ",".join(str(s) for s in ctx_sizes),
        "-n", str(gen_tokens),
        "-r", str(repetitions),
        "-t", str(threads),
        "-o", "json",
    ]
    return cmd


def parse_llama_bench_json(raw: list[dict[str, Any]]) -> list[SpeedSample]:
    """Convert llama-bench JSON output into SpeedSample list."""
    out: list[SpeedSample] = []
    for entry in raw:
        test = entry.get("test", "")
        kind = "pp" if test.startswith("pp") else "tg" if test.startswith("tg") else "?"
        n_tokens = int(entry.get("n_prompt") or entry.get("n_gen") or 0)
        out.append(SpeedSample(
            kind=kind,
            n_tokens=n_tokens,
            avg_ts=float(entry.get("avg_ts", 0.0)),
            stddev_ts=float(entry.get("stddev_ts", 0.0)),
        ))
    return out


def run_llama_bench(
    *,
    gguf_path: Path,
    ngl: int,
    ctx_sizes: list[int],
    gen_tokens: int = 128,
    repetitions: int = 3,
    threads: int = 4,
    output_json_path: Path | None = None,
) -> list[SpeedSample]:
    """Run llama-bench and return parsed samples. Optionally writes raw JSON."""
    cmd = build_llama_bench_command(
        gguf_path=gguf_path, ngl=ngl, ctx_sizes=ctx_sizes,
        gen_tokens=gen_tokens, repetitions=repetitions, threads=threads,
    )
    print(f"[llama-bench] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(result.stdout)
    if output_json_path:
        output_json_path.write_text(json.dumps(raw, indent=2))
    return parse_llama_bench_json(raw)


def samples_to_summary(samples: list[SpeedSample]) -> dict[str, Any]:
    """Summarize per-context-length tok/s for a results.json."""
    return {
        "samples": [asdict(s) for s in samples],
        "tok_per_sec_by_ctx": {
            f"ctx_{s.n_tokens}_{s.kind}": s.avg_ts for s in samples
        },
    }
