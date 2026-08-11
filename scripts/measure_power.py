#!/usr/bin/env python3
"""Measure wall-plug power draw for local inference, and derive energy per token.

Total cost of ownership models need energy per unit of work, not peak wattage.
Cloud AI is priced per token, so the local side has to reach the same
denominator before the two can be compared. This script produces that number.

It measures at the wall rather than reading nvidia-smi, because TCO covers the
whole machine: CPU, drives, fans, and power supply conversion losses. Card-only
figures typically understate the real draw by 20 to 35 percent. nvidia-smi is
still useful as a cross-check and is recorded when available.

Two measurements matter, and the less obvious one is idle. A machine serving a
20 person department sits idle most of the working day, so idle draw often
dominates annual energy use. Idle needs no workload and no model, so run that
first:

    python scripts/measure_power.py --plug-url http://192.168.1.50 \
        --label "bench-rtx3060" --idle-only --duration 300

Then measure under sustained generation to get energy per token:

    python scripts/measure_power.py --plug-url http://192.168.1.50 \
        --label "bench-rtx3060" --model models/Qwen3-4B-Q4_K_M.gguf --ngl 99

Supported meters are Shelly plugs (Gen1 and Gen2), which expose a local HTTP
endpoint needing no cloud account and no credentials. Cloud-only plugs such as
the TP-Link Tapo range are deliberately not supported; the authentication dance
is not worth it when a Shelly costs about the same.

Without --plug-url the script falls back to nvidia-smi, which measures the card
alone. That is better than nothing for relative comparisons between cards, but
do not feed it into a TCO model as though it were wall draw.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_INTERVAL_S = 1.0


# --------------------------------------------------------------------------
# Meters
# --------------------------------------------------------------------------


def read_shelly(base_url: str, timeout: float = 3.0) -> float:
    """Read instantaneous watts from a Shelly plug's local HTTP API.

    Gen1 (Shelly Plug S)      GET /meter/0            -> {"power": 12.3, ...}
    Gen2 (Shelly Plus Plug S) GET /rpc/Switch.GetStatus?id=0 -> {"apower": 12.3, ...}
    """
    base = base_url.rstrip("/")
    attempts = [
        (f"{base}/meter/0", "power"),
        (f"{base}/rpc/Switch.GetStatus?id=0", "apower"),
    ]

    last_error = None
    for url, key in attempts:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            if key in data and data[key] is not None:
                return float(data[key])
        except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError) as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not read power from {base_url}. Tried Gen1 and Gen2 endpoints. "
        f"Last error: {last_error}. Check the plug's IP and that it is on the "
        f"same network."
    )


def read_nvidia_smi(timeout: float = 3.0) -> float:
    """Sum instantaneous draw across all NVIDIA GPUs, in watts (card only)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

    watts = [float(line) for line in result.stdout.split("\n") if line.strip()]
    if not watts:
        raise RuntimeError("nvidia-smi returned no readings")
    return sum(watts)


def make_meter(plug_url):
    """Return (reader, kind). Prefers the wall plug, falls back to nvidia-smi."""
    if plug_url:
        read_shelly(plug_url)  # fail fast with a clear message
        return (lambda: read_shelly(plug_url)), "wall-plug"

    try:
        read_nvidia_smi()
    except Exception as e:
        print(
            f"Error: no --plug-url given and nvidia-smi is unusable ({e}).\n"
            f"Supply a Shelly plug URL, or run on a machine with NVIDIA drivers.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "Warning: measuring with nvidia-smi (card only). This excludes CPU, "
        "drives, fans and PSU losses, so it understates true wall draw by "
        "roughly 20 to 35 percent. Use a wall plug for TCO figures.\n",
        file=sys.stderr,
    )
    return read_nvidia_smi, "nvidia-smi"


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


class PowerSampler:
    """Samples a meter on a background thread and summarises the window."""

    def __init__(self, reader, interval=SAMPLE_INTERVAL_S):
        self._reader = reader
        self._interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._started_at = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._samples.append(self._reader())
            except Exception:
                pass  # a dropped sample is not worth aborting a long run for
            self._stop.wait(self._interval)

    def __enter__(self):
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._interval * 2 + 1)
        self.elapsed_s = time.monotonic() - self._started_at
        return False

    def summary(self) -> dict:
        s = self._samples
        if not s:
            raise RuntimeError("No power samples collected. Is the meter reachable?")
        mean_w = statistics.fmean(s)
        return {
            "mean_w": round(mean_w, 2),
            "median_w": round(statistics.median(s), 2),
            "min_w": round(min(s), 2),
            "max_w": round(max(s), 2),
            "stdev_w": round(statistics.stdev(s), 2) if len(s) > 1 else 0.0,
            "samples": len(s),
            "duration_s": round(self.elapsed_s, 1),
            "energy_wh": round(mean_w * self.elapsed_s / 3600, 4),
        }


def measure_idle(reader, duration_s: int) -> dict:
    """Sample with no workload running."""
    print(f"Measuring idle for {duration_s}s. Leave the machine alone.")
    with PowerSampler(reader) as sampler:
        time.sleep(duration_s)
    result = sampler.summary()
    print(f"  Idle: {result['mean_w']:.1f} W (min {result['min_w']}, max {result['max_w']})")
    return result


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------


def run_llama_bench(model, llama_bench, prompt_tokens, gen_tokens, ngl, repetitions, threads):
    """Run llama-bench, returning its parsed JSON output."""
    cmd = [
        llama_bench,
        "-m", str(model),
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-ngl", str(ngl),
        "-r", str(repetitions),
        "-o", "json",
    ]
    if threads:
        cmd += ["-t", str(threads)]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"llama-bench failed (exit {result.returncode})")
    return json.loads(result.stdout)


def extract_generation_rate(bench_data) -> float:
    """Pull the text-generation rate (tokens/sec) out of llama-bench JSON."""
    for entry in bench_data:
        if entry.get("test") == "tg" or str(entry.get("n_gen", 0)) not in ("0", "None"):
            rate = entry.get("avg_ts")
            if rate:
                return float(rate)
    raise RuntimeError("No generation rate found in llama-bench output")


def measure_load(reader, **bench_kwargs) -> tuple[dict, float]:
    """Sample power while llama-bench generates. Returns (power, tokens_per_s)."""
    print("Measuring under sustained generation.")
    with PowerSampler(reader) as sampler:
        bench_data = run_llama_bench(**bench_kwargs)
    power = sampler.summary()
    rate = extract_generation_rate(bench_data)
    print(f"  Load: {power['mean_w']:.1f} W at {rate:.1f} tok/s")
    return power, rate


# --------------------------------------------------------------------------
# Derived TCO figures
# --------------------------------------------------------------------------


def derive_energy_per_token(idle_w: float, load_w: float, tokens_per_s: float) -> dict:
    """Convert watts and tokens/sec into the units a TCO model can use.

    watts / (tokens/sec) = joules per token. Multiply by 1000 and divide by
    3600 to reach watt-hours per thousand tokens, which is directly comparable
    to per-token cloud pricing.

    Two figures, because they answer different questions:

    marginal  the extra energy of serving one more thousand tokens on a machine
              that is already powered on. Use this for incremental workload.
    total     the same, ignoring nothing, as though the machine existed solely
              to serve those tokens.

    Neither includes idle time between requests. That depends on the duty cycle,
    which is a modelling assumption rather than a measurement, so it belongs in
    the spreadsheet and not here.
    """
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be positive")

    marginal_w = max(load_w - idle_w, 0.0)
    to_wh_per_1k = lambda w: round(w / tokens_per_s * 1000 / 3600, 4)

    return {
        "wh_per_1k_tokens_marginal": to_wh_per_1k(marginal_w),
        "wh_per_1k_tokens_total": to_wh_per_1k(load_w),
        "marginal_w": round(marginal_w, 2),
        "idle_kwh_per_year_if_always_on": round(idle_w * 8760 / 1000, 1),
    }


def host_info() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["gpus"] = [line.strip() for line in result.stdout.split("\n") if line.strip()]
    except Exception:
        pass
    return info


def main():
    p = argparse.ArgumentParser(
        description="Measure wall-plug power and derive energy per token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Supported meters")[0].split("\n", 2)[2],
    )
    p.add_argument("--label", required=True,
                   help="Name for this configuration, e.g. 'bench-rtx3060' or 'dept-2x-p40'")
    p.add_argument("--plug-url",
                   help="Shelly plug base URL, e.g. http://192.168.1.50. "
                        "Omitted, falls back to nvidia-smi (card only).")
    p.add_argument("--idle-only", action="store_true",
                   help="Measure idle draw and stop. Needs no model.")
    p.add_argument("--duration", type=int, default=180,
                   help="Idle sampling duration in seconds (default: 180)")
    p.add_argument("--model", type=Path,
                   help="GGUF model to generate with. Required unless --idle-only.")
    p.add_argument("--llama-bench", default="llama-bench",
                   help="Path to llama-bench binary (default: llama-bench)")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=512,
                   help="Generation length; longer gives a steadier power reading "
                        "(default: 512)")
    p.add_argument("--ngl", type=int, default=99,
                   help="GPU layers; 99 = full GPU, 0 = CPU only (default: 99)")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results/power/"))
    args = p.parse_args()

    if not args.idle_only and not args.model:
        p.error("--model is required unless --idle-only is given")

    reader, meter_kind = make_meter(args.plug_url)

    record = {
        "label": args.label,
        "meter": meter_kind,
        "host": host_info(),
    }
    if meter_kind == "nvidia-smi":
        record["warning"] = (
            "Card-only measurement. Excludes CPU, drives, fans and PSU losses. "
            "Not suitable as a TCO input without adjustment."
        )

    print(f"\nConfiguration: {args.label}  (meter: {meter_kind})\n")

    idle = measure_idle(reader, args.duration)
    record["idle"] = idle

    if args.idle_only:
        record["idle_kwh_per_year_if_always_on"] = round(idle["mean_w"] * 8760 / 1000, 1)
        print(f"\n  If always on: {record['idle_kwh_per_year_if_always_on']} kWh/year at idle")
    else:
        print()
        load, rate = measure_load(
            reader,
            model=args.model,
            llama_bench=args.llama_bench,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            ngl=args.ngl,
            repetitions=args.repetitions,
            threads=args.threads,
        )
        record["load"] = load
        record["model"] = args.model.name
        record["generation_tokens_per_s"] = round(rate, 2)
        record["derived"] = derive_energy_per_token(idle["mean_w"], load["mean_w"], rate)

        d = record["derived"]
        print(f"\n  Energy per 1000 tokens: {d['wh_per_1k_tokens_marginal']} Wh marginal, "
              f"{d['wh_per_1k_tokens_total']} Wh total")
        print(f"  Idle if always on:      {d['idle_kwh_per_year_if_always_on']} kWh/year")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "idle" if args.idle_only else "load"
    out = args.output_dir / f"{args.label}-{suffix}-{stamp}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
