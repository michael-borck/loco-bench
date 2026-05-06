"""Probe GPU and process memory state for benchmark snapshots."""
from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResourceSnapshot:
    """One point-in-time snapshot of memory usage."""
    timestamp: float
    vram_used_mb: int
    vram_total_mb: int
    rss_kb: int
    mlocked_kb: int
    hwm_kb: int


def parse_nvidia_smi_output(raw: str) -> tuple[int, int]:
    """Parse `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader` output."""
    line = raw.strip().splitlines()[0]
    used, total = (int(s.strip()) for s in line.split(","))
    return used, total


def parse_proc_status(raw: str) -> dict[str, int]:
    """Parse /proc/<pid>/status into a dict keyed by 'VmRSS_kb', 'VmLck_kb', 'VmHWM_kb'."""
    out: dict[str, int] = {}
    keys = {"VmRSS": "VmRSS_kb", "VmLck": "VmLck_kb", "VmHWM": "VmHWM_kb"}
    for line in raw.splitlines():
        for prefix, dest in keys.items():
            if line.startswith(prefix + ":"):
                value = line.split(":", 1)[1].strip().split()[0]
                out[dest] = int(value)
    return out


def _run_nvidia_smi() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def _read_proc_status(pid: int) -> str:
    return Path(f"/proc/{pid}/status").read_text()


def snapshot(pid: int) -> ResourceSnapshot:
    """Take a single snapshot of GPU + process memory."""
    smi = parse_nvidia_smi_output(_run_nvidia_smi())
    proc = parse_proc_status(_read_proc_status(pid))
    return ResourceSnapshot(
        timestamp=time.time(),
        vram_used_mb=smi[0],
        vram_total_mb=smi[1],
        rss_kb=proc.get("VmRSS_kb", 0),
        mlocked_kb=proc.get("VmLck_kb", 0),
        hwm_kb=proc.get("VmHWM_kb", 0),
    )


def poll_loop(pid: int, duration_s: float, interval_s: float = 1.0) -> list[ResourceSnapshot]:
    """Poll repeatedly for duration_s, return list of snapshots."""
    snaps: list[ResourceSnapshot] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        try:
            snaps.append(snapshot(pid))
        except Exception as e:
            # Process may have exited; record and bail
            print(f"[probe] snapshot failed: {e}")
            break
        time.sleep(interval_s)
    return snaps


def snapshots_to_summary(snaps: list[ResourceSnapshot]) -> dict[str, Any]:
    """Reduce a list of snapshots to a peak summary dict."""
    if not snaps:
        return {"samples": 0}
    return {
        "samples": len(snaps),
        "peak_vram_used_mb": max(s.vram_used_mb for s in snaps),
        "vram_total_mb": snaps[-1].vram_total_mb,
        "peak_rss_kb": max(s.rss_kb for s in snaps),
        "peak_mlocked_kb": max(s.mlocked_kb for s in snaps),
        "peak_hwm_kb": max(s.hwm_kb for s in snaps),
        "snapshots": [asdict(s) for s in snaps],
    }
