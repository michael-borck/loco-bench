"""Capture host hardware fingerprint into a config.yaml host block."""
from __future__ import annotations

import platform
import re
import subprocess
from typing import Any


def parse_lscpu_output(raw: str) -> dict[str, Any]:
    """Extract model, core count, socket count from `lscpu` output."""
    out: dict[str, Any] = {}
    for line in raw.splitlines():
        if line.startswith("Model name:"):
            out["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):"):
            out["cores_total"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Socket(s):"):
            out["sockets"] = int(line.split(":", 1)[1].strip())
    return out


def parse_dmidecode_memory(raw: str) -> list[dict[str, Any]]:
    """Extract installed DIMMs from `dmidecode -t memory` output."""
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    in_device = False
    for line in raw.splitlines():
        s = line.strip()
        if s == "Memory Device":
            if current and current.get("size_mb"):
                devices.append(current)
            current = {}
            in_device = True
            continue
        if not in_device:
            continue
        if s.startswith("Size:"):
            val = s.split(":", 1)[1].strip()
            if "No Module" in val:
                continue
            m = re.match(r"(\d+)\s+(MB|GB)", val)
            if m:
                size = int(m.group(1))
                if m.group(2) == "GB":
                    size *= 1024
                current["size_mb"] = size
        elif s.startswith("Speed:"):
            m = re.search(r"(\d+)\s*MT/s", s)
            if m:
                current["speed_mts"] = int(m.group(1))
        elif s.startswith("Locator:"):
            current["locator"] = s.split(":", 1)[1].strip()
        elif s.startswith("Type:"):
            current["type"] = s.split(":", 1)[1].strip()
    if current and current.get("size_mb"):
        devices.append(current)
    return devices


def fingerprint_host() -> dict[str, Any]:
    """Run host introspection commands and return a host dict."""
    cpu_info: dict[str, Any] = {}
    dimms: list[dict[str, Any]] = []
    try:
        cpu_info = parse_lscpu_output(subprocess.check_output(["lscpu"], text=True))
    except Exception as e:
        cpu_info = {"error": str(e)}
    try:
        dimms = parse_dmidecode_memory(
            subprocess.check_output(["sudo", "-n", "dmidecode", "-t", "memory"], text=True)
        )
    except Exception as e:
        dimms = [{"error": str(e)}]
    total_ram_mb = sum(d.get("size_mb", 0) for d in dimms if "error" not in d)
    speeds = [d.get("speed_mts") for d in dimms if "speed_mts" in d]

    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu": cpu_info,
        "ram_gb": total_ram_mb // 1024,
        "ram_speed_mts": speeds[0] if speeds else None,
        "ram_dimms": dimms,
    }


def fingerprint_gpu() -> dict[str, Any]:
    """Run nvidia-smi --query-gpu and return one GPU's info."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()[0]
        name, mem, drv, pci = (s.strip() for s in raw.split(","))
        return {"model": name, "vram_total": mem, "driver": drv, "pci_bus": pci}
    except Exception as e:
        return {"error": str(e)}
