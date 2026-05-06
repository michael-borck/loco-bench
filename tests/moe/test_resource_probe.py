"""Tests for resource probe."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.moe.resource_probe import (
    ResourceSnapshot,
    parse_nvidia_smi_output,
    parse_proc_status,
    snapshot,
)


def test_parse_nvidia_smi_output_csv() -> None:
    raw = "5824, 6144\n"  # used, total in MB

    used_mb, total_mb = parse_nvidia_smi_output(raw)

    assert used_mb == 5824
    assert total_mb == 6144


def test_parse_proc_status_extracts_rss_lck_hwm() -> None:
    raw = """
Name:   llama-server
VmHWM:    24112844 kB
VmRSS:    24108200 kB
VmLck:    16776832 kB
VmPeak:   24500000 kB
"""

    parsed = parse_proc_status(raw)

    assert parsed["VmRSS_kb"] == 24108200
    assert parsed["VmLck_kb"] == 16776832
    assert parsed["VmHWM_kb"] == 24112844


@patch("scripts.moe.resource_probe._read_proc_status")
@patch("scripts.moe.resource_probe._run_nvidia_smi")
def test_snapshot_combines_sources(mock_smi, mock_proc) -> None:
    mock_smi.return_value = "5824, 6144\n"
    mock_proc.return_value = "VmRSS:    24108200 kB\nVmLck:    16776832 kB\nVmHWM:    24112844 kB\n"

    snap = snapshot(pid=12345)

    assert isinstance(snap, ResourceSnapshot)
    assert snap.vram_used_mb == 5824
    assert snap.vram_total_mb == 6144
    assert snap.rss_kb == 24108200
    assert snap.mlocked_kb == 16776832
