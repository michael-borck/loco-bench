"""Tests for hardware fingerprint capture."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.moe.hardware_fingerprint import (
    parse_dmidecode_memory,
    parse_lscpu_output,
)


def test_parse_lscpu_extracts_model_name() -> None:
    raw = """Architecture:                    x86_64
CPU(s):                          8
Model name:                      Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
Thread(s) per core:              2
Core(s) per socket:              4
Socket(s):                       1
"""

    parsed = parse_lscpu_output(raw)

    assert parsed["model"] == "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz"
    assert parsed["cores_total"] == 8
    assert parsed["sockets"] == 1


def test_parse_dmidecode_memory_lists_dimms() -> None:
    raw = """
Memory Device
        Size: 8192 MB
        Type: DDR4
        Speed: 2400 MT/s
        Locator: DIMM_A1

Memory Device
        Size: 8192 MB
        Type: DDR4
        Speed: 2400 MT/s
        Locator: DIMM_A2

Memory Device
        Size: No Module Installed
        Locator: DIMM_B1
"""

    dimms = parse_dmidecode_memory(raw)

    assert len(dimms) == 2
    assert dimms[0]["size_mb"] == 8192
    assert dimms[0]["speed_mts"] == 2400
    assert dimms[0]["locator"] == "DIMM_A1"
