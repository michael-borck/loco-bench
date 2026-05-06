"""Tests for llama.cpp server wrapper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.moe.config import Preset
from scripts.moe.server import (
    LlamaCppServer,
    ServerConfig,
    build_docker_command,
)


def test_build_docker_command_includes_preset_args() -> None:
    preset = Preset(
        name="optimized-moe",
        flags={"ngl": 99, "n_cpu_moe": 35, "no_mmap": True, "mlock": True, "ctx_size": 65536},
    )
    cfg = ServerConfig(
        image="ghcr.io/ggml-org/llama.cpp:server-cuda-test",
        gguf_path=Path("/models/qwen3-30b-a3b-q4_k_m.gguf"),
        host_port=8080,
        ipc_lock=True,
    )

    cmd = build_docker_command(cfg, preset)

    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--rm" in cmd
    assert "--gpus" in cmd and "all" in cmd
    assert "--cap-add" in cmd and "IPC_LOCK" in cmd
    assert "-p" in cmd and "8080:8080" in cmd
    assert "ghcr.io/ggml-org/llama.cpp:server-cuda-test" in cmd
    # Model arg
    assert "-m" in cmd
    # Preset args translated through
    assert "--n-cpu-moe" in cmd
    assert "35" in cmd
    assert "--no-mmap" in cmd
    assert "--mlock" in cmd


def test_build_docker_command_omits_ipc_lock_when_disabled() -> None:
    preset = Preset(name="default", flags={"ngl": 99, "ctx_size": 8192})
    cfg = ServerConfig(
        image="test:image",
        gguf_path=Path("/models/x.gguf"),
        host_port=8080,
        ipc_lock=False,
    )

    cmd = build_docker_command(cfg, preset)

    assert "IPC_LOCK" not in cmd


@patch("scripts.moe.server.requests.get")
@patch("scripts.moe.server.subprocess.Popen")
def test_server_start_polls_health(mock_popen, mock_get) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    mock_popen.return_value = mock_proc

    # Simulate /health returning 200 on second poll
    mock_get.side_effect = [
        MagicMock(status_code=503),
        MagicMock(status_code=200, json=lambda: {"status": "ok"}),
    ]

    preset = Preset(name="default", flags={"ngl": 99, "ctx_size": 8192})
    cfg = ServerConfig(
        image="test:image", gguf_path=Path("/models/x.gguf"),
        host_port=8080, ipc_lock=False,
    )
    server = LlamaCppServer(cfg, preset, health_poll_interval_s=0.0, health_timeout_s=10)

    server.start()

    assert server.url == "http://localhost:8080"
    assert mock_get.call_count == 2
    server.stop()


@patch("scripts.moe.server.requests.get")
@patch("scripts.moe.server.subprocess.Popen")
def test_server_start_raises_on_health_timeout(mock_popen, mock_get) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_get.return_value = MagicMock(status_code=503)

    preset = Preset(name="default", flags={"ngl": 99, "ctx_size": 8192})
    cfg = ServerConfig(
        image="test:image", gguf_path=Path("/models/x.gguf"),
        host_port=8080, ipc_lock=False,
    )
    server = LlamaCppServer(cfg, preset, health_poll_interval_s=0.0, health_timeout_s=0.05)

    with pytest.raises(TimeoutError):
        server.start()
    server.stop()
