"""Boot a llama.cpp server in Docker, wait for /health, return URL."""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from scripts.moe.config import Preset, preset_to_llama_cpp_args


@dataclass
class ServerConfig:
    """Container + host config for one llama.cpp server boot."""
    image: str
    gguf_path: Path
    host_port: int = 8080
    ipc_lock: bool = True
    container_name: str | None = None


def build_docker_command(cfg: ServerConfig, preset: Preset) -> list[str]:
    """Compose the `docker run ...` command line for a server boot."""
    cmd: list[str] = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-p", f"{cfg.host_port}:8080",
        "-v", f"{cfg.gguf_path.parent}:/models:ro",
    ]
    if cfg.ipc_lock:
        cmd += ["--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1"]
    if cfg.container_name:
        cmd += ["--name", cfg.container_name]

    cmd += [
        cfg.image,
        "--server",
        "--host", "0.0.0.0",
        "--port", "8080",
        "-m", f"/models/{cfg.gguf_path.name}",
    ]
    cmd += preset_to_llama_cpp_args(preset)
    return cmd


class LlamaCppServer:
    """Lifecycle wrapper for a llama.cpp Docker server."""

    def __init__(
        self,
        cfg: ServerConfig,
        preset: Preset,
        health_poll_interval_s: float = 2.0,
        health_timeout_s: float = 600.0,
    ) -> None:
        self.cfg = cfg
        self.preset = preset
        self._health_interval = health_poll_interval_s
        self._health_timeout = health_timeout_s
        self._proc: subprocess.Popen | None = None
        self.url = f"http://localhost:{cfg.host_port}"

    def start(self) -> None:
        cmd = build_docker_command(self.cfg, self.preset)
        print(f"[server] launching: {' '.join(shlex.quote(c) for c in cmd)}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_for_health()

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._health_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp server exited early with code {self._proc.returncode}"
                )
            try:
                r = requests.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(self._health_interval)
        raise TimeoutError(f"llama.cpp server did not become healthy within {self._health_timeout}s")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def __enter__(self) -> "LlamaCppServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
