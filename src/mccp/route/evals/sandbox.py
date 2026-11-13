"""Run model-written code without trusting it.

`DockerSandbox` (production): a throwaway container per run with no network, a
read-only root filesystem, a small tmpfs, all capabilities dropped,
no-new-privileges, an unprivileged user, and CPU / memory / process / wall-clock
limits. The task files are mounted read-only.

`LocalSandbox` (development and tests only): a child process in a fresh network
namespace (`unshare -n`) with CPU, memory, file-size and process rlimits. It is
weaker isolation than a container (same filesystem view), so the evaluation CLI
refuses it for real runs unless explicitly allowed.
"""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

OUTPUT_LIMIT = 4000  # characters of stdout/stderr kept for diagnosis


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    timed_out: bool
    output: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(Protocol):
    kind: str

    async def run(self, files: dict[str, str], argv: list[str], timeout_s: float) -> RunResult: ...


def _write(files: dict[str, str]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="route-eval-"))
    for name, text in files.items():
        if "/" in name or name.startswith("."):
            raise ValueError(f"unsafe file name {name!r}")
        (d / name).write_text(text, encoding="utf-8")
    os.chmod(d, 0o755)
    for p in d.iterdir():
        os.chmod(p, 0o644)
    return d


async def _communicate(proc: asyncio.subprocess.Process, timeout_s: float, on_timeout) -> tuple[bytes, bool]:
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        return out or b"", False
    except TimeoutError:
        await on_timeout()
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        except TimeoutError:
            out = b""
        return out or b"", True


class DockerSandbox:
    kind = "docker"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        memory: str = "512m",
        cpus: str = "1",
        pids: int = 128,
        docker: str = "docker",
    ):
        self.image, self.memory, self.cpus, self.pids, self.docker = image, memory, cpus, pids, docker

    def argv(self, workdir: Path, name: str, argv: list[str]) -> list[str]:
        return [
            self.docker, "run", "--rm", "--name", name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--memory", self.memory, "--memory-swap", self.memory,
            "--cpus", self.cpus, "--pids-limit", str(self.pids),
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{workdir}:/work:ro", "-w", "/work",
            self.image, *argv,
        ]  # fmt: skip

    async def run(self, files: dict[str, str], argv: list[str], timeout_s: float) -> RunResult:
        d = _write(files)
        name = f"route-eval-{os.urandom(6).hex()}"
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.argv(d, name, argv), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )

            async def kill() -> None:
                k = await asyncio.create_subprocess_exec(
                    self.docker, "kill", name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )
                await k.wait()

            out, timed_out = await _communicate(proc, timeout_s, kill)
            return RunResult(
                proc.returncode if proc.returncode is not None else -1,
                timed_out,
                out.decode("utf-8", "replace")[-OUTPUT_LIMIT:],
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)


class LocalSandbox:
    kind = "local"

    def __init__(self, python: str | None = None, memory_mb: int = 512, cpu_s: int = 20) -> None:
        import sys

        self.python = python or sys.executable
        self.memory_mb, self.cpu_s = memory_mb, cpu_s
        self.unshare = shutil.which("unshare")

    def _limits(self) -> None:  # runs in the child before exec
        mem = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_s, self.cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        os.setsid()

    async def run(self, files: dict[str, str], argv: list[str], timeout_s: float) -> RunResult:
        d = _write(files)
        cmd = [self.python if a == "python" else a for a in argv]
        if self.unshare:
            cmd = [self.unshare, "-n", *cmd] if os.geteuid() == 0 else [self.unshare, "-rn", *cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=d, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                preexec_fn=self._limits, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(d)},
            )  # fmt: skip

            async def kill() -> None:
                try:
                    os.killpg(proc.pid, 9)
                except ProcessLookupError:
                    pass

            out, timed_out = await _communicate(proc, timeout_s, kill)
            return RunResult(
                proc.returncode if proc.returncode is not None else -1,
                timed_out,
                out.decode("utf-8", "replace")[-OUTPUT_LIMIT:],
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)
