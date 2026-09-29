"""An environment backend on unprivileged Linux namespaces (docs/decisions/0025-agent-sessions.md).

Each environment is a directory holding its own copy of a base image's root filesystem. A command runs in new user,
mount and PID namespaces, chrooted into that root filesystem, as root inside (mapped to the host user outside):

- its own filesystem (the host's files are not visible), process tree and working directory (`/workspace`);
- the host's network, so the internet works without root;
- nothing kept running between commands: each command gets a fresh process tree.

This isolates processes and files; it is not a security boundary against hostile code. Environments are plain
directories, so they survive restarts of the runner.
"""

import asyncio
import os
import shutil
import signal
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

from rollout.core.harness.environments import EnvironmentSpecification, ExecutionResult
from rollout.environments.images import ImageStore

MAX_OUTPUT_CHARACTERS = 60_000

# Runs inside the new namespaces, before entering the environment: mounts, then chroot with a clean environment.
ENTER = r"""
root="$1"; cwd="$2"; command="$3"
mount --rbind /dev "$root/dev" 2>/dev/null
mount -t proc proc "$root/proc"
mount -t tmpfs tmpfs "$root/tmp"
exec chroot "$root" /usr/bin/env -i HOME=/root TERM=dumb LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    /bin/sh -c 'cd "$1" || exit 125; eval "$2"' sh "$cwd" "$command"
"""


class NamespaceEnvironments:
    """Implements `EnvironmentService`."""

    def __init__(self, directory: Path, images: ImageStore | None = None) -> None:
        self.directory = directory
        self.images = images or ImageStore(directory / "images")
        self._creating: dict[str, asyncio.Lock] = {}

    def root(self, environment_id: str) -> Path:
        return self.directory / environment_id / "rootfs"

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        lock = self._creating.setdefault(environment_id, asyncio.Lock())
        async with lock:
            home = self.directory / environment_id
            if (home / "ready").exists():
                return  # already created: creation is idempotent
            tarball = await self.images.tarball(specification.image)
            await asyncio.to_thread(_unpack, tarball, home)
            for command in specification.setup:
                result = await self.execute(environment_id, command, timeout_seconds=600, cwd=None)
                if result.exit_code != 0:
                    raise RuntimeError(f"setup command failed ({result.exit_code}): {command}\n{result.output[-2000:]}")
            await asyncio.to_thread((home / "ready").write_text, specification.model_dump_json())

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None
    ) -> ExecutionResult:
        root = self.root(environment_id)
        if not root.exists():
            raise FileNotFoundError(f"environment {environment_id} does not exist")
        process = await asyncio.create_subprocess_exec(
            "unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child",
            "sh", "-c", ENTER, "enter", str(root), cwd or "/workspace", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
        try:
            async with asyncio.timeout(timeout_seconds):
                output, _ = await process.communicate()
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = await process.communicate()
            return _result(None, output, timed_out=True)
        return _result(process.returncode, output)

    async def put(self, environment_id: str, path: str, data: bytes) -> None:
        target = self._inside(environment_id, path)

        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(write)

    async def get(self, environment_id: str, path: str) -> bytes:
        return await asyncio.to_thread(self._inside(environment_id, path).read_bytes)

    async def destroy(self, environment_id: str) -> None:
        await asyncio.to_thread(_remove, self.directory / environment_id)

    def _inside(self, environment_id: str, path: str) -> Path:
        root = self.root(environment_id).resolve()
        relative = path if path.startswith("/") else f"workspace/{path}"
        resolved = (root / relative.lstrip("/")).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"{path} is outside the environment")
        return resolved


def _result(exit_code: int | None, output: bytes, timed_out: bool = False) -> ExecutionResult:
    text = output.decode("utf-8", errors="replace")
    truncated = len(text) > MAX_OUTPUT_CHARACTERS
    if truncated:
        half = MAX_OUTPUT_CHARACTERS // 2
        text = f"{text[:half]}\n… ({len(text) - MAX_OUTPUT_CHARACTERS} characters omitted) …\n{text[-half:]}"
    return ExecutionResult(exit_code=exit_code, output=text, truncated=truncated, timed_out=timed_out)


def _unpack(tarball: Path, home: Path) -> None:
    """Unpack the image into a fresh root filesystem, with a workspace and the host's DNS settings."""
    staging = home / "staging"
    _remove(staging)
    staging.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(staging)], check=True, capture_output=True)
    (staging / "workspace").mkdir(exist_ok=True)
    resolver = Path("/etc/resolv.conf")
    if resolver.exists():
        (staging / "etc" / "resolv.conf").write_text(resolver.read_text())
    _remove(home / "rootfs")
    staging.replace(home / "rootfs")


def _remove(path: Path) -> None:
    """Remove a tree, including read-only directories a package manager may have left."""

    def allow_and_retry(function: Callable[[str], object], target: str, error: BaseException) -> None:
        os.chmod(os.path.dirname(target), stat.S_IRWXU)
        if os.path.isdir(target):
            os.chmod(target, stat.S_IRWXU)
        function(target)

    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onexc=allow_and_retry)
