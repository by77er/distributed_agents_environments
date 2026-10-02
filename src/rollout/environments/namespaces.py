"""An environment backend on unprivileged Linux namespaces (docs/environments/README.md).

Each environment is a directory holding its own copy of a base image's root filesystem. A command runs in new user,
mount and PID namespaces, chrooted into that root filesystem, as root inside (mapped to the host user outside):

- its own filesystem (the host's files are not visible), process tree and working directory (`/workspace`);
- the host's network, so the internet works without root;
- nothing kept running between commands: each command gets a fresh process tree.

This isolates processes and files; it is not a security boundary against hostile code. Environments are plain
directories, so they survive restarts of the runner.
"""

import asyncio
import subprocess
from pathlib import Path

from rollout.core.harness.environments import EnvironmentSpecification, ExecutionResult
from rollout.environments.images import ImageStore
from rollout.environments.processes import (
    IN_DIRECTORY,
    create_once,
    output_name,
    remove_tree,
    run_command,
    write_file,
)

# Runs inside the new namespaces, before entering the environment: mounts, then chroot with a clean environment.
ENTER = rf"""
root="$1"; cwd="$2"; command="$3"
mount --rbind /dev "$root/dev" 2>/dev/null
mount -t proc proc "$root/proc"
mount -t tmpfs tmpfs "$root/tmp"
exec chroot "$root" /usr/bin/env -i HOME=/root TERM=dumb LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    /bin/sh -c '{IN_DIRECTORY}' sh "$cwd" "$command"
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
        home = self.directory / environment_id

        async def prepare() -> None:
            tarball = await self.images.tarball(specification.image)
            await asyncio.to_thread(_unpack, tarball, home)

        async with self._creating.setdefault(environment_id, asyncio.Lock()):
            await create_once(
                home,
                specification,
                prepare,
                lambda command: self.execute(environment_id, command, timeout_seconds=600, cwd=None),
            )

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        root = self.root(environment_id)
        if not root.exists():
            raise FileNotFoundError(f"environment {environment_id} does not exist")

        def save(full: bytes) -> str:
            inside = f"/var/tmp/{output_name(effect_id)}"
            target = root / inside.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(full)
            return inside

        return await run_command(
            ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child",
             "sh", "-c", ENTER, "enter", str(root), cwd or "/workspace", command],
            timeout_seconds=timeout_seconds,
            save=save,
        )  # fmt: skip

    async def put(self, environment_id: str, path: str, data: bytes) -> None:
        await write_file(self._inside(environment_id, path), data)

    async def get(self, environment_id: str, path: str) -> bytes:
        return await asyncio.to_thread(self._inside(environment_id, path).read_bytes)

    async def destroy(self, environment_id: str) -> None:
        await asyncio.to_thread(remove_tree, self.directory / environment_id)

    def _inside(self, environment_id: str, path: str) -> Path:
        root = self.root(environment_id).resolve()
        relative = path if path.startswith("/") else f"workspace/{path}"
        resolved = (root / relative.lstrip("/")).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"{path} is outside the environment")
        return resolved


def _unpack(tarball: Path, home: Path) -> None:
    """Unpack the image into a fresh root filesystem, with a workspace and the host's DNS settings."""
    staging = home / "staging"
    remove_tree(staging)
    staging.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(staging)], check=True, capture_output=True)
    (staging / "workspace").mkdir(exist_ok=True)
    resolver = Path("/etc/resolv.conf")
    if resolver.exists():
        (staging / "etc" / "resolv.conf").write_text(resolver.read_text())
    remove_tree(home / "rootfs")
    staging.replace(home / "rootfs")
