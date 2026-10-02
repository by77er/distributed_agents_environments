"""An environment backend on the host itself, for uses where a sandbox matters less than convenience.

Each environment is a workspace directory on the host. A command runs in it with the host's shell, programs, files,
user and network, and inherits the server's environment variables (plus `WORKSPACE`, the workspace's path):

- relative paths (in commands, `put` and `get`) are under the workspace; absolute paths are the host's own;
- nothing kept running between commands: each command runs in its own process group, which is killed when the
  command ends (a process that starts its own session escapes this);
- the only image is `host`: an environment cannot be Alpine or anything else the host is not, and asking for another
  image is an error rather than a silent downgrade.

There is no isolation at all: a command can read and change anything the server's user can. Workspaces are plain
directories, so they survive restarts of the runner.
"""

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path

from rollout.harness.environments import EnvironmentSpecification, ExecutionResult
from rollout_computers.processes import (
    IN_DIRECTORY,
    create_once,
    output_name,
    remove_tree,
    run_command,
    write_file,
)

HOST_IMAGE = "host"


class LocalEnvironments:
    """Implements `EnvironmentService`."""

    def __init__(self, directory: Path, *, variables: Mapping[str, str] | None = None) -> None:
        """`variables` replaces the inherited environment variables when given."""
        self.directory = directory
        self.variables = dict(os.environ if variables is None else variables)
        self._creating: dict[str, asyncio.Lock] = {}

    def workspace(self, environment_id: str) -> Path:
        return self.directory / environment_id / "workspace"

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        if specification.image != HOST_IMAGE:
            raise ValueError(f"the local backend only provides the image {HOST_IMAGE!r}, not {specification.image!r}")
        home = self.directory / environment_id

        async def prepare() -> None:
            await asyncio.to_thread(remove_tree, home)  # a creation interrupted earlier starts again from nothing
            await asyncio.to_thread(self.workspace(environment_id).mkdir, parents=True)

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
        workspace = self.workspace(environment_id)
        if not workspace.exists():
            raise FileNotFoundError(f"environment {environment_id} does not exist")

        def save(full: bytes) -> str:
            target = self.directory / environment_id / "outputs" / output_name(effect_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(full)
            return str(target)

        return await run_command(
            ["/bin/sh", "-c", IN_DIRECTORY, "sh", str(workspace / (cwd or ".")), command],
            timeout_seconds=timeout_seconds,
            save=save,
            cwd=workspace,
            variables={**self.variables, "WORKSPACE": str(workspace)},
        )

    async def put(self, environment_id: str, path: str, data: bytes) -> None:
        await write_file(self._resolve(environment_id, path), data)

    async def get(self, environment_id: str, path: str) -> bytes:
        return await asyncio.to_thread(self._resolve(environment_id, path).read_bytes)

    async def destroy(self, environment_id: str) -> None:
        await asyncio.to_thread(remove_tree, self.directory / environment_id)

    def _resolve(self, environment_id: str, path: str) -> Path:
        workspace = self.workspace(environment_id)
        if not workspace.exists():
            raise FileNotFoundError(f"environment {environment_id} does not exist")
        return workspace / os.path.expanduser(path)  # an absolute path replaces the workspace
