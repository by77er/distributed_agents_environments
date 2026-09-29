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
import contextlib
import os
import signal
import tempfile
from collections.abc import Mapping
from pathlib import Path

from rollout.core.harness.environments import EnvironmentSpecification, ExecutionResult
from rollout.environments.processes import execution_result, remove_tree

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
        lock = self._creating.setdefault(environment_id, asyncio.Lock())
        async with lock:
            home = self.directory / environment_id
            if (home / "ready").exists():
                return  # already created: creation is idempotent
            await asyncio.to_thread(remove_tree, home)  # a creation interrupted earlier starts again from nothing
            await asyncio.to_thread(self.workspace(environment_id).mkdir, parents=True)
            for command in specification.setup:
                result = await self.execute(environment_id, command, timeout_seconds=600, cwd=None)
                if result.exit_code != 0:
                    raise RuntimeError(f"setup command failed ({result.exit_code}): {command}\n{result.output[-2000:]}")
            await asyncio.to_thread((home / "ready").write_text, specification.model_dump_json())

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        workspace = self.workspace(environment_id)
        if not workspace.exists():
            raise FileNotFoundError(f"environment {environment_id} does not exist")
        with tempfile.TemporaryFile() as output:  # a file, not a pipe: background processes cannot hold it open
            process = await asyncio.create_subprocess_exec(
                "/bin/sh", "-c", 'cd "$1" || exit 125; eval "$2"', "sh", str(workspace / (cwd or ".")), command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workspace,
                env={**self.variables, "WORKSPACE": str(workspace)},
                start_new_session=True,
            )  # fmt: skip
            timed_out = False
            try:
                async with asyncio.timeout(timeout_seconds):
                    await process.wait()
            except TimeoutError:
                timed_out = True
            finally:
                _kill_group(process.pid)  # what the command left running in the background ends with it
                await process.wait()
            output.seek(0)
            data = await asyncio.to_thread(output.read)
        if timed_out:
            return execution_result(None, data, timed_out=True)
        return execution_result(process.returncode, data)

    async def put(self, environment_id: str, path: str, data: bytes) -> None:
        target = self._resolve(environment_id, path)

        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(write)

    async def get(self, environment_id: str, path: str) -> bytes:
        return await asyncio.to_thread(self._resolve(environment_id, path).read_bytes)

    async def destroy(self, environment_id: str) -> None:
        await asyncio.to_thread(remove_tree, self.directory / environment_id)

    def _resolve(self, environment_id: str, path: str) -> Path:
        workspace = self.workspace(environment_id)
        if not workspace.exists():
            raise FileNotFoundError(f"environment {environment_id} does not exist")
        return workspace / os.path.expanduser(path)  # an absolute path replaces the workspace


def _kill_group(process_group: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGKILL)
