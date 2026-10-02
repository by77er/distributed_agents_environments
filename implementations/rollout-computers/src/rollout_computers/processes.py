"""What environment backends share: creating an environment once, running a command and shaping its output into a
result, and writing and removing files."""

import asyncio
import contextlib
import hashlib
import os
import shutil
import signal
import stat
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from rollout.harness.environments import EnvironmentSpecification, ExecutionResult

MAX_OUTPUT_LINES = 2000
MAX_OUTPUT_BYTES = 50 * 1024

IN_DIRECTORY = 'cd "$1" || exit 125; eval "$2"'
"""A shell script that runs the command `$2` in the directory `$1`, or exits with 125 if it cannot enter it."""


async def create_once(
    home: Path,
    specification: EnvironmentSpecification,
    prepare: Callable[[], Awaitable[None]],
    execute: Callable[[str], Awaitable[ExecutionResult]],
) -> None:
    """Create the environment kept in `home`, unless it is ready: `prepare` its files, run the specification's setup
    commands with `execute`, and mark it ready. The caller holds the environment's creation lock."""
    if (home / "ready").exists():
        return  # already created: creation is idempotent
    await prepare()
    for command in specification.setup:
        result = await execute(command)
        if result.exit_code != 0:
            raise RuntimeError(f"setup command failed ({result.exit_code}): {command}\n{result.output[-2000:]}")
    await asyncio.to_thread((home / "ready").write_text, specification.model_dump_json())


async def run_command(
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
    save: Callable[[bytes], str],
    cwd: Path | None = None,
    variables: Mapping[str, str] | None = None,
) -> ExecutionResult:
    """Run a program in its own process group and return its result (`save` as in `execution_result`). The group is
    killed when the program ends, when it times out and when the caller is cancelled: nothing it started keeps
    running, whichever way the command ends."""
    with tempfile.TemporaryFile() as output:  # a file, not a pipe: background processes cannot hold it open
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=variables,
            start_new_session=True,
        )
        timed_out = False
        try:
            async with asyncio.timeout(timeout_seconds):
                await process.wait()
        except TimeoutError:
            timed_out = True
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        output.seek(0)
        data = await asyncio.to_thread(output.read)
    return execution_result(None if timed_out else process.returncode, data, timed_out=timed_out, save=save)


def execution_result(
    exit_code: int | None, output: bytes, *, timed_out: bool = False, save: Callable[[bytes], str] | None = None
) -> ExecutionResult:
    """The result of a command. Long output keeps its end (the last `MAX_OUTPUT_LINES` lines or `MAX_OUTPUT_BYTES`
    bytes, whichever is less), which is where errors and summaries are; `save` stores the full output and returns
    its path inside the environment."""
    tail = _tail(output)
    if tail is output:
        return ExecutionResult(exit_code=exit_code, output=_decode(output), timed_out=timed_out)
    return ExecutionResult(
        exit_code=exit_code,
        output=_decode(tail),
        truncated=True,
        timed_out=timed_out,
        full_output_path=save(output) if save is not None else None,
    )


def output_name(effect_id: str) -> str:
    """A file name for a command's full output: the same on every attempt of the effect, unique otherwise."""
    return f"output-{hashlib.sha256((effect_id or uuid.uuid4().hex).encode()).hexdigest()[:16]}.log"


def _tail(output: bytes) -> bytes:
    """The end of the output, in whole lines where possible; `output` itself when it is within the limits."""
    if len(output) <= MAX_OUTPUT_BYTES and output.count(b"\n", 0, max(len(output) - 1, 0)) < MAX_OUTPUT_LINES:
        return output
    kept: list[bytes] = []
    size = 0
    for line in reversed(output.splitlines(keepends=True)):
        if len(kept) == MAX_OUTPUT_LINES or size + len(line) > MAX_OUTPUT_BYTES:
            break
        kept.append(line)
        size += len(line)
    if not kept:  # the last line alone is too long: keep its end
        return output[-MAX_OUTPUT_BYTES:]
    return b"".join(reversed(kept))


def _decode(output: bytes) -> str:
    return output.decode("utf-8", errors="replace")


async def write_file(target: Path, data: bytes) -> None:
    """Write a file, creating the directories above it."""

    def write() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    await asyncio.to_thread(write)


def remove_tree(path: Path) -> None:
    """Remove a tree, including read-only directories a package manager may have left."""

    def allow_and_retry(function: Callable[[str], object], target: str, error: BaseException) -> None:
        os.chmod(os.path.dirname(target), stat.S_IRWXU)
        if os.path.isdir(target):
            os.chmod(target, stat.S_IRWXU)
        function(target)

    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onexc=allow_and_retry)
