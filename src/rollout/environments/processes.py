"""What environment backends share: shaping a command's output into a result, and removing directories."""

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

from rollout.core.harness.environments import ExecutionResult

MAX_OUTPUT_LINES = 2000
MAX_OUTPUT_BYTES = 50 * 1024


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


def remove_tree(path: Path) -> None:
    """Remove a tree, including read-only directories a package manager may have left."""

    def allow_and_retry(function: Callable[[str], object], target: str, error: BaseException) -> None:
        os.chmod(os.path.dirname(target), stat.S_IRWXU)
        if os.path.isdir(target):
            os.chmod(target, stat.S_IRWXU)
        function(target)

    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onexc=allow_and_retry)
