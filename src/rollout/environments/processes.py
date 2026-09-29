"""What environment backends share: shaping a command's output into a result, and removing directories."""

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from rollout.core.harness.environments import ExecutionResult

MAX_OUTPUT_CHARACTERS = 60_000


def execution_result(exit_code: int | None, output: bytes, timed_out: bool = False) -> ExecutionResult:
    text = output.decode("utf-8", errors="replace")
    truncated = len(text) > MAX_OUTPUT_CHARACTERS
    if truncated:
        half = MAX_OUTPUT_CHARACTERS // 2
        text = f"{text[:half]}\n… ({len(text) - MAX_OUTPUT_CHARACTERS} characters omitted) …\n{text[-half:]}"
    return ExecutionResult(exit_code=exit_code, output=text, truncated=truncated, timed_out=timed_out)


def remove_tree(path: Path) -> None:
    """Remove a tree, including read-only directories a package manager may have left."""

    def allow_and_retry(function: Callable[[str], object], target: str, error: BaseException) -> None:
        os.chmod(os.path.dirname(target), stat.S_IRWXU)
        if os.path.isdir(target):
            os.chmod(target, stat.S_IRWXU)
        function(target)

    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onexc=allow_and_retry)
