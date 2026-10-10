"""Providers a pod's sandbox host serves in the tests, each in a process of its own: `boxes`, whose sandboxes say the
environment and working directory their process was given (`environ`); `broken`, whose code does not import; and
`flaky`, which does not import while the file `marker` names is there."""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from rollout.testing import FakeSandbox, FakeSandboxes


def _environ(sandbox: FakeSandbox, arguments: Mapping[str, JsonValue]) -> JsonValue:
    names: list[JsonValue] = [*sorted(os.environ)]
    return {"names": names, "home": os.environ.get("HOME", ""), "cwd": os.getcwd(), "uid": os.getuid()}


def boxes(directory: Path, size: int = 4) -> FakeSandboxes:
    return FakeSandboxes(size=size, operations={"environ": _environ})


def broken(directory: Path, size: int = 4) -> FakeSandboxes:
    raise ImportError("No module named 'missing': the kind's code does not import")


def flaky(directory: Path, size: int = 4, marker: str = "") -> FakeSandboxes:
    if marker and Path(marker).exists():
        raise ImportError(f"not while {marker} is there")
    return boxes(directory, size)
