"""Providers a pod's sandbox host serves in the tests, each in a process of its own: `boxes`, whose sandboxes say the
environment their process was given (`environ`), and `broken`, whose code does not import."""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from rollout.testing import FakeSandbox, FakeSandboxes


def _environ(sandbox: FakeSandbox, arguments: Mapping[str, JsonValue]) -> JsonValue:
    names: list[JsonValue] = [*sorted(os.environ)]
    return {"names": names, "home": os.environ.get("HOME", "")}


def boxes(directory: Path, size: int = 4) -> FakeSandboxes:
    return FakeSandboxes(size=size, operations={"environ": _environ})


def broken(directory: Path, size: int = 4) -> FakeSandboxes:
    raise ImportError("No module named 'missing': the kind's code does not import")
