"""Child processes: ones that end with the process that started them, and ending ones a killed process left."""

import contextlib
import ctypes
import json
import os
import signal
from collections.abc import Iterable
from pathlib import Path


def end_with_parent() -> None:
    """In a child process (as it starts, or as a `preexec_fn`): have the kernel end it when its parent dies (Linux).
    A driver that is killed would otherwise leave its trainer holding the GPU, or its servers their memory."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def children(named: str, parent: int | None = None) -> list[int]:
    """The processes started by `parent` (this process, by default) whose names begin with `named`."""
    parent = os.getpid() if parent is None else parent
    found: list[int] = []
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":\t", 1) for line in status.read_text().splitlines() if ":\t" in line)
        except OSError:
            continue  # it ended meanwhile
        if fields.get("Name", "").startswith(named) and int(fields.get("PPid", "0")) == parent:
            found.append(int(status.parent.name))
    return sorted(found)


def note_processes(record: Path, processes: Iterable[int]) -> None:
    """Write down processes this one started, with their names, so that a later process can end them if this one
    dies without doing so (a killed process cannot shut down what it started, and an engine left behind holds its
    accelerator)."""
    names: dict[str, str] = {}
    for pid in sorted(set(processes)):
        with contextlib.suppress(OSError):
            names[str(pid)] = Path(f"/proc/{pid}/comm").read_text().strip()
    record.write_text(json.dumps({"owner": os.getpid(), "processes": names}))


def end_orphans(record: Path) -> list[int]:
    """End the processes an earlier process noted in `record`, if that process is gone and they are not; returns
    the ones ended."""
    try:
        noted = json.loads(record.read_text())
        owner, processes = int(noted["owner"]), {int(pid): str(name) for pid, name in noted["processes"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []
    if owner == os.getpid() or Path(f"/proc/{owner}").exists():
        return []
    ended: list[int] = []
    for pid, name in processes.items():
        try:
            now = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            continue
        if now == name:  # (a process id may have been given to something else since)
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
                ended.append(pid)
    return ended
