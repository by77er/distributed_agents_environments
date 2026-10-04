"""How a machine is doing: its memory, its accelerators, and the disk a directory is on.

A process that beats (a runner, an engine host, a gateway replica, a launcher) measures its machine in each heartbeat
(`rollout_train.presence`); the monitor shows those, so it says how the machines that run things are doing, wherever it
is itself.
"""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def measured(directory: Path | None = None) -> dict[str, Any]:
    """Memory (bytes available and in all), each accelerator (its name, memory used and in all, how busy), and the
    disk `directory` is on, now."""
    memory: dict[str, Any] = {"available": None, "total": None}
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory = {
            "available": int(fields["MemAvailable"].split()[0]) * 1024,
            "total": int(fields["MemTotal"].split()[0]) * 1024,
        }
    except (OSError, KeyError, ValueError):
        pass
    disk = shutil.disk_usage(directory) if directory is not None and directory.exists() else None
    return {
        "at": round(time.time(), 1),
        "memory": memory,
        "accelerators": accelerators(),
        "disk": {"free": disk.free, "total": disk.total} if disk else None,
    }


def accelerators() -> list[dict[str, Any]]:
    """Each GPU `nvidia-smi` reports: its name, memory used and in all (bytes), and how busy (0 to 1)."""
    query = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        listed = subprocess.run(query, capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found: list[dict[str, Any]] = []
    for line in listed.strip().splitlines():
        name, used, total, busy = (part.strip() for part in line.split(","))
        try:
            found.append(
                {"name": name, "used": int(used) * 2**20, "total": int(total) * 2**20, "busy": int(busy) / 100}
            )
        except ValueError:  # (a field the driver does not report)
            continue
    return found


def alive(pid: int) -> bool:
    """Whether a process is there, on this machine."""
    return Path(f"/proc/{pid}").exists()
