# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray is partly untyped.)
"""The test session's Ray: a fresh local instance of its own, never a cluster this machine runs already."""

import os
import shutil
import socket
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

RAY_TESTS = Path.home() / ".cache" / "rollout" / "ray-tests"
"""Where test sessions' Rays keep their files, each in a directory Ray names `session_DATE_PID`: on disk (a machine's
/tmp may be memory). Ray's sockets are in that directory, and a socket's path may be 107 bytes at most, which leaves
no room for a directory of the process's own above it."""


def isolated() -> None:
    """Keep this process's Ray to a cluster of its own: `RAY_ADDRESS` unset (nothing joins a cluster by it), workers
    run in this environment (`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, as `ray_cluster.prepare` sets), and no token (the
    session's cluster asks for none, whatever `prepare` would say). Ray reads these once, when it is imported, so the
    session's conftest calls this before anything imports it."""
    os.environ.pop("RAY_ADDRESS", None)
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    os.environ["RAY_AUTH_MODE"] = "disabled"


@dataclass(frozen=True)
class LocalRay:
    """The test session's Ray: its address (the GCS), its dashboard's URL (where jobs are submitted), and its files
    (its session's directory)."""

    address: str
    dashboard: str
    directory: Path


@contextmanager
def started() -> Generator[LocalRay]:
    """Ray started in this process: 4 CPUs and one GPU (counted only: tests run scripted engines in what asks for it), a
    small object store, the dashboard on (jobs need it), its files in `RAY_TESTS/session_DATE_PID`, removed on the way
    out. `address="local"` starts a new instance, even where this
    machine runs a cluster already; its dashboard and the agent that runs its jobs listen on free ports, not on
    Ray's defaults, which a cluster already running holds (`ray.init` does not take the agent's port, so the
    parameters it starts the node with are given it)."""
    import ray
    from ray._private import parameter

    class Parameters(parameter.RayParams):
        def __init__(self, *given: Any, **named: Any) -> None:
            super().__init__(*given, **{"dashboard_agent_listen_port": free_port(), **named})

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(parameter, "RayParams", Parameters)
        context: Any = ray.init(
            address="local", num_cpus=4, num_gpus=1, object_store_memory=100 * 2**20, include_dashboard=True,
            dashboard_host="127.0.0.1", dashboard_port=free_port(), log_to_driver=False, _temp_dir=str(RAY_TESTS),
        )  # fmt: skip
    directory = Path(str(context.address_info["session_dir"]))
    try:
        yield LocalRay(str(context.address_info["gcs_address"]), f"http://{context.dashboard_url}", directory)
    finally:
        ray.shutdown()
        shutil.rmtree(directory, ignore_errors=True)


def submitted(client: Any, *, within: float = 60.0, **job: Any) -> str:
    """A job submitted to a `JobSubmissionClient`, asking again while the node's agent that runs jobs is starting."""
    deadline = time.monotonic() + within
    while True:
        try:
            return str(client.submit_job(**job))
        except RuntimeError as error:
            if "No available agent" not in str(error) or time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def free_port() -> int:
    """A TCP port on this machine that nothing listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
