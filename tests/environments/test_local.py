import asyncio
import os
import time
from pathlib import Path

import pytest

from rollout.core.harness import EnvironmentSpecification
from rollout.environments import LocalEnvironments

HOST = EnvironmentSpecification(image="host")


@pytest.fixture
def environments(tmp_path: Path) -> LocalEnvironments:
    return LocalEnvironments(tmp_path / "environments", variables={**os.environ, "GREETING": "hello"})


async def test_a_command_runs_on_the_host_in_its_workspace(environments: LocalEnvironments) -> None:
    await environments.create("e_one", HOST.model_copy(update={"setup": ("echo seeded > seed.txt",)}))
    workspace = environments.workspace("e_one")
    result = await environments.execute(
        "e_one", 'pwd; echo "$WORKSPACE"; cat seed.txt; echo "$GREETING"; id -u', timeout_seconds=30, cwd=None
    )
    assert result.exit_code == 0
    assert result.output.splitlines() == [str(workspace), str(workspace), "seeded", "hello", str(os.getuid())]


async def test_paths_are_relative_to_the_workspace_and_absolute_paths_are_the_hosts(
    environments: LocalEnvironments, tmp_path: Path
) -> None:
    await environments.create("e_files", HOST)
    await environments.put("e_files", "data/input.txt", b"42\n")
    assert (environments.workspace("e_files") / "data" / "input.txt").read_bytes() == b"42\n"
    result = await environments.execute("e_files", "cat input.txt", timeout_seconds=30, cwd="data")
    assert result.output == "42\n"
    (tmp_path / "outside.txt").write_text("host file")
    assert await environments.get("e_files", str(tmp_path / "outside.txt")) == b"host file"  # no sandbox
    elsewhere = await environments.execute("e_files", "pwd", timeout_seconds=30, cwd=str(tmp_path))
    assert elsewhere.output == f"{tmp_path}\n"


async def test_nothing_keeps_running_after_a_command(environments: LocalEnvironments, tmp_path: Path) -> None:
    await environments.create("e_background", HOST)
    marker = tmp_path / "survived"
    started = time.monotonic()
    result = await environments.execute(
        "e_background", f"(sleep 1; touch {marker}) & echo started", timeout_seconds=30, cwd=None
    )
    assert result.output == "started\n"
    assert time.monotonic() - started < 1  # did not wait for the background process
    await asyncio.sleep(1.5)
    assert not marker.exists()  # it was killed with the command


async def test_commands_time_out_and_report_failures(environments: LocalEnvironments) -> None:
    await environments.create("e_slow", HOST)
    started = time.monotonic()
    slow = await environments.execute("e_slow", "echo before; sleep 30", timeout_seconds=0.5, cwd=None)
    assert slow.timed_out and slow.exit_code is None and slow.output == "before\n"
    assert time.monotonic() - started < 5
    failing = await environments.execute("e_slow", "echo oops >&2; exit 3", timeout_seconds=30, cwd=None)
    assert (failing.exit_code, failing.output) == (3, "oops\n")


async def test_creation_is_idempotent_and_destroy_removes_everything(
    environments: LocalEnvironments, tmp_path: Path
) -> None:
    await environments.create("e_twice", HOST)
    await environments.execute("e_twice", "echo kept > kept.txt", timeout_seconds=30, cwd=None)
    await environments.create("e_twice", HOST)  # already exists: nothing changes
    assert await environments.get("e_twice", "kept.txt") == b"kept\n"
    await environments.destroy("e_twice")
    await environments.destroy("e_twice")  # already gone: fine
    assert not (tmp_path / "environments" / "e_twice").exists()
    with pytest.raises(FileNotFoundError):
        await environments.execute("e_twice", "true", timeout_seconds=30, cwd=None)


async def test_only_the_host_image_is_provided(environments: LocalEnvironments) -> None:
    with pytest.raises(ValueError, match="only provides the image 'host'"):
        await environments.create("e_alpine", EnvironmentSpecification(image="alpine"))
