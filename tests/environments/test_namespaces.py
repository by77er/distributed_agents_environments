import shutil
import subprocess
from pathlib import Path

import pytest

from rollout.core.harness import EnvironmentSpecification
from rollout.environments import ImageStore, NamespaceEnvironments

IMAGE_CACHE = Path.home() / ".cache" / "rollout" / "images"


def namespaces_available() -> bool:
    if shutil.which("unshare") is None:
        return False
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "true"], check=False)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not namespaces_available(), reason="unprivileged namespaces are not available")


@pytest.fixture
async def environments(tmp_path: Path) -> NamespaceEnvironments:
    service = NamespaceEnvironments(tmp_path, ImageStore(IMAGE_CACHE))
    try:
        await service.images.tarball("alpine")
    except Exception as error:  # no network: skip rather than fail
        pytest.skip(f"cannot fetch the base image: {error}")
    return service


async def test_an_environment_is_its_own_computer(environments: NamespaceEnvironments) -> None:
    await environments.create("e_one", EnvironmentSpecification(setup=["echo seeded > /workspace/seed.txt"]))
    result = await environments.execute(
        "e_one", "cat /etc/alpine-release; id -u; echo $$; pwd; cat seed.txt; ls /home", timeout_seconds=30, cwd=None
    )
    release, uid, pid, cwd, seed = result.output.splitlines()[:5]
    assert result.exit_code == 0
    assert release.startswith("3.")
    assert (uid, pid, cwd, seed) == ("0", "1", "/workspace", "seeded")  # root inside, its own process tree
    assert "bit" not in result.output  # the host's home directories are not visible


async def test_files_persist_between_commands_and_through_put_and_get(environments: NamespaceEnvironments) -> None:
    await environments.create("e_files", EnvironmentSpecification())
    await environments.execute("e_files", "echo hello > note.txt", timeout_seconds=30, cwd=None)
    assert await environments.get("e_files", "note.txt") == b"hello\n"
    await environments.put("e_files", "data/input.txt", b"42\n")
    result = await environments.execute("e_files", "cat data/input.txt", timeout_seconds=30, cwd=None)
    assert result.output == "42\n"
    with pytest.raises(ValueError, match="outside the environment"):
        await environments.get("e_files", "../../../../etc/passwd")


async def test_creation_is_idempotent_and_destroy_removes_everything(
    environments: NamespaceEnvironments, tmp_path: Path
) -> None:
    await environments.create("e_twice", EnvironmentSpecification())
    await environments.execute("e_twice", "echo kept > kept.txt", timeout_seconds=30, cwd=None)
    await environments.create("e_twice", EnvironmentSpecification())  # already exists: nothing changes
    assert await environments.get("e_twice", "kept.txt") == b"kept\n"
    await environments.execute("e_twice", "mkdir -p locked && chmod 500 locked", timeout_seconds=30, cwd=None)
    await environments.destroy("e_twice")
    await environments.destroy("e_twice")  # already gone: fine
    assert not (tmp_path / "e_twice").exists()


async def test_commands_time_out_and_report_failures(environments: NamespaceEnvironments) -> None:
    await environments.create("e_slow", EnvironmentSpecification())
    slow = await environments.execute("e_slow", "sleep 30", timeout_seconds=0.5, cwd=None)
    assert slow.timed_out and slow.exit_code is None
    failing = await environments.execute("e_slow", "echo oops >&2; exit 3", timeout_seconds=30, cwd=None)
    assert (failing.exit_code, failing.output) == (3, "oops\n")


async def test_long_output_is_saved_inside_the_environment(environments: NamespaceEnvironments) -> None:
    await environments.create("e_long", EnvironmentSpecification())
    result = await environments.execute("e_long", "seq 1 5000", timeout_seconds=30, cwd=None, effect_id="e1")
    assert result.truncated and result.output.splitlines()[0] == "3001"
    assert result.full_output_path is not None and result.full_output_path.startswith("/var/tmp/output-")
    inside = await environments.execute("e_long", f"wc -l < {result.full_output_path}", timeout_seconds=30, cwd=None)
    assert inside.output.strip() == "5000"  # the environment's own shell can read it
