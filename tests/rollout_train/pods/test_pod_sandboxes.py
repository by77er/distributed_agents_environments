"""A pod's sandbox host (`rollout_train.pods.sandboxes`): it serves the kinds its lease's settings give sources for,
each kind's pool in a process and a Python environment of its own; sizes them from the pod's spare vCPUs and memory (the
lesser of its lease's and its container's), sharing them among kinds; makes each environment once per digest and finds
it again; gives the processes a made environment, nothing of its own; keeps the other kinds served when one does not
import; admits only the run that holds the pod; and, when the run's driver is gone, another run takes the pod, or the
host is started again, a lease's key gets `SandboxLost`, never a fresh sandbox."""

import asyncio
import io
import os
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout.harness.remote import RemotePool
from rollout.harness.sandboxes import LeaseRefused, PoolUnavailable, SandboxLost, SandboxSpec
from rollout.testing import until
from rollout_train.ledger import FileLedger
from rollout_train.pods import sandboxes
from rollout_train.pods.leases import HELD, PodLease, PodLeases, pod_leases_of
from rollout_train.pods.sandboxes import Machine, SandboxHost, Venvs, machine_of, sized, worker_environment
from rollout_train.pods.sources import Project, SandboxSource
from rollout_train.publishing import packed
from tests.rollout_train.pods.authority import served_tls

ROOT = Path(__file__).resolve().parents[3]
BOX = SandboxSpec(kind="fake")
POD = "rollout-test-host-0"
WORLD = SandboxSource("minecraft", "minecraft_team.worlds:worlds")


def test_a_pool_holds_a_sandbox_per_spare_vcpu_bounded_by_its_spare_memory_and_kinds_share_them() -> None:
    gb = 1e9 / 2**30
    assert sized(Machine(16, 188 * gb), {"minecraft": WORLD}) == {"minecraft": 12}  # (an RTX PRO 6000 pod)
    assert sized(Machine(8, 125 * gb), {"minecraft": WORLD}) == {"minecraft": 4}  # (the cheapest H100 SXM pod)
    assert sized(Machine(32, 80), {"minecraft": WORLD}) == {"minecraft": 6}  # (16 GiB spare, 2.4 a world)
    assert sized(Machine(16, 188), {"minecraft": replace(WORLD, size=5)}) == {"minecraft": 5}
    assert sized(Machine(3, 188), {"minecraft": WORLD}) == {"minecraft": 0}  # (nothing spare)
    halves = {"a": replace(WORLD, share=0.5), "b": replace(WORLD, share=0.5, cpus=2)}
    assert sized(Machine(16, 188), halves) == {"a": 6, "b": 3}
    sized_each = {"a": replace(WORLD, size=4), "b": replace(WORLD, size=20)}
    assert sized(Machine(16, 188), sized_each) == {"a": 4, "b": 8}  # (b takes what a left)


def test_a_pods_vcpus_and_memory_are_the_lesser_of_its_leases_and_its_containers(tmp_path: Path) -> None:
    lease = PodLease(POD, "host", 0, "host", "image", "m", "gpu", 1.0, vcpus=16, memory_gb=188.0)
    assert machine_of(lease, tmp_path) == Machine(16, 188e9 / 2**30)  # (no limits)
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    (tmp_path / "memory.max").write_text(f"{125 * 2**30}\n")
    assert machine_of(lease, tmp_path) == Machine(8, 125)
    assert machine_of(replace(lease, vcpus=4, memory_gb=None), tmp_path) == Machine(4, 125)
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "memory").mkdir()
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("400000")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(64 * 2**30))
    assert machine_of(None, v1) == Machine(4, 64)
    unlimited = machine_of(None, tmp_path / "none")  # (no lease, no limits: the machine's own)
    assert unlimited.cpus >= 1 and unlimited.memory_gib > 0


def test_a_kinds_process_is_given_a_made_environment_and_nothing_of_the_pods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROLLOUT_LEDGER_TOKEN", "rlp1.secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    environment = worker_environment(tmp_path)
    assert set(environment) == {"PATH", "HOME", "TMPDIR", "LANG", "PYTHONUNBUFFERED"}
    assert environment["HOME"] == str(tmp_path / "home") and (tmp_path / "home").is_dir()


async def test_each_sources_environment_is_made_once_by_its_digest_and_found_again(tmp_path: Path) -> None:
    project = tmp_path / "source" / "boxes"
    (project / "boxes").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "boxes"\nversion = "0"\n')
    (project / "boxes" / "__init__.py").write_text("")
    blobs = FileBlobStore(tmp_path / "blobs")
    blob = (await blobs.put(packed(project), "application/zip")).model_dump(mode="json")
    source = SandboxSource("fake", "boxes:boxes", projects=(Project("boxes", blob, ("http",)),),
                           constraints=("httpx==0.28.1",))  # fmt: skip
    ran: list[tuple[list[str], dict[str, str]]] = []

    def run(command: Any, environment: Any) -> None:
        ran.append((list(command), dict(environment)))
        if command[:2] == ["uv", "venv"]:
            (Path(command[-1]) / "bin").mkdir(parents=True)
            (Path(command[-1]) / "bin" / "python").write_text("")

    venvs = Venvs(tmp_path / "sandboxes", blobs, run=run)
    python = await venvs.resolved(source)
    assert python == tmp_path / "sandboxes" / "venvs" / source.digest / "bin" / "python"
    (made, made_with), (installed, _) = ran
    assert made[:6] == ["uv", "venv", "--python", "3.13", "--python-preference", "only-managed"]
    unpacked = tmp_path / "sandboxes" / "projects" / str(blob["sha256"])
    assert installed[installed.index("--editable") + 1] == f"{unpacked}[http]"
    assert (unpacked / "boxes" / "__init__.py").is_file()
    assert (python.parent.parent / "constraints.txt").read_text() == "httpx==0.28.1\n"
    assert made_with["UV_PYTHON_INSTALL_DIR"].startswith(str(tmp_path / "sandboxes")) and "AWS_SECRET_ACCESS_KEY" \
        not in made_with  # fmt: skip
    again = replace(source, kind="other", settings={"heap": "1G"}, size=3)  # (what it runs, not how: the same digest)
    assert await venvs.resolved(again) == python and len(ran) == 2
    pinned = replace(source, constraints=("httpx==0.28.0",))
    assert await venvs.resolved(pinned) != python and len(ran) == 4


def zipped(**files: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


async def test_a_zip_that_names_a_path_outside_its_project_is_refused(tmp_path: Path) -> None:
    blobs = FileBlobStore(tmp_path / "blobs")
    blob = (await blobs.put(zipped(**{"../escaped.txt": "x"}), "application/zip")).model_dump(mode="json")
    source = SandboxSource("fake", "boxes:boxes", projects=(Project("boxes", blob),))
    with pytest.raises(RuntimeError, match="outside it"):
        await Venvs(tmp_path / "sandboxes", blobs, run=lambda command, environment: None).resolved(source)
    assert not (tmp_path / "sandboxes" / "escaped.txt").exists()


class Here(Venvs):
    """The tests' own Python, for every source."""

    async def resolved(self, source: SandboxSource) -> Path:
        return Path(sys.executable)


async def held_by(leases: PodLeases, run: str | None, *, renewed: float | None = None, **settings: Any) -> None:
    there = await leases.get(POD)
    assert there is not None
    changes: dict[str, Any] = {"run": run, "renewed": time.time() if renewed is None else renewed}
    if settings:
        changes["settings"] = settings
    await leases.put(replace(there, **changes), expect=there.version)


async def test_a_pod_serves_each_kind_its_lease_asks_for_in_a_process_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandboxes, "FIRST_WAIT", 0.05)
    monkeypatch.setattr(sandboxes, "GIVE_UP", 2)
    monkeypatch.setenv("ROLLOUT_LEDGER_TOKEN", "rlp1.secret")  # (the pod's: never given to a kind's process)
    leases = pod_leases_of(FileLedger(tmp_path / "ledger"))
    assert leases is not None
    fake = SandboxSource("fake", "tests.rollout_train.pods.sandbox_kinds:boxes", size=2)
    broken = SandboxSource("broken", "tests.rollout_train.pods.sandbox_kinds:broken", size=2)
    sources: dict[str, JsonValue] = {"fake": fake.to_json(), "broken": broken.to_json()}
    await leases.put(PodLease(POD, "host", 0, "host", "image", "m", "gpu", 1.0, run="run_a", state=HELD,
                              renewed=time.time(), vcpus=8, memory_gb=125.0, settings={"sandboxes": sources}),
                     expect=None)  # fmt: skip
    root = tmp_path / "sandboxes"
    passed = {"PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}

    def host_of() -> SandboxHost:
        return SandboxHost(POD, leases, Here(root, None), root, cgroup=tmp_path / "no-cgroup", passed=passed)

    host = host_of()

    async def settled() -> bool:
        states = {kind: worker.state for kind, worker in host.workers.items()}
        return states == {"fake": "running", "broken": "failed"}

    try:
        await host.followed()
        await until(settled, 60, every=0.1)  # (the broken kind given up on; the other served)
        async with served_tls(host.app()) as url:
            pool = RemotePool(f"{url}/fake")
            assert (await pool.capacity()).size == 2
            lease = await pool.acquire(BOX, "run_a/1/1/1/box")
            said = cast(dict[str, Any], (await pool.call(lease.key, "environ", {}, effect_id="e",
                                                         arguments_digest="d")).structured)  # fmt: skip
            assert "ROLLOUT_LEDGER_TOKEN" not in said["names"] and said["home"] == str(root / "home")
            with pytest.raises(LeaseRefused):
                await pool.acquire(BOX, "run_b/1/1/1/box")
            with pytest.raises(PoolUnavailable):
                await RemotePool(f"{url}/broken").acquire(SandboxSpec(kind="broken"), "run_a/1/1/1/world")

            await held_by(leases, "run_a", renewed=time.time() - 3600)  # (the run's driver is gone)
            await host.followed()
            with pytest.raises(SandboxLost):
                await pool.call(lease.key, "environ", {}, effect_id="e", arguments_digest="d")
            await held_by(leases, "run_a")  # (it is back: the lease's key is still lost, never a fresh sandbox)
            await host.followed()
            with pytest.raises(SandboxLost):
                await pool.acquire(BOX, lease.key)
            await pool.acquire(BOX, "run_a/1/2/1/box")

            await held_by(leases, "run_b")  # (another run takes the pod: read again for its first acquire)
            taken = await pool.acquire(BOX, "run_b/1/1/1/box")
            with pytest.raises(LeaseRefused):
                await pool.acquire(BOX, "run_a/1/3/1/box")
            await host.stop()

            host = host_of()  # (the host started again: what its kinds' processes held is lost)
            await host.followed()
            await until(settled, 60, every=0.1)
        async with served_tls(host.app()) as url:
            with pytest.raises(SandboxLost):
                await RemotePool(f"{url}/fake").acquire(BOX, taken.key)
            await held_by(leases, "run_b", sandboxes={})  # (the run asks for none: none is served)
            await host.followed()
            assert host.workers == {}
    finally:
        await host.stop()
    await asyncio.sleep(0)


async def test_a_pool_is_sized_again_when_its_pods_vcpus_are_known_once_it_holds_nothing(tmp_path: Path) -> None:
    leases = pod_leases_of(FileLedger(tmp_path / "ledger"))
    assert leases is not None
    fake = SandboxSource("fake", "tests.rollout_train.pods.sandbox_kinds:boxes", memory_gib=0.001)
    sources: dict[str, JsonValue] = {"fake": fake.to_json()}
    (tmp_path / "cgroup").mkdir()
    (tmp_path / "cgroup" / "cpu.max").write_text("1600000 100000\n")  # (the container's limit: 16 vCPUs)
    await leases.put(PodLease(POD, "host", 0, "host", "image", "m", "gpu", 1.0, run="run_a", state=HELD,
                              renewed=time.time(), settings={"sandboxes": sources}), expect=None)  # fmt: skip
    root = tmp_path / "sandboxes"
    passed = {"PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    host = SandboxHost(POD, leases, Here(root, None), root, cgroup=tmp_path / "cgroup", reserved_gib=0, passed=passed)

    async def running(size: int) -> bool:
        worker = host.workers.get("fake")
        return worker is not None and worker.state == "running" and worker.size == size

    try:
        await host.followed()
        await until(lambda: running(12), 60, every=0.1)  # (16 vCPUs less 4: the lease says none yet)
        there = await leases.get(POD)
        assert there is not None
        await leases.put(replace(there, vcpus=8), expect=there.version)  # (RunPod says it gave the pod 8)
        await host.followed()
        await until(lambda: running(4), 60, every=0.1)  # (the lesser of the lease's and the container's)
    finally:
        await host.stop()
