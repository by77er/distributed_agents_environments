"""A pod's sandbox host (`rollout_train.pods.sandboxes`): it serves the kinds its lease's settings give sources for,
each kind's pool in a process, a Python environment and a user of its own, on a socket only the host reaches; sizes them
from the pod's spare vCPUs and memory (the lesser of its lease's and its container's), sharing them among kinds, and
resizes them in place; makes each environment once per digest with uv (for real, offline, where uv and its cache are
here), finds it again, and deletes what is long unused; gives the processes a made environment and nothing of its own;
keeps the other kinds served when one does not import, and tries a failed one again; waits out a Python it could not
make; takes no new sandbox while its lease goes unrenewed, and loses them once its run's driver is gone; ends what a
host before it left; and, when another run takes the pod or the host is started again, a lease's key gets
`SandboxLost`, never a fresh sandbox."""

import asyncio
import io
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout.harness.remote import RemotePool
from rollout.harness.sandboxes import LeaseRefused, PoolUnavailable, SandboxLost, SandboxSpec
from rollout.testing import until
from rollout_train.ledger import FileLedger
from rollout_train.pods import sandboxes, sources
from rollout_train.pods.leases import HELD, PodLease, PodLeases, pod_leases_of
from rollout_train.pods.sandboxes import (
    Account,
    Machine,
    SandboxHost,
    Venvs,
    machine_of,
    sized,
    worker_environment,
)
from rollout_train.pods.sources import Project, SandboxSource, local_projects, sources_in
from rollout_train.publishing import packed
from tests.rollout_train.pods.authority import served_tls

ROOT = Path(__file__).resolve().parents[3]
BOX = SandboxSpec(kind="fake")
POD = "rollout-test-host-0"
WORLD = SandboxSource("minecraft", "minecraft_team.worlds:worlds", memory_gib=2.4)
KINDS = "tests.rollout_train.pods.sandbox_kinds"
PASSED = {"PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}


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
    mixed = {"a": replace(WORLD, share=0.5), "z": replace(WORLD, size=4)}  # (the size first, then a share of the rest)
    assert sized(Machine(16, 188), mixed) == {"a": 4, "z": 4}


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
    environment = worker_environment(tmp_path / "home", tmp_path / "tmp")
    assert set(environment) == {"PATH", "HOME", "TMPDIR", "LANG", "PYTHONUNBUFFERED"}
    assert environment["HOME"] == str(tmp_path / "home")


def test_a_lease_source_that_is_not_well_formed_is_left_out() -> None:
    good = SandboxSource("fake", f"{KINDS}:boxes", projects=(Project("boxes", {"sha256": "a" * 64}),))
    bad_zip = replace(good, kind="other", projects=(Project("boxes", {"sha256": "../../etc"}),))
    bad_kind = replace(good, kind="../x")
    said: dict[str, JsonValue] = {"fake": good.to_json(), "other": bad_zip.to_json(), "../x": bad_kind.to_json()}
    assert sources_in({"sandboxes": said}) == {"fake": good}


async def stored(blobs: FileBlobStore, directory: Path) -> dict[str, JsonValue]:
    return (await blobs.put(packed(directory), "application/zip")).model_dump(mode="json")


async def test_each_sources_environment_is_made_once_by_its_digest_and_found_again(tmp_path: Path) -> None:
    project = tmp_path / "source" / "boxes"
    (project / "boxes").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "boxes"\nversion = "0"\n')
    (project / "boxes" / "__init__.py").write_text("")
    blobs = FileBlobStore(tmp_path / "blobs")
    blob = await stored(blobs, project)
    source = SandboxSource("fake", "boxes:boxes", projects=(Project("boxes", blob, ("http",)),),
                           pins=("httpx==0.28.1",))  # fmt: skip
    ran: list[tuple[list[str], dict[str, str]]] = []

    def run(command: Any, environment: Any) -> None:
        ran.append((list(command), dict(environment)))
        if command[1] == "venv":
            (Path(command[-1]) / "bin").mkdir(parents=True)
            (Path(command[-1]) / "bin" / "python").write_text("")

    venvs = Venvs(tmp_path / "sandboxes", blobs, run=run)
    python = await venvs.resolved(source)
    assert python == tmp_path / "sandboxes" / "venvs" / source.digest / "bin" / "python"
    (made, made_with), (installed, _) = ran
    assert made[:6] == ["uv", "venv", "--python", "3.13", "--python-preference", "only-managed"]
    assert "--no-sources" in installed and "--no-deps" in installed  # (nothing resolved, nothing fetched by name)
    unpacked = tmp_path / "sandboxes" / "projects" / str(blob["sha256"])
    assert installed[installed.index("--editable") + 1] == f"{unpacked}[http]"
    assert (unpacked / "boxes" / "__init__.py").is_file()
    assert (python.parent.parent / "pins.txt").read_text() == "httpx==0.28.1\n"
    assert made_with["UV_PYTHON_INSTALL_DIR"].startswith(str(tmp_path / "sandboxes")) and "AWS_SECRET_ACCESS_KEY" \
        not in made_with  # fmt: skip
    again = replace(source, kind="other", settings={"heap": "1G"}, size=3)  # (what it runs, not how: the same digest)
    assert await venvs.resolved(again) == python and len(ran) == 2
    pinned = replace(source, pins=("httpx==0.28.0",))
    elsewhere = await venvs.resolved(pinned)
    assert elsewhere != python and len(ran) == 4
    old = time.time() - 30 * 86400  # (unused for a month: deleted, with the zip no kept environment uses)
    os.utime(elsewhere.parent.parent / ".used", (old, old))
    os.utime(python.parent.parent / ".used", (old, old))
    os.utime(unpacked, (old, old))
    assert venvs.collect({source.digest}) == [f"venvs/{pinned.digest}"]  # (the one in use is kept, and its zip)
    assert venvs.collect(set()) == [f"venvs/{source.digest}", f"projects/{blob['sha256']}"]


async def test_a_zip_that_names_a_path_outside_its_project_is_refused(tmp_path: Path) -> None:
    blobs = FileBlobStore(tmp_path / "blobs")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escaped.txt", "x")
    blob = (await blobs.put(buffer.getvalue(), "application/zip")).model_dump(mode="json")
    source = SandboxSource("fake", "boxes:boxes", projects=(Project("boxes", blob),))
    with pytest.raises(RuntimeError, match="outside it"):
        await Venvs(tmp_path / "sandboxes", blobs, run=lambda command, environment: None).resolved(source)
    assert not (tmp_path / "sandboxes" / "escaped.txt").exists()


UV = shutil.which("uv", path=f"{Path.home() / '.local' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
UV_CACHE = Path.home() / ".cache" / "uv"
PYTHONS = Path.home() / ".local" / "share" / "uv" / "python"


@pytest.mark.skipif(UV is None or not UV_CACHE.is_dir() or not list(PYTHONS.glob("cpython-3.13*")),
                    reason="uv, its cache or a uv-managed Python 3.13 is not on this machine")  # fmt: skip
async def test_packed_projects_install_with_real_uv_offline_and_the_provider_finds_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert UV is not None
    held = sources._directory_of  # pyright: ignore[reportPrivateUsage]
    here = {"rollout": ROOT / "libraries" / "rollout", "minecraft-team": ROOT / "environments" / "minecraft"}

    def directory_of(name: str) -> Path | None:  # (this checkout's)
        return here.get(name) or held(name)

    monkeypatch.setattr(sources, "_directory_of", directory_of)
    blobs = FileBlobStore(tmp_path / "blobs")
    local, pins = local_projects("minecraft_team.worlds:worlds")
    assert [each.name for each in local] == ["minecraft-team", "rollout"]
    projects = tuple([Project(each.name, await stored(blobs, each.directory), each.extras) for each in local])
    source = SandboxSource("minecraft", "minecraft_team.worlds:worlds", projects=projects, pins=tuple(pins))
    offline = {"UV_OFFLINE": "1", "UV_CACHE_DIR": str(UV_CACHE), "UV_PYTHON_INSTALL_DIR": str(PYTHONS)}
    python = await Venvs(tmp_path / "sandboxes", blobs, passed=offline, uv=UV).resolved(source)
    found = await asyncio.to_thread(
        subprocess.run,
        [str(python), "-P", "-c", "import minecraft_team.worlds, minecraft_team.paper as paper, rollout.harness."
         "pool_server, uvicorn; print(paper.HARNESS / 'package-lock.json', paper.PLUGIN_SOURCES, paper.CONFIG)"],
        capture_output=True, text=True, check=False, env={"PATH": "/usr/bin:/bin"},
    )  # fmt: skip
    assert found.returncode == 0, found.stderr
    lock, plugin, config = (Path(each) for each in found.stdout.split())
    assert lock.is_file() and plugin.is_dir() and config.is_dir()  # (editable: beside the shipped package)
    assert lock.is_relative_to(tmp_path / "sandboxes" / "projects")


class Here(Venvs):
    """The tests' own Python, for every source."""

    async def resolved(self, source: SandboxSource) -> Path:
        return Path(sys.executable)


async def held_by(leases: PodLeases, run: str | None, *, renewed: float | None = None, **changes: Any) -> None:
    there = await leases.get(POD)
    assert there is not None
    await leases.put(replace(there, run=run, renewed=time.time() if renewed is None else renewed, **changes),
                     expect=there.version)  # fmt: skip


@pytest.fixture
async def pod(tmp_path: Path) -> AsyncIterator[tuple[PodLeases, Path, Path]]:
    """The pod's lease (held by `run_a`, 8 vCPUs), its sandbox host's directory, and a short directory for sockets."""
    leases = pod_leases_of(FileLedger(tmp_path / "ledger"))
    assert leases is not None
    await leases.put(PodLease(POD, "host", 0, "host", "image", "m", "gpu", 1.0, run="run_a", state=HELD,
                              renewed=time.time(), vcpus=8, memory_gb=125.0), expect=None)  # fmt: skip
    sockets = Path(tempfile.mkdtemp(prefix="s"))
    yield leases, tmp_path / "sandboxes", sockets
    shutil.rmtree(sockets, ignore_errors=True)


def host_of(leases: PodLeases, root: Path, sockets: Path, venvs: Venvs | None = None, **options: Any) -> SandboxHost:
    return SandboxHost(POD, leases, venvs or Here(root, None), root, cgroup=root.parent / "no-cgroup", passed=PASSED,
                       sockets=sockets, **options)  # fmt: skip


def states_are(host: SandboxHost, **wanted: str) -> Callable[[], Any]:
    async def settled() -> bool:
        return {kind: worker.state for kind, worker in host.workers.items()} == wanted

    return settled


async def test_a_pod_serves_each_kind_its_lease_asks_for_in_a_process_of_its_own(
    pod: tuple[PodLeases, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    leases, root, sockets = pod
    monkeypatch.setattr(sandboxes, "FIRST_WAIT", 0.05)
    monkeypatch.setattr(sandboxes, "GIVE_UP", 2)
    monkeypatch.setenv("ROLLOUT_LEDGER_TOKEN", "rlp1.secret")  # (the pod's: never given to a kind's process)
    fake = SandboxSource("fake", f"{KINDS}:boxes", size=2)
    broken = SandboxSource("broken", f"{KINDS}:broken", size=2)
    await held_by(leases, "run_a", settings={"sandboxes": {"fake": fake.to_json(), "broken": broken.to_json()}})
    host = host_of(leases, root, sockets)
    try:
        await host.followed()
        await until(states_are(host, fake="running", broken="failed"), 60, every=0.1)  # (one given up, one served)
        assert stat.S_IMODE((sockets / "fake.sock").stat().st_mode) == 0o600  # (only the host reaches it)
        async with served_tls(host.app()) as url:
            pool = RemotePool(f"{url}/fake")
            assert (await pool.capacity()).size == 2
            lease = await pool.acquire(BOX, "run_a/1/1/1/box")
            said = cast(dict[str, Any], (await pool.call(lease.key, "environ", {}, effect_id="e",
                                                         arguments_digest="d")).structured)  # fmt: skip
            assert "ROLLOUT_LEDGER_TOKEN" not in said["names"]
            assert said["home"] == said["cwd"] == str(root / "kinds" / "fake" / "home")  # (its own directory)
            with pytest.raises(LeaseRefused):
                await pool.acquire(BOX, "run_b/1/1/1/box")
            with pytest.raises(PoolUnavailable):
                await RemotePool(f"{url}/broken").acquire(SandboxSpec(kind="broken"), "run_a/1/1/1/world")

            await held_by(leases, "run_a", renewed=time.time() - 600)  # (not renewed lately: no new sandboxes)
            await host.followed()
            with pytest.raises(PoolUnavailable):
                await pool.acquire(BOX, "run_a/1/2/1/box")
            await pool.call(lease.key, "environ", {}, effect_id="e", arguments_digest="d")  # (its own: still served)
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

            host = host_of(leases, root, sockets)  # (the host started again: what its kinds' processes held is lost)
            await host.followed()
            await until(states_are(host, fake="running", broken="failed"), 60, every=0.1)
        async with served_tls(host.app()) as url:
            with pytest.raises(SandboxLost):
                await RemotePool(f"{url}/fake").acquire(BOX, taken.key)
            await held_by(leases, "run_b", settings={"sandboxes": {}})  # (the run asks for none: none is served)
            await host.followed()
            assert host.workers == {}
    finally:
        await host.stop()


async def test_a_kind_given_up_on_is_tried_again_and_a_python_not_made_is_waited_out(
    pod: tuple[PodLeases, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases, root, sockets = pod
    monkeypatch.setattr(sandboxes, "FIRST_WAIT", 0.05)
    monkeypatch.setattr(sandboxes, "GIVE_UP", 2)
    marker = tmp_path / "broken-now"
    marker.write_text("")
    flaky = SandboxSource("flaky", f"{KINDS}:flaky", settings={"marker": str(marker)}, size=1)
    later = SandboxSource("later", f"{KINDS}:boxes", size=1)
    await held_by(leases, "run_a", settings={"sandboxes": {"flaky": flaky.to_json(), "later": later.to_json()}})
    failures = {"later": 4}

    class Unreachable(Here):
        """A Python not made, four times (PyPI or GitHub unreachable), for `later`."""

        async def resolved(self, source: SandboxSource) -> Path:
            if failures.get(source.kind, 0):
                failures[source.kind] -= 1
                raise RuntimeError("uv venv failed: could not reach github.com")
            return await super().resolved(source)

    host = host_of(leases, root, sockets, Unreachable(root, None))
    try:
        await host.followed()
        await until(states_are(host, flaky="failed", later="running"), 60, every=0.1)  # (waited out, not given up)
        marker.unlink()  # (what broke it is mended)
        await host.followed()
        assert host.workers["flaky"].state == "failed"  # (not tried again at once)
        monkeypatch.setattr(sandboxes, "RETRY_FAILED", 0.0)
        await host.followed()
        await until(states_are(host, flaky="running", later="running"), 60, every=0.1)
    finally:
        await host.stop()


async def test_a_pool_is_resized_in_place_when_its_pods_vcpus_are_known(pod: tuple[PodLeases, Path, Path]) -> None:
    leases, root, sockets = pod
    fake = SandboxSource("fake", f"{KINDS}:boxes", memory_gib=0.001)
    await held_by(leases, "run_a", vcpus=None, settings={"sandboxes": {"fake": fake.to_json()}})
    (root.parent / "cgroup").mkdir()
    (root.parent / "cgroup" / "cpu.max").write_text("1600000 100000\n")  # (the container's limit: 16 vCPUs)
    host = SandboxHost(POD, leases, Here(root, None), root, cgroup=root.parent / "cgroup", reserved_gib=0,
                       passed=PASSED, sockets=sockets)  # fmt: skip

    async def running(size: int) -> bool:
        worker = host.workers.get("fake")
        return worker is not None and worker.state == "running" and worker.size == size

    try:
        await host.followed()
        await until(lambda: running(12), 60, every=0.1)  # (16 vCPUs less 4: the lease says none yet)
        process = host.workers["fake"].process
        assert process is not None
        async with served_tls(host.app()) as url:
            await RemotePool(f"{url}/fake").acquire(BOX, "run_a/1/1/1/box")
            await held_by(leases, "run_a", vcpus=8)  # (RunPod says it gave the pod 8)
            await host.followed()
            assert await running(4) and host.workers["fake"].process is process  # (the same process, told)
            assert (await RemotePool(f"{url}/fake").capacity()).size == 4
    finally:
        await host.stop()


async def test_each_kind_runs_as_a_user_of_its_own_and_none_is_left_for_a_kind_too_many(
    pod: tuple[PodLeases, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    leases, root, sockets = pod
    monkeypatch.setattr(sandboxes, "FIRST_WAIT", 0.05)
    one = SandboxSource("one", f"{KINDS}:boxes", size=1)
    two = SandboxSource("two", f"{KINDS}:boxes", size=1)
    await held_by(leases, "run_a", settings={"sandboxes": {"one": one.to_json(), "two": two.to_json()}})
    mine = Account(os.getuid(), os.getgid())  # (the only user a test may switch to: its own)
    host = host_of(leases, root, sockets, accounts=[mine])
    try:
        await host.followed()
        await until(states_are(host, one="running", two="failed"), 60, every=0.1)
        assert "no user is left for the kind two" in host.workers["two"].why
        assert stat.S_IMODE((root / "kinds" / "one").stat().st_mode) == 0o700  # (its directory: its user's alone)
        assert stat.S_IMODE((root / "accounts.json").stat().st_mode) == 0o600
    finally:
        await host.stop()


async def test_a_kind_no_longer_asked_for_gives_its_user_back(
    pod: tuple[PodLeases, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    leases, root, sockets = pod
    monkeypatch.setattr(sandboxes, "FIRST_WAIT", 0.05)
    one = SandboxSource("one", f"{KINDS}:boxes", size=1)
    two = SandboxSource("two", f"{KINDS}:boxes", size=1)
    await held_by(leases, "run_a", settings={"sandboxes": {"one": one.to_json()}})
    host = host_of(leases, root, sockets, accounts=[Account(os.getuid(), os.getgid())])  # (one user, for two kinds)
    try:
        await host.followed()
        await until(states_are(host, one="running"), 60, every=0.1)
        await held_by(leases, "run_a", settings={"sandboxes": {"two": two.to_json()}})  # (the next run's kind)
        await host.followed()
        await until(states_are(host, two="running"), 60, every=0.1)  # (its user given back, and taken by two)
        assert json.loads((root / "accounts.json").read_text()) == {"two": os.getuid()}
    finally:
        await host.stop()


async def test_the_host_serves_on_a_socket_its_own_and_a_dead_kinds_socket_refuses(
    pod: tuple[PodLeases, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    leases, root, sockets = pod
    monkeypatch.setattr(sandboxes, "FIRST_WAIT", 30.0)  # (not started again while it is looked at)
    assert sandboxes.PRCTL is not None  # (looked up once, at import: never in a child before it runs Python)
    fake = SandboxSource("fake", f"{KINDS}:boxes", size=1)
    await held_by(leases, "run_a", settings={"sandboxes": {"fake": fake.to_json()}})
    host = host_of(leases, root, sockets)
    listener = sandboxes._listening(sockets / "host.sock")  # pyright: ignore[reportPrivateUsage]
    serving = asyncio.ensure_future(sandboxes.served_on(host.app(), listener))
    try:
        await host.followed()
        await until(states_are(host, fake="running"), 60, every=0.1)
        assert stat.S_IMODE((sockets / "host.sock").stat().st_mode) == 0o600  # (root's alone, on a pod)
        transport = httpx.AsyncHTTPTransport(uds=str(sockets / "host.sock"))
        async with httpx.AsyncClient(transport=transport, base_url="http://host") as client:
            assert (await client.get("/fake/capacity")).json() == {"size": 1, "leased": 0}
        worker = host.workers["fake"]
        assert worker._listener is None  # pyright: ignore[reportPrivateUsage]  (the host's copy closed)
        assert worker.process is not None
        worker.process.kill()
        await worker.process.wait()
        with socket.socket(socket.AF_UNIX) as probe, pytest.raises(ConnectionRefusedError):
            probe.connect(str(sockets / "fake.sock"))  # (refused at once, not left waiting)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        listener.close()
        await host.stop()


def test_a_host_ends_what_a_host_before_it_left(tmp_path: Path) -> None:
    left = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "rollout.harness.pool_server"],
                            start_new_session=True)  # fmt: skip
    (tmp_path / "pids").mkdir()
    (tmp_path / "pids" / "fake.pid").write_text(str(left.pid))
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    (tmp_path / "pids" / "other.pid").write_text(str(unrelated.pid))  # (not a pool's: left alone)
    try:
        host = SandboxHost(POD, None, Here(tmp_path, None), tmp_path)
        assert host.reap() == [left.pid]
        assert left.wait(10) != 0 and unrelated.poll() is None
        assert list((tmp_path / "pids").glob("*.pid")) == []
    finally:
        unrelated.kill()
        left.kill()
