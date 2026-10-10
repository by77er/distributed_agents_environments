"""Sandbox pools on a host pod (`rollout_train.pods.sandboxes`): sized from the pod's spare vCPUs and memory (as its
lease records them from RunPod, else as its container is limited), following the pod's lease (only keys of the run that
holds it, every lease released when another run or none holds it), releasing leases unused for long, and, started
again, answering `SandboxLost` for the sandboxes it lost rather than making new ones. Served under each kind, as the
pod's proxy reaches them."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.sandboxes import LeaseRefused, SandboxLost, SandboxSpec
from rollout.testing import FakeSandboxes, until
from rollout_train.ledger import FileLedger
from rollout_train.pods.leases import HELD, IDLE, PodLease, PodLeases, pod_leases_of
from rollout_train.pods.sandboxes import (
    Machine,
    PodSandboxes,
    Served,
    UsedPool,
    machine_of,
    main,
    served_pools,
    sized,
)
from rollout_train.sandboxes import FileLeases
from tests.local_ray import free_port

BOX = SandboxSpec(kind="fake")
MINECRAFT = Served("minecraft_team.worlds:worlds")
POD = "rollout-test-host-0"


def test_a_pool_holds_a_sandbox_per_spare_vcpu_bounded_by_its_spare_memory() -> None:
    gb = 1e9 / 2**30
    assert sized(Machine(16, 188 * gb), {"minecraft": MINECRAFT}) == {"minecraft": 12}  # (an RTX PRO 6000 pod)
    assert sized(Machine(8, 125 * gb), {"minecraft": MINECRAFT}) == {"minecraft": 4}  # (the cheapest H100 SXM pod)
    assert sized(Machine(32, 80), {"minecraft": MINECRAFT}) == {"minecraft": 6}  # (16 GiB spare, 2.4 a world)
    assert sized(Machine(16, 188), {"minecraft": replace(MINECRAFT, size=5)}) == {"minecraft": 5}
    assert sized(Machine(3, 188), {"minecraft": MINECRAFT}) == {"minecraft": 0}  # (nothing spare)
    assert sized(Machine(16, 188), {"minecraft": MINECRAFT}, cpus=2) == {"minecraft": 14}
    shared = sized(Machine(16, 188), {"b": replace(MINECRAFT, cpus=2), "a": replace(MINECRAFT, size=4)})
    assert shared == {"a": 4, "b": 4}  # (in order of their names: a takes 4 of the 12 spare, b 2 each of the rest)


def test_a_pods_vcpus_and_memory_are_its_leases_else_its_containers(tmp_path: Path) -> None:
    lease = PodLease(POD, "host", 0, "host", "image", "m", "gpu", 1.0, vcpus=16, memory_gb=188.0)
    assert machine_of(lease, tmp_path) == Machine(16, 188e9 / 2**30)
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    (tmp_path / "memory.max").write_text(f"{125 * 2**30}\n")
    assert machine_of(replace(lease, vcpus=None, memory_gb=None), tmp_path) == Machine(8, 125)  # (cgroup v2)
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "memory").mkdir()
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("400000")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(64 * 2**30))
    assert machine_of(None, v1) == Machine(4, 64)
    unlimited = machine_of(None, tmp_path / "none")  # (no limits: the machine's own)
    assert unlimited.cpus >= 1 and unlimited.memory_gib > 0


async def held_by(leases: PodLeases, run: str | None) -> None:
    """Say in the pod's lease that `run` holds it (none: released)."""
    there = await leases.get(POD)
    lease = replace(there, run=run, state=HELD if run else IDLE) if there is not None else PodLease(
        POD, "host", 0, "host", "image", "m", "gpu", 1.0, run=run, state=HELD if run else IDLE)  # fmt: skip
    await leases.put(lease, expect=there.version if there is not None else None)


def following(tmp_path: Path, **options: float) -> tuple[PodLeases, PodSandboxes, FakeSandboxes, UsedPool]:
    leases = pod_leases_of(FileLedger(tmp_path / "ledger"))
    assert leases is not None
    pod = PodSandboxes(POD, leases, **options)
    sandboxes = FakeSandboxes(size=4)
    pool = UsedPool(sandboxes, name=f"fake@{POD}", leases=FileLeases(tmp_path / "fake"), admits=pod.admitted)
    pod.pools["fake"] = pool
    return leases, pod, sandboxes, pool


async def test_a_pods_pool_admits_only_keys_of_the_run_that_holds_it(tmp_path: Path) -> None:
    leases, pod, sandboxes, pool = following(tmp_path)
    await held_by(leases, "run_a")
    await pod.follow()
    await pool.acquire(BOX, "run_a/1/1/1/box")
    await pool.acquire(BOX, "run_a-eval-3/1/1/1/box")  # (one of the run's evals)
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "run_b/1/1/1/box")
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "run_ab/1/1/1/box")
    assert len(sandboxes.sandboxes) == 2


async def test_every_lease_is_released_when_another_run_or_none_holds_the_pod(tmp_path: Path) -> None:
    leases, pod, sandboxes, pool = following(tmp_path)
    await held_by(leases, "run_a")
    await pod.follow()
    await pool.acquire(BOX, "run_a/1/1/1/box")
    await held_by(leases, "run_b")  # (taken by another run, before the pool's next look)
    await pool.acquire(BOX, "run_b/1/1/1/box")  # (the lease is read again for a key of another run)
    assert pod.run == "run_b" and [each.key for each in await pool.held()] == ["run_b/1/1/1/box"]
    assert len(sandboxes.deleted) == 1 and len(sandboxes.sandboxes) == 1
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "run_a/1/1/1/box")
    await held_by(leases, None)  # (released)
    await pod.follow()
    assert await pool.held() == [] and sandboxes.sandboxes == {} and pod.run is None
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "run_b/1/2/1/box")


async def test_a_lease_unused_for_long_is_released(tmp_path: Path) -> None:
    leases, pod, sandboxes, pool = following(tmp_path, idle=0.3)
    await held_by(leases, "run_a")
    await pod.follow()
    await pool.acquire(BOX, "run_a/1/1/1/box")
    await pool.acquire(BOX, "run_a/1/2/1/box")
    for _ in range(4):  # (one is used, the other is not)
        await asyncio.sleep(0.1)
        await pool.call("run_a/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
        await pod.follow()
    assert [each.key for each in await pool.held()] == ["run_a/1/1/1/box"] and len(sandboxes.deleted) == 1


async def test_a_pool_started_again_says_its_sandboxes_were_lost(tmp_path: Path) -> None:
    leases, pod, _, pool = following(tmp_path)
    await held_by(leases, "run_a")
    await pod.follow()
    await pool.acquire(BOX, "run_a/1/1/1/box")
    await pod.stop()  # (the process is stopped: its sandboxes deleted, its leases kept)
    _, again, sandboxes, pool = following(tmp_path)
    assert await again.follow() == ["run_a/1/1/1/box"]  # (marked lost)
    with pytest.raises(SandboxLost):
        await pool.acquire(BOX, "run_a/1/1/1/box")
    assert sandboxes.made == []
    await pool.acquire(BOX, "run_a/1/2/1/box")


def test_the_pools_are_read_from_the_pods_settings() -> None:
    said = {"minecraft": {"provider": "minecraft_team.worlds:worlds", "settings": {"cache": "/workspace/minecraft"},
                          "size": None, "cpus": 1.0, "memory_gib": 2.4}}  # fmt: skip
    assert served_pools({"ROLLOUT_SANDBOXES": json.dumps(said)}) == {
        "minecraft": Served("minecraft_team.worlds:worlds", {"cache": "/workspace/minecraft"}, None, 1.0, 2.4)
    }
    with pytest.raises(SystemExit, match="not JSON"):
        served_pools({"ROLLOUT_SANDBOXES": "{"})


async def test_a_pod_serves_each_pool_under_its_kind_sized_by_its_lease(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    leases = pod_leases_of(ledger)
    assert leases is not None
    await leases.put(PodLease(POD, "host", 0, "host", "image", "m", "gpu", 1.0, run="run_a", state=HELD, vcpus=8,
                              memory_gb=125.0), expect=None)  # fmt: skip
    port = free_port()
    pools: dict[str, JsonValue] = {
        "fake": {"provider": "tests.rollout_train.rollouts.games:boxes", "settings": {}, "size": None}
    }
    environ = {
        "ROLLOUT_SANDBOXES": json.dumps(pools), "ROLLOUT_POD_NAME": POD,
        "ROLLOUT_LEDGER": json.dumps({"directory": str(ledger.directory)}),
        "ROLLOUT_SANDBOX_ADDRESS": f"127.0.0.1:{port}", "ROLLOUT_SANDBOX_DIRECTORY": str(tmp_path / "sandboxes"),
    }  # fmt: skip
    serving = asyncio.ensure_future(main(environ))

    async def answers() -> bool:
        try:
            async with httpx.AsyncClient() as client:
                return (await client.get(f"http://127.0.0.1:{port}/fake/capacity")).status_code == 200
        except httpx.HTTPError:
            return False

    try:
        await until(answers, 30, every=0.05)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            assert (await client.get("/fake/capacity")).json() == {"size": 4, "leased": 0}  # (8 vCPUs less 4)
            body = {"spec": BOX.model_dump(mode="json"), "key": "run_a/1/1/1/box", "environment": {}}
            assert (await client.post("/fake/acquire", json=body)).status_code == 200
            refused = {**body, "key": "run_b/1/1/1/box"}
            assert (await client.post("/fake/acquire", json=refused)).status_code == 409
        kept = json.loads((tmp_path / "sandboxes" / "fake" / "sandboxes.json").read_text())
        assert [each["key"] for each in kept] == ["run_a/1/1/1/box"]  # (its leases on the pod's volume)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
