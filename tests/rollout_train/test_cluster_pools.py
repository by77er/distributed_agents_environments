"""A kind several pools serve (`[sandboxes.KIND] pools`), each under a name of its own (`KIND-N`): a run reaches them as
one pool (`PodPools` over `ClusterPools`), which puts a new lease on the live pool with the most room and sends each
lease's operations and release to the pool that holds it; a pool whose keeper misses beats takes no new lease and still
answers for its own, and beside a ledger that keeps no heartbeats, one that does not answer takes none; the pools'
keepers, each under its pool's name, neither fence each other out nor sweep each other's leases; and a run's driver
reaches such a kind through that pool, behind its pods where they serve the kind too."""

import asyncio
import contextlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.remote import serve_pool
from rollout.harness.sandboxes import Capacity, NoCapacity, SandboxLost, SandboxPool, SandboxSpec
from rollout.testing import FakeSandboxes, until
from rollout_train import presence
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.pods.leases import HELD, PodLease
from rollout_train.pods.pools import FilePlacements, PodPools
from rollout_train.presence import STALE, Beat, FilePresence, presence_of
from rollout_train.record import table
from rollout_train.rollouts.scheduler import CLAIMS
from rollout_train.sandboxes import ClusterPools, keep, leases_of
from tests.rollout_train.support import ask_boxed

BOX = SandboxSpec(kind="fake")


class Pools:
    """Pools of `fake` sandboxes served over HTTP in this process, `fake-N` at `http://fake-N`, their leases beside one
    ledger; and, as a ledger's heartbeats, a fresh beat of each one's keeper unless `stale` names it."""

    def __init__(self, ledger: Ledger, *sizes: int) -> None:
        self.sandboxes = [FakeSandboxes(size=size) for size in sizes]
        leases = leases_of(ledger)
        self.pools = [SandboxPool(each, name=f"fake-{n}", leases=leases) for n, each in enumerate(self.sandboxes)]
        self.urls = {pool.name: f"http://{pool.name}" for pool in self.pools}
        self.client = httpx.AsyncClient(
            mounts={f"http://{pool.name}": httpx.ASGITransport(app=serve_pool(pool)) for pool in self.pools}
        )
        self.stale: set[str] = set()

    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None: ...

    async def beats(self) -> list[Beat]:
        return [Beat(f"pools/{name}", 0.0, {}, age=STALE + 1 if name in self.stale else 1.0) for name in self.urls]


def routed(pools: Pools, tmp_path: Path) -> PodPools:
    """The pools as one, as a run's driver reaches them."""
    discover = ClusterPools(pools.urls, pools, client=pools.client)
    return PodPools("fake", discover, name="fake@r", placements=FilePlacements(tmp_path / "pools" / "fake.json"),
                    look=0.0)  # fmt: skip


async def test_new_leases_spread_over_the_pools_and_each_leases_calls_go_to_the_pool_that_holds_it(
    tmp_path: Path,
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    pools = Pools(ledger, 2, 2)
    pool = routed(pools, tmp_path)
    leases = [await pool.acquire(BOX, f"r/1/{n}/1/box") for n in range(1, 5)]
    assert [lease.pool for lease in leases] == ["fake-0", "fake-1", "fake-0", "fake-1"]  # (the most room first)
    assert await pool.capacity() == Capacity(size=4, leased=4)
    with pytest.raises(NoCapacity):
        await pool.acquire(BOX, "r/1/5/1/box")
    found = leases_of(ledger)
    assert found is not None and sorted(each.pool for each in await found.all()) == ["fake-0"] * 2 + ["fake-1"] * 2
    said = await pool.call("r/1/2/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    assert cast(dict[str, Any], said.structured)["handle"] == leases[1].handle
    assert [each.name for each in pool.operations()] == ["describe", "write", "fetch"]
    await pool.release("r/1/2/1/box")
    assert pools.sandboxes[1].deleted == [leases[1].handle] and pools.sandboxes[0].deleted == []
    again = await pool.acquire(BOX, "r/1/1/1/box")  # (a retried acquire: the same sandbox, from the same pool)
    assert again.handle == leases[0].handle and pools.sandboxes[0].made == [leases[0].handle, leases[2].handle]
    assert (await pool.acquire(BOX, "r/1/6/1/box")).pool == "fake-1"  # (the room the release left)


async def test_a_pool_whose_keeper_misses_beats_takes_no_new_lease_and_still_answers_for_its_own(
    tmp_path: Path,
) -> None:
    pools = Pools(FileLedger(tmp_path / "ledger"), 2, 2)
    pool = routed(pools, tmp_path)
    first = await pool.acquire(BOX, "r/1/1/1/box")
    assert first.pool == "fake-0"
    pools.stale.add("fake-0")
    assert await pool.capacity() == Capacity(size=2, leased=0)
    assert [(await pool.acquire(BOX, f"r/1/{n}/1/box")).pool for n in (2, 3)] == ["fake-1", "fake-1"]
    with pytest.raises(NoCapacity):  # (fake-0 has room, and takes no new lease)
        await pool.acquire(BOX, "r/1/4/1/box")
    await pool.call("r/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    assert (await pool.acquire(BOX, "r/1/1/1/box")).handle == first.handle
    assert await pool.sweep() == []  # (none of its leases is lost)
    await pool.release("r/1/1/1/box")
    assert pools.sandboxes[0].deleted == [first.handle]


async def test_beside_a_ledger_with_no_heartbeats_a_pool_that_does_not_answer_takes_no_new_lease(
    tmp_path: Path,
) -> None:
    sandboxes = FakeSandboxes(size=1)

    def silent(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no answer", request=request)

    serving = httpx.ASGITransport(app=serve_pool(SandboxPool(sandboxes)))
    client = httpx.AsyncClient(mounts={"http://fake-0": httpx.MockTransport(silent), "http://fake-1": serving})
    discover = ClusterPools({"fake-0": "http://fake-0", "fake-1": "http://fake-1"}, None, client=client)
    assert all(each.live for each in (await discover()).values())
    pool = PodPools("fake", discover, name="fake@r", placements=FilePlacements(tmp_path / "pools" / "fake.json"),
                    look=0.0)  # fmt: skip
    lease = await pool.acquire(BOX, "r/1/1/1/box")  # (fake-0 has the name first, and does not answer)
    assert lease.handle in sandboxes.sandboxes and await pool.capacity() == Capacity(size=1, leased=1)


async def test_the_keepers_of_a_kinds_pools_neither_fence_each_other_out_nor_sweep_each_others_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    pools = Pools(ledger, 2, 2)
    await ask_boxed(ledger, {1: ({}, 2)})
    fence = await ledger.take("runners/elsewhere")
    claim: JsonValue = {"runner": "elsewhere", "fence": fence.number}
    for episode in (1, 2):
        await ledger.append(table("train", CLAIMS), f"1/{episode}/1", claim, fence)
    await beats.beat("elsewhere", {"places": 2})
    keepers = [
        asyncio.create_task(keep(each, ledger, beats, beat_as=f"pools/{each.name}", every=0.02)) for each in pools.pools
    ]
    try:
        discover = ClusterPools(pools.urls, beats, client=pools.client)

        async def live() -> bool:
            return all(each.live for each in (await discover()).values())

        await until(live)  # (each keeper beats under its pool's name)
        pool = PodPools("fake", discover, name="fake@train", look=0.0)
        first, second = await pool.acquire(BOX, "train/1/1/1/box"), await pool.acquire(BOX, "train/1/2/1/box")
        assert (first.pool, second.pool) == ("fake-0", "fake-1")
        await asyncio.sleep(0.2)  # (sweeps of both: each finds the other's lease beside the ledger, and leaves it be)
        assert not any(each.done() for each in keepers)
        fences = await ledger.fences()
        assert (fences["pools/fake-0"], fences["pools/fake-1"]) == (1, 1)
        found = leases_of(ledger)
        assert found is not None and not any(each.lost for each in await found.all())
        assert [each.deleted for each in pools.sandboxes] == [[], []]
        monkeypatch.setattr(presence, "STALE", -1.0)  # (the claims' runner is gone: each keeper ends its own lease)
        await until(lambda: [each.deleted for each in pools.sandboxes] == [[first.handle], [second.handle]])
        with pytest.raises(SandboxLost):
            await pool.call("train/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    finally:
        for each in keepers:
            each.cancel()
        await asyncio.gather(*keepers, return_exceptions=True)


SEVERAL = """
[sandboxes.fake]
provider = "tests.rollout_train.rollouts.games:boxes"
url = "{base}/{{pool}}"
pools = 2
"""


async def test_a_runs_driver_reaches_a_kind_several_pools_serve_through_one_pool_of_its_own(tmp_path: Path) -> None:
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from rollout_train.jobs import Run
    from rollout_train.run_settings import RunSettings
    from rollout_train.stores import Stores
    from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
    from tests.rollout_train.machines import served

    sandboxes = [FakeSandboxes(size=1), FakeSandboxes(size=1)]
    pools = [SandboxPool(each, name=f"fake-{n}") for n, each in enumerate(sandboxes)]
    app = Starlette(routes=[Mount(f"/{n}", app=serve_pool(pool)) for n, pool in enumerate(pools)])
    async with served(app) as (base, _), contextlib.AsyncExitStack() as stack:
        cluster = a_cluster(tmp_path, more=SEVERAL.format(base=base))
        stores = Stores.open(cluster)
        beats = presence_of(stores.ledger)
        assert beats is not None
        for pool in pools:
            await beats.beat(f"pools/{pool.name}", {"kind": "pool"})
        settings = RunSettings({**POLICY, "kind": "train", "environment": WORDS, "name": "several"})
        run = Run(cluster, stores, settings, await stores.registry.create("several"), directory=tmp_path / "run",
                  sandboxes=frozenset({"fake"}))  # fmt: skip
        routing = (await run._pools(stack))["fake"]  # pyright: ignore[reportPrivateUsage]
        assert isinstance(routing, PodPools) and run.pool_bindings["fake"].local == "fake"
        assert routing.name == f"fake@{run.run.id}" and isinstance(routing.discover, ClusterPools)
        assert [each.url for each in routing.discover.pools.values()] == [f"{base}/0", f"{base}/1"]
        leases = [await routing.acquire(BOX, f"other/1/{n}/1/box") for n in (1, 2)]
        assert [lease.pool for lease in leases] == ["fake-0", "fake-1"]
        assert await routing.capacity() == Capacity(size=2, leased=2)
        said = await routing.call("other/1/2/1/box", "describe", {}, effect_id="e", arguments_digest="d")
        assert cast(dict[str, Any], said.structured)["handle"] == leases[1].handle
        await routing.release("other/1/1/1/box")
        assert sandboxes[0].deleted == [leases[0].handle] and sandboxes[1].deleted == []
        assert {each.key for each in await FilePlacements(tmp_path / "run" / "pools" / "fake.json").all()} == {
            "other/1/2/1/box"
        }  # (where each lease is, kept in the run's directory)


HOST = """
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
sandboxes = ["fake"]
[inference.h100.models."m"]
context = 8192
[sandboxes.fake]
provider = "tests.rollout_train.rollouts.games:boxes"
url = "http://sandboxes-fake-{pool}.sandboxes-fake-pools:8710"
pools = 2
on_pods = true
"""


async def test_a_kind_its_pods_serve_has_the_clusters_pools_behind_them_as_one(tmp_path: Path) -> None:
    from rollout_train.jobs import Run
    from rollout_train.run_settings import RunSettings
    from rollout_train.stores import Stores
    from tests.rollout_train.clusters import POLICY, WORDS, a_cluster

    cluster = a_cluster(tmp_path, more=HOST)
    stores = Stores.open(cluster)
    settings = RunSettings({**POLICY, "kind": "train", "environment": WORDS, "name": "worlds"})
    run = Run(cluster, stores, settings, await stores.registry.create("worlds"), directory=tmp_path / "run",
              sandboxes=frozenset({"fake"}))  # fmt: skip
    lease = PodLease("rollout-test-h100-0", "h100", 0, "host", "image", "m", "gpu", 1.0, run=run.run.id, state=HELD)
    async with contextlib.AsyncExitStack() as stack:
        run.pods = SimpleNamespace(leases={lease.pod: lease})
        pool = (await run._pools(stack))["fake"]  # pyright: ignore[reportPrivateUsage]
        assert isinstance(pool, PodPools) and isinstance(pool.fallback, PodPools)
        assert pool.name == f"fake@{run.run.id}" and pool.fallback.name == f"fake@{run.run.id}/cluster"  # (two fences)
        assert isinstance(pool.fallback.discover, ClusterPools)
        assert [each.url for each in pool.fallback.discover.pools.values()] == [
            "http://sandboxes-fake-0.sandboxes-fake-pools:8710",
            "http://sandboxes-fake-1.sandboxes-fake-pools:8710",
        ]
        run.pods = None  # (a run that leases no pod that serves the kind: the cluster's pools alone)
        run.pool_bindings.clear()
        alone = (await run._pools(stack))["fake"]  # pyright: ignore[reportPrivateUsage]
        assert isinstance(alone, PodPools) and alone.name == f"fake@{run.run.id}"
        assert run.pool_bindings["fake"].local == "fake"


def test_a_pool_of_a_kind_several_pools_serve_must_say_which_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rollout_train.cli import main
    from tests.rollout_train.clusters import a_cluster

    a_cluster(tmp_path, more=SEVERAL.format(base="http://127.0.0.1:9"))
    for given in ([], ["--name", "fake"], ["--name", "fake-2"]):
        monkeypatch.setattr(
            "sys.argv", ["rollout", "pool", "--kind", "fake", "--cluster", str(tmp_path / "cluster.toml"), *given]
        )
        with pytest.raises(SystemExit, match="--name fake-N with N from 0 to 1"):
            main()
