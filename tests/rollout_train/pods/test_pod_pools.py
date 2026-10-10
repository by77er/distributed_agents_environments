"""The sandboxes of a kind a run's pods serve, as one pool (`PodPools`): a new lease goes to the pod with the most room,
and acquires at once spread over the pods; releases and operations go to the pod that holds the lease, and only there;
capacity is the live pods' summed; a pod the run no longer holds loses its leases (`SandboxLost`), and one that only
misses beats keeps them; the cluster's pool takes what no pod has room for; a lapsed claim's lease is refused and
released on its pod, by the pool and by its keeper; where each lease is outlives the driver; acquires at once against
slow pods all get leases; an acquire whose answer is lost leaves no sandbox behind, a release that fails is tried again,
and a pod that does not answer holds no acquire up. Over HTTP, the run's pods are found by their leases and reached over
mutual TLS by their identities."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from rollout.harness.remote import serve_pool
from rollout.harness.sandboxes import (
    Capacity,
    LeaseRefused,
    NoCapacity,
    Reach,
    SandboxLost,
    SandboxPool,
    SandboxSpec,
)
from rollout.testing import FakeSandboxes
from rollout_train import presence
from rollout_train.ledger import FileLedger
from rollout_train.pods import GATEWAY_IDENTITY, pod_identity
from rollout_train.pods.identity import POD
from rollout_train.pods.leases import HELD, IDLE, PodLease, pod_leases_of
from rollout_train.pods.pools import ACQUIRING, FALLBACK, RELEASING, FilePlacements, PodPools, Reached
from rollout_train.pods.routing import LeasedPools
from rollout_train.presence import FilePresence
from rollout_train.providers import LEASED, Auth, Tls
from rollout_train.record import table
from rollout_train.rollouts.scheduler import CLAIMS
from rollout_train.sandboxes import sweep
from tests.rollout_train.pods.authority import Authority, served_tls, server_context
from tests.rollout_train.support import ask_boxed

BOX = SandboxSpec(kind="fake")


class Pods:
    """Pods standing in for those a run leases: each a pool of `fake` sandboxes of its own, found by `discover` while
    it is held (and live while it beats)."""

    def __init__(self, **sizes: int) -> None:
        self.sandboxes = {name: FakeSandboxes(size=size) for name, size in sizes.items()}
        self.pools = {name: SandboxPool(each, name=f"fake@{name}") for name, each in self.sandboxes.items()}
        self.held = set(sizes)
        self.live = set(sizes)

    async def discover(self) -> Mapping[str, Reached]:
        return {name: Reached(self.pools[name], live=name in self.live) for name in sorted(self.held)}

    def where(self, key: str) -> str:
        (found,) = [name for name, pool in self.pools.items() if key in {lease.key for lease in pool._held.values()}]  # pyright: ignore[reportPrivateUsage]
        return found


def pod_pools(pods: Pods, tmp_path: Path, **options: Any) -> PodPools:
    placements = FilePlacements(tmp_path / "pods" / "fake.json")
    return PodPools("fake", pods.discover, name="fake@run", placements=placements, look=0.0, **options)


class Slow(FakeSandboxes):
    """Sandboxes that take a while to make."""

    async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
        await asyncio.sleep(0.2)
        return await super().create(handle, spec, environment)


async def test_acquires_at_once_against_slow_pods_all_get_leases(tmp_path: Path) -> None:
    pods = Pods(a=4)
    pods.sandboxes["a"] = Slow(size=4)
    pods.pools["a"] = SandboxPool(pods.sandboxes["a"], name="fake@a")
    pool = pod_pools(pods, tmp_path)
    leases = await asyncio.gather(*(pool.acquire(BOX, f"r/1/{n}/1/box") for n in range(4)), return_exceptions=True)
    assert [type(each).__name__ for each in leases] == ["Lease"] * 4  # (those on their way are not counted twice)


class Losing:
    """A pod's pool whose answers to acquires are lost (after it made the sandbox), and whose releases fail while
    `failing`."""

    def __init__(self, pool: SandboxPool) -> None:
        self.pool = pool
        self.failing = False

    def operations(self) -> Any:
        return self.pool.operations()

    async def capacity(self) -> Capacity:
        return await self.pool.capacity()

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Any:
        await self.pool.acquire(spec, key, environment)
        raise httpx.ReadTimeout("the answer was lost")

    async def release(self, key: str) -> None:
        if self.failing:
            raise httpx.ConnectError("the pod does not answer")
        await self.pool.release(key)

    async def call(self, *arguments: Any, **options: Any) -> Any:
        return await self.pool.call(*arguments, **options)


async def test_an_acquire_whose_answer_is_lost_leaves_no_sandbox_behind(tmp_path: Path) -> None:
    pods = Pods(a=2)
    losing = Losing(pods.pools["a"])
    pods.pools["a"] = cast(Any, losing)
    pool = pod_pools(pods, tmp_path)
    with pytest.raises(NoCapacity):  # (passed on to the next pod: there is none)
        await pool.acquire(BOX, "r/1/1/1/box")
    assert pods.sandboxes["a"].sandboxes == {} and await pool.held() == []  # (released on its pod at once)
    losing.failing = True  # (and where that release fails too, the next sweep releases it)
    with pytest.raises(NoCapacity):
        await pool.acquire(BOX, "r/1/2/1/box")
    (kept,) = await FilePlacements(tmp_path / "pods" / "fake.json").all()
    assert (kept.key, kept.state) == ("r/1/2/1/box", ACQUIRING) and len(pods.sandboxes["a"].sandboxes) == 1
    losing.failing = False
    assert await pool.sweep() == ["r/1/2/1/box"]
    assert pods.sandboxes["a"].sandboxes == {} and await pool.held() == []


async def test_a_release_that_fails_is_kept_and_tried_again(tmp_path: Path) -> None:
    pods = Pods(a=2)
    pool = pod_pools(pods, tmp_path)
    lease = await pool.acquire(BOX, "r/1/1/1/box")
    losing = Losing(pods.pools["a"])
    losing.failing = True
    pods.pools["a"] = cast(Any, losing)
    await pool.release(lease.key)
    (kept,) = await FilePlacements(tmp_path / "pods" / "fake.json").all()
    assert kept.state == RELEASING and len(pods.sandboxes["a"].sandboxes) == 1
    with pytest.raises(SandboxLost):  # (its key cannot have it back meanwhile)
        await pool.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")
    losing.failing = False
    assert await pool.sweep() == [lease.key] and pods.sandboxes["a"].sandboxes == {}


async def test_a_pod_that_does_not_answer_holds_no_acquire_up(tmp_path: Path) -> None:
    pods = Pods(a=4, b=1)

    class Silent:
        async def capacity(self) -> Capacity:
            raise httpx.ConnectTimeout("no answer")

    pods.pools["a"] = cast(Any, Silent())
    pool = pod_pools(pods, tmp_path)
    await pool.acquire(BOX, "r/1/1/1/box")
    assert len(pods.sandboxes["b"].sandboxes) == 1 and await pool.capacity() == Capacity(size=1, leased=1)


async def test_a_pod_another_run_took_since_the_look_passes_the_key_to_the_next(tmp_path: Path) -> None:
    pods = Pods(a=4, b=1)

    class Taken(Losing):
        async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Any:
            raise LeaseRefused(f"{key} is not of the run this pod serves")

    pods.pools["a"] = cast(Any, Taken(pods.pools["a"]))
    pool = pod_pools(pods, tmp_path)
    assert (await pool.acquire(BOX, "r/1/1/1/box")).pool == "fake@b"


async def test_a_pod_that_does_not_answer_an_acquire_passes_it_to_the_next(tmp_path: Path) -> None:
    pods = Pods(a=4, b=1)

    class Dropping(Losing):
        async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Any:
            await self.pool.acquire(spec, key, environment)
            raise httpx.ConnectError("the connection dropped")

    pods.pools["a"] = cast(Any, Dropping(pods.pools["a"]))
    pool = pod_pools(pods, tmp_path)
    lease = await pool.acquire(BOX, "r/1/1/1/box")
    assert lease.pool == "fake@b" and pods.sandboxes["a"].sandboxes == {}  # (released on a, as far as it could be)


async def test_a_run_whose_pods_serve_no_kind_it_needs_and_no_pool_behind_says_so_and_fails_in_time(
    tmp_path: Path,
) -> None:
    from rollout_train.jobs import Run, SandboxesUnserved, _within_limits  # pyright: ignore[reportPrivateUsage]
    from rollout_train.run_settings import RunSettings
    from rollout_train.stores import Stores
    from tests.rollout_train.clusters import POLICY, WORDS, a_cluster

    cluster = a_cluster(tmp_path, more=HOST.replace('url = "http://sandboxes-fake:8710"\n', ""))
    stores = Stores.open(cluster)
    settings = RunSettings({**POLICY, "kind": "train", "environment": WORDS, "name": "unserved"})
    told: list[str] = []
    run = Run(cluster, stores, settings, await stores.registry.create("unserved"), directory=tmp_path / "run")

    async def said(detail: str) -> None:
        told.append(detail)

    run.noted = said
    pods = Pods()  # (the run's pods: none serves the kind now)
    watching = asyncio.ensure_future(run._served_by_pods("fake", pod_pools(pods, tmp_path), every=0.01,  # pyright: ignore[reportPrivateUsage]
                                                         within=0.05))  # fmt: skip
    with pytest.raises(SandboxesUnserved, match="no pod of the run's served fake sandboxes"):
        await _within_limits(run, asyncio.sleep(30))
    await watching
    assert told and "a pod that serves fake sandboxes" in told[0]


async def test_a_run_whose_pods_leases_are_lost_to_it_ends_failed_saying_so(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rollout_train.jobs import Run, _within_limits  # pyright: ignore[reportPrivateUsage]
    from rollout_train.pods.leasing import LeaseLost
    from rollout_train.run_settings import RunSettings
    from rollout_train.stores import Stores
    from tests.rollout_train.clusters import POLICY, WORDS, a_cluster

    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)
    settings = RunSettings({**POLICY, "kind": "train", "environment": WORDS, "name": "lost"})
    run = Run(cluster, stores, settings, await stores.registry.create("lost"), directory=tmp_path / "run")

    async def renewing(spent: object) -> None:
        await asyncio.sleep(0.05)
        raise LeaseLost("pod x is no longer held by run lost (its lease went stale and was reaped)")

    renewal = asyncio.ensure_future(run._renewing(SimpleNamespace(renewing=renewing)))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(LeaseLost, match="no longer held"):
        await _within_limits(run, asyncio.sleep(30))
    await renewal


async def test_a_new_lease_goes_to_the_pod_with_the_most_room(tmp_path: Path) -> None:
    pods = Pods(a=2, b=3, c=1)
    pool = pod_pools(pods, tmp_path)
    for key in ("r/1/1/1/box", "r/1/2/1/box", "r/1/3/1/box", "r/1/4/1/box", "r/1/5/1/box", "r/1/6/1/box"):
        await pool.acquire(BOX, key)
    placed = [pods.where(f"r/1/{n}/1/box") for n in range(1, 7)]
    assert placed == ["b", "a", "b", "a", "b", "c"]  # (the most room first, by name among equals)
    assert await pool.capacity() == Capacity(size=6, leased=6)
    with pytest.raises(NoCapacity):
        await pool.acquire(BOX, "r/1/7/1/box")


async def test_acquires_at_once_spread_over_the_pods(tmp_path: Path) -> None:
    pods = Pods(a=4, b=4)
    pool = pod_pools(pods, tmp_path)
    await asyncio.gather(*(pool.acquire(BOX, f"r/1/{n}/1/box") for n in range(1, 5)))
    assert sorted(len(each.sandboxes) for each in pods.sandboxes.values()) == [2, 2]


async def test_releases_and_operations_go_to_the_pod_that_holds_the_lease(tmp_path: Path) -> None:
    pods = Pods(a=1, b=1)
    pool = pod_pools(pods, tmp_path)
    first, second = await pool.acquire(BOX, "r/1/1/1/box"), await pool.acquire(BOX, "r/1/2/1/box")
    assert (first.pool, second.pool) == ("fake@a", "fake@b")  # (the pod's lease, as its pool made it)
    described = await pool.call("r/1/2/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    assert cast(dict[str, Any], described.structured)["handle"] == second.handle
    assert [each.name for each in pool.operations()] == ["describe", "write", "fetch"]
    await pool.release("r/1/2/1/box")
    assert pods.sandboxes["b"].deleted == [second.handle] and pods.sandboxes["a"].deleted == []
    assert await pool.capacity() == Capacity(size=2, leased=1)
    again = await pool.acquire(BOX, "r/1/1/1/box")  # (a retried acquire: the same sandbox, from the same pod)
    assert again.handle == first.handle and pods.sandboxes["a"].made == [first.handle]


async def test_a_pod_the_run_no_longer_holds_loses_its_leases_and_one_that_misses_beats_keeps_them(
    tmp_path: Path,
) -> None:
    pods = Pods(a=2, b=2)
    pool = pod_pools(pods, tmp_path)
    await pool.acquire(BOX, "r/1/1/1/box")  # (on a)
    await pool.acquire(BOX, "r/1/2/1/box")  # (on b)
    pods.live.discard("a")  # a misses its beats: it takes no new lease, and still answers for its own
    assert await pool.capacity() == Capacity(size=2, leased=1)
    await pool.call("r/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    await pool.acquire(BOX, "r/1/3/1/box")
    assert pods.where("r/1/3/1/box") == "b"
    pods.held.discard("a")  # the run no longer holds a (released, deleted, taken by another run)
    with pytest.raises(SandboxLost):
        await pool.call("r/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    assert await pool.sweep() == ["r/1/1/1/box"]
    assert [each.key for each in await pool.held() if each.lost] == ["r/1/1/1/box"]
    with pytest.raises(SandboxLost):  # (its sandbox is never made again elsewhere)
        await pool.acquire(BOX, "r/1/1/1/box")
    assert "r/1/1/1/box" not in {each.key for each in await pool.held()}


async def test_a_pod_whose_pool_lost_a_sandbox_says_so_and_the_lease_is_forgotten(tmp_path: Path) -> None:
    pods = Pods(a=2)
    pool = pod_pools(pods, tmp_path)
    lease = await pool.acquire(BOX, "r/1/1/1/box")
    await pods.sandboxes["a"].delete(lease.handle)  # (the pod's pool was started again: its sandboxes are gone)
    pods.pools["a"] = SandboxPool(pods.sandboxes["a"], name="fake@a", leases=pods.pools["a"].leases)
    with pytest.raises(SandboxLost):
        await pool.acquire(BOX, "r/1/1/1/box")
    assert await pool.held() == []


async def test_the_clusters_pool_takes_what_no_pod_has_room_for(tmp_path: Path) -> None:
    pods = Pods(a=1)
    cluster = FakeSandboxes(size=2)
    pool = pod_pools(pods, tmp_path, fallback=SandboxPool(cluster, name="fake"))
    await pool.acquire(BOX, "r/1/1/1/box")
    elsewhere = await pool.acquire(BOX, "r/1/2/1/box")
    assert elsewhere.pool == "fake" and cluster.made == [elsewhere.handle]
    assert [each.handle for each in await pool.held()] == [f"a/{pods.sandboxes['a'].made[0]}",
                                                           f"{FALLBACK}/{elsewhere.handle}"]  # fmt: skip
    assert await pool.capacity() == Capacity(size=3, leased=2)
    pods.held.clear()  # (the fallback's leases do not go with the pods)
    await pool.call("r/1/2/1/box", "describe", {}, effect_id="e", arguments_digest="d")
    await pool.release("r/1/2/1/box")
    assert cluster.deleted == [elsewhere.handle]


async def test_where_each_lease_is_outlives_the_driver(tmp_path: Path) -> None:
    pods = Pods(a=2, b=2)
    lease = await pod_pools(pods, tmp_path).acquire(BOX, "r/1/1/1/box")
    pods.sandboxes["b"].size = 9  # (b has more room now: the lease stays where it is)
    again = await pod_pools(pods, tmp_path).acquire(BOX, "r/1/1/1/box")  # (a driver started again, adopting it)
    assert again.handle == lease.handle and pods.where("r/1/1/1/box") == "a"
    assert pods.sandboxes["b"].made == []


async def test_a_lapsed_claims_lease_is_refused_and_released_on_its_pod(tmp_path: Path) -> None:
    pods = Pods(a=2)
    lapsed: set[str] = set()

    async def admits(key: str) -> bool:
        return key not in lapsed

    pool = pod_pools(pods, tmp_path, admits=admits)
    lease = await pool.acquire(BOX, "r/1/1/1/box")
    lapsed.add("r/1/1/1/box")
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "r/1/1/1/box")
    assert pods.sandboxes["a"].deleted == [lease.handle] and await pool.held() == []


async def test_the_keeper_releases_a_stale_claims_lease_on_its_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    pods = Pods(a=2)
    pool = pod_pools(pods, tmp_path)
    await ask_boxed(ledger, {1: ({}, 1)})
    fence = await ledger.take("runners/elsewhere")
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "elsewhere", "fence": fence.number}, fence)
    await beats.beat("elsewhere", {"places": 1})
    lease = await pool.acquire(BOX, "train/1/1/1/box")
    assert (await sweep(pool, ledger, beats))[0] == []  # (its claim holds: its runner beats)
    monkeypatch.setattr(presence, "STALE", -1.0)
    assert await sweep(pool, ledger, beats) == (["train/1/1/1/box"], {"train/1/1/1/box"})
    assert pods.sandboxes["a"].deleted == [lease.handle] and await pool.held() == []


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
url = "http://sandboxes-fake:8710"
on_pods = true
"""


async def test_a_runs_driver_reaches_a_kind_its_pods_serve_on_them_and_the_clusters_pool_behind(
    tmp_path: Path,
) -> None:
    import contextlib
    from types import SimpleNamespace

    from rollout.harness.remote import RemotePool
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
        pools = await run._pools(stack)  # pyright: ignore[reportPrivateUsage]
        pool = pools["fake"]
        assert isinstance(pool, PodPools) and run.pool_bindings["fake"].local == "fake"
        assert pool.name == f"fake@{run.run.id}" and isinstance(pool.fallback, RemotePool)
        assert pool.fallback.url == "http://sandboxes-fake:8710"
        run.pods = None  # (a run that leases no pod that serves the kind: the cluster's pool, at its url)
        run.pool_bindings.clear()
        assert await run._pools(stack) == {}  # pyright: ignore[reportPrivateUsage]
        assert run.pool_bindings["fake"].url == "http://sandboxes-fake:8710"


async def test_a_runs_pods_pools_are_found_by_their_leases_and_reached_over_mutual_tls_by_identity(
    tmp_path: Path,
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    leases = pod_leases_of(ledger)
    assert leases is not None
    authority = Authority(tmp_path / "ca")
    gateway = authority.issue("gateway", GATEWAY_IDENTITY)
    tls = Tls(ca=str(authority.root), certificate=str(gateway.certificate), key=str(gateway.key))
    sandboxes = {name: FakeSandboxes(size=2) for name in ("pod-a", "pod-b", "pod-c")}

    def proxy(name: str, identity: str) -> Any:
        """What a pod's Envoy and pools' process serve: the pool under /v1/sandboxes/fake, the gateway's alone."""
        from starlette.applications import Starlette
        from starlette.routing import Mount

        app = Starlette(routes=[Mount("/v1/sandboxes/fake", app=serve_pool(SandboxPool(sandboxes[name])))])
        return served_tls(app, server_context(authority, authority.issue(name, identity), client=GATEWAY_IDENTITY))

    async with (
        proxy("pod-a", pod_identity("pod-a")) as a,
        proxy("pod-b", pod_identity("pod-a")) as b,  # (a certificate that names another pod)
        proxy("pod-c", pod_identity("pod-c")) as c,
    ):
        for slot, (name, address, run) in enumerate([("pod-a", a, "r"), ("pod-b", b, "r"), ("pod-c", c, "other")]):
            lease = PodLease(name, "host", slot, "host", "image", "m", "gpu", 1.0, address=address, run=run,
                             state=HELD)  # fmt: skip
            await leases.put(lease, expect=None)
            await beats.beat(name, {POD: {"name": name, "identity": pod_identity(name), "role": "host",
                                          "ready": True}})  # fmt: skip
        found = LeasedPools(ledger, "r", "fake", ["host"], Auth("mtls", identity=LEASED), tls)
        try:
            reached = await found()
            assert sorted(reached) == ["pod-a", "pod-b"] and all(each.live for each in reached.values())
            assert await reached["pod-a"].pool.capacity() == Capacity(size=2, leased=0)
            with pytest.raises(httpx.ConnectError):  # (refused in the handshake: pod-b's certificate names pod-a)
                await reached["pod-b"].pool.capacity()
            pool = PodPools("fake", found, name="fake@r", look=0.0)
            lease = await pool.acquire(BOX, "r/1/1/1/box")
            assert sandboxes["pod-a"].made == [lease.handle] and await pool.capacity() == Capacity(size=2, leased=1)
            said = await pool.call("r/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
            assert json.loads(said.model_dump_json())["structured"]["handle"] == lease.handle
            first = await leases.get("pod-a")
            assert first is not None
            await leases.put(replace(first, run=None, state=IDLE), expect=first.version)
            with pytest.raises(SandboxLost):  # (released: no longer the run's)
                await pool.call("r/1/1/1/box", "describe", {}, effect_id="e", arguments_digest="d")
        finally:
            await found.aclose()
