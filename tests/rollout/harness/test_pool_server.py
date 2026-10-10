"""A pool in a process of its own, for one run at a time (`rollout.harness.pool_server`): it admits only the run's keys;
moved to another run it forgets the others' leases, and moved to none it marks every lease lost; a lost lease's key gets
`SandboxLost`, from an operation as from an acquire, never a fresh sandbox, also after the pool is started again; and a
sandbox made for a key whose run stopped being served while it was made is not kept. Over HTTP a lost key's operation
is a 410, and a 503 that does not say the pool is full is `PoolUnavailable`."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from rollout.harness.pool_server import FollowingPool, JsonLeases, of_run
from rollout.harness.remote import RemotePool, serve_pool
from rollout.harness.sandboxes import (
    LeaseRefused,
    NoCapacity,
    PoolUnavailable,
    Reach,
    SandboxLost,
    SandboxPool,
    SandboxSpec,
)
from rollout.testing import FakeSandboxes

BOX = SandboxSpec(kind="fake")


def pool_of(tmp_path: Path, sandboxes: FakeSandboxes | None = None, run: str | None = "run_a") -> FollowingPool:
    return FollowingPool(sandboxes or FakeSandboxes(), name="fake@pod", leases=JsonLeases(tmp_path / "leases.json"),
                         run=run)  # fmt: skip


def test_a_key_is_of_a_run_by_its_first_part_or_one_of_its_evals() -> None:
    assert of_run("run_a/1/1/1/box", "run_a") and of_run("run_a-eval-3/1/1/1/box", "run_a")
    assert not of_run("run_ab/1/1/1/box", "run_a") and not of_run("run_a/1/1/1/box", None)


async def test_only_the_served_runs_keys_are_admitted(tmp_path: Path) -> None:
    pool = pool_of(tmp_path)
    await pool.acquire(BOX, "run_a/1/1/1/box")
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "run_b/1/1/1/box")


async def test_with_no_run_every_lease_is_lost_and_its_key_never_gets_a_fresh_sandbox(tmp_path: Path) -> None:
    sandboxes = FakeSandboxes()
    pool = pool_of(tmp_path, sandboxes)
    lease = await pool.acquire(BOX, "run_a/1/1/1/box")
    assert await pool.follow(None) == ["run_a/1/1/1/box"]  # (its run's driver is gone)
    assert sandboxes.deleted == [lease.handle]
    with pytest.raises(SandboxLost):
        await pool.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")
    await pool.follow("run_a")  # (the driver is back: its key is still lost)
    with pytest.raises(SandboxLost):
        await pool.acquire(BOX, lease.key)
    assert sandboxes.made == [lease.handle]


async def test_another_run_forgets_the_others_leases(tmp_path: Path) -> None:
    sandboxes = FakeSandboxes()
    pool = pool_of(tmp_path, sandboxes)
    await pool.acquire(BOX, "run_a/1/1/1/box")
    await pool.follow(None)
    assert await pool.follow("run_b") == ["run_a/1/1/1/box"] and await pool.held() == []
    with pytest.raises(LeaseRefused):  # (another run's key: refused, not given a fresh sandbox)
        await pool.acquire(BOX, "run_a/1/1/1/box")
    await pool.acquire(BOX, "run_b/1/1/1/box")


async def test_a_pool_started_again_says_its_sandboxes_were_lost(tmp_path: Path) -> None:
    lease = await pool_of(tmp_path).acquire(BOX, "run_a/1/1/1/box")
    sandboxes = FakeSandboxes()
    again = pool_of(tmp_path, sandboxes)  # (its process started again: its sandboxes are gone, its leases kept)
    assert await again.sweep() == [lease.key]
    with pytest.raises(SandboxLost):
        await again.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")
    with pytest.raises(SandboxLost):
        await again.acquire(BOX, lease.key)
    assert sandboxes.made == []


async def test_a_sandbox_made_while_its_run_stopped_being_served_is_not_kept(tmp_path: Path) -> None:
    started, finish = asyncio.Event(), asyncio.Event()

    class Slow(FakeSandboxes):
        async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
            started.set()
            await finish.wait()
            return await super().create(handle, spec, environment)

    sandboxes = Slow()
    pool = pool_of(tmp_path, sandboxes)
    acquiring = asyncio.ensure_future(pool.acquire(BOX, "run_a/1/1/1/box"))
    await started.wait()
    await pool.follow("run_b")  # (while it is made: its key is in no snapshot of the leases)
    finish.set()
    with pytest.raises(LeaseRefused):
        await acquiring
    assert sandboxes.sandboxes == {} and await pool.held() == []


async def test_not_admitting_it_takes_no_new_key_and_serves_those_it_holds(tmp_path: Path) -> None:
    pool = pool_of(tmp_path)
    lease = await pool.acquire(BOX, "run_a/1/1/1/box")
    await pool.follow("run_a", admitting=False)  # (its run's driver has not renewed the machine lately)
    with pytest.raises(PoolUnavailable):
        await pool.acquire(BOX, "run_a/1/2/1/box")
    assert await pool.acquire(BOX, lease.key) == lease
    await pool.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=serve_pool(pool)))
    with pytest.raises(PoolUnavailable):  # (over HTTP too: a 503 that is not "full")
        await RemotePool("http://pool", client=client).acquire(BOX, "run_a/1/3/1/box")


async def test_a_sandbox_made_while_its_run_lost_its_driver_is_lost(tmp_path: Path) -> None:
    started, finish = asyncio.Event(), asyncio.Event()

    class Slow(FakeSandboxes):
        async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
            started.set()
            await finish.wait()
            return await super().create(handle, spec, environment)

    sandboxes = Slow()
    pool = pool_of(tmp_path, sandboxes)
    acquiring = asyncio.ensure_future(pool.acquire(BOX, "run_a/1/1/1/box"))
    await started.wait()
    await pool.follow(None)  # (no run while it is made: its driver is gone)
    finish.set()
    with pytest.raises(SandboxLost):
        await acquiring
    assert sandboxes.sandboxes == {} and [each.lost for each in await pool.held()] == [True]


async def test_a_pool_is_resized_where_its_provider_lets_it(tmp_path: Path) -> None:
    sandboxes = FakeSandboxes(size=2)
    pool = pool_of(tmp_path, sandboxes)
    assert pool.resize(5) and (await pool.capacity()).size == 5

    class Fixed(FakeSandboxes):
        @property
        def size(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
            return 2

        @size.setter
        def size(self, value: int) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
            pass

    assert not pool_of(tmp_path / "fixed", Fixed()).resize(5)


async def test_over_http_a_lost_keys_operation_is_sandbox_lost_and_an_unanswered_503_is_unavailable(
    tmp_path: Path,
) -> None:
    pool = pool_of(tmp_path, FakeSandboxes(size=1))
    lease = await pool.acquire(BOX, "run_a/1/1/1/box")
    await pool.lose(lease.key)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=serve_pool(pool)))
    remote = RemotePool("http://pool", client=client)
    with pytest.raises(SandboxLost):
        await remote.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")
    full = SandboxPool(FakeSandboxes(size=0))
    with pytest.raises(NoCapacity) as said:
        await RemotePool("http://full", client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=serve_pool(full)))).acquire(BOX, "run_a/1/2/1/box")  # fmt: skip
    assert not isinstance(said.value, PoolUnavailable)

    async def proxy(request: object) -> PlainTextResponse:  # (a proxy whose pool is not up: its own plain answer)
        return PlainTextResponse("upstream connect error", status_code=503)

    starting = Starlette(routes=[Route("/acquire", proxy, methods=["POST"])])
    with pytest.raises(PoolUnavailable):
        await RemotePool("http://starting", client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=starting))).acquire(BOX, "run_a/1/3/1/box")  # fmt: skip
