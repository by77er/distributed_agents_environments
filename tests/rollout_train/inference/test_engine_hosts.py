# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray's handles are partly untyped.)
"""Engine hosts on the session's Ray, with scripted engines: a host follows what the runs bound to it serve, keeping
as many adapters as each run's `max_lag` lets a turn sample from; turns go to a host that holds a checkpoint within
lag, through `HostServer`; full weights load replica by replica, and a turn caught by a load is sampled again; a host
started again loads what it served; and it asks Ray for its share of a GPU."""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.cluster import load
from rollout_train.inference import Generation, NotLoaded, RemoteChannel
from rollout_train.inference.hosts import HostPausable, HostServer, HostSpec, host_spec, started
from rollout_train.inference.remote import ENGINES, Unreachable
from rollout_train.ledger import Fence, FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import scope
from rollout_train.recorder import Renderer
from rollout_train.run_settings import RunSettings
from rollout_train.serving import Serving, record_serving, serving_of
from rollout_train.stores import FILES
from rollout_train.testing import PlainRenderer
from tests.local_ray import LocalRay

MODEL = "tiny"
ROOT = Path(__file__).resolve().parents[3]
OPTIONS: dict[str, Any] = {"max_tokens": 8, "temperature": 1.0, "top_p": 1.0, "stop_token_ids": [10]}
SCRIPTED = HostSpec("rollout_train.testing:scripted_engine", MODEL)


class Stores:
    """The ledger and blob store every host of a test reaches, and a run's checkpoints written into them."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
        self.ledger_at = {"directory": str(tmp_path / "ledger")}
        self.blobs_at = {"kind": FILES, "directory": str(tmp_path / "blobs")}
        self.fences: dict[str, Fence] = {}

    async def fence(self, run: str) -> Fence:
        if run not in self.fences:
            self.fences[run] = await self.checkpoints.ledger.take(scope(run))
        return self.fences[run]

    async def served(
        self, run: str, *, parent: Checkpoint | None = None, kind: str = "lora", max_lag: int | None = None
    ) -> Checkpoint:
        """A new checkpoint of `run` (its weights one small file), and the record that its channel `policy` serves
        it, as the training loop writes them."""
        fence, id = await self.fence(run), new_id()
        weights = self.tmp_path / "trained" / id
        weights.mkdir(parents=True)
        (weights / "weights.bin").write_text(id)
        parents = [parent.id] if parent else []
        made = await self.checkpoints.add(fence, id, weights=weights, run=run, base=MODEL, parents=parents, kind=kind)
        said = Serving("policy", made.id, made.depth, made.kind, made.weights, model=MODEL, max_lag=max_lag)
        await record_serving(self.checkpoints.ledger, run, said, fence)
        return made

    def presence(self) -> FilePresence:
        return FilePresence(self.tmp_path / "ledger")


@asynccontextmanager
async def hosts(
    stores: Stores,
    count: int = 1,
    *,
    spec: HostSpec = SCRIPTED,
    bound: Sequence[tuple[str, str]] = (("r", "policy"),),
    every: float = 0.1,
) -> AsyncGenerator[list[Any]]:
    """`count` engine hosts (replicas of what they serve: by default `policy` of run `r`), looking every `every`
    seconds; ended after."""
    import ray

    tag = uuid.uuid4().hex[:8]
    handles = [
        started(
            f"host-{tag}-{index}", spec, stores.ledger_at, stores.blobs_at, bound=bound, replica=(index, count),
            directory=str(stores.tmp_path / "scratch"), every=every, beating=0.2,
        )
        for index in range(count)
    ]  # fmt: skip
    try:
        yield handles
    finally:
        for handle in handles:
            ray.kill(handle, no_restart=True)


async def until(condition: Callable[[], Awaitable[bool]], seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not await condition():
        assert time.monotonic() < deadline, "it did not come to pass"
        await asyncio.sleep(0.05)


async def holds(handle: Any) -> set[str]:
    return set(await handle.models.remote())


async def test_a_host_follows_its_runs_and_keeps_what_their_lag_lets_a_turn_sample(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    stores = Stores(tmp_path)
    async with hosts(stores) as (host,):
        assert await holds(host) == {MODEL}  # (the run says nothing yet: the model's own weights)
        made: list[Checkpoint] = []
        for _ in range(4):  # (max_lag of one, unless the record says: the one served and the one before)
            made.append(await stores.served("r", parent=made[-1] if made else None))
            await until(lambda: _holding(host, made[-1].id))
        assert await holds(host) == {MODEL, made[2].id, made[3].id}
        made.append(await stores.served("r", parent=made[-1], max_lag=2))  # (the run's max_lag changed to two)
        await until(lambda: _holding(host, made[-1].id))
        assert await holds(host) == {MODEL, made[2].id, made[3].id, made[4].id}
        made.append(await stores.served("r", parent=made[-1], max_lag=0))  # (an eval's: the one served alone)
        await until(lambda: _holding(host, made[-1].id))
        assert await holds(host) == {MODEL, made[5].id}

        other = await stores.served("s")  # a second run bound to the same host: its adapters sit beside r's
        await host.bind.remote("s", "policy")
        await until(lambda: _holding(host, other.id))
        assert await holds(host) == {MODEL, made[5].id, other.id}
        server = HostServer(host, "host")
        reply = await server.generate([65], adapter=other.id, **OPTIONS)
        assert reply.model == other.id and reply.tokens
        assert (await server.generate([65], adapter=None, **OPTIONS)).model == MODEL
        with pytest.raises(NotLoaded, match=f"does not hold {made[0].id}"):
            await server.generate([65], adapter=made[0].id, **OPTIONS)
        assert (await server.generate([65], adapter=None, top=2, **OPTIONS)).top_tokens  # (a teacher's own samples)
        scores = await server.score([65, 66, 67], start=1, top=2, adapter=other.id, session="r_1/teacher")
        assert (scores.model, scores.start, scores.logprobs) == (other.id, 1, [-0.25, -0.25])
        assert scores.top_tokens == [[66, 67], [67, 68]] and scores.top_logprobs == [[-0.25, -1.25]] * 2
        assert (await server.score([65, 66], start=1, adapter=None)).model == MODEL
        with pytest.raises(NotLoaded, match=f"does not hold {made[0].id}"):
            await server.score([65, 66], start=1, adapter=made[0].id)
        with pytest.raises(ValueError, match="not a range"):  # (raised in the host, as the gateway refuses it)
            await server.score([65, 66], start=2, end=1, adapter=None)

        beats = [beat for beat in await stores.presence().beats() if beat.about.get("kind") == ENGINES]
        (beat,) = beats
        assert beat.about["runs"] == ["r", "s"] and beat.about["follows"] is None and beat.about["replica"] == 0
        listed = cast(list[dict[str, Any]], beat.about["channels"])
        adapters = {(each["run"], each["checkpoint"], each["depth"]) for each in listed[0]["engines"][0]["adapters"]}
        assert adapters == {("r", made[5].id, 6), ("s", other.id, 1)}  # (each engine says all it holds)

        await host.unbind.remote("s", "policy")
        await until(lambda: _not_holding(host, other.id))
        assert await host.bound.remote() == [("r", "policy")]


async def _holding(host: Any, name: str) -> bool:
    return name in await holds(host)


async def _not_holding(host: Any, name: str) -> bool:
    return name not in await holds(host)


async def test_turns_go_to_a_host_holding_a_checkpoint_within_lag(tmp_path: Path, local_ray: LocalRay) -> None:
    stores = Stores(tmp_path)
    first = await stores.served("r")
    async with hosts(stores) as (prompt,), hosts(stores, every=3600.0) as (lagging,):
        await until(lambda: _holding(prompt, first.id))
        await until(lambda: _holding(lagging, first.id))  # (its one look, when it started)
        second = await stores.served("r", parent=first)
        await until(lambda: _holding(prompt, second.id))
        servers = [HostServer(prompt, "prompt"), HostServer(lagging, "lagging")]

        async def wanted() -> list[Serving]:
            return await serving_of(stores.checkpoints.ledger, "r", "policy")

        channel = RemoteChannel(
            "policy", cast(Renderer, PlainRenderer()), _limits(), model=MODEL, servers=servers, wanted=wanted,
            every=0.05, patience=2.0,
        )  # fmt: skip
        await channel.refresh(now=True)
        assert channel.offered("prompt") == next(each for each in await wanted() if each.checkpoint == second.id)
        assert channel.offered("lagging") is not None and channel.offered("lagging").checkpoint == first.id  # type: ignore[union-attr]
        for session in (f"s{each}" for each in range(8)):  # (each session's turns go to one host, by its id)
            adapter, depth = await channel.weights(session)
            address = channel.server_of(session)
            assert (adapter, depth) == ((second.id, 2) if address == "prompt" else (first.id, 1))
            reply = await channel.generate([65], adapter=adapter, session=session, **OPTIONS)
            assert reply.model == adapter
        assert channel.context_limit > 0
        with pytest.raises(NotLoaded):  # (the lagging host has not loaded the newest)
            await servers[1].generate([65], adapter=second.id, **OPTIONS)


def _limits() -> Any:
    from rollout_train.inference import Limits

    return Limits()


async def test_full_weights_load_replica_by_replica_and_a_turn_caught_by_a_load_is_sampled_again(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    stores = Stores(tmp_path)
    loading = HostSpec("rollout_train.testing:scripted_engine", MODEL, {"loading": 1.0})
    first = await stores.served("r", kind="full")
    async with hosts(stores, 2, spec=loading) as (one, two):
        await until(lambda: _holding(one, first.id))
        await until(lambda: _holding(two, first.id))
        assert MODEL not in await holds(one)  # (full weights in place of the model's own)
        second = await stores.served("r", parent=first, kind="full")
        seen: list[tuple[bool, bool]] = []
        turns: list[asyncio.Task[Any]] = []
        server = HostServer(two, "two")
        while not seen or not all(seen[-1]):
            now = (second.id in await holds(one), second.id in await holds(two))
            seen.append(now)
            if now == (True, False):  # the second replica serves the checkpoint before until it loads the newest
                turns.append(asyncio.create_task(server.generate([65], adapter=first.id, **OPTIONS)))
            await asyncio.sleep(0.05)
        assert (True, False) in seen and (False, True) not in seen  # (one at a time: the first, then the second)
        ended = await asyncio.gather(*turns, return_exceptions=True)
        assert {type(each) for each in ended} <= {Generation, NotLoaded}
        assert any(isinstance(each, Generation) and each.model == first.id for each in ended)  # (before its load)
        assert any(isinstance(each, NotLoaded) for each in ended)  # (caught by its load: sampled again elsewhere)
        reply = await server.generate([65], adapter=second.id, **OPTIONS)
        assert reply.model == second.id


async def test_a_host_started_again_loads_what_it_served(tmp_path: Path, local_ray: LocalRay) -> None:
    import ray

    stores = Stores(tmp_path)
    first = await stores.served("r")
    async with hosts(stores) as (host,):
        await until(lambda: _holding(host, first.id))
        server = HostServer(host, "host")
        before = (await host.about.remote())["started"]
        ray.kill(host, no_restart=False)  # (its process dies; Ray starts it again, with new engines)
        await until(lambda: _started_after(host, before), seconds=60)
        await until(lambda: _answers_holding(server, first.id))
        assert (await server.generate([65], adapter=first.id, **OPTIONS)).model == first.id


async def _started_after(host: Any, before: float) -> bool:
    from ray.exceptions import RayActorError

    try:
        return (await host.about.remote())["started"] > before
    except RayActorError:  # (it is starting again)
        return False


async def _answers_holding(server: HostServer, name: str) -> bool:
    try:
        return name in await server.models(within=1.0)
    except Unreachable:
        return False


async def test_a_paused_host_holds_turns_back_until_it_resumes(tmp_path: Path, local_ray: LocalRay) -> None:
    stores = Stores(tmp_path)
    async with hosts(stores, bound=[]) as (host,):
        pausable = HostPausable(host)
        await pausable.pause()
        await pausable.sleep()
        turn = asyncio.create_task(HostServer(host, "host").generate([65], adapter=None, **OPTIONS))
        await asyncio.sleep(0.3)
        assert not turn.done()
        await pausable.wake()
        pausable.resume()
        assert (await turn).model == MODEL


async def test_hosts_ask_ray_for_their_share_of_a_gpu(tmp_path: Path, local_ray: LocalRay) -> None:
    stores = Stores(tmp_path)
    half = HostSpec("rollout_train.testing:scripted_engine", MODEL, gpus=0.5)
    async with hosts(stores, 2, spec=half, bound=[]) as (one, two):  # (the session's one GPU, shared)
        assert await holds(one) == await holds(two) == {MODEL}
        async with hosts(stores, spec=half, bound=[]) as (three,):
            waiting = asyncio.ensure_future(three.models.remote())
            await asyncio.sleep(1.0)
            assert not waiting.done()  # (no GPU is left for it: on Kubernetes, the autoscaler would add a node)
            import ray

            ray.kill(one, no_restart=True)  # (its half is free again)
            assert set(await asyncio.wait_for(waiting, 30)) == {MODEL}


def test_a_hosts_share_of_a_gpu_and_its_engine_come_from_the_cluster_and_the_run() -> None:
    cluster = load(ROOT / "deploy" / "clusters" / "example.toml")
    alone = host_spec(cluster, "local-vllm", "Qwen/Qwen3.5-4B")
    assert (alone.engine, alone.gpus) == ("rollout_vllm:VllmEngine", 1.0)
    offered = cluster.inference["local-vllm"]
    assert alone.options == {**offered.models["Qwen/Qwen3.5-4B"].options, "max_logprobs": 20}  # (what it declares)
    assert offered.capabilities.top_logprobs == 20

    def trained_by(trainer: str) -> RunSettings:
        return RunSettings({"trainer.provider": trainer})

    colocated = host_spec(cluster, "local-vllm", "Qwen/Qwen3.5-4B", settings=trained_by("local-lora"))
    assert colocated.gpus == 0.5  # (the trainer shares the card: each asks Ray for half)
    assert host_spec(cluster, "local-vllm", "Qwen/Qwen3.5-4B", settings=trained_by("tinker-lora")).gpus == 1.0
    with pytest.raises(ValueError, match="runs no engine host"):
        host_spec(cluster, "tinker", "Qwen/Qwen3.5-4B")
    with pytest.raises(ValueError, match="does not serve"):
        host_spec(cluster, "local-vllm", "gpt-5")
