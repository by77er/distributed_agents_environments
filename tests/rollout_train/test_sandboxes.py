"""Sandboxes in training: an episode's run leases its sandboxes under the episode's claim, a lease ends with its claim,
a runner claims only what its pools have room for, and a harness inside a sandbox reaches the recorder by itself."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness import (
    ModelBinding,
    PoolBinding,
    Program,
    ProgramReference,
    RecordedModel,
    RunBinding,
    RunContext,
    SandboxPool,
    SandboxSpec,
    register,
)
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout.testing import FakeSandbox, FakeSandboxes
from rollout_train import presence
from rollout_train.gateway import GatewayEndpoints, create_app
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.presence import FilePresence
from rollout_train.record import GROUPS, scope, table
from rollout_train.rollouts import EpisodeRunner, Plan, episodes_of, plan, playing
from rollout_train.rollouts.scheduler import CLAIMS
from rollout_train.sandboxes import FileLeases, keep, leases_of, sweep
from rollout_train.testing import plain_channel, recording

BOX = SandboxSpec(kind="fake")
GATES: dict[str, asyncio.Event] = {}


class Boxed(Program):
    """Waits at its gate, if it has one; then asks its box who it is, and says."""

    def __init__(self, parameters: Mapping[str, JsonValue]) -> None:
        self.gate = parameters.get("gate")

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"box": BOX}

    async def main(self, run: RunContext) -> None:
        if self.gate:
            await GATES.setdefault(str(self.gate), asyncio.Event()).wait()
        described: Any = (await run.sandbox("box").call("describe")).structured
        run.reward(1.0)
        await run.emit("result", {"solved": True, "handle": described["handle"], "key": run.sandbox("box").lease.key})


BINDING = RunBinding(
    models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))},
    pools={"fake": PoolBinding(local="boxes")},
)


async def ask(ledger: Ledger, groups: Mapping[int, tuple[JsonValue, int]], program: type[Program] = Boxed) -> None:
    fence = await ledger.take(scope("train"))
    await plan(ledger, "train", Plan(ProgramReference(program=register(program)), BINDING), fence)
    for number, (row, count) in groups.items():
        record: JsonValue = {"parameters": row, "episodes": count, "decided": float(number)}
        await ledger.append(table("train", GROUPS), str(number), record, fence)


def episode_runner(tmp_path: Path, pool: SandboxPool, url: str | None = None, **options: Any) -> EpisodeRunner:
    """An episode runner over the ledger and blobs in `tmp_path`, recording through a gateway in this process (served
    to harnesses at `url`, if given)."""
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    recorder = recording(plain_channel(always=[("yes\n", "stop")]), ledger=ledger, blobs=blobs, url=url)
    return EpisodeRunner(
        "here",
        ledger,
        LocalRunner(recorder=recorder, pools={"boxes": pool}),
        recorder,
        blobs,
        places=4,
        pools={"boxes": pool},
        every=0.02,
        **options,
    )


@contextlib.asynccontextmanager
async def kept(pool: SandboxPool, ledger: Ledger, beats: FilePresence | None) -> AsyncGenerator[None]:
    """The pool's keeper, sweeping often."""
    task = asyncio.create_task(keep(pool, ledger, beats, every=0.02))
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def until(condition: Any) -> None:
    async with asyncio.timeout(5.0):
        while not await condition():  # noqa: ASYNC110 (the runners write)
            await asyncio.sleep(0.01)


async def test_an_episodes_sandboxes_are_leased_under_its_claim_and_released_when_it_ends(tmp_path: Path) -> None:
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes, leases=FileLeases(tmp_path / "ledger"))
    played = episode_runner(tmp_path, pool)
    await ask(played.ledger, {1: ({}, 2)})
    async with playing(played):
        episodes = await episodes_of(played.ledger, played.blobs, "train", 1, 2, every=0.01)
    assert sorted(str(episode.info["key"]) for episode in episodes) == ["train/1/1/1/box", "train/1/2/1/box"]
    assert len({str(episode.info["handle"]) for episode in episodes}) == 2  # a sandbox each
    assert sandboxes.sandboxes == {} and len(sandboxes.deleted) == 2 and await pool.leases.all() == []


async def test_a_stale_claims_sandbox_is_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes, leases=leases_of(ledger))
    await ask(ledger, {1: ({}, 1)})
    fence = await ledger.take("runners/elsewhere")  # a runner on a machine that has since died
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "elsewhere", "fence": fence.number}, fence)
    await beats.beat("elsewhere", {"places": 1})
    lease = await pool.acquire(BOX, "train/1/1/1/box")
    unclaimed = await pool.acquire(BOX, "by-hand/box")  # a run the ledger does not know: it ends when released
    assert await sweep(pool, ledger, beats) == [] and len(sandboxes.sandboxes) == 2  # its claim holds: it beats
    monkeypatch.setattr(presence, "STALE", -1.0)  # its newest beat is now too old
    assert await sweep(pool, ledger, beats) == [lease.key]
    assert sandboxes.deleted == [lease.handle] and [each.key for each in await pool.held()] == [unclaimed.key]
    assert [each.key for each in await FileLeases(ledger.directory).all()] == [unclaimed.key]  # beside the ledger


async def test_a_new_attempt_gets_a_new_sandbox_and_the_old_one_is_deleted(tmp_path: Path) -> None:
    sandboxes = FakeSandboxes()
    ledger = FileLedger(tmp_path / "ledger")
    pool = SandboxPool(sandboxes, leases=leases_of(ledger))
    await ask(ledger, {1: ({}, 1)})
    fence = await ledger.take("runners/here")  # this runner, before it was started again
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "here", "fence": fence.number}, fence)
    old = await pool.acquire(BOX, "train/1/1/1/box")
    played = episode_runner(tmp_path, pool)  # started again: its fence moves on, and its claim lapses
    async with kept(pool, ledger, None), playing(played):
        (episode,) = await episodes_of(ledger, played.blobs, "train", 1, 1, every=0.01)
        await until(lambda: _gone(sandboxes, old.handle))
    assert episode.info["key"] == "train/1/1/2/box" and episode.info["handle"] != old.handle
    assert sorted(await ledger.read(table("train", CLAIMS))) == ["1/1/1", "1/1/2"]
    assert old.handle in sandboxes.deleted and sandboxes.sandboxes == {}


async def _gone(sandboxes: FakeSandboxes, handle: str) -> bool:
    return handle not in sandboxes.sandboxes


async def test_a_runner_claims_only_what_its_pools_have_room_for(tmp_path: Path) -> None:
    sandboxes = FakeSandboxes(size=1)
    pool = SandboxPool(sandboxes)
    held = await pool.acquire(BOX, "someone-else/box")  # the pool is full
    played = episode_runner(tmp_path, pool)
    ledger = played.ledger
    await ask(ledger, {1: ({"gate": "room"}, 3)})

    async def claimed() -> list[str]:
        return sorted(await ledger.read(table("train", CLAIMS)))

    async with playing(played):
        await asyncio.sleep(0.2)
        assert await claimed() == []  # it has places, but its pool has no room
        await pool.release(held.key)
        await until(lambda: _count(claimed, 1))
        await asyncio.sleep(0.1)
        assert await claimed() == ["1/1/1"]  # one at a time: the pool holds one
        GATES.setdefault("room", asyncio.Event()).set()
        episodes = await episodes_of(ledger, played.blobs, "train", 1, 3, every=0.01)
    assert sorted(episode.number for episode in episodes) == [1, 2, 3] and sandboxes.sandboxes == {}


async def _count(read: Any, at_least: int) -> bool:
    return len(await read()) >= at_least


async def test_a_runner_serves_only_runs_whose_pools_it_has(tmp_path: Path) -> None:
    played = episode_runner(tmp_path, SandboxPool(FakeSandboxes()))
    await ask(played.ledger, {1: ({}, 1)})
    assert [each.run for each in await played.open()] == ["train"]
    played.pools = {}
    assert await played.open() == []


async def test_leases_beside_a_database_ledger_outlive_the_pool_that_made_them(tmp_path: Path) -> None:
    pytest.importorskip("rollout_durable")
    from rollout_train.database import DatabaseLedger

    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    leases = leases_of(ledger)
    assert leases is not None
    sandboxes = FakeSandboxes()
    lease = await SandboxPool(sandboxes, name="boxes@here", leases=leases).acquire(BOX, "train/1/1/1/box")
    again = SandboxPool(sandboxes, name="boxes@here", leases=leases_of(ledger))  # the same pool, started again
    assert await again.acquire(BOX, "train/1/1/1/box") == lease and len(sandboxes.made) == 1
    other = SandboxPool(FakeSandboxes(), name="boxes@there", leases=leases_of(ledger))
    with pytest.raises(RuntimeError, match="leased from the pool boxes@here"):
        await other.acquire(BOX, "train/1/1/1/box")
    await again.release(lease.key)
    assert await leases.all() == []
    ledger.close()


class Contained(Program):
    """A program with no loop of its own: a harness inside its box plays, pointed at the recorder only by the box's
    environment; the program scores what it said."""

    def __init__(self, parameters: Mapping[str, JsonValue]) -> None:
        self.word = str(parameters["word"])

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"box": SandboxSpec(kind="fake", slots=("policy",))}

    async def main(self, run: RunContext) -> None:
        played: Any = (await run.sandbox("box").call("play", {"prompt": "Say the word."})).structured
        run.reward(1.0 if played["said"] == self.word else 0.0)
        await run.emit("result", {"solved": played["said"] == self.word})


async def test_a_harness_inside_a_sandbox_reaches_the_recorder_through_its_environment(tmp_path: Path) -> None:
    async def play(sandbox: FakeSandbox, arguments: Mapping[str, JsonValue]) -> JsonValue:
        given = sandbox.environment  # all the harness is told: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
        gateway = cast(GatewayEndpoints, played.recorder).gateway
        assert gateway is not None
        transport = httpx.ASGITransport(app=create_app(gateway))
        headers = {"Authorization": f"Bearer {given['OPENAI_API_KEY']}"}
        async with httpx.AsyncClient(transport=transport, base_url=given["OPENAI_BASE_URL"], headers=headers) as client:
            body = {"model": given["OPENAI_MODEL"], "messages": [{"role": "user", "content": arguments["prompt"]}]}
            reply = (await client.post("/chat/completions", json=body)).json()
        return {"said": reply["choices"][0]["message"]["content"]}

    pool = SandboxPool(FakeSandboxes(operations={"play": play}))
    played = episode_runner(tmp_path, pool, "http://recorder")
    await ask(played.ledger, {1: ({"word": "yes"}, 1)}, Contained)
    async with playing(played):
        (episode,) = await episodes_of(played.ledger, played.blobs, "train", 1, 1, every=0.01)
    assert episode.reward == 1.0 and episode.info == {"solved": True}
    (segment,) = episode.trajectories["policy"].segments  # what the harness sampled is the slot's trajectory
    assert "".join(chr(token) for token in segment.tokens) == "user: Say the word.\nassistant: yes\n"


def boxes(directory: Path, size: int = 4) -> FakeSandboxes:
    """What a profile's `[pools]` names: a provider, made with the run's directory."""
    return FakeSandboxes(size=size)


PROFILE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"

[pools.fake]
kind = "tests.rollout_train.test_sandboxes:boxes"
size = 2
"""


async def test_an_open_profile_leases_sandboxes_from_the_pools_it_names(tmp_path: Path) -> None:
    from rollout.harness import bind
    from rollout_train.profile import Profile

    path = tmp_path / "profile.toml"
    path.write_text(PROFILE.format(directory=tmp_path / "run"))
    async with Profile.load(path).open() as platform:
        assert platform.pool_bindings == {"fake": PoolBinding(local="fake")}
        reference = ProgramReference(program=register(Boxed))
        binding = bind(reference.model_copy(update={"parameters": {}}), "policy", pools=platform.pool_bindings)
        fence = await platform.ledger.take(scope(platform.run.id))
        await plan(platform.ledger, platform.run.id, Plan(reference, binding), fence)
        record: JsonValue = {"parameters": {}, "episodes": 3, "decided": 1.0}
        await platform.ledger.append(table(platform.run.id, GROUPS), "1", record, fence)
        episodes = await episodes_of(platform.ledger, platform.blobs, platform.run.id, 1, 3, every=0.01)
        assert sorted(str(episode.info["key"]).split("/", 1)[1] for episode in episodes) == [
            "1/1/1/box", "1/2/1/box", "1/3/1/box",
        ]  # fmt: skip
        leases = leases_of(platform.ledger)
        assert leases is not None and await leases.all() == []  # kept beside the ledger, and released
        (beat,) = await FilePresence(tmp_path / "run" / "ledger").beats()
        assert beat.about["pools"] == {"fake": {"size": 2, "leased": 0, "free": 2}}
