"""The scheduler: a run asks for each group's episodes in the ledger; runners claim them, play them and record them."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout.contracts import RunEventType
from rollout.harness import ModelBinding, RecordedModel, RunBinding, agent_program
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.record import GROUPS, scope, table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import EpisodeRunner, Outcome, Plan, Record, episodes_of, events_of, plan, playing
from rollout_train.rollouts.scheduler import CLAIMS, EPISODES, INTERRUPTED, ended
from rollout_train.testing import plain_channel
from tests.rollout_train.rollouts.games import GATES, Gated, Guess

BINDING = RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))})


class Seen:
    def __init__(self) -> None:
        self.notes: list[Mapping[str, JsonValue]] = []

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        self.notes.append(event)


def runner(
    tmp_path: Path, *says: str, name: str = "here", places: int = 8, **options: Any
) -> tuple[EpisodeRunner, Recorder, Seen]:
    """A runner over the ledger and blob store in `tmp_path`, whose policy says `says` in turn, for ever."""
    recorder = Recorder({"policy": plain_channel(always=[(f"{word}\n", "stop") for word in says])})
    seen = Seen()
    played = EpisodeRunner(
        name,
        FileLedger(tmp_path / "ledger"),
        LocalRunner(recorder=recorder),
        recorder,
        FileBlobStore(tmp_path / "blobs"),
        places,
        hooks=[seen],
        every=0.02,
        **options,
    )
    return played, recorder, seen


async def ask(ledger: Ledger, run: str, groups: Mapping[int, tuple[JsonValue, int]], task: type = Guess) -> None:
    """What a run's loop writes: how its episodes are played, and each group with how many episodes it wants."""
    fence = await ledger.take(scope(run))
    await plan(ledger, run, Plan(agent_program(task), BINDING), fence)
    for number, (row, count) in groups.items():
        record: JsonValue = {"parameters": row, "episodes": count, "decided": float(number)}
        await ledger.append(table(run, GROUPS), str(number), record, fence)


@contextlib.asynccontextmanager
async def served(*runners: EpisodeRunner) -> AsyncGenerator[None]:
    async with contextlib.AsyncExitStack() as stack:
        for each in runners:
            await stack.enter_async_context(playing(each))
        yield


async def until(condition: Any) -> None:
    async with asyncio.timeout(5.0):
        while not await condition():  # noqa: ASYNC110 (the runners write)
            await asyncio.sleep(0.01)


async def test_a_groups_episodes_are_claimed_played_and_recorded_in_the_ledger(tmp_path: Path) -> None:
    played, _, seen = runner(tmp_path, "yes", "no")
    ledger, blobs = played.ledger, played.blobs
    await ask(ledger, "train", {1: ({"word": "yes"}, 2)})
    async with served(played):
        first, second = sorted(await episodes_of(ledger, blobs, "train", 1, 2, every=0.01), key=lambda e: e.reward)
    assert (first.reward, second.reward) == (0.0, 1.0) and first.outcome is second.outcome is Outcome.COMPLETED
    assert (second.run, second.group) == ("train", 1) and {first.number, second.number} == {1, 2}
    assert second.labels == {"run": "train", "group": "1", "episode": str(second.number)}
    assert second.info == {"solved": True, "saturated": True, "duration": 1} and second.trainable
    (segment,) = second.trajectories["policy"].segments
    text = "".join(chr(token) for token in segment.tokens)
    assert text == "user: Say the word.\nassistant: yes\n" and text[segment.spans[0].start :] == "yes\n"
    claims = await ledger.read(table("train", CLAIMS))
    assert sorted(claims) == ["1/1/1", "1/2/1"] and all(claim["runner"] == "here" for claim in claims.values())  # type: ignore[index]
    assert sorted(str(each["kind"]) for each in seen.notes) == ["ended", "ended", "started", "started"]
    assert await played.open() == []  # nothing is left to play


async def test_a_record_is_small_and_names_its_trajectories_and_its_runs_events(tmp_path: Path) -> None:
    played, _, _ = runner(tmp_path, "yes")
    await ask(played.ledger, "train", {1: ({"word": "yes"}, 1)})
    async with served(played):
        (episode,) = await episodes_of(played.ledger, played.blobs, "train", 1, 1, every=0.01)
    (line,) = (await played.ledger.read(table("train", EPISODES))).values()
    record = Record.from_json(line)  # type: ignore[arg-type]
    assert len(str(line)) < 2000 and record.episode.trajectories["policy"].segments == []  # its tokens are in blobs
    events = await events_of(record, played.blobs)
    assert events[0].type is RunEventType.RUN_CREATED and events[-1].type is RunEventType.RUN_COMPLETED
    (span,) = episode.trajectories["policy"].segments[0].spans
    sampled = [event for event in events if event.type is RunEventType.EFFECT_REQUESTED]
    assert span.effect_id in {str(event.payload["effect_id"]) for event in sampled}  # type: ignore[index]
    assert record.sampled == {"policy": span.end - span.start}


async def test_a_runner_plays_as_many_as_it_has_places_oldest_group_first(tmp_path: Path) -> None:
    played, _, _ = runner(tmp_path, "yes", places=6)
    ledger = played.ledger
    await ask(ledger, "train", {1: ({"word": "yes", "gate": "a"}, 4), 2: ({"word": "yes", "gate": "b"}, 4)}, Gated)

    async def claimed() -> list[str]:
        return sorted(await ledger.read(table("train", CLAIMS)))

    async with served(played):
        await until(lambda: _count(claimed, 6))
        await asyncio.sleep(0.05)
        assert await claimed() == ["1/1/1", "1/2/1", "1/3/1", "1/4/1", "2/1/1", "2/2/1"]  # no more than its places
        GATES.setdefault("a", asyncio.Event()).set()
        assert len(await episodes_of(ledger, played.blobs, "train", 1, 4, every=0.01)) == 4
        await until(lambda: _count(claimed, 8))  # as the first group's episodes end, the rest of the second start
        GATES.setdefault("b", asyncio.Event()).set()
        episodes = await episodes_of(ledger, played.blobs, "train", 2, 4, every=0.01)
    assert sorted(episode.number for episode in episodes) == [1, 2, 3, 4]


async def _count(read: Any, at_least: int) -> bool:
    return len(await read()) >= at_least


async def test_a_run_that_fails_is_an_episode_too_and_a_guard_holds_claims_back(tmp_path: Path) -> None:
    played, _, _ = runner(tmp_path, "yes")
    await ask(
        played.ledger, "train", {1: ({"word": "yes", "broken": True}, 2), 2: ({"word": "yes", "broken": "at once"}, 1)}
    )
    async with served(played):
        failed = [
            *await episodes_of(played.ledger, played.blobs, "train", 1, 2, every=0.01),
            *await episodes_of(played.ledger, played.blobs, "train", 2, 1, every=0.01),
        ]
    for episode in failed:  # failed in its setup, or before it began
        assert episode.outcome is Outcome.FAILED and "RuntimeError: this row cannot be" in str(episode.detail)
        assert not episode.trainable and episode.reward == 0.0

    def full() -> None:
        raise MemoryError("the machine has no memory left for more runs")

    other = tmp_path / "other"
    guarded, _, _ = runner(other, "yes", guard=full)
    await ask(guarded.ledger, "train", {1: ({"word": "yes"}, 1)})
    async with served(guarded):
        await asyncio.sleep(0.1)
    assert await guarded.ledger.read(table("train", CLAIMS)) == {}  # not now: it waits for room


async def test_an_episode_its_runner_cut_short_is_claimed_again_and_played_once(tmp_path: Path) -> None:
    first, _, _ = runner(tmp_path, "yes")
    ledger = first.ledger
    await ask(ledger, "train", {1: ({"word": "yes", "gate": "closing"}, 1)}, Gated)
    async with served(first):
        await until(lambda: _count(lambda: ledger.read(table("train", CLAIMS)), 1))
    assert sorted(await ledger.read(table("train", INTERRUPTED))) == ["1/1/1"]  # noted when its runner closed
    assert await ended(ledger, first.blobs, "train", 1, 1) is None

    again, _, _ = runner(tmp_path, "yes")  # the same runner, started again: its fence moves on
    GATES.setdefault("closing", asyncio.Event()).set()
    async with served(again):
        (episode,) = await episodes_of(ledger, again.blobs, "train", 1, 1, every=0.01)
    assert episode.outcome is Outcome.COMPLETED
    assert sorted(await ledger.read(table("train", CLAIMS))) == ["1/1/1", "1/1/2"]


async def test_a_claim_holds_only_while_its_runner_keeps_its_fence(tmp_path: Path) -> None:
    played, _, _ = runner(tmp_path, "yes", name="elsewhere")
    ledger = played.ledger
    await ask(ledger, "train", {1: ({"word": "yes"}, 1)})
    fence = await ledger.take("runners/elsewhere")  # a runner that claimed and then died, as far as anyone can tell
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "elsewhere", "fence": fence.number}, fence)
    assert await played.open() == []  # held: its runner's fence is the one it claimed under
    await ledger.take("runners/elsewhere")  # it is started again
    (open,) = await played.open()
    assert (open.group, open.number, open.attempt) == (1, 1, 2)


async def test_runners_share_the_work_and_never_play_one_attempt_twice(tmp_path: Path) -> None:
    one, _, seen_one = runner(tmp_path, "yes", name="one", places=2)
    two, _, seen_two = runner(tmp_path, "yes", name="two", places=2)
    await ask(one.ledger, "train", {1: ({"word": "yes"}, 6)})
    async with served(one, two):
        episodes = await episodes_of(one.ledger, one.blobs, "train", 1, 6, every=0.01)
    assert sorted(episode.number for episode in episodes) == [1, 2, 3, 4, 5, 6]
    claims = await one.ledger.read(table("train", CLAIMS))
    assert len(claims) == 6 and all(key.endswith("/1") for key in claims)  # each attempt once
    ended_by = [str(note["runner"]) for note in (*seen_one.notes, *seen_two.notes) if note["kind"] == "ended"]
    assert sorted(set(ended_by)) == ["one", "two"]


async def test_a_runner_plays_only_the_runs_it_can_serve(tmp_path: Path) -> None:
    played, _, _ = runner(tmp_path, "yes", runs=["mine"])
    ledger = played.ledger
    await ask(ledger, "theirs", {1: ({"word": "yes"}, 1)})
    await ask(ledger, "mine", {1: ({"word": "yes"}, 1)})
    fence = await ledger.take(scope("elsewhere"))
    other = RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="a-channel-it-lacks"))})
    await plan(ledger, "elsewhere", Plan(agent_program(Guess), other), fence)
    await ledger.append(table("elsewhere", GROUPS), "1", {"parameters": {"word": "yes"}, "episodes": 1}, fence)
    assert [(each.run, each.group) for each in await played.open()] == [("mine", 1)]
    played.runs = None
    assert sorted(each.run for each in await played.open()) == ["mine", "theirs"]  # not the run it has no channel for
