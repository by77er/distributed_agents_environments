"""Rollout jobs: rows in, a stream of labelled episodes out, read with a cursor while runs are still going."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.contracts import RunEventType
from rollout.harness import ModelBinding, RecordedModel, RunBinding, agent_program
from rollout.local import LocalRunner
from rollout_train.recorder import Recorder
from rollout_train.rollouts import Episode, JobHooks, Outcome, Refused, RolloutJob, RolloutJobs, events_of
from rollout_train.testing import plain_channel
from tests.rollout_train.rollouts.games import GATES, Gated, Guess

BINDING = RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))})


class Seen(JobHooks):
    def __init__(self) -> None:
        self.events: list[Mapping[str, JsonValue]] = []

    def on_job(self, event: Mapping[str, JsonValue]) -> None:
        self.events.append(event)


def jobs(*says: str, log: Path | None = None, **options: object) -> tuple[RolloutJobs, Recorder, Seen]:
    """Jobs over a runner whose policy says `says` in turn, for ever."""
    recorder = Recorder({"policy": plain_channel(always=[(f"{word}\n", "stop") for word in says])})
    seen = Seen()
    return RolloutJobs(LocalRunner(recorder=recorder), recorder, log=log, hooks=[seen], **options), recorder, seen  # type: ignore[arg-type]


async def start(rollouts: RolloutJobs, task: type = Guess, in_flight: int = 8, name: str = "") -> RolloutJob:
    return await rollouts.start(program=agent_program(task), binding=BINDING, in_flight=in_flight, name=name)


async def test_a_ticket_yields_its_episodes_with_labels_rewards_results_and_what_was_sampled() -> None:
    rollouts, _, seen = jobs("yes", "no")
    job = await start(rollouts)
    ticket = await job.run({"word": "yes"}, labels={"group": "g1", "task": "say-yes"}, count=2)
    first, second = sorted(await ticket.episodes(), key=lambda episode: episode.reward)
    assert (first.reward, second.reward) == (0.0, 1.0) and first.outcome is second.outcome is Outcome.COMPLETED
    assert second.labels["group"] == "g1" and second.labels["task"] == "say-yes" and second.ticket == ticket.id
    assert second.info == {"solved": True, "saturated": True, "duration": 1} and second.parameters == {"word": "yes"}
    assert second.trainable and {first.cursor, second.cursor} == {1, 2}
    (epoch,) = second.traces["policy"].epochs
    text = "".join(chr(token) for token in epoch.tokens)
    assert text == "user: Say the word.\nassistant: yes\n" and text[epoch.spans[0].start :] == "yes\n"
    status = await job.status()
    assert (status.queued, status.running, status.finished, status.acknowledged) == (0, 0, 2, 0)
    kinds = [event["kind"] for event in seen.events]
    assert kinds == ["ticket", "admitted", "episode", "episode"]
    assert seen.events[-1]["labels"] == {
        "group": "g1",
        "task": "say-yes",
        "job": job.id,
        "ticket": ticket.id,
        "episode": "2",
    }
    await rollouts.close()


async def test_the_stream_is_read_with_a_cursor_while_runs_are_still_going() -> None:
    rollouts, _, _ = jobs("yes")
    job = await start(rollouts, Gated)
    held = await job.run({"word": "yes", "gate": "held"}, labels={"group": "slow"})
    quick = await job.run({"word": "yes", "gate": "open"}, labels={"group": "quick"})
    GATES.setdefault("open", asyncio.Event()).set()
    stream = job.episodes()
    first = await anext(stream)  # the quick one has ended; the slow one is still running
    assert first.labels["group"] == "quick" and (await job.status()).running == 1
    await job.acknowledge(first.cursor)
    GATES.setdefault("held", asyncio.Event()).set()
    second = await anext(stream)
    assert second.labels["group"] == "slow" and second.cursor == first.cursor + 1
    assert [episode.cursor async for episode in _until_closed(job, rollouts)] == [second.cursor]  # 1 was acknowledged
    assert await held.episodes() == [second] and len(await quick.episodes()) == 1


async def _until_closed(job: RolloutJob, rollouts: RolloutJobs, cursor: int = 0) -> AsyncIterator[Episode]:
    await rollouts.close()
    async for episode in job.episodes(cursor):
        yield episode


async def test_a_ticket_starts_when_there_is_room_for_all_of_it() -> None:
    rollouts, _, seen = jobs("yes")
    job = await start(rollouts, Gated, in_flight=5)
    one = await job.run({"word": "yes", "gate": "a"}, labels={"group": "1"}, count=4)
    two = await job.run({"word": "yes", "gate": "b"}, labels={"group": "2"}, count=4)
    await asyncio.sleep(0.05)
    assert [(s.queued, s.running) for s in [await job.status()]] == [(4, 4)]  # the second group waits, whole
    GATES.setdefault("a", asyncio.Event()).set()
    assert len(await one.episodes()) == 4  # and starts once the first has ended (three done would do: 1 + 4 <= 5)
    await asyncio.sleep(0.05)
    assert (await job.status()).running == 4
    GATES.setdefault("b", asyncio.Event()).set()
    assert len(await two.episodes()) == 4
    assert [event["kind"] for event in seen.events].count("admitted") == 2
    await rollouts.close()


async def test_a_run_that_fails_is_an_episode_too_and_a_refused_ticket_tells_its_caller() -> None:
    rollouts, _, _ = jobs("yes")
    job = await start(rollouts)
    ticket = await job.run({"word": "yes", "broken": True}, labels={"group": "g"}, count=2)
    never = await job.run({"word": "yes", "broken": "at once"}, labels={"group": "h"})
    for episode in [*await ticket.episodes(), *await never.episodes()]:  # failed in its setup, or before it began
        assert episode.outcome is Outcome.FAILED and "RuntimeError: this row cannot be" in str(episode.detail)
        assert not episode.trainable and episode.reward == 0.0 and episode.labels["group"] in ("g", "h")
    assert (await job.status()).finished == 3 and (await job.status()).running == 0
    await rollouts.close()

    def full() -> None:
        raise MemoryError("the machine has no memory left for more runs")

    guarded, _, _ = jobs("yes", guard=full)
    refused = await (await start(guarded)).run({"word": "yes"})
    with pytest.raises(Refused, match="MemoryError: the machine has no memory left"):
        await refused.episodes()
    await guarded.close()


async def test_weights_published_through_the_job_mark_what_is_sampled_afterwards() -> None:
    rollouts, recorder, seen = jobs("yes")
    job = await start(rollouts)
    (before,) = await (await job.run({"word": "yes"})).episodes()
    assert await job.publish("policy", "step-1", "/adapters/step-1") == 1
    await job.note("update", {"kl_moved": 0.01})
    (after,) = await (await job.run({"word": "yes"})).episodes()
    versions = [episode.traces["policy"].epochs[0].spans[0].version for episode in (before, after)]
    assert versions == [0, 1] and recorder.channels["policy"].adapter == "step-1"
    assert [event["kind"] for event in seen.events if event["kind"] in ("published", "update")] == [
        "published",
        "update",
    ]
    await rollouts.close()


async def test_a_job_with_a_log_keeps_every_episode_and_can_be_read_again_from_any_cursor(tmp_path: Path) -> None:
    rollouts, _, _ = jobs("yes", "no", log=tmp_path)
    job = await start(rollouts, name="main")
    episodes = await (await job.run({"word": "yes"}, labels={"group": "g"}, count=3)).episodes()
    await job.acknowledge(episodes[1].cursor)
    await rollouts.close()
    lines = (tmp_path / "main" / "episodes.jsonl").read_text().splitlines()
    assert [json.loads(line)["episode"]["cursor"] for line in lines] == [1, 2, 3]
    assert all(len(line) < 2000 for line in lines)  # a record is small: its tokens are in the blob store
    assert (tmp_path / "main" / "acknowledged").read_text() == "2"

    again, _, _ = jobs("yes", log=tmp_path)  # another process, later
    resumed = await start(again, name="main")
    status = await resumed.status()
    assert (status.finished, status.acknowledged) == (3, 2)
    kept = [episode async for episode in _until_closed(resumed, again, cursor=0)]
    assert kept == episodes  # all three, acknowledged or not: tokens, spans, logprobs and all
    (later,) = [episode async for episode in resumed.episodes(2)]
    assert later == episodes[2]

    # A record also names its run's events, which a span's effect joins its trace to.
    assert again.blobs is not None
    record = resumed.after(2)[0]
    events = await events_of(record, again.blobs)
    assert events[0].type is RunEventType.RUN_CREATED and events[-1].type is RunEventType.RUN_COMPLETED
    (span,) = later.traces["policy"].epochs[0].spans
    sampled = [event for event in events if event.type is RunEventType.EFFECT_REQUESTED]
    assert span.effect_id in {str(event.payload["effect_id"]) for event in sampled}  # type: ignore[index]
    assert record.sampled == {"policy": span.end - span.start}


async def test_a_job_with_no_log_holds_episodes_until_they_are_acknowledged() -> None:
    rollouts, _, _ = jobs("yes")
    job = await start(rollouts)
    episodes = await (await job.run({"word": "yes"}, count=2)).episodes()
    assert [record.episode.cursor for record in job.after(0)] == [1, 2]
    await job.acknowledge(1)
    assert [record.episode.cursor for record in job.after(0)] == [2] and await job.episode(job.after(0)[0]) == episodes[
        1
    ]
    await rollouts.close()


async def test_a_job_started_again_under_its_name_takes_the_place_of_the_one_before(tmp_path: Path) -> None:
    cancels: list[str] = []

    class Watched(LocalRunner):
        async def cancel(self, run_id: str, *, reason: str) -> None:
            cancels.append(run_id)
            await super().cancel(run_id, reason=reason)

    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop")])})
    rollouts = RolloutJobs(Watched(recorder=recorder), recorder, log=tmp_path)
    first = await start(rollouts, task=Gated, name="main")
    waiting = await first.run({"word": "yes", "gate": "later"}, key="group-1")
    await asyncio.sleep(0.05)
    second = await start(rollouts, name="main", task=Gated)  # the first is closed, and its run with it
    with pytest.raises(Refused, match="closed before the ticket was over"):
        await waiting.episodes()
    assert rollouts.job("main") is second and len(cancels) == 1  # in the runner too: it is not left running
    (cut,) = second.after(0)
    assert cut.episode.outcome is Outcome.CANCELLED and cut.episode.run_id == cancels[0]

    # What was asked for is still owed: the new job runs it again, and asking again by its key finds it.
    again = await second.run({"word": "yes", "gate": "later"}, key="group-1")
    await asyncio.sleep(0.05)
    assert again.id == waiting.id == "t_group-1" and (await second.status()).running == 1
    GATES["later"].set()
    (episode,) = await again.episodes()
    assert episode.outcome is Outcome.COMPLETED and episode.cursor == 2  # the log is one log
    assert len((tmp_path / "main" / "tickets.jsonl").read_text().splitlines()) == 1  # asked for once
    await rollouts.close()


async def test_a_ticket_can_be_read_again_until_its_episodes_are_acknowledged() -> None:
    rollouts, _, _ = jobs("yes")
    job = await start(rollouts)
    ticket = await job.run({"word": "yes"}, count=2)
    assert await ticket.ready(5.0) and len(await ticket.episodes()) == 2
    assert [e.cursor for e in await job.ticket(ticket.id).episodes()] == [1, 2]  # by its id, as a service finds it
    await job.acknowledge(2)
    with pytest.raises(KeyError):
        job.ticket(ticket.id)
    await rollouts.close()
