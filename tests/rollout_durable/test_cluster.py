"""Several runners, each its own process, share one Postgres: messages through any of them reach the right run, and
runs survive their runner being killed, evicted and woken anywhere (docs/durability/runners.md)."""

import asyncio
import random
from collections import Counter
from pathlib import Path

import pytest
from cluster import Cluster, until

from rollout.contracts import RunEventType

CHAT = "scenarios/chat"


def gapless(cluster: Cluster, run_id: str) -> bool:
    events = cluster.events(run_id)
    return [e.seq for e in events] == list(range(len(events))) and sum(
        e.type is RunEventType.RUN_CREATED for e in events
    ) == 1


def outcome(cluster: Cluster, run_id: str) -> str | None:
    record = cluster.store.run(run_id)
    return record.status if record is not None and record.outcome is not None else None


async def test_messages_through_any_runner_reach_one_run_in_order(tmp_path: Path, postgres: str) -> None:
    async with Cluster(tmp_path, postgres, 3) as cluster:
        for index in range(6):  # each through a different runner
            await cluster.call(index % 3, "send", address=f"{CHAT}/c1", text=f"m{index}", key=f"c1-m{index}")
        (run_id,) = cluster.conversation_runs(f"{CHAT}/c1")
        await until(lambda: len(cluster.replies(run_id)) == 6)
        assert cluster.replies(run_id) == [f"reply to m{index}" for index in range(6)]
        assert cluster.received(run_id) == [f"c1-m{index}" for index in range(6)]

        # A new conversation messaged through every runner at once starts one run, which gets every message.
        await asyncio.gather(
            *(
                cluster.call(index, "send", address=f"{CHAT}/c2", text=f"hello {index}", key=f"c2-{index}")
                for index in range(3)
            )
        )
        (run_id,) = cluster.conversation_runs(f"{CHAT}/c2")
        await until(lambda: len(cluster.replies(run_id)) == 3)
        assert sorted(cluster.received(run_id)) == ["c2-0", "c2-1", "c2-2"]

        # A retried message is delivered once, whichever runner the retry goes through.
        await asyncio.gather(
            *(cluster.call(index, "send", address=f"{CHAT}/c2", text="again", key="c2-again") for index in range(3))
        )
        await until(lambda: len(cluster.replies(run_id)) == 4)
        await asyncio.sleep(1)
        assert cluster.received(run_id).count("c2-again") == 1
        assert gapless(cluster, run_id)


async def test_the_runs_of_a_killed_runner_are_taken_over(tmp_path: Path, postgres: str) -> None:
    async with Cluster(tmp_path, postgres, 3, takeover_after=3) as cluster:
        await cluster.call(0, "start", program="counting", run_id="r_takeover")
        await until(lambda: len(cluster.ledger("models")) >= 2)  # the second model call is in flight
        cluster.kill(0)  # and it stays dead: another runner must take the run over
        calls_before = [entry["effect_id"] for entry in cluster.ledger("models")]

        await until(lambda: outcome(cluster, "r_takeover") is not None, seconds=90)
        assert outcome(cluster, "r_takeover") == "completed"
        models = cluster.ledger("models")
        calls = Counter(entry["effect_id"] for entry in models)
        assert len(calls) == 5  # one model effect per turn
        assert [effect for effect, count in calls.items() if count > 1] in ([], [calls_before[-1]])
        assert {entry["runner"] for entry in models[len(calls_before) :]} <= {"runner-1", "runner-2"}
        assert cluster.replies("r_takeover") == [f"reply to turn {turn}" for turn in range(1, 6)]
        assert gapless(cluster, "r_takeover")


async def test_a_restarted_runner_releases_its_runs_at_once(tmp_path: Path, postgres: str) -> None:
    """A runner restarted with its id puts its runs back on the queue as it starts, long before a takeover would;
    whichever runner dequeues each one continues it."""
    async with Cluster(tmp_path, postgres, 2, takeover_after=600) as cluster:  # no takeover in this test
        await cluster.call(0, "start", program="counting", run_id="r_restart")
        await until(lambda: len(cluster.ledger("models")) >= 2)
        calls_before = [entry["effect_id"] for entry in cluster.ledger("models")]
        cluster.kill(0)
        await cluster.start(0)
        await until(lambda: outcome(cluster, "r_restart") is not None, seconds=30)
        assert outcome(cluster, "r_restart") == "completed"
        calls = Counter(entry["effect_id"] for entry in cluster.ledger("models"))
        assert len(calls) == 5
        assert [effect for effect, count in calls.items() if count > 1] in ([], [calls_before[-1]])
        assert gapless(cluster, "r_restart")


async def test_a_command_interrupted_by_a_crash_is_not_run_again_by_another_runner(
    tmp_path: Path, postgres: str
) -> None:
    async with Cluster(tmp_path, postgres, 2, takeover_after=3) as cluster:
        await cluster.call(0, "start", program="deploy", run_id="r_guarded")
        await until(lambda: any(entry["operation"] == "execute" for entry in cluster.ledger("environments")))
        cluster.kill(0)  # while the command runs
        await until(lambda: outcome(cluster, "r_guarded") is not None, seconds=90)
        operations = Counter(entry["operation"] for entry in cluster.ledger("environments"))
        assert operations == {"create": 1, "execute": 1, "destroy": 1}
        assert cluster.replies("r_guarded") == ["outcome unknown"]
        destroyed_by = [entry["runner"] for entry in cluster.ledger("environments") if entry["operation"] == "destroy"]
        assert destroyed_by == ["runner-1"]


async def test_an_evicted_run_wakes_on_a_message_through_another_runner(tmp_path: Path, postgres: str) -> None:
    async with Cluster(tmp_path, postgres, 3, evict_after=0.5) as cluster:
        await cluster.call(0, "send", address=f"{CHAT}/c3", text="m0", key="c3-m0")
        (run_id,) = cluster.conversation_runs(f"{CHAT}/c3")
        await until(lambda: cluster.replies(run_id) == ["reply to m0"])
        await until(lambda: cluster.store.is_evicted(run_id))
        resident = [await cluster.call(index, "resident") for index in range(3)]
        assert all(run_id not in runs for runs in resident)  # unloaded everywhere

        await cluster.call(2, "send", address=f"{CHAT}/c3", text="m1", key="c3-m1")
        await until(lambda: cluster.replies(run_id) == ["reply to m0", "reply to m1"])
        assert len(cluster.ledger("models")) == 2  # waking replayed the first turn without calling the model
        await until(lambda: cluster.store.evictions() >= 2)  # and it was evicted again, wherever it woke
        assert gapless(cluster, run_id)

        # A run evicted while waiting with a timeout is woken at its deadline by some runner, and ends.
        await cluster.call(1, "send", address="scenarios/shortwait/c4", text="hi", key="c4-hi")
        (short,) = cluster.conversation_runs("scenarios/shortwait/c4")
        await until(lambda: outcome(cluster, short) is not None, seconds=30)
        assert outcome(cluster, short) == "completed"


async def test_cancelling_an_evicted_run_through_another_runner_tears_it_down(tmp_path: Path, postgres: str) -> None:
    async with Cluster(tmp_path, postgres, 2, evict_after=0.5) as cluster:
        await cluster.call(0, "send", address=f"{CHAT}/c5", text="hi", key="c5-hi")
        (run_id,) = cluster.conversation_runs(f"{CHAT}/c5")
        await until(lambda: cluster.store.is_evicted(run_id))
        await cluster.call(1, "cancel", run_id=run_id)
        assert outcome(cluster, run_id) == "cancelled"
        assert [entry["run_id"] for entry in cluster.ledger("teardowns")] == [run_id]


@pytest.mark.parametrize("seed", [1, 2])
async def test_messages_racing_eviction_are_never_lost(tmp_path: Path, postgres: str, seed: int) -> None:
    """Eviction and delivery race on every runner: every message is answered once, in order."""
    rng = random.Random(seed)
    async with Cluster(tmp_path, postgres, 3, evict_after=0.2) as cluster:
        for index in range(15):
            await cluster.call(rng.randrange(3), "send", address=f"{CHAT}/race", text=f"m{index}", key=f"race-{index}")
            await asyncio.sleep(rng.uniform(0, 0.6))  # sometimes before eviction, sometimes after
        (run_id,) = cluster.conversation_runs(f"{CHAT}/race")
        await until(
            lambda: len(cluster.replies(run_id)) == 15, seconds=90, message=f"replies: {cluster.replies(run_id)}"
        )
        assert cluster.received(run_id) == [f"race-{index}" for index in range(15)]
        assert cluster.replies(run_id) == [f"reply to m{index}" for index in range(15)]
        assert len(cluster.ledger("models")) == 15  # no model call repeated by any replay
        assert cluster.store.evictions() > 0
        assert gapless(cluster, run_id)
