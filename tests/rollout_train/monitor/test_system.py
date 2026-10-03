"""Where a run stands, read from its directory: the monitor's system view."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import FileLedger, Policies, train
from rollout_train.layout import BLOBS, FEED, JOBS, LEDGER
from rollout_train.monitor import FeedReader, RunFeed, System
from rollout_train.monitor.system import DECIDED, ENDED, MADE, PLAYING, STEPPING, WAITING
from rollout_train.record import GROUPS, ITERATIONS, STEPS, scope, table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import RolloutJobs
from rollout_train.rollouts.episodes import Episode, Outcome, Record, Trajectory
from rollout_train.rollouts.jobs import EPISODES, INTERRUPTED, TICKETS
from rollout_train.testing import plain_channel
from tests.rollout_train.rollouts.games import Words
from tests.rollout_train.training.test_loop import Counting


async def test_a_run_that_trained_is_shown_as_its_ledger_its_log_and_its_feed_have_it(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from rollout_train.monitor.app import create_app

    feed = RunFeed(tmp_path / FEED)
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    blobs = FileBlobStore(tmp_path / BLOBS)
    runner = LocalRunner(recorder=recorder, hooks=[feed])
    jobs = RolloutJobs(runner, recorder, log=tmp_path / JOBS, blobs=blobs, hooks=[feed])
    policies = Policies(FileLedger(tmp_path / LEDGER), blobs)
    await train(
        jobs, Words(), Counting(), policies, policy="words", channel="policy", directory=tmp_path / "versions",
        groups=3, seed=1,
    )  # fmt: skip
    await jobs.close()
    feed.close()

    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        system = (await client.get("/api/system")).json()
        group = (await client.get("/api/groups/train/1")).json()
        assert "monitor.js" in (await client.get("/")).text and (await client.get("/monitor.js")).status_code == 200
        assert (await client.get("/api/groups/train/9")).status_code == 404
        episode = (await client.get(f"/api/episodes/{group['episodes'][0]['run_id']}")).json()
    assert group["number"] == 1 and group["stage"] == "done" and group["outcome"]["iteration"] == 1
    assert len(group["episodes"]) == 4 and all(
        each["state"] == "completed" and "info" in each for each in group["episodes"]
    )
    assert (
        episode["source"] == "feed"
        and episode["labels"]["iteration"] == "1"
        and episode["ended"]["state"] == "completed"
    )
    assert any(line["kind"] == "sample" and line["messages"] for line in episode["lines"])

    # Once the feed has let an episode go, its replies and tool calls are read back from the events the job kept.
    kept = await System(tmp_path, FeedReader(tmp_path / "elsewhere")).episode(episode["run_id"])
    samples = [line for line in kept["lines"] if line["kind"] == "sample"]
    assert kept["source"] == "archive" and samples and all(not each["messages"] and each["reply"] for each in samples)
    assert [line["reply"] for line in samples] == [
        line["reply"] for line in episode["lines"] if line["kind"] == "sample"
    ]
    (run,) = system["runs"]
    assert run["run"] == "train" and run["fence"] == 1 and run["decided"] == 3 and run["open"] == []
    assert [line["iteration"] for line in run["iterations"]] == [1, 2, 3]
    trained = [line for line in run["iterations"] if line["update"]]
    assert trained  # (the policy says yes and no in turn: some group has something to compare)

    (policy,) = system["policies"]
    assert policy["policy"] == "words" and policy["fence"] == 1 and policy["head"] == trained[-1]["adapter"]
    assert [version["name"] for version in policy["versions"]] == [line["adapter"] for line in trained]
    first = policy["versions"][0]
    assert (
        first["parent"] is None and first["weights"]["files"] == 1 and first["state"]["files"] == 1 and first["batch"]
    )

    (job,) = system["jobs"]
    assert job == {**job, "job": "train", "tickets": 3, "episodes": 12, "outcomes": {"completed": 12}}
    assert job["last"] == job["acknowledged"] == 12 and job["sampled"] > 0
    (channel,) = system["channels"]
    assert channel["channel"] == "policy" and channel["adapter"] == policy["head"]
    assert system["ledger"]["fences"] == {scope("train"): 1, "policies/words": 1}
    assert system["ledger"]["tables"][table("train", GROUPS)] == 3
    assert system["kept"]["versions"] > 0 and system["kept"]["episodes"] > 0
    assert system["machine"]["now"]["disk"]["total"] > 0 and system["written"] <= system["at"]


async def test_a_group_in_flight_is_at_the_stage_a_loop_starting_now_would_find_it_at(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    policies = Policies(ledger, FileBlobStore(tmp_path / BLOBS))
    fence = await ledger.take(scope("train"))
    writer = await policies.writer("miner")
    feed = RunFeed(tmp_path / FEED)
    system = System(tmp_path, FeedReader(tmp_path / FEED))
    log = tmp_path / JOBS / "train"
    log.mkdir(parents=True)

    async def group() -> dict[str, Any]:
        (run,) = (await system.snapshot())["runs"]
        (open_group,) = run["open"]
        return open_group

    def ended(cursor: int, run_id: str, detail: str | None = None) -> None:
        outcome = Outcome.CANCELLED if detail else Outcome.COMPLETED
        labels = {"episode": str(cursor), "ticket": "t_train-0001"}
        trajectories = {"ada": Trajectory([], {"default": float(cursor)})}
        episode = Episode(
            cursor, "train", "t_train-0001", run_id, labels, None, outcome, detail, trajectories=trajectories
        )
        with (log / EPISODES).open("a") as file:
            file.write(json.dumps(Record(episode, sampled={"ada": 40}).to_json()) + "\n")

    await ledger.append(table("train", GROUPS), "1", {"task": "t003", "title": "chests", "decided": 5.0}, fence)
    assert (await group()) == {**(await group()), "number": 1, "task": "t003", "stage": DECIDED, "ticket": None}

    ticket = {"id": "t_train-0001", "parameters": None, "labels": {"iteration": "1"}, "count": 2}
    (log / TICKETS).write_text(json.dumps(ticket) + "\n")
    assert (await group())["stage"] == WAITING

    created: JsonValue = {"labels": {"ticket": "t_train-0001", "episode": "1"}}
    feed._write("r_one", {"kind": "event", "type": "run.created", "at": 6.0, "payload": created})  # pyright: ignore[reportPrivateUsage]
    feed._write("r_one", {"kind": "sample", "slot": "ada", "at": 7.0})  # pyright: ignore[reportPrivateUsage]
    playing = await group()
    assert playing["stage"] == PLAYING and playing["ended"] == 0
    (episode,) = playing["episodes"]
    assert episode == {**episode, "run_id": "r_one", "episode": "1", "state": "running", "samples": 1, "in_feed": True}

    ended(1, "r_one")
    ended(2, "r_cut", INTERRUPTED)  # (a run a closed job left: it is run again, and does not count)
    playing = await group()
    assert playing["stage"] == PLAYING and playing["ended"] == 1
    assert [(each["run_id"], each["state"], each["reward"]) for each in playing["episodes"]] == [
        ("r_one", "completed", 1.0),
        ("r_cut", "cancelled", 2.0),
    ]
    ended(3, "r_two")
    assert (await group())["stage"] == ENDED

    step: dict[str, JsonValue] = {
        "policy": "miner",
        "parent": None,
        "number": 1,
        "segments": 8,
        "batch": {"uri": "…"},
        "seed": 1,
    }
    await ledger.append(table("train", STEPS), "1", step, fence)
    stepping = await group()
    assert stepping["stage"] == STEPPING
    assert stepping["step"] == {**{key: value for key, value in step.items() if key != "batch"}, "makes": "miner@1"}

    weights = tmp_path / "adapter.bin"
    weights.write_text("weights")
    await policies.add(writer, "miner", 1, weights=weights)
    assert (await group())["stage"] == MADE

    outcome: JsonValue = {"iteration": 1, "task": "t003", "failures": ["x", "x"]}
    await ledger.append(table("train", ITERATIONS), "1", outcome, fence)
    (run,) = (await system.snapshot())["runs"]
    (done,) = run["iterations"]
    assert run["open"] == [] and {key: done[key] for key in ("iteration", "task", "failures")} == {
        "iteration": 1,
        "task": "t003",
        "failures": ["x"],
    }
    assert [(each["run_id"], each["interrupted"]) for each in done["episodes"]] == [
        ("r_one", False),
        ("r_cut", True),
        ("r_two", False),
    ]
    feed.close()


async def test_a_directory_that_is_no_run_has_nothing_to_show_and_is_left_as_it_is(tmp_path: Path) -> None:
    system = await System(tmp_path, FeedReader(tmp_path / FEED)).snapshot()
    assert system["runs"] == system["policies"] == system["jobs"] == system["channels"] == []
    assert system["written"] is None and system["processes"] is None
    assert not (tmp_path / LEDGER).exists()  # (reading makes no ledger)
