"""Where a run stands, read from its ledger and its directory: the monitor's system view."""

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import Budget, Checkpoint, FileLedger, Step, Versions, Weighted, train
from rollout_train.layout import BLOBS, FEED, LEDGER
from rollout_train.ledger import Ledger
from rollout_train.monitor import FeedReader, RunFeed, System
from rollout_train.monitor.system import DONE, ENDED, PLAYING, WAITING
from rollout_train.record import GROUPS, RESULTS, STARTS, STEPS, scope, table
from rollout_train.recorder import Recorder
from rollout_train.registry import registry_of
from rollout_train.rollouts import EpisodeRunner, playing
from rollout_train.rollouts.episodes import Episode, Outcome, Record, Trajectory
from rollout_train.rollouts.scheduler import CLAIMS, CLOSED, EPISODES, INTERRUPTED, runner_scope
from rollout_train.testing import plain_channel
from rollout_train.trainer import STATE, WEIGHTS
from tests.rollout_train.rollouts.games import Words


class Trains:
    """A trainer that trains nothing and leaves files as a trainer would."""

    budget = Budget(segments=3)

    def __init__(self) -> None:
        self.steps = 0

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step:
        self.steps += 1
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"weights after {self.steps} steps")
        (into / STATE).mkdir()
        (into / STATE / "optimizer.bin").write_text(f"moments after {self.steps} steps")
        return Step({"segments": float(len(batch))})


async def test_a_run_that_trained_is_shown_as_its_ledger_and_its_feed_have_it(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from rollout_train.monitor.app import create_app

    feed = RunFeed(tmp_path / FEED)
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    blobs = FileBlobStore(tmp_path / BLOBS)
    ledger = FileLedger(tmp_path / LEDGER)
    versions = Versions(ledger, blobs)
    local = LocalRunner(recorder=recorder, hooks=[feed])
    runner = EpisodeRunner("here", ledger, local, recorder, blobs, places=4, hooks=[feed], every=0.05)
    async with playing(runner):
        await train(
            Words(), Trains(), versions, base="tiny", channel="policy", directory=tmp_path / "versions",
            publish=recorder.publish, groups=3, groups_per_step=1, seed=1, hooks=[feed],
            started={"directory": str(tmp_path)},
        )  # fmt: skip
    await local.close()
    feed.close()
    registry = registry_of(ledger)
    assert registry is not None
    head = await versions.head("train")
    assert head is not None
    await registry.bookmark("best", head.id)

    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        system = (await client.get("/api/system")).json()
        group = (await client.get("/api/groups/train/1")).json()
        assert "monitor.js" in (await client.get("/")).text and (await client.get("/monitor.js")).status_code == 200
        assert (await client.get("/api/groups/train/9")).status_code == 404
        episode = (await client.get(f"/api/episodes/{group['episodes'][0]['run_id']}")).json()
    assert group["number"] == 1 and group["stage"] == "done" and group["outcome"]["group"] == 1
    assert len(group["episodes"]) == 4 and all(
        each["state"] == "completed" and "info" in each for each in group["episodes"]
    )
    assert (
        episode["source"] == "feed" and episode["labels"]["group"] == "1" and episode["ended"]["state"] == "completed"
    )
    assert any(line["kind"] == "sample" and line["messages"] for line in episode["lines"])

    # Once the feed has let an episode go, its replies and tool calls are read back from the events its runner kept.
    kept = await System(tmp_path, FeedReader(tmp_path / "elsewhere")).episode(episode["run_id"])
    samples = [line for line in kept["lines"] if line["kind"] == "sample"]
    assert kept["source"] == "archive" and samples and all(not each["messages"] and each["reply"] for each in samples)
    assert [line["reply"] for line in samples] == [
        line["reply"] for line in episode["lines"] if line["kind"] == "sample"
    ]
    (run,) = system["runs"]
    assert run["run"] == "train" and run["fence"] == 1 and run["decided"] == 3 and run["open"] == []
    # The loop wrote down where it ran; its directory is the one opened, whose feed is read.
    assert run["starts"] == 1 and run["from"] is None and run["host"] and run["started"] <= system["at"]
    assert run["name"] == "train"  # (no registry entry: called by its id)
    assert run["episodes_at"] == "here" and run["state"] == "running"
    assert run["played"] == {**run["played"], "episodes": 12, "outcomes": {"completed": 12}, "playing": 0}
    assert run["played"]["sampled"] > 0
    assert [line["group"] for line in run["done"]] == [1, 2, 3]
    trained = [line for line in run["done"] if line["update"]]
    assert trained  # (the policy says yes and no in turn: some group has something to compare)
    # Each group is listed once: with the step it went into, or, if it gave nothing to train on, with the step decided
    # after it (or toward the next one, when none was).
    members = [number for step in run["steps"] for number in [*step["groups"], *step["skipped"]]]
    assert sorted([*members, *run["next"]]) == [1, 2, 3]
    # (groups play at once: a step covers every group queued when it begins, one or more)
    assert sorted(number for step in run["steps"] for number in step["groups"]) == [line["group"] for line in trained]
    made = {step["step"]: step["makes"] for step in run["steps"]}
    assert all(line["adapter"] == made[line["step"]] for line in trained)

    shown = system["versions"]
    assert [version["id"] for version in shown] == [step["makes"] for step in run["steps"]]
    first, last = shown[0], shown[-1]
    assert first["parents"] == [] and first["base"] == "tiny" and first["depth"] == 1 and first["run"] == "train"
    assert first["weights"]["files"] == 1 and first["state"]["files"] == 1 and first["batch"]
    assert all(each["parents"] == [before["id"]] for before, each in itertools.pairwise(shown))
    assert last["id"] == head.id and last["depth"] == len(shown) and last["bookmarks"] == ["best"]
    assert system["bookmarks"] == {"best": head.id} and last["short"] and head.id.startswith(last["short"])

    (runner_seen,) = system["runners"]
    assert runner_seen == {**runner_seen, "runner": "here", "fence": 1, "playing": [], "claims": 12}
    (channel,) = system["channels"]
    assert channel["channel"] == "policy" and channel["adapter"] == head.id
    assert system["ledger"]["fences"] == {scope("train"): 1, runner_scope("here"): 1}
    assert system["ledger"]["tables"][table("train", GROUPS)] == 3
    assert system["kept"]["versions"] > 0 and system["kept"]["episodes"] > 0
    assert system["machine"]["now"]["disk"]["total"] > 0 and system["written"] <= system["at"]


async def ended(ledger: Ledger, run: str, group: int, number: int, run_id: str, runner: str = "here") -> None:
    """An episode a runner recorded: its reward is its number."""
    fence = await ledger.take(runner_scope(f"{runner}-records"))  # (records are written under a fence of their own)
    trajectories = {"ada": Trajectory([], {"default": float(number)})}
    labels = {"run": run, "group": str(group), "episode": str(number)}
    episode = Episode(run, group, number, run_id, labels, Outcome.COMPLETED, trajectories=trajectories)
    record: JsonValue = Record(episode, sampled={"ada": 40}).to_json()
    await ledger.append(table(run, EPISODES), f"{group}/{number}", record, fence)


async def test_a_group_in_flight_is_at_the_stage_a_loop_starting_now_would_find_it_at(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    versions = Versions(ledger, FileBlobStore(tmp_path / BLOBS))
    fence = await ledger.take(scope("train"))
    feed = RunFeed(tmp_path / FEED)
    system = System(tmp_path, FeedReader(tmp_path / FEED))

    async def group() -> dict[str, Any]:
        (run,) = (await system.snapshot())["runs"]
        if run["open"]:
            (open_group,) = run["open"]
            return open_group
        found = await system.group("train", 1)  # (one that is done is no longer in flight)
        assert found is not None
        return found

    start: JsonValue = {"from": None, "host": "here", "started": 5.0, "directory": str(tmp_path)}
    await ledger.append(table("train", STARTS), str(fence.number), start, fence)
    decided: JsonValue = {"task": "t003", "title": "chests", "decided": 5.0, "episodes": 2}
    await ledger.append(table("train", GROUPS), "1", decided, fence)
    waiting = await group()
    assert waiting == {**waiting, "number": 1, "task": "t003", "stage": WAITING, "count": 2, "playing": []}

    runner = await ledger.take(runner_scope("here"))
    claim: JsonValue = {"runner": "here", "fence": runner.number, "at": 6.0}
    await ledger.append(table("train", CLAIMS), "1/1/1", claim, runner)
    claimed = await group()
    assert claimed["stage"] == PLAYING and claimed["ended"] == 0
    assert claimed["playing"] == [{"group": 1, "episode": "1", "attempt": 1, "runner": "here", "at": 6.0}]

    created: JsonValue = {"labels": {"run": "train", "group": "1", "episode": "1"}}
    feed._write("r_one", {"kind": "event", "type": "run.created", "at": 6.0, "payload": created})  # pyright: ignore[reportPrivateUsage]
    feed._write("r_one", {"kind": "sample", "slot": "ada", "at": 7.0})  # pyright: ignore[reportPrivateUsage]
    (episode,) = (await group())["episodes"]
    assert episode == {**episode, "run_id": "r_one", "episode": "1", "state": "running", "samples": 1, "in_feed": True}

    # The second episode's first attempt is cut short when its runner closes: it is claimed again, and played again.
    await ledger.append(table("train", CLAIMS), "1/2/1", claim, runner)
    created = {"labels": {"run": "train", "group": "1", "episode": "2"}}
    feed._write("r_cut", {"kind": "event", "type": "run.created", "at": 6.5, "payload": created})  # pyright: ignore[reportPrivateUsage]
    feed._write("r_cut", {"kind": "event", "type": "run.cancelled", "at": 7.5, "payload": {}})  # pyright: ignore[reportPrivateUsage]
    await ledger.append(table("train", INTERRUPTED), "1/2/1", {"why": CLOSED, "at": 7.5}, runner)
    await ended(ledger, "train", 1, 1, "r_one")
    playing_now = await group()
    assert playing_now["stage"] == PLAYING and playing_now["ended"] == 1 and playing_now["playing"] == []
    assert [
        (each["run_id"], each["state"], each["reward"], each["interrupted"]) for each in playing_now["episodes"]
    ] == [
        ("r_one", "completed", 1.0, False),
        ("r_cut", "cancelled", None, True),
    ]
    await ended(ledger, "train", 1, 2, "r_two")
    assert (await group())["stage"] == ENDED

    result: JsonValue = {"group": 1, "time": 9.0, "task": "t003", "failures": ["x", "x"], "segments": 8}
    await ledger.append(table("train", RESULTS), "1", result, fence)
    recorded = await group()
    assert recorded["stage"] == DONE and recorded["step"] is None  # (done: it waits toward the next step)
    (run,) = (await system.snapshot())["runs"]
    assert run["next"] == [1] and run["open"] == [] and run["done"][0]["step_state"] is None

    step: dict[str, JsonValue] = {
        "groups": [1],
        "parent": None,
        "makes": "minerone",
        "segments": 8,
        "batch": {"uri": "…"},
        "seed": 1,
    }
    await ledger.append(table("train", STEPS), "1", step, fence)
    stepping = await group()
    assert stepping["stage"] == DONE  # the step is its own thing: the group is done, and in it
    assert stepping["step"] == {
        **{key: value for key, value in step.items() if key != "batch"},
        "step": 1,
        "makes": "minerone",
        "state": "stepping",
    }
    (run,) = (await system.snapshot())["runs"]
    assert run["next"] == [] and [(each["step"], each["state"]) for each in run["steps"]] == [(1, "stepping")]
    assert (run["done"][0]["step"], run["done"][0]["step_state"]) == (1, "stepping")

    weights = tmp_path / "adapter.bin"
    weights.write_text("weights")
    await versions.add(fence, "minerone", weights=weights, run="train", step=1)  # the step's version: it is done
    (run,) = (await system.snapshot())["runs"]
    (done,) = run["done"]
    assert run["open"] == [] and {key: done[key] for key in ("group", "task", "failures", "adapter")} == {
        "group": 1,
        "task": "t003",
        "failures": ["x"],
        "adapter": "minerone",
    }
    assert run["steps"] == [{**run["steps"][0], "step": 1, "groups": [1], "state": "committed", "makes": "minerone"}]
    assert [(each["run_id"], each["interrupted"]) for each in done["episodes"]] == [
        ("r_one", False),
        ("r_cut", True),
        ("r_two", None),
    ]
    feed.close()


async def test_a_directory_that_is_no_run_has_nothing_to_show_and_is_left_as_it_is(tmp_path: Path) -> None:
    system = await System(tmp_path, FeedReader(tmp_path / FEED)).snapshot()
    assert system["runs"] == system["versions"] == system["runners"] == system["channels"] == []
    assert system["bookmarks"] == {}
    assert system["written"] is None and system["processes"] is None
    assert not (tmp_path / LEDGER).exists()  # (reading makes no ledger)


async def test_a_claim_holds_while_its_runner_keeps_the_fence_it_was_made_under(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    fence = await ledger.take(scope("train"))
    decided: JsonValue = {"task": "t003", "decided": 5.0, "episodes": 2}
    await ledger.append(table("train", GROUPS), "1", decided, fence)
    first, other = await ledger.take(runner_scope("first")), await ledger.take(runner_scope("other"))
    for key, runner, at in (("1/1/1", first, 6.0), ("1/2/1", other, 7.0)):
        claim: JsonValue = {"runner": runner.scope.removeprefix("runners/"), "fence": runner.number, "at": at}
        await ledger.append(table("train", CLAIMS), key, claim, runner)
    system = System(ledger=ledger)
    runners = {each["runner"]: each for each in (await system.snapshot())["runners"]}
    assert [claim["episode"] for claim in runners["first"]["playing"]] == ["1"]
    assert [claim["episode"] for claim in runners["other"]["playing"]] == ["2"]

    await ledger.take(runner_scope("first"))  # started again: what it claimed before is anyone's
    snapshot = await system.snapshot()
    runners = {each["runner"]: each for each in snapshot["runners"]}
    assert runners["first"] == {**runners["first"], "fence": 2, "playing": [], "claims": 1, "last": 6.0}
    assert [each["runner"] for each in snapshot["runners"]] == ["other", "first"]  # (playing ones first)
    (run,) = snapshot["runs"]
    assert run["played"]["playing"] == 1 and [claim["runner"] for claim in run["open"][0]["playing"]] == ["other"]
