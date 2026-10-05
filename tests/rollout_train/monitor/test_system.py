"""Where a run stands, read from its ledger and its directory: the monitor's system view."""

import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.contracts import Message
from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import Budget, Checkpoints, FileLedger, Files, Step, Weighted, train
from rollout_train.evals import subject_table, suite_table
from rollout_train.gateway import Attempt
from rollout_train.launches import EVAL
from rollout_train.layout import BLOBS, FEED, LEDGER
from rollout_train.ledger import Ledger
from rollout_train.machine import measured
from rollout_train.monitor import FeedReader, RunFeed, System
from rollout_train.monitor.system import DONE, ENDED, PLAYING, WAITING
from rollout_train.presence import presence_of
from rollout_train.record import GROUPS, RESULTS, STARTS, STEPS, scope, table
from rollout_train.registry import registry_of
from rollout_train.rollouts import EpisodeRunner, playing
from rollout_train.rollouts.episodes import Episode, Outcome, Record, Trajectory, stored
from rollout_train.rollouts.scheduler import CLAIMS, CLOSED, EPISODES, INTERRUPTED, runner_scope
from rollout_train.testing import Policy, gateway_endpoints, plain_channel, sample_request
from rollout_train.trainer import STATE, WEIGHTS
from tests.rollout_train.rollouts.games import Words
from tests.rollout_train.support import monitor_client


class Trains:
    """A trainer that trains nothing and leaves files as a trainer would."""

    budget = Budget(segments=3)
    weights = "lora"

    def __init__(self) -> None:
        self.steps = 0

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        self.steps += 1
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"weights after {self.steps} steps")
        (into / STATE).mkdir()
        (into / STATE / "optimizer.bin").write_text(f"moments after {self.steps} steps")
        return Step({"segments": float(len(batch))})


async def test_a_run_that_trained_is_shown_as_its_ledger_and_its_feed_have_it(tmp_path: Path) -> None:
    pytest.importorskip("starlette")

    feed = RunFeed(tmp_path / FEED)
    policy = Policy(plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")]))
    blobs = FileBlobStore(tmp_path / BLOBS)
    ledger = FileLedger(tmp_path / LEDGER)
    recorder = policy.gateway(ledger, blobs)
    checkpoints = Checkpoints(ledger, blobs)
    local = LocalRunner(gateway=recorder, hooks=[feed])
    presence = presence_of(ledger)
    assert presence is not None
    channel = policy.channels["policy"]

    def about() -> dict[str, Any]:  # (what the platform says of its machine and its channel in each beat)
        serving = {"channel": "policy", "adapter": channel.adapter, "version": channel.version, **channel.take()}
        return {"run": "train", "machine": measured(tmp_path), "channels": [serving]}

    runner = EpisodeRunner(
        "here", ledger, local, recorder, blobs, places=4, hooks=[feed], every=0.05,
        presence=presence, about=about, beating=0.05,
    )  # fmt: skip
    async with playing(runner):
        await train(
            Words(), Trains(), checkpoints, base="tiny", channel="policy", directory=tmp_path / "checkpoints",
            publish=policy.publish, groups=3, groups_per_step=1, seed=1, hooks=[feed],
            started={"directory": str(tmp_path)},
        )  # fmt: skip
    await presence.beat("here", {**about(), "places": 4, "playing": 0})  # (its last beat, the last checkpoint served)
    await local.close()
    feed.close()
    registry = registry_of(ledger)
    assert registry is not None
    head = await checkpoints.head("train")
    assert head is not None
    await registry.bookmark("best", head.id)

    async with monitor_client(tmp_path) as client:
        system = (await client.get("/api/system")).json()
        machines = (await client.get("/api/machines")).json()
        group = (await client.get("/api/groups/train/1")).json()
        script = re.search(r'src="\./(assets/[^"]+\.js)"', (await client.get("/")).text)
        assert script and (await client.get(f"/{script.group(1)}")).status_code == 200  # (the page, as built)
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

    shown = system["checkpoints"]
    assert [checkpoint["id"] for checkpoint in shown] == [step["makes"] for step in run["steps"]]
    first, last = shown[0], shown[-1]
    assert first["parents"] == [] and first["base"] == "tiny" and first["depth"] == 1 and first["run"] == "train"
    assert first["weights"]["files"] == 1 and first["state"]["files"] == 1
    assert all(each["parents"] == [before["id"]] for before, each in itertools.pairwise(shown))
    assert last["id"] == head.id and last["depth"] == len(shown) and last["bookmarks"] == ["best"]
    assert system["bookmarks"] == {"best": head.id} and last["short"] and head.id.startswith(last["short"])

    (runner_seen,) = system["runners"]
    assert runner_seen == {**runner_seen, "runner": "here", "fence": 1, "playing": [], "claims": 12}
    (channel,) = system["channels"]
    assert channel["channel"] == "policy" and channel["adapter"] == head.id
    assert system["ledger"]["fences"] == {scope("train"): 1, runner_scope("here"): 1}
    assert system["ledger"]["tables"][table("train", GROUPS)] == 3
    assert system["kept"]["checkpoints"] > 0 and system["kept"]["episodes"] > 0
    (machine,) = machines["runners"]
    assert machine["name"] == "here" and machine["alive"] and machine["run"] == "train" and machine["places"] == 4
    (host,) = machines["hosts"]
    assert host["machine"]["disk"]["total"] > 0 and len(host["history"]) > 1
    assert system["written"] <= system["at"]


async def ended(
    ledger: Ledger,
    run: str,
    group: int,
    number: int,
    run_id: str,
    runner: str = "here",
    info: dict[str, JsonValue] | None = None,
) -> None:
    """An episode a runner recorded: its reward is its number; `info` is what it reported."""
    fence = await ledger.take(runner_scope(f"{runner}-records"))  # (records are written under a fence of their own)
    trajectories = {"ada": Trajectory([], {"default": float(number)})}
    labels = {"run": run, "group": str(group), "episode": str(number)}
    episode = Episode(run, group, number, run_id, labels, Outcome.COMPLETED, info=info or {}, trajectories=trajectories)
    record: JsonValue = Record(episode, sampled={"ada": 40}).to_json()
    await ledger.append(table(run, EPISODES), f"{group}/{number}", record, fence)


async def test_a_group_in_flight_is_at_the_stage_a_loop_starting_now_would_find_it_at(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / BLOBS))
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
    assert episode == {**episode, "run_id": "r_one", "episode": "1", "state": "running", "samples": 1}

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
    await checkpoints.add(fence, "minerone", weights=weights, run="train", step=1)  # the step's checkpoint: it is done
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
    assert system["runs"] == system["checkpoints"] == system["runners"] == system["channels"] == []
    assert system["bookmarks"] == {}
    assert system["written"] is None and "processes" not in system
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


async def test_a_task_that_never_says_whether_it_solved_is_shown_saying_nothing_of_it(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    fence = await ledger.take(scope("words"))
    start: JsonValue = {"from": None, "host": "here", "started": 5.0}
    await ledger.append(table("words", STARTS), str(fence.number), start, fence)
    for number, episodes in (("1", 2), ("2", 1)):
        decided: JsonValue = {"task": "say", "decided": 5.0, "episodes": episodes}
        await ledger.append(table("words", GROUPS), number, decided, fence)
    await ended(ledger, "words", 1, 1, "r_one", info={"turns": 3})  # (group 1 says nothing of solving)
    await ended(ledger, "words", 1, 2, "r_two")
    await ended(ledger, "words", 2, 1, "r_three", info={"solved": True})
    for number, solved in (("1", [False, False]), ("2", [True])):  # (training reads what is not said as not solved)
        rewards: list[JsonValue] = [1.0] * len(solved)
        result: JsonValue = {"time": 9.0, "rewards": rewards, "solved": list[JsonValue](solved)}
        await ledger.append(table("words", RESULTS), number, result, fence)
    system = System(ledger=ledger)

    (run,) = (await system.snapshot())["runs"]
    assert [line["solved"] for line in run["done"]] == [[None, None], [True]]
    assert [each["solved"] for line in run["done"] for each in line["episodes"]] == [None, None, True]
    group = await system.group("words", 1)
    assert group is not None and group["result"]["solved"] == group["outcome"]["solved"] == [None, None]
    statistics = (await system.statistics())["runs"][0]["groups"]
    assert [each["solved"] for each in statistics] == [[None, None], [True]]


async def test_a_checkpoint_says_what_its_weights_are(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / BLOBS))
    fence = await ledger.take(scope("train"))
    weights = tmp_path / "weights.bin"
    weights.write_text("weights")
    await checkpoints.add(fence, "fullone", weights=weights, run="train", step=1, kind="full", base="small")
    await checkpoints.add(fence, "adapterone", weights=weights, run="train", step=2, parents=["fullone"])
    shown = {each["id"]: each for each in (await System(ledger=ledger).snapshot())["checkpoints"]}
    assert (shown["fullone"]["kind"], shown["fullone"]["base"]) == ("full", "small")
    assert (shown["adapterone"]["kind"], shown["adapterone"]["base"]) == ("lora", "fullone")  # (an adapter over it)


async def test_an_eval_whose_task_never_says_whether_it_solved_counts_no_solves(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    fence = await ledger.take(scope("words-eval"))
    begun: JsonValue = {"kind": EVAL, "suite": "words-v1", "checkpoint": None, "started": 5.0}
    await ledger.append(table("words-eval", STARTS), str(fence.number), begun, fence)
    await ledger.append(table("words-eval", GROUPS), "1", {"task": "say", "episodes": 1}, fence)
    version: JsonValue = {"entries": [{"environment": "games:words", "starts": [{"task": "say", "seed": 1}]}]}
    await ledger.append(suite_table("words-v1", "suite"), "suite", version, fence)
    who: JsonValue = {
        "kind": "model",
        "version": "words-v1@1",
        "parts": [{"environment": "games:words", "run": "words-eval"}],
    }
    await ledger.append(subject_table("words-v1", "words-eval", "subject"), "subject", who, fence)
    outcome: JsonValue = {"run_id": "r_one", "reward": 0.5, "solved": False}
    await ledger.append(subject_table("words-v1", "words-eval", "results"), "1-1", outcome, fence)
    await ended(ledger, "words-eval", 1, 1, "r_one")
    evals = await System(ledger=ledger).evals()
    assert [each["solved"] for each in evals["evals"]] == [None]
    (suite,) = evals["suites"]
    (subject,) = suite["subjects"]
    assert subject["solved"] is None
    assert subject["results"] == {"1": [{"solved": None, "reward": 0.5, "run_id": "r_one"}]}


async def test_an_episode_playing_is_shown_with_its_reward_so_far_and_each_slots(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / LEDGER)
    fence = await ledger.take(scope("train"))
    start: JsonValue = {"from": None, "host": "here", "started": 5.0, "directory": str(tmp_path)}
    await ledger.append(table("train", STARTS), str(fence.number), start, fence)
    await ledger.append(table("train", GROUPS), "1", {"task": "t", "decided": 5.0, "episodes": 1}, fence)
    feed = RunFeed(tmp_path / FEED)
    created: JsonValue = {"labels": {"run": "train", "group": "1", "episode": "1"}}
    feed._write("r_one", {"kind": "event", "type": "run.created", "at": 6.0, "payload": created})  # pyright: ignore[reportPrivateUsage]
    for slot, value in (("ada", 1.0), ("bo", 0.0)):
        reward: JsonValue = {"slot": slot, "value": value}
        feed._write("r_one", {"kind": "event", "type": "reward.assigned", "at": 7.0, "payload": reward})  # pyright: ignore[reportPrivateUsage]
    (run,) = (await System(tmp_path, FeedReader(tmp_path / FEED)).snapshot())["runs"]
    (episode,) = run["open"][0]["episodes"]
    assert episode["reward"] == 0.5 and episode["rewards"] == {"ada": 1.0, "bo": 0.0}
    feed.close()


async def test_an_episode_whose_samples_a_gateway_elsewhere_recorded_shows_their_replies(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / LEDGER), FileBlobStore(tmp_path / BLOBS)
    fence = await ledger.take(scope("train"))
    start: JsonValue = {"from": None, "host": "here", "started": 5.0, "directory": str(tmp_path)}
    await ledger.append(table("train", STARTS), str(fence.number), start, fence)
    await ledger.append(table("train", GROUPS), "1", {"task": "t", "decided": 5.0, "episodes": 1}, fence)
    feed = RunFeed(tmp_path / FEED)  # (what the runner's hooks saw: its events, none of its harness's samples)
    created: JsonValue = {"labels": {"run": "train", "group": "1", "episode": "1"}}
    feed._write("r_one", {"kind": "event", "type": "run.created", "at": 6.0, "payload": created})  # pyright: ignore[reportPrivateUsage]
    feed.close()
    endpoints = gateway_endpoints(
        plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")]), ledger=ledger, blobs=blobs
    )
    endpoints.admit("r_one", Attempt("train", await ledger.take("runs/train/episodes/1/1"), "1/1", 1))
    endpoint = endpoints.endpoint(RecordedModel(channel="policy"))
    for turn in range(2):
        await endpoint.sample(sample_request([Message.user(f"Go {turn}.")], f"h{turn}", session_id="r_one/policy"))

    playing_now = await System(tmp_path, FeedReader(tmp_path / FEED)).episode("r_one")
    assert playing_now["source"] == "turns" and playing_now["labels"]["run"] == "train"
    said = [(line["slot"], line["reply"]["text"]) for line in playing_now["lines"]]
    assert said == [("policy", "yes"), ("policy", "no")]
    later = await System(tmp_path, FeedReader(tmp_path / FEED)).episode("r_one", after=1)
    assert [line["effect_id"] for line in later["lines"]] == ["h1"]

    # Once it ended, and the feed has let it go: its kept events hold no sample either, and the turns are read.
    episode = Episode("train", 1, 1, "r_one", {"run": "train"}, Outcome.COMPLETED)
    record = await stored(episode, [], blobs)
    await ledger.append(table("train", EPISODES), "1/1", record.to_json(), fence)
    ended = await System(tmp_path, FeedReader(tmp_path / "elsewhere")).episode("r_one")
    assert ended["source"] == "turns" and ended["ended"]["state"] == "completed"
    assert [line["reply"]["text"] for line in ended["lines"]] == ["yes", "no"]
