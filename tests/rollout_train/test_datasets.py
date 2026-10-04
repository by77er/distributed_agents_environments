"""Datasets: episodes picked by a rule, turns by filters, kept as a record in the ledger and a manifest in a blob, and
a supervised step on one that makes a checkpoint learned from the checkpoints that sampled it."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.contracts import RunEvent
from rollout.harness.blobs import FileBlobStore
from rollout_train import Checkpoints, FileLedger
from rollout_train.checkpoints import new_id
from rollout_train.datasets import (
    LEFT_OUT,
    Turn,
    dataset_of,
    datasets_in,
    examples,
    make,
    manifest_of,
    resolved_dataset,
    served_at,
    turn_filter,
)
from rollout_train.imitation import GUIDANCE, imitate
from rollout_train.record import GROUPS, STARTS, scope, table
from rollout_train.recorder import Segment, Span
from rollout_train.registry import Taken, registry_of
from rollout_train.rollouts import Episode, Outcome, Trajectory, stored
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.stores import FILES
from rollout_train.testing import plain_renderer
from tests.rollout_train.training.test_loop import Counting

WAY = "How to get there. Place the table."


def segment(prompt: str, sampled: str, version: int) -> Segment:
    tokens = [ord(character) for character in prompt + sampled]
    return Segment(tokens, [Span(len(prompt), len(tokens), version, f"r:{version}")], [-0.5] * len(sampled))


class Played:
    """A run's episodes in a ledger of files, as runners leave them."""

    def __init__(self, tmp_path: Path, run: str = "train") -> None:
        self.ledger = FileLedger(tmp_path / "ledger")
        self.blobs = FileBlobStore(tmp_path / "blobs")
        self.run = run
        self.at: dict[str, JsonValue] = {"kind": FILES, "directory": str(tmp_path / "blobs")}

    async def group(self, number: int, task: str, *episodes: Mapping[str, Any]) -> None:
        fence = await self.ledger.take(scope(self.run))
        decided: JsonValue = {"task": task, "title": task}
        await self.ledger.append(table(self.run, GROUPS), str(number), decided, fence)
        for index, said in enumerate(episodes, start=1):
            info: dict[str, JsonValue] = {"solved": said.get("solved", False), "duration": said.get("duration")}
            if "guidance" in said:
                info[GUIDANCE] = said["guidance"]
            segments: list[Segment] = said.get("segments") or [segment("user: guess\nassistant: ", "apple", 0)]
            trajectory = Trajectory(segments, {"default": float(said.get("reward", 0.0))})
            outcome = Outcome.COMPLETED if said.get("completed", True) else Outcome.FAILED
            episode = Episode(self.run, number, index, f"r{number}-{index}", {}, outcome, info=info,
                              trajectories={"policy": trajectory})  # fmt: skip
            record = (await stored(episode, [], self.blobs)).to_json()
            await self.ledger.append(table(self.run, EPISODES), f"{number}/{index}", record, fence)


async def picked(played: Played, rule: str, per_task: int | None = None) -> list[str]:
    made = await make(played.ledger, rule, [played.run], into=played.blobs, at=played.at, per_task=per_task)
    return [str(line["source"]).rsplit("/", 2)[0] for line in await manifest_of(made)]


async def test_each_rule_picks_its_episodes(tmp_path: Path) -> None:
    played = Played(tmp_path)
    await played.group(1, "t1", {"solved": True, "reward": 2.0, "duration": 9}, {"solved": True, "reward": 3.0,
                       "duration": 30}, {"solved": False, "reward": 5.0})  # fmt: skip
    await played.group(2, "t1", {"solved": True, "reward": 1.0, "duration": 5}, {"solved": True, "reward": 1.0,
                       "duration": 4}, {"solved": True, "reward": 1.0, "duration": 4})  # fmt: skip
    await played.group(3, "t2", {"solved": False}, {"solved": True, "reward": 7.0, "completed": False})
    await played.group(4, "t1", {"solved": True, "reward": 0.5})
    assert await picked(played, "solved-all") == ["train/1/1", "train/1/2", "train/2/1", "train/2/2", "train/2/3",
                                                  "train/4/1"]  # fmt: skip
    # The highest reward, then the shortest, then the first; a group with no solved episode that completed gives none.
    assert await picked(played, "best-of-group") == ["train/1/2", "train/2/2", "train/4/1"]
    assert await picked(played, "capped-per-task", per_task=2) == ["train/1/1", "train/1/2"]
    assert len(await picked(played, "capped-per-task")) == 3
    with pytest.raises(ValueError, match="no episode rule"):
        await picked(played, "best-of-everything")


def even_turns(
    turns: Sequence[Turn], events: Sequence[RunEvent], info: Mapping[str, JsonValue]
) -> list[dict[str, Any]]:
    """A turn filter an environment might supply: every other turn, and what it saw of each."""
    return [{"parity": "even"} if turn.index % 2 == 0 else {LEFT_OUT: "odd", "parity": "odd"} for turn in turns]


def no_turns(turns: Sequence[Turn], events: Sequence[RunEvent], info: Mapping[str, JsonValue]) -> list[dict[str, Any]]:
    return [{LEFT_OUT: "none wanted"} for _ in turns]


async def test_turn_filters_keep_the_turns_every_one_keeps_and_say_why_the_others_went(tmp_path: Path) -> None:
    played = Played(tmp_path)
    turns = [segment(f"user: turn {n}\nassistant: ", "move", 0) for n in range(5)]
    turns.insert(2, Segment([ord("x")] * 4, [], []))  # (a segment that sampled nothing)
    await played.group(1, "t1", {"solved": True, "segments": turns})
    made = await make(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at,
                      turns=["all", "tests.rollout_train.test_datasets:even_turns"])  # fmt: skip
    lines = await manifest_of(made)
    assert [line["source"] for line in lines] == ["train/1/1/policy/0", "train/1/1/policy/4"]
    assert all(line["parity"] == "even" for line in lines) and LEFT_OUT not in lines[0]
    assert made.left_out == {"odd": 3, "nothing sampled": 1} and made.counts["turns_seen"] == 6
    assert turn_filter("all")([], [], {}) == []
    with pytest.raises(ValueError, match="no turn filter"):
        turn_filter("worked")
    with pytest.raises(ValueError, match="no turn"):  # nothing is kept: no dataset
        await make(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at,
                   turns=["tests.rollout_train.test_datasets:no_turns"])  # fmt: skip


async def test_a_dataset_is_a_record_and_a_manifest_that_read_back(tmp_path: Path) -> None:
    played = Played(tmp_path)
    checkpoints = Checkpoints(played.ledger, played.blobs)
    fence = await played.ledger.take(scope("train"))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "adapter.bin").write_text("weights")
    first = await checkpoints.add(fence, new_id(), weights=tmp_path / "weights", run="train", step=1)
    second = await checkpoints.add(fence, new_id(), weights=tmp_path / "weights", run="train", step=2,
                                   parents=[first.id])  # fmt: skip
    assert await served_at(played.ledger, "train") == {1: first.id, 2: second.id}
    at_depth = [segment("user: a\nassistant: ", "apple", version) for version in (0, 1, 2)]
    await played.group(1, "t1", {"solved": True, "reward": 1.0, "segments": at_depth, "guidance": {"way": WAY}})
    made = await make(played.ledger, "best-of-group", ["train"], into=played.blobs, at=played.at, by="me@here")
    assert await dataset_of(played.ledger, made.id) == made and await datasets_in(played.ledger) == [made]
    assert made.checkpoints == [first.id, second.id] and made.by == "me@here" and made.cut == ["way"]
    assert made.counts == {"episodes": 1, "groups": 1, "tasks": 1, "turns": 3, "sampled_tokens": 15,
                           "context_tokens": 3 * len("user: a\nassistant: apple"), "turns_seen": 3}  # fmt: skip
    lines = await manifest_of(made)
    assert [(line["depth"], line["checkpoint"]) for line in lines] == [(0, None), (1, first.id), (2, second.id)]
    assert lines[0] == {"source": "train/1/1/policy/0", "task": "t1", "reward": 1.0, "depth": 0, "checkpoint": None,
                        "tokens": 24, "sampled": 5, "guidance": ["way"]}  # fmt: skip
    # Another process finds it by its id, the start of it, or its name.
    elsewhere = FileLedger(tmp_path / "ledger")
    registry = registry_of(elsewhere)
    assert registry is not None
    await registry.name_dataset("first-guesses", made.id)
    for reference in (made.id, made.id[:6], "first-guesses"):
        assert await resolved_dataset(elsewhere, registry, reference) == made.id
    with pytest.raises(Taken):
        await registry.name_dataset("first-guesses", "another")
    with pytest.raises(KeyError):
        await resolved_dataset(elsewhere, registry, "nothing")
    assert [(each.name, each.dataset) for each in await registry.datasets()] == [("first-guesses", made.id)]


async def guesses(played: Played, tmp_path: Path) -> tuple[Checkpoints, list[str]]:
    """Two checkpoints the run `train` made, and a solved episode sampled at depths 0 to 2, guided."""
    checkpoints = Checkpoints(played.ledger, played.blobs)
    fence = await played.ledger.take(scope("train"))
    (tmp_path / "weights").mkdir(exist_ok=True)
    (tmp_path / "weights" / "adapter.bin").write_text("weights")
    first = await checkpoints.add(fence, new_id(), weights=tmp_path / "weights", run="train", base="qwen", step=1)
    second = await checkpoints.add(fence, new_id(), weights=tmp_path / "weights", run="train", step=2,
                                   parents=[first.id])  # fmt: skip
    told = f"system: Goal.\n\n{WAY}\nuser: guess\nassistant: "
    sampled = [segment(told, word, version) for word, version in (("apple", 2), ("river", 1), ("candle", 0))]
    await played.group(1, "t1", {"solved": True, "reward": 1.0, "segments": sampled, "guidance": {"way": WAY}})
    return checkpoints, [first.id, second.id]


async def test_a_step_on_a_dataset_learns_from_the_checkpoints_that_sampled_it(tmp_path: Path) -> None:
    played = Played(tmp_path)
    checkpoints, (first, second) = await guesses(played, tmp_path)
    made = await make(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at)
    taught = await examples(played.ledger, made, plain_renderer("plain"))
    assert (taught.dataset, taught.episodes, taught.left_out) == (made.id, 1, 0)
    assert [each.source for each in taught.segments] == [f"train/1/1/policy/{index}" for index in range(3)]
    assert all(each.advantage == 1.0 for each in taught.segments)
    text = "".join(chr(token) for token in taught.segments[0].segment.tokens)
    assert text == "system: Goal.\nuser: guess\nassistant: apple"  # (the way, cut)
    assert taught.sampled_by == {"train/1/1/policy/0": second, "train/1/1/policy/1": first}

    other = Checkpoints(played.ledger, played.blobs)  # another run's checkpoint, to train from
    start = await other.add(await played.ledger.take(scope("sft")), new_id(), weights=tmp_path / "weights",
                            run="elsewhere", base="qwen")  # fmt: skip
    trainer = Counting()
    fence = await played.ledger.take(scope("sft"))
    made_by = await imitate(checkpoints, trainer, taught, fence=fence, run="sft", start=start.id, base="qwen",
                            directory=tmp_path / "checkpoints")  # fmt: skip
    assert made_by.parents == (start.id, first, second) and made_by.depth == 2  # the start, then the samplers by depth
    assert made_by.dataset == made.id and (await checkpoints.checkpoint(made_by.id)).dataset == made.id
    assert made_by.metrics["imitated_segments"] == 3.0 and trainer.parents == ["weights"]
    assert made_by.batch is not None
    batch = json.loads(await played.blobs.read(made_by.batch))
    assert [source for source, _ in batch] == [each.source for each in taught.segments]

    again = await imitate(checkpoints, trainer, taught, fence=fence, run="sft", start=None, base="qwen",
                          directory=tmp_path / "checkpoints")  # fmt: skip
    assert again.parents == (made_by.id, first, second)  # (the run goes on from what it made)
    alone = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("fresh")), run="fresh",
                          start=None, base="qwen", directory=tmp_path / "checkpoints")  # fmt: skip
    assert alone.parents == () and alone.depth == 1 and alone.dataset == made.id  # (from the base model)


class Trains:
    """A trainer a profile names, that trains nothing."""

    weights = "lora"

    def __init__(self, model: str, **settings: Any) -> None:
        from rollout_train import Budget

        self.budget = Budget()
        self.settings = settings

    async def step(self, batch: Sequence[Any], *, seed: int, parent: Any, into: Path) -> Any:
        from rollout_train import Step
        from rollout_train.trainer import WEIGHTS

        assert self.settings["objective"] == "likelihood"
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"trained on {len(batch)} segments")
        return Step({"segments": float(len(batch))})


PROFILE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"

[trainer]
kind = "tests.rollout_train.test_datasets:Trains"
channel = "policy"
"""


def test_the_commands_make_list_and_train_on_a_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from rollout_train.cli import main

    played = Played(tmp_path / "played")
    checkpoints, (first, second) = asyncio.run(guesses(played, tmp_path))
    where = str(tmp_path / "played" / "ledger")

    def run(*arguments: str) -> str:
        monkeypatch.setattr("sys.argv", ["rollout", *arguments])
        main()
        return capsys.readouterr().out

    said = run("dataset", "make", "best-of-group", "--run", "train", "--name", "guesses", "--ledger", where)
    assert said.startswith("made the dataset ") and "(guesses): 1 episodes, 1 groups, 1 tasks, 3 turns" in said
    (made,) = asyncio.run(datasets_in(played.ledger))
    assert made.blobs == played.at  # (beside the episodes)
    assert run("dataset", "list", "--ledger", where).split()[:4] == [made.id[:4], "[guesses]", "best-of-group", "3"]
    with pytest.raises(SystemExit, match="no run"):
        run("dataset", "make", "solved-all", "--run", "nobody", "--ledger", where)
    with pytest.raises(SystemExit, match="another dataset"):
        run("dataset", "make", "solved-all", "--run", "train", "--name", "guesses", "--ledger", where)

    profile = tmp_path / "profile.toml"
    profile.write_text(PROFILE.format(directory=tmp_path / "played"))
    monkeypatch.setattr("sys.argv", ["rollout", "imitate", str(profile), "--dataset", "guesses", "--start",
                                     second, "--name", "sft-guesses"])  # fmt: skip
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert "3 segments of 1 episodes (0 left out)" in out and f"(from {second}, {first})" in out
    (made_by,) = [each for each in asyncio.run(checkpoints.all()) if each.dataset is not None]
    assert made_by.parents == (second, first) and made_by.dataset == made.id

    async def started() -> dict[str, Any]:
        registry = registry_of(played.ledger)
        assert registry is not None
        (entry,) = [each for each in await registry.runs() if each.name == "sft-guesses"]
        assert made_by.run == entry.id
        return dict(await played.ledger.read(table(entry.id, STARTS)))

    (start,) = asyncio.run(started()).values()
    assert isinstance(start, dict) and start["kind"] == "imitation" and start["dataset"] == made.id
    assert start["from"] == second


@pytest.mark.parametrize("kind", ["files", "database"])
async def test_a_datasets_name_says_one_dataset(tmp_path: Path, kind: str) -> None:
    from rollout_train.database import DatabaseLedger

    ledger = FileLedger(tmp_path / "files") if kind == "files" else DatabaseLedger(f"sqlite:///{tmp_path / 'db'}")
    registry = registry_of(ledger)
    assert registry is not None and await registry.datasets() == []
    await registry.create("a-run")
    named = await registry.name_dataset("guesses", "kkkkmmmm")
    assert (await registry.name_dataset("guesses", "kkkkmmmm")).dataset == named.dataset  # (again: the same)
    await registry.name_dataset("another", "llllmmmm")
    for taken in ("guesses", "a/b", " "):
        with pytest.raises(Taken):
            await registry.name_dataset(taken, "nnnnmmmm")
    assert [(each.name, each.dataset) for each in await registry.datasets()] == [
        ("another", "llllmmmm"),
        ("guesses", "kkkkmmmm"),
    ]
    assert [each.name for each in await registry.runs()] == ["a-run"]  # (runs and bookmarks are as they were)
