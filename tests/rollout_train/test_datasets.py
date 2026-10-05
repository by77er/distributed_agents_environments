"""Datasets: episodes picked by a rule, turns by filters, kept as a record in the ledger and a manifest in a blob, and
a supervised step on one that makes a checkpoint learned from the checkpoints that sampled it."""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pydantic import JsonValue

from rollout.contracts import RunEvent
from rollout.harness.blobs import FileBlobStore
from rollout_train import Checkpoints, FileLedger, Files, Step, Weighted
from rollout_train.checkpoints import new_id
from rollout_train.datasets import (
    LEFT_OUT,
    Turn,
    dataset_of,
    datasets_in,
    examples,
    make_dataset,
    manifest_of,
    resolved_dataset,
    served_at,
    turn_filter,
)
from rollout_train.imitation import GUIDANCE, imitate, passes_for
from rollout_train.record import GROUPS, STARTS, scope, table
from rollout_train.recorder import TOKEN_LEVEL, Segment, Span, TeacherScores
from rollout_train.registry import Taken, registry_of
from rollout_train.rollouts import Episode, Outcome, Trajectory, stored
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.stores import FILES
from rollout_train.testing import plain_renderer
from rollout_train.trainer import Distilled, Item, Labelled, Pair
from tests.rollout_train.support import Counting

WAY = "How to get there. Place the table."


def segment(prompt: str, sampled: str, version: int, sampled_with: tuple[str, ...] = TOKEN_LEVEL) -> Segment:
    tokens = [ord(character) for character in prompt + sampled]
    spans = [Span(len(prompt), len(tokens), version, f"r:{version}")]
    return Segment(tokens, spans, [-0.5] * len(sampled), sampled_with=sampled_with)


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
    made = await make_dataset(played.ledger, rule, [played.run], into=played.blobs, at=played.at, per_task=per_task)
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
    with pytest.raises(ValueError, match="there is no rule"):
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
    made = await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at,
                      turns=["all", "tests.rollout_train.test_datasets:even_turns"])  # fmt: skip
    lines = await manifest_of(made)
    assert [line["source"] for line in lines] == ["train/1/1/policy/0", "train/1/1/policy/4"]
    assert all(line["parity"] == "even" for line in lines) and LEFT_OUT not in lines[0]
    assert made.left_out == {"odd": 3, "nothing sampled": 1} and made.counts["turns_seen"] == 6
    assert turn_filter("all")([], [], {}) == []
    with pytest.raises(ValueError, match="no turn filter"):
        turn_filter("worked")
    with pytest.raises(ValueError, match="no turn"):  # nothing is kept: no dataset
        await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at,
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
    made = await make_dataset(played.ledger, "best-of-group", ["train"], into=played.blobs, at=played.at, by="me@here")
    assert await dataset_of(played.ledger, made.id) == made and await datasets_in(played.ledger) == [made]
    assert made.checkpoints == [first.id, second.id] and made.by == "me@here" and made.cut == ["way"]
    assert made.supervision == "importance"  # (every turn was sampled with its exact tokens and logprobs)
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
    made = await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at)
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
    assert made_by.supervision == again.supervision == "importance"
    alone = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("fresh")), run="fresh",
                          start=None, base="qwen", directory=tmp_path / "checkpoints")  # fmt: skip
    assert alone.parents == () and alone.depth == 1 and alone.dataset == made.id  # (from the base model)


class Stateful(Counting):
    """A trainer that trains nothing and notes the trainer state each step was given."""

    def __init__(self) -> None:
        super().__init__()
        self.states: list[str | None] = []

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        self.states.append((parent.state / "optimizer.bin").read_text() if parent and parent.state else None)
        return await super().step(batch, seed=seed, parent=parent, into=into)


async def test_a_step_starts_its_optimizer_afresh_unless_told_to_resume_it(tmp_path: Path) -> None:
    played = Played(tmp_path)
    checkpoints, (_, second) = await guesses(played, tmp_path)
    (tmp_path / "moments").mkdir()
    (tmp_path / "moments" / "optimizer.bin").write_text("moments of a policy gradient")
    trained = await checkpoints.add(await played.ledger.take(scope("train")), new_id(), weights=tmp_path / "weights",
                                    state=tmp_path / "moments", run="train", step=3, parents=[second])  # fmt: skip
    made = await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at)
    taught = await examples(played.ledger, made, plain_renderer("plain"))
    trainer = Stateful()
    fresh = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("sft")), run="sft",
                          start=trained.id, base="qwen", directory=tmp_path / "checkpoints")  # fmt: skip
    assert trainer.parents == ["weights"] and trainer.states == [None]  # (the start's weights, not its moments)
    assert fresh.metrics["optimizer_resumed"] == 0.0 and fresh.parents[0] == trained.id
    resumed = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("again")), run="again",
                            start=trained.id, base="qwen", directory=tmp_path / "checkpoints",
                            resume_optimizer=True)  # fmt: skip
    assert trainer.states == [None, "moments of a policy gradient"] and resumed.metrics["optimizer_resumed"] == 1.0


class Trains:
    """A trainer a profile names, that trains nothing (and keeps what each step was given)."""

    weights = "lora"
    given: ClassVar[list[list[Any]]] = []

    def __init__(self, model: str, **settings: Any) -> None:
        from rollout_train import Budget

        self.budget = Budget()
        self.settings = settings

    async def step(self, batch: Sequence[Any], *, seed: int, parent: Any, into: Path) -> Any:
        from rollout_train import Step
        from rollout_train.trainer import WEIGHTS

        assert self.settings["objective"]["family"] in ("likelihood", "preference")  # (resolved)
        Trains.given.append(list(batch))
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
    registry = registry_of(played.ledger)
    assert registry is not None
    asyncio.run(registry.create("train", "train"))  # (a run is found by the registry)

    def run(*arguments: str) -> str:
        monkeypatch.setattr("sys.argv", ["rollout", *arguments])
        main()
        return capsys.readouterr().out

    said = run("dataset", "make", "best-of-group", "--run", "train", "--name", "guesses", "--ledger", where)
    assert said.startswith("made the dataset ") and "(guesses): 1 episodes, 1 groups, 1 tasks, 3 turns" in said
    (made,) = asyncio.run(datasets_in(played.ledger))
    assert made.blobs == played.at  # (beside the episodes)
    listed = run("dataset", "list", "--ledger", where).split()
    assert listed[:3] == [made.id[:4], "best-of-group", "3"] and listed[-2:] == ["train", "[guesses]"]
    with pytest.raises(SystemExit, match="no run"):
        run("dataset", "make", "solved-all", "--run", "nobody", "--ledger", where)
    with pytest.raises(SystemExit, match="another dataset"):
        run("dataset", "make", "solved-all", "--run", "train", "--name", "guesses", "--ledger", where)

    profile = tmp_path / "profile.toml"
    profile.write_text(PROFILE.format(directory=tmp_path / "played"))
    monkeypatch.setattr("sys.argv", ["rollout", "imitate", str(profile), "--dataset", "guesses", "--start",
                                     second, "--name", "sft-guesses", "--set", "imitation.passes=2"])  # fmt: skip
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert (
        "3 segments of 1 episodes (0 left out), importance" in out
        and "2 passes at" in out
        and f"(from {second}, {first})" in out
    )
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
    assert start["from"] == second and start["supervision"] == "importance"
    fixed = cast(dict[str, Any], start["run_settings"])[
        "fixed"
    ]  # (its run settings: the profile's, with what was said over them)
    assert fixed["kind"] == "imitate" and fixed["imitation.dataset"] == "guesses" and fixed["start"] == second
    assert fixed["imitation.passes"] == 2 and fixed["objective.preset"] == "sft"
    assert start["run_settings"]["objective"]["family"] == "likelihood"
    assert start["run_settings"]["preset"] is None


def test_a_dataset_of_pairs_is_trained_on_by_the_preference_preset_a_run_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from rollout_train.cli import main

    played = Played(tmp_path / "played")
    asyncio.run(played.group(1, "t1", {"reward": 1.0}, {"reward": 0.0}, {"reward": 0.5}))
    registry = registry_of(played.ledger)
    assert registry is not None
    asyncio.run(registry.create("train", "train"))
    where = str(tmp_path / "played" / "ledger")
    monkeypatch.setattr("sys.argv", ["rollout", "dataset", "make", "best-and-worst", "--run", "train", "--name",
                                     "pairs", "--ledger", where])  # fmt: skip
    main()
    assert "1 episodes" not in capsys.readouterr().out  # (two: the best and the worst)
    profile = tmp_path / "profile.toml"
    profile.write_text(PROFILE.format(directory=tmp_path / "played"))

    def imitate(*more: str) -> None:
        monkeypatch.setattr("sys.argv", ["rollout", "imitate", str(profile), "--dataset", "pairs", *more])
        main()

    with pytest.raises(SystemExit, match="a dataset of pairs is trained on by a preference preset, not sft"):
        imitate("--name", "nothing-named")
    with pytest.raises(SystemExit, match="a likelihood or preference preset, not dapo"):
        imitate("--name", "a-policy-gradient", "--set", "objective.preset=dapo")
    Trains.given.clear()
    with pytest.raises(SystemExit) as exited:
        imitate("--name", "simpo-pairs", "--set", "objective.preset=simpo")
    assert exited.value.code == 0 and "1 pairs of 2 episodes" in capsys.readouterr().out
    ((pair,),) = Trains.given
    assert isinstance(pair, Pair) and pair.source == "train/1/1>2"


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


def test_a_small_dataset_takes_passes_enough_for_its_updates() -> None:
    twelve = [Weighted(segment("user: a\nassistant: ", "x" * 10, 0), 1.0) for _ in range(12)]  # 120 sampled tokens
    assert passes_for(twelve, 512) == 8  # (a pass is one update: eight passes make eight)
    assert passes_for(twelve, 40) == 3  # (three updates a pass)
    assert passes_for(twelve * 100, 512) == 1  # (large enough: one pass)


async def test_a_dataset_of_turns_sampled_without_behaviour_logprobs_is_supervised(tmp_path: Path) -> None:
    played = Played(tmp_path)
    replied = [segment("user: a\nassistant: ", "apple", 0), segment("user: b\nassistant: ", "pear", 0, sampled_with=())]
    await played.group(1, "t1", {"solved": True, "segments": replied})
    made = await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at)
    assert made.supervision == "supervised" and await dataset_of(played.ledger, made.id) == made
    taught = await examples(played.ledger, made, plain_renderer("plain"))
    assert taught.supervision == "supervised"
    checkpoints, trainer = Checkpoints(played.ledger, played.blobs), Counting()
    made_by = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("sft")), run="sft",
                            start=None, base="qwen", directory=tmp_path / "checkpoints")  # fmt: skip
    assert (
        made_by.supervision == "supervised" and (await checkpoints.checkpoint(made_by.id)).supervision == "supervised"
    )


def taught_segment(prompt: str, sampled: str, version: int, top: int = 2) -> Segment:
    """A segment a teacher scored: its logprob of each sampled token, and its top `top` tokens at each."""
    made = segment(prompt, sampled, version)
    scores = TeacherScores("teacher", [-0.25] * len(sampled), [[ord("a"), ord("b")][:top]] * len(sampled),
                           [[-0.1, -2.5][:top]] * len(sampled))  # fmt: skip
    return dataclasses.replace(made, teacher=scores)


async def test_a_dataset_of_teacher_samples_is_of_teacher_supervision_and_its_examples_are_distilled(
    tmp_path: Path,
) -> None:
    played = Played(tmp_path, run="teacher-eval")
    told = f"system: Goal.\n\n{WAY}\nuser: guess\nassistant: "
    sampled = [taught_segment(told, "apple", 0), taught_segment(told, "pear", 0)]
    await played.group(1, "t1", {"solved": True, "segments": sampled, "guidance": {"way": WAY}})
    made = await make_dataset(played.ledger, "solved-all", ["teacher-eval"], into=played.blobs, at=played.at)
    assert made.supervision == "teacher" and await dataset_of(played.ledger, made.id) == made
    assert {line["teacher"] for line in await manifest_of(made)} == {"teacher"}
    taught = await examples(played.ledger, made, plain_renderer("plain"))
    assert taught.supervision == "teacher" and not taught.segments and len(taught.distilled) == 2
    first = taught.distilled[0]
    assert isinstance(first, Distilled) and first.scores.top == 2 and first.source == "teacher-eval/1/1/policy/0"
    assert WAY not in "".join(map(chr, first.segment.tokens))  # (the guidance cut, the teacher's scores kept)
    assert first.scores.logprobs == [-0.25] * 5 and first.segment.sampled == 5
    checkpoints, trainer = Checkpoints(played.ledger, played.blobs), Counting()
    made_by = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("distil")),
                            run="distil", start=None, base="qwen", directory=tmp_path / "checkpoints")  # fmt: skip
    assert made_by.supervision == "teacher" and all(isinstance(each, Distilled) for each in trainer.batches[0])


async def test_a_dataset_with_a_turn_no_teacher_scored_is_not_of_teacher_supervision(tmp_path: Path) -> None:
    played = Played(tmp_path)
    mixed = [taught_segment("user: a\nassistant: ", "apple", 0), segment("user: b\nassistant: ", "pear", 0)]
    await played.group(1, "t1", {"solved": True, "segments": mixed})
    made = await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at)
    assert made.supervision == "importance"
    assert len((await examples(played.ledger, made, plain_renderer("plain"))).segments) == 2
    played = Played(tmp_path / "without-top")
    await played.group(1, "t1", {"solved": True, "segments": [taught_segment("user: a\nassistant: ", "apple", 0, 0)]})
    made = await make_dataset(played.ledger, "solved-all", ["train"], into=played.blobs, at=played.at)
    assert made.supervision == "importance"  # (logprobs of the sampled tokens alone are an on-policy run's, no dataset)


async def test_a_dataset_of_pairs_prefers_each_groups_best_episode_to_its_worst(tmp_path: Path) -> None:
    played = Played(tmp_path)
    checkpoints, (_, second) = await guesses(played, tmp_path)  # (group 1: one episode, nothing to compare)
    told = f"system: Goal.\n\n{WAY}\nuser: guess\nassistant: "
    replies = {
        word: [segment(told, word, 2), segment(f"{told}{word}\nuser: again\nassistant: ", word, 1)]
        for word in ("yes", "no")
    }
    await played.group(
        2, "t1",
        {"reward": 0.5, "segments": replies["no"], "guidance": {"way": WAY}},
        {"reward": 1.0, "segments": replies["yes"], "guidance": {"way": WAY}, "duration": 3},
        {"reward": 0.0, "segments": [segment("user: guess\nassistant: ", "nothing", 0, sampled_with=())]},
        {"reward": 1.0, "segments": replies["no"], "guidance": {"way": WAY}, "duration": 2},
    )  # fmt: skip
    await played.group(3, "t2", {"reward": 1.0}, {"reward": 1.0})  # (every reward the same: no pair)
    made = await make_dataset(played.ledger, "best-and-worst", ["train"], into=played.blobs, at=played.at)
    assert made.kind == "pairs" and made.counts["pairs"] == 1 and made.counts["episodes"] == 2
    assert made.supervision == "supervised"  # (the worst was sampled without behaviour logprobs)
    (line,) = await manifest_of(made)
    assert line["source"] == "train/2/4>3" and line["rewards"] == [1.0, 0.0]  # (the best and the shortest)
    taught = await examples(played.ledger, made, plain_renderer("plain"))
    (pair,) = taught.preferences
    assert isinstance(pair, Pair) and not taught.segments and taught.items == [pair]
    assert len(pair.chosen) == 2 and len(pair.rejected) == 1
    assert "".join(chr(token) for token in pair.chosen[0].tokens) == "system: Goal.\nuser: guess\nassistant: no"
    assert taught.sampled_by["train/2/4/policy/0"] == second
    trainer = Counting()
    made_by = await imitate(checkpoints, trainer, taught, fence=await played.ledger.take(scope("dpo")), run="dpo",
                            start=None, base="qwen", directory=tmp_path / "checkpoints")  # fmt: skip
    assert trainer.batches == [[pair]] and made_by.metrics["imitated_segments"] == 3.0
    assert made_by.dataset == made.id and made_by.parents == ()  # (from the base model: none)

    labelled = await make_dataset(played.ledger, "above-and-below", ["train"], into=played.blobs, at=played.at)
    assert labelled.kind == "labelled" and labelled.counts["labelled"] == 4
    examples_of = (await examples(played.ledger, labelled, plain_renderer("plain"))).preferences
    assert [(each.source, each.desirable) for each in examples_of if isinstance(each, Labelled)] == [
        ("train/2/1", False), ("train/2/2", True), ("train/2/3", False), ("train/2/4", True)
    ]  # fmt: skip
