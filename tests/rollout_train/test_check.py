"""`rollout env check`: an environment's rows, starts, eval data and one scripted episode; and groups played by a model,
with the ones that teach nothing said plainly."""

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.environment import Description, Row, Start, binding_for
from rollout.harness.blobs import FileBlobStore
from rollout_train.check import CHECK, checked, played, scripted
from rollout_train.ledger import FileLedger
from rollout_train.record import RESULTS, STARTS, table
from rollout_train.testing import Policy, plain_channel
from tests.rollout_train.rollouts.games import Words, guessing, words
from tests.rollout_train.support import answering, here

ENVIRONMENT = "tests.rollout_train.rollouts.games:words"


class Careless(Words):
    """Everything an environment can get wrong before it is played."""

    version = ""

    def rows(self) -> Sequence[Row]:
        return [Row("same", "a"), Row("same", "b", counts_for=("gone",))]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {"at": random.random()}  # (not the random numbers it is given)

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {}


careless = Careless()


class Unstartable(Words):
    """Words whose say-maybe has no start to draw, and whose eval data is written down rather than drawn."""

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        if row.key == "say-maybe":
            raise KeyError("no such board")
        return super().start(row, rng)

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"words-held-out": [Start("say-yes", "say yes", 1, {"word": "yes", "seed": 1001})]}


class Unmeasured(Words):
    def evals(self) -> Mapping[str, Sequence[Start]]:
        raise RuntimeError("eval data missing")


class Unsaid(Words):
    """Words whose description says its results report no duration, and rewards of 2 or more."""

    description = Description(rewards=(2.0, 3.0), saturated=True)


def test_an_environment_that_holds_together_passes_every_check() -> None:
    found = checked(words)
    assert [each.check for each in found] == ["rows", "description", "starts", "train and eval"]
    assert all(each.passed for each in found), found
    assert "6 eval starts in words-held-out (6)" in found[3].said


def test_what_an_environment_gets_wrong_is_said() -> None:
    (rows, description) = checked(Careless())  # (rows that do not build stop the rest)
    assert not rows.passed and "two rows are 'same'" in rows.said and "counts for gone" in rows.said
    assert not description.passed and "no version" in description.said

    class Wandering(Careless):
        version = "1"

        def rows(self) -> Sequence[Row]:
            return [Row("one", "one")]

    found = {each.check: each for each in checked(Wandering())}
    assert not found["starts"].passed and "drew two different starts" in found["starts"].said
    assert not found["train and eval"].passed and "no eval data" in found["train and eval"].said


async def test_what_an_environment_raises_is_said_as_a_failed_check() -> None:
    found = {each.check: each for each in checked(Unstartable())}
    assert not found["starts"].passed and "say-maybe with seed 0: start() raised KeyError" in found["starts"].said
    assert found["train and eval"].said == "say-maybe: start() raised KeyError: 'no such board'"
    found = {each.check: each for each in checked(Unmeasured())}
    assert found["train and eval"].said == "evals() raised RuntimeError: eval data missing"
    assert not found["starts"].passed and not found["train and eval"].passed
    for environment, row in ((Unstartable(), "say-maybe"), (Unmeasured(), "say-yes")):
        episode = await scripted(environment, row=row, reply="yes")
        assert not episode.passed and episode.said.startswith(f"{row}'s start could not be drawn: ")


async def test_one_episode_plays_with_a_scripted_model_and_its_result_is_held_to_the_description() -> None:
    said = await scripted(words, reply="yes")
    assert said.passed and said.said.startswith("say-yes, 1 model turn: reward 1")
    missed = await scripted(words, row="say-no", reply="yes")
    assert missed.passed and "reward 0" in missed.said
    unsaid = await scripted(Unsaid(), reply="yes")
    assert not unsaid.passed and "outside [2, 3]" in unsaid.said and "says a duration" in unsaid.said
    assert not (await scripted(words, row="say-perhaps")).passed
    chosen = await scripted(guessing, reply="<think>river? apple?</think>apple")
    assert chosen.passed and '"solved": true' in chosen.said


async def test_groups_whose_episodes_all_scored_the_same_are_flagged(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    binding = binding_for(words, "policy")
    async with here(ledger, answering(), blobs):  # (it says yes, then no: a group of say-yes is half solved)
        found = await played(words, ledger, blobs, run="check-1", binding=binding, groups=3, episodes=4)
    assert [each.check for each in found] == ["group 1", "group 2", "group 3", "groups"]
    assert all(each.passed for each in found) and "groups have something to teach" in found[-1].said
    assert all(each.said.startswith("say-maybe") for each in found[:3] if each.flagged)  # (no one says maybe)
    start: Any = next(iter((await ledger.read(table("check-1", STARTS))).values()))
    assert start["kind"] == CHECK and start["version"] == "1"
    assert len(await ledger.read(table("check-1", RESULTS))) == 3

    never = Policy(plain_channel(always=[("perhaps\n", "stop")]))
    async with here(ledger, never, blobs):
        found = await played(words, ledger, blobs, run="check-2", binding=binding, groups=2, episodes=4)
    assert all(each.flagged and "every episode scored the same" in each.said for each in found[:2])
    assert not found[-1].passed and "training would take no step" in found[-1].said
    lines: Any = await ledger.read(table("check-2", RESULTS))
    assert all("every episode scored the same" in line["skipped"] for line in lines.values())


def test_the_command_checks_an_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from rollout_train.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))  # (its scratch files go under ~/.cache/rollout/checks)

    def run(*arguments: str) -> tuple[int, str]:
        monkeypatch.setattr("sys.argv", ["rollout", "env", "check", *arguments])
        with pytest.raises(SystemExit) as ended:
            main()
        return int(ended.value.code or 0), capsys.readouterr().out

    code, out = run(ENVIRONMENT, "--reply", "yes")
    assert code == 0 and [line.split()[0] for line in out.splitlines()] == ["ok"] * 5
    code, out = run("tests.rollout_train.test_check:careless")
    assert code == 1 and "FAIL  rows" in out
