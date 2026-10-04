"""A run's settings: fixed ones and changeable ones; what is wanted of them, kept beside the ledger and changed in
place; and a running loop that reads them each time it is about to decide a step, takes only the changeable ones from
that step on, and says in each step's record which settings it used."""

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout_train import loop as loop_module
from rollout_train import train
from rollout_train.checkpoints import Checkpoints
from rollout_train.colocated import Colocated
from rollout_train.evals import Schedule, make_suite
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.profile import EvalsSpec, Profile
from rollout_train.record import EVALS, STEPS, table
from rollout_train.rollouts.scheduler import episodes_of
from rollout_train.settings import (
    CHANGEABLE,
    EVALS_EVERY,
    EVALS_SUITE,
    GROUPS_PER_STEP,
    applied,
    changeable,
    checked,
    desired_settings_of,
    fixed,
)
from rollout_train.trainer import Changeable, Files, Step, Weighted
from tests.rollout_train.rollouts.games import words
from tests.rollout_train.test_evals import ENVIRONMENT, a_schedule
from tests.rollout_train.test_profile import write
from tests.rollout_train.training.test_loop import Counting, answering, here, made_by


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))


class Rated(Counting):
    """A trainer that trains nothing and takes its learning rate between steps: it writes down the rate each step
    was taken with."""

    def __init__(self) -> None:
        super().__init__()
        self.learning_rate = 1e-4
        self.rates: list[float] = []
        self.changes: list[dict[str, JsonValue]] = []

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return {"learning_rate": self.learning_rate}

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        if set(settings) - {"learning_rate"} or not isinstance(settings.get("learning_rate", 0.0), float):
            raise ValueError(f"cannot take {dict(settings)}")
        self.changes.append(dict(settings))
        self.learning_rate = float(str(settings["learning_rate"]))

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        self.rates.append(self.learning_rate)
        return await super().step(batch, seed=seed, parent=parent, into=into)


def database(tmp_path: Path) -> Ledger:
    pytest.importorskip("rollout_durable")
    from rollout_train.database import DatabaseLedger

    return DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")


@pytest.mark.parametrize("kind", ["files", "database"])
async def test_what_is_wanted_of_a_runs_settings_is_kept_beside_the_ledger_and_changed_in_place(
    tmp_path: Path, kind: str
) -> None:
    ledger = FileLedger(tmp_path / "ledger") if kind == "files" else database(tmp_path)
    desired = desired_settings_of(ledger)
    assert desired is not None and await desired.desired("train") is None
    first = await desired.want("train", {"trainer.learning_rate": 3e-5, EVALS_SUITE: "words-v1"})
    second = await desired.want("train", {EVALS_SUITE: None, EVALS_EVERY: 2})  # (each key given is replaced)
    assert second.settings == {"trainer.learning_rate": 3e-5, EVALS_SUITE: None, EVALS_EVERY: 2}
    assert second.changed >= first.changed and await desired.desired("train") == second
    assert await desired.desired("another") is None
    assert await ledger.tables() == []  # (none of it is in the ledger's record)


def test_only_a_changeable_setting_is_taken_and_only_with_a_value_it_can_have() -> None:
    current: dict[str, JsonValue] = {GROUPS_PER_STEP: 4, EVALS_SUITE: None, EVALS_EVERY: 1}
    current["trainer.learning_rate"] = 1e-4
    wanted: dict[str, JsonValue] = {
        GROUPS_PER_STEP: 2.0,  # (a whole number, as JSON may say it)
        EVALS_SUITE: "words-v1",
        EVALS_EVERY: 0,  # (not a value it can have: left as it is)
        "trainer.learning_rate": 3e-5,
        "trainer.rank": 8,  # (fixed: not taken)
        "model": "another",
    }
    assert applied(current, wanted) == {
        GROUPS_PER_STEP: 2, EVALS_SUITE: "words-v1", EVALS_EVERY: 1, "trainer.learning_rate": 3e-5,
    }  # fmt: skip
    assert checked(EVALS_SUITE, "") is None
    for key, value in ((GROUPS_PER_STEP, 0), (EVALS_EVERY, 1.5), (EVALS_SUITE, 3), (GROUPS_PER_STEP, True)):
        with pytest.raises(ValueError):
            checked(key, value)


def test_a_runs_settings_are_its_profiles_split_into_fixed_and_changeable(tmp_path: Path) -> None:
    profile = Profile.load(write(tmp_path), settings={"trainer.learning_rate": 3e-5})
    trainer = Rated()
    said = fixed(profile, trainer, groups=40, seed=1)
    assert said["model"] == "a-checkpoint" and said["weights"] == "lora" and said["trainer.segment_tokens"] == 900
    assert said["channels.policy.engines"] == 2 and said["episodes_at_once"] == 6 and said["groups"] == 40
    assert "trainer.learning_rate" not in said  # (the trainer takes it between steps)
    assert changeable(trainer, groups_per_step=4, max_lag=1, evals=EvalsSpec("words-v1", every=2)) == {
        GROUPS_PER_STEP: 4, "max_lag": 1, EVALS_SUITE: "words-v1", EVALS_EVERY: 2, "evals.episodes": 1,
        "trainer.learning_rate": 1e-4,
    }  # fmt: skip
    assert set(changeable(Counting(), groups_per_step=4, max_lag=1)) == set(CHANGEABLE)
    wrapped = Colocated(trainer, [])
    assert isinstance(wrapped, Changeable) and wrapped.changeable == {"learning_rate": 1e-4}
    wrapped.change({"learning_rate": 5e-5})
    assert trainer.learning_rate == 5e-5
    with pytest.raises(ValueError):
        Colocated(Counting(), []).change({"learning_rate": 5e-5})


async def test_a_running_loop_takes_the_changeable_settings_wanted_from_the_next_step_and_says_each_steps(
    tmp_path: Path,
) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "words-v1", ENVIRONMENT, words, rows=["say-yes"], seeds=[1])
    desired = desired_settings_of(ledger)
    assert desired is not None
    read: list[int] = []

    async def wanted() -> Mapping[str, JsonValue]:
        decided = await ledger.read(table("train", STEPS))
        read.append(len(decided))
        if decided:  # (once the first step is decided, someone changes the run's settings)
            await desired.want("train", {
                "trainer.learning_rate": 3e-5, GROUPS_PER_STEP: 2, EVALS_SUITE: "words-v1", EVALS_EVERY: 1,
                "trainer.rank": 8, "episodes_at_once": 1,
            })  # fmt: skip
        found = await desired.desired("train")
        return found.settings if found else {}

    async def scheduled(name: str, every: int, episodes: int) -> Schedule | None:
        return a_schedule(suite, every=every) if name == suite.name else None

    recorder, trainer = answering(), Rated()
    async with here(ledger, recorder, blobs):
        await train(
            words, trainer, checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=8, groups_per_step=1, seed=1, desired=wanted, scheduled=scheduled,
        )  # fmt: skip
    steps: Any = await ledger.read(table("train", STEPS))
    used = [steps[key]["settings"] for key in sorted(steps, key=int)]
    assert len(used) >= 3
    assert used[0] == {GROUPS_PER_STEP: 1, "max_lag": 1, EVALS_SUITE: None, EVALS_EVERY: 1, "evals.episodes": 1,
                       "trainer.learning_rate": 1e-4}  # fmt: skip
    assert all(each == used[1] for each in used[1:])  # (taken from the step after the change, and kept)
    assert used[1] == {GROUPS_PER_STEP: 2, "max_lag": 1, EVALS_SUITE: "words-v1", EVALS_EVERY: 1, "evals.episodes": 1,
                       "trainer.learning_rate": 3e-5}  # fmt: skip
    assert trainer.rates == [1e-4] + [3e-5] * (len(used) - 1) and trainer.changes == [{"learning_rate": 3e-5}]
    groups = [len(steps[key]["groups"]) for key in sorted(steps, key=int)]
    assert groups[0] == 1 and all(count >= 2 for count in groups[1:-1])  # (the last step takes whatever is left)
    evaluated: Any = await ledger.read(table("train", EVALS))
    made = {checkpoint.step: checkpoint.id for checkpoint in await made_by(checkpoints)}
    assert sorted(int(key) for key in evaluated) == sorted(step for step in made if step and step > 1)
    assert len(read) >= len(used)  # (read before each step was decided)


async def test_a_value_the_trainer_cannot_take_leaves_its_settings_as_they_were(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)

    async def wanted() -> Mapping[str, JsonValue]:
        return {"trainer.learning_rate": "fast", GROUPS_PER_STEP: 2}

    recorder, trainer = answering(), Rated()
    async with here(ledger, recorder, blobs):
        await train(
            words, trainer, checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=4, groups_per_step=1, seed=1, desired=wanted,
        )  # fmt: skip
    steps: Any = await ledger.read(table("train", STEPS))
    assert steps and all(
        step["settings"]["trainer.learning_rate"] == 1e-4 and step["settings"][GROUPS_PER_STEP] == 2
        for step in steps.values()
    )
    assert trainer.changes == [] and set(trainer.rates) == {1e-4}
