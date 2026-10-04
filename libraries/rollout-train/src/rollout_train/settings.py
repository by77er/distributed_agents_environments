"""A run's settings: fixed ones, which make what the run is, and changeable ones, which a running run takes from its
next step on; and the settings someone wants a run to have from now on (its desired settings).

A training run's settings are named by dotted key, as a profile's are (`rollout_train.profile.Profile.load`). Fixed
ones are fixed when it starts: the model, the trainer's kind and what its weights are (`lora`, `full`), the adapter's
rank, the channels and their engines, how many episodes it plays at once (`fixed`). Changeable ones can change between
two steps without breaking it (`CHANGEABLE`): how many groups a step waits for (`groups_per_step`), the evals it makes
of its checkpoints (`evals.suite`: a suite by name, which follows its newest version, or one version by id, `NAME@N`;
`evals.every`; `evals.episodes`, none for the suite's own), and whatever settings its trainer says it takes between
steps (`trainer.learning_rate`, say: `rollout_train.trainer.Changeable`).

A run's desired settings are ordinary state, changed in place, not part of the ledger's append-only record: a file
beside a ledger of files (`FileDesiredSettings`), a table in a database ledger's database
(`rollout_train.database.DatabaseDesiredSettings`). Whoever wants a change (the monitor's run settings page, say)
writes it there; the training loop reads them each time it is about to decide a step, takes the changeable ones it
knows for that step and those after it, and writes the settings each step used in the step's record
(`rollout_train.record`). A run that is not running takes them when it is started again.

Beside them, a run's desired settings say whether it is paused (`PAUSED`). A paused run's loop decides no group and no
step, and runners claim none of its episodes, nor those of the evals it asked for or is part of (`paused`); what is
playing plays out and is recorded, and a step being taken is finished. Its process stays, beating and holding its
engines. Paused false, it goes on.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout_train.ledger import FileLedger, Ledger, locked
from rollout_train.record import STARTS, newest_record, table

GROUPS_PER_STEP = "groups_per_step"
MAX_LAG = "max_lag"
"""How many checkpoints behind the newest a turn of the trained channel may begin (0: only the newest)."""
EVALS_SUITE, EVALS_EVERY, EVALS_EPISODES = "evals.suite", "evals.every", "evals.episodes"
TRAINER = "trainer."
"""The start of a setting the trainer takes (`trainer.learning_rate`): what follows is its name for it."""
CHANGEABLE = (GROUPS_PER_STEP, MAX_LAG, EVALS_SUITE, EVALS_EVERY, EVALS_EPISODES)
"""The changeable settings every training run has; its trainer's (`trainer.…`) are beside them."""
PAUSED = "paused"
"""The key of a run's desired settings that says whether it is paused (`true`): not a setting a step takes."""


@dataclass(frozen=True)
class Desired:
    """What someone wants a run's changeable settings to be, by dotted key, and when that was last changed."""

    run: str
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    changed: float = 0.0


class DesiredSettings(Protocol):
    async def desired(self, run: str) -> Desired | None:
        """What is wanted of a run's settings, if anything is."""
        ...

    async def want(self, run: str, settings: Mapping[str, JsonValue]) -> Desired:
        """Want these settings of a run from now on: each key given is replaced, the others are kept."""
        ...


def desired_settings_of(ledger: Ledger) -> DesiredSettings | None:
    """The desired settings beside a ledger: a file beside a ledger of files, a table in a database ledger's
    database."""
    if isinstance(ledger, FileLedger):
        return FileDesiredSettings(ledger.directory)
    return getattr(ledger, "desired_settings", None)


async def paused(ledger: Ledger, run: str, store: DesiredSettings | None = None) -> bool:
    """Whether a run is paused: its desired settings (in `store`, by default those beside `ledger`) say so, or those of
    a run its newest start says it is played for: the training run whose schedule asked for it (`by`), or the eval it
    is a part of (`part_of`)."""
    store = store if store is not None else desired_settings_of(ledger)
    if store is None:
        return False
    seen: set[str] = set()
    waiting = [run]
    while waiting:
        each = waiting.pop()
        if each in seen:
            continue
        seen.add(each)
        found = await store.desired(each)
        if found is not None and found.settings.get(PAUSED) is True:
            return True
        starts = await ledger.read(table(each, STARTS))
        newest = newest_record(starts)
        waiting += [str(newest[key]) for key in ("by", "part_of") if newest.get(key)]
    return False


def checked(key: str, value: JsonValue) -> JsonValue:
    """A changeable setting's value, as a run takes it; raises `ValueError` for one it cannot take (`evals.every`
    below 1, say). A trainer's own settings are checked by the trainer."""
    if key == EVALS_EPISODES and value is None:  # (the suite's own)
        return None
    if key in (GROUPS_PER_STEP, EVALS_EVERY, EVALS_EPISODES):
        if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value or value < 1:
            raise ValueError(f"{key} is a whole number, 1 at least (not {value!r})")
        return int(value)
    if key == MAX_LAG:
        if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value or value < 0:
            raise ValueError(f"{key} is a whole number, 0 at least (not {value!r})")
        return int(value)
    if key == EVALS_SUITE:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} names a suite or one of its versions, or is null for no evals (not {value!r})")
        return value or None
    return value


def applied(current: Mapping[str, JsonValue], desired: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """The settings a run uses from its next step: `current` (its changeable settings in effect, by dotted key), with
    each desired one it has and can take in its place. A desired key it does not have, or a value it cannot take, is
    left as it is."""
    settings = dict(current)
    for key, value in desired.items():
        if key not in settings:
            continue
        try:
            settings[key] = checked(key, value)
        except ValueError:
            continue
    return settings


def fixed(profile: Any, trainer: Any, **loop: JsonValue) -> dict[str, JsonValue]:
    """A training run's fixed settings, by dotted key: what its profile (`rollout_train.profile.Profile`) says of its
    model, trainer, channels and machine, what its trainer makes (`weights`), and what the loop was started with
    (`loop`: `groups`, `seed`), less the settings its trainer takes between steps."""
    described = profile.trainer
    taken = set(_changeable_of(trainer))
    said: dict[str, Any] = {"model": profile.channels[described.channel].model, "weights": trainer.weights}
    said |= {"trainer.kind": described.kind, "trainer.channel": described.channel}
    said |= {"trainer.colocated": described.colocated}
    said |= {f"{TRAINER}{key}": value for key, value in described.settings.items() if key not in taken}
    for name, channel in profile.channels.items():
        said |= {f"channels.{name}.model": channel.model, f"channels.{name}.engine": channel.engine}
        said |= {f"channels.{name}.engines": len(channel.engines), f"channels.{name}.renderer": channel.renderer}
        said |= {f"channels.{name}.thinking_tokens": channel.thinking_tokens}
        said |= {f"channels.{name}.answer_tokens": channel.answer_tokens, f"channels.{name}.reshard": channel.reshard}
    said |= {"episodes_at_once": profile.episodes_at_once, **loop}
    return {key: _json(value) for key, value in said.items()}


def changeable(trainer: Any, *, groups_per_step: int, max_lag: int, evals: Any = None) -> dict[str, JsonValue]:
    """A training run's changeable settings as it starts, by dotted key: the groups a step waits for, how far behind
    the newest checkpoint a turn may begin, its evals (an `rollout_train.profile.EvalsSpec`, or none), and the settings
    its trainer takes between steps with their values."""
    said: dict[str, JsonValue] = {GROUPS_PER_STEP: groups_per_step, MAX_LAG: max_lag}
    said |= {EVALS_SUITE: evals.suite if evals else None, EVALS_EVERY: evals.every if evals else 1}
    said |= {EVALS_EPISODES: evals.episodes if evals else None}
    return said | {f"{TRAINER}{key}": _json(value) for key, value in _changeable_of(trainer).items()}


def _changeable_of(trainer: Any) -> Mapping[str, Any]:
    return getattr(trainer, "changeable", None) or {}


def _json(value: Any) -> JsonValue:
    return value if value is None or isinstance(value, str | int | float | bool) else json.loads(json.dumps(value))


class FileDesiredSettings:
    """`DesiredSettings` in `settings.json` in a ledger's directory, under the lock the ledger's files are written
    under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "settings.json"

    async def desired(self, run: str) -> Desired | None:
        return await asyncio.to_thread(lambda: self._read().get(run))

    async def want(self, run: str, settings: Mapping[str, JsonValue]) -> Desired:
        def changed() -> Desired:
            with locked(self.directory):
                every = self._read()
                was = every.get(run)
                now = Desired(run, {**(was.settings if was else {}), **settings}, round(time.time(), 1))
                every[run] = now
                staged = self.path.with_suffix(".staged")
                staged.write_text(json.dumps([asdict(each) for each in every.values()]))
                staged.replace(self.path)
                return now

        return await asyncio.to_thread(changed)

    def _read(self) -> dict[str, Desired]:
        if not self.path.exists():
            return {}
        return {each["run"]: Desired(**each) for each in json.loads(self.path.read_text())}
