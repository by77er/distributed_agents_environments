"""An environment: what a run trains on and an eval measures.

Whoever builds an environment supplies one: the program that plays an episode; the situations it can be set in (rows,
easiest first) and how one start of a row is drawn; its eval data (named lists of starts, never drawn for training);
what its results say (a `Description`); and a version. It may supply a curriculum of its own (`curriculum()`, a
`rollout.curriculum.Curriculum`); without one, the generic curriculum decides what is trained on next. Every run of a
group is given the same start, so that their differences are the players' doing. Nothing else about the environment
is known to what trains on it: how an episode went comes back in its result (`Episode.info`).

**Train and eval.** Training draws each start through `train_start`, which draws again while what it drew is one of
the environment's eval starts: no training group is given an eval start, whatever the environment's `start` does with
its random numbers. (Seeds kept apart would hold only for an environment whose starts differ whenever their seeds do;
comparing the starts themselves holds for every environment.) A start is told apart by its parameters alone, so the
guarantee is as fine as they are: two starts that differ only in a number the episode never reads are the same
situation. An eval that should hold out whole situations lists starts of rows the environment does not train on.
"""

import json
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from pydantic import JsonValue

from rollout.harness.imports import ToolBinding
from rollout.harness.runner import ProgramReference, RunBinding, bind, with_row
from rollout.harness.sandboxes import PoolBinding

__all__ = [
    "Description",
    "Environment",
    "Row",
    "Start",
    "binding_for",
    "drawn",
    "held_out",
    "start_key",
    "train_start",
]


@dataclass(frozen=True)
class Row:
    key: str
    """Its name among the environment's rows."""
    title: str
    """What it is, for people."""
    parameters: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    counts_for: tuple[str, ...] = ()
    """The keys of other rows that a group of this one is evidence about too: the same situation with more help, say.
    What it teaches about this row it teaches about them."""


@dataclass(frozen=True)
class Start:
    """One start of a row, drawn with a seed of its own: what an eval plays."""

    task: str
    """The row's key."""
    title: str
    seed: int
    parameters: JsonValue
    """What every episode of it is given: the row's start, drawn with `seed`."""


@dataclass(frozen=True)
class Description:
    """What an environment's results say, for whatever shows or compares them."""

    rewards: tuple[float | None, float | None] = (0.0, 1.0)
    """The range an episode's reward falls in (None: no bound on that side)."""
    solved: bool = True
    """Whether its results say `solved`."""
    saturated: bool = False
    """Whether its results say `saturated` (nothing was left to earn)."""
    duration: str | None = None
    """What a result's `duration` counts (`turns`, `minutes of game time`); None: its results say no duration."""
    observations: str | None = None
    """How its observations are shown (`minecraft`, say); None: as text."""

    def to_json(self) -> dict[str, JsonValue]:
        said = asdict(self)
        return {**said, "rewards": list(self.rewards)}


class Environment(Protocol):
    @property
    def program(self) -> ProgramReference:
        """What a run executes; its parameters are a start."""
        ...

    @property
    def version(self) -> str:
        """Changed whenever its rows, starts, eval data or scoring change: runs and suites record it."""
        ...

    @property
    def description(self) -> Description: ...

    def rows(self) -> Sequence[Row]:
        """Every situation, easiest first."""
        ...

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        """The parameters of one start of `row` (a seed drawn with `rng`, say): what every run of a group is given."""
        ...

    def evals(self) -> Mapping[str, Sequence[Start]]:
        """Its eval data: named lists of starts, never drawn for training. Each is frozen as a suite of its name the
        first time it is played (`rollout_train.evals`). `drawn` derives one from rows and seeds."""
        ...


def drawn(environment: Environment, *, seeds: Sequence[int], rows: Sequence[str] | None = None) -> list[Start]:
    """A start of each row (of `rows`, by key; else every row) for each seed, drawn with `random.Random(seed)`: eval
    data derived from rows and seeds. Raises `ValueError` for a row the environment lacks, or no seeds."""
    known = {row.key: row for row in environment.rows()}
    if missing := [key for key in rows or [] if key not in known]:
        raise ValueError(f"the environment has no row {', '.join(missing)}")
    if not seeds:
        raise ValueError("eval starts need a seed at least")
    chosen = [known[key] for key in rows] if rows else list(known.values())
    return [
        Start(row.key, row.title, seed, environment.start(row, random.Random(seed))) for row in chosen for seed in seeds
    ]


def start_key(parameters: JsonValue) -> str:
    """A start's parameters as canonical JSON: two starts are the same start when their keys are equal."""
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def held_out(environment: Environment) -> frozenset[str]:
    """The keys of every one of the environment's eval starts (`start_key`): what training never draws."""
    return frozenset(start_key(start.parameters) for starts in environment.evals().values() for start in starts)


DRAWS = 100
"""Starts `train_start` draws of a row before it gives up: past that, the row's starts are all eval starts."""


def train_start(environment: Environment, row: Row, rng: random.Random, held: Collection[str]) -> JsonValue:
    """A start of `row` for training, drawn with `rng`, and drawn again while it is an eval start (its key in `held`,
    `held_out`). Raises `ValueError` when `DRAWS` draws in a row are."""
    for _ in range(DRAWS):
        parameters = environment.start(row, rng)
        if start_key(parameters) not in held:
            return parameters
    raise ValueError(f"every start of {row.key} drawn was an eval start: training has none of it to draw")


def binding_for(
    environment: Environment,
    channel: str,
    tools: Mapping[str, ToolBinding] | None = None,
    pools: Mapping[str, PoolBinding] | None = None,
) -> RunBinding:
    """How an environment's runs are served: every model slot of its program from `channel`, each of its imports from
    the tool set of its own name, or where `tools` says, and each kind of sandbox from the pool of its own name, or
    where `pools` says. (A program says which slots, imports and sandboxes it has once it is given a row: the
    environment's first.)"""
    first = with_row(environment.program, environment.start(environment.rows()[0], random.Random(0)))
    return bind(first, channel, tools=tools, pools=pools)
