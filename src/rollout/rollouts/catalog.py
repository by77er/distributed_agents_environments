"""A catalog: what an environment offers to be trained on, as rows.

Whoever builds an environment supplies one: the program that plays an episode, the situations it can be set in
(rows, easiest first), and how one start of a row is drawn. Every run of a group is given the same start, so that
their differences are the players' doing. Nothing else about the environment is known to what trains on it: how an
episode went comes back in its result (`Episode.info`).
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue

from rollout.core.harness.runner import ProgramReference


@dataclass(frozen=True)
class Row:
    key: str
    """Its name among the catalog's rows."""
    title: str
    """What it is, for people."""
    parameters: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


class Catalog(Protocol):
    program: ProgramReference
    """What a run executes; its parameters are a start."""

    def rows(self) -> Sequence[Row]:
        """Every situation, easiest first."""
        ...

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        """The parameters of one start of `row` (a seed drawn with `rng`, say): what every run of a group is given."""
        ...
