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

from rollout.harness.imports import ToolBinding
from rollout.harness.runner import ProgramReference, RunBinding, bind, with_row

__all__ = ["Catalog", "Row", "binding_for"]


@dataclass(frozen=True)
class Row:
    key: str
    """Its name among the catalog's rows."""
    title: str
    """What it is, for people."""
    parameters: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    counts_for: tuple[str, ...] = ()
    """The keys of other rows that a group of this one is evidence about too: the same situation with more help, say.
    What it teaches about this row it teaches about them."""


class Catalog(Protocol):
    program: ProgramReference
    """What a run executes; its parameters are a start."""

    def rows(self) -> Sequence[Row]:
        """Every situation, easiest first."""
        ...

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        """The parameters of one start of `row` (a seed drawn with `rng`, say): what every run of a group is given."""
        ...


def binding_for(catalog: Catalog, channel: str, tools: Mapping[str, ToolBinding] | None = None) -> RunBinding:
    """How a catalog's runs are served: every model slot of its program from `channel`, and each of its imports
    from the tool set of its own name, or where `tools` says. (A program says which slots and imports it has once
    it is given a row: the catalog's first.)"""
    first = with_row(catalog.program, catalog.start(catalog.rows()[0], random.Random(0)))
    return bind(first, channel, tools=tools)
