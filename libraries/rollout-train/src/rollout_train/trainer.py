"""What training asks of a trainer, in terms that say nothing of where it runs or what it trains.

An algorithm decides what to train on: segments, and how much each should count (`Weighted`), for a policy gradient
or a likelihood; pairs of episodes' segments, one preferred to the other (`Pair`), or episodes' segments labelled
desirable or undesirable (`Labelled`), for a preference loss. Which the trainer's objective takes is its family's
(`rollout_train.objectives`), which a trainer says (`objective`; the `default` preset if it says none). A trainer takes
a batch, moves the policy, and says where the new weights are (`Step`). It also says what it can take (`Budget`): the
longest segment, and how many a step can afford. Those come from its hardware, and nothing above it chooses them. A
trainer may take some of its settings between steps (`Changeable`): a run's settings page changes them for the run's
next step (`rollout_train.settings`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from rollout_train.objectives import DEFAULT, Objective
from rollout_train.recorder import Segment


@dataclass(frozen=True)
class Weighted:
    """A segment to train on, and its advantage: every token the policy sampled in it counts by that much."""

    segment: Segment
    advantage: float
    source: str = ""
    """Where the segment is from, for the record of what a step trained on: `RUN/GROUP/EPISODE/SLOT/INDEX`."""


@dataclass(frozen=True)
class Pair:
    """Two sides over a shared context (one start), the chosen preferred to the rejected: each the segments of one
    episode (every turn the policy sampled in it), of which only the tokens the policy sampled count."""

    chosen: tuple[Segment, ...]
    rejected: tuple[Segment, ...]
    source: str = ""
    """Where the sides are from, for the record of what a step trained on: `RUN/GROUP/CHOSEN>REJECTED` (the episodes'
    numbers)."""

    @property
    def segments(self) -> tuple[Segment, ...]:
        return (*self.chosen, *self.rejected)


@dataclass(frozen=True)
class Labelled:
    """The segments of one episode, labelled desirable or not (KTO's unpaired examples)."""

    side: tuple[Segment, ...]
    desirable: bool
    source: str = ""
    """`RUN/GROUP/EPISODE`."""

    @property
    def segments(self) -> tuple[Segment, ...]:
        return self.side


type Item = Weighted | Pair | Labelled
"""What a batch holds: weighted segments, pairs or labelled examples (never a mixture)."""


def segments_of(item: Item) -> tuple[Segment, ...]:
    """The segments an item holds."""
    return (item.segment,) if isinstance(item, Weighted) else item.segments


def weight_of(item: Item) -> float:
    """What the record of a step says an item counted for: a segment's advantage, a pair's 1, an example's 1 or -1."""
    if isinstance(item, Weighted):
        return item.advantage
    if isinstance(item, Labelled):
        return 1.0 if item.desirable else -1.0
    return 1.0


@dataclass(frozen=True)
class Budget:
    segment_tokens: int | None = None
    """The longest segment the trainer can train on (None: any)."""
    segments: int | None = None
    """How many segments a step can afford (None: any number), counting each of a pair's or an example's."""


@dataclass(frozen=True)
class Files:
    """A checkpoint's files on this machine: what a step starts from."""

    weights: Path
    state: Path | None = None
    """What the trainer left for itself beside the weights (an optimizer's state, say), if it left any."""


@dataclass(frozen=True)
class Step:
    metrics: Mapping[str, float]


WEIGHTS = "weights"
"""Under a step's directory: the new weights, as engines load them."""
STATE = "state"
"""Under a step's directory: what the trainer goes on from, and what the step did."""


class StepFailed(Exception):
    """A step did not produce weights: the policy is as it was, and a later step may succeed."""


class Trainer(Protocol):
    """A trainer keeps nothing between steps that it cannot be given again: a step says what it starts from and
    where its files go, so any trainer can take any step of any policy."""

    budget: Budget
    weights: str
    """What its steps make: `lora` (an adapter over the weights the engines hold) or `full` (all the weights)."""

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        """Train on the batch, starting from `parent` (None: from the base model). The new weights are left in
        `into/weights`, and what a later step starts from in `into/state`. Raises `StepFailed` if the step
        produced no weights."""
        ...


def objective_of(trainer: object) -> Objective:
    """The objective a trainer trains with: its `objective`, else the `default` preset."""
    found = getattr(trainer, "objective", None)
    return found if isinstance(found, Objective) else DEFAULT


@runtime_checkable
class Changeable(Protocol):
    """A trainer that takes some of its settings between steps (its learning rate, say): those that change neither
    what its weights are nor what it can take (`Budget`)."""

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        """The settings it takes between steps, by its name for each, with their values now: a component of its
        objective by its run setting's key (`objective.kl.coefficient`)."""
        ...

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        """Take these settings (some of `changeable`) from its next step on. Raises `ValueError` for one it does not
        take, or a value it cannot."""
        ...
