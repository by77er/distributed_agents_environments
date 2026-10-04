"""What training asks of a trainer, in terms that say nothing of where it runs or what it trains.

An algorithm decides which segments to train on and how much each should count (`Weighted`). A trainer takes a
batch, moves the policy, and says where the new weights are (`Step`). It also says what it can take (`Budget`): the
longest segment, and how many a step can afford. Those come from its hardware, and nothing above it chooses them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rollout_train.recorder import Segment


@dataclass(frozen=True)
class Weighted:
    """A segment to train on, and its advantage: every token the policy sampled in it counts by that much."""

    segment: Segment
    advantage: float
    source: str = ""
    """Where the segment is from, for the record of what a step trained on: `RUN/GROUP/EPISODE/SLOT/INDEX`."""


@dataclass(frozen=True)
class Budget:
    segment_tokens: int | None = None
    """The longest segment the trainer can train on (None: any)."""
    segments: int | None = None
    """How many segments a step can afford (None: any number)."""


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

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        """Train on the batch, starting from `parent` (None: from the base model). The new weights are left in
        `into/weights`, and what a later step starts from in `into/state`. Raises `StepFailed` if the step
        produced no weights."""
        ...
