"""What training asks of a trainer, in terms that say nothing of where it runs or what it trains.

An algorithm decides which sequences to train on and how much each should count (`Weighted`). A trainer takes a
batch, moves the policy, and says where the new weights are (`Step`). It also says what it can take (`Budget`): the
longest sequence, and how many a step can afford. Those come from its hardware, and nothing above it chooses them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rollout_train.recorder import Epoch


@dataclass(frozen=True)
class Weighted:
    """A sequence to train on, and its advantage: every token the policy sampled in it counts by that much."""

    epoch: Epoch
    advantage: float
    source: str = ""
    """Where the sequence is from, for the record of what a step trained on: `cursor/slot/index` in the job's log."""


@dataclass(frozen=True)
class Budget:
    sequence_tokens: int | None = None
    """The longest sequence the trainer can train on (None: any)."""
    sequences: int | None = None
    """How many sequences a step can afford (None: any number)."""


@dataclass(frozen=True)
class Step:
    adapter: str
    """The name of the new weights."""
    path: str
    """Where engines read them."""
    metrics: Mapping[str, float]
    artifacts: Mapping[str, str] = field(default_factory=dict[str, str])
    """What the step left behind, by name, as paths of files and directories: the weights, and whatever the trainer
    would need to go on from exactly here. Whoever keeps a record of the run keeps these."""


class StepFailed(Exception):
    """A step did not produce weights: the policy is as it was, and a later step may succeed."""


class Trainer(Protocol):
    budget: Budget

    @property
    def latest(self) -> tuple[str, str] | None:
        """The newest weights' name and path, if a step has been taken (by this trainer or one before it)."""
        ...

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        """Train on the batch, and return the new weights. Raises `StepFailed` if the step produced none."""
        ...
