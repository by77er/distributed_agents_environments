"""What training asks of a trainer, in terms that say nothing of where it runs or what it trains.

An algorithm decides which sequences to train on and how much each should count (`Weighted`). A trainer takes a
batch, moves the policy, and says where the new weights are (`Step`). It also says what it can take (`Budget`): the
longest sequence, and how many a step can afford. Those come from its hardware, and nothing above it chooses them.

`LoraTrainer` trains a LoRA adapter in a process of its own (`rollout.training.worker`). `Colocated` wraps any
trainer that shares an accelerator with the engines serving the policy: they sleep while it steps.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rollout.inference import Channel
from rollout.recorder import Epoch


@dataclass(frozen=True)
class Weighted:
    """A sequence to train on, and its advantage: every token the policy sampled in it counts by that much."""

    epoch: Epoch
    advantage: float


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


class Trainer(Protocol):
    budget: Budget

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        """Train on the batch, and return the new weights."""
        ...


class LoraTrainer:
    """Trains a LoRA adapter, one step at a time, each in a fresh process on the GPU. Adapters are kept under
    `directory/adapters/step-N`, and the optimizer's state under `directory/trainer`; a trainer made again over the
    same directory goes on from the latest step."""

    def __init__(
        self,
        checkpoint: str,
        directory: Path,
        *,
        rank: int = 32,
        learning_rate: float = 5e-5,
        budget: Budget | None = None,
    ) -> None:
        from rollout.training.worker import TrainerProcess, TrainerSettings

        self.budget = budget = budget or Budget()
        self._adapters = directory / "adapters"
        self._adapters.mkdir(parents=True, exist_ok=True)
        self._process = TrainerProcess(
            TrainerSettings(
                checkpoint,
                state=directory / "trainer",
                rank=rank,
                alpha=2.0 * rank,
                learning_rate=learning_rate,
                max_sequence_tokens=budget.sequence_tokens,
            )
        )
        steps = [
            int(path.name.removeprefix("step-"))
            for path in self._adapters.glob("step-*")
            if (path / "adapter_config.json").exists()
        ]
        self.steps = max(steps, default=0)
        """Steps taken so far (over every process that has used the directory)."""

    @property
    def latest(self) -> tuple[str, str] | None:
        """The newest adapter's name and path, if a step has been taken."""
        return (f"step-{self.steps}", str(self._adapters / f"step-{self.steps}")) if self.steps else None

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        name = f"step-{self.steps + 1}"
        previous = self._adapters / f"step-{self.steps}" if self.steps else None
        metrics = await self._process.step(batch, seed=seed, adapter=self._adapters / name, previous=previous)
        self.steps += 1
        return Step(name, str(self._adapters / name), metrics)


class Colocated:
    """A trainer that shares an accelerator with the engines of some channels: requests to them are held back and
    the engines sleep while it steps. `guard` is called once they are asleep and raises if the step should not
    start (too little memory, say)."""

    def __init__(
        self, trainer: Trainer, channels: Sequence[Channel], *, guard: Callable[[], None] | None = None
    ) -> None:
        self._trainer = trainer
        self._channels = channels
        self._guard = guard
        self.budget = trainer.budget

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        started = time.monotonic()
        for channel in self._channels:
            await channel.pause()  # no request may be in flight when an engine goes to sleep
        waited = time.monotonic() - started
        try:
            for channel in self._channels:
                await channel.sleep()
            try:
                if self._guard is not None:
                    self._guard()
                step = await self._trainer.step(batch, seed=seed)
            finally:
                for channel in self._channels:
                    await channel.wake()
        finally:
            for channel in self._channels:
                channel.resume()
        timing = {"waited_for_requests_seconds": waited, "update_seconds": time.monotonic() - started}
        return Step(step.adapter, step.path, {**step.metrics, **timing})
