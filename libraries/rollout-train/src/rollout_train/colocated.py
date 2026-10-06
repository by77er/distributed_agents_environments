"""A trainer that shares an accelerator with the engines serving the policy: they take turns."""

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from rollout_train.objectives import Objective
from rollout_train.trainer import (
    Changeable,
    Files,
    Item,
    Progress,
    Progressing,
    Resident,
    Step,
    Trainer,
    objective_of,
)


class Pausable(Protocol):
    """What sharing an accelerator needs of whatever serves the policy (`rollout_train.inference.Channel` is one)."""

    async def pause(self) -> None: ...
    def resume(self) -> None: ...
    async def sleep(self) -> None: ...
    async def wake(self) -> None: ...


class Colocated:
    """A trainer that shares an accelerator with the engines of some channels: requests to them are held back and
    the engines sleep while it steps. `guard` is called once they are asleep and raises if the step should not
    start (too little memory, say)."""

    def __init__(
        self, trainer: Trainer, channels: Sequence[Pausable], *, guard: Callable[[], None] | None = None
    ) -> None:
        self._trainer = trainer
        self._channels = channels
        self._guard = guard
        self.budget = trainer.budget
        self.weights = trainer.weights
        self.objective: Objective = objective_of(trainer)

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        """The settings the trainer it wraps takes between steps (`rollout_train.trainer.Changeable`), if any."""
        return self._trainer.changeable if isinstance(self._trainer, Changeable) else {}

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        if not isinstance(self._trainer, Changeable):
            raise ValueError(f"the trainer takes no settings between steps (not {', '.join(settings)})")
        self._trainer.change(settings)

    @property
    def holding(self) -> str | None:
        """What the trainer it wraps holds between steps (`rollout_train.trainer.Resident`), if it holds anything."""
        return self._trainer.holding if isinstance(self._trainer, Resident) else None

    def watch(self, told: Callable[[Progress], None] | None) -> None:
        """Have the trainer it wraps tell `told` how far each step has got, where it says (`Progressing`)."""
        if isinstance(self._trainer, Progressing):
            self._trainer.watch(told)

    def close(self) -> None:
        """End what the trainer it wraps keeps running between steps, if it keeps anything."""
        if isinstance(self._trainer, Resident):
            self._trainer.close()

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
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
                step = await self._trainer.step(batch, seed=seed, parent=parent, into=into)
            finally:
                for channel in self._channels:
                    await channel.wake()
        finally:
            for channel in self._channels:
                channel.resume()
        timing = {"waited_for_requests_seconds": waited, "update_seconds": time.monotonic() - started}
        return Step({**step.metrics, **timing})
