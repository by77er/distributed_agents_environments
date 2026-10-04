"""A trainer that shares an accelerator with the engines serving the policy: they take turns."""

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from rollout_train.trainer import Files, Step, Trainer, Weighted


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

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
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
