"""`LoraTrainer` and `FullTrainer`: `Trainer`s that train a LoRA adapter over a model, or all of a model's weights,
each step in a process of its own."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rollout_lora.settings import LoraSettings
from rollout_lora.worker import TrainerProcess
from rollout_train.trainer import Budget, Files, Step, Weighted


class LoraTrainer:
    """Trains a LoRA adapter over `model`'s checkpoint, one step at a time, each in a fresh process on the GPU
    (`rollout_lora.worker`). It keeps nothing between steps: a step starts from the adapter and the optimizer's
    state it is given and leaves the new ones where it is told. `settings` are `LoraSettings`' fields."""

    weights = "lora"

    def __init__(self, model: str, **settings: Any) -> None:
        self.settings = LoraSettings(**settings)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        self._process = TrainerProcess(model, self.settings, self.weights)

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        return Step(await self._process.step(batch, seed=seed, parent=parent, into=into))


class FullTrainer(LoraTrainer):
    """Trains every weight of a text model (`rollout_lora.full`), one step at a time in a fresh process: a step
    starts from its parent's full weights (the model's own for the first) and the optimizer's state, and leaves the
    new ones where it is told. `settings` are `LoraSettings`' fields; the adapter's (`rank`) are not used."""

    weights = "full"
