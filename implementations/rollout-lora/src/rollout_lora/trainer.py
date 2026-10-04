"""`LoraTrainer` and `FullTrainer`: `Trainer`s that train a LoRA adapter over a model, or all of a model's weights,
each step in a process of its own."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout_lora.settings import LoraSettings
from rollout_lora.worker import TrainerProcess
from rollout_train.trainer import Budget, Files, Step, Weighted


class LoraTrainer:
    """Trains a LoRA adapter over `model`'s checkpoint, one step at a time, each in a fresh process on the GPU
    (`rollout_lora.worker`). It keeps nothing between steps: a step starts from the adapter and the optimizer's
    state it is given and leaves the new ones where it is told. `settings` are `LoraSettings`' fields; those in
    `CHANGEABLE` it takes between steps (`rollout_train.trainer.Changeable`)."""

    weights = "lora"

    def __init__(self, model: str, **settings: Any) -> None:
        self.settings = LoraSettings(**settings)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        self._process = TrainerProcess(model, self.settings, self.weights)

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return self.settings.changeable()

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        self.settings = self.settings.changed(settings)
        self._process.settings = self.settings

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        return Step(await self._process.step(batch, seed=seed, parent=parent, into=into))


class FullTrainer(LoraTrainer):
    """Trains every weight of a text model (`rollout_lora.full`), one step at a time in a fresh process: a step
    starts from its parent's full weights (the model's own for the first) and the optimizer's state, and leaves the
    new ones where it is told. `settings` are `LoraSettings`' fields; `rank` is not
    used."""

    weights = "full"
