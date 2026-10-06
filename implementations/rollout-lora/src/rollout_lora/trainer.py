"""`LoraTrainer` and `FullTrainer`: `Trainer`s that train a LoRA adapter over a model, or all of a model's weights, in
a process per GPU (`rollout_lora.resident.Workers`): kept between steps where the trainer has its GPUs to itself, a
fresh one for each step beside an engine. Kept, they keep each step's full state in a blob store after the step, where
they are told one (`rollout_train.trainer.Keeps`, as a training pod tells them)."""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout_lora.resident import Workers, visible_gpus
from rollout_lora.settings import LoraSettings
from rollout_train.checkpoints import Manifest
from rollout_train.objectives import Objective
from rollout_train.trainer import Budget, Files, Item, Step


class LoraTrainer:
    """Trains a LoRA adapter over `model`'s checkpoint, one step at a time. `settings` are `LoraSettings`' fields (its
    `objective` among them); those in `CHANGEABLE`, and the changeable components of its objective, it takes between
    steps (`rollout_train.trainer.Changeable`). Its reference is the model with the adapter switched off.

    `gpus` is how many GPUs it steps on (by default those it is given: `CUDA_VISIBLE_DEVICES`, which Ray sets for a
    trainer's actor, else the machine's), a process on each (`rollout_lora.workers`), the policy sharded over them on
    more than one (`rollout_lora.sharded`). `colocated` says it shares its GPU with an inference engine, which sleeps
    while it steps: then each step's processes end after it, and give the engine back the memory. Otherwise they are
    kept between steps: a step from the checkpoint the last one made goes on from what they hold
    (`rollout_train.trainer.Resident`). Every step leaves the files a later one starts from (the full state every
    `state_every` steps): in `into/state` before it returns, or, told a blob store (`keep_in`), kept there by the
    processes after it returns (`kept`), its weights served meanwhile."""

    weights = "lora"

    def __init__(self, model: str, *, gpus: int | None = None, colocated: bool = False, **settings: Any) -> None:
        self.settings = LoraSettings(**settings)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        count = gpus if gpus is not None else visible_gpus()
        self._process = Workers(model, self.settings, self.weights, max(1, count), kept=not colocated)

    @property
    def objective(self) -> Objective:
        return self.settings.loss

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return self.settings.changeable()

    @property
    def holding(self) -> str | None:
        """What its processes hold between steps (none beside an engine, where they end after each step)."""
        return self._process.holding

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        self.settings = self.settings.changed(settings)
        self._process.settings = self.settings

    def keep_in(self, blobs: Mapping[str, JsonValue]) -> None:
        """Have the processes keep each step's full state in the blob store at this location after the step returns
        (where they are kept between steps: beside an engine they end after each step, and write it before)."""
        if self._process.kept:
            self._process.keep = dict(blobs)

    async def kept(self, into: str) -> Manifest:
        """What the processes kept of the full state of the step that wrote into `into`, once they all have."""
        files = await asyncio.to_thread(self._process.kept_state, into)
        return Manifest({path: BlobReference.model_validate(reference) for path, reference in files.items()})

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        metrics = await self._process.step(batch, seed=seed, parent=parent, into=into)
        return Step(metrics, keeping=self._process.keeping(into.name))

    def close(self) -> None:
        """End its processes, and what they hold (the next step starts them again)."""
        self._process.close()


class FullTrainer(LoraTrainer):
    """Trains every weight of a text model (`rollout_lora.full`): a step starts from its parent's weights (the model's
    own for the first) and the optimizer's state, and leaves the new ones where it is told. The weights, gradients and
    optimizer's state are sharded over its GPUs (on one too), and a step leaves the weights in bfloat16 (what engines
    serve) and the full state (the float32 weights and the optimizer's) every `state_every` steps. `settings` are
    `LoraSettings`' fields; `rank` is not used. It holds a reference (a frozen copy of the model) only when asked
    (`frozen_reference`)."""

    weights = "full"
