"""`LoraTrainer` and `FullTrainer`: `Trainer`s that train a LoRA adapter over a model, or all of a model's weights: on
one GPU each step in a process of its own, on several in processes kept between steps, one per GPU, the policy sharded
over them."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout_lora.resident import Workers, visible_gpus
from rollout_lora.settings import LoraSettings
from rollout_lora.worker import TrainerProcess
from rollout_train.objectives import Objective
from rollout_train.trainer import Budget, Files, Item, Step


class LoraTrainer:
    """Trains a LoRA adapter over `model`'s checkpoint, one step at a time. `settings` are `LoraSettings`' fields (its
    `objective` among them); those in `CHANGEABLE`, and the changeable components of its objective, it takes between
    steps (`rollout_train.trainer.Changeable`). Its reference is the model with the adapter switched off.

    `gpus` is how many GPUs it steps on (by default those it is given: `CUDA_VISIBLE_DEVICES`, which Ray sets for a
    trainer's actor, else the machine's). On one, each step runs in a fresh process (`rollout_lora.worker`) that keeps
    nothing: a step starts from the adapter and the optimizer's state it is given and leaves the new ones where it is
    told. On more, the steps run in a process per GPU kept between steps (`rollout_lora.resident`), the policy sharded
    over them (`rollout_lora.sharded`): a step from the checkpoint the last one made goes on from what they hold
    (`rollout_train.trainer.Resident`), and every step still leaves the files a later one can start from."""

    weights = "lora"

    def __init__(self, model: str, *, gpus: int | None = None, **settings: Any) -> None:
        self.settings = LoraSettings(**settings)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        count = gpus if gpus is not None else visible_gpus()
        self._process: TrainerProcess | Workers = (
            Workers(model, self.settings, self.weights, count)
            if count > 1
            else TrainerProcess(model, self.settings, self.weights)
        )

    @property
    def objective(self) -> Objective:
        return self.settings.loss

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return self.settings.changeable()

    @property
    def holding(self) -> str | None:
        """What its processes hold between steps (none on one GPU, which holds nothing)."""
        return self._process.holding if isinstance(self._process, Workers) else None

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        self.settings = self.settings.changed(settings)
        self._process.settings = self.settings

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        return Step(await self._process.step(batch, seed=seed, parent=parent, into=into))

    def close(self) -> None:
        """End its processes on several GPUs, and what they hold (the next step starts them again)."""
        if isinstance(self._process, Workers):
            self._process.close()


class FullTrainer(LoraTrainer):
    """Trains every weight of a text model (`rollout_lora.full`): a step starts from its parent's weights (the model's
    own for the first) and the optimizer's state, and leaves the new ones where it is told; on several GPUs the
    weights, gradients and optimizer's state are sharded over them, and a step leaves the weights in bfloat16 (what
    engines serve) and the full state every `state_every` steps. `settings` are `LoraSettings`' fields; `rank` is not
    used. It holds a reference (a frozen copy of the model) only when asked (`frozen_reference`)."""

    weights = "full"
