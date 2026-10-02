"""`LoraTrainer`: a `Trainer` that trains a LoRA adapter over a 4-bit checkpoint, each step in a process of its own."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rollout.lora.settings import LoraSettings
from rollout.lora.worker import Job, TrainerProcess
from rollout.training.trainer import Budget, Step, Weighted


class LoraTrainer:
    """Trains a LoRA adapter over `model`'s checkpoint, one step at a time, each in a fresh process on the GPU
    (`rollout.lora.worker`). Adapters are kept under `directory/adapters/step-N`, and the optimizer's state under
    `directory/trainer`; a trainer made again over the same directory goes on from the latest step. `settings` are
    `LoraSettings`' fields."""

    def __init__(self, model: str, directory: Path, **settings: Any) -> None:
        self.settings = LoraSettings(**settings)
        self.budget = Budget(self.settings.sequence_tokens, self.settings.sequences_per_step)
        self._adapters = directory / "adapters"
        self._adapters.mkdir(parents=True, exist_ok=True)
        self._process = TrainerProcess(Job(model, directory / "trainer", self.settings))
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
