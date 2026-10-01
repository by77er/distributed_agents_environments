"""The trainer in a process of its own, next to an inference engine.

Each step runs in a fresh spawned process: it loads the policy onto the GPU, restores the LoRA weights and the
optimizer's state from the previous step's files, trains, saves and exits. Exiting is what frees the GPU and the
memory: a trainer parked in system memory between steps, next to a sleeping engine's offloaded weights, can exhaust a
small machine. An engine's client libraries can also change how transformers builds models in the process that uses
them (vLLM swaps in its own configuration classes), which a separate process avoids.
"""

import asyncio
import multiprocessing
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from rollout.training.grpo import TrainingSequence


@dataclass(frozen=True)
class TrainerSettings:
    checkpoint: str
    state: Path
    """Where the optimizer's state is kept between steps."""
    rank: int = 32
    alpha: float = 64.0
    learning_rate: float = 2e-5
    max_sequence_tokens: int | None = None
    """Sequences longer than this are left out of a step (see `GroupRelativeTrainer`)."""


class TrainerProcess:
    def __init__(self, settings: TrainerSettings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    async def step(
        self, sequences: Sequence[TrainingSequence], *, seed: int, adapter: Path, previous: Path | None
    ) -> dict[str, float]:
        """Train one step on the GPU (the engine must have freed it) from the `previous` adapter, and save `adapter`."""
        async with self._lock:
            return await asyncio.to_thread(self._run, list(sequences), seed, adapter, previous)

    def _run(
        self, sequences: list[TrainingSequence], seed: int, adapter: Path, previous: Path | None
    ) -> dict[str, float]:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_step, args=(child, self.settings, sequences, seed, adapter, previous), name="trainer"
        )
        process.start()
        child.close()
        try:
            kind, payload = parent.recv()
        except EOFError:
            process.join()
            raise RuntimeError(f"the trainer exited without a result (exit code {process.exitcode})") from None
        process.join()
        if kind == "error":
            raise RuntimeError(f"the trainer failed:\n{payload}")
        return payload


def _step(
    connection: Connection,
    settings: TrainerSettings,
    sequences: list[TrainingSequence],
    seed: int,
    adapter: Path,
    previous: Path | None,
) -> None:
    try:
        import torch

        from rollout.training.grpo import GroupRelativeTrainer
        from rollout.training.lora import load_adapter
        from rollout.training.policy import Policy

        policy = Policy.load(settings.checkpoint, rank=settings.rank, alpha=settings.alpha)
        if previous is not None:
            load_adapter(policy.model, previous)
        trainer = GroupRelativeTrainer(
            policy, learning_rate=settings.learning_rate, max_sequence_tokens=settings.max_sequence_tokens
        )
        optimizer_state = settings.state / "optimizer.pt"
        if previous is not None and optimizer_state.exists():
            trainer.optimizer.load_state_dict(torch.load(optimizer_state, map_location="cuda"))
        metrics: dict[str, Any] = trainer.step(sequences, seed=seed)
        policy.save(adapter)
        settings.state.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.optimizer.state_dict(), optimizer_state)
        metrics["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        connection.send(("done", metrics))
    except Exception:
        connection.send(("error", traceback.format_exc()))
