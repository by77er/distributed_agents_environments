"""The trainer in a process of its own, next to an inference engine.

Each step runs in a fresh spawned process: it loads the policy onto the GPU, restores the LoRA weights and the
optimizer's state from the previous step's files, trains, saves and exits. Exiting is what frees the GPU and the
memory: a trainer parked in system memory between steps, next to a sleeping engine's offloaded weights, can exhaust a
small machine. An engine's client libraries can also change how transformers builds models in the process that uses
them (vLLM swaps in its own configuration classes), which a separate process avoids.

The process may use the GPU memory that is free when it starts and no more: where a driver lets a process spill into
system memory (Windows does), a step that needs more would crawl instead of failing, and a minibatch that fails is
counted and left out (`minibatches_out_of_memory`). The process ends with the process that started it.
"""

import asyncio
import multiprocessing
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from rollout.processes import end_with_parent
from rollout.training.trainer import Weighted


@dataclass(frozen=True)
class TrainerSettings:
    checkpoint: str
    state: Path
    """Where the optimizer's state is kept between steps."""
    rank: int = 32
    alpha: float = 64.0
    learning_rate: float = 5e-5
    max_sequence_tokens: int | None = None
    """Sequences longer than this are left out of a step (see `GroupRelativeTrainer`)."""


class TrainerProcess:
    def __init__(self, settings: TrainerSettings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._process: BaseProcess | None = None

    async def step(
        self, sequences: Sequence[Weighted], *, seed: int, adapter: Path, previous: Path | None
    ) -> dict[str, float]:
        """Train one step on the GPU (the engine must have freed it) from the `previous` adapter, and save `adapter`."""
        async with self._lock:
            try:
                return await asyncio.to_thread(self._run, list(sequences), seed, adapter, previous)
            except asyncio.CancelledError:  # whoever waited is gone: the step is not left running for nobody
                if self._process is not None and self._process.is_alive():
                    self._process.terminate()
                raise

    def _run(self, sequences: list[Weighted], seed: int, adapter: Path, previous: Path | None) -> dict[str, float]:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_step, args=(child, self.settings, sequences, seed, adapter, previous), name="trainer"
        )
        process.start()
        self._process = process
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


MEMORY_MARGIN = 256 * 2**20
"""GPU memory left free of what was free when a step started (other programs' use moves a little)."""


def _step(
    connection: Connection,
    settings: TrainerSettings,
    sequences: list[Weighted],
    seed: int,
    adapter: Path,
    previous: Path | None,
) -> None:
    try:
        import os

        end_with_parent()
        # Reserve close to what is used: fragmentation would otherwise cost about 0.7 GiB at the peak.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        free, total = torch.cuda.mem_get_info()
        allowed = max(0.05, min(1.0, (free - MEMORY_MARGIN) / total))
        torch.cuda.set_per_process_memory_fraction(allowed)  # pyright: ignore[reportUnknownMemberType]

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
            for group in trainer.optimizer.param_groups:  # (the saved state carries the rate it was saved with)
                group["lr"] = settings.learning_rate
        metrics: dict[str, Any] = trainer.step(sequences, seed=seed)
        policy.save(adapter)
        settings.state.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.optimizer.state_dict(), optimizer_state)
        metrics["peak_gpu_gib"] = torch.cuda.max_memory_reserved() / 2**30
        metrics["free_gpu_gib"] = free / 2**30  # when the step started: what it was allowed, less the margin
        connection.send(("done", metrics))
    except Exception:
        connection.send(("error", traceback.format_exc()))
