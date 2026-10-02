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
import json
import multiprocessing
import traceback
from collections.abc import Sequence
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from rollout.processes import end_with_parent
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import STATE, WEIGHTS, Checkpoint, StepFailed, Weighted


class TrainerProcess:
    def __init__(self, checkpoint: str, settings: LoraSettings) -> None:
        self.checkpoint = checkpoint
        self.settings = settings
        self._lock = asyncio.Lock()
        self._process: BaseProcess | None = None

    async def step(
        self, sequences: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path
    ) -> dict[str, float]:
        """Train one step on the GPU (the engine must have freed it) from `parent`, and leave the adapter in
        `into/weights` and the optimizer's state in `into/state`."""
        async with self._lock:
            try:
                return await asyncio.to_thread(self._run, list(sequences), seed, parent, into)
            except asyncio.CancelledError:  # whoever waited is gone: the step is not left running for nobody
                if self._process is not None and self._process.is_alive():
                    self._process.terminate()
                raise

    def _run(self, sequences: list[Weighted], seed: int, parent: Checkpoint | None, into: Path) -> dict[str, float]:
        context = multiprocessing.get_context("spawn")
        ours, child = context.Pipe()
        arguments = (child, self.checkpoint, self.settings, sequences, seed, parent, into)
        process = context.Process(target=_step, args=arguments, name="trainer")
        process.start()
        self._process = process
        child.close()
        try:
            kind, payload = ours.recv()
        except EOFError:
            process.join()
            raise StepFailed(f"the trainer exited without a result (exit code {process.exitcode})") from None
        process.join()
        if kind == "error":
            raise StepFailed(f"the trainer failed:\n{payload}")
        return payload


OPTIMIZER = "optimizer.pt"
"""In a step's state: the optimizer's state after it."""
MINIBATCHES = "minibatches.jsonl"
"""In a step's state: what each of its minibatches did, one line each."""
MEMORY_MARGIN = 256 * 2**20
"""GPU memory left free of what was free when a step started (other programs' use moves a little)."""


def _step(
    connection: Connection,
    checkpoint: str,
    settings: LoraSettings,
    sequences: list[Weighted],
    seed: int,
    parent: Checkpoint | None,
    into: Path,
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

        from rollout_lora.layers import load_adapter
        from rollout_lora.policy import Policy
        from rollout_lora.step import ClippedPolicyGradient

        policy = Policy.load(checkpoint, rank=settings.rank, alpha=settings.alpha)
        if parent is not None:
            load_adapter(policy.model, parent.weights)
        trainer = ClippedPolicyGradient(policy, settings)
        if parent is not None and parent.state is not None and (parent.state / OPTIMIZER).exists():
            trainer.optimizer.load_state_dict(torch.load(parent.state / OPTIMIZER, map_location="cuda"))
            for group in trainer.optimizer.param_groups:  # (the saved state carries the rate it was saved with)
                group["lr"] = settings.learning_rate
        metrics: dict[str, Any] = trainer.step(sequences, seed=seed)
        policy.save(into / WEIGHTS)
        (into / STATE).mkdir(parents=True, exist_ok=True)
        torch.save(trainer.optimizer.state_dict(), into / STATE / OPTIMIZER)
        (into / STATE / MINIBATCHES).write_text("".join(json.dumps(each) + "\n" for each in trainer.minibatches))
        metrics["peak_gpu_gib"] = torch.cuda.max_memory_reserved() / 2**30
        metrics["free_gpu_gib"] = free / 2**30  # when the step started: what it was allowed, less the margin
        connection.send(("done", metrics))
    except Exception:
        connection.send(("error", traceback.format_exc()))
