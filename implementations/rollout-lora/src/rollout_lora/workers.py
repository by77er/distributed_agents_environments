# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of the distributed API untyped.)
"""The processes a trainer on several GPUs steps in, one per GPU, each holding its shard of the policy and of the
optimizer between steps (`rollout_lora.sharded`). `rollout_lora.resident.Workers` starts them under torchrun:

    python -m torch.distributed.run --standalone --nproc-per-node N -m rollout_lora.workers

Each connects to the trainer that started them (`ROLLOUT_WORKERS`, its address, `HOST:PORT`; `ROLLOUT_WORKERS_KEY`,
the key it takes connections with, in hex), joins the others (`ROLLOUT_WORKERS_DEVICE`: `cuda`, a GPU each, or `cpu`,
on gloo), says it is ready, and takes the steps it is sent (`Asked`), answering each with the step's metrics (rank 0)
or nothing (the others), or with the traceback of what failed. After a failure it ends: the processes are out of step,
and the trainer ends them and starts others. It also ends when the trainer's connection closes, and when its parent
(torchrun's agent) ends.

A step goes on from what the processes hold when its parent is the checkpoint they made last (its state's `HELD` says
the name they gave it); else they load the policy from the step's parent (an adapter over the model, or full weights,
and the optimizer's state: the full state where the parent has it, `rollout_lora.sharded.SHARDS`, else `optimizer.pt`,
else afresh) or from the model for the first step. Rank 0 writes what the step leaves: the weights, `minibatches.jsonl`,
`HELD`, and the optimizer's state every `state_every` steps since the processes loaded (an adapter's as `optimizer.pt`,
a full-weight trainer's full state under `SHARDS`, which every process writes its shards of).
"""

import contextlib
import gc
import json
import os
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rollout.processes import end_with_parent
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import HELD, STATE, WEIGHTS, Files, Item

if TYPE_CHECKING:
    import torch

__all__ = ["OPTIMIZER", "Asked", "main"]

OPTIMIZER = "optimizer.pt"
"""In a step's state: an adapter's optimizer's state after it (as a step on one GPU writes it)."""
MEMORY_MARGIN = 256 * 2**20
"""GPU memory left free of what was free when a process started."""
SEED = 0
"""What the processes seed torch's generator with before they load a policy."""


@dataclass(frozen=True)
class Asked:
    """A step, as the trainer sends it to every process."""

    checkpoint: str
    """The model the trainer trains (an adapter's base, a full-weight trainer's first weights)."""
    settings: LoraSettings
    weights: str
    """`lora` or `full`."""
    segments: list[Item]
    seed: int
    parent: Files | None
    into: Path
    held: str
    """What the processes call what they hold after this step (written to its state's `HELD`)."""


@dataclass
class _Held:
    """What the processes hold between steps."""

    policy: Any
    optimizer: "torch.optim.Optimizer"
    fresh: bool
    name: str = ""
    """What they called it (`Asked.held`), after a step; empty before any."""
    since: int = 0
    """Steps since they loaded or wrote the full state."""
    whole_base: bool = False
    loaded: bool = True
    """Whether the next step's policy was loaded from files (rather than gone on from memory)."""


def _holds(held: _Held | None, parent: Files | None) -> bool:
    """Whether the processes hold `parent`: its state says the name they gave what they hold."""
    if held is None or not held.name or parent is None or parent.state is None:
        return False
    with contextlib.suppress(OSError):
        return (parent.state / HELD).read_text().strip() == held.name
    return False


def _whole_base(asked: Asked, ranks: Any, device: "torch.device") -> bool:
    """Whether each GPU holds the whole frozen model (`LoraSettings.whole_base`): as the settings say, else where the
    model's files take at most half of the smallest GPU's memory. Every process decides alike, before any shards."""
    import torch

    from rollout_lora.models import local

    if asked.settings.whole_base is not None:
        return asked.settings.whole_base
    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    smallest = min(ranks.gathered(total))
    files = sum(each.stat().st_size for each in local(asked.checkpoint).glob("*.safetensors"))
    return device.type == "cuda" and files <= smallest / 2


def _loaded(asked: Asked, ranks: Any, device: "torch.device", mesh: Any) -> _Held:
    """The policy and its optimizer from the step's parent (or the model), sharded."""
    import torch

    from rollout_lora.full import FullPolicy
    from rollout_lora.layers import load_adapter
    from rollout_lora.policy import Policy
    from rollout_lora.sharded import SHARDS, in_turn, read_optimizer, read_state, shard_adapter, shard_full

    settings, parent = asked.settings, asked.parent
    whole_base = asked.weights == "lora" and _whole_base(asked, ranks, device)

    def make() -> Any:
        if asked.weights == "full":
            reference = asked.checkpoint if settings.frozen_reference and settings.loss.needs_reference else None
            loaded = str(parent.weights) if parent is not None else asked.checkpoint
            policy = FullPolicy.load(loaded, reference=reference, device="cpu")
            shard_full(policy, mesh, device)
            return policy
        policy = Policy.load(asked.checkpoint, rank=settings.rank, alpha=settings.alpha, device="cpu")
        if parent is not None:
            load_adapter(policy.model, parent.weights)
        shard_adapter(policy, mesh, device, whole_base=whole_base)
        return policy

    torch.manual_seed(SEED)  # (a new adapter is drawn alike in every process: each keeps its share of the same one)
    policy = in_turn(make, ranks)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings.learning_rate, weight_decay=0.0)
    fresh = True
    state = parent.state if parent is not None else None
    if state is not None and asked.weights == "full" and (state / SHARDS).is_dir():
        read_state(policy.model, optimizer, state / SHARDS, ranks)
        fresh = False
    elif state is not None and (state / OPTIMIZER).exists():
        read_optimizer(optimizer, state / OPTIMIZER, policy.parameters())
        fresh = False
    return _Held(policy, optimizer, fresh, whole_base=whole_base)


def _step(
    asked: Asked, held: _Held | None, ranks: Any, device: "torch.device", mesh: Any
) -> tuple[dict[str, float], _Held]:
    import torch

    from rollout_lora.sharded import SHARDS, write_adapter, write_optimizer, write_serving_copy, write_state
    from rollout_objectives.step import MINIBATCHES, PolicyStep

    if held is None or not _holds(held, asked.parent):
        del held  # (what the processes held is not the step's parent: freed before the parent is loaded)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        held = _loaded(asked, ranks, device, mesh)
    settings = asked.settings
    for group in held.optimizer.param_groups:  # (the saved state carries the rate it was saved with)
        group["lr"] = settings.learning_rate
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stepping = PolicyStep(held.policy, settings, fresh=held.fresh, ranks=ranks, optimizer_given=held.optimizer)
    metrics: dict[str, Any] = stepping.step(asked.segments, seed=asked.seed)
    loaded, held.loaded, held.fresh = held.loaded, False, False
    into = asked.into
    if asked.weights == "full":
        write_serving_copy(held.policy, into / WEIGHTS, ranks, source=asked.checkpoint)
    else:
        write_adapter(held.policy, into / WEIGHTS, ranks)
    held.since += 1
    writes_state = held.since >= settings.state_every_for(asked.weights)
    if writes_state and asked.weights == "full":
        write_state(held.policy.model, held.optimizer, into / STATE / SHARDS, ranks)
    elif writes_state:
        write_optimizer(held.optimizer, into / STATE / OPTIMIZER, ranks)
    if writes_state:
        held.since = 0
    if ranks.rank == 0:
        (into / STATE).mkdir(parents=True, exist_ok=True)
        (into / STATE / MINIBATCHES).write_text("".join(json.dumps(each) + "\n" for each in stepping.minibatches))
        (into / STATE / HELD).write_text(asked.held)
    held.name = asked.held
    peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    metrics |= {
        "peak_gpu_gib": ranks.most(peak),
        "gpus": float(ranks.size),
        "whole_base": float(held.whole_base),
        "loaded_from_files": float(loaded),
        "full_state": float(writes_state),
    }
    return metrics, held


def _bounded(device: "torch.device") -> float:
    """Bound the process's GPU memory to what was free when it started (as a step on one GPU is bounded:
    `rollout_lora.worker`); what was free, in GiB."""
    import torch

    if device.type != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, (free - MEMORY_MARGIN) / total)), device)
    return free / 2**30


def main() -> None:
    end_with_parent()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    host, _, port = os.environ["ROLLOUT_WORKERS"].rpartition(":")
    connection: Connection = Client((host, int(port)), authkey=bytes.fromhex(os.environ["ROLLOUT_WORKERS_KEY"]))
    import torch.distributed as distributed

    from rollout_lora.sharded import joined

    try:
        ranks, device, mesh = joined(os.environ.get("ROLLOUT_WORKERS_DEVICE", "cuda"))
        free = _bounded(device)
    except Exception:
        connection.send(("error", traceback.format_exc()))
        raise
    connection.send(("ready", ranks.rank))
    held: _Held | None = None
    while True:
        try:
            message = connection.recv()
        except EOFError:  # (the trainer is gone)
            break
        if message[0] != "step":
            break
        try:
            metrics, held = _step(message[1], held, ranks, device, mesh)
            metrics["free_gpu_gib"] = min(ranks.gathered(free))  # (when the processes started)
            connection.send(("done", metrics if ranks.rank == 0 else {}))
        except Exception:
            connection.send(("error", traceback.format_exc()))
            break
    with contextlib.suppress(Exception):
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
