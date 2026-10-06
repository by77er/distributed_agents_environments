# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of the distributed API untyped.)
"""The processes a trainer steps in, one per GPU, and what they load and write: `rollout_lora.resident.Workers`
starts them under torchrun, whatever their number and whether they are kept between steps:

    python -m torch.distributed.run --standalone --nproc-per-node N -m rollout_lora.workers

Each connects to the trainer that started them (`ROLLOUT_WORKERS`, its address, `HOST:PORT`; `ROLLOUT_WORKERS_KEY`,
the key it takes connections with, in hex), joins the others (`ROLLOUT_WORKERS_DEVICE`: `cuda`, a GPU each, or `cpu`,
on gloo), bounds its GPU memory to what was free when it started, less `MEMORY_MARGIN` (where a driver lets a process
spill into system memory, as Windows does, a step that needs more would crawl instead of failing), says it is ready,
and takes the steps it is sent (`Asked`), answering each with the step's metrics (rank 0) or nothing (the others), or
with the traceback of what failed. After a failure it ends: the processes are out of step, and the trainer ends them
and starts others. It also ends when the trainer's connection closes, and when its parent (torchrun's agent) ends.

**Loading.** A step goes on from what the processes hold when its parent is the checkpoint they made last (its state's
`HELD` says the name they gave it). Else they drop what they hold, and free its memory, before they load the policy:
an adapter over the model (the parent's adapter, else a new one), or full weights (the parent's, else the model's),
torch's generator seeded alike first (`SEED`: a new adapter is the same in every process and every trainer). On one
process an adapter's policy is loaded onto the GPU as it is; on several it is sharded over them
(`rollout_lora.sharded`). Full weights are sharded on any number, one included, so that one GPU computes as several do.
The optimizer goes on from the parent's full state (`full_state`): `state/shards` (`SHARDS`: the
float32 weights and the optimizer's state, read back by however many processes there are) or `state/optimizer.pt`
(`OPTIMIZER`: an adapter's optimizer). A parent without a state starts it afresh. A parent whose state left its full
state out is refused before any process is asked (`refusal`).

**Writing.** Rank 0 writes the weights (an adapter in PEFT's layout in float32; full weights' bfloat16 serving copy),
`minibatches.jsonl` and, where the processes are kept, `HELD`. The full state is written every `Asked.state_every` steps
since the processes loaded: an adapter's optimizer as `optimizer.pt` (gathered, in the layout of one process's state),
full weights' under `shards` (every process writes its own).
"""

import contextlib
import gc
import json
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rollout.processes import end_with_parent
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import HELD, STATE, WEIGHTS, Files, Item

if TYPE_CHECKING:
    import torch

__all__ = ["MEMORY_MARGIN", "OPTIMIZER", "SEED", "Asked", "full_state", "main", "refusal"]

OPTIMIZER = "optimizer.pt"
"""In a step's state: an adapter's optimizer's state after it (`torch.save` of its `state_dict`, by parameter index)."""
MEMORY_MARGIN = 256 * 2**20
"""GPU memory left free of what was free when a process started (other programs' use moves a little)."""
SEED = 0
"""What the processes seed torch's generator with before they load a policy."""
SHARDS = "shards"
"""Under a step's state: full weights' full state, as PyTorch's distributed checkpoint writes it
(`rollout_lora.sharded.write_state`)."""


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
    held: str | None
    """What the processes call what they hold after this step (written to its state's `HELD`); none where they end
    after it."""
    state_every: int = 1
    """Every how many steps since they loaded the processes write the full state."""


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


def full_state(state: Path) -> bool:
    """Whether a step's state holds the trainer's full state: what a step from it goes on from as the step that wrote
    it would."""
    return (state / SHARDS).is_dir() or (state / OPTIMIZER).is_file()


def _named(state: Path | None) -> str | None:
    """The name a state's `HELD` says, if it has one."""
    if state is None:
        return None
    with contextlib.suppress(OSError):
        return (state / HELD).read_text().strip()
    return None


def refusal(parent: Files | None, holding: str | None, state_every: int) -> str | None:
    """Why a step from `parent` cannot be taken by processes that hold `holding` (none: nothing), if it cannot: its
    state is a kept trainer's that left the full state out (`LoraSettings.state_every`), and they do not hold it. Going
    on would start the optimizer afresh, and full weights from their bfloat16 serving copy."""
    if parent is None or parent.state is None or full_state(parent.state):
        return None
    named = _named(parent.state)
    if named is None or (holding is not None and named == holding):
        return None
    return (
        f"the parent's state has no full state (the trainer writes it every {state_every} steps), and no process "
        "holds the parent any longer: going on would lose its optimizer's state. Start a run from the newest "
        "checkpoint with its full state, or have the trainer write it every step (state_every = 1)"
    )


def _holds(held: _Held, parent: Files | None) -> bool:
    """Whether the processes hold `parent`: its state says the name they gave what they hold."""
    return bool(held.name) and parent is not None and _named(parent.state) == held.name


def _whole_base(asked: Asked, ranks: Any, device: "torch.device") -> bool:
    """Whether each GPU holds the whole frozen model (`LoraSettings.whole_base`): as the settings say; else on one
    GPU; else as the memory estimate decides (`rollout_train.memory.holds_whole_base`, from the smallest GPU's memory
    by its name, else as it reports it). Every process decides alike, before any shards."""
    import torch

    from rollout_lora.models import local
    from rollout_train.memory import gpu_memory_gib, holds_whole_base

    if ranks.size == 1:
        return True
    if asked.settings.whole_base is not None:
        return asked.settings.whole_base
    if device.type != "cuda":
        return False
    found = torch.cuda.get_device_properties(device)
    gib = gpu_memory_gib((found.name,)) or found.total_memory / 2**30
    files = sum(each.stat().st_size for each in local(asked.checkpoint).glob("*.safetensors"))
    return holds_whole_base(files, min(ranks.gathered(gib)))


def _loaded(asked: Asked, ranks: Any, device: "torch.device", mesh: Any) -> _Held:
    """The policy and its optimizer from the step's parent (or the model): on the device, or sharded."""
    import torch

    from rollout_lora.full import FullPolicy
    from rollout_lora.layers import load_adapter
    from rollout_lora.policy import Policy
    from rollout_lora.sharded import in_turn, read_optimizer, read_state, shard_adapter, shard_full

    settings, parent = asked.settings, asked.parent
    whole_base = asked.weights == "lora" and _whole_base(asked, ranks, device)
    shared = ranks.size > 1
    staged = "cpu" if shared else str(device)  # (several: loaded onto the CPU in turn, then sharded)

    def make() -> Any:
        if asked.weights == "full":
            reference = asked.checkpoint if settings.frozen_reference and settings.loss.needs_reference else None
            loaded = str(parent.weights) if parent is not None else asked.checkpoint
            policy = FullPolicy.load(loaded, reference=reference, device=staged)
            shard_full(policy, mesh, device)
            return policy
        policy = Policy.load(asked.checkpoint, rank=settings.rank, alpha=settings.alpha, device=staged)
        if parent is not None:
            load_adapter(policy.model, parent.weights)
        if shared:
            shard_adapter(policy, mesh, device, whole_base=whole_base)
        return policy

    torch.manual_seed(SEED)  # (a new adapter is drawn alike in every process: each keeps its share of the same one)
    policy = in_turn(make, ranks)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings.learning_rate, weight_decay=0.0)
    fresh = True
    state = parent.state if parent is not None else None
    if state is not None and (state / SHARDS).is_dir():
        read_state(policy.model, optimizer, state / SHARDS, ranks)
        fresh = False
    elif state is not None and (state / OPTIMIZER).is_file():
        read_optimizer(optimizer, state / OPTIMIZER, policy.parameters())
        fresh = False
    return _Held(policy, optimizer, fresh, whole_base=whole_base)


def _stepped(asked: Asked, held: _Held, ranks: Any, device: "torch.device") -> dict[str, float]:
    """The step taken on what the processes hold, its files written; its metrics."""
    import torch

    from rollout_lora.sharded import write_adapter, write_optimizer, write_serving_copy, write_state
    from rollout_objectives.step import MINIBATCHES, PolicyStep

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
    writes_state = held.since >= asked.state_every
    if writes_state and asked.weights == "full":
        write_state(held.policy.model, held.optimizer, into / STATE / SHARDS, ranks)
    elif writes_state:
        write_optimizer(held.optimizer, into / STATE / OPTIMIZER, ranks)
    if writes_state:
        held.since = 0
    if ranks.rank == 0:
        (into / STATE).mkdir(parents=True, exist_ok=True)
        (into / STATE / MINIBATCHES).write_text("".join(json.dumps(each) + "\n" for each in stepping.minibatches))
        if asked.held is not None:
            (into / STATE / HELD).write_text(asked.held)
    held.name = asked.held or ""
    peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    return metrics | {
        "peak_gpu_gib": ranks.most(peak),
        "gpus": float(ranks.size),
        "whole_base": float(held.whole_base),
        "loaded_from_files": float(loaded),
        "full_state": float(writes_state),
    }


def _freed(device: "torch.device") -> None:
    """Free the memory of what nothing refers to any longer (the GPU's too)."""
    import torch

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _bounded(device: "torch.device") -> float:
    """Bound the process's GPU memory to what was free when it started, less `MEMORY_MARGIN`; what was free, in GiB."""
    import torch

    if device.type != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, (free - MEMORY_MARGIN) / total)), device)
    return free / 2**30


def _serve(
    receive: Callable[[], Any], send: Callable[[Any], None], ranks: Any, device: "torch.device", mesh: Any, free: float
) -> None:
    """Take the steps the trainer sends until it stops, a step fails or its connection closes."""
    held: _Held | None = None
    while True:
        try:
            message = receive()
        except EOFError:  # (the trainer is gone)
            return
        if message[0] != "step":
            return
        asked: Asked = message[1]
        try:
            if held is not None and not _holds(held, asked.parent):
                held = None  # (what they hold is not the parent: dropped and freed before the parent is loaded)
                _freed(device)
            if held is None:
                held = _loaded(asked, ranks, device, mesh)
            metrics = _stepped(asked, held, ranks, device)
            metrics["free_gpu_gib"] = min(ranks.gathered(free))  # (when the processes started)
            send(("done", metrics if ranks.rank == 0 else {}))
        except Exception:
            send(("error", traceback.format_exc()))
            return


def main() -> None:
    end_with_parent()
    # Reserve close to what is used: fragmentation would otherwise cost about 0.7 GiB at the peak.
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
    _serve(connection.recv, connection.send, ranks, device, mesh, free)
    with contextlib.suppress(Exception):
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
