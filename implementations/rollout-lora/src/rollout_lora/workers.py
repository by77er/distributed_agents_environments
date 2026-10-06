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
with the traceback of what failed. While a step runs, rank 0 says how far it has got after each pack and each minibatch
(`("progress", Progress)`, `rollout_objectives.step.PolicyStep.progress`). After a failure it ends: the processes are
out of step, and the trainer ends them and starts others. It also ends when the trainer's connection closes, and when
its parent (torchrun's agent) ends.

**Loading.** A step goes on from what the processes hold when its parent is the checkpoint they made last (its state's
`HELD` says the name they gave it). Else they drop what they hold, and free its memory, before they load the policy:
an adapter over the model (the parent's adapter, else a new one), or full weights (the parent's, else the model's),
torch's generator seeded alike first (`SEED`: a new adapter is the same in every process and every trainer). On one
process an adapter's policy is loaded onto the GPU as it is; on several it is sharded over them
(`rollout_lora.sharded`). Full weights are sharded on any number, one included, so that one GPU computes as several do.
The optimizer goes on from the parent's full state (`full_state`): `state/shards` (`SHARDS`: the
float32 weights and the optimizer's state, read back by however many processes there are) or `state/optimizer.pt`
(`OPTIMIZER`: an adapter's optimizer), and an adapter from the state's float32 copy (`MASTER`) where it has one, else
from the weights. A parent without a state starts it afresh. A parent whose state left its full state out is refused
before any process is asked (`refusal`).

**Writing.** Rank 0 writes the weights engines load, in bfloat16 (an adapter in PEFT's layout, half the bytes of its
float32; full weights' serving copy), `minibatches.jsonl` and, where the processes are kept, `HELD`. The full state is
written every `Asked.state_every` steps since the processes loaded, the same files from one process as from several: an
adapter's optimizer as `optimizer.pt` (gathered, in the layout of one process's state) and the adapter in float32 as
`master.safetensors`; full weights' under `shards` (every process writes its own).

**Keeping the full state after the step.** Told where to keep it (`Asked.keep`, a blob store's location), the processes
copy the full state off the GPU at the end of the step, in the same order in every process (an adapter's gathered into
rank 0's host memory, pinned; full weights' shares written by each process into its own memory with the distributed
checkpoint, every collective done before the step answers), answer, and keep it in the blob store from host memory in a
thread of each process's own (`_Keeper`), which reads and writes nothing on the GPU and takes part in no collective, so
the next step trains on while it uploads. Each process says what it kept (`("kept", NAME, …)`, the step's
directory's name; the trainer merges them, `rollout_lora.resident.Workers.kept_state`), or why it could not, after
trying `KEEP_TRIES` times. A snapshot is taken in host memory only where every process's share together takes at most
`HOST_SHARE` of the memory the machine has available (within its container's limit); else it is written to the pod's
disk (`SPILLED`) and kept from there. One snapshot is kept at a time: a step's snapshot waits until the one before is
kept, and a process that ends finishes keeping it first. Without `Asked.keep`, the full state is written into the
step's `state` before the step answers.
"""

import asyncio
import contextlib
import gc
import io
import json
import math
import os
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rollout.processes import end_with_parent
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import HELD, STATE, WEIGHTS, Files, Item, Progress

if TYPE_CHECKING:
    import torch

__all__ = ["HOST_SHARE", "KEEP_TRIES", "MASTER", "MEMORY_MARGIN", "OPTIMIZER", "SEED", "SPILLED", "Asked",
           "full_state", "main", "refusal"]  # fmt: skip

OPTIMIZER = "optimizer.pt"
"""In a step's state: an adapter's optimizer's state after it (`torch.save` of its `state_dict`, by parameter index)."""
MASTER = "master.safetensors"
"""In a step's state, beside `OPTIMIZER`: the adapter in float32, by PEFT's names, which a step from it loads (the
weights' copy, which engines load, is bfloat16)."""
SPILLED = "spilled"
"""Under a step's directory: the full state written to disk to be kept from there, where host memory cannot hold it."""
HOST_SHARE = 0.5
"""The most of the host memory available that the processes' snapshots of a full state take together."""
KEEP_TRIES = 5
"""Times a process tries to keep each file of a full state, waiting twice as long after each failure from a second."""
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
    keep: Mapping[str, Any] | None = None
    """Where they keep the full state after answering (a blob store's location, `rollout_train.stores.opened`); none:
    written into `into/state` before they answer."""


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
    from rollout_lora.models import fetched
    from rollout_lora.policy import Policy
    from rollout_lora.sharded import in_turn, read_optimizer, read_state, shard_adapter, shard_full

    settings, parent = asked.settings, asked.parent
    fetched(asked.checkpoint)  # (before anything reads its files: a pod's cache starts empty)
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
        if parent is not None:  # (from the float32 adapter where the state has it, never the bfloat16 serving copy)
            master = parent.state / MASTER if parent.state is not None else None
            load_adapter(policy.model, master if master is not None and master.is_file() else parent.weights)
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


def _stepped(
    asked: Asked,
    held: _Held,
    ranks: Any,
    device: "torch.device",
    keeper: "_Keeper",
    told: Callable[[Progress], None] | None = None,
) -> dict[str, float]:
    """The step taken on what the processes hold, its files written (its full state kept after it, by `keeper`, where
    the step says where), `told` how far it has got as it goes; its metrics."""
    import torch

    from rollout_lora.sharded import write_adapter, write_serving_copy
    from rollout_objectives.step import MINIBATCHES, PolicyStep

    settings = asked.settings
    for group in held.optimizer.param_groups:  # (the saved state carries the rate it was saved with)
        group["lr"] = settings.learning_rate
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    stepping = PolicyStep(held.policy, settings, fresh=held.fresh, ranks=ranks, optimizer_given=held.optimizer,
                          progress=told)  # fmt: skip
    metrics: dict[str, Any] = stepping.step(asked.segments, seed=asked.seed)
    trained, began = time.monotonic() - began, time.monotonic()
    loaded, held.loaded, held.fresh = held.loaded, False, False
    into = asked.into
    master = None
    if asked.weights == "full":
        write_serving_copy(held.policy, into / WEIGHTS, ranks, source=asked.checkpoint)
    else:
        master = write_adapter(held.policy, into / WEIGHTS, ranks)
    saved, began = time.monotonic() - began, time.monotonic()
    held.since += 1
    writes_state = held.since >= asked.state_every
    waited, in_memory = 0.0, 0.0
    if writes_state:
        waited = keeper.wait()  # (the state before kept first: one is kept at a time, and its memory is freed)
        began = time.monotonic()
        snapshot = _snapshot(asked, held, master, ranks)
        in_memory = float(snapshot.in_memory)
        if asked.keep is not None:
            keeper.keep(into.name, snapshot, asked.keep)
        held.since = 0
    copied = time.monotonic() - began if writes_state else 0.0
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
        "train_seconds": round(ranks.most(trained), 3),
        "save_adapter_seconds": round(ranks.most(saved), 3),
        "state_wait_seconds": round(ranks.most(waited), 3),
        "snapshot_seconds": round(ranks.most(copied), 3),
        "snapshot_in_memory": in_memory,
    }


@dataclass
class _Snapshot:
    """A step's full state, off the GPU, as one process keeps it: each of its files, by its path within the state, as
    what makes its bytes from host memory, or as a file on disk."""

    files: dict[str, Callable[[], bytes] | Path] = field(default_factory=dict[str, Callable[[], bytes] | Path])
    in_memory: bool = False
    """Whether it is in host memory (else on disk, or written into the step's state)."""
    memory: str | None = None
    """Where in the process's own memory it was written, to be freed once kept."""


def _snapshot(asked: Asked, held: _Held, master: "dict[str, torch.Tensor] | None", ranks: Any) -> _Snapshot:
    """The step's full state copied off the GPU, every process alike (each gathers and writes in the same order, and
    decides alike where): into host memory where it fits and the step keeps it after (`_fits`), else onto disk (the
    step's state, or `SPILLED` to be kept from there)."""
    import torch

    from rollout_lora.sharded import MEMORY, gathered_optimizer, state_bytes, write_state

    into = asked.into
    if asked.weights == "full":
        shares = state_bytes(held.policy.model)
        in_memory = asked.keep is not None and _fits(shares, ranks)
        place = into / (STATE if asked.keep is None else SPILLED) / SHARDS
        where = f"{MEMORY}rollout-state/{os.getpid()}/{into.name}/{SHARDS}" if in_memory else place
        write_state(held.policy.model, held.optimizer, where, ranks)
        if asked.keep is None:
            return _Snapshot()
        if in_memory:
            return _Snapshot(_in_memory(str(where), SHARDS), in_memory=True, memory=str(where))
        if ranks.rank != 0:  # (every process wrote its files into one directory: rank 0 keeps them all)
            return _Snapshot()
        return _Snapshot({f"{SHARDS}/{each.relative_to(place)}": each for each in sorted(place.rglob("*"))
                          if each.is_file()})  # fmt: skip
    trained = sum(4 * each.numel() for each in held.policy.parameters())  # (float32, whatever its shards)
    optimizer = gathered_optimizer(held.optimizer, ranks)
    in_memory = asked.keep is not None and _fits(3 * trained if ranks.rank == 0 else 0, ranks)
    if optimizer is None or master is None:  # (rank 0 holds the adapter and its optimizer's state)
        return _Snapshot(in_memory=in_memory)
    if in_memory:
        return _Snapshot({OPTIMIZER: lambda: _saved(optimizer), MASTER: lambda: _tensors(master)}, in_memory=True)
    place = into / (STATE if asked.keep is None else SPILLED)
    place.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer, place / OPTIMIZER)
    (place / MASTER).write_bytes(_tensors(master))
    return _Snapshot({OPTIMIZER: place / OPTIMIZER, MASTER: place / MASTER})


def _saved(state: Any) -> bytes:
    """What `torch.save` writes of `state`."""
    import torch

    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _tensors(tensors: "dict[str, torch.Tensor]") -> bytes:
    """Tensors by name as a safetensors file's bytes."""
    from safetensors.torch import save

    return save(tensors)


def _in_memory(where: str, under: str) -> dict[str, Callable[[], bytes] | Path]:
    """The files this process wrote into its memory at `where`, by their paths `under` a state."""
    import fsspec  # pyright: ignore[reportMissingTypeStubs]

    from rollout_lora.sharded import MEMORY

    memory = fsspec.filesystem("memory")
    root = "/" + where.removeprefix(MEMORY)  # (as the in-memory filesystem names its paths)
    found: dict[str, Callable[[], bytes] | Path] = {}
    for path in sorted(memory.find(root)):

        def read(path: str = path) -> bytes:
            return memory.cat_file(path)

        found[f"{under}/{path.removeprefix(root).lstrip('/')}"] = read
    return found


def host_memory() -> float:
    """Bytes of host memory this process may still take: what the machine has available, within its container's limit
    (its cgroup's, version 2 or 1, less what it holds that cannot be reclaimed)."""
    available = math.inf
    with contextlib.suppress(OSError, ValueError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    cgroup = Path("/sys/fs/cgroup")
    for limit_at, used_at, stat_at, inactive in (
        (cgroup / "memory.max", cgroup / "memory.current", cgroup / "memory.stat", "inactive_file"),
        (cgroup / "memory" / "memory.limit_in_bytes", cgroup / "memory" / "memory.usage_in_bytes",
         cgroup / "memory" / "memory.stat", "total_inactive_file"),
    ):  # fmt: skip
        with contextlib.suppress(OSError, ValueError):
            limit = limit_at.read_text().strip()
            if limit == "max" or int(limit) >= 2**60:  # (no limit)
                continue
            stat = dict(line.split() for line in stat_at.read_text().splitlines())
            used = int(used_at.read_text()) - int(stat.get(inactive, 0))
            available = min(available, int(limit) - used)
    return available


def _fits(share: int, ranks: Any) -> bool:
    """Whether every process's share of a snapshot (this one's: `share` bytes) together fits in host memory: at most
    `HOST_SHARE` of what the machine has available. Every process calls it at once, and all decide alike."""
    shares = ranks.gathered((share, host_memory()))
    return sum(each for each, _ in shares) <= HOST_SHARE * min(available for _, available in shares)


class _Keeper:
    """Keeps a process's snapshots of steps' full state in a blob store, one at a time, each in a thread of its own,
    and says what it kept, or why it could not (`send`, as `("kept", NAME, {"files": …, "seconds": …})` or
    `("kept", NAME, {"error": …})`)."""

    def __init__(self, send: Callable[[Any], None]) -> None:
        self.send = send
        self._thread: threading.Thread | None = None

    def wait(self) -> float:
        """Wait until the snapshot being kept is kept (or could not be); how long it waited, in seconds."""
        began = time.monotonic()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return time.monotonic() - began

    def keep(self, name: str, snapshot: _Snapshot, location: Mapping[str, Any]) -> None:
        """Keep `snapshot` (of the step that wrote into a directory called `name`) in the store at `location`, in the
        background, once the one before is kept."""
        self.wait()
        self._thread = threading.Thread(target=self._kept, args=(name, snapshot, location), name="keeping", daemon=True)
        self._thread.start()

    def _kept(self, name: str, snapshot: _Snapshot, location: Mapping[str, Any]) -> None:
        began = time.monotonic()
        try:
            files = asyncio.run(_put(snapshot, location)) if snapshot.files else {}
            said: dict[str, Any] = {"files": files, "seconds": round(time.monotonic() - began, 3)}
        except Exception as error:
            said = {"error": f"{type(error).__name__}: {error}"[-2000:]}
        finally:
            if snapshot.memory is not None:  # (its memory freed, kept or not)
                with contextlib.suppress(Exception):
                    import fsspec  # pyright: ignore[reportMissingTypeStubs]

                    fsspec.filesystem("memory").rm(snapshot.memory, recursive=True)
        with contextlib.suppress(OSError):  # (the trainer is gone: there is no one to tell)
            self.send(("kept", name, said))


async def _put(snapshot: _Snapshot, location: Mapping[str, Any]) -> dict[str, Any]:
    """Each of a snapshot's files kept in the store at `location`, tried `KEEP_TRIES` times; their references, by path,
    as JSON."""
    from rollout_train.stores import opened

    blobs = opened(location)
    kept: dict[str, Any] = {}
    for path, source in snapshot.files.items():
        data = await asyncio.to_thread(source.read_bytes) if isinstance(source, Path) else source()
        for attempt in range(1, KEEP_TRIES + 1):
            try:
                kept[path] = (await blobs.put(data, "application/octet-stream")).model_dump(mode="json")
                break
            except Exception:
                if attempt == KEEP_TRIES:
                    raise
                await asyncio.sleep(2.0 ** (attempt - 1))
        del data
    return kept


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
    """Take the steps the trainer sends until it stops, a step fails or its connection closes; a full state being kept
    is kept before it returns."""
    sending = threading.Lock()

    def said(message: Any) -> None:  # (the keeper's thread tells the trainer too)
        with sending:
            send(message)

    def told(progress: Progress) -> None:
        with contextlib.suppress(OSError):  # (the trainer is gone: the step's answer fails it)
            said(("progress", progress))

    keeper = _Keeper(said)
    held: _Held | None = None
    try:
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
                metrics = _stepped(asked, held, ranks, device, keeper, told if ranks.rank == 0 else None)
                metrics["free_gpu_gib"] = min(ranks.gathered(free))  # (when the processes started)
                said(("done", metrics if ranks.rank == 0 else {}))
            except Exception:
                said(("error", traceback.format_exc()))
                return
    finally:
        keeper.wait()


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
