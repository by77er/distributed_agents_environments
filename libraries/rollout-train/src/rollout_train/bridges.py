"""Bridges: how a checkpoint's files in a trainer's format become files an inference provider loads.

A trainer writes its checkpoints in one format (`peft`, `full`, `tinker`:
`rollout_train.providers.TrainerCapabilities`); a provider loads some formats (`Capabilities.loads`). A bridge
turns one format into another (`Bridge`), and the registry (`BRIDGES`) holds every bridge by its pair of formats.
`path` finds the cheapest chain of bridges from a checkpoint's format to any format a provider loads; most pairs are
one bridge. A pair with no path is refused, and the pairs refused on purpose say why (`REFUSED`): Tinker samples only
checkpoints Tinker trained.

| From | To | Bridge | Work |
|---|---|---|---|
| `tinker` | `tinker` | `none` | the serving record names the checkpoint's own files |
| `tinker` | `peft` | `peft-from-tinker` | the sampler checkpoint's archive, its tensor names remapped to PEFT's |
| `peft` | `peft` | `verbatim` | linked as they are (also onto a model quantized from the adapter's base) |
| `full` | `full` | `full-reload` | linked as they are, loaded under the checkpoint's name replica by replica |
| `peft` | `full` | `merge-quantize` | merged into the base: full weights, quantized as they load; only when asked |

A bridge may change an adapter's rank as the provider sees it (`rank_factor`): Tinker's adapters for Qwen3.5's
linear-attention layers carry an A of their own for each of q, k and v, joined into one projection of three times the
rank.

Each bridge's work is a function, named as `module:name` (`Bridge.task`): `task(weights, into, context)` writes the
target's files into `into` from the source's in `weights`, and returns what it says about them. `Context` tells it the
checkpoint, the model its weights are over, the model the provider serves it on, and the bridge's own settings.
`bridged` runs a chain of bridges in this process; `on_ray` runs each as a Ray task of its own, asking for the CPUs
and memory the bridge declares, on whichever node has room. A bridge notes in the ledger when it begins
(`checkpoints/resharding`) and what it made (`checkpoints/resharded`: the bridge, what its task said, and a manifest
of the files, in the blob store), keyed `CHECKPOINT@BRIDGE`, so that one checkpoint is bridged once for each format it
is served in, under the fence of the run that made the checkpoint. A checkpoint bridged before is not bridged again.
"""

import asyncio
import fnmatch
import heapq
import os
import shutil
import tempfile
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.names import named
from rollout_train.checkpoints import Checkpoints, Manifest, kept
from rollout_train.ledger import Fence, Ledger
from rollout_train.trainer import WEIGHTS

__all__ = [
    "BRIDGED",
    "BRIDGES",
    "BRIDGING",
    "FORMATS",
    "REFUSED",
    "Bridge",
    "Context",
    "NoBridge",
    "bridge_of",
    "bridged",
    "by_name",
    "checkpoint_of",
    "format_of",
    "key",
    "made",
    "on_ray",
    "path",
    "rank_factor",
    "verbatim",
]

FORMATS = ("peft", "full", "tinker")
"""The checkpoint formats: an adapter in PEFT's layout, full weights (safetensors and a `config.json`), and pointers
to Tinker's sampler checkpoint and state."""
AUTO, MERGE_QUANTIZE = "auto", "merge-quantize"
"""What a run's `channels.NAME.bridge` may say: the cheapest path, or one through `merge-quantize`."""


@dataclass(frozen=True)
class Bridge:
    """One bridge: from a format to another, the task that does it, and what the task needs."""

    name: str
    source: str
    target: str
    task: str | None
    """`module:name` of the function that writes the target's files from the source's; none: the files are served as
    they are, with nothing written."""
    says: str
    cpus: float = 1
    memory_gib: float = 1
    network: bool = False
    """Whether the task fetches from outside the cluster (Tinker's archive)."""
    rank_factors: tuple[tuple[str, int], ...] = ()
    """By model pattern (`fnmatch`): how many times the trained rank the provider sees."""
    explicit: bool = False
    """Chosen only when the run asks for it (`channels.NAME.bridge`), never by the path search alone."""
    cost: int = 1
    """Its weight in the path search."""


@dataclass(frozen=True)
class NoBridge:
    """A pair of formats with no path between them, and why."""

    source: str
    target: str
    reason: str


VERBATIM = "rollout_train.bridges:verbatim"
BRIDGES: tuple[Bridge, ...] = (
    Bridge(
        "none", "tinker", "tinker", None, "served as it is: Tinker's sampler reads the checkpoint's pointer", cost=0
    ),
    Bridge(
        "peft-from-tinker",
        "tinker",
        "peft",
        "rollout_tinker.bridges:peft",
        "Tinker's adapter downloaded and written in PEFT's layout",
        cpus=2,
        network=True,
        rank_factors=(("Qwen/Qwen3.5-*", 3),),
    ),
    Bridge("verbatim", "peft", "peft", VERBATIM, "the adapter's files, linked as they are"),
    Bridge(
        "full-reload",
        "full",
        "full",
        VERBATIM,
        "the full weights' files, linked as they are and loaded under the checkpoint's name, replica by replica",
    ),
    Bridge(
        MERGE_QUANTIZE,
        "peft",
        "full",
        "rollout_lora.bridges:merge_quantize",
        "the adapter merged into its base, as full weights a provider quantizes as it loads them",
        cpus=8,
        memory_gib=48,
        explicit=True,
        cost=10,
    ),
)
"""Every bridge, by its pair of formats."""

_NO_UPLOAD = "Tinker samples only checkpoints Tinker trained: there is no upload"
REFUSED: tuple[NoBridge, ...] = (
    NoBridge("peft", "tinker", _NO_UPLOAD),
    NoBridge("full", "tinker", _NO_UPLOAD),
    NoBridge("full", "peft", "full weights are not an adapter: serve them on a provider that reloads full weights"),
)
"""Pairs refused on purpose, with the reason a run is told."""


def path(source: str, loads: Collection[str], *, wanted: str = AUTO) -> tuple[Bridge, ...] | NoBridge:
    """The cheapest chain of bridges from `source` to one of the formats in `loads`. `wanted` is the run's
    `channels.NAME.bridge`: `auto`, or `merge-quantize` for a path through it (a bridge marked `explicit` is used
    only then)."""
    if wanted not in (AUTO, MERGE_QUANTIZE):
        return NoBridge(source, ",".join(sorted(loads)), f"a bridge is {AUTO} or {MERGE_QUANTIZE}, not {wanted!r}")
    if not loads:
        return NoBridge(source, "", "the provider serves base models only: it loads no checkpoint")
    # Dijkstra over (format, whether the wanted bridge was crossed).
    must = wanted == MERGE_QUANTIZE
    waiting: list[tuple[int, int, str, bool, tuple[Bridge, ...]]] = [(0, 0, source, not must, ())]
    seen: set[tuple[str, bool, bool]] = set()  # (a format, whether the wanted bridge was crossed, whether any was)
    order = 0
    while waiting:
        cost, _, at, crossed, chain = heapq.heappop(waiting)
        if (at, crossed, bool(chain)) in seen:
            continue
        seen.add((at, crossed, bool(chain)))
        if crossed and at in loads and chain:  # (every pair, even a format to itself, is a bridge)
            return chain
        for bridge in BRIDGES:
            if bridge.source != at or (bridge.explicit and not must) or bridge in chain:
                continue
            order += 1
            through = crossed or bridge.name == wanted
            heapq.heappush(waiting, (cost + bridge.cost, order, bridge.target, through, (*chain, bridge)))
    for refused in REFUSED:
        if refused.source == source and refused.target in loads:
            return refused
    targets = " or ".join(sorted(loads))
    if must:
        return NoBridge(source, targets, f"no path from {source} to {targets} goes through {MERGE_QUANTIZE}")
    return NoBridge(source, targets, f"no bridge turns a {source} checkpoint into {targets}")


def rank_factor(chain: Collection[Bridge], model: str) -> int:
    """How many times the trained rank a provider sees, after the bridges of `chain`, for an adapter over `model`."""
    factor = 1
    for bridge in chain:
        factor *= next((each for pattern, each in bridge.rank_factors if fnmatch.fnmatch(model, pattern)), 1)
    return factor


def format_of(files: Collection[str]) -> frozenset[str]:
    """The formats a checkpoint's files are in, by their paths within it (a manifest's, with or without the leading
    `weights/`): `tinker.json` is `tinker`; `adapter_config.json` with `adapter_model.safetensors` is `peft`;
    `config.json` with safetensors weights is `full`. A checkpoint can be in two at once."""
    names = {each.removeprefix("weights/") for each in files}
    found: set[str] = set()
    if "tinker.json" in names:
        found.add("tinker")
    if {"adapter_config.json", "adapter_model.safetensors"} <= names:
        found.add("peft")
    if "config.json" in names and any(each.endswith(".safetensors") and "adapter" not in each for each in names):
        found.add("full")
    return frozenset(found)


def by_name(name: str) -> Bridge:
    """The bridge of the registry called `name`; `KeyError`, saying the names, for none."""
    for bridge in BRIDGES:
        if bridge.name == name:
            return bridge
    raise KeyError(f"no bridge is called {name!r} (these are: {', '.join(each.name for each in BRIDGES)})")


BRIDGING = "checkpoints/resharding"
"""The ledger's table of bridges begun, keyed `CHECKPOINT@BRIDGE` (`key`)."""
BRIDGED = "checkpoints/resharded"
"""The ledger's table of what each bridge made, keyed `CHECKPOINT@BRIDGE`."""
SCRATCH = "~/.cache/rollout/scratch"
"""Where a bridge on a Ray worker reads the source's files and writes the target's, unless it is told: on disk (a
machine's /tmp may be memory)."""

_MANIFEST = TypeAdapter(Manifest)


@dataclass(frozen=True)
class Context:
    """What a bridge's task is told beside the files it reads and writes."""

    checkpoint: str
    """The checkpoint it bridges, by id."""
    model: str | None = None
    """The model the checkpoint's weights are over (its record's `base`), by name or directory."""
    target: str | None = None
    """The model the provider serves it on (a copy of `model` quantized, say); none: `model`."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The bridge's own settings, from whoever runs it (the service Tinker's bridge asks, say)."""


def verbatim(weights: Path, into: Path, context: Context) -> dict[str, JsonValue]:
    """The provider loads the trainer's files as they are: each is linked (or copied) into `into`."""
    for each in sorted(weights.rglob("*")):
        if each.is_file():
            target = into / each.relative_to(weights)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(each, target)
            except OSError:
                shutil.copy2(each, target)
    return {"kind": "verbatim"}


def key(checkpoint: str, bridge: str) -> str:
    """A checkpoint's entry in the bridges' tables, for a bridge by name: `CHECKPOINT@BRIDGE`."""
    return f"{checkpoint}@{bridge}"


def checkpoint_of(entry: str) -> str:
    """The checkpoint an entry of the bridges' tables is of (`key`)."""
    return entry.partition("@")[0]


async def made(ledger: Ledger, checkpoint: str, bridge: str) -> Manifest | None:
    """What a bridge, by name, made of a checkpoint, if it did."""
    record: Any = (await ledger.read(BRIDGED)).get(key(checkpoint, bridge))
    return _MANIFEST.validate_python(record["files"]) if record else None


async def bridge_of(ledger: Ledger, checkpoint: str) -> str | None:
    """The bridge, by name, that last made files of a checkpoint, if one did."""
    found: tuple[float, str] | None = None
    for entry, record in (await ledger.read(BRIDGED)).items():
        if checkpoint_of(entry) == checkpoint and isinstance(record, dict):
            at, name = float(str(record.get("at") or 0)), str(record.get("bridge") or "")
            if name and (found is None or at >= found[0]):
                found = (at, name)
    return found[1] if found else None


async def bridged(
    checkpoints: Checkpoints,
    fence: Fence,
    checkpoint: str,
    chain: Sequence[Bridge],
    scratch: Path,
    *,
    target: str | None = None,
    settings: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> Manifest:
    """The files a provider loads for `checkpoint` (by id), made in this process by each bridge of `chain` (`path`'s)
    in turn, each from what the one before made, the first from the checkpoint's weights, unless it made them before.
    A bridge with no task (`none`) passes on the files it is given. `scratch` is where the files are read to and
    written, on this machine, for the while it takes; `target` is the model the provider serves, and `settings` each
    bridge's own, by its name."""
    record = await checkpoints.checkpoint(checkpoint)
    if record.weights is None:
        raise ValueError(f"{checkpoint} was released: its weights were deleted")
    files = record.weights
    for bridge in chain:
        context = Context(checkpoint, record.base, target, dict((settings or {}).get(bridge.name, {})))
        files = await _step(checkpoints, fence, bridge, files, context, scratch)
    return files


async def _step(
    checkpoints: Checkpoints, fence: Fence, bridge: Bridge, source: Manifest, context: Context, scratch: Path
) -> Manifest:
    """What one bridge makes of `source`, noted in the ledger: made now, or what it made before."""
    if bridge.task is None:
        return source
    if (done := await made(checkpoints.ledger, context.checkpoint, bridge.name)) is not None:
        return done
    entry = key(context.checkpoint, bridge.name)
    begun: JsonValue = {"at": round(time.time(), 1), "bridge": bridge.name, "host": os.uname().nodename}
    await checkpoints.ledger.append(BRIDGING, entry, begun, fence)
    await asyncio.to_thread(scratch.mkdir, parents=True, exist_ok=True)
    work = Path(await asyncio.to_thread(tempfile.mkdtemp, dir=scratch, prefix=f"{context.checkpoint}-"))
    try:
        weights = await checkpoints.files(source, work / WEIGHTS)
        into = work / "bridged"
        said = await asyncio.to_thread(named(bridge.task), weights, into, context)
        manifest = await kept(into, checkpoints.blobs)
    finally:
        await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)
    record: dict[str, Any] = {
        "at": round(time.time(), 1),
        "bridge": bridge.name,
        "task": bridge.task,
        "said": said,
        "files": _MANIFEST.dump_python(manifest, mode="json"),
    }
    if not await checkpoints.ledger.append(BRIDGED, entry, record, fence):
        return await made(checkpoints.ledger, context.checkpoint, bridge.name) or manifest
    return manifest


def _opened(ledger_at: Mapping[str, Any], blobs_at: Mapping[str, Any]) -> Checkpoints:
    """The checkpoints of the ledger and the blob store where they are."""
    from rollout_train.ledger import opened
    from rollout_train.stores import opened as store_at

    return Checkpoints(opened(ledger_at), store_at(blobs_at))


def _on_worker(
    ledger_at: Mapping[str, Any],
    blobs_at: Mapping[str, Any],
    fence: tuple[str, int],
    checkpoint: str,
    bridge: Bridge,
    source: Mapping[str, Any] | None,
    target: str | None,
    settings: Mapping[str, JsonValue],
    scratch: str,
) -> dict[str, Any]:
    """One bridge in a Ray worker: it opens the ledger and the blob store from where they are, and returns the
    manifest of what the bridge made as JSON (what crosses between processes). `source` is what the bridge before
    made; none: the checkpoint's weights."""

    async def run() -> Manifest:
        checkpoints = _opened(ledger_at, blobs_at)
        record = await checkpoints.checkpoint(checkpoint)
        files = _MANIFEST.validate_python(source) if source is not None else record.weights
        if files is None:
            raise ValueError(f"{checkpoint} was released: its weights were deleted")
        context = Context(checkpoint, record.base, target, settings)
        return await _step(checkpoints, Fence(*fence), bridge, files, context, work)

    work = Path(scratch).expanduser() / "bridges"

    return _MANIFEST.dump_python(asyncio.run(run()), mode="json")


async def on_ray(
    ledger_at: Mapping[str, Any],
    blobs_at: Mapping[str, Any],
    fence: Fence,
    checkpoint: str,
    chain: Sequence[Bridge],
    *,
    target: str | None = None,
    settings: Mapping[str, Mapping[str, JsonValue]] | None = None,
    scratch: str = SCRATCH,
) -> Manifest:
    """`bridged`, with each bridge of `chain` a Ray task of its own on the cluster this process is connected to
    (`ray.init`), asking for the CPUs and memory the bridge declares. `ledger_at` and `blobs_at` say where a worker
    finds the ledger and the blob store; `scratch` is where it works, on its own machine. A chain whose bridges write
    nothing (`none`) serves the checkpoint's own files."""
    import ray

    files: dict[str, Any] | None = None
    for bridge in chain:
        if bridge.task is None:
            continue
        task = ray.remote(_on_worker).options(  # pyright: ignore[reportUnknownMemberType]
            num_cpus=bridge.cpus, memory=int(bridge.memory_gib * 2**30)
        )
        told = dict((settings or {}).get(bridge.name, {}))
        reference = task.remote(
            dict(ledger_at), dict(blobs_at), (fence.scope, fence.number), checkpoint, bridge, files, target, told,
            scratch,
        )  # fmt: skip
        files = await asyncio.wrap_future(reference.future())
    if files is None:
        record = await _opened(ledger_at, blobs_at).checkpoint(checkpoint)
        if record.weights is None:
            raise ValueError(f"{checkpoint} was released: its weights were deleted")
        return record.weights
    return _MANIFEST.validate_python(files)
