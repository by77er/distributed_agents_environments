"""Bridges, declared: how a checkpoint's files in a trainer's format become files an inference provider loads.

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
| `peft` | `full` | `merge-quantize` | merged into the base, quantized as the provider's model is; only when asked |

A bridge may change an adapter's rank as the provider sees it (`rank_factor`): Tinker's adapters for Qwen3.5's
linear-attention layers carry an A of their own for each of q, k and v, joined into one projection of three times the
rank.

These are declarations only: what runs each bridge (a task on a CPU worker, its record in the ledger) is
`rollout_train.resharding` for `verbatim`, and the rest comes with the bridges' tasks.
"""

import fnmatch
import heapq
from collections.abc import Collection
from dataclasses import dataclass

__all__ = ["BRIDGES", "FORMATS", "REFUSED", "Bridge", "NoBridge", "format_of", "path", "rank_factor"]

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
    Bridge("verbatim", "peft", "peft", "rollout_train.resharding:verbatim", "the adapter's files, linked as they are"),
    Bridge(
        "full-reload",
        "full",
        "full",
        "rollout_train.resharding:verbatim",
        "the full weights' files, linked as they are and loaded under the checkpoint's name, replica by replica",
    ),
    Bridge(
        MERGE_QUANTIZE,
        "peft",
        "full",
        "rollout_lora.bridges:merge_quantize",
        "the adapter merged into its base and quantized as the provider's model is",
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
