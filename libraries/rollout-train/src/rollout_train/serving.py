"""What each run's channels should serve: the desired state, written down by whoever trains, followed by whoever serves.

A run's channels are named within the run (`RUN/NAME`, `qualified`). Each time the training loop serves a checkpoint, it
appends to the run's `serving` table, under its fence, that the channel serves it from now on (`Serving`): the
checkpoint's id, depth and kind, the files its engines load (the checkpoint's weights, or what a bridge made of
them), the full checkpoint an adapter is served over, and the longest turn the trainer can train on. A run that starts
from the base model says so first, with no checkpoint.

Whatever serves the channel, on any machine, reads `wanted` and loads what it says (`rollout_train.following`);
whatever samples it elsewhere reads it too, to ask for that checkpoint by name, or for one close enough to it
(`rollout_train.inference.remote`). A channel never goes back: what a channel should serve now is its record of the
greatest depth.

**A channel's mode** is in the run's start (its run settings, `channels.NAME.mode`), and says whose records it serves
(`source_of`, `serving_of`):

- the trained channel, and a channel whose mode says nothing: its own records;
- `follows`, with `follows` and `lag`: the followed channel's records, `lag` records behind its newest (the base model
  until it has more than `lag`), under its own name;
- `fixed` on a checkpoint (`channels.NAME.checkpoint`, by id): that checkpoint, at its depth, from the start; `fixed`
  on none: its own records, which nothing writes, so the base model.
"""

import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout_train.checkpoints import Manifest, checkpoints_in
from rollout_train.ledger import Fence, Ledger
from rollout_train.record import recorded_settings, table

SERVING = "serving"
"""A run's table of what its channels should serve, keyed `NAME/CHECKPOINT` (`NAME/base` for the base model)."""
BASE = "base"


@dataclass(frozen=True)
class Serving:
    """That a run's channel serves a checkpoint from now on (or, with no checkpoint, the base model)."""

    channel: str
    """The channel's name within the run."""
    checkpoint: str | None = None
    """By id; None: the model the channel's engines are started with."""
    depth: int = 0
    """The checkpoint's depth: the version its samples are stamped with."""
    kind: str = "lora"
    """`lora`, an adapter; `full`, weights loaded in place of the engines' own."""
    files: Manifest | None = None
    """What the engines load: the checkpoint's weights, or the files a bridge made of them."""
    layout: str | None = None
    """The bridge that made `files` (`rollout_train.bridges`), by name, if one did."""
    over: str | None = None
    """For an adapter over a full checkpoint, that checkpoint, by id: the engines hold its weights first."""
    model: str | None = None
    """The model the channel's line began from, by name."""
    sequence: int | None = None
    """The longest turn the trainer can train on (`Limits.sequence`), for every runner that samples the channel."""
    max_lag: int | None = None
    """How many checkpoints behind this a sample may be, where the run says (0 for an eval, which plays one
    checkpoint); None: as the runner's channel says."""
    at: float = field(default_factory=lambda: round(time.time(), 1))

    def to_json(self) -> dict[str, JsonValue]:
        return _SERVING.dump_python(self, mode="json")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Serving":
        return _SERVING.validate_python(dict(data))


_SERVING = TypeAdapter(Serving)


def qualified(run: str, channel: str) -> str:
    """A run's channel, named within the run: `RUN/NAME`."""
    return f"{run}/{channel}"


def parts(name: str) -> tuple[str, str]:
    """The run and the channel a channel's name says: a qualified name's (`qualified`), or for a name without a run,
    no run (`""`) and the name."""
    if "/" not in name:
        return "", name
    run, _, channel = name.rpartition("/")
    if not run or not channel:
        raise ValueError(f"{name!r} names no run's channel (RUN/NAME)")
    return run, channel


async def record_serving(ledger: Ledger, run: str, serving: Serving, fence: Fence) -> bool:
    """Write down, under the run's fence, that its channel serves `serving` from now on; False if it was written
    before (a loop started again serves what it served)."""
    key = f"{serving.channel}/{serving.checkpoint or BASE}"
    return await ledger.append(table(run, SERVING), key, serving.to_json(), fence)


@dataclass(frozen=True)
class Source:
    """Whose serving records a run's channel serves, as its start says (none of the three: its own)."""

    follows: str | None = None
    """The channel it follows, by name within the run."""
    lag: int = 0
    """How many records behind the followed channel's newest."""
    checkpoint: str | None = None
    """The checkpoint a fixed channel serves, by id."""


async def source_of(ledger: Ledger, run: str, channel: str) -> Source:
    """Whose records a run's channel serves: what the run's newest start says of its mode (its run settings); its own,
    where the start says nothing (a run recorded before run settings, the trained channel, a channel without a
    mode)."""
    settings = await recorded_settings(ledger, run) or {}
    if settings.get("kind", "train") == "train" and settings.get("trainer.channel", "policy") == channel:
        return Source()
    prefix = f"channels.{channel}."
    mode, follows, checkpoint = (settings.get(prefix + key) for key in ("mode", "follows", "checkpoint"))
    if mode == "follows" and isinstance(follows, str):
        return Source(follows=follows, lag=int(settings.get(prefix + "lag") or 0))
    if mode == "fixed" and isinstance(checkpoint, str):
        return Source(checkpoint=checkpoint)
    return Source()


async def serving_of(ledger: Ledger, run: str, channel: str, *, seen: Collection[str] = ()) -> list[Serving]:
    """Every checkpoint a run's channel serves, once each, in the order they were written: its own records; for a
    channel that follows another, that channel's records but the newest `lag` (deepest last); for one fixed on a
    checkpoint, that checkpoint (`source_of`). Raises `ValueError` for a channel that follows itself, by way of
    others, or is fixed on a checkpoint that does not exist or was released."""
    source = await source_of(ledger, run, channel)
    if source.follows is not None:
        if source.follows in {*seen, channel}:
            raise ValueError(f"channel {channel} of run {run} follows itself, by way of {source.follows}")
        followed = await serving_of(ledger, run, source.follows, seen={*seen, channel})
        followed.sort(key=lambda each: each.depth)
        kept = followed[: max(0, len(followed) - source.lag)]
        return [replace(each, channel=channel) for each in kept]
    if source.checkpoint is not None:
        return [await _pinned(ledger, channel, source.checkpoint)]
    return await recorded_for(ledger, run, channel)


async def recorded_for(ledger: Ledger, run: str, channel: str) -> list[Serving]:
    """Every checkpoint a run has written down that its channel serves, once each, in the order they were written."""
    return [
        Serving.from_json(record)
        for record in (await ledger.read(table(run, SERVING))).values()
        if isinstance(record, dict) and record.get("channel") == channel
    ]


async def _pinned(ledger: Ledger, channel: str, id: str) -> Serving:
    """What a channel fixed on a checkpoint serves: the checkpoint's weights, at its depth, over the full checkpoint
    it was trained over, if it was; turns sample it alone (`max_lag` 0)."""
    every = {each.id: each for each in await checkpoints_in(ledger)}
    checkpoint = every.get(id)
    if checkpoint is None or checkpoint.weights is None:
        raise ValueError(
            f"channel {channel} is fixed on {id}, which {'was released' if checkpoint else 'is no checkpoint'}"
        )
    over = checkpoint.base if checkpoint.kind != "full" and checkpoint.base in every else None
    return Serving(channel, checkpoint.id, checkpoint.depth, checkpoint.kind, checkpoint.weights, over=over, max_lag=0)


async def wanted(ledger: Ledger, run: str, channel: str) -> Serving | None:
    """What a run's channel should serve now: the record of the greatest depth it serves (`serving_of`; the newest
    among equals); None if there is none (the base model)."""
    found: Serving | None = None
    for each in await serving_of(ledger, run, channel):
        if found is None or each.depth >= found.depth:
            found = each
    return found
