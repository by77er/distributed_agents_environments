"""Processes of one role, from a profile: an engine host, which keeps the servers on its machine serving what a run says
(`rollout engines`), and a runner, which plays runs' episodes and nothing else (`rollout runner`). Neither trains; the
trainer writes down what its channel serves (`rollout_train.serving`), and they meet through the ledger, the heartbeats
beside it and the blob store (docs/guide/deploying.md).
"""

import asyncio
import socket
from collections.abc import Sequence

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Follower
from rollout_train.layout import BLOBS, LEDGER, PROCESSES
from rollout_train.ledger import Ledger
from rollout_train.ledger import opened as ledger_at
from rollout_train.presence import presence_of
from rollout_train.profile import Profile, machine_of
from rollout_train.registry import found, registry_of
from rollout_train.stores import opened


def ledger_of(profile: Profile) -> Ledger:
    """The ledger a profile names (by default files under its directory)."""
    return ledger_at(dict(profile.ledger) or {"directory": str(profile.directory / LEDGER)})


def blobs_of(profile: Profile) -> Blobs:
    """The blob store a profile names (by default files under its directory)."""
    return opened(dict(profile.blobs)) if profile.blobs else FileBlobStore(profile.directory / BLOBS)


async def run_id(ledger: Ledger, who: str) -> str:
    """A run's id, from its name or its id."""
    registry = registry_of(ledger)
    entry = found(await registry.runs(), who) if registry is not None else None
    return entry.id if entry is not None else who


async def host_engines(profile: Profile, run: str, *, name: str | None = None, every: float = 2.0) -> None:
    """Keep the vLLM servers at the addresses the profile's channels name serving what `run` (its name or id) says
    those channels should, until cancelled: each LoRA checkpoint is loaded as an adapter named by its id, from files
    fetched under the profile's directory, which the servers must read on this machine. It reads what the run says
    every `every` seconds, and beats as `name` (by default this machine's name) with what each serves."""
    ledger = ledger_of(profile)
    followed = await run_id(ledger, run)
    async with profile.engines() as channels:
        if not channels:
            raise SystemExit("the profile names no channel whose engines are servers (`RemoteEngine`)")
        follower = Follower(
            name or socket.gethostname(), Checkpoints(ledger, blobs_of(profile)), followed, channels,
            profile.directory / "checkpoints", presence=presence_of(ledger),
            about=lambda: machine_of(profile.directory, profile.directory / PROCESSES), every=every,
        )  # fmt: skip
        await follower.serve()


async def run_episodes(profile: Profile, runs: Sequence[str]) -> None:
    """Play the episodes of `runs` (their names or ids; none: every run whose channels this process reaches) until
    cancelled, with the profile's channels, tool sets and pools, and no trainer."""
    ledger = ledger_of(profile)
    plays = [await run_id(ledger, each) for each in runs]
    async with profile.open(plays=plays):
        await asyncio.Event().wait()
