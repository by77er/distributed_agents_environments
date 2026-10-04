"""Processes of one role, from a profile: an engine host, which serves the profile's engines to other machines and keeps
them serving what a run says (`rollout engines`), and a runner, which plays runs' episodes and nothing else (`rollout
runner`). Neither trains; the trainer writes down what its channel serves (`rollout_train.serving`), and they meet
through the ledger, the heartbeats beside it and the blob store (docs/guide/deploying.md).
"""

import asyncio
import socket
import ssl
from collections.abc import Sequence
from typing import Any

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Follower
from rollout_train.inference import serve_engines
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


async def host_engines(
    profile: Profile,
    run: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8820,
    address: str | None = None,
    name: str | None = None,
    token: str | None = None,
    certificate: str | None = None,
    key: str | None = None,
    ca: str | None = None,
    every: float = 2.0,
) -> None:
    """Serve the profile's engines at `host:port` until cancelled, as the server `name` (by default this machine's
    name and the port: the start of each replica's id), keeping them serving what `run` (its name or id) says its
    channels should, and beating with each replica's id and `address` (what others reach the server at; by default
    the host and port). With `token`, requests must carry it as a bearer token; with `certificate` and `key`, it serves
    TLS; with `ca`, it asks clients for certificates that CA signed. What the run says is read every `every`
    seconds."""
    import uvicorn

    ledger = ledger_of(profile)
    followed = await run_id(ledger, run)
    server = name or f"{socket.gethostname()}-{port}"
    reached = address or f"{'https' if certificate else 'http'}://{host}:{port}"
    async with profile.engines() as channels:
        if not channels:
            raise SystemExit("the profile names no channel whose engines serve here")
        follower = Follower(
            server, Checkpoints(ledger, blobs_of(profile)), followed, channels, profile.directory / "checkpoints",
            address=reached, server=server, presence=presence_of(ledger),
            about=lambda: machine_of(profile.directory, profile.directory / PROCESSES), every=every,
        )  # fmt: skip
        options: dict[str, Any] = {}
        if certificate is not None:
            options = {"ssl_certfile": certificate, "ssl_keyfile": key}
        if ca is not None:
            options |= {"ssl_ca_certs": ca, "ssl_cert_reqs": ssl.CERT_REQUIRED}
        app = serve_engines(channels, server, token=token)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off", **options)
        await asyncio.gather(uvicorn.Server(config).serve(), follower.serve())


async def run_episodes(profile: Profile, runs: Sequence[str]) -> None:
    """Play the episodes of `runs` (their names or ids; none: every run whose channels this process reaches) until
    cancelled, with the profile's channels, tool sets and pools, and no trainer."""
    ledger = ledger_of(profile)
    plays = [await run_id(ledger, each) for each in runs]
    async with profile.open(plays=plays):
        await asyncio.Event().wait()
