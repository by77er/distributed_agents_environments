"""The gateway: a horizontally scalable service between programs and harnesses and the endpoints that sample a
policy (docs/libraries/rollout-train/gateway.md).

- `service`: `Gateway` renders a request, chooses the checkpoint, samples with the thinking budget, records the turn and
  replies; `create_app` serves it in OpenAI's and Anthropic's APIs, and natively. It keeps no session.
- `keys`: `Grant`, what a signed key lets its holder do, and `Keyring`, the secrets that sign and verify keys.
- `turns`: `TurnStore`, every turn in the ledger and the blob store, and the segments a program's run recorded.
- `client`: `GatewayEndpoints`, what a runner needs to have its recorded slots served by the gateway.

`deployed` makes a replica from a profile's `[gateway]` table.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING

from rollout_train.gateway.client import Attempt, GatewayEndpoint, GatewayEndpoints
from rollout_train.gateway.keys import Grant, KeyRefused, Keyring
from rollout_train.gateway.service import Gateway, Refused, create_app
from rollout_train.gateway.turns import Link, Reply, TurnRecord, TurnStore, turns_table, unaccepted

if TYPE_CHECKING:
    from rollout_train.profile import Profile

__all__ = [
    "Attempt",
    "Gateway",
    "GatewayEndpoint",
    "GatewayEndpoints",
    "Grant",
    "KeyRefused",
    "Keyring",
    "Link",
    "Refused",
    "Reply",
    "TurnRecord",
    "TurnStore",
    "create_app",
    "deployed",
    "turns_table",
    "unaccepted",
]


async def deployed(profile: "Profile", stack: contextlib.AsyncExitStack) -> Gateway:
    """A replica of the gateway a profile describes: its ledger and blob store, its `[gateway]` table's keys, and its
    channels, sampled as a runner samples them: those whose engines serve elsewhere routed to their servers (each run's
    from what it says its channel serves), and any other with its engines started here (each closed by `stack`)."""
    from pathlib import Path

    from rollout.harness.blobs import FileBlobStore
    from rollout.names import named
    from rollout_train.inference import Routes
    from rollout_train.layout import BLOBS, LEDGER
    from rollout_train.ledger import opened as ledger_at
    from rollout_train.profile import GatewaySpec, started_engines
    from rollout_train.stores import opened

    spec = profile.gateway or GatewaySpec()
    ledger = ledger_at(dict(profile.ledger) or {"directory": str(profile.directory / LEDGER)})
    blobs = opened(dict(profile.blobs)) if profile.blobs else FileBlobStore(profile.directory / BLOBS)
    models = {name: channel.model for name, channel in profile.channels.items()}
    await asyncio.to_thread(
        profile.directory.mkdir, parents=True, exist_ok=True
    )  # (where engines' processes are noted)
    channels = started_engines(profile, stack, models)
    routed = {
        name: channel.route(named(channel.renderer)(channel.model))
        for name, channel in profile.channels.items()
        if channel.routed
    }
    routes = Routes(routed, ledger) if routed else None
    if routes is not None:
        stack.callback(routes.close)
    keyring = Keyring.load(Path(spec.keys).expanduser()) if spec.keys else Keyring.from_environment()  # noqa: ASYNC240
    return Gateway(TurnStore(ledger, blobs), keyring, channels, routes, models)
