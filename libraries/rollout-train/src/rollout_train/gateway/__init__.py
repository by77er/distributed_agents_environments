"""The gateway: a horizontally scalable service between programs and harnesses and the endpoints that sample a
policy (docs/libraries/rollout-train/gateway.md).

- `service`: `Gateway` renders a request, chooses the checkpoint, samples with the thinking budget, records the turn and
  replies; it also scores tokens it is handed (`ScoreRequest`), recording that as a turn too. `create_app` serves it
  in OpenAI's and Anthropic's APIs, and natively. It keeps no session.
- `keys`: `Grant`, what a signed key lets its holder do, and `Keyring`, the secrets that sign and verify keys.
- `turns`: `TurnStore`, every turn in the ledger and the blob store, and the segments a program's run recorded.
- `client`: `GatewayEndpoints`, what a runner needs to have its recorded slots served by the gateway.
- `directory`: `ChannelDirectory`, every channel a run's start names, built over its providers' servers.

`rollout gateway --cluster` serves a replica over the cluster config's stores, keys and providers
(`ChannelDirectory.of`); a run's driver serves one of its own over its channels (`rollout_train.jobs`).
"""

from rollout_train.gateway.client import Attempt, GatewayEndpoint, GatewayEndpoints
from rollout_train.gateway.directory import ChannelDirectory, Provided
from rollout_train.gateway.keys import Grant, KeyRefused, Keyring
from rollout_train.gateway.service import Gateway, Refused, ScoreRequest, create_app
from rollout_train.gateway.turns import Link, Reply, TurnRecord, TurnStore, turns_table, unaccepted

__all__ = [
    "Attempt",
    "ChannelDirectory",
    "Gateway",
    "GatewayEndpoint",
    "GatewayEndpoints",
    "Grant",
    "KeyRefused",
    "Keyring",
    "Link",
    "Provided",
    "Refused",
    "Reply",
    "ScoreRequest",
    "TurnRecord",
    "TurnStore",
    "create_app",
    "turns_table",
    "unaccepted",
]
