"""The names certificates carry, and the pods that say in their heartbeats that they are alive.

Every certificate of the cluster carries one URI SAN in the trust domain `rollout`: the gateway's is
`spiffe://rollout/gateway`; a pod's is `spiffe://rollout/pod/NAME`, where NAME is the name its starter gave it when it
asked for it (before the pod existed, so before any id the provider gives it). A pod beats beside the ledger with its
name, the identity its certificate carries, whether it is ready, and its certificate's serial. A beat never says where
the pod is reached: that is its lease's `address`, which the platform read from RunPod's API
(`rollout_train.pods.leases.PodLease.address`). Whoever reaches pods (the gateway, a run's trainer) reaches only those
whose beat is fresh and whose identity is the one named for the pod, at the address its lease says, over `https`, and
checks that identity in the TLS handshake (`rollout_train.inference.Connection.identity`).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from rollout_train.presence import Beat, alive

TRUST_DOMAIN = "rollout"
GATEWAY_IDENTITY = f"spiffe://{TRUST_DOMAIN}/gateway"
"""The identity of the gateway's client certificate: the only client a pod takes requests from."""
POD = "pod"
"""The key of a pod's own part of its beat: `{"name", "identity", "role", "ready", "serial"}`."""
_NAME = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")


def pod_identity(name: str) -> str:
    """The identity a pod's certificate carries: `spiffe://rollout/pod/NAME`. A name is lowercase letters, digits and
    hyphens, at most 63 characters, as a DNS label is."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a pod's name (lowercase letters, digits and hyphens, at most 63)")
    return f"spiffe://{TRUST_DOMAIN}/pod/{name}"


@dataclass(frozen=True)
class LivePod:
    """A pod that is alive, as its newest beat says: its name, the identity its certificate must carry, its role
    (`inference` or `trainer`), whether it is ready, and its certificate's serial (for whoever started the pod, which
    has the certificate of a pod it no longer counts as its own revoked). Where it is reached is its lease's."""

    name: str
    identity: str
    role: str
    ready: bool
    serial: str | None = None


def live(beats: Iterable[Beat], role: str | None = None) -> list[LivePod]:
    """The pods whose newest beat is fresh and names the identity named for the pod (of `role`, if given), by name. A
    beat that says another identity than its pod's name is left out: nothing reaches it. Whatever else a beat says (an
    address, say) is not read."""
    found: list[LivePod] = []
    for beat in beats:
        about: Any = beat.about.get(POD)
        if not alive(beat) or not isinstance(about, dict):
            continue
        said = cast(dict[str, Any], about)
        name, identity = said.get("name"), said.get("identity")
        if not isinstance(name, str) or name != beat.runner:
            continue
        if not _NAME.fullmatch(name) or not isinstance(identity, str) or identity != pod_identity(name):
            continue
        if role is not None and said.get("role") != role:
            continue
        serial = said.get("serial")
        found.append(LivePod(name, identity, str(said.get("role")), said.get("ready") is True,
                             serial if isinstance(serial, str) else None))  # fmt: skip
    return sorted(found, key=lambda pod: pod.name)
