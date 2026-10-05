"""The servers of a run's channel on RunPod's pods: the pods the run's leases name for the channel, reached at the
addresses their beats say, over mutual TLS with the gateway's certificate, each checked by its own identity.

`LeasedServers` is what a `RemoteChannel` asks at each look for its servers (`discover`): for each lease of the run on
one of the channel's providers, serving the channel, whose pod beats fresh, ready for the run, with the identity named
for it (`rollout_train.pods.identity.live`), a `RemoteEngine` at the pod's public address whose connection carries the
gateway's client certificate and requires the pod's certificate to carry `spiffe://rollout/pod/NAME`. A pod that stops
beating, or is released, is no longer a server; a pod taken warm by the run becomes one once it beats ready for it.
"""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from rollout_train.inference.remote import CheckpointServer, RemoteEngine
from rollout_train.ledger import Ledger
from rollout_train.pods.identity import live
from rollout_train.pods.leases import IDLE, pod_leases_of
from rollout_train.presence import presence_of

if TYPE_CHECKING:
    from rollout_train.providers import Auth, Tls

__all__ = ["LeasedServers"]


class LeasedServers:
    """The pods that serve `channel` of `run` on `providers` (by name), as `RemoteEngine`s of `model`, reached as
    `auth` says with the cluster's `tls` (the pod's identity from its beat)."""

    def __init__(
        self,
        ledger: Ledger,
        run: str,
        channel: str,
        providers: Collection[str],
        model: str,
        auth: "Auth",
        tls: "Tls | None",
    ) -> None:
        self.ledger = ledger
        self.run = run
        self.channel = channel
        self.providers = set(providers)
        self.model = model
        self.auth = auth
        self.tls = tls
        self._engines: dict[tuple[str, str], RemoteEngine] = {}

    async def __call__(self) -> Sequence[CheckpointServer]:
        leases = pod_leases_of(self.ledger)
        presence = presence_of(self.ledger)
        if leases is None or presence is None:
            return []
        names = {
            each.pod for each in await leases.all() if each.run == self.run and each.channel == self.channel
            and each.provider in self.providers and each.state != IDLE
        }  # fmt: skip
        found: list[CheckpointServer] = []
        for pod in live(await presence.beats()):
            if pod.name not in names or not pod.ready:
                continue
            key = (pod.address, pod.identity)
            if key not in self._engines:
                connection = self.auth.connection(self.tls, identity=pod.identity)
                self._engines[key] = RemoteEngine(self.model, address=pod.address, connection=connection)
            found.append(self._engines[key])
        return found

    def close(self) -> None:
        """Close the clients of the pods it reached."""
        for engine in self._engines.values():
            engine.close()
        self._engines.clear()
