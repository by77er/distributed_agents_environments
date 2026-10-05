"""The servers of a run's channel on RunPod's pods: the pods the run's leases name for the channel, reached at the
addresses their leases say, over mutual TLS with the gateway's certificate, each checked by its own identity.

`LeasedServers` is what a `RemoteChannel` asks at each look for its servers (`discover`): for each lease of the run on
one of the channel's providers, serving the channel, whose pod beats fresh, ready for the run, with the identity named
for it (`rollout_train.pods.identity.live`), a `RemoteEngine` at the address its lease says (where RunPod's API says the
pod is reached, `https://IP:PORT`) whose connection carries the gateway's client certificate and requires the pod's
certificate to carry `spiffe://rollout/pod/NAME`. What a pod's beat says decides only whether it is ready, never where
it is reached: a lease with no address yet, or one that is not `https`, is no server. A pod that stops beating, or is
released, is no longer a server; a pod taken warm by the run becomes one once it beats ready for it.
"""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from rollout_train.inference.remote import CheckpointServer, RemoteEngine
from rollout_train.ledger import Ledger
from rollout_train.pods.identity import live, pod_identity
from rollout_train.pods.leases import IDLE, pod_leases_of
from rollout_train.presence import presence_of

if TYPE_CHECKING:
    from rollout_train.providers import Auth, Tls

__all__ = ["LeasedServers"]


class LeasedServers:
    """The pods that serve `channel` of `run` on `providers` (by name), as `RemoteEngine`s of `model`, reached at
    their leases' addresses as `auth` says with the cluster's `tls` (each pod's identity the one named for it)."""

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
        addresses = {
            each.pod: each.address for each in await leases.all() if each.run == self.run
            and each.channel == self.channel and each.provider in self.providers and each.state != IDLE
            and each.address is not None and each.address.startswith("https://")
        }  # fmt: skip
        found: list[CheckpointServer] = []
        for pod in live(await presence.beats()):
            address = addresses.get(pod.name)
            if address is None or not pod.ready:
                continue
            identity = pod_identity(pod.name)
            key = (address, identity)
            if key not in self._engines:
                connection = self.auth.connection(self.tls, identity=identity)
                self._engines[key] = RemoteEngine(self.model, address=address, connection=connection)
            found.append(self._engines[key])
        return found

    def close(self) -> None:
        """Close the clients of the pods it reached."""
        for engine in self._engines.values():
            engine.close()
        self._engines.clear()
