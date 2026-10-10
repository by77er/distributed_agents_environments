"""The servers of a run's channel on RunPod's pods, and the sandbox pools its pods serve: the pods the run's leases
name, reached at the addresses their leases say, over mutual TLS with the gateway's certificate, each checked by its
own identity.

`LeasedServers` is what a `RemoteChannel` asks at each look for its servers (`discover`): for each lease of the run on
one of the channel's providers, serving the channel, whose pod beats fresh, ready for the run, with the identity named
for it (`rollout_train.pods.identity.live`), a `RemoteEngine` at the address its lease says (where RunPod's API says the
pod is reached, `https://IP:PORT`) whose connection carries the gateway's client certificate and requires the pod's
certificate to carry `spiffe://rollout/pod/NAME`. What a pod's beat says decides only whether it is ready, never where
it is reached: a lease with no address yet, or one that is not `https`, is no server. A pod that stops beating, or is
released, is no longer a server; a pod taken warm by the run becomes one once it beats ready for it.

`LeasedPools` is what a run's `PodPools` asks at each look for the pools of a kind its pods serve: for each lease of
the run on one of the providers whose pods serve the kind, with an `https` address, a `RemotePool` at
`ADDRESS/v1/sandboxes/KIND`, reached the same way (the gateway's certificate, the pod's identity checked); live while
the pod's beat is fresh and names the identity named for it. A pod's pool takes leases whether or not its engine is
ready.
"""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import httpx

from rollout.harness.remote import RemotePool
from rollout_train.inference.remote import CheckpointServer, RemoteEngine
from rollout_train.ledger import Ledger
from rollout_train.pods.identity import live, pod_identity
from rollout_train.pods.leases import IDLE, pod_leases_of
from rollout_train.pods.pools import Reached
from rollout_train.presence import presence_of

if TYPE_CHECKING:
    from rollout_train.providers import Auth, Tls

__all__ = ["SANDBOXES", "LeasedPools", "LeasedServers"]

SANDBOXES = "/v1/sandboxes"
"""Where a pod's proxy serves its sandbox pools, each under its kind (`/v1/sandboxes/KIND/acquire`)."""
POOL_TIMEOUT = httpx.Timeout(600.0, connect=5.0)
"""How long a request to a pod's pool may take: an acquire starts a sandbox (a world: a Paper server, its bots, its
task), but a pod that does not answer is given up on in 5 seconds. Asking a pool how full it is takes 10 at most
(`rollout.harness.remote.QUICK`)."""


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


class LeasedPools:
    """The pools of `kind` the pods of `run` on `providers` (by name) serve, as `RemotePool`s at their leases'
    addresses, reached as `auth` says with the cluster's `tls` (each pod's identity the one named for it)."""

    def __init__(
        self, ledger: Ledger, run: str, kind: str, providers: Collection[str], auth: "Auth", tls: "Tls | None"
    ) -> None:
        self.ledger = ledger
        self.run = run
        self.kind = kind
        self.providers = set(providers)
        self.auth = auth
        self.tls = tls
        self._pools: dict[tuple[str, str], RemotePool] = {}

    async def __call__(self) -> dict[str, Reached]:
        leases = pod_leases_of(self.ledger)
        presence = presence_of(self.ledger)
        if leases is None or presence is None:
            return {}
        addresses = {
            each.pod: each.address for each in await leases.all() if each.run == self.run
            and each.provider in self.providers and each.state != IDLE and each.address is not None
            and each.address.startswith("https://")
        }  # fmt: skip
        alive = {pod.name for pod in live(await presence.beats())}
        wanted = {(address, pod_identity(pod)) for pod, address in addresses.items()}
        for key in [each for each in self._pools if each not in wanted]:  # (a pod no longer the run's: closed)
            await self._pools.pop(key).aclose()
        found: dict[str, Reached] = {}
        for pod, address in sorted(addresses.items()):
            identity = pod_identity(pod)
            key = (address, identity)
            if key not in self._pools:
                client = self.auth.connection(self.tls, identity=identity).client(POOL_TIMEOUT)
                self._pools[key] = RemotePool(f"{address}{SANDBOXES}/{self.kind}", client=client)
            found[pod] = Reached(self._pools[key], live=pod in alive)
        return found

    async def aclose(self) -> None:
        """Close the clients of the pods it reached."""
        for pool in self._pools.values():
            await pool.aclose()
        self._pools.clear()
