"""A run's channels as its start names them: what a gateway samples for a run whose settings say its channels'
providers.

A run's newest start records its run settings (`rollout_train.run_settings.recorded`): for each channel, its
providers, model, renderer, thinking budget and mode. A `ChannelDirectory` reads them the first time a run is asked
for (`load`) and builds a `RemoteChannel` for every channel whose providers it knows, over those providers' servers:
nothing registers a channel with the gateway, and a run's channels exist once its start is written. Each channel asks
what it should serve of the run's serving records (`rollout_train.serving.serving_of`), so the trained channel serves
what the run trains, a `follows` channel the followed channel's checkpoint `lag` records back, and a `fixed` channel
its pinned checkpoint or the base model. A channel served by several providers samples on all of their servers, a
session's turns going to one of them by its id.

A channel on a hosted API (a provider of the kind `api`) is an `ApiChannel` instead: it samples the channel's model
through the endpoint its provider names, at most the provider's `concurrency` requests at once across every run's
channels on it in this process, and needs no renderer.

A channel on RunPod's pods (`runpod-inference`, `runpod-host`) is a `RemoteChannel` whose servers are found at each
look: the pods the run's leases name for the channel that beat ready, each reached over mutual TLS with the gateway's
certificate and checked by its own identity (`rollout_train.pods.routing.LeasedServers`).

Providers are known by name with the servers each is reached at (`Provided`): from the cluster config (`of`: every
provider whose servers answer vLLM's API at its endpoints, every `api` provider, `Hosted`, and every provider on pods),
or given (a test's scripted servers).
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rollout.names import named
from rollout_train.inference.api import ApiChannel, Hosted
from rollout_train.inference.channel import MAX_LAG, Limits
from rollout_train.inference.remote import CheckpointServer, Connection, RemoteChannel
from rollout_train.ledger import Ledger
from rollout_train.record import recorded_settings
from rollout_train.run_settings import RunSettings
from rollout_train.serving import Serving, qualified, serving_of

if TYPE_CHECKING:
    from rollout_train.cluster import Cluster
    from rollout_train.providers import Auth, Tls
    from rollout_train.recorder.renderers import Renderer

__all__ = ["ON_PODS", "SERVED_AT_ENDPOINTS", "ChannelDirectory", "Provided", "started_settings"]

type Built = RemoteChannel | ApiChannel
"""A run's channel as the directory builds it: on servers elsewhere, or on a hosted API."""

SERVED_AT_ENDPOINTS = ("vllm", "vllm-servers")
"""The kinds of inference provider whose servers answer vLLM's API at the endpoints the cluster config names."""
ON_PODS = ("runpod-inference", "runpod-host")
"""The kinds whose servers are RunPod's pods: a run's channel on one is served by the pods the run's leases name, found
by their beats (`rollout_train.pods.routing.LeasedServers`)."""


@dataclass(frozen=True)
class Provided:
    """How a provider's servers are reached: each a URL (a vLLM server, a router in front of several) or any
    `CheckpointServer`, and the connection URLs are reached with."""

    servers: tuple["str | CheckpointServer", ...]
    connection: Connection = field(default_factory=Connection)


def _renderer(renderer: str, model: str) -> "Renderer":
    return named(renderer)(model)


async def started_settings(ledger: Ledger, run: str) -> RunSettings | None:
    """The run settings a run's newest start records (none: it records none)."""
    recorded = await recorded_settings(ledger, run)
    return RunSettings(recorded) if recorded is not None else None


class ChannelDirectory:
    """Every run's channels, built from its start when the run is first asked for (`load`), over the servers of the
    providers in `providers`, rendered with the renderer each channel names (`renderers`, given its `module:name` and
    the model; by default the renderer itself, called with the model), and over the hosted APIs in `hosted`. What each
    channel should serve and what its servers have are asked again every `every` seconds; a turn waits up to
    `patience` seconds for a server with a checkpoint close enough."""

    def __init__(
        self,
        ledger: Ledger,
        providers: Mapping[str, Provided],
        *,
        hosted: Mapping[str, Hosted] | None = None,
        pods: Mapping[str, "Auth"] | None = None,
        tls: "Tls | None" = None,
        renderers: Callable[[str, str], "Renderer"] = _renderer,
        every: float | None = None,
        patience: float = 300.0,
    ) -> None:
        self.ledger = ledger
        self.providers = dict(providers)
        self.hosted = dict(hosted or {})
        self.pods = dict(pods or {})
        """The providers whose servers are RunPod's pods, each with how its pods are reached (with `tls`)."""
        self.tls = tls
        self.renderers = renderers
        self._every = every
        self._patience = patience
        self._runs: dict[str, dict[str, Built]] = {}
        self._loading: dict[str, asyncio.Lock] = {}

    @classmethod
    def of(cls, cluster: "Cluster", ledger: Ledger, **options: Any) -> "ChannelDirectory":
        """A directory over the cluster config's providers whose servers answer vLLM's API at their endpoints
        (`SERVED_AT_ENDPOINTS`), each reached as its auth says, and its `api` providers."""
        providers = {
            name: Provided(tuple(provider.endpoints), provider.auth.connection(cluster.tls))
            for name, provider in cluster.inference.items()
            if provider.kind in SERVED_AT_ENDPOINTS and provider.endpoints
        }
        hosted = {name: Hosted.of(provider) for name, provider in cluster.inference.items() if provider.kind == "api"}
        pods = {name: provider.auth for name, provider in cluster.inference.items() if provider.kind in ON_PODS}
        return cls(ledger, providers, hosted=hosted, pods=pods, tls=cluster.tls, **options)

    async def load(self, run: str) -> dict[str, Built]:
        """A run's channels, by name, built from its newest start the first time (a run whose start names no
        provider this directory knows has none; one with no start yet is asked again next time)."""
        if run in self._runs:
            return self._runs[run]
        async with self._loading.setdefault(run, asyncio.Lock()):
            if run in self._runs:
                return self._runs[run]
            settings = await started_settings(self.ledger, run)
            if settings is None:
                return {}
            self._runs[run] = {
                name: channel for name in settings.channels if (channel := self._built(run, name, settings)) is not None
            }
            return self._runs[run]

    def channel(self, run: str, name: str) -> Built | None:
        """A run's channel, once the run is loaded."""
        return self._runs.get(run, {}).get(name)

    def channels(self) -> dict[str, Built]:
        """Every channel built so far, by its name within its run (`RUN/NAME`)."""
        return {qualified(run, name): channel for run, built in self._runs.items() for name, channel in built.items()}

    def _built(self, run: str, name: str, settings: RunSettings) -> Built | None:
        """A channel of a run, over its providers' servers or on its hosted API; none where it names a provider this
        directory does not know, or no model, or (on servers) no renderer."""
        providers = settings.providers(name)
        model, renderer = settings.get(f"channels.{name}.model"), settings.get(f"channels.{name}.renderer")
        thinking, answer = (
            settings.get(f"channels.{name}.thinking_tokens"),
            settings.get(f"channels.{name}.answer_tokens"),
        )
        limits = Limits(thinking if isinstance(thinking, int) else None, answer if isinstance(answer, int) else None)
        if len(providers) == 1 and providers[0] in self.hosted:
            hosted = self.hosted[providers[0]]
            if model is None or str(model) not in hosted.models:
                return None
            return ApiChannel(name, hosted, str(model), limits)
        known = {*self.providers, *self.pods}
        if not providers or any(each not in known for each in providers) or model is None or renderer is None:
            return None
        servers: list[str | CheckpointServer] = [
            server for each in providers if each in self.providers for server in self.providers[each].servers
        ]
        leased = [each for each in providers if each in self.pods]
        discover = None
        if leased:
            from rollout_train.pods.routing import LeasedServers

            discover = LeasedServers(self.ledger, run, name, leased, str(model), self.pods[leased[0]], self.tls)
        connection = next((self.providers[each].connection for each in providers if each in self.providers), None)
        lag = settings.get("max_lag") if settings.mode(name) == "trained" else None

        async def wanted() -> Sequence[Serving]:
            return await serving_of(self.ledger, run, name)

        return RemoteChannel(
            name,
            self.renderers(str(renderer), str(model)),
            limits,
            model=str(model),
            servers=servers,
            wanted=wanted,
            max_lag=lag if isinstance(lag, int) else MAX_LAG,
            connection=connection,
            every=self._every,
            patience=self._patience,
            discover=discover,
        )

    def close(self) -> None:
        for built in self._runs.values():
            for channel in built.values():
                channel.close()
