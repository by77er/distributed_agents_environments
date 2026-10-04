# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray's handles and options are partly untyped.)
"""Engine hosts: one replica's engines as a Ray actor, serving what the runs bound to it should, by checkpoint name.

An `EngineHost` holds the engines of one replica (`engine`, `module:name`, made with the model and its options: vLLM in
production, scripted engines in tests) and a `Follower` that keeps them serving what each run's channel bound to it
should (`rollout_train.serving`). The ledger is the only thing it shares with whoever trains: it reads the runs'
serving records and the blob store, and needs no trainer, placement group or Ray cluster in common with them. So the
same actor is a run's own replica, started by its job, or a replica of a long-lived pool that serves every run bound
to it; the runs it serves may change while it runs (`bind`, `unbind`). Several runs' adapters sit side by side on its
engines, and full weights are loaded under the checkpoint's id, replica by replica (`rollout_train.following`).

It is asked as a server elsewhere is (`rollout_train.inference.remote.CheckpointServer`): `models` lists what it holds
(as vLLM's `/v1/models` does), and `generate` samples the checkpoint a request names, refusing one it does not hold
(`NotLoaded`). A turn caught by a full checkpoint's load waits for it, and is then refused unless what it named is still
held, so it is sampled again from the start. `HostServer` is that server over an actor handle, for a `RemoteChannel` in
the same Ray cluster; `HostPausable` holds its requests back and puts its engines to sleep for a colocated trainer
(`rollout_train.colocated`).

It asks Ray for GPUs (`host_spec`): a replica's share of its provider's, half of it where the run's trainer shares the
card. On Kubernetes, a share no node has free makes KubeRay's autoscaler start a GPU worker for it. Ray starts it again
when it dies (`max_restarts=-1`): its engines start afresh, and its follower loads what the serving records say again.
"""

import asyncio
import contextlib
import socket
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout.names import named
from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Binding, Follower
from rollout_train.inference.channel import Channel, Engine, Generation, NotLoaded
from rollout_train.inference.remote import Unreachable
from rollout_train.presence import presence_of
from rollout_train.serving import qualified

if TYPE_CHECKING:
    from rollout_train.cluster import Cluster
    from rollout_train.recorder.renderers import Renderer
    from rollout_train.run_settings import RunSettings

__all__ = ["EngineHost", "HostPausable", "HostServer", "HostSpec", "host_spec", "started"]

SCRATCH = "~/.cache/rollout/scratch"
"""Where an engine host keeps the files of the checkpoints it serves, unless it is told: on disk, on its own node."""


class EngineHost:
    """One replica's engines and the follower that keeps them serving what the runs bound to it should: started as a
    Ray actor (`started`), named `name`. `ledger_at` and `blobs_at` say where the ledger and the blob store are (as
    `rollout_train.ledger.opened` and `rollout_train.stores.opened` read them); `engine` (`module:name`) makes the
    engine with `model` and `options`. `bound` is the runs' channels it serves at first, `(run, channel)`. `replica`
    is its index among the replicas of what it serves, and how many there are. It keeps checkpoints' files under
    `directory`, and looks at what to serve every `every` seconds."""

    def __init__(
        self,
        name: str,
        ledger_at: Mapping[str, Any],
        blobs_at: Mapping[str, Any],
        engine: str,
        model: str,
        options: Mapping[str, JsonValue] | None = None,
        *,
        bound: Collection[Binding] = (),
        replica: tuple[int, int] = (0, 1),
        directory: str = SCRATCH,
        every: float = 2.0,
        beating: float = 15.0,
    ) -> None:
        from rollout_train.ledger import opened
        from rollout_train.stores import opened as store_at

        self.name = name
        self.model = model
        self.started = round(time.time(), 3)
        """When this incarnation started: Ray starts the actor again when it dies."""
        self.engine: Engine = named(engine)(model, **dict(options or {}))
        ledger = opened(ledger_at)
        self._bound: set[Binding] = {(str(run), str(channel)) for run, channel in bound}
        self.base = self._channel(name)
        """What samples the model the engine was started with, for a request that names it."""
        self.follower = Follower(
            name, Checkpoints(ledger, store_at(blobs_at)), None, {}, Path(directory).expanduser() / "engines" / name,
            bindings=self._bindings, opened=lambda run, channel: self._channel(qualified(run, channel)),
            replica=replica, presence=presence_of(ledger), about=self._about, every=every, beating=beating,
        )  # fmt: skip
        self._following = asyncio.get_event_loop().create_task(self.follower.serve())

    def _channel(self, name: str) -> Channel:
        # (nothing samples through the host's channels by tokens of text: they need no renderer)
        return Channel(name, [self.engine], cast("Renderer", None), model=self.model)

    async def _bindings(self) -> Collection[Binding]:
        return set(self._bound)

    def _about(self) -> Mapping[str, JsonValue]:
        return {"host": socket.gethostname(), "model": self.model, "started": self.started}

    async def about(self) -> Mapping[str, JsonValue]:
        """What its beats say of it: its machine, its model, and when it started."""
        return self._about()

    async def bind(self, run: str, channel: str) -> None:
        """Serve a run's channel too, from the next look on."""
        self._bound.add((run, channel))

    async def unbind(self, run: str, channel: str) -> None:
        """Serve a run's channel no more: its adapters are removed at the next look."""
        self._bound.discard((run, channel))

    async def bound(self) -> list[Binding]:
        """The runs' channels it serves, `(run, channel)`."""
        return sorted(self._bound)

    async def follow(self) -> bool:
        """Look at what to serve now, as the follower does every few seconds; whether anything changed."""
        return await self.follower.follow()

    async def models(self) -> dict[str, Any]:
        """What it holds, by name, each as a card of vLLM's `/v1/models`: the model it was started with (unless full
        weights replaced it), each adapter (its `parent` the model), each full checkpoint, with the depth each was
        published as."""
        accepted = self.engine.max_model_len
        channels = list(self.follower.bound.values())
        cards: dict[str, Any] = {}
        if not any(channel.held for channel in channels):
            cards[self.model] = {"id": self.model, "object": "model", "parent": None, "max_model_len": accepted}
        for channel in channels:
            for name, depth, full in channel.adapters():
                card = {"id": name, "object": "model", "parent": None if full else self.model}
                cards[name] = card | {"max_model_len": accepted, "depth": depth}
        return cards

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        session: str = "",
        request: str | None = None,
    ) -> Generation:
        """Sample the checkpoint `adapter` names (None: the model), as a `CheckpointServer`; `NotLoaded` where it does
        not hold it (once a load in progress has ended)."""
        channel = self._holding(adapter)
        return await channel.sample(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids,
            name=adapter, session=session,
        )  # fmt: skip

    def _holding(self, name: str | None) -> Channel:
        """The channel that holds what `name` names: the run's channel it was published on, or the model's own."""
        channels = list(self.follower.bound.values())
        for channel in channels:
            if name is not None and (name in channel.loaded or name == channel.held):
                return channel
        if name not in (None, self.model):
            raise NotLoaded(f"{self.name} does not hold {name}")
        if any(channel.held for channel in channels):
            raise NotLoaded(f"{self.name} holds full weights in place of {self.model}")
        return self.base

    async def served(self) -> list[JsonValue]:
        """What each channel serves, as its beats say (`Follower.served`)."""
        return self.follower.served()

    async def pause(self) -> None:
        """Hold new requests back, and wait for those in flight to finish."""
        await asyncio.gather(*(channel.pause() for channel in self._channels()))

    async def resume(self) -> None:
        for channel in self._channels():
            channel.resume()

    async def sleep(self) -> None:
        """Free the GPU (for a trainer that shares it); requests should be held back first (`pause`)."""
        await self.engine.sleep()

    async def wake(self) -> None:
        await self.engine.wake()

    async def close(self) -> None:
        """Stop following and end the engines (before the actor is ended)."""
        self._following.cancel()
        with contextlib.suppress(BaseException):
            await self._following
        self.engine.close()

    def _channels(self) -> list[Channel]:
        return [self.base, *self.follower.bound.values()]


@dataclass(frozen=True)
class HostSpec:
    """What starts an engine host of a provider's model: its engine (`module:name`), model and options, and what it
    asks Ray for."""

    engine: str
    model: str
    options: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    gpus: float = 0.0
    """GPUs it asks for: a fraction shares a card."""
    resources: Mapping[str, float] = field(default_factory=dict[str, float])
    """Custom resources it asks for (`[placement.engines]`), which steer it to the nodes that have them."""


def host_spec(cluster: "Cluster", provider: str, model: str, *, settings: "RunSettings | None" = None) -> HostSpec:
    """An engine host of `provider`'s `model` (an `[inference.NAME]` of the cluster, of a kind its engines run in an
    engine host, `vllm`): its kind's engine, the model's options, a replica's GPUs, and `[placement.engines]`. A run
    whose trainer shares the provider's card (`colocate_with`, by its `settings`) has its host ask for half of the
    replica's GPUs, and its trainer for the other half."""
    from rollout_train.providers import INFERENCE_KINDS

    offered = cluster.inference[provider]
    engine = INFERENCE_KINDS[offered.kind].implementation
    if engine is None or offered.kind != "vllm":
        raise ValueError(f"provider {provider} ({offered.kind}) runs no engine host: its servers are elsewhere")
    if model not in offered.models:
        raise ValueError(f"provider {provider} does not serve {model} (it serves {', '.join(offered.models)})")
    gpus = offered.gpus
    trainer = cluster.trainers.get(str(settings["trainer.provider"])) if settings is not None else None
    if trainer is not None and trainer.colocate_with == provider:
        gpus /= 2
    resources = dict(cluster.placement.get("engines", {}))
    return HostSpec(engine, model, dict(offered.models[model].options), gpus, resources)


def started(
    name: str,
    spec: HostSpec,
    ledger_at: Mapping[str, Any],
    blobs_at: Mapping[str, Any],
    *,
    bound: Collection[Binding] = (),
    replica: tuple[int, int] = (0, 1),
    namespace: str | None = None,
    detached: bool = False,
    directory: str = SCRATCH,
    every: float = 2.0,
    beating: float = 15.0,
) -> Any:
    """An engine host started as a Ray actor named `name` on the cluster this process is connected to, asking for
    what `spec` says and started again whenever it dies; its handle. A run's own host goes with the job that started it;
    a pool's is `detached`, and lives until it is ended."""
    import ray

    options: dict[str, Any] = {
        "name": name, "num_gpus": spec.gpus, "resources": dict(spec.resources), "max_restarts": -1,
        "max_concurrency": 1000,
    }  # fmt: skip
    if namespace is not None:
        options["namespace"] = namespace
    if detached:
        options["lifetime"] = "detached"
    actor = ray.remote(EngineHost).options(**options)
    return actor.remote(
        name, dict(ledger_at), dict(blobs_at), spec.engine, spec.model, dict(spec.options), bound=list(bound),
        replica=replica, directory=directory, every=every, beating=beating,
    )  # fmt: skip


class HostServer:
    """A `CheckpointServer` over an engine host actor's handle, for a `RemoteChannel` in the Ray cluster it runs in:
    each call is an actor call. A host that does not answer (it died, or is starting again) is `Unreachable`."""

    def __init__(self, handle: Any, address: str) -> None:
        self.handle = handle
        self.address = address
        self.max_model_len = 0

    async def models(self, within: float = 2.0) -> dict[str, Any]:
        try:
            listing: dict[str, Any] = await asyncio.wait_for(self.handle.models.remote(), within)
        except (TimeoutError, *_gone()) as error:
            raise Unreachable(f"{self.address}: {type(error).__name__}: {error}") from error
        with contextlib.suppress(StopIteration, KeyError, TypeError, ValueError):
            self.max_model_len = int(next(iter(listing.values()))["max_model_len"])
        return listing

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        session: str = "",
        request: str | None = None,
    ) -> Generation:
        try:
            return await self.handle.generate.remote(
                list(prompt), max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                stop_token_ids=list(stop_token_ids), adapter=adapter, session=session, request=request,
            )  # fmt: skip
        except NotLoaded as error:  # (Ray raises the host's own error, as an instance of its class)
            raise NotLoaded(str(error).splitlines()[-1]) from error
        except _gone() as error:
            raise Unreachable(f"{self.address}: {type(error).__name__}: {error}") from error

    def close(self) -> None:
        pass


class HostPausable:
    """`rollout_train.colocated.Pausable` over an engine host actor's handle: a colocated trainer holds the host's
    requests back and puts its engines to sleep while it steps."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    async def pause(self) -> None:
        await self.handle.pause.remote()

    def resume(self) -> None:
        self.handle.resume.remote()

    async def sleep(self) -> None:
        await self.handle.sleep.remote()

    async def wake(self) -> None:
        await self.handle.wake.remote()


def _gone() -> tuple[type[BaseException], ...]:
    """What Ray raises for an actor that is not there to answer: it died, or is starting again."""
    from ray.exceptions import RayActorError

    return (RayActorError,)
