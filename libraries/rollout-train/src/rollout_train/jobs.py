# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray's handles and options are partly untyped.)
"""A run's job: its driver builds the run from the cluster config and the run's settings, claims what it needs, and
runs the loop of the run's kind.

`python -m rollout_train.jobs LAUNCH` is every run's entrypoint (`rollout_train.submitting.submit` starts it as a Ray
job, or as a RayJob on Kubernetes; `rollout train --here` runs it in the calling process). It reads the cluster config
(`rollout_train.cluster.located`: the one it was handed, else the file found), opens the stores
(`rollout_train.stores.Stores`), reads its launch (`rollout_train.launches`), and:

1. checks the run's settings again against the cluster config, with what it finds now (`rollout_train.launching
   .checked`): a refusal ends the run, the reasons recorded as its end and on its launch;
2. asks Ray for what the run needs (`Run.start`): an engine host actor of its own for each replica of a channel on a
   `vllm` provider (`rollout_train.inference.hosts`), the trainer as an actor with its share of a GPU (half of the
   card where it is colocated with the trained channel's engines, which then sleep while it steps), on the driver's
   node, since a step's files are handed to it by path. While something waits for Ray, the driver beats as
   `run/RUN` saying what it waits for, and notes it on its launch;
3. samples every channel its settings name through a gateway in its own process (`rollout_train.gateway.Gateway`):
   a channel on engine hosts or on servers elsewhere (`vllm-servers`, RunPod pods) is a routed channel, sampled by
   checkpoint name from what the run's serving records say (`rollout_train.inference.Routes`); a channel on Tinker is
   sampled by engines in this process. Its start records its settings, so the cluster's gateway can serve its
   channels on providers it reaches (`rollout_train.gateway.ChannelDirectory`);
4. plays its episodes with a runner in its own process (`rollout_train.rollouts.EpisodeRunner` over
   `rollout.local.LocalRunner`), with the sandbox pools of the cluster's `[sandboxes]` its environment's programs
   declare, the tool sets of its `[tools]`, and the gateway, served on this node for harnesses;
5. runs the loop of its kind (`train`, `evaluate`, `imitate`, `check`), with bridges as Ray tasks
   (`rollout_train.bridges.on_ray`) where the trained channel's provider loads another format than the trainer makes;
6. on the way out, ends what it started (Ray ends the actors with the job in any case), and notes how it ended.

A run's channel on a `vllm` provider gets engine hosts of its own (`Run.start`, `hosted`).
"""

import argparse
import asyncio
import contextlib
import dataclasses
import inspect
import json
import math
import os
import secrets
import shutil
import socket
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout.environment import Environment, binding_for
from rollout.harness.imports import ToolBinding
from rollout.harness.sandboxes import MemoryLeases, Pool, PoolBinding, SandboxPool
from rollout.local import LocalRunner
from rollout.names import named
from rollout_train.bridges import AUTO, Bridge, NoBridge, format_of, on_ray, path
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest
from rollout_train.cluster import Cluster, located
from rollout_train.colocated import Colocated
from rollout_train.gateway import Gateway, GatewayEndpoints, Keyring, TurnStore
from rollout_train.inference import Channel, Limits, Route, Routes
from rollout_train.inference.channel import MAX_LAG
from rollout_train.launching import Refused, checked, declared, ray_capacity
from rollout_train.ledger import Fence
from rollout_train.machine import measured
from rollout_train.presence import presence_of
from rollout_train.providers import INFERENCE_KINDS, TrainerProvider, settings_of
from rollout_train.record import ENDS, STARTS, ending, scope, start_header, table, trained_objective
from rollout_train.record import FAILED as RUN_FAILED
from rollout_train.registry import Entry, registry_of, resolved
from rollout_train.rollouts.scheduler import EpisodeRunner
from rollout_train.run_settings import KINDS, RunSettings, is_trainers, objective_in, recorded
from rollout_train.sandboxes import admits, keep, leases_of
from rollout_train.stores import Stores, blobs_at, ledger_at
from rollout_train.trainer import Budget, Files, Item, Step, Trainer, objective_of

if TYPE_CHECKING:
    from rollout_train.objectives import Objective

__all__ = [
    "NotEnoughMemory",
    "Run",
    "TrainerActor",
    "TrainerClient",
    "driven",
    "imitated",
    "main",
    "ran",
    "run_directory",
    "taken_by",
]

TRAIN, EVAL, IMITATE, CHECK = KINDS
ENTRYPOINT = "rollout_train.jobs"
"""The module a run's job runs (`python -m rollout_train.jobs LAUNCH`)."""
LOOK = 2.0
"""Seconds between the driver's looks at what it waits for from Ray."""
FEED = "feed"
"""Under a run's directory: the monitor's feed."""


class NotEnoughMemory(Exception):
    """Stopping is better than exhausting the machine (a host may shut down rather than kill one process)."""


def available_memory_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return math.inf
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return math.inf


def require_memory(gib: float, purpose: str) -> None:
    available = available_memory_gib()
    if available < gib:
        raise NotEnoughMemory(f"{available:.1f} GiB of system memory is available; {gib:.0f} GiB is needed {purpose}")


def _needs(gib: float, purpose: str) -> Callable[[], None] | None:
    return (lambda: require_memory(gib, purpose)) if gib else None


def run_directory(cluster: Cluster, run: str) -> Path:
    """Where a run keeps its files on its driver's node: its feed, the checkpoints in use, fetched bases."""
    return Path(cluster.scratch).expanduser() / "runs" / run


class TrainerActor:
    """A trainer in a Ray actor: made with `implementation` (`module:name`), the model and its settings, and asked
    for steps by a `TrainerClient`."""

    def __init__(self, implementation: str, model: str, settings: Mapping[str, Any]) -> None:
        self.trainer: Trainer = named(implementation)(model, **dict(settings))

    def described(self) -> dict[str, Any]:
        """What the client says of the trainer: its budget, its weights, its objective, its changeable settings."""
        changeable = getattr(self.trainer, "changeable", None)
        return {
            "budget": self.trainer.budget,
            "weights": self.trainer.weights,
            "objective": objective_of(self.trainer),
            "changeable": dict(changeable) if isinstance(changeable, Mapping) else None,
        }

    async def step(self, batch: list[Item], seed: int, parent: Files | None, into: Path) -> Step:
        return await self.trainer.step(batch, seed=seed, parent=parent, into=into)

    def change(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        change = getattr(self.trainer, "change", None)
        if change is None:
            raise ValueError(f"the trainer takes no settings between steps (not {', '.join(settings)})")
        change(settings)
        return dict(getattr(self.trainer, "changeable", {}))


class TrainerClient:
    """A `Trainer` over a `TrainerActor`'s handle: each step is an actor call (an error the trainer raised is raised
    as itself), and what it takes between steps is changed there."""

    def __init__(self, handle: Any, described: Mapping[str, Any]) -> None:
        self.handle = handle
        self.budget: Budget = described["budget"]
        self.weights: str = described["weights"]
        self.objective = described["objective"]
        self._changeable: dict[str, JsonValue] | None = described["changeable"]

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return dict(self._changeable or {})

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        import ray

        if self._changeable is None:
            raise ValueError(f"the trainer takes no settings between steps (not {', '.join(settings)})")
        self._changeable = cast(
            dict[str, JsonValue], _raised(lambda: ray.get(self.handle.change.remote(dict(settings))))
        )

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        from ray.exceptions import RayTaskError

        try:
            return await self.handle.step.remote(list(batch), seed, parent, into)
        except RayTaskError as error:
            raise _cause(error) from None


def _raised[T](call: Callable[[], T]) -> T:
    from ray.exceptions import RayTaskError

    try:
        return call()
    except RayTaskError as error:
        raise _cause(error) from None


def _cause(error: Any) -> BaseException:
    """The error an actor's method raised, as itself."""
    cause = getattr(error, "cause", None)
    return cause if isinstance(cause, BaseException) else error


def takes_objective(making: Any) -> bool:
    """Whether what makes a trainer takes an objective among its settings (`objective`, or any keyword)."""
    return "objective" in taken_by(making, {"objective": None})


def taken_by(making: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Those of `settings` that what makes a trainer takes: every one where it takes any keyword, else those it names
    (a run's recorded settings hold every setting of its trainer's kind, which a trainer of its own may not all
    take)."""
    try:
        parameters = inspect.signature(making).parameters.values()
    except (TypeError, ValueError):
        return dict(settings)
    if any(each.kind is inspect.Parameter.VAR_KEYWORD for each in parameters):
        return dict(settings)
    names = {each.name for each in parameters}
    return {key: value for key, value in settings.items() if key in names}


@dataclass
class Waiting:
    """What the driver asked Ray for and has not got yet: each actor by name, with what it asked for."""

    actors: dict[str, tuple[Any, str]] = field(default_factory=dict[str, tuple[Any, str]])

    def add(self, name: str, handle: Any, wants: str) -> None:
        self.actors[name] = (handle, wants)


@dataclass
class Run:
    """A run being built from its settings, and what it started: everything the loop of its kind is given."""

    cluster: Cluster
    stores: Stores
    settings: RunSettings
    run: Entry
    preset: str | None = None
    """The preset its settings came from (`NAME@N`), recorded as provenance."""
    resumes: bool = False
    """Whether it goes on from where a start of it before stopped (it trains with the objective that one did)."""
    noted: Callable[[str], Awaitable[None]] | None = None
    """Told what the run waits for, as it changes (its launch's detail)."""
    environment: Environment | None = None
    started: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the run's start records beside what its loop knows."""
    directory: Path = field(default_factory=Path)
    origin: str | None = None
    """The checkpoint it starts from, by id (none: the base model)."""
    trainer: Trainer | None = None
    hosts: dict[str, list[Any]] = field(default_factory=dict[str, list[Any]])
    """The engine hosts of each channel, by channel."""
    channels: dict[str, Channel] = field(default_factory=dict[str, Channel])
    """The channels whose engines are in this process (Tinker's)."""
    routes: Routes | None = None
    gateway: Gateway | None = None
    recorder: GatewayEndpoints | None = None
    runner: EpisodeRunner | None = None
    feed: Any = None
    tool_bindings: dict[str, ToolBinding] = field(default_factory=dict[str, ToolBinding])
    pool_bindings: dict[str, PoolBinding] = field(default_factory=dict[str, PoolBinding])
    runs: set[str] = field(default_factory=set[str])
    """The runs its runner plays: its own, and its evals'."""
    chain: tuple[Bridge, ...] = ()
    """The bridges the trained channel's files are made by (none: the trainer's files as they are)."""
    objective: Mapping[str, JsonValue] | None = None
    """The objective its trainer is made with, where the run's settings do not say it all (an imitate run's)."""
    waiting: Waiting = field(default_factory=Waiting)
    trainer_handle: Any = None

    @property
    def ledger(self) -> Any:
        return self.stores.ledger

    @property
    def checkpoints(self) -> Checkpoints:
        return self.stores.checkpoints

    @property
    def kind(self) -> str:
        return self.settings.kind

    @property
    def channel(self) -> str:
        """The channel the run trains or plays: the trained one, else the first its settings name."""
        trained = self.settings["trainer.channel"]
        channels = self.settings.channels
        return str(trained) if trained in channels or not channels else channels[0]

    async def start(self, stack: contextlib.AsyncExitStack, *, training: bool) -> None:
        """Start what the run needs, registering with `stack` how each is stopped: the environment, the trainer (with
        `training`), each channel's engines, the gateway, the pools and the runner; then wait for what Ray has yet to
        give."""
        self.directory = run_directory(self.cluster, self.run.id)
        await asyncio.to_thread(self.directory.mkdir, parents=True, exist_ok=True)
        self.runs.add(self.run.id)
        start = self.settings["start"]
        if isinstance(start, str):
            self.origin = await resolved(self.ledger, registry_of(self.ledger), start)
        if training:
            await self._trainer()
        self._channels()
        await self._waited()
        if training and self.trainer is not None:
            self._colocated()
        if self.kind in (TRAIN, EVAL, CHECK):
            await self._played(stack)

    async def _trainer(self) -> None:
        """The trainer, as an actor on this node asking for its GPUs (half of the card, where the trained channel's
        engine hosts share it)."""
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        provider = self.cluster.trainers[str(self.settings["trainer.provider"])]
        model = self.settings.trainer_model
        if model is None:
            raise ValueError("the run says no model its trainer trains (trainer.model, or the trained channel's)")
        model = await self._over(provider, model)
        given: dict[str, Any] = {
            key.removeprefix("trainer."): value for key, value in self.settings.values.items() if is_trainers(key)
        }
        making = provider.implementation
        if self.kind in (TRAIN, IMITATE):
            kept = await trained_objective(self.ledger, self.run.id)  # (a run started again: what it trained with)
            objective = kept if kept is not None and self.resumes else None
            with contextlib.suppress(ValueError, ImportError):
                if takes_objective(named(making)):
                    given["objective"] = self.objective or objective or objective_in(self.settings).to_json()
        if (project := provider.secrets.get("project")) is not None and (said := project.resolve()) is not None:
            given["project"] = said
        gpus = provider.gpus / 2 if self._colocating(provider) else provider.gpus
        name = f"run/{self.run.id}/trainer"
        node = ray.get_runtime_context().get_node_id()
        actor = ray.remote(TrainerActor).options(
            name=name, num_gpus=gpus, num_cpus=1, max_concurrency=4,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False),
        )  # fmt: skip
        with contextlib.suppress(ImportError, ValueError):  # (one this process cannot import is told everything)
            given = taken_by(named(making), given)
        handle = actor.remote(making, model, given)
        self.waiting.add(name, handle, f"{gpus:g} GPU and 1 CPU on the driver's node" if gpus else "1 CPU")
        self.trainer_handle = handle

    def _colocating(self, provider: TrainerProvider) -> bool:
        """Whether the trainer shares the GPU of the trained channel's engine hosts."""
        trained = self.settings.trained
        if trained is None or provider.colocate_with is None:
            return False
        return provider.colocate_with in self.settings.providers(trained)

    async def _over(self, provider: TrainerProvider, model: str) -> str:
        """What the trainer is made over: the model; for a run that starts from a full checkpoint (or an adapter over
        one), that checkpoint's files, fetched here."""
        if self.origin is None:
            return model
        under = await self.checkpoints.under(await self.checkpoints.checkpoint(self.origin))
        if under is None or under.weights is None:
            return model
        return str(await self.checkpoints.files(under.weights, self.directory / "bases" / under.id))

    def _channels(self) -> None:
        """Each channel's engines: engine hosts for a `vllm` provider, servers elsewhere for a provider reached at
        addresses, engines in this process for Tinker."""
        routes: dict[str, Route] = {}
        sequence = self._sequence()
        for channel in self.settings.channels:
            providers = self.settings.providers(channel)
            model = self.settings.get(f"channels.{channel}.model")
            renderer = self.settings.get(f"channels.{channel}.renderer")
            if not providers or model is None or renderer is None:
                continue  # (a channel no slot samples: validation refused one a slot does)
            thinking = self.settings.get(f"channels.{channel}.thinking_tokens")
            answer = self.settings.get(f"channels.{channel}.answer_tokens")
            limits = Limits(
                thinking if isinstance(thinking, int) else None,
                answer if isinstance(answer, int) else None,
                sequence=sequence if channel == self.settings.trained else None,
            )
            made = named(str(renderer))(str(model))
            kinds = {self.cluster.inference[each].kind for each in providers}
            first = providers[0]
            if "tinker" in kinds:
                if len(providers) > 1:
                    raise ValueError(f"channel {channel}: a channel Tinker samples has no other provider")
                self.channels[channel] = self._tinker(channel, first, str(model), made, limits)
                continue
            if "api" in kinds:
                raise ValueError(f"channel {channel}: an api provider's channel is sampled by the cluster's gateway")
            servers: list[Any] = []
            connection = None
            for name in providers:
                provider = self.cluster.inference[name]
                if provider.kind == "vllm":
                    servers += self.hosted(channel, name, str(model))
                else:
                    via = provider.settings.get("via")
                    servers += [str(via)] if via else list(provider.endpoints)
                    connection = connection or provider.auth.connection(self.cluster.tls)
            lag = self.settings["max_lag"] if channel == self.settings.trained else None
            routes[channel] = Route(
                made, str(model), tuple(servers), limits, max_lag=lag if isinstance(lag, int) else MAX_LAG,
                **({"connection": connection} if connection is not None else {}),
            )  # fmt: skip
        self.routes = Routes(routes, self.ledger) if routes else None

    def _sequence(self) -> int | None:
        """The longest turn the trained channel takes: the longest segment its trainer trains on."""
        provider = self.cluster.trainers.get(str(self.settings["trainer.provider"]))
        if self.kind != TRAIN or provider is None:
            return None
        said = self.settings.get("trainer.segment_tokens")
        return said if isinstance(said, int) else provider.segment_tokens

    def hosted(self, channel: str, provider: str, model: str) -> list[Any]:
        """The servers of a channel on a `vllm` provider: engine hosts of the run's own, one per replica, each bound to
        the run's channel and asking Ray for its share of a GPU."""
        from rollout_train.inference.hosts import HostServer, host_spec, started

        offered = self.cluster.inference[provider]
        asked = self.settings[f"channels.{channel}.replicas"]
        count = asked if isinstance(asked, int) else offered.replicas
        spec = host_spec(self.cluster, provider, model, settings=self.settings)
        servers: list[Any] = []
        for index in range(count):
            name = f"run/{self.run.id}/engine/{channel}/{index}"
            handle = started(
                name, spec, ledger_at(self.cluster), blobs_at(self.cluster), bound=[(self.run.id, channel)],
                replica=(index, count), directory=self.cluster.scratch,
            )  # fmt: skip
            self.hosts.setdefault(channel, []).append(handle)
            self.waiting.add(name, handle, f"{spec.gpus:g} GPU" if spec.gpus else "its engine")
            servers.append(HostServer(handle, name))
        return servers

    def _tinker(self, channel: str, provider: str, model: str, renderer: Any, limits: Limits) -> Channel:
        offered = self.cluster.inference[provider]
        kind = INFERENCE_KINDS[offered.kind]
        assert kind.implementation is not None
        options = dict(offered.models[model].options) if model in offered.models else {}
        options.setdefault("max_model_len", offered.models[model].context if model in offered.models else 32_768)
        if (project := offered.secrets.get("project")) is not None and (said := project.resolve()) is not None:
            options["project"] = said
        return Channel(channel, [named(kind.implementation)(model, **options)], renderer, limits, model=model)

    async def _waited(self) -> None:
        """Wait for every actor asked for to be ready, beating as `run/RUN` and noting on the launch what it waits
        for, every `LOOK` seconds."""
        import ray

        pending = dict(self.waiting.actors)
        if not pending:
            return
        presence = presence_of(self.ledger)
        readies = {name: handle.__ray_ready__.remote() for name, (handle, _) in pending.items()}
        said: str | None = None
        while readies:
            done, _ = await asyncio.to_thread(ray.wait, list(readies.values()), num_returns=len(readies), timeout=LOOK)
            for name in [name for name, ready in readies.items() if ready in done]:
                try:
                    ray.get(readies.pop(name))
                except Exception as error:  # (an actor that could not start: the run cannot go on)
                    raise RuntimeError(f"{name} did not start: {_cause(error)}") from None
            if not readies:
                break
            waits = [f"{name} ({pending[name][1]}{_state(pending[name][0])})" for name in sorted(readies)]
            now = "waits for " + ", ".join(waits)
            if now != said:
                said = now
                if self.noted is not None:
                    await self.noted(now)
            if presence is not None:
                about: dict[str, JsonValue] = {"kind": "run", "run": self.run.id, "host": socket.gethostname(),
                                               "waiting": cast(JsonValue, waits)}  # fmt: skip
                with contextlib.suppress(Exception):
                    await presence.beat(f"run/{self.run.id}", about)
        if said is not None and self.noted is not None:
            await self.noted("running")
        handle = self.trainer_handle
        if handle is not None:
            described = cast(dict[str, Any], _raised(lambda: ray.get(handle.described.remote())))
            self.trainer = TrainerClient(handle, described)

    def _colocated(self) -> None:
        """The trainer, taking turns with the trained channel's engine hosts where it shares their GPU."""
        from rollout_train.inference.hosts import HostPausable

        provider = self.cluster.trainers[str(self.settings["trainer.provider"])]
        trained = self.settings.trained
        if self.trainer is None or trained is None or not self._colocating(provider):
            return
        hosts = [HostPausable(each) for each in self.hosts.get(trained, [])]
        if hosts:
            guard = _needs(self.cluster.guards.training_gib, "to train")
            self.trainer = Colocated(self.trainer, hosts, guard=guard)

    async def _played(self, stack: contextlib.AsyncExitStack) -> None:
        """The gateway (served on this node for harnesses), the feed, the tool sets and pools, and the runner."""
        from rollout_train.monitor import RunFeed

        feed = RunFeed(self.directory / FEED, keep=self.cluster.monitor.feed_episodes)
        stack.callback(feed.close)
        self.feed = feed
        if self.routes is not None:
            stack.callback(self.routes.close)
        keyring = Keyring.parse([("run", secrets.token_urlsafe(32))])  # (keys only this process mints and takes)
        models = {
            name: str(self.settings.get(f"channels.{name}.model")) for name in self.settings.channels
            if self.settings.get(f"channels.{name}.model") is not None
        }  # fmt: skip
        store = TurnStore(self.ledger, self.stores.blobs)
        self.gateway = Gateway(store, keyring, self.channels, self.routes, models, hooks=[feed])
        port = _free_port()
        self.recorder = GatewayEndpoints.of(self.gateway, f"http://127.0.0.1:{port}")
        _background(stack, _serve(self.gateway, port))
        for name, tool in self.cluster.tools.items():
            self.tool_bindings[name] = ToolBinding(url=tool.url)
        pools = await self._pools(stack)
        runner = LocalRunner(gateway=self.recorder, tool_sets={}, pools=pools, hooks=[feed], blobs=self.stores.blobs)
        self.runner = EpisodeRunner(
            f"run/{self.run.id}", self.ledger, runner, self.recorder, self.stores.blobs,
            places=int(cast(int, self.settings["episodes_at_once"])), pools=pools, runs=self.runs, hooks=[feed],
            guard=_needs(self.cluster.guards.runs_gib, "to run more episodes"), presence=presence_of(self.ledger),
            about=self._about,
        )  # fmt: skip
        await self.runner.prepare()
        await runner.launch()
        stack.push_async_callback(runner.close)
        _background(stack, self.runner.serve())

    async def _pools(self, stack: contextlib.AsyncExitStack) -> dict[str, Pool]:
        """A pool of each kind of sandbox the environment's programs declare, from the cluster's `[sandboxes]`, with
        its keeper; its leases are kept beside the ledger."""
        pools: dict[str, Pool] = {}
        if self.environment is None:
            return pools
        kinds, _ = await asyncio.to_thread(declared, self.environment)
        for kind in sorted(kinds):
            section = self.cluster.sandboxes.get(kind)
            if section is None:
                continue  # (validation refused a run whose environment needs it)
            provider = named(section.provider)(self.directory, size=section.size, **dict(section.settings))
            beats = presence_of(self.ledger)
            pool = SandboxPool(
                provider, name=f"{kind}@{self.run.id}", leases=leases_of(self.ledger) or MemoryLeases(),
                admits=admits(self.ledger, beats),
            )  # fmt: skip
            stack.push_async_callback(pool.close)  # (after the runner: its runs release theirs first)
            _background(stack, keep(pool, self.ledger, presence_of(self.ledger)))
            pools[kind] = pool
            self.pool_bindings[kind] = PoolBinding(local=kind)
        return pools

    def _about(self) -> dict[str, JsonValue]:
        """What the runner says in each beat: its machine, the run, and what each channel serves."""
        channels: list[JsonValue] = [
            {"channel": name, "adapter": channel.serving, "version": channel.version, **channel.take()}
            for name, channel in self.channels.items()
        ]
        for name, channel in (self.routes.channels() if self.routes is not None else {}).items():
            servers: list[JsonValue] = list(channel.servers())
            channels.append({"channel": name, **channel.take(), "servers": servers})
        return {
            "host": socket.gethostname(), "directory": str(self.directory), "machine": measured(self.directory),
            "run": self.run.id, "channels": channels,
        }  # fmt: skip

    def binding(self, environment: Environment) -> Any:
        """How an environment's episodes are played: each slot from the channel the settings bind it to (the run's
        channel unless said), each import and pool where the cluster serves it."""
        slots = {
            key.removeprefix("slots."): str(value) for key, value in self.settings.values.items()
            if key.startswith("slots.") and isinstance(value, str)
        }  # fmt: skip
        return binding_for(environment, self.channel, self.tool_bindings, self.pool_bindings, slots)

    async def publish(self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
                      ) -> int:  # fmt: skip
        """Serve a checkpoint on a channel: on engines in this process (Tinker's), loaded now; elsewhere, its engine
        hosts and servers follow the run's serving record, and this returns the version given."""
        if channel in self.channels:
            return await self.channels[channel].publish(adapter, path, version, full=full)
        return version or 0

    async def bridged(self, checkpoint: Checkpoint, fence: Fence) -> Manifest:
        """The files the trained channel's engines load for a checkpoint, made by its bridges as Ray tasks."""
        return await self._bridge(self.chain, checkpoint, fence)

    async def _bridge(self, chain: Sequence[Bridge], checkpoint: Checkpoint, fence: Fence) -> Manifest:
        told = {
            name: {key: value for key, value in (("cpus", each.cpus), ("memory_gib", each.memory_gib)) if value}
            for name, each in self.cluster.bridges.items()
        }
        model = self.settings.get(f"channels.{self.channel}.model")
        return await on_ray(
            ledger_at(self.cluster), blobs_at(self.cluster), fence, checkpoint.id, chain,
            target=str(model) if model is not None else None, settings=cast(Any, told), scratch=self.cluster.scratch,
        )  # fmt: skip

    def chosen(self, formats: frozenset[str]) -> tuple[Bridge, ...]:
        """The bridges that make what the run's channel's first provider loads from checkpoints in `formats` (none
        where the files are served as they are); `ValueError` where none does."""
        providers = self.settings.providers(self.channel)
        if not providers:
            return ()
        loads = self.cluster.inference[providers[0]].capabilities.loads
        wanted = str(self.settings[f"channels.{self.channel}.bridge"] or AUTO)
        found: tuple[Bridge, ...] | NoBridge = NoBridge("", "", "nothing to bridge")
        for each in sorted(formats):
            found = path(each, loads, wanted=wanted)
            if not isinstance(found, NoBridge):
                break
        if isinstance(found, NoBridge):
            raise ValueError(found.reason)
        return found if any(each.task is not None for each in found) else ()

    async def eval_run(self, step: int | None = None, part: int | None = None) -> str:
        """The run of an eval, by id, which the runner plays: with `step`, the eval of the checkpoint this run made at
        that step (`RUN-eval-STEP`, called `NAME-eval-STEP`); else this run (an eval itself). With `part`, the run that
        plays that entry of an eval of several (its id and `-PART`)."""
        from rollout_train.registry import _registered  # pyright: ignore[reportPrivateUsage]

        if step is None and part is None:
            return self.run.id
        id, name = self.run.id, self.run.name
        if step is not None:
            id, name = f"{id}-eval-{step}", f"{name}-eval-{step}"
        if part is not None:
            id, name = f"{id}-{part}", f"{name}-{part}"
        entry = await _registered(registry_of(self.ledger), id, name)
        self.runs.add(entry.id)
        return entry.id

    async def bookmarked(self) -> set[str]:
        """The checkpoints bookmarks name (which keep their files)."""
        registry = registry_of(self.ledger)
        return {mark.checkpoint for mark in await registry.bookmarks()} if registry else set()

    async def made(self, checkpoint: Checkpoint) -> None:
        """Carry the run's bookmark, if it names one, to a checkpoint it made."""
        registry, bookmark = registry_of(self.ledger), self.settings["bookmark"]
        if registry is not None and isinstance(bookmark, str) and bookmark:
            await registry.bookmark(bookmark, checkpoint.id)


def _state(handle: Any) -> str:
    """What Ray says of an actor it has not made ready, where it says (its state: pending creation, restarting)."""
    with contextlib.suppress(Exception):
        from ray.util.state import get_actor

        found: Any = get_actor(handle._actor_id.hex(), timeout=2)
        if found is not None and found.state != "ALIVE":
            return f": {str(found.state).lower().replace('_', ' ')}"
    return ""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def _serve(gateway: Gateway, port: int) -> None:
    """The run's gateway over HTTP on this node, for harnesses."""
    import uvicorn

    from rollout_train.gateway import create_app

    config = uvicorn.Config(create_app(gateway), host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    await uvicorn.Server(config).serve()


def _background(stack: contextlib.AsyncExitStack, work: Coroutine[Any, Any, None]) -> None:
    """Run `work` until the stack is closed."""
    task = asyncio.ensure_future(work)

    async def stop() -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    stack.push_async_callback(stop)


@contextlib.asynccontextmanager
async def started(run: Run, *, training: bool) -> AsyncGenerator[Run]:
    """A run started (`Run.start`), and everything it started stopped on the way out."""
    async with contextlib.AsyncExitStack() as stack:
        stack.callback(_ended, run)
        await run.start(stack, training=training)
        yield run


def _ended(run: Run) -> None:
    """End the actors a run started (Ray ends them with the job in any case; a run in a process that goes on, such as
    `--here`, ends them now)."""
    import ray

    for handle in [*(each for listed in run.hosts.values() for each in listed), run.trainer_handle]:
        if handle is not None:
            with contextlib.suppress(Exception):
                ray.kill(handle, no_restart=True)


async def _environment(run: Run) -> tuple[Environment, dict[str, JsonValue] | None]:
    """The run's environment, imported here (a published one from its version, in the runtime environment this job
    was given), and what its start records of the version."""
    from rollout_train.published import loaded, provenance

    name = run.settings["environment"]
    if not isinstance(name, str):
        raise ValueError(f"a {run.kind} run names its environment")
    found, version = await loaded(name, run.ledger)
    return found, provenance(version) if version is not None else None


async def _refused(run: Run, refusals: Sequence[Any]) -> None:
    """Record a refused run: a start that says its settings, and an end that says why."""
    fence = await run.ledger.take(scope(run.run.id))
    said = recorded(run.settings, (), run.preset)
    record: dict[str, JsonValue] = {**start_header(kind=run.kind), **run.started, "run_settings": said}
    await run.ledger.append(table(run.run.id, STARTS), str(fence.number), record, fence)
    reason = "; ".join(f"{each.key}: {each.reason}" for each in refusals)
    await run.ledger.append(table(run.run.id, ENDS), str(fence.number), {"how": RUN_FAILED, "detail": reason[:2000]},
                            fence)  # fmt: skip


def _specs(run: Run) -> tuple[Any, ...]:
    provider = run.cluster.trainers.get(str(run.settings["trainer.provider"]))
    if provider is None:
        return ()
    try:
        return settings_of(provider.runs)
    except ImportError:  # (a trainer whose package is not installed here: its settings as given)
        return ()


async def ran(run: Run) -> None:
    """Check the run's settings again, then run it as its kind says."""
    from rollout_train.published import is_published

    loaded = None
    name = run.settings["environment"]
    offered = isinstance(name, str) and (name in run.cluster.environments or is_published(name))
    if run.kind in (TRAIN, EVAL, CHECK) and offered:  # (one the cluster does not offer is refused, not imported)
        run.environment, published = await _environment(run)
        loaded = run.environment
        run.started["environment"] = str(run.settings["environment"])
        if published is not None:
            run.started["published"] = published
    _, free = ray_capacity()
    findings = await checked(run.settings, run.cluster, run.ledger, loaded=loaded, own=run.run.id, gpus_free=free)
    refusing = [each for each in findings if each.refuses]
    if refusing:
        await _refused(run, refusing)
        raise Refused(findings)
    run.started["run_settings"] = recorded(run.settings, _specs(run), run.preset)
    if run.kind == TRAIN:
        await _train(run)
    elif run.kind == EVAL:
        await _evaluate(run)
    elif run.kind == IMITATE:
        await _imitate(run)
    else:
        await _check(run)


async def _train(run: Run) -> None:
    from rollout_train.evals import Schedule, environments_of, suite_of
    from rollout_train.loop import train
    from rollout_train.settings import desired_settings_of

    environment = run.environment
    assert environment is not None
    name = str(run.settings["environment"])
    async with started(run, training=True) as live:
        assert live.trainer is not None
        provider = run.cluster.trainers[str(run.settings["trainer.provider"])]
        live.chain = live.chosen(frozenset({provider.capabilities.format}))
        wanting = desired_settings_of(run.ledger)

        async def scheduled(suite_name: str, every: int, episodes: int | None) -> Schedule | None:
            """The evals of a suite, by name (the version it points to now) or a version's id; none for a suite the
            ledger does not have, or one whose environments do not load here."""
            try:
                suite = await suite_of(run.ledger, suite_name)
                if suite is None:
                    return None
                played = environments_of(suite, {name: environment})
            except (KeyError, ValueError):
                return None
            return Schedule(suite, live.eval_run, every, episodes, played, live.binding)

        async def desired() -> Mapping[str, JsonValue]:
            found = await wanting.desired(run.run.id) if wanting is not None else None
            return found.settings if found is not None else {}

        schedule = None
        if isinstance(suite := run.settings["evals.suite"], str):
            every, episodes = run.settings["evals.every"], run.settings["evals.episodes"]
            schedule = await scheduled(suite, int(cast(int, every)), cast(int | None, episodes))
            if schedule is not None:
                schedule = dataclasses.replace(schedule, named=suite)
        run.started["blobs"] = dict(run.stores.location)
        run.started["directory"] = str(live.directory)
        async with ending(run.ledger, run.run.id):
            await train(
                environment, live.trainer, run.checkpoints, start=live.origin, channel=live.channel,
                base=str(run.settings.get(f"channels.{live.channel}.model")), directory=live.directory / "checkpoints",
                publish=live.publish, groups=int(cast(int, run.settings["groups"])),
                groups_per_step=int(cast(int, run.settings["groups_per_step"])),
                max_lag=int(cast(int, run.settings["max_lag"])), seed=int(cast(int, run.settings["seed"])),
                episodes_at_once=int(cast(int, run.settings["episodes_at_once"])), binding=live.binding(environment),
                run=run.run.id, started=run.started, hooks=[live.feed], kept=live.bookmarked, made=live.made,
                reshard=live.bridged if live.chain else None, evals=schedule, desired=desired, scheduled=scheduled,
            )  # fmt: skip


async def _evaluate(run: Run) -> None:
    from rollout_train.evals import environments_of, evaluate, suite_of
    from rollout_train.published import is_published, loaded

    named_suite = run.settings["eval.suite"]
    suite = await suite_of(run.ledger, str(named_suite)) if isinstance(named_suite, str) else None
    if suite is None:
        raise KeyError(f"there is no suite {named_suite!r}")
    imported = {each: (await loaded(each, run.ledger))[0] for each in suite.environments if is_published(each)}
    played = environments_of(suite, {**imported, **({str(run.settings["environment"]): run.environment}
                                                    if run.environment is not None else {})})  # fmt: skip
    episodes = run.settings["eval.episodes"]
    try:
        async with started(run, training=False) as live:
            subject = live.origin
            chain: tuple[Bridge, ...] = ()
            if subject is not None:
                record = await run.checkpoints.checkpoint(subject)
                files = record.weights.files if record.weights is not None else {}
                chain = live.chosen(format_of(files))

            async def reshard(checkpoint: Checkpoint, fence: Fence) -> Manifest:
                return await live._bridge(chain, checkpoint, fence)  # pyright: ignore[reportPrivateUsage]

            async def part(number: int) -> str:
                return await live.eval_run(part=number)

            run.started["blobs"] = dict(run.stores.location)
            run.started["directory"] = str(live.directory)
            run.started["environment"] = suite.environments[0]
            async with ending(run.ledger, run.run.id):
                said = await evaluate(
                    run.checkpoints, run=run.run.id, suite=suite, subject=subject,
                    base=str(run.settings.get(f"channels.{live.channel}.model")), channel=live.channel,
                    directory=live.directory / "checkpoints", publish=live.publish, environments=played,
                    binding=live.binding, parts=part, episodes=cast(int | None, episodes), started=run.started,
                    reshard=reshard if chain else None, hooks=[live.feed],
                )  # fmt: skip
    finally:  # (the files fetched to serve the checkpoint are needed only while it plays; a full one is a whole model)
        for fetched in ("bases", "checkpoints"):
            await asyncio.to_thread(shutil.rmtree, run_directory(run.cluster, run.run.id) / fetched, ignore_errors=True)
    for each in said["entries"]:
        solved = f"solved {each['solved']} of {each['played']}" if each["solved"] is not None else str(each["played"])
        print(f"{suite.id} {each['environment']}: {solved} episodes (mean reward {each['reward']})", flush=True)


def imitated(settings: RunSettings, kind: str) -> "Objective":
    """The objective an imitate run trains with: the likelihood or preference preset its settings name (as the
    dataset's `kind` holds examples, or pairs and labelled examples), else `sft` with those of their components a
    likelihood takes. Raises `ValueError` for a policy-gradient preset named, or one that does not fit the dataset."""
    from rollout_train.objectives import DEFAULT, LIKELIHOOD, POLICY_GRADIENT, PREFERENCE, PRESETS, component

    preset = str(settings["objective.preset"])
    asked = PRESETS.get(preset)
    if asked is not None and asked.objective.family == POLICY_GRADIENT:
        if preset != DEFAULT.preset:
            raise ValueError(f"an imitate run trains on a dataset's examples, which have no advantages: a likelihood "
                             f"or preference preset, not {preset}")  # fmt: skip

        def taken(key: str) -> bool:
            found = component(key.removeprefix("objective."))
            return not key.startswith("objective.") or (found is not None and LIKELIHOOD in found.families)

        values = {key: value for key, value in settings.values.items() if taken(key)}
        settings = RunSettings({**values, "objective.preset": "sft"})
    objective = objective_in(settings)
    if (objective.family == PREFERENCE) != (kind != "examples"):
        wanted = "a preference preset" if kind != "examples" else "a likelihood preset (sft)"
        raise ValueError(f"a dataset of {kind} is trained on by {wanted}, not {objective.preset}")
    return objective


async def _imitate(run: Run) -> None:
    import time

    from rollout_train.datasets import dataset_of, resolved_dataset
    from rollout_train.datasets import examples as dataset_examples
    from rollout_train.imitation import IMITATION, RATES, WARMUP, imitate, passes_for
    from rollout_train.record import PROCESS

    ledger, registry = run.ledger, registry_of(run.ledger)
    made = await dataset_of(ledger, await resolved_dataset(ledger, registry, str(run.settings["imitation.dataset"])))
    channel = run.channel
    renderer_name = run.settings.get(f"channels.{channel}.renderer")
    model = run.settings.trainer_model
    if renderer_name is None or model is None:
        raise ValueError(f"an imitate run renders its examples with channels.{channel}.renderer over its model")
    taught = await dataset_examples(ledger, made, named(str(renderer_name))(str(model)))
    if not taught.items:
        raise ValueError("the dataset has no examples")
    objective = imitated(run.settings, made.kind)
    provider = run.cluster.trainers[str(run.settings["trainer.provider"])]
    limit = run.settings["imitation.limit"]
    chosen = taught.items if not isinstance(limit, int) else taught.items[:limit]
    given = run.settings.values
    rate = given.get("trainer.learning_rate", RATES.get(provider.capabilities.produces, 1e-6))
    warmup = given.get("imitation.warmup", WARMUP)
    passes = given.get("imitation.passes") or passes_for(chosen, int(cast(int, given.get("trainer.tokens_per_step",
                                                                                         4096))))  # fmt: skip
    run.settings = RunSettings({**given, "trainer.learning_rate": rate, "trainer.warmup_updates": warmup,
                                "trainer.passes": passes})  # fmt: skip
    run.objective = objective.to_json()
    shown = f"{len(taught.segments)} segments" if taught.segments else f"{len(taught.preferences)} {made.kind}"
    print(f"{shown} of {taught.episodes} episodes ({taught.left_out} left out), {taught.supervision}; {passes} passes "
          f"at {rate:g}, warmed up over {warmup} updates", flush=True)  # fmt: skip
    async with started(run, training=True) as live:
        assert live.trainer is not None
        fence = await ledger.take(scope(run.run.id))  # (the run is stopped: imitation writes as it)
        said = recorded(run.settings, _specs(run), run.preset)
        said["objective"] = run.objective
        start: dict[str, JsonValue] = {
            **run.started, "kind": IMITATION, "from": live.origin, "dataset": made.id,
            "supervision": taught.supervision,
            "host": socket.gethostname(), "process": PROCESS, "started": round(time.time(), 1),
            "blobs": dict(run.stores.location), "directory": str(live.directory), "run_settings": said,
        }  # fmt: skip
        await ledger.append(table(run.run.id, STARTS), str(fence.number), start, fence)
        async with ending(ledger, run.run.id):
            checkpoint = await imitate(
                run.checkpoints, live.trainer, taught, fence=fence, run=run.run.id, start=live.origin,
                base=str(model), directory=live.directory / "checkpoints",
                limit=limit if isinstance(limit, int) else None, seed=int(cast(int, run.settings["seed"])),
                resume_optimizer=bool(run.settings["imitation.resume_optimizer"]),
            )  # fmt: skip
    parents = ", ".join(checkpoint.parents) or "the base model"
    metrics = {key: round(value, 4) for key, value in checkpoint.metrics.items()}
    print(f"made {checkpoint.id} (from {parents}): {json.dumps(metrics)}", flush=True)


async def _check(run: Run) -> None:
    from rollout_train.algorithm import Grpo
    from rollout_train.check import played

    environment = run.environment
    assert environment is not None
    async with started(run, training=False) as live:
        run.started["blobs"] = dict(run.stores.location)
        run.started["directory"] = str(live.directory)
        episodes = run.settings["check.episodes"]
        async with ending(run.ledger, run.run.id):
            found = await played(
                environment, run.ledger, run.stores.blobs, run=run.run.id, binding=live.binding(environment),
                groups=int(cast(int, run.settings["groups"])),
                episodes=episodes if isinstance(episodes, int) else Grpo().group_size,
                seed=int(cast(int, run.settings["seed"])), started=run.started,
            )  # fmt: skip
    for each in found:
        print(each, flush=True)
    if not all(each.passed for each in found):
        raise ValueError("the check found what does not hold together")


async def driven(launch: str, cluster: Cluster, stores: Stores | None = None) -> None:
    """Run a launch's run, noting on the launch that it runs, what it waits for, and how it ended: ended, failed (with
    why: its settings' refusals among them) or stopped (cancelled)."""
    from rollout_train.launches import ASKED, RUNNING, STOPPED, STOPPING, SUBMITTED, launch_of, launches_of

    stores = stores or Stores.open(cluster)
    launches = launches_of(stores.ledger)
    if launches is None:
        raise KeyError("this ledger keeps no launches")
    found = await launch_of(launches, launch)
    if found.run is None:
        raise ValueError(f"launch {launch} names no run")
    noted = await launches.note(launch, expect=(ASKED, SUBMITTED), state=RUNNING, detail="running")
    if noted.state != RUNNING:
        if noted.state == STOPPING:
            await launches.note(launch, expect=(STOPPING,), state=STOPPED, detail="stopped before it started")
        return
    registry = registry_of(stores.ledger)
    entry = next((each for each in await registry.runs() if each.id == found.run), None) if registry else None
    asked = found.asked
    settings = RunSettings({**asked.settings, "kind": asked.kind, "name": entry.name if entry else asked.name})

    async def said(detail: str) -> None:
        from rollout_train.launches import OPEN

        with contextlib.suppress(KeyError):
            await launches.note(launch, expect=OPEN, detail=detail)

    run = Run(
        cluster, stores, settings, entry or Entry(found.run, asked.name, 0.0), preset=asked.preset,
        resumes=asked.resumes is not None, noted=said,
    )  # fmt: skip
    run.started |= {"cluster": cluster.name, "launch": found.id}
    if found.job:
        run.started["job"] = found.job
    await _driven(launches, found.id, ran(run))


async def _driven(launches: Any, id: str, work: Coroutine[Any, Any, None]) -> None:
    """Do `work`, and note on the launch how it ended."""
    from rollout_train.launches import ENDED, FAILED, OPEN, STOPPED

    try:
        await work
    except asyncio.CancelledError:
        await asyncio.shield(launches.note(id, expect=OPEN, state=STOPPED, detail="stopped"))
        raise
    except Refused as refused:
        await launches.note(id, expect=OPEN, state=FAILED, detail=f"refused: {refused}")
        raise
    except BaseException as error:
        await launches.note(id, expect=OPEN, state=FAILED, detail=f"{type(error).__name__}: {error}"[-4000:])
        raise
    else:
        await launches.note(id, expect=OPEN, state=ENDED, detail="ended")


def main(arguments: Sequence[str] | None = None) -> None:
    """`python -m rollout_train.jobs LAUNCH`: a run's job."""
    from rollout_train.cli import until_signalled
    from rollout_train.ray_cluster import connect

    parser = argparse.ArgumentParser(prog="python -m rollout_train.jobs", description="A run's job.")
    parser.add_argument("launch", help="the launch whose run it runs")
    parser.add_argument("--cluster", help="the cluster config (by default the one handed to the job, else found)")
    given = parser.parse_args(arguments)
    cluster = located(given.cluster)
    said = cluster.ray.address
    connect(said if said != "auto" else os.environ.get("RAY_ADDRESS") or said)  # (a job: the cluster it was given)
    try:
        code = asyncio.run(until_signalled(driven(given.launch, cluster)))
    except Refused as refused:
        print(f"refused: {refused}", file=sys.stderr, flush=True)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
