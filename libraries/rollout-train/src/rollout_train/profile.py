"""A deployment, described: the channels and the engines behind them, the trainer, the runner, where tool sets and
sandbox pools live.

Whoever deploys writes this down once (a TOML file, or the dataclasses below) and opens it; whoever trains gets the
`policies`, a `trainer` and a way to `publish` checkpoints, while a runner plays the episodes the run asks for in the
ledger, and never learns what stands behind them. Scaling is a change here: more engines behind a
channel, a tool set or a pool at a URL instead of in this process.

Engines, renderers, the trainer, tool sets and sandbox providers are named as `module:name`, and what the profile says
of each is passed to it: this module knows no engine and no trainer. docs/guide/deploying.md describes the file.
"""

import asyncio
import contextlib
import dataclasses
import json
import math
import secrets
import socket
import tomllib
from collections.abc import AsyncGenerator, Callable, Collection, Coroutine, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.harness.imports import ToolBinding, ToolSet
from rollout.harness.sandboxes import MemoryLeases, Pool, PoolBinding, SandboxPool
from rollout.local import LocalRunner
from rollout.names import named
from rollout.processes import end_orphans, note_processes
from rollout_train import Checkpoint, Checkpoints, Colocated, Fence, Ledger, Manifest, Trainer
from rollout_train.bridges import bridged, by_name, on_ray
from rollout_train.following import Follower
from rollout_train.gateway import Gateway, GatewayEndpoints, Keyring, TurnStore
from rollout_train.inference import Channel, Connection, Engine, Limits, Route, Routes
from rollout_train.inference.channel import MAX_LAG
from rollout_train.layout import BLOBS, FEED, LEDGER, PROCESSES
from rollout_train.ledger import LOCATION
from rollout_train.ledger import opened as ledger_at
from rollout_train.machine import alive, measured
from rollout_train.presence import presence_of
from rollout_train.ray_cluster import connect, disconnect
from rollout_train.registry import Entry, Registry, registry_of, resolved, run_of
from rollout_train.rollouts.scheduler import EpisodeRunner
from rollout_train.sandboxes import admits, keep, leases_of
from rollout_train.stores import location, opened

__all__ = ["ChannelSpec", "EvalsSpec", "GatewaySpec", "NotEnoughMemory", "Platform", "Profile", "TrainerSpec"]


@dataclass(frozen=True)
class ChannelSpec:
    model: str
    """The checkpoint every engine of the channel serves."""
    renderer: str
    """`module:name` of the model family's renderer, called with `model`."""
    engine: str
    """`module:name` of what makes an engine, called with `model` and one entry of `engines`."""
    engines: tuple[Mapping[str, Any], ...] = ({},)
    """One entry per replica: what that engine is told (its share of a GPU, which device, where it listens)."""
    thinking_tokens: int | None = None
    """Tokens of thinking per turn before it is closed by force (`Limits.thinking`); none: no thinking budget."""
    answer_tokens: int | None = None
    """Room for the answer after the thinking (`Limits.answer`); none: whatever room the turn has left. With neither,
    a turn may fill what the context leaves."""
    reshard: str | None = None
    """The bridge, by name (`rollout_train.bridges.BRIDGES`: `verbatim`, `peft-from-tinker`, …), that makes the files
    the engines load from a checkpoint's; none: the trainer's files as they are, with no bridge."""
    max_lag: int = MAX_LAG
    """For a channel whose engines serve elsewhere (`engine` is `RemoteEngine`, each entry of `engines` a server's
    `address`): how many checkpoints behind what the channel should serve a sample may be, where its server does not
    have the newest yet."""
    via: str | None = None
    """For a channel whose engines serve elsewhere: the URL its runners send every request to (a router or a proxy in
    front of its servers); none: its servers' addresses. Its engine hosts load checkpoints at the addresses."""
    connection: Mapping[str, str] = field(default_factory=dict[str, str])
    """How servers elsewhere are reached (`Connection`): `token_env` or `token_file`, `ca`, `certificate`, `key`."""

    def __post_init__(self) -> None:
        for key in ("thinking_tokens", "answer_tokens"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} is a whole number, 1 at least, or none for no budget (not {value!r})")
        if self.reshard is not None:
            by_name(self.reshard)

    @property
    def routed(self) -> bool:
        """Whether its engines serve elsewhere (said by name: an engine's module is not imported to load a profile)."""
        return self.engine == REMOTE

    def route(self, renderer: Any, sequence: int | None = None) -> Route:
        """How a runner samples it on its servers elsewhere; `sequence`, the trainer's longest turn, where this process
        trains it."""
        servers = (self.via,) if self.via else tuple(str(each["address"]) for each in self.engines)
        return Route(
            renderer,
            self.model,
            servers,
            Limits(self.thinking_tokens, self.answer_tokens, sequence=sequence),
            max_lag=self.max_lag,
            connection=Connection(**self.connection),
        )


REMOTE = "rollout_train.inference:RemoteEngine"
"""What a channel whose engines serve elsewhere names as its `engine`."""


@dataclass(frozen=True)
class TrainerSpec:
    kind: str
    """`module:name` of what makes the trainer, called with the channel's model and `settings`."""
    channel: str
    """The channel that serves the policy it trains."""
    start: str | None = None
    """The checkpoint a new run trains from (`rollout_train.registry.resolved`: a bookmark, `RUN:STEP`, `RUN`, or a
    checkpoint's id or the start of one); by default the base model. A run started again goes on from the newest
    checkpoint it made."""
    bookmark: str | None = None
    """A bookmark the run carries: moved to each checkpoint it makes."""
    colocated: bool = False
    """Whether it shares the channels' accelerator: their engines then sleep while it steps."""
    settings: Mapping[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class EvalsSpec:
    """Evals a training run makes of its checkpoints as it makes them (`rollout_train.evals.Schedule`)."""

    suite: str
    """The suite each plays: by name, the version its name points to when each step is decided (the environment's eval
    data of that name, frozen on first use, or a suite made by hand: `rollout suite make`); or one version, by id
    (`NAME@N`)."""
    every: int = 1
    """The checkpoint of every `every`th step is evaluated."""
    episodes: int | None = None
    """Episodes of each of the suite's starts; none: the suite's own."""

    def __post_init__(self) -> None:
        if self.every < 1 or (self.episodes is not None and self.episodes < 1):
            raise ValueError("evals: `every` and `episodes` are 1 at least")


@dataclass(frozen=True)
class GatewaySpec:
    """The gateway (`rollout_train.gateway`): a stateless service that samples the channels and records every turn,
    which `rollout gateway PROFILE` serves, as many replicas as wanted. A run's runner records through it: through the
    replicas at `url`, or, with none, through a gateway in its own process (served to harnesses at the profile's
    `serve`)."""

    url: str | None = None
    """Where programs and harnesses reach it (its base URL, as a proxy in front of it presents it); none: a runner
    samples through a gateway in its own process. A runner that records through replicas elsewhere starts no engine: a
    routed channel is sampled on its servers, and any other is hosted by the replicas (its engines in their processes),
    which say what it guarantees. A hosted channel samples what its engines serve: the base model, never trained."""
    listen: str = "127.0.0.1:8830"
    """`host:port` a replica serves on."""
    keys: str | None = None
    """A file of the secrets keys are signed with (`rollout_train.gateway.keys`); by default the environment's, and for
    a gateway in a runner's own process with none there, a secret it makes when it starts."""
    lifetime: float = 6 * 3600.0
    """Seconds a key minted for a slot is good for."""


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


@dataclass(frozen=True)
class Profile:
    directory: Path
    """The run's state: checkpoints' files while in use, the monitor's feed, and (unless the profile names other places)
    its ledger and blobs."""
    channels: Mapping[str, ChannelSpec]
    trainer: TrainerSpec | None = None
    serve: str | None = None
    """`host:port` to serve the gateway in the runner's own process on, for harnesses (none: a harness cannot be given
    an address)."""
    address: str | None = None
    """The URL others reach `serve` at (by default `http://` and `serve`)."""
    tools: Mapping[str, str] = field(default_factory=dict[str, str])
    """Each tool set by name: a URL, or `module:name` of what makes it, called with `directory`."""
    pools: Mapping[str, str | Mapping[str, Any]] = field(default_factory=dict[str, str | Mapping[str, Any]])
    """Each sandbox pool by the kind of sandbox it serves: a URL, or `module:name` of the provider that makes them,
    called with `directory`; or a table whose `kind` is that and whose other entries are passed to it too."""
    ledger: Mapping[str, Any] = field(default_factory=dict[str, Any])
    """Where the run's tables and the checkpoints are kept (`rollout_train.ledger.opened`): `{"directory":
    …}`, in files; `{"kind": "module:name", …}`, what that makes from the other entries, such as a database
    (`rollout_train.database:DatabaseLedger` with a `url`). By default files under `directory/ledger`. Runs that share a
    ledger see each other's checkpoints."""
    blobs: Mapping[str, Any] = field(default_factory=dict[str, Any])
    """Where episodes (and what programs store) are kept: `kind` is `module:name` of what makes the store, called
    with the other entries. Without one, files under `directory/blobs`."""
    runs_gib: float = 0.0
    """System memory that must be available to admit runs."""
    training_gib: float = 0.0
    """And to start a step of a colocated trainer."""
    episodes_at_once: int = 6
    """The most episodes a run plays at once (whatever groups they are of): what the machine's engines and its memory
    for the programs' worlds can take."""
    feed_runs: int | None = None
    """Episodes kept in the monitor's feed, where it should not keep `RunFeed`'s own number (the oldest are
    deleted)."""
    ray: str | None = None
    """The Ray cluster to connect to (`auto`, or `ray://host:port`): reshards then run as Ray tasks on it."""
    name: str | None = None
    """What a run first started in `directory` is called (by default the directory's name). It is named again with
    `rollout rename`; its id, in the directory's `run.json`, never changes."""
    evals: EvalsSpec | None = None
    """The evals a training run makes of its checkpoints as it makes them."""
    gateway: GatewaySpec | None = None
    """The gateway that samples the channels and records turns: what `rollout gateway PROFILE` serves, and what a
    run's runner records through (by default a gateway in its own process)."""

    @classmethod
    def load(cls, path: Path, *, directory: Path | None = None, settings: Mapping[str, Any] | None = None) -> "Profile":
        """The profile a TOML file describes; `directory` replaces the file's (one profile, many runs), and
        `settings` replace or add its keys, by dotted name (`trainer.learning_rate`, `episodes_at_once`). A key the
        file has and a profile does not is an error: a misspelt guard would otherwise be no guard. A channel's
        `thinking_tokens` or `answer_tokens` of `"none"` is no budget (TOML has no null: what a setting that removes
        the file's budget says)."""
        described = tomllib.loads(path.read_text())
        for dotted, value in (settings or {}).items():
            *tables, key = dotted.split(".")
            place: dict[str, Any] = described
            for name in tables:
                place = place.setdefault(name, {})
            place[key] = value
        channels: dict[str, ChannelSpec] = {}
        for name, channel in _table(described, "channels").items():
            known = (
                "model", "renderer", "engine", "engines", "thinking_tokens", "answer_tokens", "reshard", "max_lag",
                "via", "connection",
            )  # fmt: skip
            given = _only(dict(channel), f"channels.{name}", *known)
            for budget in ("thinking_tokens", "answer_tokens"):  # (`none`: no budget, where a setting unsets one)
                if given.get(budget) == "none":
                    given[budget] = None
            engines = tuple(given.pop("engines", [{}]))
            channels[name] = ChannelSpec(**given, engines=engines)
            if channels[name].routed and not all("address" in each for each in engines):
                raise ValueError(f"channels.{name}: each entry of engines is a server's address")
        trainer = _table(described, "trainer")
        memory = _only(_table(described, "memory"), "memory", "runs_gib", "training_gib")
        blobs = _table(described, "blobs")
        evals = _only(_table(described, "evals"), "evals", "suite", "every", "episodes")
        gateway = _only(
            _table(described, "gateway"),
            "gateway",
            *(each.name for each in dataclasses.fields(GatewaySpec)),
        )
        known = (
            "directory", "ledger", "serve", "address", "tools", "pools", "feed_runs", "episodes_at_once", "ray",
        )  # fmt: skip
        top = _only(described, "the profile", *known)
        top["directory"] = directory or Path(top["directory"]).expanduser()
        if isinstance(top.get("ledger"), str):  # (`ledger = "path"`: a directory of files)
            top["ledger"] = {"directory": str(Path(top["ledger"]).expanduser())}
        return cls(
            **top,
            **memory,
            blobs=blobs,
            channels=channels,
            evals=EvalsSpec(**evals) if evals.get("suite") else None,  # (`suite = ""`: no evals)
            gateway=GatewaySpec(**gateway) if gateway else None,
            trainer=TrainerSpec(
                kind=trainer.pop("kind"),
                channel=trainer.pop("channel"),
                start=trainer.pop("start", None),
                bookmark=trainer.pop("bookmark", None),
                colocated=trainer.pop("colocated", False),
                settings=trainer,
            )
            if trainer
            else None,
        )

    @property
    def hosted(self) -> list[str]:
        """The channels the gateway at `[gateway] url` hosts, by name: with a URL, every channel not routed (a runner
        starts none of their engines, and samples them there); without one, none."""
        if self.gateway is None or self.gateway.url is None:
            return []
        return [name for name, channel in self.channels.items() if not channel.routed]

    @contextlib.asynccontextmanager
    async def open(self, *, training: bool = True, plays: Collection[str] | None = None) -> AsyncGenerator["Platform"]:
        """Start what the profile describes, and stop it on the way out (also if starting fails half way). Without
        `training` (an eval), no trainer is made: the trained channel's engines still load what the trainer's `start`
        is served over. With `plays`, a runner and nothing else (`rollout runner`): see `Platform.start`."""
        async with contextlib.AsyncExitStack() as stack:
            yield await Platform.start(self, stack, training=training, plays=plays)

    @contextlib.asynccontextmanager
    async def engines(self) -> AsyncGenerator[dict[str, Channel]]:
        """The channels whose engines are servers elsewhere, as clients of the servers at their addresses, and nothing
        else: what an engine host (`rollout engines`) loads checkpoints into. Closed on the way out."""
        async with contextlib.AsyncExitStack() as stack:
            self.directory.mkdir(parents=True, exist_ok=True)
            models = {name: spec.model for name, spec in self.channels.items()}
            yield started_engines(self, stack, models, servers=True)


def _table(described: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(described.pop(name, {}))


def _only(table: dict[str, Any], where: str, *known: str) -> dict[str, Any]:
    if unknown := sorted(set(table) - set(known)):
        raise ValueError(f"{where} has no {', '.join(unknown)} (it has {', '.join(known)})")
    return table


class Platform:
    """An open profile: its `run`, the checkpoint it trains from (`origin`), the `checkpoints`, a `trainer` to step,
    `publish` to serve a checkpoint, and a `runner` that plays the episodes its run asks for
    (`rollout_train.rollouts.scheduler.EpisodeRunner`)."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.location: dict[str, Any] = dict(profile.ledger) or {"directory": str(profile.directory / LEDGER)}
        """Where the ledger is; written into the run's directory, for whatever reads the run."""
        self.ledger: Ledger = ledger_at(self.location)
        self.checkpoints: Checkpoints
        """The checkpoints, in the ledger and the blob store."""
        self.registry: Registry | None = registry_of(self.ledger)
        """What the runs are called, and the bookmarks."""
        self.run: Entry
        """The run in the profile's directory (none for a runner and nothing else: `plays`)."""
        self.plays: Collection[str] | None = None
        """For a runner and nothing else, the runs it plays (none named: every run whose channels it reaches)."""
        self.origin: str | None = None
        """The checkpoint a new run trains from, by id (None: the base model)."""
        self.channels: dict[str, Channel] = {}
        """The channels whose engines serve in this process."""
        self.routes: Routes | None = None
        """The channels whose engines serve elsewhere, each run's sampled from the checkpoints it says they serve."""
        self.gateway: Gateway | None = None
        """The gateway in this process, where the runner records through one here (no `[gateway] url`)."""
        self.recorder: GatewayEndpoints
        """What the runner's recorded slots sample through: the gateway here, or the one at `[gateway] url`."""
        self.runner: EpisodeRunner
        self.feed: Any
        """The monitor's feed (`RunFeed`): the loop's results and steps go there too."""
        self.trainer: Trainer | None = None
        self.tool_bindings: dict[str, ToolBinding] = {}
        """Where a run finds each tool set the profile names (for a run's binding)."""
        self.pool_bindings: dict[str, PoolBinding] = {}
        """Where a run acquires each kind of sandbox the profile names a pool for (for a run's binding)."""
        self.blobs: Blobs
        """Where episodes' trajectories and events, and checkpoints' files, are kept."""
        self.blobs_at: dict[str, Any] = {}
        """Where that is, as any process opens it (`rollout_train.stores`), for the run's `starts` record."""
        self._runs: set[str] = set()
        """The runs whose episodes the runner plays: the profile's, and its evals'."""

    @classmethod
    async def start(
        cls,
        profile: Profile,
        stack: contextlib.AsyncExitStack,
        *,
        training: bool = True,
        plays: Collection[str] | None = None,
    ) -> "Platform":
        """Start everything (the trainer only with `training`), registering with `stack` how each thing is stopped
        (the engines last). With `plays`, a runner and nothing else (`rollout runner`): no run is registered in the
        directory and no trainer is made; the runner plays those runs (by id), or with none named every run whose
        channels it reaches, and the channels whose engines serve in this process follow what the one run named says
        they should serve (`rollout_train.following`). Without, every channel of this process but the trained one
        follows what the run says it serves: nothing, unless its start says the channel follows another or is fixed on
        a checkpoint (`rollout_train.serving.source_of`)."""
        from rollout_train.monitor import RunFeed

        self = cls(profile)
        self.plays = plays
        directory = profile.directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / LOCATION).write_text(json.dumps(self.location))
        if profile.ray:  # (reshards run as Ray tasks on that cluster)
            await asyncio.to_thread(connect, profile.ray)
            stack.callback(disconnect)
        if plays is None:
            self.run = await run_of(directory, self.ledger, self.registry, profile.name)
            self._runs.add(self.run.id)
        if plays is None and profile.trainer is not None and profile.trainer.start is not None:
            self.origin = await resolved(self.ledger, self.registry, profile.trainer.start)
        self.blobs = opened(dict(profile.blobs)) if profile.blobs else FileBlobStore(directory / BLOBS)
        self.blobs_at = location(profile.blobs, directory / BLOBS)
        self.checkpoints = Checkpoints(self.ledger, self.blobs)
        models = {name: spec.model for name, spec in profile.channels.items()}
        """What each channel's engines load: the profile's model; for the trained channel of a run that starts from a
        full checkpoint, or an adapter over one, that full checkpoint's files (the model its adapters, or its full
        weights, are trained over)."""
        hosted = profile.hosted
        if profile.trainer is not None and profile.trainer.channel in hosted and training and plays is None:
            raise ValueError(
                f"channels.{profile.trainer.channel}: a channel the gateway at [gateway] url hosts samples what its "
                "own engines serve, so it cannot be trained"
            )
        trained_here = profile.trainer is not None and not profile.channels[profile.trainer.channel].routed
        if self.origin is not None and profile.trainer is not None and trained_here:
            under = await self.checkpoints.under(await self.checkpoints.checkpoint(self.origin))
            if under is not None and under.weights is not None:
                fetched = await self.checkpoints.files(under.weights, directory / "bases" / under.id)
                models[profile.trainer.channel] = str(fetched)
        record = directory / PROCESSES
        end_orphans(record)  # an engine a killed process left behind holds its accelerator
        described = profile.trainer
        learner: Trainer | None = None
        if described is not None and training and plays is None:
            learner = named(described.kind)(models[described.channel], **described.settings)
        sequence = learner.budget.segment_tokens if learner is not None else None
        self.channels = started_engines(profile, stack, models, sequence, leaving=hosted)
        trained = described.channel if described is not None else None
        routes = {
            name: spec.route(named(spec.renderer)(spec.model), sequence if name == trained else None)
            for name, spec in profile.channels.items()
            if spec.routed
        }
        if routes:
            self.routes = Routes(routes, self.ledger)
            stack.callback(self.routes.close)
        feed = RunFeed(directory / FEED, **({"keep": profile.feed_runs} if profile.feed_runs else {}))
        stack.callback(feed.close)
        self.recorder = self._recording(feed)
        if hosted:  # (what each guarantees, as the gateway that hosts it says)
            await self.recorder.hosted(hosted)
        tool_sets: dict[str, ToolSet] = {}
        for name, where in profile.tools.items():
            if where.startswith(("http://", "https://")):
                self.tool_bindings[name] = ToolBinding(url=where)
                continue
            tool_sets[name] = named(where)(directory)
            self.tool_bindings[name] = ToolBinding(local=name)
            stack.push_async_callback(_closed, tool_sets[name])
        pools = await self._pools(stack)
        runner = LocalRunner(gateway=self.recorder, tool_sets=tool_sets, pools=pools, hooks=[feed], blobs=self.blobs)
        self.feed = feed
        self.runner = EpisodeRunner(
            f"{socket.gethostname()}/{directory.name}",  # (the same name when started again: what it claimed is free)
            self.ledger,
            runner,
            self.recorder,
            self.blobs,
            places=profile.episodes_at_once,
            imports=list(tool_sets),
            pools=pools,
            runs=self._runs if plays is None else (set(plays) or None),
            hooks=[feed],
            guard=_needs(profile.runs_gib, "to run more episodes"),
            presence=presence_of(self.ledger),
            about=lambda: self._about(record),
        )
        await self.runner.prepare()
        await runner.launch()
        stack.push_async_callback(runner.close)
        _background(stack, self.runner.serve())
        if plays is not None and len(plays) == 1 and self.channels:  # (its own engines serve what that run says)
            (followed,) = plays
            name = f"{socket.gethostname()}/{directory.name}"
            follower = Follower(name, self.checkpoints, followed, self.channels, directory / "checkpoints")
            _background(stack, follower.serve())
        others = {name: channel for name, channel in self.channels.items() if name != trained}
        if plays is None and others:  # (a channel the run's start says follows another, or is pinned, loads it here)
            name = f"{socket.gethostname()}/{directory.name}/channels"
            follower = Follower(name, self.checkpoints, self.run.id, others, directory / "following")
            _background(stack, follower.serve())
        self.trainer = learner
        if learner is not None and described is not None and described.colocated:
            ready = _needs(profile.training_gib, "to train")
            self.trainer = Colocated(learner, list(self.channels.values()), guard=ready)
        if profile.serve and self.gateway is not None:
            _background(stack, self._serve(profile.serve, self.gateway))
        return self

    def _recording(self, feed: Any) -> GatewayEndpoints:
        """What the runner records through: the gateway at `[gateway] url`, or one in this process over its channels
        and routes, which tells the feed of the samples harnesses ask for (served to them at `serve`)."""
        profile = self.profile
        spec = profile.gateway or GatewaySpec()
        if spec.keys:
            keyring = Keyring.load(Path(spec.keys).expanduser())
        else:
            try:
                keyring = Keyring.from_environment()
            except KeyError:  # (keys only this process mints and takes: a secret of its own)
                keyring = Keyring.parse([("local", secrets.token_urlsafe(32))])
        store = TurnStore(self.ledger, self.blobs)
        if spec.url is not None:
            return GatewayEndpoints(spec.url, keyring, store, routes=self.routes, lifetime=spec.lifetime)
        models = {name: channel.model for name, channel in profile.channels.items()}
        self.gateway = Gateway(store, keyring, self.channels, self.routes, models, hooks=[feed])
        address = profile.address or (f"http://{profile.serve}" if profile.serve else None)
        return GatewayEndpoints.of(self.gateway, address, lifetime=spec.lifetime)

    async def _pools(self, stack: contextlib.AsyncExitStack) -> dict[str, Pool]:
        """The pools the profile names that live in this process, each with its keeper; the rest are bound by URL.
        A pool's leases are kept beside the ledger, under a name of this machine and run; it refuses a key whose claim
        has lapsed."""
        pools: dict[str, Pool] = {}
        directory = self.profile.directory
        for kind, where in self.profile.pools.items():
            if isinstance(where, str) and where.startswith(("http://", "https://")):
                self.pool_bindings[kind] = PoolBinding(url=where)
                continue
            options = {"kind": where} if isinstance(where, str) else dict(where)
            provider = named(str(options.pop("kind")))(directory, **options)
            name = f"{kind}@{socket.gethostname()}/{directory.name}"
            beats = presence_of(self.ledger)
            pool = SandboxPool(
                provider, name=name, leases=leases_of(self.ledger) or MemoryLeases(), admits=admits(self.ledger, beats)
            )
            stack.push_async_callback(pool.close)  # (after the runner: its runs release theirs first)
            _background(stack, keep(pool, self.ledger, presence_of(self.ledger)))
            pools[kind] = pool
            self.pool_bindings[kind] = PoolBinding(local=kind)
        return pools

    @property
    def layout(self) -> str | None:
        """The bridge, by name, that makes the files the trained channel's engines load, if one does."""
        trainer = self.profile.trainer
        return self.profile.channels[trainer.channel].reshard if trainer is not None else None

    async def reshard(self, checkpoint: Checkpoint, fence: Fence) -> Manifest:
        """The files the trained channel's engines load for a checkpoint, made by its bridge: as a Ray task when the
        profile names a Ray cluster, else here."""
        assert self.layout is not None
        chain = (by_name(self.layout),)
        if self.profile.ray:
            return await on_ray(self.location, self.blobs_at, fence, checkpoint.id, chain)
        return await bridged(self.checkpoints, fence, checkpoint.id, chain, self.profile.directory / "resharding")

    async def eval_run(self, step: int | None = None, part: int | None = None) -> str:
        """The run of an eval, by id, which the runner plays: with `step`, the eval of the checkpoint this run made at
        that step (`rollout_train.evals.Schedule`), registered the first time as `NAME-eval-STEP` and kept in
        `directory/evals`; else this run (an eval itself). With `part`, the run that plays that entry (by its number
        from 1) of the eval of a suite of several: that eval's name and `-PART`, kept beside it."""
        if step is None and part is None:
            return self.run.id
        where, name = self.profile.directory, self.run.name
        if step is not None:
            where, name = where / "evals" / f"{self.run.id}-eval-{step}", f"{name}-eval-{step}"
        if part is not None:
            where, name = where / "parts" / str(part), f"{name}-{part}"
        entry = await run_of(where, self.ledger, self.registry, name)
        self._runs.add(entry.id)
        return entry.id

    async def bookmarked(self) -> set[str]:
        """The checkpoints bookmarks name (which keep their files)."""
        return {mark.checkpoint for mark in await self.registry.bookmarks()} if self.registry else set()

    async def made(self, checkpoint: Checkpoint) -> None:
        """Carry the profile's bookmark, if it names one, to a checkpoint the run made."""
        if self.registry is not None and self.profile.trainer is not None and self.profile.trainer.bookmark:
            await self.registry.bookmark(self.profile.trainer.bookmark, checkpoint.id)

    async def publish(
        self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
    ) -> int:
        """Serve new weights on a channel from now on (with `full`, a full checkpoint's); returns the number its
        samples are stamped with (a checkpoint's depth). The runner beats at once, saying what the channel serves. A
        channel whose engines serve elsewhere is served there: they follow what the training loop wrote down that it
        serves (`rollout_train.serving`), and this returns the version given."""
        if channel not in self.channels and self.routes is not None and self.routes.routed(channel):
            return version or 0
        if channel in self.profile.hosted:
            raise ValueError(
                f"the channel {channel!r} is hosted by the gateway at [gateway] url, which samples what its own "
                "engines serve: it plays the base model only"
            )
        served = await self.channels[channel].publish(adapter, path, version, full=full)
        with contextlib.suppress(Exception):  # (a beat missed is said at the next)
            await self.runner.beat()
        return served

    def _about(self, record: Path) -> dict[str, JsonValue]:
        """What the runner says of this machine in each beat: its host, the run, its measurements, the engines'
        processes, what each channel serves and how fast since the beat before, and each run's channel whose engines
        are elsewhere (`RUN/NAME`), with what each of its servers would sample from."""
        said = machine_of(self.profile.directory, record)
        channels: list[JsonValue] = [
            {"channel": name, "adapter": channel.serving, "version": channel.version, **channel.take()}
            for name, channel in self.channels.items()
        ]
        routed = self.routes.channels() if self.routes is not None else {}
        for name, channel in routed.items():
            servers: list[JsonValue] = list(channel.servers())
            channels.append({"channel": name, **channel.take(), "servers": servers})
        return {**said, **({"run": self.run.id} if self.plays is None else {}), "channels": channels}

    async def _serve(self, address: str, gateway: Gateway) -> None:
        """The gateway in this process, over HTTP, for harnesses."""
        import uvicorn

        from rollout_train.gateway import create_app

        host, _, port = address.rpartition(":")
        config = uvicorn.Config(create_app(gateway), host=host, port=int(port), log_level="warning")
        await uvicorn.Server(config).serve()


def started_engines(
    profile: Profile,
    stack: contextlib.AsyncExitStack,
    models: Mapping[str, str],
    sequence: int | None = None,
    *,
    servers: bool = False,
    leaving: Collection[str] = (),
) -> dict[str, Channel]:
    """The channels whose engines serve in this process, by name, each engine started from `models[name]` and closed by
    `stack`, their processes noted in the directory (`PROCESSES`) for whoever must end them if this process is killed;
    with `servers`, those whose engines are servers elsewhere instead, as clients of them. `sequence` (the trainer's
    longest segment) is the longest turn of the channel the profile's trainer trains. The channels named in `leaving`
    (those a gateway elsewhere hosts) are left out."""
    record = profile.directory / PROCESSES
    trained = profile.trainer.channel if profile.trainer is not None else None
    started: list[Engine] = []
    channels: dict[str, Channel] = {}
    for name, spec in profile.channels.items():
        if spec.routed != servers or name in leaving:
            continue
        reached = {"connection": Connection(**spec.connection)} if spec.routed else {}
        engines: list[Engine] = []
        for options in spec.engines:
            engine: Engine = named(spec.engine)(models[name], **options, **reached)
            stack.callback(engine.close)
            engines.append(engine)
            started.append(engine)
            note_processes(record, [pid for each in started for pid in each.processes])
        channels[name] = Channel(
            name,
            engines,
            named(spec.renderer)(models[name]),
            Limits(spec.thinking_tokens, spec.answer_tokens, sequence=sequence if name == trained else None),
        )
    return channels


def machine_of(directory: Path, record: Path) -> dict[str, JsonValue]:
    """What a process says of its machine in each beat: its host, its directory, the machine's measurements, and the
    engines' processes noted in `record` and whether each is alive."""
    processes: Any = None
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        noted = json.loads(record.read_text())
        started = [
            {"pid": int(pid), "name": str(name), "alive": alive(int(pid))} for pid, name in noted["processes"].items()
        ]
        processes = {"owner": int(noted["owner"]), "started": started}
    return {
        "host": socket.gethostname(),
        "directory": str(directory),
        "machine": measured(directory),
        "processes": processes,
    }


def _needs(gib: float, purpose: str) -> Callable[[], None] | None:
    return (lambda: require_memory(gib, purpose)) if gib else None


async def _closed(thing: Any) -> None:
    """Close something whose `close` may be a coroutine or not."""
    closing = thing.close()
    if asyncio.iscoroutine(closing):
        await closing


def _background(stack: contextlib.AsyncExitStack, work: Coroutine[Any, Any, None]) -> None:
    """Run `work` until the stack is closed."""
    task = asyncio.ensure_future(work)

    async def stop() -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    stack.push_async_callback(stop)
