"""A deployment, described: the channels and the engines behind them, the trainer, the runner, where tool sets and
sandbox pools live.

Whoever deploys writes this down once (a TOML file, or the dataclasses below) and opens it; whoever trains gets the
`policies`, a `trainer` and a way to `publish` checkpoints, while a runner plays the episodes the run asks for in the
ledger, and never learns what stands behind them. Scaling is a change here: more engines behind a
channel, a durable runner instead of an in-process one, a tool set or a pool at a URL instead of in this process.

Engines, renderers, the trainer, tool sets and sandbox providers are named as `module:name`, and what the profile says
of each is passed to it: this module knows no engine and no trainer. docs/guide/deploying.md describes the file.
"""

import asyncio
import contextlib
import json
import math
import socket
import tomllib
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.harness.imports import ToolBinding, ToolSet
from rollout.harness.runner import Runner
from rollout.harness.sandboxes import MemoryLeases, Pool, PoolBinding, SandboxPool
from rollout.names import named
from rollout.processes import end_orphans, note_processes
from rollout_train import Checkpoint, Checkpoints, Colocated, Fence, Ledger, Manifest, Trainer
from rollout_train.inference import Channel, Engine, Limits
from rollout_train.layout import BLOBS, FEED, LEDGER, PROCESSES
from rollout_train.ledger import LOCATION
from rollout_train.ledger import opened as ledger_at
from rollout_train.machine import alive, measured
from rollout_train.presence import presence_of
from rollout_train.recorder import Recorder
from rollout_train.recorder.recorder import SERVED_UNDER
from rollout_train.registry import Entry, Registry, registry_of, resolved, run_of
from rollout_train.resharding import connect, disconnect, on_ray, reshard
from rollout_train.rollouts.scheduler import EpisodeRunner
from rollout_train.sandboxes import admits, keep, leases_of
from rollout_train.stores import location, opened

__all__ = ["ChannelSpec", "EvalsSpec", "NotEnoughMemory", "Platform", "Profile", "TrainerSpec"]


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
    """Tokens of thinking per turn, and of answer after it, where the channel should not use `Limits`' own."""
    answer_tokens: int | None = None
    reshard: str | None = None
    """`module:name` of the layout the engines load a checkpoint's files in (`rollout_train.resharding`); none: the
    trainer's files as they are, with no reshard."""


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
    """The suite each plays, by name: the environment's eval data of that name (frozen on first use), or a suite made
    by hand (`rollout suite make`)."""
    every: int = 1
    """The checkpoint of every `every`th step is evaluated."""
    episodes: int = 1
    """Episodes of each of the suite's starts."""

    def __post_init__(self) -> None:
        if self.every < 1 or self.episodes < 1:
            raise ValueError("evals: `every` and `episodes` are 1 at least")


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
    runner: str = "local"
    """`local` runs episodes in this process; `durable` records them so that they survive it."""
    serve: str | None = None
    """`host:port` to serve the model endpoint for harnesses on."""
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

    @classmethod
    def load(cls, path: Path, *, directory: Path | None = None, settings: Mapping[str, Any] | None = None) -> "Profile":
        """The profile a TOML file describes; `directory` replaces the file's (one profile, many runs), and
        `settings` replace or add its keys, by dotted name (`trainer.learning_rate`, `episodes_at_once`). A key the
        file has and a profile does not is an error: a misspelt guard would otherwise be no guard."""
        described = tomllib.loads(path.read_text())
        for dotted, value in (settings or {}).items():
            *tables, key = dotted.split(".")
            place: dict[str, Any] = described
            for name in tables:
                place = place.setdefault(name, {})
            place[key] = value
        channels: dict[str, ChannelSpec] = {}
        for name, channel in _table(described, "channels").items():
            known = ("model", "renderer", "engine", "engines", "thinking_tokens", "answer_tokens", "reshard")
            given = _only(dict(channel), f"channels.{name}", *known)
            engines = tuple(given.pop("engines", [{}]))
            channels[name] = ChannelSpec(**given, engines=engines)
        trainer = _table(described, "trainer")
        memory = _only(_table(described, "memory"), "memory", "runs_gib", "training_gib")
        blobs = _table(described, "blobs")
        evals = _only(_table(described, "evals"), "evals", "suite", "every", "episodes")
        known = (
            "directory", "ledger", "runner", "serve", "address", "tools", "pools", "feed_runs", "episodes_at_once",
            "ray",
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
            evals=EvalsSpec(**evals) if evals else None,
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

    @contextlib.asynccontextmanager
    async def open(self, *, training: bool = True) -> AsyncGenerator["Platform"]:
        """Start what the profile describes, and stop it on the way out (also if starting fails half way). Without
        `training` (an eval), no trainer is made: the trained channel's engines still load what the trainer's `start`
        is served over."""
        async with contextlib.AsyncExitStack() as stack:
            yield await Platform.start(self, stack, training=training)


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
        """The run in the profile's directory."""
        self.origin: str | None = None
        """The checkpoint a new run trains from, by id (None: the base model)."""
        self.channels: dict[str, Channel] = {}
        self.recorder: Recorder
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
    async def start(cls, profile: Profile, stack: contextlib.AsyncExitStack, *, training: bool = True) -> "Platform":
        """Start everything (the trainer only with `training`), registering with `stack` how each thing is stopped
        (the engines last)."""
        from rollout_train.monitor import RunFeed

        self = cls(profile)
        directory = profile.directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / LOCATION).write_text(json.dumps(self.location))
        if profile.ray:  # (reshards run as Ray tasks on that cluster)
            await asyncio.to_thread(connect, profile.ray)
            stack.callback(disconnect)
        self.run = await run_of(directory, self.ledger, self.registry, profile.name)
        self._runs.add(self.run.id)
        if profile.trainer is not None and profile.trainer.start is not None:
            self.origin = await resolved(self.ledger, self.registry, profile.trainer.start)
        self.blobs = opened(dict(profile.blobs)) if profile.blobs else FileBlobStore(directory / BLOBS)
        self.blobs_at = location(profile.blobs, directory / BLOBS)
        self.checkpoints = Checkpoints(self.ledger, self.blobs)
        models = {name: spec.model for name, spec in profile.channels.items()}
        """What each channel's engines load: the profile's model; for the trained channel of a run that starts from a
        full checkpoint, or an adapter over one, that full checkpoint's files (the model its adapters, or its full
        weights, are trained over)."""
        if self.origin is not None and profile.trainer is not None:
            under = await self.checkpoints.under(await self.checkpoints.checkpoint(self.origin))
            if under is not None and under.weights is not None:
                fetched = await self.checkpoints.files(under.weights, directory / "bases" / under.id)
                models[profile.trainer.channel] = str(fetched)
        record = directory / PROCESSES
        end_orphans(record)  # an engine a killed process left behind holds its accelerator
        described = profile.trainer
        learner: Trainer | None = None
        if described is not None and training:
            learner = named(described.kind)(models[described.channel], **described.settings)
        started: list[Engine] = []
        for name, spec in profile.channels.items():
            engines: list[Engine] = []
            for options in spec.engines:
                engine: Engine = named(spec.engine)(models[name], **options)
                stack.callback(engine.close)
                engines.append(engine)
                started.append(engine)
                note_processes(record, [pid for each in started for pid in each.processes])
            trained = learner if described is not None and described.channel == name else None
            limits = {"thinking": spec.thinking_tokens, "answer": spec.answer_tokens}
            self.channels[name] = Channel(
                name,
                engines,
                named(spec.renderer)(models[name]),
                Limits(
                    **{key: value for key, value in limits.items() if value is not None},
                    sequence=trained.budget.segment_tokens if trained is not None else None,
                ),
            )
        address = profile.address or (f"http://{profile.serve}" if profile.serve else None)
        self.recorder = Recorder(self.channels, base_url=f"{address}{SERVED_UNDER}" if address else None)
        feed = RunFeed(directory / FEED, **({"keep": profile.feed_runs} if profile.feed_runs else {}))
        stack.callback(feed.close)
        tool_sets: dict[str, ToolSet] = {}
        for name, where in profile.tools.items():
            if where.startswith(("http://", "https://")):
                self.tool_bindings[name] = ToolBinding(url=where)
                continue
            tool_sets[name] = named(where)(directory)
            self.tool_bindings[name] = ToolBinding(local=name)
            stack.push_async_callback(_closed, tool_sets[name])
        pools = await self._pools(stack)
        runner: Runner
        if profile.runner == "durable":
            from rollout_durable import DurableRunner

            runner = DurableRunner(
                directory / "runs",
                recorder=self.recorder,
                tool_sets=tool_sets,
                pools=pools,
                hooks=[feed],
                blobs=self.blobs,
            )
        elif profile.runner == "local":
            from rollout.local import LocalRunner

            runner = LocalRunner(
                recorder=self.recorder, tool_sets=tool_sets, pools=pools, hooks=[feed], blobs=self.blobs
            )
        else:
            raise ValueError(f"runner is {profile.runner!r}: it is local or durable")
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
            runs=self._runs,
            hooks=[feed],
            guard=_needs(profile.runs_gib, "to run more episodes"),
            presence=presence_of(self.ledger),
            about=lambda: self._about(record),
        )
        await self.runner.prepare()  # (before the runner recovers its runs: they find their claims adopted)
        await runner.launch()
        stack.push_async_callback(runner.close)
        _background(stack, self.runner.serve())
        self.trainer = learner
        if learner is not None and described is not None and described.colocated:
            ready = _needs(profile.training_gib, "to train")
            self.trainer = Colocated(learner, list(self.channels.values()), guard=ready)
        if profile.serve:
            _background(stack, self._serve(profile.serve))
        return self

    async def _pools(self, stack: contextlib.AsyncExitStack) -> dict[str, Pool]:
        """The pools the profile names that live in this process, each with its keeper; the rest are bound by URL.
        A pool's leases are kept beside the ledger, under a name of this machine and run; it refuses a key whose claim
        has lapsed. Under a durable runner, closing leaves its leases for the runs resumed when it starts again."""
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
            release = self.profile.runner != "durable"
            stack.push_async_callback(pool.close, release=release)  # (after the runner: its runs release theirs first)
            _background(stack, keep(pool, self.ledger, presence_of(self.ledger)))
            pools[kind] = pool
            self.pool_bindings[kind] = PoolBinding(local=kind)
        return pools

    @property
    def layout(self) -> str | None:
        """The layout the trained channel's engines load checkpoints in, if they are resharded."""
        trainer = self.profile.trainer
        return self.profile.channels[trainer.channel].reshard if trainer is not None else None

    async def reshard(self, checkpoint: Checkpoint, fence: Fence) -> Manifest:
        """A checkpoint's files in the trained channel's layout: resharded as a Ray task when the profile names a Ray
        cluster, else here."""
        assert self.layout is not None
        if self.profile.ray:
            return await on_ray(self.location, self.blobs_at, fence, checkpoint.id, self.layout)
        return await reshard(self.checkpoints, fence, checkpoint.id, self.layout, self.profile.directory / "resharding")

    async def eval_run(self, step: int) -> str:
        """The run of the eval of the checkpoint this run made at `step` (`rollout_train.evals.Schedule`), by id:
        registered the first time as `NAME-eval-STEP` and kept in `directory/evals`; the runner plays its episodes."""
        where = self.profile.directory / "evals" / f"{self.run.id}-eval-{step}"
        entry = await run_of(where, self.ledger, self.registry, f"{self.run.name}-eval-{step}")
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
        samples are stamped with (a checkpoint's depth). The runner beats at once, saying what the channel serves."""
        served = await self.recorder.publish(channel, adapter, path, version, full=full)
        with contextlib.suppress(Exception):  # (a beat missed is said at the next)
            await self.runner.beat()
        return served

    def _about(self, record: Path) -> dict[str, JsonValue]:
        """What the runner says of this machine in each beat: its host, the run, its measurements, the engines'
        processes, and what each channel serves and how fast since the beat before."""
        processes: Any = None
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            noted = json.loads(record.read_text())
            started = [
                {"pid": int(pid), "name": str(name), "alive": alive(int(pid))}
                for pid, name in noted["processes"].items()
            ]
            processes = {"owner": int(noted["owner"]), "started": started}
        return {
            "host": socket.gethostname(),
            "run": self.run.id,
            "directory": str(self.profile.directory),
            "machine": measured(self.profile.directory),
            "processes": processes,
            "channels": [
                {"channel": name, "adapter": channel.serving, "version": channel.version, **channel.take()}
                for name, channel in self.channels.items()
            ],
        }

    async def _serve(self, address: str) -> None:
        """The harness endpoint, over HTTP."""
        import uvicorn
        from starlette.applications import Starlette

        from rollout_train.recorder.compat import create_app as harness_endpoint

        host, _, port = address.rpartition(":")
        app = Starlette(routes=list(harness_endpoint(self.recorder).routes))
        await uvicorn.Server(uvicorn.Config(app, host=host, port=int(port), log_level="warning")).serve()


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
