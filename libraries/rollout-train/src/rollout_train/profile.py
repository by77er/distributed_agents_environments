"""A deployment, described: the channels and the engines behind them, the trainer, the runner, where tool sets live.

Whoever deploys writes this down once (a TOML file, or the dataclasses below) and opens it; whoever trains gets the
`policies`, a `trainer` and a way to `publish` versions, while a runner plays the episodes the run asks for in the
ledger, and never learns what stands behind them. Scaling is a change here: more engines behind a
channel, a durable runner instead of an in-process one, a tool set at a URL instead of in this process.

Engines, renderers, the trainer and tool sets are named as `module:name`, and what the profile says of each is
passed to it: this module knows no engine and no trainer. docs/guide/deploying.md describes the file.
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
from rollout.names import named
from rollout.processes import end_orphans, note_processes
from rollout_train import Colocated, Ledger, Trainer, Version, Versions
from rollout_train.inference import Channel, Engine, Limits
from rollout_train.layout import BLOBS, FEED, LEDGER, PROCESSES
from rollout_train.ledger import LOCATION
from rollout_train.ledger import opened as ledger_at
from rollout_train.machine import alive, measured
from rollout_train.presence import presence_of
from rollout_train.recorder import Recorder
from rollout_train.recorder.recorder import SERVED_UNDER
from rollout_train.registry import Entry, Registry, registry_of, resolved, run_of
from rollout_train.rollouts.scheduler import EpisodeRunner
from rollout_train.stores import location, opened

__all__ = ["ChannelSpec", "NotEnoughMemory", "Platform", "Profile", "TrainerSpec"]


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


@dataclass(frozen=True)
class TrainerSpec:
    kind: str
    """`module:name` of what makes the trainer, called with the channel's model and `settings`."""
    channel: str
    """The channel that serves the policy it trains."""
    start: str | None = None
    """The version a new run trains from (`rollout_train.registry.resolved`: a bookmark, `RUN:STEP`, `RUN`, or a
    version's id or the start of one); by default the base model. A run started again goes on from the newest version
    it made."""
    bookmark: str | None = None
    """A bookmark the run carries: moved to each version it makes."""
    colocated: bool = False
    """Whether it shares the channels' accelerator: their engines then sleep while it steps."""
    settings: Mapping[str, Any] = field(default_factory=dict[str, Any])


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
    """The run's state: versions' files while in use, the monitor's feed, and (unless the profile names other places)
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
    ledger: Mapping[str, Any] = field(default_factory=dict[str, Any])
    """Where the run's tables and the policies' versions are kept (`rollout_train.ledger.opened`): `{"directory": …}`,
    in files; `{"kind": "module:name", …}`, what that makes from the other entries, such as a database
    (`rollout_train.database:DatabaseLedger` with a `url`). By default files under `directory/ledger`. Runs that share
    a ledger see each other's versions."""
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
    name: str | None = None
    """What a run first started in `directory` is called (by default the directory's name). It is named again with
    `rollout rename`; its id, in the directory's `run.json`, never changes."""

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
            known = ("model", "renderer", "engine", "engines", "thinking_tokens", "answer_tokens")
            given = _only(dict(channel), f"channels.{name}", *known)
            engines = tuple(given.pop("engines", [{}]))
            channels[name] = ChannelSpec(**given, engines=engines)
        trainer = _table(described, "trainer")
        memory = _only(_table(described, "memory"), "memory", "runs_gib", "training_gib")
        blobs = _table(described, "blobs")
        known = ("directory", "ledger", "runner", "serve", "address", "tools", "feed_runs", "episodes_at_once")
        top = _only(described, "the profile", *known)
        top["directory"] = directory or Path(top["directory"]).expanduser()
        if isinstance(top.get("ledger"), str):  # (`ledger = "path"`: a directory of files)
            top["ledger"] = {"directory": str(Path(top["ledger"]).expanduser())}
        return cls(
            **top,
            **memory,
            blobs=blobs,
            channels=channels,
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
    async def open(self) -> AsyncGenerator["Platform"]:
        """Start what the profile describes, and stop it on the way out (also if starting fails half way)."""
        async with contextlib.AsyncExitStack() as stack:
            yield await Platform.start(self, stack)


def _table(described: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(described.pop(name, {}))


def _only(table: dict[str, Any], where: str, *known: str) -> dict[str, Any]:
    if unknown := sorted(set(table) - set(known)):
        raise ValueError(f"{where} has no {', '.join(unknown)} (it has {', '.join(known)})")
    return table


class Platform:
    """An open profile: its `run`, the version it trains from (`origin`), the `versions`, a `trainer` to step,
    `publish` to serve a version, and a `runner` that plays the episodes its run asks for
    (`rollout_train.rollouts.scheduler.EpisodeRunner`)."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.location: dict[str, Any] = dict(profile.ledger) or {"directory": str(profile.directory / LEDGER)}
        """Where the ledger is; written into the run's directory, for whatever reads the run."""
        self.ledger: Ledger = ledger_at(self.location)
        self.versions: Versions
        """The versions, in the ledger and the blob store."""
        self.registry: Registry | None = registry_of(self.ledger)
        """What the runs are called, and the bookmarks."""
        self.run: Entry
        """The run in the profile's directory."""
        self.origin: str | None = None
        """The version a new run trains from, by id (None: the base model)."""
        self.channels: dict[str, Channel] = {}
        self.recorder: Recorder
        self.runner: EpisodeRunner
        self.feed: Any
        """The monitor's feed (`RunFeed`): the loop's results and steps go there too."""
        self.trainer: Trainer | None = None
        self.tool_bindings: dict[str, ToolBinding] = {}
        """Where a run finds each tool set the profile names (for a run's binding)."""
        self.blobs: Blobs
        """Where episodes' trajectories and events, and versions' files, are kept."""
        self.blobs_at: dict[str, Any] = {}
        """Where that is, as any process opens it (`rollout_train.stores`), for the run's `starts` record."""

    @classmethod
    async def start(cls, profile: Profile, stack: contextlib.AsyncExitStack) -> "Platform":
        """Start everything, registering with `stack` how each thing is stopped (the engines last)."""
        from rollout_train.monitor import RunFeed

        self = cls(profile)
        directory = profile.directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / LOCATION).write_text(json.dumps(self.location))
        self.run = await run_of(directory, self.ledger, self.registry, profile.name)
        if profile.trainer is not None and profile.trainer.start is not None:
            self.origin = await resolved(self.ledger, self.registry, profile.trainer.start)
        record = directory / PROCESSES
        end_orphans(record)  # an engine a killed process left behind holds its accelerator
        described = profile.trainer
        learner: Trainer | None = None
        if described is not None:
            learner = named(described.kind)(profile.channels[described.channel].model, **described.settings)
        started: list[Engine] = []
        for name, spec in profile.channels.items():
            engines: list[Engine] = []
            for options in spec.engines:
                engine: Engine = named(spec.engine)(spec.model, **options)
                stack.callback(engine.close)
                engines.append(engine)
                started.append(engine)
                note_processes(record, [pid for each in started for pid in each.processes])
            trained = learner if described is not None and described.channel == name else None
            limits = {"thinking": spec.thinking_tokens, "answer": spec.answer_tokens}
            self.channels[name] = Channel(
                name,
                engines,
                named(spec.renderer)(spec.model),
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
        self.blobs = opened(dict(profile.blobs)) if profile.blobs else FileBlobStore(directory / BLOBS)
        self.blobs_at = location(profile.blobs, directory / BLOBS)
        self.versions = Versions(self.ledger, self.blobs)
        runner: Runner
        if profile.runner == "durable":
            from rollout_durable import DurableRunner

            runner = DurableRunner(
                directory / "runs", recorder=self.recorder, tool_sets=tool_sets, hooks=[feed], blobs=self.blobs
            )
        elif profile.runner == "local":
            from rollout.local import LocalRunner

            runner = LocalRunner(recorder=self.recorder, tool_sets=tool_sets, hooks=[feed], blobs=self.blobs)
        else:
            raise ValueError(f"runner is {profile.runner!r}: it is local or durable")
        await runner.launch()
        stack.push_async_callback(runner.close)
        self.feed = feed
        self.runner = EpisodeRunner(
            f"{socket.gethostname()}/{directory.name}",  # (the same name when started again: what it claimed is free)
            self.ledger,
            runner,
            self.recorder,
            self.blobs,
            places=profile.episodes_at_once,
            imports=list(tool_sets),
            runs=[self.run.id],
            hooks=[feed],
            guard=_needs(profile.runs_gib, "to run more episodes"),
            presence=presence_of(self.ledger),
            about=lambda: self._about(record),
        )
        _background(stack, self.runner.serve())
        self.trainer = learner
        if learner is not None and described is not None and described.colocated:
            ready = _needs(profile.training_gib, "to train")
            self.trainer = Colocated(learner, list(self.channels.values()), guard=ready)
        if profile.serve:
            _background(stack, self._serve(profile.serve))
        return self

    async def bookmarked(self) -> set[str]:
        """The versions bookmarks name (which keep their files)."""
        return {mark.version for mark in await self.registry.bookmarks()} if self.registry else set()

    async def made(self, version: Version) -> None:
        """Carry the profile's bookmark, if it names one, to a version the run made."""
        if self.registry is not None and self.profile.trainer is not None and self.profile.trainer.bookmark:
            await self.registry.bookmark(self.profile.trainer.bookmark, version.id)

    async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int:
        """Serve new weights on a channel from now on; returns the version they are served as."""
        return await self.recorder.publish(channel, adapter, path, version)

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
                {"channel": name, "adapter": channel.adapter, "version": channel.version, **channel.take()}
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
