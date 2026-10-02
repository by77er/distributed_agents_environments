"""A deployment, described: the channels and the engines behind them, the trainer, the runner, where tool sets live.

Whoever deploys writes this down once (a TOML file, or the dataclasses below) and opens it; whoever trains gets
`jobs` and a `trainer` and never learns what stands behind them. Scaling is a change here: more engines behind a
channel, a durable runner instead of an in-process one, a tool set at a URL instead of in this process.

Engines, renderers, the trainer and tool sets are named as `module:name`, and what the profile says of each is
passed to it: this module knows no engine and no trainer. docs/guide/deploying.md describes the file.
"""

import asyncio
import contextlib
import math
import tomllib
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.harness.imports import ToolBinding, ToolSet
from rollout.harness.runner import Runner
from rollout.names import named
from rollout.processes import end_orphans, note_processes
from rollout_train import Colocated, FileLedger, Ledger, Policies, Trainer
from rollout_train.inference import Channel, Engine, Limits
from rollout_train.ledger import LEDGER
from rollout_train.recorder import Recorder
from rollout_train.recorder.recorder import SERVED_UNDER
from rollout_train.rollouts import RolloutJobs

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
    policy: str | None = None
    """The policy it trains, by name (by default the run directory's name). A policy that has versions is gone on
    with."""
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
    """The run's state: adapters, the job's log, metrics, the monitor's feed."""
    channels: Mapping[str, ChannelSpec]
    trainer: TrainerSpec | None = None
    runner: str = "local"
    """`local` runs episodes in this process; `durable` records them so that they survive it."""
    serve: str | None = None
    """`host:port` to serve the rollout jobs and the model endpoint for harnesses on."""
    address: str | None = None
    """The URL others reach `serve` at (by default `http://` and `serve`)."""
    tools: Mapping[str, str] = field(default_factory=dict[str, str])
    """Each tool set by name: a URL, or `module:name` of what makes it, called with `directory`."""
    ledger: Path | None = None
    """Where the run's tables and the policies' versions are kept, in files (by default `directory/ledger`).
    Runs that share it see each other's policies."""
    blobs: Mapping[str, Any] = field(default_factory=dict[str, Any])
    """Where episodes (and what programs store) are kept: `kind` is `module:name` of what makes the store, called
    with the other entries. Without one, files under `directory/blobs`."""
    runs_gib: float = 0.0
    """System memory that must be available to admit runs."""
    training_gib: float = 0.0
    """And to start a step of a colocated trainer."""
    feed_runs: int | None = None
    """Episodes kept in the monitor's feed, where it should not keep `RunFeed`'s own number (the oldest are
    deleted)."""

    @classmethod
    def load(cls, path: Path, *, directory: Path | None = None) -> "Profile":
        """The profile a TOML file describes; `directory` replaces the file's (one profile, many runs). A key the
        file has and a profile does not is an error: a misspelt guard would otherwise be no guard."""
        described = tomllib.loads(path.read_text())
        channels: dict[str, ChannelSpec] = {}
        for name, channel in _table(described, "channels").items():
            known = ("model", "renderer", "engine", "engines", "thinking_tokens", "answer_tokens")
            given = _only(dict(channel), f"channels.{name}", *known)
            engines = tuple(given.pop("engines", [{}]))
            channels[name] = ChannelSpec(**given, engines=engines)
        trainer = _table(described, "trainer")
        memory = _only(_table(described, "memory"), "memory", "runs_gib", "training_gib")
        blobs = _table(described, "blobs")
        known = ("directory", "ledger", "runner", "serve", "address", "tools", "feed_runs")
        top = _only(described, "the profile", *known)
        top["directory"] = directory or Path(top["directory"]).expanduser()
        if "ledger" in top:
            top["ledger"] = Path(top["ledger"]).expanduser()
        return cls(
            **top,
            **memory,
            blobs=blobs,
            channels=channels,
            trainer=TrainerSpec(
                kind=trainer.pop("kind"),
                channel=trainer.pop("channel"),
                policy=trainer.pop("policy", None),
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
    """An open profile: `jobs` to run episodes with, a `trainer` to step, and the `policies` it trains."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.ledger: Ledger = FileLedger(profile.ledger or profile.directory / LEDGER)
        self.policies: Policies
        """The policies' versions, in the ledger and the blob store."""
        self.policy = (profile.trainer.policy if profile.trainer else None) or profile.directory.name
        """The policy the profile's trainer trains."""
        self.channels: dict[str, Channel] = {}
        self.recorder: Recorder
        self.jobs: RolloutJobs
        self.trainer: Trainer | None = None
        self.tool_bindings: dict[str, ToolBinding] = {}
        """Where a run finds each tool set the profile names (for a run's binding)."""
        self.blobs: Blobs
        """Where the jobs keep their episodes."""

    @classmethod
    async def start(cls, profile: Profile, stack: contextlib.AsyncExitStack) -> "Platform":
        """Start everything, registering with `stack` how each thing is stopped (the engines last)."""
        from rollout_train.monitor import RunFeed

        self = cls(profile)
        directory = profile.directory
        directory.mkdir(parents=True, exist_ok=True)
        record = directory / "engine.json"
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
                    sequence=trained.budget.sequence_tokens if trained is not None else None,
                ),
            )
        address = profile.address or (f"http://{profile.serve}" if profile.serve else None)
        self.recorder = Recorder(self.channels, base_url=f"{address}{SERVED_UNDER}" if address else None)
        feed = RunFeed(directory / "feed", **({"keep": profile.feed_runs} if profile.feed_runs else {}))
        stack.callback(feed.close)
        tool_sets: dict[str, ToolSet] = {}
        for name, where in profile.tools.items():
            if where.startswith(("http://", "https://")):
                self.tool_bindings[name] = ToolBinding(url=where)
                continue
            tool_sets[name] = named(where)(directory)
            self.tool_bindings[name] = ToolBinding(local=name)
            stack.push_async_callback(_closed, tool_sets[name])
        store = dict(profile.blobs)
        self.blobs = named(store.pop("kind"))(**store) if store else FileBlobStore(directory / "blobs")
        self.policies = Policies(self.ledger, self.blobs)
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
        guard = _needs(profile.runs_gib, "to run more episodes")
        self.jobs = RolloutJobs(
            runner, self.recorder, log=directory / "jobs", blobs=self.blobs, hooks=[feed], guard=guard
        )
        stack.push_async_callback(self.jobs.close)
        self.trainer = learner
        if learner is not None and described is not None and described.colocated:
            ready = _needs(profile.training_gib, "to train")
            self.trainer = Colocated(learner, list(self.channels.values()), guard=ready)
        _background(stack, self._measure(feed))
        if profile.serve:
            _background(stack, self._serve(profile.serve))
        return self

    async def _measure(self, feed: Any, every: float = 60.0) -> None:
        """Tell whoever watches how the engines are doing, once a minute."""
        while True:
            await asyncio.sleep(every)
            for name, channel in self.channels.items():
                counts = channel.take()
                if counts["requests"]:
                    feed.on_job({"kind": "inference", "channel": name, "version": channel.version, **counts})

    async def _serve(self, address: str) -> None:
        """Rollout jobs and the harness endpoint, over HTTP."""
        import uvicorn
        from starlette.applications import Starlette

        from rollout_train.recorder.compat import create_app as harness_endpoint
        from rollout_train.rollouts.service import create_app as rollout_service

        host, _, port = address.rpartition(":")
        app = Starlette(routes=[*harness_endpoint(self.recorder).routes, *rollout_service(self.jobs).routes])
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
