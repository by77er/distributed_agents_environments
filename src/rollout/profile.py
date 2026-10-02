"""A deployment, described: the channels and the engines behind them, the trainer, the runner, where tool sets live.

Whoever deploys writes this down once (a TOML file, or the dataclasses below) and opens it; whoever trains gets
`jobs` and a `trainer` and never learns what stands behind them. Scaling is a change here: more engines behind a
channel, a durable runner instead of an in-process one, a tool set at a URL instead of in this process.

```toml
directory = "~/.cache/rollout/runs/first"     # the run's state: adapters, the job's log, metrics, the monitor's feed
runner = "local"                              # or "durable": runs survive this process
serve = "127.0.0.1:8900"                      # optional: rollout jobs and the model endpoint for harnesses, over HTTP

[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "qwen3.5"
thinking_tokens = 1024
answer_tokens = 400
engines = [{ gpu_share = 0.78 }]              # one entry per replica

[trainer]
rank = 32
learning_rate = 5e-5
sequence_tokens = 8000                        # the longest turn it can train on: the channels take it as their limit
sequences_per_step = 384
colocated = true                              # it shares the engines' GPU: they sleep while it steps

[tools]
minecraft = "minecraft_swarm.worlds:tools"    # made in this process by `tools(directory)`; or "http://worlds:8700"

[memory]
runs_gib = 6                                  # must be available to admit runs
training_gib = 4                              # and to start a step
```
"""

import asyncio
import contextlib
import importlib
import math
import random
import tomllib
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rollout.core.harness.imports import ToolBinding, ToolSet
from rollout.core.harness.runner import Runner, bind, with_row
from rollout.inference import Channel, Engine, Limits
from rollout.recorder import Recorder
from rollout.rollouts import Catalog, RolloutJobs
from rollout.training import Budget, Colocated, Directory, Grpo, LoraTrainer, Trainer, train

__all__ = ["ChannelSpec", "EngineSpec", "NotEnoughMemory", "Platform", "Profile", "TrainerSpec"]


@dataclass(frozen=True)
class EngineSpec:
    gpu_share: float = 0.78
    """Of its GPU's memory, while awake. The weights take what they take; the rest is its cache."""
    max_model_len: int = 8192
    concurrency: int = 32
    """Requests it works on at once."""


@dataclass(frozen=True)
class ChannelSpec:
    model: str
    renderer: str
    thinking_tokens: int = 1024
    answer_tokens: int = 400
    engines: tuple[EngineSpec, ...] = (EngineSpec(),)


@dataclass(frozen=True)
class TrainerSpec:
    rank: int = 32
    learning_rate: float = 5e-5
    sequence_tokens: int = 8000
    sequences_per_step: int = 384
    colocated: bool = True


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
    channels: Mapping[str, ChannelSpec]
    trainer: TrainerSpec = TrainerSpec()
    runner: str = "local"
    serve: str | None = None
    tools: Mapping[str, str] = field(default_factory=dict[str, str])
    runs_gib: float = 0.0
    training_gib: float = 0.0
    feed_runs: int = 80
    """Episodes kept in the monitor's feed (the oldest are deleted)."""

    @classmethod
    def load(cls, path: Path, *, directory: Path | None = None) -> "Profile":
        """The profile a TOML file describes; `directory` replaces the file's (one profile, many runs)."""
        described = tomllib.loads(path.read_text())
        channels = {
            name: ChannelSpec(
                **{key: value for key, value in channel.items() if key != "engines"},
                engines=tuple(EngineSpec(**engine) for engine in channel.get("engines", [{}])),
            )
            for name, channel in described.get("channels", {}).items()
        }
        memory = described.get("memory", {})
        return cls(
            directory=directory or Path(described["directory"]).expanduser(),
            channels=channels,
            trainer=TrainerSpec(**described.get("trainer", {})),
            runner=described.get("runner", "local"),
            serve=described.get("serve"),
            tools=described.get("tools", {}),
            runs_gib=float(memory.get("runs_gib", 0.0)),
            training_gib=float(memory.get("training_gib", 0.0)),
        )

    @contextlib.asynccontextmanager
    async def open(self) -> AsyncGenerator["Platform"]:
        """Start what the profile describes, and stop it on the way out."""
        platform = await Platform.start(self)
        try:
            yield platform
        finally:
            await platform.close()


class Platform:
    """An open profile: `jobs` to run episodes with, a `trainer` to step, and `train` for the loop over both."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.store = Directory(profile.directory)
        self.recorder: Recorder
        self.jobs: RolloutJobs
        self.trainer: Trainer
        self.tool_bindings: dict[str, ToolBinding] = {}
        self._closing: list[Any] = []
        self._tasks: list[asyncio.Task[None]] = []

    @classmethod
    async def start(cls, profile: Profile) -> "Platform":
        from transformers import AutoTokenizer

        from rollout.inference.vllm import VllmEngine, end_orphaned_engines, note_engines
        from rollout.monitor import RunFeed
        from rollout.recorder import renderer_for
        from rollout.recorder.renderers import Tokenizer

        self = cls(profile)
        directory = profile.directory
        directory.mkdir(parents=True, exist_ok=True)
        end_orphaned_engines(directory / "engine.json")  # one a killed process left behind holds the GPU
        learner = LoraTrainer(
            next(iter(profile.channels.values())).model,
            directory,
            rank=profile.trainer.rank,
            learning_rate=profile.trainer.learning_rate,
            budget=Budget(profile.trainer.sequence_tokens, profile.trainer.sequences_per_step),
        )
        channels: dict[str, Channel] = {}
        for name, spec in profile.channels.items():
            tokenizer = cast(Tokenizer, AutoTokenizer.from_pretrained(spec.model))  # pyright: ignore[reportUnknownMemberType]
            engines: list[Engine] = [
                VllmEngine(
                    spec.model,
                    gpu_memory_utilization=engine.gpu_share,
                    max_model_len=engine.max_model_len,
                    max_num_seqs=engine.concurrency,
                    max_lora_rank=profile.trainer.rank,
                )
                for engine in spec.engines
            ]
            limits = Limits(spec.thinking_tokens, spec.answer_tokens, profile.trainer.sequence_tokens)
            channels[name] = Channel(name, engines, renderer_for(spec.renderer, tokenizer), limits)
            self._closing.append(channels[name])
            if learner.latest is not None:  # a run that is started again serves its newest weights
                await channels[name].publish(*learner.latest)
        note_engines(directory / "engine.json")
        self.recorder = Recorder(channels, base_url=f"http://{profile.serve}/v1" if profile.serve else None)
        feed = RunFeed(directory / "feed", keep=profile.feed_runs)
        self._closing.append(feed)
        tool_sets: dict[str, ToolSet] = {}
        for name, where in profile.tools.items():
            if where.startswith(("http://", "https://")):
                self.tool_bindings[name] = ToolBinding(url=where)
                continue
            module, _, factory = where.partition(":")
            tool_sets[name] = getattr(importlib.import_module(module), factory)(directory)
            self.tool_bindings[name] = ToolBinding(local=name)
            self._closing.append(tool_sets[name])
        runner: Runner
        if profile.runner == "durable":
            from rollout.durable import DurableRunner

            durable = DurableRunner(directory / "runs", recorder=self.recorder, tool_sets=tool_sets, hooks=[feed])
            await durable.launch()
            self._closing.append(durable)
            runner = cast(Runner, durable)
        else:
            from rollout.core.local import LocalRunner

            runner = cast(Runner, LocalRunner(recorder=self.recorder, tool_sets=tool_sets, hooks=[feed]))
        guard = (lambda: require_memory(profile.runs_gib, "to run more episodes")) if profile.runs_gib else None
        self.jobs = RolloutJobs(runner, self.recorder, log=directory / "jobs", hooks=[feed], guard=guard)
        self._closing.append(self.jobs)
        self.trainer = learner
        if profile.trainer.colocated:
            ready = (lambda: require_memory(profile.training_gib, "to train")) if profile.training_gib else None
            self.trainer = Colocated(learner, list(channels.values()), guard=ready)
        self._tasks.append(asyncio.create_task(self._measure(feed, channels)))
        if profile.serve:
            self._tasks.append(asyncio.create_task(self._serve(profile.serve)))
        return self

    async def train(self, catalog: Catalog, *, groups: int = 100, algorithm: Grpo | None = None, seed: int = 0) -> None:
        """The training loop over this platform, on the first channel."""
        channel = next(iter(self.profile.channels))
        first = with_row(catalog.program, catalog.start(catalog.rows()[0], random.Random(0)))
        binding = bind(first, channel, tools=self.tool_bindings)
        await train(
            self.jobs, catalog, self.trainer, self.store, channel=channel, algorithm=algorithm or Grpo(),
            groups=groups, seed=seed, binding=binding,
        )  # fmt: skip

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for thing in reversed(self._closing):  # the jobs first, the engines last
            close = getattr(thing, "close", None)
            closed = close() if close is not None else None
            if asyncio.iscoroutine(closed):
                with contextlib.suppress(Exception):
                    await closed

    async def _measure(self, feed: Any, channels: Mapping[str, Channel], every: float = 60.0) -> None:
        """Tell whoever watches how the engines are doing, once a minute."""
        while True:
            await asyncio.sleep(every)
            for name, channel in channels.items():
                counts = channel.take()
                if counts["requests"]:
                    feed.on_job({"kind": "inference", "channel": name, "version": channel.version, **counts})

    async def _serve(self, address: str) -> None:
        """Rollout jobs and the harness endpoint, over HTTP."""
        import uvicorn
        from starlette.applications import Starlette

        from rollout.recorder.compat import create_app as harness_endpoint
        from rollout.rollouts.service import create_app as rollout_service

        host, _, port = address.rpartition(":")
        app = Starlette(routes=[*harness_endpoint(self.recorder).routes, *rollout_service(self.jobs).routes])
        await uvicorn.Server(uvicorn.Config(app, host=host, port=int(port), log_level="warning")).serve()
