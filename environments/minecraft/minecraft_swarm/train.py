"""Train the swarm: curriculum → group of episodes → group-relative update → new adapter, on one GPU.

Each iteration the curriculum picks a task; a group of episodes runs it concurrently from identical starts (same
world seed, same layout seed), with the four agents of every episode sampling from one recorded channel. Each
episode's reward is its task's objective, scored from ground truth. When every episode of a group has finished, and
if their rewards differ, the engine sleeps, a trainer process takes one step on a sample of the group's turns and
exits, and the new LoRA adapter is loaded into the engine once it is awake again.

Groups overlap, so that one slow episode does not hold the GPU idle: the next group starts when at most
`stragglers` episodes of earlier groups are still running. No episode is left out: a group is trained on only when
its last episode is done, with advantages over all of them. Its turns may then be an update or two old, and a
straggler plays on under the newer adapter; the update is PPO's clipped ratio against the logprobs recorded when
each token was sampled, which is what corrects for that. During an update every running episode waits, its world
frozen between turns.

A run that is started again in the same directory goes on from its latest adapter, its curriculum and its count of
iterations.

Memory is checked before each group and each update: the run stops with `NotEnoughMemory` rather than exhaust the
machine.

Outputs (in `directory`): `metrics.jsonl` (one line per iteration), `episodes.jsonl` (each episode's ground truth),
`transcripts/` (one agent's turns per iteration), `adapters/step-N/` (PEFT adapters), `trainer/` (the optimizer's
state), `curriculum.json`, and `feed/` (every episode as it happens: what each agent saw, thought and did; watch it
with `rollout-monitor DIRECTORY/feed`).
"""

import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from minecraft_swarm.curriculum import Curriculum
from minecraft_swarm.episode import SwarmEpisode
from minecraft_swarm.prompts import TEAM
from minecraft_swarm.tasks import Task, catalog
from minecraft_swarm.worlds import MinecraftTools, MinecraftWorlds
from rollout.core.contracts import RunEventType
from rollout.core.harness import (
    ModelBinding,
    ProgramReference,
    RecordedModel,
    RunBinding,
    RunSpecification,
    RunStatus,
    SamplingParameters,
    ToolBinding,
    register,
)
from rollout.core.local import LocalRunner
from rollout.core.local.runner import LocalRunHandle
from rollout.monitor import RunFeed
from rollout.recorder import Channel, MeteredEngine, Recorder, renderer_for
from rollout.recorder.engines import VllmEngine
from rollout.recorder.renderers import Tokenizer


@dataclass
class TrainingSettings:
    directory: Path
    model: str = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
    renderer: str = "qwen3.5"
    iterations: int = 100
    group_size: int = 4
    worlds: int = 12
    """How many world seeds episodes draw from (a template is generated for each, once)."""
    max_minutes: float | None = None
    """Caps each task's budget of game time (None: the task's own)."""
    max_turns: int | None = None
    """Caps each episode's turns (for smoke tests)."""
    max_sequence_tokens: int = 8000
    """No turn is longer (prompt and completion): the recorder gives a long prompt less room to think, so that the
    trainer can train on every turn. The trainer's peak on the GPU grows with length (10.7 GiB at 5,000 tokens, 12.4
    at 8,000, 13.7 at 10,000); past the card's memory, Windows spills into system memory and the step crawls (a
    10,000-token turn took 74 s instead of 9)."""
    update_turns: int = 384
    """At most this many agent turns per update, sampled evenly from the group's (long episodes have thousands)."""
    thinking_budget: int = 1024
    """Tokens of thinking per turn before it is closed by force. Wide on purpose: on this environment's
    observations the model's thoughts run to a median of 530 tokens and a 95th percentile of 820, and a thought cut
    off in the middle decides nothing."""
    answer_tokens: int = 400
    """Room for an answer after the thinking: a tool call takes about 40 tokens, a summary of old turns up to this."""
    temperature: float = 1.0
    learning_rate: float = 2e-5
    lora_rank: int = 32
    window_ticks: int = 100
    gpu_memory_utilization: float = 0.78
    """The engine's share of the GPU while it is awake (the trainer runs only while it sleeps, its memory freed).
    What the weights leave is the engine's cache: at 0.72 it held 63,000 tokens, less than sixteen agents' contexts.
    The engine takes about 0.8 GiB more than its share, and the desktop up to 1.7 GiB: at 0.85 the card was full,
    memory spilled into the system's, and a turn went from 10 s to minutes."""
    seed: int = 0
    tasks: list[str] | None = None
    """Restrict the curriculum to these task ids (all tasks when None)."""
    group_memory_gib: float = 6.0
    """System memory that must be available to start a group of episodes (each runs a Paper server)."""
    update_memory_gib: float = 4.0
    """System memory that must be available to train, once the engine has gone to sleep."""
    stragglers: int = 1
    """The next group starts when at most this many episodes of earlier groups are still running (0: when none is)."""
    feed_runs: int = 80
    """Episodes kept in `feed/` for the monitor (the oldest are deleted)."""
    exercise_updates: bool = False
    """For smoke tests: update with small synthetic advantages when a group carries no signal."""


async def train(settings: TrainingSettings) -> None:
    from transformers import AutoTokenizer

    from rollout.training.grpo import TrainingSequence, group_advantages

    directory = settings.directory
    (directory / "transcripts").mkdir(parents=True, exist_ok=True)
    (directory / "adapters").mkdir(exist_ok=True)
    (directory / "settings.json").write_text(json.dumps(asdict(settings), default=str, indent=1))
    rng = random.Random(settings.seed)
    world_seeds = [
        random.Random(f"world-{settings.seed}-{index}").randrange(1 << 31) for index in range(settings.worlds)
    ]
    tokenizer = cast(Tokenizer, AutoTokenizer.from_pretrained(settings.model))  # pyright: ignore[reportUnknownMemberType]
    renderer = renderer_for(settings.renderer, tokenizer)
    engine = VllmEngine(
        settings.model,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        max_num_seqs=len(TEAM) * (settings.group_size + settings.stragglers),
        max_lora_rank=settings.lora_rank,
    )
    metered = MeteredEngine(engine)
    channel = Channel(
        metered,
        renderer,
        thinking_budget=settings.thinking_budget,
        answer_tokens=settings.answer_tokens,
        max_sequence_tokens=settings.max_sequence_tokens,
    )
    recorder = Recorder({"policy": channel})
    worlds = MinecraftWorlds(window_ticks=settings.window_ticks, logs=directory / "logs")
    (directory / "logs").mkdir(exist_ok=True)
    feed = RunFeed(directory / "feed", keep=settings.feed_runs)
    runner = LocalRunner(recorder=recorder, tool_sets={"minecraft": MinecraftTools(worlds)}, hooks=[feed])
    tasks = [task for task in catalog() if settings.tasks is None or task.id in settings.tasks]
    curriculum = Curriculum(tasks, rng, start=len(tasks) if settings.tasks else 3)
    if (directory / "curriculum.json").exists():
        curriculum.load(directory / "curriculum.json")
    sampling = SamplingParameters(temperature=settings.temperature)
    binding = RunBinding(
        models={name: ModelBinding(recorded=RecordedModel(channel="policy", sampling=sampling)) for name in TEAM},
        imports={"minecraft": ToolBinding(local="minecraft")},
    )
    learner = Learner(settings, engine, metered, channel)
    await learner.resume()
    metrics = directory / "metrics.jsonl"
    done = len(metrics.read_text().splitlines()) if metrics.exists() else 0
    flight = Flight()
    finishing = asyncio.Lock()  # groups are scored, trained on and logged one at a time

    async def start(iteration: int) -> Group:
        task = curriculum.sample()
        world_seed, layout_seed = rng.choice(world_seeds), rng.randrange(1 << 30)
        minutes = task.minutes if settings.max_minutes is None else min(task.minutes, settings.max_minutes)
        parameters: dict[str, JsonValue] = {
            "task": task.id,
            "world_seed": world_seed,
            "layout_seed": layout_seed,
            "minutes": minutes,
            "turns": settings.max_turns,
        }
        labels = {"group": f"{iteration:04d}", "iteration": str(iteration), "task": task.id, "title": task.title}
        specification = RunSpecification(
            program=ProgramReference(program=register(SwarmEpisode), parameters=parameters), binding=binding
        )
        overlapped = flight.count
        handles: list[LocalRunHandle] = []
        for episode in range(1, settings.group_size + 1):  # identical starts: the same world and layout
            handles.append(await runner.start(specification, labels={**labels, "episode": str(episode)}))
            flight.took_off()
        watchers = [asyncio.create_task(flight.follow(handle)) for handle in handles]
        return Group(iteration, task, world_seed, minutes, handles, watchers, time.monotonic(), overlapped)

    async def finish(group: Group) -> None:
        await asyncio.gather(*group.watchers)
        waited = time.monotonic() - group.started
        async with finishing:
            iteration, task, handles = group.iteration, group.task, group.handles
            completed = [
                handle for handle in handles if handle.outcome and handle.outcome.status is RunStatus.COMPLETED
            ]
            rewards = [_reward(handle) for handle in completed]
            results = [_result(handle) for handle in completed]
            for handle, result in zip(completed, results, strict=True):
                _append(directory / "episodes.jsonl", {"iteration": iteration, "run": handle.run_id, **result})
            if completed:
                _transcript(
                    directory / "transcripts" / f"{iteration:04d}.txt", recorder, completed[0].run_id, tokenizer
                )
            turns = [
                (turn, handle)
                for handle in completed
                for turns_of_agent in recorder.sessions_of(handle.run_id).values()
                for turn in turns_of_agent
            ]
            line: dict[str, Any] = {
                "iteration": iteration,
                "time": round(time.time(), 1),
                "task": task.id,
                "title": task.title,
                "difficulty": task.difficulty,
                "world_seed": group.world_seed,
                "minutes": group.minutes,
                "game_minutes": [round(float(result.get("game_minutes", 0)), 1) for result in results],
                "turns": [result.get("turns") for result in results],
                "rewards": rewards,
                "solved": [bool(result.get("solved")) for result in results],
                "failed": len(handles) - len(completed),
                "failures": [handle.outcome.detail for handle in handles if handle not in completed and handle.outcome],
                "adapter_step": learner.step,
                "sampled_under": sorted({turn.adapter_version for turn, _ in turns}),
                "overlapped": group.overlapped,
                "rollout_seconds": round(waited, 1),
                "inference": metered.take(),
                "memory_available_gib": round(available_memory_gib(), 1),
            }
            if rewards:
                curriculum.update(task, rewards, [bool(result.get("solved")) for result in results])
            advantages = group_advantages(rewards) if len(rewards) >= 2 else None
            if advantages is None and settings.exercise_updates and len(rewards) >= 2:
                advantages = [0.1 * (index - (len(rewards) - 1) / 2) for index in range(len(rewards))]  # smoke tests
            if advantages is None:
                line["update"] = "skipped: every episode scored the same"
            else:
                advantage_of = {
                    handle.run_id: advantage for handle, advantage in zip(completed, advantages, strict=True)
                }
                line["turns_recorded"] = len(turns)
                if len(turns) > settings.update_turns:  # an even sample: every episode and agent keeps its share
                    turns = random.Random(iteration).sample(turns, settings.update_turns)
                sequences = [  # built only for the turns trained on: a long episode records tens of thousands
                    TrainingSequence(
                        tokens=[*turn.prompt, *turn.completion],
                        loss_mask=[False] * len(turn.prompt) + turn.loss_mask,
                        behavior_logprobs=[math.nan] * len(turn.prompt) + turn.logprobs,
                        advantage=advantage_of[handle.run_id],
                    )
                    for turn, handle in turns
                ]
                line["update"] = await learner.update(sequences, seed=iteration)
                line["adapter_step"] = learner.step
            for handle in handles:
                recorder.forget(handle.run_id)
            line["seconds"] = round(time.monotonic() - group.started, 1)
            line["unlocked"] = len(curriculum.unlocked())
            _append(metrics, line)
            curriculum.save(directory / "curriculum.json")
            print(json.dumps(line), flush=True)

    finishers: list[asyncio.Task[None]] = []
    try:
        for iteration in range(done + 1, done + settings.iterations + 1):
            await flight.at_most(settings.stragglers)
            for finisher in finishers:  # a group that failed to finish (no memory, a trainer error) ends the run
                if finisher.done() and finisher.exception() is not None:
                    raise cast(BaseException, finisher.exception())
            require_memory(settings.group_memory_gib, "to run a group of episodes")
            finishers.append(asyncio.create_task(finish(await start(iteration))))
        await asyncio.gather(*finishers)
    finally:
        for finisher in finishers:
            finisher.cancel()
        await worlds.close()
        engine.close()
        feed.close()


@dataclass
class Group:
    """A group of episodes of one task from identical starts."""

    iteration: int
    task: Task
    world_seed: int
    minutes: float
    handles: list[LocalRunHandle]
    watchers: list[asyncio.Task[None]]
    started: float
    overlapped: int
    """Episodes of earlier groups still running when this one started."""


class Flight:
    """The episodes in flight, so that the next group can start when few enough are left."""

    def __init__(self) -> None:
        self.count = 0
        self._changed = asyncio.Condition()

    def took_off(self) -> None:
        self.count += 1

    async def follow(self, handle: LocalRunHandle) -> None:
        """Wait for an episode to end, however it ends."""
        try:
            await handle.result()
        finally:
            async with self._changed:
                self.count -= 1
                self._changed.notify_all()

    async def at_most(self, count: int) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self.count <= count)


class Learner:
    """Takes turns with the engine on the GPU: hold back the episodes still running, sleep the engine, train one step
    in a fresh trainer process (which exits, freeing the GPU and its memory), wake the engine, switch the channel to
    the new adapter and let the episodes go on. The adapter before stays loaded: a turn that was waiting to be
    sampled finishes under the adapter it started with."""

    def __init__(self, settings: TrainingSettings, engine: VllmEngine, gate: MeteredEngine, channel: Channel) -> None:
        from rollout.training.worker import TrainerProcess, TrainerSettings

        self.settings = settings
        self.engine = engine
        self.gate = gate
        self.channel = channel
        self.step = 0
        self.loaded: list[str] = []
        """Adapters in the engine, oldest first: the current one and the one before."""
        self.trainer = TrainerProcess(
            TrainerSettings(
                settings.model,
                state=settings.directory / "trainer",
                rank=settings.lora_rank,
                alpha=2.0 * settings.lora_rank,
                learning_rate=settings.learning_rate,
                max_sequence_tokens=settings.max_sequence_tokens,
            )
        )

    async def resume(self) -> None:
        """Go on from the latest adapter in the run's directory, if there is one."""
        adapters = self.settings.directory / "adapters"
        steps = [
            int(path.name.removeprefix("step-"))
            for path in adapters.glob("step-*")
            if (path / "adapter_config.json").exists()
        ]
        if steps:
            self.step = max(steps)
            await self._switch(f"step-{self.step}")

    async def update(self, sequences: list[Any], *, seed: int) -> dict[str, float]:
        started = time.monotonic()
        adapters = self.settings.directory / "adapters"
        name = f"step-{self.step + 1}"
        previous = adapters / f"step-{self.step}" if self.step else None
        await self.gate.pause()  # no request may be in flight when the engine goes to sleep
        waited = time.monotonic() - started
        try:
            await self.engine.sleep()
            try:
                require_memory(self.settings.update_memory_gib, "to train")
                metrics = await self.trainer.step(sequences, seed=seed, adapter=adapters / name, previous=previous)
            finally:
                await self.engine.wake()
            self.step += 1
            await self._switch(name)
        finally:
            self.gate.resume()
        metrics["waited_for_requests_seconds"] = waited
        metrics["update_seconds"] = time.monotonic() - started
        return {key: round(value, 5) for key, value in metrics.items()}

    async def _switch(self, name: str) -> None:
        """Load an adapter and sample from it from now on; keep the one before, drop the one before that."""
        await self.engine.load_adapter(name, str(self.settings.directory / "adapters" / name))
        self.channel.adapter, self.channel.adapter_version = name, self.step
        self.loaded.append(name)
        while len(self.loaded) > 2:
            await self.engine.remove_adapter(self.loaded.pop(0))


class NotEnoughMemory(RuntimeError):
    """Stopping is better than exhausting the machine (WSL shuts down rather than killing one process)."""


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return math.inf


def require_memory(gib: float, purpose: str) -> None:
    available = available_memory_gib()
    if available < gib:
        raise NotEnoughMemory(f"{available:.1f} GiB of system memory is available; {gib:.0f} GiB is needed {purpose}")


def _reward(handle: LocalRunHandle) -> float:
    for event in handle.recorded_events():
        if event.type is RunEventType.REWARD_ASSIGNED and isinstance(event.payload, dict):
            value = event.payload.get("value")
            if isinstance(value, int | float):
                return float(value)
    return 0.0


def _result(handle: LocalRunHandle) -> dict[str, Any]:
    for event in handle.recorded_events():
        if event.type is RunEventType.OUTPUT_EMITTED and isinstance(event.payload, dict):
            payload = event.payload.get("payload")
            if isinstance(payload, dict):
                return dict(payload)
    return {}


def _transcript(path: Path, recorder: Recorder, run_id: str, tokenizer: Tokenizer) -> None:
    """One agent's turns: the end of each prompt (its latest observation) and what it sampled."""
    parts: list[str] = []
    for index, turn in enumerate(recorder.sessions_of(run_id).get(TEAM[0], []), start=1):
        observation = tokenizer.decode(list(turn.prompt[-600:]))
        sampled = tokenizer.decode(turn.completion)
        parts.append(
            f"===== turn {index} (adapter {turn.adapter}) =====\n...{observation}\n----- sampled -----\n{sampled}\n"
        )
    path.write_text("\n".join(parts))


def _append(path: Path, line: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(line, default=str) + "\n")
