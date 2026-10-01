"""Train the swarm: curriculum → group of episodes → group-relative update → new adapter, on one GPU.

Each iteration the curriculum picks a task; a group of episodes runs it concurrently from identical starts (same
world seed, same layout seed), with the four agents of every episode sampling from one recorded channel. Each
episode's reward is its task's objective, scored from ground truth. If the group's rewards differ, the engine sleeps,
a trainer process takes one step on a sample of the group's turns and exits, and the new LoRA adapter is loaded into
the engine once it is awake again.

Memory is checked before each group and each update: the run stops with `NotEnoughMemory` rather than exhaust the
machine.

Outputs (in `directory`): `metrics.jsonl` (one line per iteration), `episodes.jsonl` (each episode's ground truth),
`transcripts/` (one agent's turns per iteration), `adapters/step-N/` (PEFT adapters), `trainer/` (the optimizer's
state), `curriculum.json`.
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
from minecraft_swarm.tasks import catalog
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
from rollout.recorder import Channel, Recorder, renderer_for
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
    update_turns: int = 384
    """At most this many agent turns per update, sampled evenly from the group's (long episodes have thousands)."""
    thinking_budget: int = 384
    answer_tokens: int = 256
    temperature: float = 1.0
    learning_rate: float = 2e-5
    lora_rank: int = 32
    window_ticks: int = 100
    gpu_memory_utilization: float = 0.72
    seed: int = 0
    tasks: list[str] | None = None
    """Restrict the curriculum to these task ids (all tasks when None)."""
    group_memory_gib: float = 6.0
    """System memory that must be available to start a group of episodes (each runs a Paper server)."""
    update_memory_gib: float = 4.0
    """System memory that must be available to train, once the engine has gone to sleep."""
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
        max_num_seqs=4 * settings.group_size,
        max_lora_rank=settings.lora_rank,
    )
    channel = Channel(engine, renderer, thinking_budget=settings.thinking_budget, answer_tokens=settings.answer_tokens)
    recorder = Recorder({"policy": channel})
    worlds = MinecraftWorlds(window_ticks=settings.window_ticks, logs=directory / "logs")
    (directory / "logs").mkdir(exist_ok=True)
    runner = LocalRunner(recorder=recorder, tool_sets={"minecraft": MinecraftTools(worlds)})
    tasks = [task for task in catalog() if settings.tasks is None or task.id in settings.tasks]
    curriculum = Curriculum(tasks, rng, start=len(tasks) if settings.tasks else 3)
    if (directory / "curriculum.json").exists():
        curriculum.load(directory / "curriculum.json")
    sampling = SamplingParameters(temperature=settings.temperature)
    binding = RunBinding(
        models={name: ModelBinding(recorded=RecordedModel(channel="policy", sampling=sampling)) for name in TEAM},
        imports={"minecraft": ToolBinding(local="minecraft")},
    )
    learner = Learner(settings, engine, channel)
    try:
        for iteration in range(1, settings.iterations + 1):
            started = time.monotonic()
            require_memory(settings.group_memory_gib, "to run a group of episodes")
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
            handles = await _run_group(runner, binding, parameters, settings.group_size)
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
            line: dict[str, Any] = {
                "iteration": iteration,
                "task": task.id,
                "title": task.title,
                "difficulty": task.difficulty,
                "world_seed": world_seed,
                "minutes": minutes,
                "game_minutes": [round(float(result.get("game_minutes", 0)), 1) for result in results],
                "turns": [result.get("turns") for result in results],
                "rewards": rewards,
                "solved": [bool(result.get("solved")) for result in results],
                "failed": len(handles) - len(completed),
                "failures": [handle.outcome.detail for handle in handles if handle not in completed and handle.outcome],
                "adapter_step": learner.step,
                "rollout_seconds": round(time.monotonic() - started, 1),
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
                recorded = [
                    (turn, advantage)
                    for handle, advantage in zip(completed, advantages, strict=True)
                    for turns_of_agent in recorder.sessions_of(handle.run_id).values()
                    for turn in turns_of_agent
                ]
                line["turns_recorded"] = len(recorded)
                if len(recorded) > settings.update_turns:  # an even sample: every episode and agent keeps its share
                    recorded = random.Random(iteration).sample(recorded, settings.update_turns)
                sequences = [  # built only for the turns trained on: a long episode records tens of thousands
                    TrainingSequence(
                        tokens=[*turn.prompt, *turn.completion],
                        loss_mask=[False] * len(turn.prompt) + turn.loss_mask,
                        behavior_logprobs=[math.nan] * len(turn.prompt) + turn.logprobs,
                        advantage=advantage,
                    )
                    for turn, advantage in recorded
                ]
                line["update"] = await learner.update(sequences, seed=iteration)
                line["adapter_step"] = learner.step
            for handle in handles:
                recorder.forget(handle.run_id)
            line["seconds"] = round(time.monotonic() - started, 1)
            line["unlocked"] = len(curriculum.unlocked())
            _append(directory / "metrics.jsonl", line)
            curriculum.save(directory / "curriculum.json")
            print(json.dumps(line), flush=True)
    finally:
        await worlds.close()
        engine.close()


class Learner:
    """Takes turns with the engine on the GPU: sleep it, train one step in a fresh trainer process (which exits, freeing
    the GPU and its memory), wake it, and switch the channel to the new adapter."""

    def __init__(self, settings: TrainingSettings, engine: VllmEngine, channel: Channel) -> None:
        from rollout.training.worker import TrainerProcess, TrainerSettings

        self.settings = settings
        self.engine = engine
        self.channel = channel
        self.step = 0
        self.trainer = TrainerProcess(
            TrainerSettings(
                settings.model,
                state=settings.directory / "trainer",
                rank=settings.lora_rank,
                alpha=2.0 * settings.lora_rank,
                learning_rate=settings.learning_rate,
            )
        )

    async def update(self, sequences: list[Any], *, seed: int) -> dict[str, float]:
        started = time.monotonic()
        adapters = self.settings.directory / "adapters"
        name = f"step-{self.step + 1}"
        previous = adapters / f"step-{self.step}" if self.step else None
        await self.engine.sleep()
        slept = time.monotonic() - started
        try:
            require_memory(self.settings.update_memory_gib, "to train (the engine's weights are in system memory)")
            metrics = await self.trainer.step(sequences, seed=seed, adapter=adapters / name, previous=previous)
        finally:
            await self.engine.wake()
        self.step += 1
        await self.engine.load_adapter(name, str(adapters / name))
        retired = self.channel.adapter
        self.channel.adapter, self.channel.adapter_version = name, self.step
        if retired is not None:
            await self.engine.remove_adapter(retired)
        metrics["engine_sleep_seconds"] = slept
        metrics["update_seconds"] = time.monotonic() - started
        return {key: round(value, 5) for key, value in metrics.items()}


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


async def _run_group(
    runner: LocalRunner, binding: RunBinding, parameters: dict[str, JsonValue], size: int
) -> list[LocalRunHandle]:
    """Episodes of one task from identical starts (the same world and layout), run at once."""
    specification = RunSpecification(
        program=ProgramReference(program=register(SwarmEpisode), parameters=parameters), binding=binding
    )
    handles = [await runner.start(specification) for _ in range(size)]
    await asyncio.gather(*(handle.result() for handle in handles))
    return handles


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
