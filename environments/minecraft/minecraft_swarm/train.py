"""Train the swarm: curriculum → group of episodes → group-relative update → new adapter, on one GPU.

Each iteration the curriculum picks a task; a group of episodes runs it concurrently from identical starts (same
world seed, same layout seed), with the four agents of every episode sampling from one recorded channel. The team's
diamonds are each episode's reward. If the group's rewards differ, the engine sleeps, the trainer takes one step on
every agent's turns, and the new LoRA adapter is loaded into the engine before it wakes.

Outputs (in `directory`): `metrics.jsonl` (one line per iteration), `episodes.jsonl` (each episode's ground truth),
`transcripts/` (one agent's turns per iteration), `adapters/step-N/` (PEFT adapters), `curriculum.json`.
"""

import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
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
    world_seeds: list[int] = field(default_factory=lambda: [12345, 2024, 777])
    max_turns: int = 16
    """Caps each task's turns (episodes cost time in proportion)."""
    thinking_budget: int = 384
    answer_tokens: int = 256
    temperature: float = 1.0
    learning_rate: float = 2e-5
    lora_rank: int = 32
    window_ticks: int = 100
    gpu_memory_utilization: float = 0.72
    seed: int = 0


async def train(settings: TrainingSettings) -> None:
    from transformers import AutoTokenizer

    from rollout.training.grpo import GroupRelativeTrainer, TrainingSequence, group_advantages
    from rollout.training.policy import Policy

    directory = settings.directory
    (directory / "transcripts").mkdir(parents=True, exist_ok=True)
    (directory / "adapters").mkdir(exist_ok=True)
    (directory / "settings.json").write_text(json.dumps(asdict(settings), default=str, indent=1))
    rng = random.Random(settings.seed)
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
    curriculum = Curriculum(catalog(), rng)
    if (directory / "curriculum.json").exists():
        curriculum.load(directory / "curriculum.json")
    sampling = SamplingParameters(temperature=settings.temperature)
    binding = RunBinding(
        models={name: ModelBinding(recorded=RecordedModel(channel="policy", sampling=sampling)) for name in TEAM},
        imports={"minecraft": ToolBinding(local="minecraft")},
    )
    trainer: GroupRelativeTrainer | None = None
    policy: Policy | None = None
    step = 0
    try:
        for iteration in range(1, settings.iterations + 1):
            started = time.monotonic()
            task = curriculum.sample()
            world_seed, layout_seed = rng.choice(settings.world_seeds), rng.randrange(1 << 30)
            turns = min(task.turns, settings.max_turns)
            handles = await _run_group(runner, binding, task, world_seed, layout_seed, turns, settings.group_size)
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
                "turns": turns,
                "rewards": rewards,
                "failed": len(handles) - len(completed),
                "adapter_step": step,
                "rollout_seconds": round(time.monotonic() - started, 1),
            }
            if rewards:
                curriculum.update(task, rewards)
            advantages = group_advantages(rewards) if len(rewards) >= 2 else None
            if advantages is None:
                line["update"] = "skipped: every episode scored the same"
            else:
                sequences = [
                    TrainingSequence(
                        tokens=[*turn.prompt, *turn.completion],
                        loss_mask=[False] * len(turn.prompt) + turn.loss_mask,
                        behavior_logprobs=[math.nan] * len(turn.prompt) + turn.logprobs,
                        advantage=advantage,
                    )
                    for handle, advantage in zip(completed, advantages, strict=True)
                    for turns_of_agent in recorder.sessions_of(handle.run_id).values()
                    for turn in turns_of_agent
                ]
                await engine.sleep()
                if trainer is None or policy is None:  # loaded once, on the first update; kept on the CPU between
                    policy = Policy.load(settings.model, rank=settings.lora_rank, alpha=2.0 * settings.lora_rank)
                    trainer = GroupRelativeTrainer(policy, learning_rate=settings.learning_rate)
                else:
                    trainer.to("cuda")
                metrics = trainer.step(sequences, seed=iteration)
                step += 1
                name = f"step-{step}"
                adapter = policy.save(directory / "adapters" / name)
                trainer.to("cpu")
                await engine.wake()
                await engine.load_adapter(name, str(adapter))
                previous = channel.adapter
                channel.adapter, channel.adapter_version = name, step
                if previous is not None:
                    await engine.remove_adapter(previous)
                line["update"] = {key: round(value, 5) for key, value in metrics.items()}
                line["adapter_step"] = step
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


async def _run_group(
    runner: LocalRunner,
    binding: RunBinding,
    task: Task,
    world_seed: int,
    layout_seed: int,
    turns: int,
    size: int,
) -> list[LocalRunHandle]:
    parameters: dict[str, JsonValue] = {
        "task": task.id,
        "world_seed": world_seed,
        "layout_seed": layout_seed,
        "turns": turns,
    }
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
        observation = tokenizer.decode(turn.prompt[-600:])
        sampled = tokenizer.decode(turn.completion)
        parts.append(
            f"===== turn {index} (adapter {turn.adapter}) =====\n...{observation}\n----- sampled -----\n{sampled}\n"
        )
    path.write_text("\n".join(parts))


def _append(path: Path, line: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(line, default=str) + "\n")
