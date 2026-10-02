"""The training loop: a curriculum over a catalog's rows, groups of episodes, a step per group, weights published.

It is written against `Jobs`, `Trainer` and `Store` only: the same loop runs with everything in one process and
with the runs, the engines and the trainer on machines of their own. It never waits for all running episodes: two
groups are kept outstanding, the job starts the second as soon as there is room beside what is left of the first,
and each group is trained on when its last episode ends, while the next group's episodes run on (their tokens then
carry two weights versions, which the clipped objective corrects for).

What it writes (to the store): `metrics.jsonl`, one line per group; `curriculum.json`. The same lines go to the job
as `iteration` notes, for whoever watches. A loop started again over the same store goes on after the last group
logged, on starts drawn anew.
"""

import asyncio
import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.core.harness.runner import RunBinding, bind, with_row
from rollout.rollouts import Catalog, Episode, Jobs, Row
from rollout.training.algorithm import Grpo, fastest_of_the_saturated
from rollout.training.curriculum import Curriculum
from rollout.training.trainer import Trainer

FAILED_UPDATES = 3
"""Updates that may fail in a row (each is logged, and the weights stay as they were) before the loop stops."""


class Store(Protocol):
    """Where a training run keeps its small state: a directory, or anything else that holds named texts."""

    def read(self, name: str) -> str | None: ...
    def write(self, name: str, text: str) -> None: ...
    def append(self, name: str, line: str) -> None: ...


@dataclass(frozen=True)
class Directory:
    """A `Store` in a directory."""

    path: Path

    def read(self, name: str) -> str | None:
        file = self.path / name
        return file.read_text() if file.exists() else None

    def write(self, name: str, text: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / name).write_text(text)

    def append(self, name: str, line: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / name).open("a") as file:
            file.write(line + "\n")


def iterations(store: Store) -> list[dict[str, Any]]:
    """The groups a run has logged, in the order they were logged."""
    return [json.loads(line) for line in (store.read("metrics.jsonl") or "").splitlines() if line.strip()]


@dataclass
class _Group:
    number: int
    row: Row
    started: float


async def train(
    jobs: Jobs,
    catalog: Catalog,
    trainer: Trainer,
    store: Store,
    *,
    channel: str = "policy",
    algorithm: Grpo | None = None,
    groups: int = 100,
    overlap: int = 1,
    seed: int = 0,
    binding: RunBinding | None = None,
    curriculum: Curriculum | None = None,
) -> None:
    """Train `channel`'s policy on `catalog` for `groups` more groups. `overlap`: the next group starts when at most
    this many episodes of earlier groups are still running. `binding` says how the program's model slots and imports
    are served (by default: every slot from `channel`, each import from the tool set of its own name)."""
    algorithm = algorithm or Grpo()
    done = max((int(line["iteration"]) for line in iterations(store)), default=0)
    rng = random.Random(f"{seed}-{done}")  # a run that is started again draws new starts, not the same ones
    curriculum = curriculum or Curriculum(catalog.rows(), rng)
    if saved := store.read("curriculum.json"):
        curriculum.restore(json.loads(saved))
    if binding is None:  # (a program says which slots and imports it has once it is given a row)
        binding = bind(with_row(catalog.program, catalog.start(catalog.rows()[0], random.Random(0))), channel)
    job = await jobs.start(
        program=catalog.program,
        binding=binding,
        in_flight=algorithm.group_size + overlap,
        name="train",
    )
    # A job that is found again may hold episodes of groups a stopped loop never finished: they are groups no more.
    consumed = (await job.status()).finished
    await job.acknowledge(consumed)
    outstanding: dict[asyncio.Task[list[Episode]], _Group] = {}
    number, failed_updates = done, 0
    read: set[int] = set()

    async def submit() -> None:
        nonlocal number
        number += 1
        row = curriculum.sample([group.row.key for group in outstanding.values()])
        labels = {"group": f"{number:04d}", "iteration": str(number), "task": row.key, "title": row.title}
        ticket = await job.run(catalog.start(row, rng), labels=labels, count=algorithm.group_size)
        outstanding[asyncio.create_task(ticket.episodes())] = _Group(number, row, time.time())

    try:
        while outstanding or number < done + groups:
            while len(outstanding) < 2 and number < done + groups:
                await submit()
            finished, _ = await asyncio.wait(outstanding, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                group, episodes = outstanding.pop(task), task.result()
                line: dict[str, JsonValue] = {
                    "iteration": group.number,
                    "time": round(time.time(), 1),
                    "task": group.row.key,
                    "title": group.row.title,
                    "rollout_seconds": round(time.time() - group.started, 1),
                }
                good = [episode for episode in episodes if episode.trainable]
                line |= _results(episodes, good)
                if good:
                    curriculum.update(group.row, [e.reward for e in good], [bool(e.info.get("solved")) for e in good])
                else:
                    curriculum.failed(group.row)
                bonus = fastest_of_the_saturated(good) if algorithm.tie_break else []
                if any(bonus):
                    line["speed_bonus"] = list(bonus)
                batch = algorithm.batch(good, trainer.budget, random.Random(group.number))
                if batch is None:
                    line["update"] = (
                        "skipped: every episode scored the same"
                        if len(good) >= 2
                        else f"skipped: {len(good)} of {len(episodes)} episodes completed"
                    )
                else:
                    line["sequences_recorded"] = sum(len(t.epochs) for e in good for t in e.traces.values())
                    line["sequences_trained"] = len(batch)
                    try:
                        step = await trainer.step(batch, seed=group.number)
                        line["version"] = await job.publish(channel, step.adapter, step.path)
                        line["adapter"] = step.adapter
                        line["update"] = {key: round(value, 5) for key, value in step.metrics.items()}
                        failed_updates = 0
                    except RuntimeError as error:  # the trainer failed: the weights stay, and the run goes on
                        failed_updates += 1
                        line["update"] = f"failed: {str(error).strip().splitlines()[-1][:300]}"
                        if failed_updates >= FAILED_UPDATES:
                            raise
                line["seconds"] = round(time.time() - group.started, 1)
                line["unlocked"] = len(curriculum.unlocked())
                store.append("metrics.jsonl", json.dumps(line))
                store.write("curriculum.json", json.dumps(curriculum.saved(), indent=1))
                await job.note("iteration", line)
                read.update(episode.cursor for episode in episodes)
                while consumed + 1 in read:  # everything through here has been trained on or set aside
                    consumed += 1
                await job.acknowledge(consumed)
    finally:
        for task in outstanding:
            task.cancel()


def _results(episodes: list[Episode], good: list[Episode]) -> Mapping[str, JsonValue]:
    """A group's episodes as its line of metrics tells them."""
    failed = [episode for episode in episodes if not episode.trainable]
    return {
        "rewards": [episode.reward for episode in good],
        "solved": [bool(episode.info.get("solved")) for episode in good],
        "durations": [episode.info.get("duration") for episode in good],
        "failed": len(failed),
        "failures": [str(episode.detail or episode.excluded or episode.outcome.value) for episode in failed],
    }
