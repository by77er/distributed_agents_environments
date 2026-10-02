"""The training loop: a curriculum over a catalog's rows, groups of episodes, a step per group, weights published.

It is written against `Jobs`, `Trainer`, `Algorithm` and `Store` only: the same loop runs with everything in one
process and with the runs, the engines and the trainer on machines of their own. It never waits for all running
episodes: `OUTSTANDING` groups are kept submitted, the job starts the next as soon as there is room beside what is
left of the one before, and each group is trained on when its last episode ends, while the next group's episodes
run on (their tokens then carry two weights versions, which the trainer's objective corrects for).

What it writes (to the store): `metrics.jsonl`, one `Iteration` per group; `curriculum.json`. The same lines go to
the job as `iteration` notes, for whoever watches. A loop started again over the same store goes on after the last
group logged, on starts drawn anew.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass

from rollout.catalog import Catalog, Row, binding_for
from rollout.harness.runner import RunBinding
from rollout_train.algorithm import Algorithm, Grpo
from rollout_train.curriculum import Curriculum
from rollout_train.record import METRICS, Iteration, Store, iterations
from rollout_train.rollouts import Episode, Jobs
from rollout_train.trainer import StepFailed, Trainer

OUTSTANDING = 2
"""Groups kept submitted: one playing, and one that starts as the first one's last episodes end."""
FAILED_UPDATES = 3
"""Steps that may fail in a row (each is logged, and the weights stay as they were) before the loop stops."""


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
    channel: str,
    algorithm: Algorithm | None = None,
    groups: int = 100,
    overlap: int = 1,
    seed: int = 0,
    binding: RunBinding | None = None,
    curriculum: Curriculum | None = None,
) -> None:
    """Train `channel`'s policy on `catalog` for `groups` more groups. `algorithm` is `Grpo()` unless given.
    `overlap`: the next group starts when at most this many episodes of earlier groups are still running. `binding`
    says how the program's model slots and imports are served (by default: every slot from `channel`, each import
    from the tool set of its own name)."""
    algorithm = algorithm if algorithm is not None else Grpo()
    done = max((line.iteration for line in iterations(store)), default=0)
    rng = random.Random(f"{seed}-{done}")  # a run that is started again draws new starts, not the same ones
    curriculum = curriculum or Curriculum(catalog.rows(), rng)
    if saved := store.read("curriculum.json"):
        curriculum.restore(json.loads(saved))
    job = await jobs.start(
        program=catalog.program,
        binding=binding or binding_for(catalog, channel),
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
            while len(outstanding) < OUTSTANDING and number < done + groups:
                await submit()
            finished, _ = await asyncio.wait(outstanding, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                group, episodes = outstanding.pop(task), task.result()
                good = [episode for episode in episodes if episode.trainable]
                failed = [episode for episode in episodes if not episode.trainable]
                line = Iteration(
                    iteration=group.number,
                    time=0.0,
                    task=group.row.key,
                    title=group.row.title,
                    rollout_seconds=round(time.time() - group.started, 1),
                    rewards=[episode.reward for episode in good],
                    solved=[episode.solved for episode in good],
                    durations=[episode.duration for episode in good],
                    failed=len(failed),
                    failures=[str(episode.detail or episode.excluded or episode.outcome.value) for episode in failed],
                )
                if good:  # the curriculum sees the task's own rewards, whatever the algorithm adds to them
                    curriculum.update(group.row, line.rewards, line.solved)
                else:
                    curriculum.failed(group.row)
                batch = algorithm.batch(episodes, trainer.budget, random.Random(group.number))
                line.notes, line.skipped = batch.notes, batch.skipped
                if batch.sequences:
                    line.sequences_recorded = sum(len(t.epochs) for e in good for t in e.traces.values())
                    line.sequences_trained = len(batch.sequences)
                    try:
                        step = await trainer.step(batch.sequences, seed=group.number)
                    except StepFailed as error:  # the weights stay as they were, and the run goes on
                        failed_updates += 1
                        line.error = str(error).strip().splitlines()[-1][:300]
                        if failed_updates >= FAILED_UPDATES:
                            raise
                    else:
                        failed_updates = 0
                        line.version = await job.publish(channel, step.adapter, step.path)
                        line.adapter = step.adapter
                        line.update = {key: round(value, 5) for key, value in step.metrics.items()}
                line.time = round(time.time(), 1)
                line.seconds = round(line.time - group.started, 1)
                line.unlocked = len(curriculum.unlocked())
                store.append(METRICS, json.dumps(line.to_json()))
                store.write("curriculum.json", json.dumps(curriculum.saved(), indent=1))
                await job.note("iteration", line.to_json())
                read.update(episode.cursor for episode in episodes)
                while consumed + 1 in read:  # everything through here has been trained on or set aside
                    consumed += 1
                await job.acknowledge(consumed)
    finally:
        for task in outstanding:
            task.cancel()
