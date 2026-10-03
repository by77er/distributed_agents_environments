"""The training loop: a curriculum over a catalog's rows, groups of episodes, a step per group, weights published.

It is written against `Jobs`, `Trainer`, `Algorithm` and `Policies` only: the same loop runs with everything in one
process and with the runs, the engines and the trainer on machines of their own. It never waits for all running
episodes: `OUTSTANDING` groups are kept asked for, the job starts the next as soon as there is room beside what is
left of the one before, and each group is trained on when its last episode ends, while the next group's episodes
run on (their tokens then carry two versions, which the trainer's objective corrects for).

**It can die at any moment and be started again.** It keeps nothing it cannot read back: what it decides and what
happens are appended to the run's tables in the ledger (`rollout_train.record`), each under the group's number, and
every action is one that can be taken twice.

| It died | Started again, it |
|---|---|
| after deciding a group | asks for that group again, under the same key: the job gives back the ticket it has |
| while a group played | waits for the episodes the job still owes; the others are in the job's log |
| after a group ended | finds the group has no outcome, and takes it from there |
| during a step | finds the step decided and no version made, and takes the step again from the same parent |
| after the step | finds the version, and goes on to serve it and to write the group's outcome |

A step that was decided is finished before any other is decided: a decision names the version it will make, and
only one group may make it.

Taking the run's fence and the policy's when it starts shuts out a loop it replaced: that one's next write is
refused.
"""

import asyncio
import json
import random
import shutil
import time
from pathlib import Path

from pydantic import JsonValue

from rollout.catalog import Catalog, binding_for
from rollout.contracts import BlobReference
from rollout.harness.runner import RunBinding
from rollout_train.algorithm import Algorithm, Batch, Grpo
from rollout_train.curriculum import Curriculum
from rollout_train.policies import Policies, Version, named
from rollout_train.record import GROUPS, ITERATIONS, STEPS, Iteration, scope, table
from rollout_train.rollouts import Episode, Jobs
from rollout_train.trainer import STATE, WEIGHTS, Checkpoint, StepFailed, Trainer

OUTSTANDING = 2
"""Groups kept asked for: one playing, and one that starts as the first one's last episodes end."""
FAILED_UPDATES = 3
"""Steps that may fail in a row (each is logged, and the weights stay as they were) before the loop stops."""


async def train(
    jobs: Jobs,
    catalog: Catalog,
    trainer: Trainer,
    policies: Policies,
    *,
    policy: str,
    channel: str,
    directory: Path,
    run: str = "train",
    algorithm: Algorithm | None = None,
    groups: int = 100,
    overlap: int = 1,
    seed: int = 0,
    binding: RunBinding | None = None,
    curriculum: Curriculum | None = None,
) -> None:
    """Train `policy` on `catalog` until `groups` more groups are done with (those a stopped loop left unfinished
    among them), serving it on `channel`. `directory` is where versions' files are kept on this machine while they
    are in use: the one being served and the one before it (a turn in progress finishes under the weights it began
    with); every version's files are in the blob store. `algorithm` is `Grpo()` unless given. `overlap`: the next
    group starts when at most this many episodes of earlier groups are still running. `binding` says how the
    program's model slots and imports are served (by default: every slot from `channel`, each import from the tool
    set of its own name). `curriculum` is one that has recorded nothing: the run's iterations are folded into it."""
    algorithm = algorithm if algorithm is not None else Grpo()
    ledger, blobs = policies.ledger, policies.blobs
    fence = await ledger.take(scope(run))  # whoever ran this before can no longer write
    writer = await policies.writer(policy)
    decided = {int(number): _mapping(group) for number, group in (await ledger.read(table(run, GROUPS))).items()}
    logged = await ledger.read(table(run, ITERATIONS))
    curriculum = curriculum or Curriculum(catalog.rows())
    for number in sorted(logged, key=int):
        curriculum.recorded(Iteration.from_json(_mapping(logged[number])))
    job = await jobs.start(
        program=catalog.program,
        binding=binding or binding_for(catalog, channel),
        in_flight=algorithm.group_size + overlap,
        name=run,
    )

    async def files(version: Version) -> Checkpoint:
        """A version's files on this machine, read from the blob store if they are not here."""
        here = directory / version.name
        weights = await policies.files(version.weights, here / WEIGHTS)
        return Checkpoint(weights, await policies.files(version.state, here / STATE) if version.state else None)

    async def serve(version: Version) -> None:
        nonlocal served
        if version.number <= served:
            return  # (a group whose step was taken before a newer one: the channel does not go back)
        await job.publish(channel, version.name, str((await files(version)).weights), version.number)
        served = version.number
        keep = {version.name, version.parent}
        for old in await asyncio.to_thread(lambda: [each for each in directory.iterdir() if each.name not in keep]):
            await asyncio.to_thread(shutil.rmtree, old, ignore_errors=True)

    served = 0
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    head = await policies.head(policy)
    if head is not None:
        await serve(head)  # (a loop that died between making a version and serving it serves it now)

    outstanding: dict[asyncio.Task[list[Episode]], int] = {}

    async def ask(number: int) -> None:
        group = decided[number]
        labels = {
            "group": f"{number:04d}",
            "iteration": str(number),
            "task": str(group["task"]),
            "title": str(group["title"]),
        }
        ticket = await job.run(
            group["parameters"], labels=labels, count=algorithm.group_size, key=f"{run}-{number:04d}"
        )
        outstanding[asyncio.create_task(ticket.episodes())] = number

    async def decide() -> None:
        number = max(decided, default=0) + 1
        rng = random.Random(f"{seed}-{number}")
        row = curriculum.sample([str(decided[each]["task"]) for each in outstanding.values()], rng)
        group: dict[str, JsonValue] = {
            "task": row.key,
            "title": row.title,
            "parameters": catalog.start(row, rng),
            "decided": round(time.time(), 1),
        }
        await ledger.append(table(run, GROUPS), str(number), group, fence)  # before it is asked for
        decided[number] = group
        await ask(number)

    async def stepped(number: int, batch: Batch) -> Version:
        """The version the group's step makes: decided, then made, then recorded, each once."""
        steps = await ledger.read(table(run, STEPS))
        if str(number) not in steps:
            parent = await policies.head(policy)
            trained = [[weighted.source, weighted.advantage] for weighted in batch.segments]
            manifest = await blobs.put(json.dumps(trained).encode(), "application/json")
            decision: dict[str, JsonValue] = {
                "policy": policy,
                "parent": parent.name if parent else None,
                "number": (parent.number if parent else 0) + 1,
                "segments": len(trained),
                "decided": round(time.time(), 1),
                "batch": manifest.model_dump(mode="json"),
                "seed": number,
            }
            await ledger.append(table(run, STEPS), str(number), decision, fence)  # before the trainer is called
            steps = {**steps, str(number): decision}
        intent = _mapping(steps[str(number)])
        made = int(str(intent["number"]))
        try:
            return await policies.version(named(policy, made))  # made before this loop died: not taken again
        except KeyError:
            pass
        parent_name = str(intent["parent"]) if intent["parent"] else None
        start = await files(await policies.version(parent_name)) if parent_name else None
        into = directory / named(policy, made)
        await asyncio.to_thread(shutil.rmtree, into, ignore_errors=True)  # (what a step that died left)
        step = await trainer.step(batch.segments, seed=int(str(intent["seed"])), parent=start, into=into)
        return await policies.add(
            writer,
            policy,
            made,
            weights=into / WEIGHTS,
            state=into / STATE if await asyncio.to_thread((into / STATE).exists) else None,
            parent=parent_name,
            batch=BlobReference.model_validate(intent["batch"]),
            metrics=step.metrics,
        )

    async def finish(number: int, episodes: list[Episode]) -> None:
        nonlocal failed_updates
        group = decided[number]
        began = float(str(group.get("decided", time.time())))
        good = [episode for episode in episodes if episode.trainable]
        failed = [episode for episode in episodes if not episode.trainable]
        line = Iteration(
            iteration=number,
            time=0.0,
            task=str(group["task"]),
            title=str(group["title"]),
            rollout_seconds=round(time.time() - began, 1),
            rewards=[episode.reward for episode in good],
            solved=[episode.solved for episode in good],
            durations=[episode.duration for episode in good],
            failed=len(failed),
            failures=[str(episode.detail or episode.excluded or episode.outcome.value) for episode in failed],
        )
        curriculum.recorded(line)  # (the curriculum sees the task's own rewards, whatever the algorithm adds)
        batch = algorithm.batch(episodes, trainer.budget, random.Random(number))
        line.notes, line.skipped = batch.notes, batch.skipped
        if batch.segments:
            line.segments_recorded = sum(
                len(trajectory.segments) for episode in good for trajectory in episode.trajectories.values()
            )
            line.segments_trained = len(batch.segments)
            try:
                version = await stepped(number, batch)
            except StepFailed as error:  # the weights stay as they were, and the run goes on
                failed_updates += 1
                line.error = str(error).strip().splitlines()[-1][:300]
                if failed_updates >= FAILED_UPDATES:
                    raise
            else:
                failed_updates = 0
                await serve(version)
                line.adapter, line.version = version.name, version.number
                line.update = {key: round(value, 5) for key, value in version.metrics.items()}
        line.time = round(time.time(), 1)
        line.seconds = round(line.time - began, 1)
        line.unlocked = len(curriculum.unlocked())
        await ledger.append(table(run, ITERATIONS), str(number), line.to_json(), fence)  # the group is done with
        await job.note("iteration", line.to_json())
        await job.acknowledge(max((episode.cursor for episode in episodes), default=0))

    failed_updates = 0
    try:
        for number in sorted(decided):
            if str(number) not in logged:
                await ask(number)  # what a stopped loop left unfinished
        steps = await ledger.read(table(run, STEPS))
        begun = sorted(
            (number for number in outstanding.values() if str(number) in steps),
            key=lambda number: int(str(_mapping(steps[str(number)])["number"])),
        )
        for number in begun:  # a step that was decided is finished before another is decided
            task = next(task for task, each in outstanding.items() if each == number)
            episodes = await task
            del outstanding[task]
            await finish(number, episodes)
        owed = groups - len(outstanding) - len(begun)
        while outstanding or owed > 0:
            while len(outstanding) < OUTSTANDING and owed > 0:
                await decide()
                owed -= 1
            finished, _ = await asyncio.wait(outstanding, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(finished, key=lambda each: outstanding[each]):
                await finish(outstanding.pop(task), task.result())
    finally:
        for task in outstanding:
            task.cancel()


def _mapping(record: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(record, dict)
    return record
