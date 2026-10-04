"""The training loop: a curriculum over an environment's rows, groups of episodes, steps over the groups played,
checkpoints.

It is written against the ledger, a `Trainer`, an `Algorithm` and `Checkpoints` only: it asks for each group's episodes
in the ledger, and runners, wherever they are, play them (`rollout_train.rollouts.scheduler`). The same loop runs with
everything in one process and with the runners, the engines and the trainer on machines of their own.

**Play and training go their own ways.** Enough groups are kept asked for that `episodes_at_once` episodes have work
waiting, whatever groups they are of. When a group's last episode ends its result is written down at once, and what the
algorithm finds to train on in it joins a queue. A step is taken over every group queued once there are at least
`groups_per_step` (so that no step leans toward one task), while play goes on; at the end of the run, over whatever is
left. Tokens sampled under an older checkpoint than the one a step starts from are corrected for by the trainer's
objective. One step is taken at a time, each from the checkpoint the one before made; the first from the checkpoint the
run starts from (`start`: any checkpoint, of this run or another; the base model if none).

**It can die at any moment and be started again.** It keeps nothing it cannot read back: what it decides and what
happens are appended to the run's tables in the ledger (`rollout_train.record`), and every action is one that can be
taken twice.

| It died | Started again, it |
|---|---|
| after deciding a group, or while it played | waits for its episodes: runners play them (again, any a runner dropped) |
| after a group ended | finds no result, and writes it |
| with groups queued | finds results to train on that no step covers, and queues them again |
| during a step | finds the step decided and no checkpoint made, and takes it again over the same groups |
| after the step | finds the checkpoint, serves it, and goes on |

Taking the run's fence when it starts shuts out a loop it replaced: that one's next write is refused. The checkpoints it
makes are appended under that fence.

**It can evaluate its checkpoints as it makes them** (`evals`, a `rollout_train.evals.Schedule`). After a step whose
checkpoint the schedule names is served, the suite is asked for as an eval of that checkpoint, a run of its own, and
the next step waits until every start has been played: all that time the channel serves that checkpoint, while
training groups go on being played under it. The eval's results are folded into the curriculum. Started again, the
loop finishes an eval it left unfinished before it decides anything.
"""

import asyncio
import json
import random
import shutil
import socket
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.environment import Environment, binding_for
from rollout.harness.runner import RunBinding
from rollout_train.algorithm import Algorithm, Grpo, spread
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest, Retention, new_id
from rollout_train.curriculum import Curriculum
from rollout_train.evals import Schedule, evaluate
from rollout_train.ledger import Fence
from rollout_train.record import EVALS, FAILURES, GROUPS, RESULTS, STARTS, STEPS, Result, results, scope, table
from rollout_train.rollouts import Episode
from rollout_train.rollouts.scheduler import Hooks, Plan, episodes_of, plan
from rollout_train.trainer import STATE, WEIGHTS, Files, StepFailed, Trainer, Weighted


class Publisher(Protocol):
    """Serves new weights on a channel from now on (with `full`, a full checkpoint's in place of the engines'); returns
    the number its samples are stamped with (the checkpoint's depth)."""

    async def __call__(
        self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
    ) -> int: ...


FAILED_UPDATES = 3
"""Steps that may fail in a row (each is written down, and the weights stay as they were) before the loop stops."""


async def train(
    environment: Environment,
    trainer: Trainer,
    checkpoints: Checkpoints,
    *,
    start: str | None = None,
    base: str | None = None,
    channel: str,
    directory: Path,
    publish: Publisher,
    run: str = "train",
    algorithm: Algorithm | None = None,
    groups: int = 100,
    groups_per_step: int = 4,
    episodes_at_once: int = 6,
    seed: int = 0,
    binding: RunBinding | None = None,
    curriculum: Curriculum | None = None,
    retention: Retention | None = None,
    started: Mapping[str, JsonValue] | None = None,
    hooks: Sequence[Hooks] = (),
    kept: Callable[[], Awaitable[Collection[str]]] | None = None,
    made: Callable[[Checkpoint], Awaitable[object]] | None = None,
    reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None,
    evals: Schedule | None = None,
) -> None:
    """Train from `start` (a checkpoint's id; else the base model, named `base`) on `environment` until `groups` more
    groups have been played (those a stopped loop left unplayed among them) and every group played has been trained on,
    serving each checkpoint made on `channel`; a run started again goes on from the newest checkpoint it made. A step is
    taken over the groups queued once at least `groups_per_step` have something to train on (and, at the end, over what
    is left). `directory` is where checkpoints' files are kept on this machine while they are in use: the one being
    served and the one before it (a turn in progress finishes under the weights it began with); every checkpoint's files
    are in the blob store; `publish` serves a checkpoint on `channel`. `algorithm` is `Grpo()` unless given.
    `episodes_at_once` is how many episodes the run keeps work waiting for, whatever groups they are of (runners play
    them, as many at once as each has places). `binding` says how the program's model slots and imports are served (by
    default: every slot from `channel`, each import from the tool set of its own name). `curriculum` is one that has
    recorded nothing: the run's results are folded into it. `retention` says which of the checkpoints the run made keep
    their files (weights and trainer state) once a newer one is served (`Retention()` unless given); besides those, what
    is served, what any run starts from, and whatever `kept` says (the bookmarked checkpoints, say) keep theirs.
    `started` is what the run's `starts` record says beside what the loop knows (where it starts from, this host, the
    time): where the run's directory is, where the monitor on its machine serves (`address`), and what profile started
    it, say. `hooks` are told of each result and step; `made` is called with each checkpoint made, once it is served (to
    move a bookmark, say). `reshard` gives the files the engines load for a checkpoint (in their layout:
    `rollout_train.resharding`), told the run's fence to note it under; without it, they load the trainer's. `evals`
    says which checkpoints the run evaluates as it makes them, between their step and the next."""
    algorithm = algorithm if algorithm is not None else Grpo()
    retention = retention if retention is not None else Retention()
    ledger, blobs = checkpoints.ledger, checkpoints.blobs
    fence = await ledger.take(scope(run))  # whoever ran this before can no longer write
    decided = {int(number): _mapping(group) for number, group in (await ledger.read(table(run, GROUPS))).items()}
    recorded = {
        int(key): Result.from_json(_mapping(line), int(key), decided.get(int(key), {}))
        for key, line in (await ledger.read(table(run, RESULTS))).items()
    }
    steps = {int(key): _mapping(step) for key, step in (await ledger.read(table(run, STEPS))).items()}
    failures = {int(key) for key in await ledger.read(table(run, FAILURES))}
    evaluated = {int(key): _mapping(said) for key, said in (await ledger.read(table(run, EVALS))).items()}
    curriculum = curriculum or Curriculum(environment.rows())
    for number in sorted(recorded):
        curriculum.recorded(recorded[number])
    for key in sorted(evaluated):
        said = evaluated[key]
        curriculum.evaluated(str(said["suite"]), str(said["checkpoint"]), await results(ledger, str(said["run"])))
    if start is not None and trainer.weights == "full" and (await checkpoints.checkpoint(start)).kind != "full":
        raise ValueError(f"{start} is an adapter: merge it (`rollout merge`) to train every weight from it")
    await plan(ledger, run, Plan(environment.program, binding or binding_for(environment, channel)), fence)
    here = {"from": start, "host": socket.gethostname(), "started": round(time.time(), 1)}
    await ledger.append(table(run, STARTS), str(fence.number), {**here, **(started or {})}, fence)
    asking = -(-(episodes_at_once + algorithm.group_size - 1) // algorithm.group_size)
    """Groups kept asked for: when one of the episodes running ends, another is waiting (a group is decided only once
    the last of one before it has ended)."""

    def note(kind: str, payload: Mapping[str, JsonValue]) -> None:
        event: dict[str, JsonValue] = {"kind": kind, "run": run, "at": round(time.time(), 3), **payload}
        for hook in hooks:
            hook.on_note(event)

    async def files(checkpoint: Checkpoint) -> Files:
        """A checkpoint's files on this machine, read from the blob store if they are not here."""
        here = directory / checkpoint.id
        if checkpoint.weights is None:
            raise ValueError(f"{checkpoint.id} was released: its weights are gone")
        weights = await checkpoints.files(checkpoint.weights, here / WEIGHTS)
        return Files(weights, await checkpoints.files(checkpoint.state, here / STATE) if checkpoint.state else None)

    async def serve(checkpoint: Checkpoint) -> None:
        nonlocal served
        if served is not None and checkpoint.depth <= served.depth:
            return  # (the channel does not go back)
        if reshard is not None:
            loaded = await checkpoints.files(await reshard(checkpoint, fence), directory / checkpoint.id / "resharded")
        else:
            loaded = (await files(checkpoint)).weights
        served_as = await publish(channel, checkpoint.id, str(loaded), checkpoint.depth, full=checkpoint.kind == "full")
        note("published", {"channel": channel, "adapter": checkpoint.id, "version": served_as})
        served = checkpoint
        keep = {checkpoint.id, checkpoint.parent}
        for old in await asyncio.to_thread(lambda: [each for each in directory.iterdir() if each.name not in keep]):
            await asyncio.to_thread(shutil.rmtree, old, ignore_errors=True)

    async def current() -> Checkpoint | None:
        """The checkpoint the next step goes on from: the newest the run made, else the one it starts from."""
        head = await checkpoints.head(run)
        return head if head is not None else await checkpoints.checkpoint(start) if start else None

    async def keeping() -> set[str]:
        """The checkpoints that keep their files whatever retention says."""
        starts = [await ledger.read(name) for name in await ledger.tables() if name.endswith(f"/{STARTS}")]
        begun = {
            str(record["from"]) for each in starts for record in map(_mapping, each.values()) if record.get("from")
        }
        serving = {served.id, *served.parents} if served is not None else set[str]()
        return begun | serving | set(await kept() if kept is not None else ())

    async def evaluated_with(checkpoint: Checkpoint) -> None:
        """Play the schedule's suite with a checkpoint the run made and serves, if the schedule names it and it has
        not been evaluated yet."""
        if evals is None or not evals.due(checkpoint, run) or int(str(checkpoint.step)) in evaluated:
            return
        step = int(str(checkpoint.step))
        eval_run = await evals.run(step)
        said = await evaluate(
            evals.environment, checkpoints, run=eval_run, suite=evals.suite, subject=checkpoint.id, base=base,
            channel=channel, directory=directory, publish=None, episodes=evals.episodes, binding=evals.binding,
            started={"from": None, "by": run, "step": step},  # (whether its files are kept is the run's retention's)
            asked_by="by its run's schedule", hooks=hooks,
        )  # fmt: skip
        summary: dict[str, JsonValue] = {"played": said["played"], "solved": said["solved"], "reward": said["reward"]}
        record: dict[str, JsonValue] = {"suite": evals.suite.name, "checkpoint": checkpoint.id, "run": eval_run}
        record |= summary | {"at": round(time.time(), 1)}
        await ledger.append(table(run, EVALS), str(step), record, fence)
        evaluated[step] = record
        curriculum.evaluated(evals.suite.name, checkpoint.id, said["results"])

    served: Checkpoint | None = None
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    if (now := await current()) is not None:
        await serve(now)  # (a loop that died between making a checkpoint and serving it serves it now)
        await evaluated_with(now)  # (and one that died while evaluating it finishes the eval)

    outstanding: dict[asyncio.Task[list[Episode]], int] = {}
    """Groups being played, by the task that waits for their episodes."""
    segments: dict[int, list[Weighted]] = {}
    """What the algorithm found to train on in each group with a result that is not yet done with."""
    queue: list[int] = []
    """Groups with something to train on that no step covers yet."""

    async def played(number: int) -> list[Episode]:
        count = int(str(decided[number].get("episodes", algorithm.group_size)))
        return await episodes_of(ledger, blobs, run, number, count)

    def ask(number: int) -> None:
        outstanding[asyncio.create_task(played(number))] = number

    async def decide() -> None:
        number = max(decided, default=0) + 1
        rng = random.Random(f"{seed}-{number}")
        row = curriculum.sample([str(decided[each]["task"]) for each in outstanding.values()], rng)
        group: dict[str, JsonValue] = {
            "task": row.key,
            "title": row.title,
            "parameters": environment.start(row, rng),
            "episodes": algorithm.group_size,
            "decided": round(time.time(), 1),
        }
        await ledger.append(table(run, GROUPS), str(number), group, fence)  # runners play it from here
        decided[number] = group
        ask(number)

    def held(number: int, episodes: list[Episode]) -> list[Weighted]:
        """What the algorithm trains on in a group (the same, from the same episodes, each time it is asked)."""
        return list(algorithm.batch(episodes, trainer.budget, random.Random(number)).segments)

    async def record(number: int, episodes: list[Episode]) -> None:
        group = decided[number]
        began = float(str(group.get("decided", time.time())))
        good = [episode for episode in episodes if episode.trainable]
        failed = [episode for episode in episodes if not episode.trainable]
        batch = algorithm.batch(episodes, trainer.budget, random.Random(number))
        line = Result(
            group=number,
            time=round(time.time(), 1),
            task=str(group["task"]),
            title=str(group["title"]),
            rollout_seconds=round(time.time() - began, 1),
            rewards=[episode.reward for episode in good],
            solved=[episode.solved for episode in good],
            durations=[episode.duration for episode in good],
            failed=len(failed),
            failures=[str(episode.detail or episode.excluded or episode.outcome.value) for episode in failed],
            notes=batch.notes,
            segments_recorded=sum(
                len(trajectory.segments) for episode in good for trajectory in episode.trajectories.values()
            ),
            segments=len(batch.segments),
            skipped=batch.skipped,
        )
        curriculum.recorded(line)  # (the curriculum sees the task's own rewards, whatever the algorithm adds)
        line.unlocked = len(curriculum.unlocked())
        await ledger.append(table(run, RESULTS), str(number), line.to_json(), fence)  # before anything trains on it
        recorded[number] = line
        note("result", {"group": number, **line.to_json()})
        segments[number] = held(number, episodes)
        if segments[number]:
            queue.append(number)
        else:
            done_with([number])

    def done_with(numbers: Sequence[int]) -> None:
        for number in numbers:
            segments.pop(number, None)

    async def take(key: int, numbers: list[int]) -> None:
        """A step over `numbers`: decided (unless it was), made once, served."""
        nonlocal failed_updates
        given = spread([weighted for number in numbers for weighted in segments[number]], trainer.budget.segments,
                       random.Random(f"{seed}-step-{key}"))  # fmt: skip
        if key not in steps:
            parent = await current()
            trained: list[JsonValue] = [[weighted.source, weighted.advantage] for weighted in given]
            manifest = await blobs.put(json.dumps(trained).encode(), "application/json")
            decision: dict[str, JsonValue] = {
                "groups": list[JsonValue](numbers),
                "parent": parent.id if parent else None,
                "makes": new_id(),
                "segments": len(given),
                "decided": round(time.time(), 1),
                "batch": manifest.model_dump(mode="json"),
                "seed": key,
            }
            await ledger.append(table(run, STEPS), str(key), decision, fence)  # before the trainer is called
            steps[key] = decision
        intent = steps[key]
        makes = str(intent["makes"])
        try:
            checkpoint = await checkpoints.checkpoint(makes)  # made before this loop died: not made again
        except KeyError:
            parent_id = str(intent["parent"]) if intent["parent"] else None
            parent = await checkpoints.checkpoint(parent_id) if parent_id else None
            # An adapter's first step over full weights begins a new adapter: the model it trains over is those weights.
            begin = await files(parent) if parent is not None and parent.kind == trainer.weights else None
            into = directory / makes
            await asyncio.to_thread(shutil.rmtree, into, ignore_errors=True)  # (what a step that died left)
            try:
                step = await trainer.step(given, seed=int(str(intent["seed"])), parent=begin, into=into)
            except StepFailed as error:  # the weights stay as they were, and the run goes on
                failed_updates += 1
                said: dict[str, JsonValue] = {
                    "error": str(error).strip().splitlines()[-1][:300],
                    "at": round(time.time(), 1),
                }
                await ledger.append(table(run, FAILURES), str(key), said, fence)
                failures.add(key)
                note("step", {"step": key, "groups": list[JsonValue](numbers), **said})
                done_with(numbers)
                if failed_updates >= FAILED_UPDATES:
                    raise
                return
            checkpoint = await checkpoints.add(
                fence,
                makes,
                weights=into / WEIGHTS,
                run=run,
                base=base,
                kind=trainer.weights,
                step=key,
                state=into / STATE if await asyncio.to_thread((into / STATE).exists) else None,
                parents=[parent_id] if parent_id else [],
                batch=BlobReference.model_validate(intent["batch"]),
                metrics=step.metrics,
            )
        failed_updates = 0
        await serve(checkpoint)
        if made is not None:
            await made(checkpoint)
        await checkpoints.thin(fence, run, retention, await keeping())
        metrics: dict[str, JsonValue] = {name: round(value, 5) for name, value in checkpoint.metrics.items()}
        note("step", {"step": key, "groups": list[JsonValue](numbers), "checkpoint": checkpoint.id, "metrics": metrics})
        done_with(numbers)
        await evaluated_with(checkpoint)  # (the next step waits for it: the checkpoint is served until it ends)

    failed_updates = 0
    stepping: asyncio.Task[None] | None = None
    try:
        # What a stopped loop left: groups decided and not played out, groups played out and not done with, a step
        # decided and not finished.
        covered = {group: key for key, step in steps.items() for group in _groups(step)}
        unfinished = [key for key in sorted(steps) if key not in failures and not await _made(checkpoints, steps[key])]
        for number in sorted(decided):
            if number not in recorded:
                ask(number)
            elif recorded[number].segments and (number not in covered or covered[number] in unfinished):
                segments[number] = held(number, await played(number))  # (the ledger has them)
                if number not in covered:
                    queue.append(number)
        for key in unfinished:  # a step that was decided is finished before another is decided
            await take(key, _groups(steps[key]))
        owed = groups - len(outstanding)
        while outstanding or owed > 0 or queue or stepping is not None:
            while len(outstanding) < asking and owed > 0:
                await decide()
                owed -= 1
            last = not outstanding and owed <= 0
            if stepping is None and queue and (len(queue) >= groups_per_step or last):
                numbers, queue[:] = list(queue), []
                stepping = asyncio.create_task(take(max(steps, default=0) + 1, numbers))
            waited: set[asyncio.Task[Any]] = {*outstanding, *([stepping] if stepping else [])}
            if not waited:
                continue
            finished, _ = await asyncio.wait(waited, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted((task for task in finished if task in outstanding), key=lambda each: outstanding[each]):
                await record(outstanding.pop(task), task.result())
            if stepping is not None and stepping in finished:
                done, stepping = stepping, None
                done.result()  # (a step that failed too often stops the loop)
    finally:
        for task in [*outstanding, *([stepping] if stepping else [])]:
            task.cancel()
        if stepping is not None:
            await asyncio.gather(stepping, return_exceptions=True)


async def _made(checkpoints: Checkpoints, step: dict[str, JsonValue]) -> bool:
    try:
        await checkpoints.checkpoint(str(step["makes"]))
    except KeyError:
        return False
    return True


def _groups(step: dict[str, JsonValue]) -> list[int]:
    """The groups a step covers, by their numbers."""
    listed = step.get("groups")
    return [int(str(group)) for group in listed] if isinstance(listed, list) else []


def _mapping(record: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(record, dict)
    return record
