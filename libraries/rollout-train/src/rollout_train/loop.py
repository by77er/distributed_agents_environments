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
makes are appended under that fence. What it does outside the ledger, it does only while its fence is the newest: it
looks before it publishes a checkpoint, before it deletes files in the run's directory, and before it moves a bookmark
(`made`). Its trainer writes a step's files into a directory of the loop's own (`making/FENCE/MAKES`), renamed to the
checkpoint's (`MAKES`) once the checkpoint is added, so a loop that was replaced while its trainer ran never writes
where its replacement does.

**What its channel serves is written down** (`rollout_train.serving`): each time it serves a checkpoint, it appends that
the channel serves it from now on (the base model until the first), and then publishes it to the engines in its own
process, if it has any. Engines on other machines follow the record (`rollout_train.following`).

**It can evaluate its checkpoints as it makes them** (`evals`, a `rollout_train.evals.Schedule`). After a step whose
checkpoint the schedule names is served, the suite is asked for as an eval of that checkpoint, a run of its own (and a
run for each entry, for a suite of several environments: each played on the same channel, with its own binding), and
the next step waits until every start has been played: all that time the channel serves that checkpoint, while
training groups go on being played under it. Each entry's results are folded into the curriculum, the entry named.
Started again, the loop finishes an eval it left unfinished before it decides anything.

**Its changeable settings can change while it runs** (`rollout_train.settings`): how many groups a step waits for, its
evals, and the settings its trainer takes between steps. Each time it is about to decide a step it reads what is wanted
of them (`desired`), and decides the step with them; the step's record says which settings it used, and a step taken
again after a stop uses those. Whether a checkpoint is evaluated is the evals its step was decided with, and the
version of the suite its name pointed to then (`rollout_train.evals`): an edit of the suite applies from the next step.

**It can be paused** (`rollout_train.settings.PAUSED`, among the desired settings). Each time it is about to decide a
group or a step it looks; paused, it decides neither, while the episodes playing play out and are recorded and a step
being taken is finished (runners claim none of its episodes meanwhile: `rollout_train.rollouts.scheduler`). It looks
again every `PAUSE_LOOK` seconds, and goes on once the run is no longer paused. A pause and a resume are noted to the
hooks (`paused`, `resumed`).
"""

import asyncio
import json
import os
import random
import shutil
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.curriculum import Curriculum, curriculum_of
from rollout.environment import Environment, binding_for, held_out, train_start
from rollout.harness.runner import RunBinding
from rollout_train.algorithm import Algorithm, algorithm_for, within
from rollout_train.bridges import bridge_of
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest, Retention, new_id
from rollout_train.evals import Publisher, Schedule, evaluate
from rollout_train.inference.channel import MAX_LAG as MAX_LAG_DEFAULT
from rollout_train.ledger import Fence, Fenced, Ledger
from rollout_train.record import (
    EVALS,
    FAILURES,
    GROUPS,
    RESULTS,
    STARTS,
    STEPS,
    Result,
    described,
    mapping,
    results,
    scope,
    start_header,
    table,
)
from rollout_train.rollouts import Episode
from rollout_train.rollouts.scheduler import Hooks, Plan, episodes_of, plan
from rollout_train.serving import Serving, record_serving
from rollout_train.settings import (
    EVALS_EPISODES,
    EVALS_EVERY,
    EVALS_SUITE,
    GROUPS_PER_STEP,
    MAX_LAG,
    PAUSED,
    applied,
    run_key,
    trainer_key,
)
from rollout_train.trainer import STATE, WEIGHTS, Changeable, Files, Item, StepFailed, Trainer, objective_of, weight_of

FAILED_UPDATES = 3
"""Steps that may fail in a row (each is written down, and the weights stay as they were) before the loop stops."""
PAUSE_LOOK = 1.0
"""Seconds between a paused loop's looks at whether it is still paused."""


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
    max_lag: int = MAX_LAG_DEFAULT,
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
    desired: Callable[[], Awaitable[Mapping[str, JsonValue]]] | None = None,
    scheduled: Callable[[str, int, int | None], Awaitable[Schedule | None]] | None = None,
) -> None:
    """Train from `start` (a checkpoint's id; else the base model, named `base`) on `environment` until `groups` more
    groups have been played (those a stopped loop left unplayed among them) and every group played has been trained on,
    serving each checkpoint made on `channel`; a run started again goes on from the newest checkpoint it made. A step is
    taken over the groups queued once at least `groups_per_step` have something to train on (and, at the end, over what
    is left). `directory` is where checkpoints' files are kept on this machine while they are in use: the one being
    served and the one before it (a turn in progress finishes under the weights it began with); every checkpoint's files
    are in the blob store; `publish` serves a checkpoint on `channel`. `algorithm` is the one the family of the
    trainer's objective takes, unless given (`rollout_train.algorithm.algorithm_for`).
    `episodes_at_once` is how many episodes the run keeps work waiting for, whatever groups they are of (runners play
    them, as many at once as each has places). `binding` says how the program's model slots and imports are served (by
    default: every slot from `channel`, each import from the tool set of its own name). `curriculum` is one that has
    recorded nothing (by default the environment's own, else the generic one: `curriculum_of`): the run's results are
    folded into it. Each group's start is drawn with `train_start`, never one of the environment's eval starts.
    `retention` says which of the checkpoints the run made keep their files (weights and trainer state) once a newer one
    is served (`Retention()` unless given); besides those, what is served, what any run starts from, and whatever `kept`
    says (the bookmarked checkpoints, say) keep theirs. `started` is what the run's `starts` record says beside what the
    loop knows (where it starts from, this host, the time): where the run's directory is, where the monitor on its
    machine serves (`address`), and what profile started it, say. `hooks` are told of each result and step; `made` is
    called with each checkpoint made, once it is served (to move a bookmark, say). `reshard` gives the files the engines
    load for a checkpoint (made by a bridge: `rollout_train.bridges`), told the run's fence to note it under; without
    it, they load the trainer's. `evals` says which checkpoints the run evaluates as it makes them, between their step
    and the next. `desired` reads what is wanted of the run's changeable settings (`rollout_train.settings`:
    `groups_per_step`, `evals.…`, `trainer.…`), each time a step is about to be decided; `scheduled` makes the schedule
    of evals they name (a suite by name or a version by id, every, episodes; None for a suite the run cannot play),
    without which only `evals`' suite can be played. A suite named by its name is played in the version its name points
    to when each step is decided: the step's record says which (`suite_version`)."""
    algorithm = algorithm if algorithm is not None else algorithm_for(objective_of(trainer))
    retention = retention if retention is not None else Retention()
    ledger, blobs = checkpoints.ledger, checkpoints.blobs
    fence = await ledger.take(scope(run))  # whoever ran this before can no longer write
    decided = {int(number): mapping(group) for number, group in (await ledger.read(table(run, GROUPS))).items()}
    recorded = {
        int(key): Result.from_json(mapping(line), int(key), decided.get(int(key), {}))
        for key, line in (await ledger.read(table(run, RESULTS))).items()
    }
    steps: dict[int, dict[str, JsonValue]] = {
        int(key): mapping(step) for key, step in (await ledger.read(table(run, STEPS))).items()
    }
    failures = {int(key) for key in await ledger.read(table(run, FAILURES))}
    evaluated = {int(key): mapping(said) for key, said in (await ledger.read(table(run, EVALS))).items()}
    curriculum = curriculum or curriculum_of(environment)
    eval_starts = held_out(environment)  # (which training never draws)
    for number in sorted(recorded):
        curriculum.recorded(recorded[number])
    for key in sorted(evaluated):
        said = evaluated[key]
        for entry, played_by in _entries_of(said):
            curriculum.evaluated(str(said["suite"]), str(said["checkpoint"]), await results(ledger, played_by), entry)
    if start is not None and trainer.weights == "full" and (await checkpoints.checkpoint(start)).kind != "full":
        raise ValueError(f"{start} is an adapter: merge it (`rollout merge`) to train every weight from it")
    await plan(ledger, run, Plan(environment.program, binding or binding_for(environment, channel)), fence)
    here = start_header(**{"from": start})
    here |= described(environment)
    await ledger.append(table(run, STARTS), str(fence.number), {**here, **(started or {})}, fence)
    asking = -(-(episodes_at_once + algorithm.group_size - 1) // algorithm.group_size)
    """Groups kept asked for: when one of the episodes running ends, another is waiting (a group is decided only once
    the last of one before it has ended)."""
    settings: dict[str, JsonValue] = {
        GROUPS_PER_STEP: groups_per_step,
        MAX_LAG: max_lag,
        EVALS_SUITE: (evals.named or evals.suite.name) if evals else None,
        EVALS_EVERY: evals.every if evals else 1,
        EVALS_EPISODES: evals.episodes if evals else None,
    }
    settings |= {run_key(key): value for key, value in _changeable(trainer).items()}
    """The changeable settings in effect: those the next step is decided with."""
    schedules: dict[tuple[str, int, int | None], Schedule | None] = {}

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
        if checkpoint.weights is None:
            raise ValueError(f"{checkpoint.id} was released: its weights are gone")
        manifest = await reshard(checkpoint, fence) if reshard is not None else checkpoint.weights
        layout = await bridge_of(ledger, checkpoint.id) if reshard is not None else None
        wanted = Serving(
            channel, checkpoint.id, checkpoint.depth, checkpoint.kind, manifest, layout,
            await _over(checkpoints, checkpoint), base, trainer.budget.segment_tokens,
            max_lag=int(str(settings[MAX_LAG])),
        )  # fmt: skip
        await record_serving(ledger, run, wanted, fence)  # (whatever serves the channel elsewhere follows it)
        if reshard is not None:
            loaded = await checkpoints.files(manifest, directory / checkpoint.id / "resharded")
        else:
            loaded = (await files(checkpoint)).weights
        await newest(ledger, fence)
        served_as = await publish(channel, checkpoint.id, str(loaded), checkpoint.depth, full=checkpoint.kind == "full")
        note("published", {"channel": channel, "adapter": checkpoint.id, "version": served_as})
        served = checkpoint
        keep = {checkpoint.id, checkpoint.parent}
        await newest(ledger, fence)  # (a loop that was replaced deletes nothing of its replacement's)
        for old in await asyncio.to_thread(lambda: [each for each in directory.iterdir() if each.name not in keep]):
            await asyncio.to_thread(shutil.rmtree, old, ignore_errors=True)

    async def current() -> Checkpoint | None:
        """The checkpoint the next step goes on from: the newest the run made, else the one it starts from."""
        head = await checkpoints.head(run)
        return head if head is not None else await checkpoints.checkpoint(start) if start else None

    async def keeping() -> set[str]:
        """The checkpoints that keep their files whatever retention says."""
        starts = [await ledger.read(name) for name in await ledger.tables() if name.endswith(f"/{STARTS}")]
        begun = {str(record["from"]) for each in starts for record in map(mapping, each.values()) if record.get("from")}
        serving = {served.id, *served.parents} if served is not None else set[str]()
        return begun | serving | set(await kept() if kept is not None else ())

    def cadence(said: Mapping[str, JsonValue]) -> tuple[int, int | None]:
        """How often and how many episodes of each start the evals in `said` play (none: the suite's)."""
        episodes = said.get(EVALS_EPISODES)
        return int(str(said.get(EVALS_EVERY) or 1)), int(str(episodes)) if episodes else None

    async def schedule_for(suite: str, every: int, episodes: int | None) -> Schedule | None:
        """The evals of a suite (by name: the version its name points to now; or a version, by id)."""
        if evals is not None and (
            suite == evals.suite.id or (scheduled is None and suite in (evals.named, evals.suite.name))
        ):
            return replace(evals, every=every, episodes=episodes)
        return await scheduled(suite, every, episodes) if scheduled is not None else None

    async def pinned() -> str | None:
        """The version of the suite the evals in effect name, as its name points now: what the step about to be decided
        is evaluated with."""
        if not settings.get(EVALS_SUITE):
            return None
        schedule = await schedule_for(str(settings[EVALS_SUITE]), *cadence(settings))
        return schedule.suite.id if schedule is not None else None

    async def schedule_of(step: int) -> Schedule | None:
        """The evals a step was decided with (those of `evals` for a step whose record says no settings): the version
        its record names."""
        said = steps.get(step, {}).get("settings")
        if not isinstance(said, dict):
            return evals
        if not said.get(EVALS_SUITE):
            return None
        version = str(steps[step]["suite_version"])
        key = (version, *cadence(said))
        if key not in schedules:
            schedules[key] = await schedule_for(*key)
        return schedules[key]

    async def evaluated_with(checkpoint: Checkpoint) -> None:
        """Play the suite its step's evals name with a checkpoint the run made and serves, if they name it and it has
        not been evaluated yet."""
        if checkpoint.run != run or checkpoint.step is None or checkpoint.step in evaluated:
            return
        step = checkpoint.step
        schedule = await schedule_of(step)
        if schedule is None or not schedule.due(checkpoint, run):
            return
        eval_run = await schedule.run(step)

        async def part(number: int) -> str:
            return await schedule.run(step, number)

        said = await evaluate(
            checkpoints, run=eval_run, suite=schedule.suite, subject=checkpoint.id, base=base, channel=channel,
            directory=directory, publish=None, environments=schedule.environments, binding=schedule.binding,
            parts=part, episodes=schedule.episodes,
            started={"from": None, "by": run, "step": step},  # (whether its files are kept is the run's retention's)
            asked_by="by its run's schedule", hooks=hooks,
        )  # fmt: skip
        summary: dict[str, JsonValue] = {"played": said["played"], "solved": said["solved"], "reward": said["reward"]}
        record: dict[str, JsonValue] = {"suite": schedule.suite.name, "version": schedule.suite.id}
        record |= {"checkpoint": checkpoint.id, "run": eval_run}
        record |= summary | {"at": round(time.time(), 1)}
        record["entries"] = [
            {key: value for key, value in each.items() if key != "results"} for each in said["entries"]
        ]
        await ledger.append(table(run, EVALS), str(step), record, fence)
        evaluated[step] = record
        for each in said["entries"]:
            curriculum.evaluated(schedule.suite.name, checkpoint.id, each["results"], each["environment"])

    async def refresh() -> None:
        """Take what is wanted of the changeable settings, for the step about to be decided and those after it."""
        nonlocal settings
        if desired is None:
            return
        wanted = applied(settings, await desired())
        try:
            trained_with(_trainers(wanted))
        except (ValueError, TypeError) as error:  # (a value the trainer cannot take: its settings stay as they were)
            note("settings", {"error": str(error)})
            wanted |= {key: value for key, value in settings.items() if trainer_key(key) is not None}
        if changes := {key: value for key, value in wanted.items() if settings.get(key) != value}:
            note("settings", {"changed": changes})
        settings = wanted

    async def pausing() -> bool:
        """Whether the run is paused now, as its desired settings say; a change is noted."""
        nonlocal halted
        now = desired is not None and (await desired()).get(PAUSED) is True
        if now != halted:
            note("paused" if now else "resumed", {})
            halted = now
        return now

    def trained_with(said: Mapping[str, JsonValue]) -> None:
        """Have the trainer take these of its settings from its next step on, where they differ from what it has."""
        if isinstance(trainer, Changeable):
            now = trainer.changeable
            if differ := {key: value for key, value in said.items() if now.get(key) != value}:
                trainer.change(differ)

    served: Checkpoint | None = None
    halted = False
    """Whether the run was paused when last looked."""
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    if (now := await current()) is not None:
        await serve(now)  # (a loop that died between making a checkpoint and serving it serves it now)
        await evaluated_with(now)  # (and one that died while evaluating it finishes the eval)
    else:  # (the base model, until the first checkpoint)
        first = Serving(channel, model=base, sequence=trainer.budget.segment_tokens, max_lag=max_lag)
        await record_serving(ledger, run, first, fence)

    outstanding: dict[asyncio.Task[list[Episode]], int] = {}
    """Groups being played, by the task that waits for their episodes."""
    segments: dict[int, list[Item]] = {}
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
            "parameters": train_start(environment, row, rng, eval_starts),
            "episodes": algorithm.group_size,
            "decided": round(time.time(), 1),
        }
        await ledger.append(table(run, GROUPS), str(number), group, fence)  # runners play it from here
        decided[number] = group
        ask(number)

    def held(number: int, episodes: list[Episode]) -> list[Item]:
        """What the algorithm trains on in a group (the same, from the same episodes, each time it is asked)."""
        return list(algorithm.batch(episodes, trainer.budget, random.Random(number)).items)

    async def record(number: int, episodes: list[Episode]) -> None:
        group = decided[number]
        began = float(str(group.get("decided", time.time())))
        good = [episode for episode in episodes if episode.trainable]
        batch = algorithm.batch(episodes, trainer.budget, random.Random(number))
        line = Result.of(
            episodes,
            group=number,
            task=str(group["task"]),
            title=str(group["title"]),
            rollout_seconds=round(time.time() - began, 1),
            notes=batch.notes,
            segments_recorded=sum(
                len(trajectory.segments) for episode in good for trajectory in episode.trajectories.values()
            ),
            segments=len(batch.items),
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
        given = within([item for number in numbers for item in segments[number]], trainer.budget.segments,
                       random.Random(f"{seed}-step-{key}"))  # fmt: skip
        if key not in steps:
            parent = await current()
            trained: list[JsonValue] = [[item.source, weight_of(item)] for item in given]
            manifest = await blobs.put(json.dumps(trained).encode(), "application/json")
            decision: dict[str, JsonValue] = {
                "groups": list[JsonValue](numbers),
                "parent": parent.id if parent else None,
                "makes": new_id(),
                "segments": len(given),
                "decided": round(time.time(), 1),
                "batch": manifest.model_dump(mode="json"),
                "seed": key,
                "settings": dict(settings),
                "suite_version": await pinned(),
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
            into = directory / "making" / str(fence.number) / makes  # (its own: one it replaced may write its own)
            await asyncio.to_thread(shutil.rmtree, into, ignore_errors=True)  # (what a step that died left)
            await asyncio.to_thread(into.parent.mkdir, parents=True, exist_ok=True)
            if isinstance(used := intent.get("settings"), dict):  # (the settings it was decided with, taken again too)
                trained_with(_trainers(used))
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
            # The checkpoint is this loop's (another's add under an older fence is refused since this one took its
            # fence, and it was not there when the step began): its files are where the checkpoint's are looked for.
            await newest(ledger, fence)
            await asyncio.to_thread(_moved, into, directory / makes)
        failed_updates = 0
        await serve(checkpoint)
        if made is not None:
            await newest(ledger, fence)  # (a loop that was replaced does not move a bookmark back)
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
            paused = await pausing()  # (paused, nothing is decided: what is in flight goes on)
            while not paused and len(outstanding) < asking and owed > 0:
                await decide()
                owed -= 1
            last = not outstanding and owed <= 0
            if not paused and stepping is None and queue:
                await refresh()
            if (
                not paused
                and stepping is None
                and queue
                and (len(queue) >= int(str(settings[GROUPS_PER_STEP])) or last)
            ):
                numbers, queue[:] = list(queue), []
                stepping = asyncio.create_task(take(max(steps, default=0) + 1, numbers))
            waited: set[asyncio.Task[Any]] = {*outstanding, *([stepping] if stepping else [])}
            if not waited:
                if paused:
                    await asyncio.sleep(PAUSE_LOOK)
                continue
            finished, _ = await asyncio.wait(
                waited, timeout=PAUSE_LOOK if paused else None, return_when=asyncio.FIRST_COMPLETED
            )
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


def _entries_of(said: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The entries an eval in a run's `evals` table played, each its environment and the run that played it."""
    entries = cast(list[Mapping[str, Any]], said.get("entries") or [])
    return [(str(each["environment"]), str(each["run"])) for each in entries]


async def _over(checkpoints: Checkpoints, checkpoint: Checkpoint) -> str | None:
    """The full checkpoint an adapter is served over, by id, if it is over one (not over a model)."""
    if checkpoint.kind == "full":
        return None
    try:
        under = await checkpoints.under(checkpoint)
    except ValueError:  # (released: whatever serves the adapter holds its weights already, or cannot)
        return checkpoint.base
    return under.id if under is not None else None


async def newest(ledger: Ledger, fence: Fence) -> None:
    """Raise `Fenced` unless `fence` is still its scope's newest: before a side effect outside the ledger."""
    if (await ledger.fences()).get(fence.scope) != fence.number:
        raise Fenced(f"{fence.scope} has a newer writer than fence {fence.number}")


def _moved(source: Path, target: Path) -> None:
    """Put the directory `source` at `target`, replacing what is there."""
    shutil.rmtree(target, ignore_errors=True)
    os.replace(source, target)


async def _made(checkpoints: Checkpoints, step: dict[str, JsonValue]) -> bool:
    try:
        await checkpoints.checkpoint(str(step["makes"]))
    except KeyError:
        return False
    return True


def _changeable(trainer: Trainer) -> Mapping[str, JsonValue]:
    """The settings a trainer takes between steps, with their values now (none for one that takes none)."""
    return trainer.changeable if isinstance(trainer, Changeable) else {}


def _trainers(settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """The trainer's own of a run's settings (`trainer.…`, and its objective's `objective.…`), by its names for
    them."""
    return {name: value for key, value in settings.items() if (name := trainer_key(key)) is not None}


def _groups(step: dict[str, JsonValue]) -> list[int]:
    """The groups a step covers, by their numbers."""
    listed = step.get("groups")
    return [int(str(group)) for group in listed] if isinstance(listed, list) else []
