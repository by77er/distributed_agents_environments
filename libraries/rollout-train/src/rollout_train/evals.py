"""Evaluations: a frozen suite of starts, played by one checkpoint (or the base model), with nothing trained.

A **suite** is a named list of starts of a catalog's rows, each a row's start drawn with a seed of its own: what every
subject plays, start for start, so that subjects compare. It is kept in the ledger, written once under the suite's
fence (`suites/NAME`) and never changed: what it is (`evaluations/SUITE/suite`: its catalog, rows and seeds) and its
starts (`evaluations/SUITE/starts`, by number from 1: the row's key and title, the seed, the start's parameters).

An **eval** is one suite played by one subject: a checkpoint (by any reference `rollout_train.registry.resolved`
takes), or the base model. It is a run of its own, registered and fenced like any run, whose start says what it is
(`kind: eval`, the suite, the checkpoint). It serves the subject on the channel, asks for one group per start with
`episodes` episodes each (runners play them as they play any run's), and records each episode's outcome under
`evaluations/SUITE/EVAL/results` (`START-EPISODE`), beside a record of the subject (`evaluations/SUITE/EVAL/subject`).
Each group's result is written to the run's own `results` too, so the eval reads like any run. Started again, it goes
on: what it decided and what it recorded are not done twice.
"""

import asyncio
import random
import socket
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.catalog import Catalog, binding_for
from rollout.harness.runner import RunBinding
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest
from rollout_train.launches import EVAL
from rollout_train.ledger import Fence, Ledger, between
from rollout_train.record import GROUPS, RESULTS, STARTS, Result, scope, table
from rollout_train.registry import valid
from rollout_train.rollouts.episodes import Episode
from rollout_train.rollouts.scheduler import Hooks, Plan, episodes_of, plan
from rollout_train.trainer import WEIGHTS

EVALUATIONS = "evaluations/"
NOTHING_TRAINED = "an evaluation trains on nothing"


def suite_table(suite: str, part: str) -> str:
    """A suite's table: `suite` (what it is) or `starts`."""
    return f"{EVALUATIONS}{suite}/{part}"


def subject_table(suite: str, subject: str, part: str) -> str:
    """A subject's table under a suite: `subject` (who played) or `results` (how each episode went)."""
    return f"{EVALUATIONS}{suite}/{subject}/{part}"


@dataclass(frozen=True)
class Start:
    """One start of a suite: a row's start, drawn with a seed of its own."""

    task: str
    """The row's key."""
    title: str
    seed: int
    parameters: JsonValue
    """What every episode of it is given: the row's start, drawn with `seed`."""


@dataclass(frozen=True)
class Suite:
    """A named list of starts of a catalog's rows, frozen: what every subject plays, start for start."""

    name: str
    catalog: str
    """The catalog, as `module:name`."""
    starts: list[Start]
    """Its starts, in order: each row it names, once with each seed."""
    made: float = 0.0
    rows: list[str] | None = None
    """The rows it names, by key."""
    seeds: list[int] | None = None


async def make_suite(
    ledger: Ledger, name: str, catalog_name: str, catalog: Catalog, *, rows: Sequence[str] | None, seeds: Sequence[int]
) -> Suite:
    """Make a suite of `catalog`: a start of each row (of `rows`, by key; else every row) for each seed. Raises
    `ValueError` for a name that is no name or is taken (a suite is never changed), or a row the catalog lacks."""
    name = valid(name)
    if await suite_of(ledger, name) is not None:
        raise ValueError(f"there is a suite {name!r} already: a suite is never changed, make another")
    known = {row.key: row for row in catalog.rows()}
    if missing := [key for key in rows or [] if key not in known]:
        raise ValueError(f"the catalog has no row {', '.join(missing)}")
    chosen = [known[key] for key in rows] if rows else list(known.values())
    if not seeds:
        raise ValueError("a suite needs a seed at least")
    starts = [
        Start(row.key, row.title, seed, catalog.start(row, random.Random(seed))) for row in chosen for seed in seeds
    ]
    made = Suite(name, catalog_name, starts, round(time.time(), 1), [row.key for row in chosen], list(seeds))
    fence = await ledger.take(f"suites/{name}")
    about: Any = {"catalog": catalog_name, "made": made.made, "rows": made.rows, "seeds": made.seeds}
    await ledger.append(suite_table(name, "suite"), "suite", about, fence)
    for number, start in enumerate(starts, start=1):
        record: Any = asdict(start)
        await ledger.append(suite_table(name, "starts"), str(number), record, fence)
    return made


async def suite_of(ledger: Ledger, name: str) -> Suite | None:
    """A suite, if there is one by that name."""
    about: Any = (await ledger.read(suite_table(name, "suite"))).get("suite")
    starts = await ledger.read(suite_table(name, "starts"))
    if about is None and not starts:
        return None
    about = about or {}
    listed = [Start(**record) for _, record in sorted(starts.items(), key=lambda item: int(item[0]))]  # type: ignore[arg-type]
    return Suite(name, str(about.get("catalog") or ""), listed, float(about.get("made") or 0.0),
                 about.get("rows"), about.get("seeds"))  # fmt: skip


async def suites_in(ledger: Ledger) -> list[str]:
    """Every suite a ledger has, by name."""
    return sorted({name for each in await ledger.tables() if (name := between(each, EVALUATIONS, "/starts"))})


class Publisher(Protocol):
    async def __call__(
        self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
    ) -> int: ...


async def evaluate(
    catalog: Catalog,
    checkpoints: Checkpoints,
    *,
    run: str,
    suite: Suite,
    subject: str | None,
    base: str | None,
    channel: str,
    directory: Path,
    publish: Publisher,
    episodes: int = 1,
    binding: RunBinding | None = None,
    started: Mapping[str, JsonValue] | None = None,
    asked_by: str = "by hand",
    reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None,
    hooks: Sequence[Hooks] = (),
) -> dict[str, Any]:
    """Play `suite` with `subject` (a checkpoint's id; None: the base model, named `base`) served on `channel`,
    `episodes` episodes of each start, as the run `run`; returns how it went (`played`, `solved`, `reward`). `publish`
    serves a checkpoint on the channel (a full one in place of the engines' weights: the channel's engines must hold
    the model it is an adapter over, which `rollout eval` sees to), `reshard` gives its files in the engines' layout (`rollout_train.resharding`);
    `directory` holds its files on this machine."""
    ledger, blobs = checkpoints.ledger, checkpoints.blobs
    fence = await ledger.take(scope(run))
    await plan(ledger, run, Plan(catalog.program, binding or binding_for(catalog, channel)), fence)
    here: dict[str, JsonValue] = {"kind": EVAL, "suite": suite.name, "checkpoint": subject, "from": subject}
    here |= {"host": socket.gethostname(), "started": round(time.time(), 1)}
    await ledger.append(table(run, STARTS), str(fence.number), {**here, **(started or {})}, fence)
    who: dict[str, JsonValue] = {
        "kind": "checkpoint" if subject else "model",
        "checkpoint": subject,
        "model": base,
        "episodes": episodes,
        "asked_by": asked_by,
        "decided": round(time.time(), 1),
        "run": run,
    }
    await ledger.append(subject_table(suite.name, run, "subject"), "subject", who, fence)

    def note(kind: str, payload: Mapping[str, JsonValue]) -> None:
        event: dict[str, JsonValue] = {"kind": kind, "run": run, "at": round(time.time(), 3), **payload}
        for hook in hooks:
            hook.on_note(event)

    if subject is not None:
        served = await checkpoints.checkpoint(subject)
        if served.weights is None:
            raise ValueError(f"{subject} was released: its weights were deleted, so it cannot be played")
        manifest = await reshard(served, fence) if reshard is not None else served.weights
        files = await checkpoints.files(manifest, directory / served.id / ("resharded" if reshard else WEIGHTS))
        version = await publish(channel, served.id, str(files), served.depth, full=served.kind == "full")
        note("published", {"channel": channel, "adapter": served.id, "version": version})

    decided = await ledger.read(table(run, GROUPS))
    for number, start in enumerate(suite.starts, start=1):
        if str(number) not in decided:
            group: JsonValue = {
                "task": start.task,
                "title": start.title,
                "parameters": start.parameters,
                "episodes": episodes,
                "decided": round(time.time(), 1),
            }
            await ledger.append(table(run, GROUPS), str(number), group, fence)  # runners play it from here

    async def played(number: int) -> list[Episode]:
        found = await episodes_of(ledger, blobs, run, number, episodes)
        await record(number, found)
        return found

    async def record(number: int, found: list[Episode]) -> None:
        start = suite.starts[number - 1]
        for episode in found:
            outcome: JsonValue = {
                "run_id": episode.run_id,
                "reward": episode.reward,
                "solved": episode.solved,
                "duration": episode.duration,
                "outcome": episode.outcome.value,
                "detail": episode.detail,
                "time": round(time.time(), 1),
            }
            await ledger.append(subject_table(suite.name, run, "results"), f"{number}-{episode.number}", outcome, fence)
        good = [episode for episode in found if episode.trainable]
        line = Result(
            group=number,
            time=round(time.time(), 1),
            task=start.task,
            title=start.title,
            rewards=[episode.reward for episode in good],
            solved=[episode.solved for episode in good],
            durations=[episode.duration for episode in good],
            failed=len(found) - len(good),
            failures=[str(each.detail or each.excluded or each.outcome.value) for each in found if not each.trainable],
            skipped=NOTHING_TRAINED,
        )
        await ledger.append(table(run, RESULTS), str(number), line.to_json(), fence)
        note("result", {"group": number, **line.to_json()})

    every = await asyncio.gather(*(played(number) for number in range(1, len(suite.starts) + 1)))
    done = [episode for found in every for episode in found]
    rewards = [episode.reward for episode in done if episode.trainable]
    return {
        "played": len(done),
        "solved": sum(episode.solved for episode in done),
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
    }
