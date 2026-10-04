"""Evaluations: a frozen suite of starts, played by one checkpoint (or the base model), with nothing trained.

A **suite** is a named list of starts of an environment's rows, each a row's start drawn with a seed of its own: what
every subject plays, start for start, so that subjects compare. Most come from an environment's eval data
(`Environment.evals()`), frozen under their name the first time they are played (`suite_for`): training never draws
those starts. One can be made by hand too, of rows and seeds (`make_suite`), with no such promise. It is kept in the
ledger as one record, written once under the suite's fence (`suites/NAME`) and never changed
(`evaluations/SUITE/suite`): its environment and its version, its rows and seeds, whether it is held out of training,
and its starts in order (each the row's key and title, the seed, the start's parameters). Two makers of one suite at
once leave one of their suites whole: the one whose record was appended, which the other then reads and plays. A suite
made before suites were one record keeps its starts in a table of their own (`evaluations/SUITE/starts`, by number from
1), and reads the same.

An **eval** is one suite played by one subject: a checkpoint (by any reference `rollout_train.registry.resolved`
takes), or the base model. It is a run of its own, registered and fenced like any run, whose start says what it is
(`kind: eval`, the suite, the checkpoint). It serves the subject on the channel, asks for one group per start with
`episodes` episodes each (runners play them as they play any run's), and records each episode's outcome under
`evaluations/SUITE/EVAL/results` (`START-EPISODE`), beside a record of the subject (`evaluations/SUITE/EVAL/subject`).
Each group's result is written to the run's own `results` too, so the eval reads like any run. Started again, it goes
on: what it decided and what it recorded are not done twice.

A training run can evaluate its own checkpoints as it makes them (a `Schedule`): the loop plays the suite with the
checkpoint of every `every`th step between that step and the next, on the channel that already serves it, each eval a
run of its own (`rollout_train.loop.train`).
"""

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue

from rollout.environment import Environment, Start, binding_for, drawn
from rollout.harness.runner import RunBinding
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest
from rollout_train.launches import EVAL
from rollout_train.ledger import Fence, Fenced, Ledger, appended, between
from rollout_train.record import (
    ENDS,
    FINISHED,
    GROUPS,
    PROCESS,
    RESULTS,
    STARTS,
    Result,
    described,
    results,
    scope,
    table,
)
from rollout_train.registry import valid
from rollout_train.rollouts.episodes import Episode
from rollout_train.rollouts.scheduler import Hooks, Plan, episodes_of, plan
from rollout_train.serving import Serving, record_serving
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
class Suite:
    """A named list of starts of an environment's rows, frozen: what every subject plays, start for start."""

    name: str
    environment: str
    """The environment, as `module:name`."""
    starts: list[Start]
    """Its starts, in order: each row it names, once with each seed."""
    made: float = 0.0
    rows: list[str] | None = None
    """The rows it names, by key."""
    seeds: list[int] | None = None
    version: str | None = None
    """The environment's version when the suite was made."""
    held_out: bool = False
    """Whether it is the environment's eval data, whose starts training never draws."""


async def make_suite(
    ledger: Ledger,
    name: str,
    environment_name: str,
    environment: Environment,
    *,
    rows: Sequence[str] | None = None,
    seeds: Sequence[int] = (),
    starts: Sequence[Start] | None = None,
) -> Suite:
    """Make a suite of `environment`: `starts`, the environment's eval data of that name (held out of training); or,
    by hand, a start of each row (of `rows`, by key; else every row) for each seed (`drawn`). Raises `ValueError` for a
    name that is no name or is taken (a suite is never changed), a row the environment lacks, or no starts."""
    name = valid(name)
    if await suite_of(ledger, name) is not None:
        raise ValueError(f"there is a suite {name!r} already: a suite is never changed, make another")
    listed = list(starts) if starts is not None else drawn(environment, seeds=seeds, rows=rows)
    if not listed:
        raise ValueError("a suite needs a start at least")
    made = Suite(
        name, environment_name, listed, round(time.time(), 1), list(dict.fromkeys(start.task for start in listed)),
        list(dict.fromkeys(start.seed for start in listed)), environment.version, held_out=starts is not None,
    )  # fmt: skip
    fence = await ledger.take(f"suites/{name}")
    whole: Any = {key: value for key, value in asdict(made).items() if key != "name"}
    there = await appended(ledger, suite_table(name, "suite"), "suite", whole, fence)
    if there.wrote:
        return made
    if "starts" in (record := _mapping(there.record)):  # another maker's, made meanwhile: the suite is that one
        return _as_suite(name, record, {})
    return await suite_of(ledger, name) or made  # (one made as a record and a table of starts)


async def suite_for(ledger: Ledger, name: str, environment_name: str, environment: Environment) -> Suite:
    """The suite `name`: the one in the ledger, or else the environment's eval data of that name, frozen now (on first
    use). Raises `KeyError` when neither has it, `ValueError` when the ledger's is another environment's."""
    found = await suite_of(ledger, name)
    if found is not None:
        if found.environment and found.environment != environment_name:
            raise ValueError(f"the suite {name!r} is of {found.environment}, not {environment_name}")
        return found
    data = environment.evals()
    if name not in data:
        known = ", ".join(sorted(data)) or "none"
        raise KeyError(f"there is no suite {name!r}, and {environment_name} has no eval data of that name ({known})")
    for attempt in range(MAKERS):  # (another process may make it at the same time: then it is read from the ledger)
        try:
            return await make_suite(ledger, name, environment_name, environment, starts=data[name])
        except Fenced:  # another maker took the suite's fence after this one did
            await asyncio.sleep(0.05 * (attempt + 1))
        except ValueError:  # another maker made it since it was looked for
            if (found := await suite_of(ledger, name)) is None:
                raise
            return found
        if (found := await suite_of(ledger, name)) is not None:
            return found
    raise RuntimeError(f"the suite {name!r} was being made by others {MAKERS} times over, and none of them made it")


MAKERS = 5
"""How many times `suite_for` makes a suite that others are making at the same time, before it gives up."""


async def suite_of(ledger: Ledger, name: str) -> Suite | None:
    """A suite, if there is one by that name: one record (`suite`) holding all of it, or, for a suite made before
    suites were, a record of what it is and its starts in a table of their own."""
    found = (await ledger.read(suite_table(name, "suite"))).get("suite")
    about = _mapping(found)
    starts = {} if "starts" in about else await ledger.read(suite_table(name, "starts"))
    if found is None and not starts:
        return None
    return _as_suite(name, about, starts)


def _as_suite(name: str, about: Mapping[str, Any], starts: Mapping[str, JsonValue]) -> Suite:
    """A suite from its record, and its starts by number where the record does not hold them."""
    if "starts" in about:
        listed = [Start(**record) for record in about["starts"]]
    else:
        listed = [Start(**record) for _, record in sorted(starts.items(), key=lambda item: int(item[0]))]  # type: ignore[arg-type]
    environment = str(about.get("environment") or about.get("catalog") or "")  # (a suite made as a catalog's says so)
    return Suite(name, environment, listed, float(about.get("made") or 0.0), about.get("rows"), about.get("seeds"),
                 about.get("version"), bool(about.get("held_out")))  # fmt: skip


def starts_in(tables: Mapping[str, Mapping[str, Any]], suite: str) -> dict[str, Any]:
    """A suite's starts by number from 1, from a ledger's tables as read (by name), whichever way the suite was
    written."""
    whole: Any = _mapping(tables.get(suite_table(suite, "suite"), {}).get("suite")).get("starts")
    if isinstance(whole, list):
        return {str(number): start for number, start in enumerate(cast(list[Any], whole), start=1)}
    return dict(tables.get(suite_table(suite, "starts"), {}))


def suites_among(names: Iterable[str]) -> list[str]:
    """The suites tables are of, from their names."""
    found = {between(each, EVALUATIONS, part) for each in names for part in ("/suite", "/starts")}
    return sorted(name for name in found if name and "/" not in name)


def _mapping(record: JsonValue) -> Mapping[str, Any]:
    return record if isinstance(record, dict) else {}


async def suites_in(ledger: Ledger) -> list[str]:
    """Every suite a ledger has, by name."""
    return suites_among(await ledger.tables())


class Publisher(Protocol):
    async def __call__(
        self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
    ) -> int: ...


@dataclass(frozen=True)
class Schedule:
    """Evals a training run makes of its own checkpoints: `suite` played by the checkpoint of every `every`th step,
    `episodes` episodes of each start, between that step and the next. `environment` is the suite's environment, and
    `binding` how its episodes are played (by default every slot from the trained channel). `run` gives the eval's run
    for a step: the same each time it is asked for that step, and one the run's episode runners play."""

    suite: Suite
    environment: Environment
    run: Callable[[int], Awaitable[str]]
    every: int = 1
    episodes: int = 1
    binding: RunBinding | None = None

    def due(self, checkpoint: Checkpoint, run: str) -> bool:
        """Whether `checkpoint` is evaluated: a checkpoint `run` made at a step the schedule names."""
        return checkpoint.run == run and checkpoint.step is not None and checkpoint.step % self.every == 0


async def evaluate(
    environment: Environment,
    checkpoints: Checkpoints,
    *,
    run: str,
    suite: Suite,
    subject: str | None,
    base: str | None,
    channel: str,
    directory: Path,
    publish: Publisher | None,
    episodes: int = 1,
    binding: RunBinding | None = None,
    started: Mapping[str, JsonValue] | None = None,
    asked_by: str = "by hand",
    reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None,
    hooks: Sequence[Hooks] = (),
    served_by: str | None = None,
) -> dict[str, Any]:
    """Play `suite` with `subject` (a checkpoint's id; None: the base model, named `base`) served on `channel`,
    `episodes` episodes of each start, as the run `run`; returns how it went (`played`, `solved`, `reward`, and each
    start's `results`, a `Result` each). `publish` serves a checkpoint on the channel (a full one in place of the
    engines' weights; for an adapter over a full checkpoint, the engines must already hold that checkpoint's weights, as
    `rollout eval` sees to); None: the channel serves `subject` already (a training run's newest checkpoint). `reshard`
    gives its files in the engines' layout (`rollout_train.resharding`); `directory` holds its files on this machine.
    What the eval's channel serves is written down (`rollout_train.serving`), so that runners anywhere play it on
    replicas that serve `subject` and no other checkpoint; `served_by` names the channel whose replicas serve it
    (`RUN/NAME`: the training run's, for an eval its schedule asks for), where it is not the eval's own."""
    ledger, blobs = checkpoints.ledger, checkpoints.blobs
    fence = await ledger.take(scope(run))
    await plan(ledger, run, Plan(environment.program, binding or binding_for(environment, channel)), fence)
    here: dict[str, JsonValue] = {"kind": EVAL, "suite": suite.name, "checkpoint": subject, "from": subject}
    here |= {"host": socket.gethostname(), "process": PROCESS, "started": round(time.time(), 1)} | described(
        environment
    )
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

    if subject is None:  # (the model the channel's engines are started with)
        await record_serving(ledger, run, Serving(channel, model=base, served_by=served_by, max_lag=0), fence)
    elif publish is None:  # (the channel serves it already)
        known = await checkpoints.checkpoint(subject)
        said = Serving(channel, known.id, known.depth, known.kind, model=base, served_by=served_by, max_lag=0)
        await record_serving(ledger, run, said, fence)
    else:
        served = await checkpoints.checkpoint(subject)
        if served.weights is None:
            raise ValueError(f"{subject} was released: its weights were deleted, so it cannot be played")
        manifest = await reshard(served, fence) if reshard is not None else served.weights
        said = Serving(channel, served.id, served.depth, served.kind, manifest, model=base, max_lag=0)
        await record_serving(ledger, run, said, fence)
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
    await ledger.append(table(run, ENDS), str(fence.number), {"how": FINISHED, "at": round(time.time(), 1)}, fence)
    return {
        "played": len(done),
        "solved": sum(episode.solved for episode in done),
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "results": await results(ledger, run),  # (as first recorded)
    }
