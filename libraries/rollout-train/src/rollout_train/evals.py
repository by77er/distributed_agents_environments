"""Evaluations: a suite, an eval configuration kept in versions, played by one checkpoint (or the base model), with
nothing trained.

A **suite** is an eval configuration, by name: an environment (`module:name`, and its version then), the starts every
subject plays (each a row's start drawn with a seed of its own), the episodes of each start, and the sampling limits the
eval's channel takes (`thinking_tokens`, `answer_tokens`; none: the channel's own). Its starts are one of three: the
environment's eval data of a name (`Environment.evals()`); a start of each of some rows for each of some seeds
(`drawn`); or starts given as they are. Training never draws the environment's eval starts: a suite says whether all of
its starts are among them (`held_out`). Every subject of one version plays the same starts, start for start, so that
subjects compare.

A suite is kept in **versions**. Each is one record with an id of its own (`NAME@NUMBER`), written once under the
suite's fence (`suites/NAME`) and never changed or deleted: editing a suite (`edit_suite`) makes its next version. The
suite's name points to its newest version: the registry beside the ledger holds where
(`rollout_train.registry.SuiteName`), and each edit moves it; a name the registry holds nothing for is its newest
version in the ledger. Every version is in `evaluations/NAME/suite`: version 1 under the key `suite`, each later one
under its number. A suite made before suites had versions is its version 1: one record, or, older still, a record
and its starts in a table of their own (`evaluations/NAME/starts`, by number from 1).

Most suites are an environment's eval data, frozen as version 1 of a suite of that name the first time it is played
(`suite_for`). One can be made by hand too (`make_suite`). Two makers of one suite at once leave one of their suites
whole: the one whose record was appended, which the other then reads and plays. Two editors at once make two versions,
one after the other, and the name points to the later.

An **eval** is one version of a suite played by one subject: a checkpoint (by any reference
`rollout_train.registry.resolved` takes), or the base model. It is a run of its own, registered and fenced like any run,
whose start says what it is (`kind: eval`, the suite, its version, the checkpoint). It serves the subject on the
channel, asks for one group per start with `episodes` episodes each (runners play them as they play any run's), and
records each episode's outcome under `evaluations/SUITE/EVAL/results` (`START-EPISODE`, the start by its number in the
version), beside a record of the subject and the version it played (`evaluations/SUITE/EVAL/subject`; one recorded
before suites had versions played version 1). Each group's result is written to the run's own `results` too, so the
eval reads like any run. Started again, it goes on: what it decided and what it recorded are not done twice.

A training run can evaluate its own checkpoints as it makes them (a `Schedule`): the loop plays the suite with the
checkpoint of every `every`th step between that step and the next, on the channel that already serves it, each eval a
run of its own (`rollout_train.loop.train`). Each step is decided with the version the suite's name points to then.
"""

import asyncio
import hashlib
import socket
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue

from rollout.environment import Environment, Start, binding_for, drawn, held_out, start_key
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
from rollout_train.registry import registry_of, valid, version_number
from rollout_train.rollouts.episodes import Episode
from rollout_train.rollouts.scheduler import Hooks, Plan, episodes_of, plan
from rollout_train.serving import Serving, record_serving
from rollout_train.trainer import WEIGHTS

EVALUATIONS = "evaluations/"
NOTHING_TRAINED = "an evaluation trains on nothing"
FIRST = "suite"
"""The key of a suite's version 1 in its table; each later version's is its number."""
EVAL_DATA, DRAWN, GIVEN = "eval data", "rows and seeds", "starts"
"""How a version's starts were chosen: the environment's eval data of a name; a start of each row for each seed; or
given as they are."""
MAKERS = 5
"""How many times a suite that others are making or editing at the same time is tried again, before giving up."""


def suite_table(suite: str, part: str) -> str:
    """A suite's table: `suite` (its versions) or `starts` (version 1's starts, for a suite made before suites were one
    record)."""
    return f"{EVALUATIONS}{suite}/{part}"


def subject_table(suite: str, subject: str, part: str) -> str:
    """A subject's table under a suite: `subject` (who played) or `results` (how each episode went)."""
    return f"{EVALUATIONS}{suite}/{subject}/{part}"


def version_id(name: str, number: int) -> str:
    """A version's id: `NAME@NUMBER`."""
    return f"{name}@{number}"


def parsed(reference: str) -> tuple[str, int | None]:
    """The suite a reference names, and the number of the version it says: `NAME@NUMBER`, that version (0 for a number
    that is none); `NAME`, none (the version the name points to)."""
    name, at, number = reference.partition("@")
    if not at:
        return reference, None
    return name, int(number) if number.isdigit() else 0


def played_version(subject: Mapping[str, Any], suite: str) -> str:
    """The version an eval played, by id, from its subject's record: version 1 for one recorded before suites had
    versions."""
    return str(subject.get("version") or version_id(suite, 1))


@dataclass(frozen=True)
class Suite:
    """One version of a suite: what every subject of it plays, start for start, and how."""

    name: str
    environment: str
    """The environment, as `module:name`."""
    starts: list[Start]
    """Its starts, in order."""
    made: float = 0.0
    rows: list[str] | None = None
    """The rows it names, by key."""
    seeds: list[int] | None = None
    environment_version: str | None = None
    """The environment's version when this version was made."""
    held_out: bool = False
    """Whether every start is one of the environment's eval starts, which training never draws."""
    number: int = 1
    chosen: str = DRAWN
    """How its starts were chosen: `EVAL_DATA`, `DRAWN` or `GIVEN`."""
    eval_data: str | None = None
    """The name of the environment's eval data its starts are, where they are."""
    episodes: int = 1
    """Episodes of each start an eval plays, unless it is asked for another number."""
    thinking_tokens: int | None = None
    """The eval's channel's tokens of thinking per turn, and of answer after it; none: the channel's own."""
    answer_tokens: int | None = None
    edited_from: int | None = None
    """The version it was edited from, by number."""

    @property
    def id(self) -> str:
        return version_id(self.name, self.number)

    @property
    def limits(self) -> dict[str, int]:
        """The sampling limits it gives the eval's channel, by a channel's names for them (`thinking_tokens`,
        `answer_tokens`), where it gives any."""
        said = {"thinking_tokens": self.thinking_tokens, "answer_tokens": self.answer_tokens}
        return {key: value for key, value in said.items() if value is not None}

    def record(self) -> dict[str, Any]:
        """What the ledger keeps of it (its name and number are its table and key)."""
        return {key: value for key, value in asdict(self).items() if key not in ("name", "number")}

    def configured(self) -> tuple[Any, ...]:
        """What makes an eval of it what it is: the environment's version, the starts, the episodes and the limits."""
        return (self.environment_version, self.starts, self.episodes, self.thinking_tokens, self.answer_tokens)


def chosen_starts(
    environment: Environment,
    *,
    rows: Sequence[str] | None = None,
    seeds: Sequence[int] = (),
    starts: Sequence[Start] | None = None,
    eval_data: str | None = None,
) -> tuple[list[Start], str]:
    """A version's starts, and how they were chosen: the environment's eval data of the name `eval_data`; `starts`, as
    they are; or a start of each row (of `rows`, by key; else every row) for each seed (`drawn`). Raises `KeyError` for
    eval data the environment does not have, `ValueError` for a row it does not have or no seeds."""
    if eval_data is not None:
        data = environment.evals()
        if eval_data not in data:
            known = ", ".join(sorted(data)) or "none"
            raise KeyError(f"the environment has no eval data {eval_data!r} ({known})")
        return list(data[eval_data]), EVAL_DATA
    if starts is not None:
        return list(starts), GIVEN
    return drawn(environment, seeds=seeds, rows=rows), DRAWN


def _version(
    name: str,
    number: int,
    environment_name: str,
    environment: Environment,
    listed: list[Start],
    chosen: str,
    eval_data: str | None,
    *,
    episodes: int,
    thinking_tokens: int | None,
    answer_tokens: int | None,
    edited_from: int | None = None,
) -> Suite:
    """A version of a suite, made now. Raises `ValueError` for no starts, or a count that is no whole number of 1 at
    least."""
    if not listed:
        raise ValueError("a suite needs a start at least")
    counts: dict[str, Any] = {"episodes": episodes, "thinking_tokens": thinking_tokens, "answer_tokens": answer_tokens}
    for what, count in counts.items():
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ValueError(f"{what} is a whole number, 1 at least (not {count!r})")
    held = held_out(environment)
    return Suite(
        name, environment_name, listed, round(time.time(), 1), list(dict.fromkeys(start.task for start in listed)),
        list(dict.fromkeys(start.seed for start in listed)), environment.version,
        all(start_key(start.parameters) in held for start in listed), number, chosen, eval_data, episodes,
        thinking_tokens, answer_tokens, edited_from,
    )  # fmt: skip


async def make_suite(
    ledger: Ledger,
    name: str,
    environment_name: str,
    environment: Environment,
    *,
    rows: Sequence[str] | None = None,
    seeds: Sequence[int] = (),
    starts: Sequence[Start] | None = None,
    eval_data: str | None = None,
    episodes: int = 1,
    thinking_tokens: int | None = None,
    answer_tokens: int | None = None,
) -> Suite:
    """Make version 1 of a suite of `environment`, its starts chosen as `chosen_starts` says, with `episodes` of each
    start and the limits its evals' channel takes. Raises `ValueError` for a name that is no name or is taken (a suite
    is edited instead: `edit_suite`), a row the environment does not have, no starts, or a count below 1; `KeyError`
    for eval data the environment does not have."""
    name = valid(name)
    if await suite_of(ledger, name) is not None:
        raise ValueError(f"there is a suite {name!r} already: edit it (a version of its own), or make another")
    listed, chosen = chosen_starts(environment, rows=rows, seeds=seeds, starts=starts, eval_data=eval_data)
    made = _version(
        name, 1, environment_name, environment, listed, chosen, eval_data, episodes=episodes,
        thinking_tokens=thinking_tokens, answer_tokens=answer_tokens,
    )  # fmt: skip
    fence = await ledger.take(f"suites/{name}")
    there = await appended(ledger, suite_table(name, FIRST), FIRST, made.record(), fence)
    if there.wrote:
        return made
    if "starts" in (record := _mapping(there.record)):  # another maker's, made meanwhile: the suite is that one
        return _as_suite(name, 1, record, {})
    return await suite_of(ledger, version_id(name, 1)) or made  # (one made as a record and a table of starts)


async def edit_suite(
    ledger: Ledger,
    name: str,
    environment_name: str,
    environment: Environment,
    *,
    rows: Sequence[str] | None = None,
    seeds: Sequence[int] = (),
    starts: Sequence[Start] | None = None,
    eval_data: str | None = None,
    same_starts: bool = False,
    episodes: int = 1,
    thinking_tokens: int | None = None,
    answer_tokens: int | None = None,
    base: int | None = None,
) -> Suite:
    """Make a suite's next version, and point its name to it: its starts chosen as `chosen_starts` says, or (with
    `same_starts`) those of the version its name points to now, with `episodes` of each start and the limits its evals'
    channel takes. `base` is the version the edit was made from, by number: an edit of another than the one the name
    points to is refused (someone edited it meanwhile). Raises `KeyError` for a suite there is not or eval data the
    environment does not have; `ValueError` for another environment than the suite's, an edit that changes nothing,
    or what `make_suite` refuses."""
    for attempt in range(MAKERS):
        current = await suite_of(ledger, name)
        if current is None:
            raise KeyError(f"there is no suite {name!r}")
        if current.environment and current.environment != environment_name:
            raise ValueError(f"the suite {name!r} is of {current.environment}, not {environment_name}: make another")
        if base is not None and base != current.number:
            raise ValueError(f"the suite {name!r} was edited meanwhile: it is at version {current.number}, not {base}")
        if same_starts:
            listed, chosen, data = current.starts, current.chosen, current.eval_data
        else:
            listed, chosen = chosen_starts(environment, rows=rows, seeds=seeds, starts=starts, eval_data=eval_data)
            data = eval_data
        newest = max(each.number for each in await versions_of(ledger, name))
        made = _version(
            name, newest + 1, environment_name, environment, listed, chosen, data, episodes=episodes,
            thinking_tokens=thinking_tokens, answer_tokens=answer_tokens, edited_from=current.number,
        )  # fmt: skip
        if made.configured() == current.configured():
            raise ValueError(f"that is {current.id} as it is: nothing changed")
        fence = await ledger.take(f"suites/{name}")
        try:
            there = await appended(ledger, suite_table(name, FIRST), str(made.number), made.record(), fence)
        except Fenced:  # another editor took the suite's fence after this one did
            await asyncio.sleep(0.05 * (attempt + 1))
            continue
        if there.wrote:
            registry = registry_of(ledger)
            if registry is not None:
                await registry.point_suite(name, made.id, forward=True)
            return made
    raise RuntimeError(f"the suite {name!r} was being edited by others {MAKERS} times over")


async def suite_for(ledger: Ledger, reference: str, environment_name: str, environment: Environment) -> Suite:
    """The version of a suite a reference says (`NAME`: the one its name points to; `NAME@NUMBER`: that one): the
    ledger's, or else the environment's eval data of that name, frozen now as its version 1 (on first use). Raises
    `KeyError` when neither has it, `ValueError` when the ledger's is another environment's."""
    found = await suite_of(ledger, reference)
    if found is not None:
        if found.environment and found.environment != environment_name:
            raise ValueError(f"the suite {found.name!r} is of {found.environment}, not {environment_name}")
        return found
    name, number = parsed(reference)
    if number not in (None, 1):
        raise KeyError(f"there is no version {reference!r} of a suite")
    data = environment.evals()
    if name not in data:
        known = ", ".join(sorted(data)) or "none"
        raise KeyError(f"there is no suite {name!r}, and {environment_name} has no eval data of that name ({known})")
    for attempt in range(MAKERS):  # (another process may make it at the same time: then it is read from the ledger)
        try:
            return await make_suite(ledger, name, environment_name, environment, eval_data=name)
        except Fenced:  # another maker took the suite's fence after this one did
            await asyncio.sleep(0.05 * (attempt + 1))
        except ValueError:  # another maker made it since it was looked for
            if (found := await suite_of(ledger, name)) is None:
                raise
            return found
        if (found := await suite_of(ledger, name)) is not None:
            return found
    raise RuntimeError(f"the suite {name!r} was being made by others {MAKERS} times over, and none of them made it")


async def suite_of(ledger: Ledger, reference: str) -> Suite | None:
    """The version of a suite a reference says, if there is one: `NAME@NUMBER`, that version; `NAME`, the one the
    registry points the name to, else its newest."""
    name, number = parsed(reference)
    versions = await versions_of(ledger, name)
    if not versions:
        return None
    if number is not None:
        return next((each for each in versions if each.number == number), None)
    pointed = await pointed_to(ledger, name)
    return next((each for each in versions if each.number == pointed), versions[-1])


async def pointed_to(ledger: Ledger, name: str) -> int | None:
    """The number of the version the registry beside a ledger points a suite's name to, if it points it anywhere."""
    registry = registry_of(ledger)
    if registry is None:
        return None
    found = next((each for each in await registry.suites() if each.name == name), None)
    return version_number(found.version) if found is not None else None


async def versions_of(ledger: Ledger, name: str) -> list[Suite]:
    """Every version of a suite, oldest first (none: there is no such suite)."""
    records = await ledger.read(suite_table(name, FIRST))
    first = _mapping(records.get(FIRST))
    starts = {} if "starts" in first else await ledger.read(suite_table(name, "starts"))
    return _versions(name, records, starts)


def versions_in(tables: Mapping[str, Mapping[str, JsonValue]], name: str) -> list[Suite]:
    """Every version of a suite, oldest first, from a ledger's tables as read (by name)."""
    return _versions(name, tables.get(suite_table(name, FIRST), {}), tables.get(suite_table(name, "starts"), {}))


def _versions(name: str, records: Mapping[str, JsonValue], starts: Mapping[str, JsonValue]) -> list[Suite]:
    found: list[Suite] = []
    if records.get(FIRST) is not None or starts:
        found.append(_as_suite(name, 1, _mapping(records.get(FIRST)), starts))
    for key, record in records.items():
        if key.isdigit() and int(key) > 1:
            found.append(_as_suite(name, int(key), _mapping(record), {}))
    return sorted(found, key=lambda each: each.number)


def _as_suite(name: str, number: int, about: Mapping[str, Any], starts: Mapping[str, JsonValue]) -> Suite:
    """A version from its record, and its starts by number where the record does not hold them. A record from before
    suites had versions says no more than its environment, its version (`version`), rows, seeds and whether it is
    held out: one held out was the environment's eval data of the suite's name."""
    if "starts" in about:
        listed = [_start(record) for record in about["starts"]]
    else:
        listed = [_start(record) for _, record in sorted(starts.items(), key=lambda item: int(item[0]))]
    environment = str(about.get("environment") or about.get("catalog") or "")  # (a suite made as a catalog's says so)
    held = bool(about.get("held_out"))
    chosen = str(about.get("chosen") or (EVAL_DATA if held else DRAWN))
    eval_data = about.get("eval_data") or (name if "chosen" not in about and held else None)
    return Suite(
        name, environment, listed, float(about.get("made") or 0.0), about.get("rows"), about.get("seeds"),
        about.get("environment_version", about.get("version")), held, number, chosen, eval_data,
        int(about.get("episodes") or 1), about.get("thinking_tokens"), about.get("answer_tokens"),
        about.get("edited_from"),
    )  # fmt: skip


def _start(record: JsonValue) -> Start:
    """A start as a version's record holds it (one that says less reads with what it says)."""
    said = _mapping(record)
    return Start(str(said.get("task") or ""), str(said.get("title") or ""), said.get("seed", 0), said.get("parameters"))


def starts_in(tables: Mapping[str, Mapping[str, Any]], suite: str, number: int = 1) -> dict[str, Any]:
    """A version's starts (version 1 unless `number` says another) by number from 1, from a ledger's tables as read (by
    name), whichever way the suite was written."""
    records = tables.get(suite_table(suite, FIRST), {})
    whole: Any = _mapping(records.get(FIRST if number == 1 else str(number))).get("starts")
    if isinstance(whole, list):
        return {str(place): start for place, start in enumerate(cast(list[Any], whole), start=1)}
    return dict(tables.get(suite_table(suite, "starts"), {})) if number == 1 else {}


def start_identity(start: Mapping[str, Any]) -> str:
    """A short name for a start that is the same in every version that has it: its row and its parameters, hashed."""
    said = f"{start.get('task')}|{start_key(start.get('parameters'))}"
    return hashlib.blake2b(said.encode(), digest_size=6).hexdigest()


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
    """Evals a training run makes of its own checkpoints: `suite` (one version) played by the checkpoint of every
    `every`th step, `episodes` episodes of each start (none: the suite's), between that step and the next.
    `environment` is the suite's environment, and `binding` how its episodes are played (by default every slot from the
    trained channel). `run` gives the eval's run for a step: the same each time it is asked for that step, and one the
    run's episode runners play. `named` is what the run's settings call the suite (`evals.suite`): its name (by
    default), which follows its newest version, or the version's id, which does not."""

    suite: Suite
    environment: Environment
    run: Callable[[int], Awaitable[str]]
    every: int = 1
    episodes: int | None = None
    binding: RunBinding | None = None
    named: str | None = None

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
    episodes: int | None = None,
    binding: RunBinding | None = None,
    started: Mapping[str, JsonValue] | None = None,
    asked_by: str = "by hand",
    reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None,
    hooks: Sequence[Hooks] = (),
    served_by: str | None = None,
    limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Play `suite` (one version) with `subject` (a checkpoint's id; None: the base model, named `base`) served on
    `channel`, `episodes` episodes of each start (none: the suite's), as the run `run`; returns how it went (`played`,
    `solved`, `reward`, and each start's `results`, a `Result` each). `publish` serves a checkpoint on the channel (a
    full one in place of the engines' weights; for an adapter over a full checkpoint, the engines must already hold that
    checkpoint's weights, as `rollout eval` sees to); None: the channel serves `subject` already (a training run's
    newest checkpoint). `reshard` gives its files in the engines' layout (`rollout_train.resharding`); `directory`
    holds its files on this machine.
    What the eval's channel serves is written down (`rollout_train.serving`), so that runners anywhere play it on
    replicas that serve `subject` and no other checkpoint; `served_by` names the channel whose replicas serve it
    (`RUN/NAME`: the training run's, for an eval its schedule asks for), where it is not the eval's own. `limits` are
    the sampling limits the channel was given (the suite's, where it serves the eval's own channel), recorded with who
    played."""
    ledger, blobs = checkpoints.ledger, checkpoints.blobs
    episodes = episodes or suite.episodes
    fence = await ledger.take(scope(run))
    await plan(ledger, run, Plan(environment.program, binding or binding_for(environment, channel)), fence)
    here: dict[str, JsonValue] = {"kind": EVAL, "suite": suite.name, "version": suite.id, "checkpoint": subject}
    here["from"] = subject
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
        "version": suite.id,
    }
    if limits:
        who["limits"] = dict(limits)
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
