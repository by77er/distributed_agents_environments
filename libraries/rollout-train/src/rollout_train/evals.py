"""Evaluations: a suite, an eval configuration kept in versions, played by one checkpoint (or the base model), with
nothing trained.

A **suite** is an eval configuration, by name: a list of **entries**, one for each environment it plays (an environment
is in a version once). An entry says its environment (`module:name`, and its version then), the starts every subject
plays of it (each a row's start drawn with a seed of its own), the episodes of each start, and the sampling limits its
episodes take (`thinking_tokens`, `answer_tokens`; none: the channel's own). An entry's starts are one of three: the
environment's eval data of a name (`Environment.evals()`); a start of each of some rows for each of some seeds
(`drawn`); or starts given as they are. Training never draws the environment's eval starts: an entry says whether all of
its starts are among them (`held_out`). Every subject of one version plays the same starts, start for start, so that
subjects compare. A version's starts are numbered from 1 across its entries, in order: the first entry's, then the
next's.

A suite is kept in **versions**. Each is one record with an id of its own (`NAME@NUMBER`), written once under the
suite's fence (`suites/NAME`) and never changed or deleted: editing a suite (`edit_suite`) makes its next version. The
suite's name points to its newest version: the registry beside the ledger holds where
(`rollout_train.registry.SuiteName`), and each edit moves it; a name the registry holds nothing for is its newest
version in the ledger. Every version is in `evaluations/NAME/suite`: version 1 under the key `suite`, each later one
under its number.

Most suites are an environment's eval data, frozen as version 1 of a suite of that name the first time it is played
(`suite_for`). One can be made by hand too (`make_suite`, of entries `suite_entry` makes). Two makers of one suite at
once leave one of their suites whole: the one whose record was appended, which the other then reads and plays. Two
editors at once make two versions, one after the other, and the name points to the later.

An **eval** is one version of a suite played by one subject: a checkpoint (by any reference
`rollout_train.registry.resolved` takes), or the base model. It is a run of its own, registered and fenced like any run,
whose start says what it is (`kind: eval`, the suite, its version as `suite_version`, the checkpoint; and, for a version
of one entry, its environment's version and description, as a training run's start does). It serves the subject on the
channel. Each entry is played by a run of its own: the eval's run, for a version of one entry; else a run for each entry
(its **parts**), each with its own plan (the entry's program, and its binding: its tool sets, pools and limits), so a
runner plays an entry only where it has what the entry's environment needs. A part's start says the eval it is part of
(`part_of`). Each run asks for one group per start of its entry with that entry's episodes each (runners play them as
they play any run's), and each group's result is written to that run's own `results`, so a part reads like any run. The
eval records each episode's outcome under `evaluations/SUITE/EVAL/results` (`START-EPISODE`, the start by its number in
the version), beside a record of the subject, the version it played and its parts (`evaluations/SUITE/EVAL/subject`),
and, once every start has been played, each entry's scores (`scores`, in the same table): episodes played, solved where
its environment's results say it, and the mean reward. Entries are scored apart: environments' rewards do not compare.
Started again, it goes on: what it decided and what it recorded are not done twice.

A training run can evaluate its own checkpoints as it makes them (a `Schedule`): the loop plays the suite with the
checkpoint of every `every`th step between that step and the next, on the channel that already serves it (the policy is
the same whatever the environment), each eval a run of its own, and each entry played with its own binding
(`rollout_train.loop.train`). Each step is decided with the version the suite's name points to then.
"""

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue

from rollout.environment import Environment, Start, binding_for, drawn, held_out, start_key
from rollout.harness.runner import RunBinding
from rollout.names import named
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest
from rollout_train.launches import EVAL
from rollout_train.ledger import Fence, Fenced, Ledger, between
from rollout_train.record import (
    ENDS,
    FINISHED,
    GROUPS,
    RESULTS,
    STARTS,
    Result,
    described,
    mapping,
    results,
    scope,
    start_header,
    table,
)
from rollout_train.registry import registry_of, valid, version_number
from rollout_train.rollouts.episodes import Episode
from rollout_train.rollouts.scheduler import EPISODES, Hooks, Plan, episodes_of, plan
from rollout_train.serving import Serving, record_serving
from rollout_train.trainer import WEIGHTS

EVALUATIONS = "evaluations/"
NOTHING_TRAINED = "an evaluation trains on nothing"
FIRST = "suite"
"""The key of a suite's version 1 in its table; each later version's is its number."""
SUITE_VERSION = "suite_version"
"""What an eval's start says the version it plays under (by id); its `version` is its environment's, as a training
run's start says."""
SCORES = "scores"
"""The key of an eval's entries' scores in its subject's table, once every start has been played."""
EVAL_DATA, DRAWN, GIVEN = "eval data", "rows and seeds", "starts"
"""How an entry's starts were chosen: the environment's eval data of a name; a start of each row for each seed; or
given as they are."""
MAKERS = 5
"""How many times a suite that others are making or editing at the same time is tried again, before giving up."""


def suite_table(suite: str, part: str) -> str:
    """A suite's table: `suite` (its versions)."""
    return f"{EVALUATIONS}{suite}/{part}"


def subject_table(suite: str, subject: str, part: str) -> str:
    """A subject's table under a suite: `subject` (who played, and its scores) or `results` (how each episode went)."""
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


def started_version(start: Mapping[str, Any], subject: Mapping[str, Any]) -> str | None:
    """The version an eval's start played, by id: what it says (`suite_version`), else what its subject's record says
    (a start written before it said so has its environment's version under `version`, as a training run's start
    does)."""
    found = start.get(SUITE_VERSION) or subject.get("version")
    return str(found) if found else None


def played_version(subject: Mapping[str, Any]) -> str | None:
    """The version an eval played, by id, from its subject's record."""
    found = subject.get("version")
    return str(found) if found else None


@dataclass(frozen=True)
class SuiteEntry:
    """One environment of a suite's version: what every subject plays of it, start for start, and how."""

    environment: str
    """The environment, as `module:name`."""
    starts: list[Start]
    """Its starts, in order."""
    environment_version: str | None = None
    """The environment's version when the entry was made."""
    chosen: str = DRAWN
    """How its starts were chosen: `EVAL_DATA`, `DRAWN` or `GIVEN`."""
    eval_data: str | None = None
    """The name of the environment's eval data its starts are, where they are."""
    rows: list[str] | None = None
    """The rows it names, by key."""
    seeds: list[int] | None = None
    held_out: bool = False
    """Whether every start is one of the environment's eval starts, which training never draws."""
    episodes: int = 1
    """Episodes of each start an eval plays, unless it is asked for another number."""
    thinking_tokens: int | None = None
    """Its episodes' tokens of thinking per turn, and of answer after it; none: the channel's own (which may be no
    budget)."""
    answer_tokens: int | None = None

    @property
    def limits(self) -> dict[str, int]:
        """The sampling limits it gives its episodes, by a channel's names for them (`thinking_tokens`,
        `answer_tokens`), where it gives any."""
        said = {"thinking_tokens": self.thinking_tokens, "answer_tokens": self.answer_tokens}
        return {key: value for key, value in said.items() if value is not None}

    def configured(self) -> tuple[Any, ...]:
        """What makes it what it is in an eval: the environment and its version, the starts, the episodes and the
        limits."""
        return (
            self.environment, self.environment_version, self.starts, self.episodes, self.thinking_tokens,
            self.answer_tokens,
        )  # fmt: skip


@dataclass(frozen=True)
class Suite:
    """One version of a suite: its entries, each what every subject of it plays, start for start, and how."""

    name: str
    entries: list[SuiteEntry]
    made: float = 0.0
    number: int = 1
    edited_from: int | None = None
    """The version it was edited from, by number."""

    @property
    def id(self) -> str:
        return version_id(self.name, self.number)

    @property
    def environments(self) -> list[str]:
        """Its entries' environments, in order."""
        return [each.environment for each in self.entries]

    @property
    def starts(self) -> list[Start]:
        """Every start, numbered from 1 in this order: the first entry's, then the next's."""
        return [start for each in self.entries for start in each.starts]

    @property
    def held_out(self) -> bool:
        """Whether every entry is held out of training."""
        return bool(self.entries) and all(each.held_out for each in self.entries)

    def entry(self, environment: str) -> SuiteEntry | None:
        """Its entry of an environment, if it has one."""
        return next((each for each in self.entries if each.environment == environment), None)

    def record(self) -> dict[str, Any]:
        """What the ledger keeps of it (its name and number are its table and key)."""
        return {"made": self.made, "edited_from": self.edited_from, "entries": [asdict(each) for each in self.entries]}

    def configured(self) -> tuple[Any, ...]:
        """What makes an eval of it what it is: each entry's configuration, in order."""
        return tuple(each.configured() for each in self.entries)


def offsets(suite: Suite) -> list[int]:
    """How many starts come before each entry's first: its starts are numbered from that number and 1."""
    found: list[int] = []
    before = 0
    for each in suite.entries:
        found.append(before)
        before += len(each.starts)
    return found


def chosen_starts(
    environment: Environment,
    *,
    rows: Sequence[str] | None = None,
    seeds: Sequence[int] = (),
    starts: Sequence[Start] | None = None,
    eval_data: str | None = None,
) -> tuple[list[Start], str]:
    """An entry's starts, and how they were chosen: the environment's eval data of the name `eval_data`; `starts`, as
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


def suite_entry(
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
) -> SuiteEntry:
    """An entry of `environment` (named `environment_name`), its starts chosen as `chosen_starts` says, with `episodes`
    of each start and the limits its episodes take. Raises `ValueError` for a row the environment does not have, no
    starts, or a count that is no whole number of 1 at least; `KeyError` for eval data the environment does not
    have."""
    listed, chosen = chosen_starts(environment, rows=rows, seeds=seeds, starts=starts, eval_data=eval_data)
    if not listed:
        raise ValueError("an entry needs a start at least")
    counts: dict[str, Any] = {"episodes": episodes, "thinking_tokens": thinking_tokens, "answer_tokens": answer_tokens}
    for what, count in counts.items():
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ValueError(f"{what} is a whole number, 1 at least (not {count!r})")
    held = held_out(environment)
    return SuiteEntry(
        environment_name, listed, environment.version, chosen, eval_data if chosen == EVAL_DATA else None,
        list(dict.fromkeys(start.task for start in listed)), list(dict.fromkeys(start.seed for start in listed)),
        all(start_key(start.parameters) in held for start in listed), episodes, thinking_tokens, answer_tokens,
    )  # fmt: skip


def _version(name: str, number: int, entries: Sequence[SuiteEntry], edited_from: int | None = None) -> Suite:
    """A version of a suite, made now. Raises `ValueError` for no entries, or an environment in two."""
    if not entries:
        raise ValueError("a suite needs an entry at least")
    environments = [each.environment for each in entries]
    if twice := sorted({each for each in environments if environments.count(each) > 1}):
        raise ValueError(f"an environment is in a version once ({', '.join(twice)} is in more than one entry)")
    return Suite(name, list(entries), round(time.time(), 1), number, edited_from)


async def make_suite(ledger: Ledger, name: str, entries: Sequence[SuiteEntry]) -> Suite:
    """Make version 1 of a suite of these entries (`suite_entry`). Raises `ValueError` for a name that is no name or is
    taken (a suite is edited instead: `edit_suite`), no entries, or an environment in two."""
    name = valid(name)
    if await suite_of(ledger, name) is not None:
        raise ValueError(f"there is a suite {name!r} already: edit it (a version of its own), or make another")
    made = _version(name, 1, entries)
    fence = await ledger.take(f"suites/{name}")
    there = await ledger.append_returning(suite_table(name, FIRST), FIRST, made.record(), fence)
    if there.wrote:
        return made
    return _as_suite(name, 1, mapping(there.record))  # (another maker's, made meanwhile)


async def edit_suite(ledger: Ledger, name: str, entries: Sequence[SuiteEntry], *, base: int | None = None) -> Suite:
    """Make a suite's next version of these entries, and point its name to it. An entry whose starts are those of the
    current version's entry of its environment keeps how they were chosen. `base` is the version the edit was made
    from, by number: an edit of another than the one the name points to is refused (someone edited it meanwhile).
    Raises `KeyError` for a suite there is not; `ValueError` for an edit that changes nothing, or what `make_suite`
    refuses."""
    for attempt in range(MAKERS):
        current = await suite_of(ledger, name)
        if current is None:
            raise KeyError(f"there is no suite {name!r}")
        if base is not None and base != current.number:
            raise ValueError(f"the suite {name!r} was edited meanwhile: it is at version {current.number}, not {base}")
        kept = [_kept(each, current.entry(each.environment)) for each in entries]
        newest = max(each.number for each in await versions_of(ledger, name))
        made = _version(name, newest + 1, kept, current.number)
        if made.configured() == current.configured():
            raise ValueError(f"that is {current.id} as it is: nothing changed")
        fence = await ledger.take(f"suites/{name}")
        try:
            there = await ledger.append_returning(suite_table(name, FIRST), str(made.number), made.record(), fence)
        except Fenced:  # another editor took the suite's fence after this one did
            await asyncio.sleep(0.05 * (attempt + 1))
            continue
        if there.wrote:
            registry = registry_of(ledger)
            if registry is not None:
                await registry.point_suite(name, made.id, forward=True)
            return made
    raise RuntimeError(f"the suite {name!r} was being edited by others {MAKERS} times over")


def _kept(entry: SuiteEntry, before: SuiteEntry | None) -> SuiteEntry:
    """An edited entry, saying how its starts were chosen as the entry before it says, where its starts are those."""
    if before is None or before.starts != entry.starts or entry.chosen == before.chosen:
        return entry
    return SuiteEntry(
        entry.environment, entry.starts, entry.environment_version, before.chosen, before.eval_data, before.rows,
        before.seeds, entry.held_out, entry.episodes, entry.thinking_tokens, entry.answer_tokens,
    )  # fmt: skip


async def suite_for(ledger: Ledger, reference: str, environment_name: str, environment: Environment) -> Suite:
    """The version of a suite a reference says (`NAME`: the one its name points to; `NAME@NUMBER`: that one): the
    ledger's, whatever its environments, or else `environment`'s eval data of that name, frozen now as its version 1 (on
    first use). Raises `KeyError` when neither has it."""
    found = await suite_of(ledger, reference)
    if found is not None:
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
            return await make_suite(ledger, name, [suite_entry(environment_name, environment, eval_data=name)])
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
    return _versions(name, await ledger.read(suite_table(name, FIRST)))


def versions_in(tables: Mapping[str, Mapping[str, JsonValue]], name: str) -> list[Suite]:
    """Every version of a suite, oldest first, from a ledger's tables as read (by name)."""
    return _versions(name, tables.get(suite_table(name, FIRST), {}))


def _versions(name: str, records: Mapping[str, JsonValue]) -> list[Suite]:
    found: list[Suite] = []
    if records.get(FIRST) is not None:
        found.append(_as_suite(name, 1, mapping(records.get(FIRST))))
    for key, record in records.items():
        if key.isdigit() and int(key) > 1:
            found.append(_as_suite(name, int(key), mapping(record)))
    return sorted(found, key=lambda each: each.number)


def _as_suite(name: str, number: int, about: Mapping[str, Any]) -> Suite:
    """A version from its record."""
    entries = [_as_entry(mapping(each)) for each in cast(list[Any], about.get("entries") or [])]
    return Suite(name, entries, float(about.get("made") or 0.0), number, about.get("edited_from"))


def _as_entry(about: Mapping[str, Any]) -> SuiteEntry:
    """An entry from its record."""
    return SuiteEntry(
        str(about.get("environment") or ""), [_start(record) for record in cast(list[Any], about.get("starts") or [])],
        about.get("environment_version"), str(about.get("chosen") or DRAWN), about.get("eval_data"), about.get("rows"),
        about.get("seeds"), bool(about.get("held_out")), int(about.get("episodes") or 1), about.get("thinking_tokens"),
        about.get("answer_tokens"),
    )  # fmt: skip


def _start(record: JsonValue) -> Start:
    """A start as a version's record holds it (one that says less reads with what it says)."""
    said = mapping(record)
    return Start(str(said.get("task") or ""), str(said.get("title") or ""), said.get("seed", 0), said.get("parameters"))


def start_identity(start: Mapping[str, Any]) -> str:
    """A short name for a start that is the same in every version that has it: its row and its parameters, hashed."""
    said = f"{start.get('task')}|{start_key(start.get('parameters'))}"
    return hashlib.blake2b(said.encode(), digest_size=6).hexdigest()


def suites_among(names: Iterable[str]) -> list[str]:
    """The suites tables are of, from their names."""
    found = {between(each, EVALUATIONS, "/suite") for each in names}
    return sorted(name for name in found if name and "/" not in name)


async def suites_in(ledger: Ledger) -> list[str]:
    """Every suite a ledger has, by name."""
    return suites_among(await ledger.tables())


def parts_of(tables: Mapping[str, Mapping[str, Any]], suite: str, run: str) -> list[dict[str, Any]]:
    """The runs an eval played its version's entries in, from a ledger's tables as read (by name): each its
    `environment`, its `run`, its `episodes` of each start, and the number of its first start in the version less one
    (`offset`)."""
    about = mapping(tables.get(subject_table(suite, run, "subject"), {}).get("subject"))
    version = next((each for each in versions_in(tables, suite) if each.id == played_version(about)), None)
    said = cast(list[Any], about.get("parts") or [])
    places = offsets(version) if version is not None and len(version.entries) == len(said) else [0] * len(said)
    return [dict(mapping(each)) | {"offset": place} for each, place in zip(said, places, strict=True)]


def eval_episodes(tables: Mapping[str, Mapping[str, Any]], suite: str, run: str) -> dict[str, Any]:
    """An eval's episodes as its runs recorded them, by `START/EPISODE` (the start by its number in the version), from a
    ledger's tables as read (by name)."""
    found: dict[str, Any] = {}
    for part in parts_of(tables, suite, run):
        for key, record in tables.get(table(str(part["run"]), EPISODES), {}).items():
            group, _, episode = key.partition("/")
            if group.isdigit():
                found[f"{int(group) + int(part['offset'])}/{episode}"] = record
    return found


def environments_of(suite: Suite, given: Mapping[str, Environment] | None = None) -> dict[str, Environment]:
    """Each entry's environment, by `module:name`: as `given`, else imported here. Raises `KeyError` for one that does
    not load here."""
    found: dict[str, Environment] = {}
    for each in suite.environments:
        if given is not None and each in given:
            found[each] = given[each]
            continue
        try:
            found[each] = named(each)
        except Exception as error:  # (whatever importing it raises: it is not here)
            raise KeyError(f"the environment {each!r} does not load here ({error})") from None
    return found


def limited(binding: RunBinding, entry: SuiteEntry) -> RunBinding:
    """A binding whose recorded models take an entry's sampling limits, where it gives any."""
    if not entry.limits:
        return binding
    models = dict(binding.models)
    for slot, model in binding.models.items():
        if model.recorded is not None:
            sampling = model.recorded.sampling.model_copy(update=entry.limits)
            models[slot] = model.model_copy(
                update={"recorded": model.recorded.model_copy(update={"sampling": sampling})}
            )
    return binding.model_copy(update={"models": models})


def entry_scores(environment: str, rewards: Sequence[float], solved: Sequence[bool] | None) -> dict[str, Any]:
    """How an eval did at an entry, from the rewards of its episodes that count, and whether each solved its start
    (None where its environment's results do not say): `played`, `solved` and the share solved (`share`), and the mean
    `reward`."""
    return {
        "environment": environment,
        "played": len(rewards),
        "solved": sum(solved) if solved is not None else None,
        "share": round(sum(solved) / len(solved), 4) if solved else None,
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
    }


class Publisher(Protocol):
    """Serves new weights on a channel from now on (with `full`, a full checkpoint's in place of the engines'); returns
    the number its samples are stamped with (the checkpoint's depth)."""

    async def __call__(
        self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
    ) -> int: ...


class EvalRuns(Protocol):
    async def __call__(self, step: int, part: int | None = None) -> str:
        """The run of the eval of the checkpoint made at `step`; with `part`, the run that plays that entry (by its
        number from 1) of a suite of several. The same each time it is asked for."""
        ...


@dataclass(frozen=True)
class Schedule:
    """Evals a training run makes of its own checkpoints: `suite` (one version) played by the checkpoint of every
    `every`th step, `episodes` episodes of each start (none: each entry's), between that step and the next. `run` gives
    the eval's run for a step, and each part's: runs the run's episode runners play. `environments` are the entries'
    environments, by `module:name` (any not given are imported), and `binding` how an environment's episodes are played
    (by default every slot from the trained channel); each entry's limits are its own. `named` is what the run's
    settings call the suite (`evals.suite`): its name (by default), which follows its newest version, or the version's
    id, which does not."""

    suite: Suite
    run: EvalRuns
    every: int = 1
    episodes: int | None = None
    environments: Mapping[str, Environment] = field(default_factory=dict[str, Environment])
    binding: Callable[[Environment], RunBinding] | None = None
    named: str | None = None

    def due(self, checkpoint: Checkpoint, run: str) -> bool:
        """Whether `checkpoint` is evaluated: a checkpoint `run` made at a step the schedule names."""
        return checkpoint.run == run and checkpoint.step is not None and checkpoint.step % self.every == 0


async def evaluate(
    checkpoints: Checkpoints,
    *,
    run: str,
    suite: Suite,
    subject: str | None,
    base: str | None,
    channel: str,
    directory: Path,
    publish: Publisher | None,
    environments: Mapping[str, Environment] | None = None,
    binding: Callable[[Environment], RunBinding] | None = None,
    parts: Callable[[int], Awaitable[str]] | None = None,
    episodes: int | None = None,
    started: Mapping[str, JsonValue] | None = None,
    asked_by: str = "by hand",
    reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None,
    hooks: Sequence[Hooks] = (),
) -> dict[str, Any]:
    """Play `suite` (one version) with `subject` (a checkpoint's id; None: the base model, named `base`) served on
    `channel`, `episodes` episodes of each start (none: each entry's), as the run `run`. Returns how it went: `played`,
    `solved`, `reward` and every start's `results` (a `Result` each), and each entry's (`entries`: its environment, the
    run that played it, its scores, `entry_scores`, and its `results`).

    `environments` are the entries' environments by `module:name` (any not given are imported: `environments_of`);
    `binding` says how an environment's episodes are played (by default every slot from `channel`), and each entry's
    binding takes its limits. A suite of several entries plays each in a run of its own: `parts` gives the run of an
    entry, by its number from 1 (by default `RUN-NUMBER`). `publish` serves a checkpoint on the channel (a full one in
    place of the engines' weights; for an adapter over a full checkpoint, the engines must already hold that
    checkpoint's weights, as `rollout eval` sees to); None: the channel serves `subject` already (a training run's
    newest checkpoint). `reshard` gives the files its engines load (made by a bridge: `rollout_train.bridges`);
    `directory` holds its files on this machine. What the eval's channel serves is written down for each of its runs
    (`rollout_train.serving`), so that runners anywhere play it on replicas that serve `subject` and no other
    checkpoint. Raises `KeyError` for an entry's environment that does not load here."""
    ledger, blobs = checkpoints.ledger, checkpoints.blobs
    loaded = environments_of(suite, environments)
    fence = await ledger.take(scope(run))
    several = len(suite.entries) > 1
    runs = (
        [run] if not several else [await (parts or _part(run))(number) for number in range(1, len(suite.entries) + 1)]
    )
    fences = [fence] if not several else [await ledger.take(scope(each)) for each in runs]
    counts = [episodes or entry.episodes for entry in suite.entries]
    here: dict[str, JsonValue] = {"kind": EVAL, "suite": suite.name, SUITE_VERSION: suite.id, "checkpoint": subject}
    here |= start_header(**{"from": subject})
    first = loaded[suite.entries[0].environment]
    whole = {**here, **(described(first) if not several else {"parts": cast(JsonValue, runs)}), **(started or {})}
    await ledger.append(table(run, STARTS), str(fence.number), whole, fence)
    for number, (entry, played_by, its_fence) in enumerate(zip(suite.entries, runs, fences, strict=True), start=1):
        environment = loaded[entry.environment]
        bound = limited(binding(environment) if binding is not None else binding_for(environment, channel), entry)
        await plan(ledger, played_by, Plan(environment.program, bound), its_fence)
        if several:
            part = {**here, **described(environment), **(started or {}), "environment": entry.environment}
            await ledger.append(
                table(played_by, STARTS), str(its_fence.number), part | {"part_of": run} | {"entry": number}, its_fence
            )
    same = len(set(counts)) == 1
    who: dict[str, JsonValue] = {
        "kind": "checkpoint" if subject else "model",
        "checkpoint": subject,
        "model": base,
        "episodes": counts[0] if same else None,
        "asked_by": asked_by,
        "decided": round(time.time(), 1),
        "run": run,
        "version": suite.id,
        "parts": [
            {
                "environment": entry.environment,
                "run": played_by,
                "episodes": count,
                "limits": cast(JsonValue, entry.limits),
            }
            for entry, played_by, count in zip(suite.entries, runs, counts, strict=True)
        ],
    }
    await ledger.append(subject_table(suite.name, run, "subject"), "subject", who, fence)

    def note(kind: str, payload: Mapping[str, JsonValue]) -> None:
        event: dict[str, JsonValue] = {"kind": kind, "run": run, "at": round(time.time(), 3), **payload}
        for hook in hooks:
            hook.on_note(event)

    serving: Serving
    if subject is None:  # (the model the channel's engines are started with)
        serving = Serving(channel, model=base, max_lag=0)
    elif publish is None:  # (the channel serves it already)
        known = await checkpoints.checkpoint(subject)
        serving = Serving(channel, known.id, known.depth, known.kind, model=base, max_lag=0)
    else:
        served = await checkpoints.checkpoint(subject)
        if served.weights is None:
            raise ValueError(f"{subject} was released: its weights were deleted, so it cannot be played")
        manifest = await reshard(served, fence) if reshard is not None else served.weights
        serving = Serving(channel, served.id, served.depth, served.kind, manifest, model=base, max_lag=0)
    for each, its_fence in {run: fence, **dict(zip(runs, fences, strict=True))}.items():
        await record_serving(ledger, each, serving, its_fence)
    if subject is not None and publish is not None:
        assert serving.files is not None
        files = await checkpoints.files(serving.files, directory / subject / ("resharded" if reshard else WEIGHTS))
        version = await publish(channel, subject, str(files), serving.depth, full=serving.kind == "full")
        note("published", {"channel": channel, "adapter": subject, "version": version})

    for entry, played_by, its_fence, count in zip(suite.entries, runs, fences, counts, strict=True):
        decided = await ledger.read(table(played_by, GROUPS))
        for number, start in enumerate(entry.starts, start=1):
            if str(number) not in decided:
                group: JsonValue = {
                    "task": start.task,
                    "title": start.title,
                    "parameters": start.parameters,
                    "episodes": count,
                    "decided": round(time.time(), 1),
                }
                await ledger.append(
                    table(played_by, GROUPS), str(number), group, its_fence
                )  # runners play it from here

    async def played(place: int, number: int) -> list[Episode]:
        """The episodes of the start `number` of the entry at `place`, once they have all ended, recorded."""
        entry, played_by, its_fence = suite.entries[place], runs[place], fences[place]
        found = await episodes_of(ledger, blobs, played_by, number, counts[place])
        start, at = entry.starts[number - 1], offsets(suite)[place] + number
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
            await ledger.append(subject_table(suite.name, run, "results"), f"{at}-{episode.number}", outcome, fence)
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
        await ledger.append(table(played_by, RESULTS), str(number), line.to_json(), its_fence)
        note("result", {"group": number, **line.to_json()} | ({"part": played_by} if several else {}))
        return found

    every = await asyncio.gather(
        *(asyncio.gather(*(played(place, number) for number in range(1, len(entry.starts) + 1)))
          for place, entry in enumerate(suite.entries))
    )  # fmt: skip
    entries: list[dict[str, Any]] = []
    for place, entry in enumerate(suite.entries):
        done = [episode for found in every[place] for episode in found if episode.trainable]
        says = loaded[entry.environment].description.solved
        said = entry_scores(
            entry.environment, [each.reward for each in done], [each.solved for each in done] if says else None
        )
        entries.append(
            said | {"run": runs[place], "results": await results(ledger, runs[place])}
        )  # (as first recorded)
    kept: dict[str, JsonValue] = {
        "entries": [{key: value for key, value in each.items() if key != "results"} for each in entries]
    }
    await ledger.append(subject_table(suite.name, run, "subject"), SCORES, kept, fence)
    for each, its_fence in {**dict(zip(runs, fences, strict=True)), run: fence}.items():
        await ledger.append(
            table(each, ENDS), str(its_fence.number), {"how": FINISHED, "at": round(time.time(), 1)}, its_fence
        )
    everything = [episode for found in every for listed in found for episode in listed]
    rewards = [episode.reward for episode in everything if episode.trainable]
    return {
        "played": len(everything),
        "solved": sum(episode.solved for episode in everything),
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "results": [line for each in entries for line in each["results"]],
        "entries": entries,
    }


def _part(run: str) -> Callable[[int], Awaitable[str]]:
    """The run of each entry of an eval of several, by its number: `RUN-NUMBER`."""

    async def part(number: int) -> str:
        return f"{run}-{number}"

    return part
