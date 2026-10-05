"""Environments from the system's point of view: every environment it knows of (`listed`), and what one is and what
was done with it (`page_of`).

An environment is known by `module:name` (a published one by `NAME@VERSION`, `rollout_train.published`) from several
sources (`SOURCES`), each a function from what the monitor read (`Read`) to what it saw of environments
(`Sighting`s): the cluster config that offers one (`offered`), the runs started on one (`started`: training runs,
evals' runs and checks), the suites whose versions play one (`in_suites`), and the published versions the ledger keeps
(`published`). `listed` folds every sighting of an environment into one line of the list, a published one's with where
its source came from and what its check found (`source_of`); a new source is a function added to `SOURCES`.

An environment's page (`page_of`) is what it says of itself where it loads in the monitor's process (`described`: its
version, description, rows, eval data and curriculum; a published one's as its check recorded it) beside what the
ledger has of it: each row's groups and episodes in the training runs on it (matched by the row's key), those runs, the
suites with a version that plays it, every eval of such a version with its score at this environment's entry, and its
newest check (`rollout env check` with a model's settings: a run of its own, whose groups each say whether every
episode scored the same).
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import JsonValue

from rollout.curriculum import curriculum_of
from rollout.environment import Environment, binding_for
from rollout_train.check import CHECK
from rollout_train.evals import EVAL, parsed, started_version, subject_table, suites_among, versions_in
from rollout_train.monitor.scores import evals_in
from rollout_train.published import EnvironmentVersion, short
from rollout_train.published import parsed as published_parts
from rollout_train.record import ENDS, GROUPS, RESULTS, STARTS, named_runs, newest_record, table

GENERIC = {"environment", "env", "environments", "main"}
"""Names an environment's object is often given, which say nothing of it."""


def readable(environment: str) -> str:
    """An environment's `module:name`, in a word: its object's name, or its package's where that name says nothing; a
    published one's `NAME@VERSION` with the version's first few characters."""
    if (said := published_parts(environment)) is not None:
        return f"{said[0]}@{short(said[1])}"
    module, _, attribute = environment.partition(":")
    if attribute and attribute.lower() not in GENERIC:
        return attribute
    return module.split(".")[0] or environment


@dataclass(frozen=True)
class Read:
    """What the sources read: the ledger's tables, by name; the environments the cluster offers; and the
    registry's names (the version each suite's name points to, under `suites`)."""

    tables: Mapping[str, Mapping[str, JsonValue]]
    offered: frozenset[str] = frozenset()
    names: Mapping[str, Any] = field(default_factory=dict[str, Any])
    published: tuple[EnvironmentVersion, ...] = ()


@dataclass(frozen=True)
class Sighting:
    """One source's word of an environment: a version of it, whether the cluster offers it, a training run on it,
    a suite that plays it, and when something last played it."""

    environment: str
    version: str | None = None
    offered: bool = False
    run: str | None = None
    suite: str | None = None
    at: float | None = None


Source = Callable[[Read], Iterable[Sighting]]


def offered(read: Read) -> Iterable[Sighting]:
    """The environments the cluster offers."""
    return [Sighting(each, offered=True) for each in read.offered]


def started(read: Read) -> Iterable[Sighting]:
    """The environments runs were started on: each start's version of its environment, a training run on it, and when
    each start began. An eval that names no environment played its suite's version's;
    one of several environments is seen in its parts."""
    found: list[Sighting] = []
    for run in named_runs(read.tables):
        for start in read.tables.get(table(run, STARTS), {}).values():
            if not isinstance(start, dict) or start.get("parts"):
                continue
            begun = cast(dict[str, Any], start)
            at = float(begun["started"]) if isinstance(begun.get("started"), int | float) else None
            for environment in _played_by(begun, run, read.tables):
                found.append(
                    Sighting(
                        environment,
                        version=str(begun["version"]) if begun.get("version") is not None else None,
                        run=run if _training(begun) else None,
                        at=at,
                    )
                )
    return found


def in_suites(read: Read) -> Iterable[Sighting]:
    """The environments suites' versions play, with the version of each an entry was made with."""
    return [
        Sighting(entry.environment, version=entry.environment_version, suite=suite)
        for suite in suites_among(read.tables)
        for version in versions_in(read.tables, suite)
        for entry in version.entries
        if entry.environment
    ]


def published(read: Read) -> Iterable[Sighting]:
    """The published environments' versions the ledger keeps, each with the version it says it is, when imported."""
    return [Sighting(each.reference, version=_said_version(each), at=None) for each in read.published]


SOURCES: tuple[Source, ...] = (offered, started, in_suites, published)
"""Where the system knows environments from."""


def _said_version(version: EnvironmentVersion) -> str | None:
    said = version.description.get("version")
    return str(said) if said is not None else None


def source_of(version: EnvironmentVersion) -> dict[str, Any]:
    """Where a published version's source came from, and what its check found: its URL, ref, commit, subdirectory and
    entry point, when it was imported, its findings, and whether every one passed."""
    return {
        "source": version.source, "ref": version.ref, "commit": version.commit, "subdirectory": version.subdirectory,
        "entry_point": version.entry_point, "imported": version.imported, "check": list(version.check),
        "passed": all(bool(each.get("passed")) for each in version.check), "dependencies": list(version.dependencies),
        "version": version.version,
    }  # fmt: skip


def listed(read: Read, sources: Iterable[Source] = SOURCES) -> list[dict[str, Any]]:
    """Every environment the sources saw, by readable name: its `module:name`, the versions of it seen, whether the
    cluster offers it, its training runs (by id) and the suites with a version that plays it, and when a run last
    started on it (`used`)."""
    seen: dict[str, list[Sighting]] = {}
    for source in sources:
        for sighting in source(read):
            seen.setdefault(sighting.environment, []).append(sighting)
    imported = {each.reference: each for each in read.published}
    lines: list[dict[str, Any]] = []
    for environment, sightings in seen.items():
        times = [each.at for each in sightings if each.at is not None]
        lines.append(
            {
                "environment": environment,
                "name": readable(environment),
                "versions": sorted({each.version for each in sightings if each.version}),
                "offered": any(each.offered for each in sightings),
                "runs": sorted({each.run for each in sightings if each.run}),
                "suites": sorted({each.suite for each in sightings if each.suite}),
                "used": max(times) if times else None,
                "published": source_of(imported[environment]) if environment in imported else None,
            }
        )
    return sorted(lines, key=lambda each: (each["name"], each["environment"]))


def described(environment: Environment) -> dict[str, Any]:
    """What an environment says of itself: its version and description, its rows (easiest first: key and title), its
    eval data (each list's name and the row of each start), its curriculum (its class's name, whether it is the
    environment's own, and the rows it unlocks at first and past the hardest one solved) and the kinds of sandbox its
    program declares (`sandboxes`, `sandboxes_of`)."""
    curriculum = curriculum_of(environment)
    return {
        "version": environment.version,
        "description": environment.description.to_json(),
        "rows": [{"key": row.key, "title": row.title} for row in environment.rows()],
        "evals": {name: [start.task for start in starts] for name, starts in environment.evals().items()},
        "curriculum": {
            "name": type(curriculum).__name__,
            "own": callable(getattr(environment, "curriculum", None)),
            "start": getattr(curriculum, "start", None),
            "reach": getattr(curriculum, "reach", None),
        },
        "sandboxes": sandboxes_of(environment),
    }


def sandboxes_of(environment: Environment) -> list[str] | None:
    """The kinds of sandbox an environment's program declares, given its first row's start (a run acquires one of
    each from a pool of its kind); none where its program cannot be made here."""
    try:
        return sorted(binding_for(environment, "policy").pools)
    except Exception:  # (an environment with no rows, or whose program does not import here)
        return None


def page_of(
    read: Read, environment: str, said: Mapping[str, Any] | None = None, error: str | None = None
) -> dict[str, Any] | None:
    """An environment's page, from what it says of itself where it loads (`said`, as `described` gives it; else
    `error` says why it does not load) and what the ledger has of it. None where it neither loads nor is known.

    - `version` (as it loads; else none) and `versions` (seen in training runs' starts);
    - `description`: as it loads; else as its newest run's start says;
    - `rows`: its rows, easiest first (where it does not load: those its runs played, in the order first played), then
      the rows of its eval data it has no row of (`trains` false); each with the groups and episodes the training runs
      played of it, how many of those episodes solved it (`solved`, of `said`: none where no run's results say), the
      mean reward, and how many eval starts it has (`held`);
    - `evals`: each list of eval data and how many starts it has; `curriculum`, as `described` says it;
    - `runs`: the training runs on it, newest first, each with what it played;
    - `suites`: each suite with a version that plays it, the version its name points to, and those versions;
    - `scores`: every eval of such a version, newest first, with its subject and its score at this environment's entry;
    - `check`: its newest check, each group with its rewards and whether every episode scored the same (`flagged`)."""
    known = next((each for each in listed(read) if each["environment"] == environment), None)
    if known is None and said is None:
        return None
    tables, called = read.tables, read.names.get("runs", {})
    runs: list[dict[str, Any]] = []
    rows: dict[str, dict[str, Any]] = {}
    check: dict[str, Any] | None = None
    description: Any = said["description"] if said else None
    newest_start = -1.0
    for run in named_runs(tables):
        starts = cast(dict[str, Any], tables.get(table(run, STARTS), {}))
        mine: list[dict[str, Any]] = [
            cast(dict[str, Any], each)
            for each in (starts[key] for key in sorted(starts, key=lambda key: int(key) if key.isdigit() else 0))
            if isinstance(each, dict) and environment in _played_by(cast(dict[str, Any], each), run, tables)
        ]
        if not mine:
            continue
        latest: dict[str, Any] = mine[-1]
        begun = float(latest.get("started") or 0.0)
        if said is None and latest.get("description") and begun >= newest_start:
            description, newest_start = latest["description"], begun
        if latest.get("kind") == CHECK:
            if check is None or begun >= float(check["started"] or 0.0):
                check = _check(tables, run, latest, called)
            continue
        if not _training(latest):
            continue
        about: dict[str, Any] = latest.get("description") or {}
        reports = bool(about.get("solved", True))
        played = _played(tables, run, rows, reports)
        runs.append(
            {
                "run": run,
                "name": called.get(run, run),
                "version": latest.get("version"),
                "started": latest.get("started"),
                **played,
            }
        )
    runs.sort(key=lambda each: -float(each["started"] or 0.0))
    if said is not None:  # (its own rows, in its order, with what was played of each)
        ordered = [
            rows.pop(row["key"], _row(row["key"], row["title"])) | {"title": row["title"]} for row in said["rows"]
        ]
        rows = {each["key"]: each for each in ordered}  # (those played that it has no longer are left out)
    eval_tasks: dict[str, list[str]] = said["evals"] if said else {}
    for tasks in eval_tasks.values():
        for task in tasks:
            if task not in rows:
                rows[task] = _row(task, task) | {"trains": False}
            rows[task]["held"] += 1
    for each in rows.values():
        rewards = each.pop("rewards")
        each["reward"] = round(sum(rewards) / len(rewards), 4) if rewards else None
    return {
        "environment": environment,
        "name": readable(environment),
        "offered": bool(known and known["offered"]),
        "loads": said is not None,
        "error": error if said is None else None,
        "version": said["version"] if said else None,
        "versions": known["versions"] if known else [],
        "description": description,
        "curriculum": said["curriculum"] if said else None,
        "rows": list(rows.values()),
        "evals": {name: len(tasks) for name, tasks in eval_tasks.items()},
        "runs": runs,
        "suites": _suites(read, environment),
        "scores": _scores(read, environment),
        "check": check,
        "published": known["published"] if known else None,
    }


def _training(start: Mapping[str, Any]) -> bool:
    """Whether a start is a training run's: it names its environment, and is no eval and no check."""
    return bool(start.get("environment")) and start.get("kind") not in (EVAL, CHECK)


def _played_by(start: Mapping[str, Any], run: str, tables: Mapping[str, Mapping[str, JsonValue]]) -> list[str]:
    """The environments a start plays: the one it names; else, for an eval, its suite's version's."""
    if start.get("environment"):
        return [str(start["environment"])]
    if start.get("kind") != EVAL or not start.get("suite"):
        return []
    subject: Any = tables.get(subject_table(str(start["suite"]), run, "subject"), {}).get("subject") or {}
    version = started_version(start, subject)
    if version is None:
        return []
    name, number = parsed(version)
    versions = versions_in(tables, name)
    found = next((each for each in versions if each.number == number), versions[-1] if versions else None)
    return [each for each in found.environments if each] if found is not None else []


def _row(key: str, title: str) -> dict[str, Any]:
    counts: dict[str, Any] = {"groups": 0, "played": 0, "solved": None, "said": 0, "rewards": [], "held": 0}
    return {"key": key, "title": title, "trains": True, **counts}


def _played(
    tables: Mapping[str, Mapping[str, JsonValue]], run: str, rows: dict[str, dict[str, Any]], reports: bool
) -> dict[str, Any]:
    """What a training run played, added to each row's counts (by its key) and summed: its groups with a result, their
    episodes (failed ones too), and how many of those fit to train on (`said`) solved their start (none where its
    results do not say)."""
    groups = cast(dict[str, Any], tables.get(table(run, GROUPS), {}))
    results = cast(dict[str, Any], tables.get(table(run, RESULTS), {}))
    total: dict[str, Any] = {"groups": 0, "played": 0, "solved": None, "said": 0}
    for number in sorted(results, key=lambda key: int(key) if key.isdigit() else 0):
        if not isinstance(results[number], dict):
            continue
        result, group = cast(dict[str, Any], results[number]), cast(dict[str, Any], groups.get(number) or {})
        said: list[Any] = result.get("solved") or []
        given: list[Any] = result.get("rewards") or []
        task = str(group.get("task") or "")
        row = rows.setdefault(task, _row(task, str(group.get("title") or task)))
        rewards = [float(each) for each in given]
        played = len(rewards) + int(result.get("failed") or 0)
        solved = sum(bool(each) for each in said) if reports else None
        for counts in (row, total):
            counts["groups"] += 1
            counts["played"] += played
            if solved is not None:
                counts["solved"] = (counts["solved"] or 0) + solved
                counts["said"] += len(rewards)
        row["rewards"] += rewards
    return total


def _suites(read: Read, environment: str) -> list[dict[str, Any]]:
    """The suites with a version that plays the environment: the version each name points to (by id), whether that one
    plays it, and the versions that do."""
    pointed: Mapping[str, str] = read.names.get("suites", {})
    found: list[dict[str, Any]] = []
    for suite in suites_among(read.tables):
        versions = versions_in(read.tables, suite)
        playing = [each for each in versions if environment in each.environments]
        if not playing:
            continue
        current = next((each for each in versions if each.id == pointed.get(suite)), versions[-1])
        shown = [(each, each.entry(environment)) for each in playing]
        found.append(
            {
                "suite": suite,
                "version": current.id,
                "current": environment in current.environments,
                "versions": [
                    {
                        "id": each.id,
                        "number": each.number,
                        "starts": len(entry.starts) if entry else 0,
                        "made": each.made,
                    }
                    for each, entry in shown
                ],
            }
        )
    return found


def _scores(read: Read, environment: str) -> list[dict[str, Any]]:
    """Every eval of a version that plays the environment, newest first: its run, suite and version, its subject (a
    checkpoint, or a model by name), how far it got, and its score at the environment's entry."""
    found: list[dict[str, Any]] = []
    for each in evals_in(read.tables, read.names):
        entry = next((entry for entry in each["entries"] if entry["environment"] == environment), None)
        if entry is None:
            continue
        found.append(
            {key: each[key] for key in ("run", "name", "suite", "version", "kind", "checkpoint", "model", "started")}
            | {"done": each["done"]}
            | {key: entry[key] for key in ("played", "solved", "said", "share", "reward")}
        )
    return sorted(found, key=lambda each: -float(each["started"] or 0.0))


def _check(
    tables: Mapping[str, Mapping[str, JsonValue]], run: str, start: Mapping[str, Any], called: Mapping[str, str]
) -> dict[str, Any]:
    """A check's run as its page shows it: when it began, the version it checked, how it ended, and each group: its
    row, its rewards, whether each solved, its failed episodes and why, and whether every episode scored the same
    (`flagged`, with what the check said of it)."""
    groups = cast(dict[str, Any], tables.get(table(run, GROUPS), {}))
    results = cast(dict[str, Any], tables.get(table(run, RESULTS), {}))
    ends = cast(dict[str, Any], tables.get(table(run, ENDS), {}))
    shown: list[dict[str, Any]] = []
    for number in sorted(groups, key=lambda key: int(key) if key.isdigit() else 0):
        group, result = cast(dict[str, Any], groups[number]), results.get(number)
        line = cast(dict[str, Any], result) if isinstance(result, dict) else None
        shown.append(
            {
                "group": int(number),
                "task": group.get("task"),
                "episodes": group.get("episodes"),
                "rewards": line.get("rewards", []) if line else None,
                "solved": line.get("solved", []) if line else None,
                "failed": int(line.get("failed") or 0) if line else 0,
                "failures": sorted(set(line.get("failures") or [])) if line else [],
                "flagged": bool(line and line.get("skipped")),
                "skipped": line.get("skipped") if line else None,
            }
        )
    ended = dict(newest_record(ends)) or None
    return {
        "run": run,
        "name": called.get(run, run),
        "started": start.get("started"),
        "version": start.get("version"),
        "ended": ended,
        "groups": shown,
    }
