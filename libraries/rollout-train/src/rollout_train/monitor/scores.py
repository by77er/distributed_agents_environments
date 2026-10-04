"""Evals from a checkpoint's point of view: every eval a checkpoint has had, and its scores along its line.

An eval is a run of its own (`rollout_train.evals`): who played is in `evaluations/SUITE/EVAL/subject`, how each episode
went in `evaluations/SUITE/EVAL/results`, and its start (`runs/EVAL/starts`) says when it began and, for one a training
run's schedule asked for, which run and step (`by`, `step`); that run's `evals` table says so too. A checkpoint's
evals (`evals_of`) are every eval whose subject it is, by hand or by a schedule, each with its score at each entry of
its suite (an environment: entries are scored apart, as environments' rewards do not compare): the mean reward of its
episodes, and the share solved where its episodes say whether they solved their start.

A checkpoint's line (`path_of`) is its first parents back to the root, and the root's base model before it: the model
every checkpoint on it builds on. It crosses runs (a run that starts from another's checkpoint), merges (a full
checkpoint made from an adapter) and forks. Each point on it has its score at each entry of each version of a suite,
pooled over every eval of it on that version (scores of two versions do not compare); the base model's are the evals of
that model by name.

A subject is what an eval played: a checkpoint (by id) or a base model (by name). The subjects (`subjects_in`) are every
one that has had an eval, each with what it is and its evals; a subject's history (`history_of`) is every eval it has
had, newest first, across suites, versions and environments.
"""

from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoint, short
from rollout_train.evals import (
    EVALUATIONS,
    eval_episodes,
    parsed,
    parts_of,
    played_version,
    subject_table,
    versions_in,
)
from rollout_train.ledger import between
from rollout_train.monitor.statistics import reported
from rollout_train.record import EVALS, GROUPS, STARTS, newest_record, table

CHECKPOINT, MODEL = "checkpoint", "model"
"""What a subject is: a checkpoint, or a base model."""
BY_HAND, BY_SCHEDULE = "by hand", "schedule"
"""Who asked for an eval: someone, by hand (`rollout eval`, the page), or a training run's schedule."""


def evals_of(
    tables: Mapping[str, Mapping[str, JsonValue]], checkpoint: str, names: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Every eval whose subject is `checkpoint` (by id), newest first: the suite and the version it played, the eval's
    run, who asked for it (by hand, or the training run and step whose schedule did), episodes of each start, how far
    it got, its score, and its score at each entry (`entries`)."""
    return _shown([each for each in _evals(tables, names or {}) if _subject_of(each) == (CHECKPOINT, checkpoint)])


def subjects_in(
    tables: Mapping[str, Mapping[str, JsonValue]],
    checkpoints: Mapping[str, Checkpoint],
    names: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every subject that has had an eval, the one evaluated last first: what it is, and its evals (`_subject`)."""
    said = names or {}
    evals: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for each in _evals(tables, said):
        evals.setdefault(_subject_of(each), []).append(each)
    shorter = short(checkpoints)
    found = [
        _subject(kind, reference, checkpoints, shorter, said, listed) for (kind, reference), listed in evals.items()
    ]
    return sorted(found, key=lambda each: -(each["started"] or 0.0))


def history_of(
    tables: Mapping[str, Mapping[str, JsonValue]],
    checkpoints: Mapping[str, Checkpoint],
    kind: str,
    reference: str,
    names: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """A subject's history: what it is (`subject`) and every eval it has had (`evals`), newest first, each with its
    suite, the version it played, who asked for it and when, and its score at each entry of that version. None for a
    subject that is neither a checkpoint here, nor a base model a checkpoint here builds on, nor evaluated."""
    said = names or {}
    found = [each for each in _evals(tables, said) if _subject_of(each) == (kind, reference)]
    known = (
        reference in checkpoints if kind == CHECKPOINT else any(each.base == reference for each in checkpoints.values())
    )
    if not found and not known:
        return None
    subject = _subject(kind, reference, checkpoints, short(checkpoints), said, found)
    return {"subject": subject, "evals": _shown(found)}


def path_of(
    tables: Mapping[str, Mapping[str, JsonValue]],
    checkpoints: Mapping[str, Checkpoint],
    checkpoint: str,
    names: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A checkpoint's line, from the base model to it along first parents (`points`, by depth: the base model at 0),
    each point with what it is (its run, step, weights and bookmarks) and its score at each version of a suite
    (`scores`, by the version's id, pooled over every eval of it there; and at each of the version's entries,
    `entries`, by environment); and the versions any point was evaluated on (`suites`), each with its suite's name,
    its number, its environments, and how the page labels it (the name, and the version where the line has more than
    one of that suite)."""
    said = names or {}
    called: Mapping[str, str] = said.get("runs", {})
    marks: dict[str, list[str]] = {}
    for mark, each in said.get("bookmarks", {}).items():
        marks.setdefault(str(each), []).append(mark)
    line: list[Checkpoint] = []
    at = checkpoints.get(checkpoint)
    while at is not None and at.id not in {each.id for each in line}:  # (a cycle would be no line: stopped)
        line.insert(0, at)
        at = checkpoints.get(at.parents[0]) if at.parents else None
    evals = _evals(tables, said)
    shorter = short(checkpoints)
    base = line[0].base if line else None
    points: list[dict[str, Any]] = [
        {
            "id": None,
            "short": "base",
            "depth": 0,
            "model": base,
            "run": None,
            "name": None,
            "step": None,
            "kind": "model",
            "bookmarks": [],
            "scores": _pooled([each for each in evals if each["kind"] == "model" and each["model"] == base]),
        }
    ]
    for each in line:
        points.append(
            {
                "id": each.id,
                "short": shorter.get(each.id, each.id),
                "depth": each.depth,
                "model": None,
                "run": each.run,
                "name": called.get(each.run, each.run) if each.run else None,
                "step": each.step,
                "kind": each.kind,
                "bookmarks": sorted(marks.get(each.id, [])),
                "scores": _pooled([found for found in evals if found["checkpoint"] == each.id]),
            }
        )
    versions = sorted({version for point in points for version in point["scores"]}, key=_by_number)
    named = [parsed(version)[0] for version in versions]
    suites: list[dict[str, Any]] = []
    for version in versions:
        name, number = parsed(version)
        label = name if named.count(name) == 1 else f"{name} v{number}"
        suites.append({"suite": version, "name": name, "number": number, "label": label}
                      | {"environments": _environments(tables, name, number or 1)})  # fmt: skip
    return {"checkpoint": checkpoint, "points": points if line else [], "suites": suites}


def _shown(evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evals as the page lists them, newest first: without each episode's reward."""
    return [
        {key: value for key, value in each.items() if key != "rewards"}
        | {"entries": [{key: value for key, value in entry.items() if key != "rewards"} for entry in each["entries"]]}
        for each in sorted(evals, key=lambda each: -(each["started"] or 0.0))
    ]


def _subject_of(each: Mapping[str, Any]) -> tuple[str, str]:
    """What an eval played: a base model by name, or a checkpoint by id."""
    if each["kind"] == MODEL:
        return MODEL, str(each["model"])
    return CHECKPOINT, str(each["checkpoint"])


def _subject(
    kind: str,
    reference: str,
    checkpoints: Mapping[str, Checkpoint],
    shorter: Mapping[str, str],
    names: Mapping[str, Any],
    evals: list[dict[str, Any]],
) -> dict[str, Any]:
    """What a subject is: a checkpoint (its shortest id, the run and step that made it, what it builds on, the bookmarks
    that name it) or a base model (its name); and its evals (`evals`, by run, newest first), the suites they played,
    when the newest began and how many are playing still."""
    called: Mapping[str, str] = names.get("runs", {})
    made = checkpoints.get(reference) if kind == CHECKPOINT else None
    marks: Mapping[str, Any] = names.get("bookmarks", {}) if kind == CHECKPOINT else {}
    newest = sorted(evals, key=lambda each: -(each["started"] or 0.0))
    return {
        "kind": kind,
        "id": reference,
        "short": shorter.get(reference, reference) if kind == CHECKPOINT else reference,
        "run": made.run if made else None,
        "name": called.get(made.run, made.run) if made and made.run else None,
        "step": made.step if made else None,
        "base": made.base if made else None,
        "bookmarks": sorted(mark for mark, each in marks.items() if str(each) == reference),
        "evals": [each["run"] for each in newest],
        "suites": sorted({each["suite"] for each in evals}),
        "started": newest[0]["started"] if newest else None,
        "playing": sum(not each["done"] for each in evals),
    }


def evals_in(
    tables: Mapping[str, Mapping[str, JsonValue]], names: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Every eval in the tables, with its score and its score at each entry of the version it played (`entries`)."""
    return _evals(tables, names or {})


def _evals(tables: Mapping[str, Mapping[str, JsonValue]], names: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every eval in the tables, with its score."""
    called: Mapping[str, str] = names.get("runs", {})
    scheduled: dict[str, tuple[str, int]] = {}  # (an eval's run: the training run and step whose schedule asked)
    for name, records in tables.items():
        if (trained := between(name, "runs/", f"/{EVALS}")) is not None:
            for key, record in records.items():
                if isinstance(record, dict) and record.get("run"):
                    scheduled[str(record["run"])] = (trained, int(key))
    found: list[dict[str, Any]] = []
    for name in sorted(tables):
        middle = between(name, EVALUATIONS, "/subject")
        if middle is None or "/" not in middle:
            continue
        suite, run = middle.split("/", 1)
        about = cast(dict[str, Any], tables[name].get("subject") or {})
        starts = cast(dict[str, Any], tables.get(table(run, STARTS), {}))
        begun = newest_record(starts)
        parts = parts_of(tables, suite, run)
        groups = [
            group
            for part in parts
            for group in cast(dict[str, Any], tables.get(table(str(part["run"]), GROUPS), {})).values()
        ]
        episodes = eval_episodes(tables, suite, run)
        results = cast(dict[str, Any], tables.get(subject_table(suite, run, "results"), {}))
        by, step = scheduled.get(run, (begun.get("by"), begun.get("step")))
        version = played_version(about)
        said = [result for key, result in results.items() if reported(episodes.get(key.replace("-", "/", 1)))]
        rewards = [float(result["reward"]) for result in results.values() if result.get("reward") is not None]
        each_start = int(about.get("episodes") or 1)
        played_on = next((each for each in versions_in(tables, suite) if each.id == version), None)
        listed = list(zip(played_on.entries if played_on else [], parts, strict=False))
        expected = sum(int(group.get("episodes") or 0) for group in groups) or sum(
            len(entry.starts) * int(part.get("episodes") or each_start) for entry, part in listed
        )
        entries = [
            entry_of(str(part.get("environment")), results, episodes, int(part["offset"]), len(entry.starts))
            for entry, part in listed
        ]
        found.append(
            {
                "suite": suite,
                "version": version,
                "run": run,
                "name": called.get(run, run),
                "kind": about.get("kind", "checkpoint"),
                "checkpoint": about.get("checkpoint"),
                "model": about.get("model"),
                "asked_by": BY_SCHEDULE if by else BY_HAND,
                "by": by,
                "by_name": called.get(str(by), by) if by else None,
                "step": step,
                "episodes": each_start,
                "played": len(results),
                "expected": expected,
                "solved": sum(bool(result.get("solved")) for result in said) if said else None,
                "said": len(said),
                "rewards": rewards,
                "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
                "started": begun.get("started") or about.get("decided"),
                "at": max((float(result.get("time") or 0.0) for result in results.values()), default=None),
                "done": bool(results) and len(results) >= expected,
                "entries": entries,
            }
        )
    for each in found:
        each["share"] = round(each["solved"] / each["said"], 4) if each["said"] else None
    return found


def entry_of(
    environment: str, results: Mapping[str, Any], episodes: Mapping[str, Any], offset: int, starts: int
) -> dict[str, Any]:
    """How an eval did at one entry of its version, whose starts are those after `offset` (`starts` of them): its
    environment, episodes played, solved and the share solved of those that say, and the mean reward."""
    mine = {key: result for key, result in results.items() if offset < int(key.partition("-")[0]) <= offset + starts}
    said = [result for key, result in mine.items() if reported(episodes.get(key.replace("-", "/", 1)))]
    rewards = [float(result["reward"]) for result in mine.values() if result.get("reward") is not None]
    solved = sum(bool(result.get("solved")) for result in said) if said else None
    return {
        "environment": environment,
        "played": len(mine),
        "solved": solved,
        "said": len(said),
        "share": round(solved / len(said), 4) if said and solved is not None else None,
        "rewards": rewards,
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
    }


def _by_number(version: str) -> tuple[str, int]:
    """A version's id in the order versions are listed: by suite, then by number."""
    name, number = parsed(version)
    return name, number or 0


def _pooled(evals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each version's score over these evals together, by the version's id: the mean reward of every episode, the share
    solved of those that say, how many episodes, and which evals; and the same of each of its entries (`entries`, by
    environment)."""
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for each in evals:
        if each["played"]:
            by_suite.setdefault(each["version"], []).append(each)
    pooled: dict[str, dict[str, Any]] = {}
    for suite, listed in by_suite.items():
        rewards = [reward for each in listed for reward in each["rewards"]]
        said = sum(each["said"] for each in listed)
        solved = sum(each["solved"] or 0 for each in listed)
        pooled[suite] = {
            "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "solved": round(solved / said, 4) if said else None,
            "played": sum(each["played"] for each in listed),
            "evals": [each["run"] for each in listed],
            "entries": _pooled_entries([entry for each in listed for entry in each["entries"]]),
        }
    return pooled


def _pooled_entries(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each environment's score over these entries of evals together, by `module:name`."""
    by_environment: dict[str, list[dict[str, Any]]] = {}
    for each in entries:
        if each["played"]:
            by_environment.setdefault(each["environment"], []).append(each)
    pooled: dict[str, dict[str, Any]] = {}
    for environment, listed in by_environment.items():
        rewards = [reward for each in listed for reward in each["rewards"]]
        said = sum(each["said"] for each in listed)
        solved = sum(each["solved"] or 0 for each in listed)
        pooled[environment] = {
            "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "solved": round(solved / said, 4) if said else None,
            "played": sum(each["played"] for each in listed),
        }
    return pooled


def _environments(tables: Mapping[str, Mapping[str, JsonValue]], suite: str, number: int) -> list[str]:
    """A version's entries' environments, as `module:name`."""
    found = next((each for each in versions_in(tables, suite) if each.number == number), None)
    return [each for each in found.environments if each] if found is not None else []
