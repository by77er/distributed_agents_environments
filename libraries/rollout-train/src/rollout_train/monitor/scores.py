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

Every eval (`evals_in`) and every suite with each subject's results at each start (`suites_in`) are what the Evals page
lists.
"""

from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoint, short
from rollout_train.evals import (
    EVALUATIONS,
    Suite,
    eval_episodes,
    offsets,
    parsed,
    parts_of,
    played_version,
    start_identity,
    subject_table,
    suites_among,
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
    return _shown([each for each in evals_in(tables, names or {}) if _subject_of(each) == (CHECKPOINT, checkpoint)])


def subjects_in(
    tables: Mapping[str, Mapping[str, JsonValue]],
    checkpoints: Mapping[str, Checkpoint],
    names: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every subject that has had an eval, the one evaluated last first: what it is, and its evals (`_subject`)."""
    said = names or {}
    evals: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for each in evals_in(tables, said):
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
    found = [each for each in evals_in(tables, said) if _subject_of(each) == (kind, reference)]
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
    evals = evals_in(tables, said)
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
    called: Mapping[str, str] = (names or {}).get("runs", {})
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


def suites_in(
    tables: Mapping[str, Mapping[str, JsonValue]], names: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Each suite: the version its name points to (`version`, by id; where the registry's `names` say, else its
    newest) with that version's starts, every version (`versions`, oldest first, each with its starts), and each
    subject that played it (a checkpoint, or another model), with the version it played and how it did at each
    start (by the start's number in that version) and at each of that version's entries (`entries`)."""
    suites: list[dict[str, Any]] = []
    pointed: Mapping[str, str] = (names or {}).get("suites", {})
    for suite in suites_among(tables):
        versions = versions_in(tables, suite)
        if not versions:
            continue
        current = next((each for each in versions if each.id == pointed.get(suite)), versions[-1])
        by_id = {each.id: each for each in versions}
        subjects: list[dict[str, Any]] = []
        played_by = {each for name in tables if (each := between(name, f"{EVALUATIONS}{suite}/", "/results"))}
        for subject in sorted(played_by):
            about = cast(dict[str, Any], tables.get(subject_table(suite, subject, "subject"), {}).get("subject") or {})
            version = played_version(about)
            by_start: dict[str, list[dict[str, Any]]] = {}
            episodes = eval_episodes(tables, suite, subject)  # (an eval's subject is its run)
            for key, result in cast(dict[str, Any], tables.get(subject_table(suite, subject, "results"), {})).items():
                said = bool(reported(episodes.get(key.replace("-", "/", 1))))
                by_start.setdefault(key.partition("-")[0], []).append(
                    {"solved": bool(result.get("solved")) if said else None, "reward": result.get("reward")}
                    | {"run_id": result.get("run_id")}
                )
            played = [each for listed in by_start.values() for each in listed]
            solved = [each["solved"] for each in played if each["solved"] is not None]
            rewards = [float(each["reward"]) for each in played if each["reward"] is not None]
            subjects.append(
                {
                    "subject": subject,
                    "kind": about.get("kind", "checkpoint"),
                    "checkpoint": about.get("checkpoint"),
                    "model": about.get("model"),
                    "episodes": _episodes(about),
                    "asked_by": about.get("asked_by"),
                    "version": version,
                    "starts": len(by_id[version].starts) if version in by_id else 0,
                    "results": by_start,
                    "entries": _entries(by_id.get(version or ""), by_start, parts_of(tables, suite, subject)),
                    "played": len(played),
                    "solved": sum(solved) if solved or not played else None,
                    "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
                }
            )
        suites.append(
            {
                "suite": suite,
                "version": current.id,
                "number": current.number,
                "starts": _starts(current),
                "versions": [_described(each) for each in versions],
                "subjects": subjects,
            }
        )
    return suites


def _starts(suite: Suite) -> list[dict[str, Any]]:
    """A version's starts as the page shows them: each by its number in the version, its entry's environment, its row,
    title and seed, and what it is in every version that has it (`identity`)."""
    return [
        {"start": str(before + number), "environment": entry.environment or None, "task": start.task}
        | {"title": start.title, "seed": start.seed}
        | {"identity": start_identity({"task": start.task, "parameters": start.parameters})}
        for entry, before in zip(suite.entries, offsets(suite), strict=True)
        for number, start in enumerate(entry.starts, start=1)
    ]


def _described(suite: Suite) -> dict[str, Any]:
    """A version as the page shows it: what it is, its entries (each with how many of its starts come before its
    first: `offset`), and its starts."""
    return {
        "id": suite.id,
        "number": suite.number,
        "environments": [each or None for each in suite.environments],
        "made": suite.made or None,
        "held_out": suite.held_out,
        "entries": [
            {
                "environment": entry.environment or None,
                "environment_version": entry.environment_version,
                "chosen": entry.chosen,
                "eval_data": entry.eval_data,
                "rows": entry.rows,
                "seeds": entry.seeds,
                "held_out": entry.held_out,
                "episodes": entry.episodes,
                "thinking_tokens": entry.thinking_tokens,
                "answer_tokens": entry.answer_tokens,
                "offset": before,
                "starts": len(entry.starts),
            }
            for entry, before in zip(suite.entries, offsets(suite), strict=True)
        ],
        "starts": _starts(suite),
    }


def _entries(
    version: Suite | None, by_start: Mapping[str, list[dict[str, Any]]], parts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """How a subject did at each entry of the version it played (each its environment, its episodes of each start,
    episodes played, solved where its episodes say, and the mean reward), from its episodes by start and the runs that
    played its entries."""
    if version is None:
        return []
    found: list[dict[str, Any]] = []
    for place, (entry, before) in enumerate(zip(version.entries, offsets(version), strict=True)):
        played = [each for number in range(1, len(entry.starts) + 1) for each in by_start.get(str(before + number), [])]
        solved = [each["solved"] for each in played if each["solved"] is not None]
        rewards = [float(each["reward"]) for each in played if each["reward"] is not None]
        episodes = parts[place].get("episodes") if place < len(parts) else None
        found.append(
            {
                "environment": entry.environment or None,
                "episodes": int(episodes or entry.episodes),
                "played": len(played),
                "solved": sum(solved) if solved or not played else None,
                "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
            }
        )
    return found


def _by_number(version: str) -> tuple[str, int]:
    """A version's id in the order versions are listed: by suite, then by number."""
    name, number = parsed(version)
    return name, number or 0


def _pooled(evals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each version's score over these evals together, by the version's id (`_scored`), with which evals; and the
    same of each of its entries (`entries`, by environment)."""
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for each in evals:
        if each["played"]:
            by_suite.setdefault(each["version"], []).append(each)
    return {
        suite: _scored(listed)
        | {"evals": [each["run"] for each in listed]}
        | {"entries": _pooled_entries([entry for each in listed for entry in each["entries"]])}
        for suite, listed in by_suite.items()
    }


def _pooled_entries(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each environment's score over these entries of evals together, by `module:name` (`_scored`)."""
    by_environment: dict[str, list[dict[str, Any]]] = {}
    for each in entries:
        if each["played"]:
            by_environment.setdefault(each["environment"], []).append(each)
    return {environment: _scored(listed) for environment, listed in by_environment.items()}


def _scored(listed: list[dict[str, Any]]) -> dict[str, Any]:
    """The score of evals (or their entries) together: the mean reward of every episode, the share solved of those
    that say, and how many episodes."""
    rewards = [reward for each in listed for reward in each["rewards"]]
    said = sum(each["said"] for each in listed)
    solved = sum(each["solved"] or 0 for each in listed)
    return {
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "solved": round(solved / said, 4) if said else None,
        "played": sum(each["played"] for each in listed),
    }


def _environments(tables: Mapping[str, Mapping[str, JsonValue]], suite: str, number: int) -> list[str]:
    """A version's entries' environments, as `module:name`."""
    found = next((each for each in versions_in(tables, suite) if each.number == number), None)
    return [each for each in found.environments if each] if found is not None else []


def _episodes(subject: Mapping[str, Any]) -> int | None:
    """A subject's episodes of each start, from its record: none where its entries play different numbers of them."""
    if subject.get("episodes") is None and subject.get("parts"):
        return None
    return int(subject.get("episodes") or 1)
