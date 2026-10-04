"""Evals from a checkpoint's point of view: every eval a checkpoint has had, and its scores along its line.

An eval is a run of its own (`rollout_train.evals`): who played is in `evaluations/SUITE/EVAL/subject`, how each episode
went in `evaluations/SUITE/EVAL/results`, and its start (`runs/EVAL/starts`) says when it began and, for one a training
run's schedule asked for, which run and step (`by`, `step`); that run's `evals` table says so too. A checkpoint's
evals (`evals_of`) are every eval whose subject it is, by hand or by a schedule, each with its score: the mean reward of
its episodes, and the share solved where its episodes say whether they solved their start.

A checkpoint's line (`path_of`) is its first parents back to the root, and the root's base model before it: the model
every checkpoint on it builds on. It crosses runs (a run that starts from another's checkpoint), merges (a full
checkpoint made from an adapter) and forks. Each point on it has its score at each suite, pooled over every eval of it
on that suite; the base model's are the evals of that model by name.
"""

from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoint, short
from rollout_train.ledger import between
from rollout_train.monitor.statistics import reported
from rollout_train.record import EVALS, GROUPS, STARTS, table

EVALUATIONS = "evaluations/"
BY_HAND, BY_SCHEDULE = "by hand", "schedule"
"""Who asked for an eval: someone, by hand (`rollout eval`, the page), or a training run's schedule."""


def evals_of(
    tables: Mapping[str, Mapping[str, JsonValue]], checkpoint: str, names: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Every eval whose subject is `checkpoint` (by id), newest first: the suite, the eval's run, who asked for it
    (by hand, or the training run and step whose schedule did), episodes of each start, how far it got and its score."""
    found = [each for each in _evals(tables, names or {}) if each["checkpoint"] == checkpoint]
    return [
        {key: value for key, value in each.items() if key != "rewards"}  # (each episode's reward stays here)
        for each in sorted(found, key=lambda each: -(each["started"] or 0.0))
    ]


def path_of(
    tables: Mapping[str, Mapping[str, JsonValue]],
    checkpoints: Mapping[str, Checkpoint],
    checkpoint: str,
    names: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A checkpoint's line, from the base model to it along first parents (`points`, by depth: the base model at 0),
    each point with what it is (its run, step, weights and bookmarks) and its score at each suite (`scores`, pooled
    over every eval of it there); and the suites any point was evaluated on, with their catalogs."""
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
    suites = sorted({suite for point in points for suite in point["scores"]})
    return {
        "checkpoint": checkpoint,
        "points": points if line else [],
        "suites": [{"suite": suite, "catalog": _about(tables, suite).get("catalog")} for suite in suites],
    }


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
        begun: dict[str, Any] = starts[max(starts, key=int)] if starts else {}
        groups = cast(dict[str, Any], tables.get(table(run, GROUPS), {}))
        episodes = cast(dict[str, Any], tables.get(table(run, "episodes"), {}))
        results = cast(dict[str, Any], tables.get(f"{EVALUATIONS}{suite}/{run}/results", {}))
        by, step = scheduled.get(run, (begun.get("by"), begun.get("step")))
        said = [result for key, result in results.items() if reported(episodes.get(key.replace("-", "/", 1)))]
        rewards = [float(result["reward"]) for result in results.values() if result.get("reward") is not None]
        each_start = int(about.get("episodes") or 1)
        expected = sum(int(group.get("episodes") or 0) for group in groups.values()) or each_start * len(
            tables.get(f"{EVALUATIONS}{suite}/starts", {})
        )
        found.append(
            {
                "suite": suite,
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
            }
        )
    for each in found:
        each["share"] = round(each["solved"] / each["said"], 4) if each["said"] else None
    return found


def _pooled(evals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each suite's score over these evals together: the mean reward of every episode, the share solved of those that
    say, how many episodes, and which evals."""
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for each in evals:
        if each["played"]:
            by_suite.setdefault(each["suite"], []).append(each)
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
        }
    return pooled


def _about(tables: Mapping[str, Mapping[str, JsonValue]], suite: str) -> dict[str, Any]:
    return cast(dict[str, Any], tables.get(f"{EVALUATIONS}{suite}/suite", {}).get("suite") or {})
