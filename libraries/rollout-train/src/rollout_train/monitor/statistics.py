"""Every training run of a ledger in figures: what the monitor's statistics page draws (evals are on the evals page).

For each run (`rollout_train.record`): its groups, each with its row, when it was decided and when its result was
written, its episodes' rewards and whether each solved (null for each where none of the group's episodes said whether it
solved its task: `unreported`), what it gave to train on (or why nothing) and what its step did with it; its steps,
each with the checkpoint it made and the trainer's statistics; how many groups were in flight and how many waited
toward a step, over time; and its engines' throughput, from the `inference` notes in its feed (for a run whose
directory is where the feed can be read).

It knows rows, groups, episodes, rewards and durations, not what an environment plays. A group whose start carries a
list under `names` says how many it names (`names`), so that groups can be counted by it.
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.checkpoints import CHECKPOINTS
from rollout_train.launches import EVAL
from rollout_train.record import FAILURES, GROUPS, RESULTS, STARTS, STEPS, Result, named_runs, table
from rollout_train.rollouts.scheduler import EPISODES

STEP_METRICS = (
    "kl_moved",
    "kl_floor",
    "clip_fraction",
    "mean_mismatch",
    "mean_weight",
    "truncated_fraction",
    "loss",
    "segments",
    "tokens",
    "optimizer_steps",
    "start_seconds",
    "update_seconds",
    "seconds",
)
"""What a step's checkpoint says of the update, as far as it says it (older checkpoints say less)."""
THROUGHPUT = 720
"""Throughput measurements of a channel at most: more are merged, neighbours together."""
COMMITTED, STEPPING, FAILED = "committed", "stepping", "failed"


def statistics(
    tables: Mapping[str, Mapping[str, JsonValue]],
    notes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """The figures a ledger's tables (by name) hold, with each run's engines' throughput from its feed's job lines
    (`notes`, by run)."""
    now = time.time() if now is None else now
    made = _checkpoints(tables)
    runs = [run for run in named_runs(tables) if not _evaluates(tables, run)]
    notes = notes or {}
    return {
        "now": round(now, 1),
        "runs": [
            _run(
                run,
                {name: tables.get(table(run, name), {}) for name in (GROUPS, RESULTS, STEPS, FAILURES, EPISODES)},
                made,
            )
            | {"inference": _throughput(notes.get(run, ()))}
            for run in runs
        ],
    }


def _evaluates(tables: Mapping[str, Mapping[str, JsonValue]], run: str) -> bool:
    """Whether a run is an eval, or a part of one: its newest start says so."""
    starts = tables.get(table(run, STARTS), {})
    newest: Any = starts[max(starts, key=int)] if starts else {}
    return isinstance(newest, dict) and cast(dict[str, Any], newest).get("kind") == EVAL


def _checkpoints(tables: Mapping[str, Mapping[str, JsonValue]]) -> dict[str, dict[str, Any]]:
    """Every checkpoint's record, by its id."""
    return {key: cast(dict[str, Any], record) for key, record in tables.get(CHECKPOINTS, {}).items()}


def _makes(step: Mapping[str, Any]) -> str | None:
    makes = step.get("makes")
    return str(makes) if makes else None


def _run(run: str, tables: Mapping[str, Mapping[str, Any]], made: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    covering: dict[int, dict[str, Any]] = {}
    for key, intent in sorted(tables[STEPS].items(), key=lambda item: int(item[0])):
        makes = _makes(intent)
        checkpoint = made.get(makes or "")
        metrics: Mapping[str, Any] = (checkpoint or {}).get("metrics") or {}
        step = {
            "step": int(key),
            "checkpoint": makes,
            "decided": intent.get("decided"),
            "made": checkpoint.get("made") if checkpoint else None,
            "state": FAILED if key in tables[FAILURES] else COMMITTED if checkpoint else STEPPING,
            "groups": len(intent.get("groups") or []),
            "segments": intent.get("segments"),
            "metrics": {name: metrics[name] for name in STEP_METRICS if name in metrics},
        }
        steps.append(step)
        listed: list[Any] = intent.get("groups") or []
        for group in listed:
            covering[int(group)] = step
    groups: list[dict[str, Any]] = []
    unsaid = unreported(tables[EPISODES])
    for key, record in sorted(tables[GROUPS].items(), key=lambda item: int(item[0])):
        number, result = int(key), tables[RESULTS].get(key)
        joined = asdict(Result.from_json(result, number, record)) if result else None
        parameters: Any = record.get("parameters")
        names: Any = cast(dict[str, Any], parameters).get("names") if isinstance(parameters, dict) else None
        step = covering.get(number)
        groups.append(
            {
                "group": number,
                "task": record.get("task"),
                "title": record.get("title"),
                "decided": record.get("decided"),
                "time": joined["time"] if joined else None,
                "rollout_seconds": joined["rollout_seconds"] if joined else None,
                "rewards": joined["rewards"] if joined else [],
                "solved": solved_of(joined["solved"], key in unsaid) if joined else [],
                "failed": joined["failed"] if joined else 0,
                "segments": joined["segments"] if joined else None,
                "skipped": joined["skipped"] if joined else None,
                "step": step["step"] if step else None,
                "trained": step["state"] if step else None,
                "names": len(cast(list[Any], names)) if isinstance(names, list) else None,
            }
        )
    return {
        "run": run,
        "wrote": newest(
            [group["decided"] for group in groups]
            + [group["time"] for group in groups]
            + [step["decided"] for step in steps]
            + [step["made"] for step in steps]
        ),
        "groups": groups,
        "steps": steps,
        "flight": _flight(groups, steps),
    }


def reported(line: Any) -> bool | None:
    """Whether an episode's record (as the ledger keeps it) says if it solved its task (`info["solved"]`); None
    without a record."""
    if not isinstance(line, dict):
        return None
    episode: Any = cast(dict[str, Any], line).get("episode") or {}
    return "solved" in (episode.get("info") or {})


def unreported(episodes: Mapping[str, Any]) -> set[str]:
    """The groups (by number) whose episodes ended, none of them saying whether it solved its task: training counts
    each as not solved, and the page says nothing of it (a task may never say)."""
    said: dict[str, bool] = {}
    for key, line in episodes.items():
        group = key.split("/")[0]
        said[group] = said.get(group, False) or bool(reported(line))
    return {group for group, any_said in said.items() if not any_said}


def solved_of(solved: Sequence[bool], unsaid: bool) -> list[bool | None]:
    """A result's `solved` as the page shows it: as written, or null for each where nothing said it (`unsaid`)."""
    return [None] * len(solved) if unsaid else list(solved)


def newest(times: Iterable[Any]) -> float | None:
    """The latest of some times (those that are numbers), if any: when a run last wrote."""
    found = [float(each) for each in times if isinstance(each, int | float)]
    return round(max(found), 1) if found else None


def _flight(groups: list[dict[str, Any]], steps: list[dict[str, Any]]) -> list[list[float]]:
    """How many groups were in flight (decided, their result not written) and how many waited toward a step (their
    result written with something to train on, and no step begun over them), over time: `[time, in flight, waiting]`
    at every change. A step begins when it was decided, or, where it did not say, once its checkpoint was made less
    the update's time."""

    def began(step: Mapping[str, Any]) -> float | None:
        if step["decided"]:
            return float(step["decided"])
        if step["made"]:
            return float(step["made"]) - float(step["metrics"].get("update_seconds") or 0.0)
        return None

    by_number = {step["step"]: step for step in steps}
    changes: list[tuple[float, int, int]] = []
    for group in groups:
        decided, ended = group["decided"], group["time"]
        if decided:
            changes.append((float(decided), 1, 0))
            if ended:
                changes.append((float(ended), -1, 0))
        if ended and group["segments"]:
            changes.append((float(ended), 0, 1))
            step = by_number.get(group["step"]) if group["step"] is not None else None
            if step and (start := began(step)) is not None:
                changes.append((max(start, float(ended)), 0, -1))
    series: list[list[float]] = []
    playing = waiting = 0
    for at, more, queued in sorted(changes):
        playing, waiting = playing + more, waiting + queued
        if series and series[-1][0] == at:
            series[-1] = [at, playing, waiting]
        else:
            series.append([at, playing, waiting])
    return series


def _throughput(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each channel's measurements, `[time, tokens a second, requests at once, tokens generated, requests]`, merged
    neighbours together down to `THROUGHPUT` (means of the rates, sums of the counts)."""
    channels: dict[str, list[list[float]]] = {}
    for note in notes:
        if note.get("kind") != "inference" or not note.get("at"):
            continue
        channels.setdefault(str(note.get("channel")), []).append(
            [float(note["at"])]
            + [float(note.get(name) or 0.0) for name in ("tokens_per_second", "mean_concurrency")]
            + [float(note.get(name) or 0.0) for name in ("generated_tokens", "requests")]
        )
    shown: list[dict[str, Any]] = []
    for channel, points in channels.items():
        every = -(-len(points) // THROUGHPUT)
        merged = [
            [
                chunk[-1][0],
                round(sum(each[1] for each in chunk) / len(chunk), 1),
                round(sum(each[2] for each in chunk) / len(chunk), 2),
                sum(each[3] for each in chunk),
                sum(each[4] for each in chunk),
            ]
            for chunk in (points[start : start + every] for start in range(0, len(points), every))
        ]
        shown.append({"channel": channel, "every": every, "points": merged})
    return shown
