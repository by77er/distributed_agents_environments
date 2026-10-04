"""The checkpoints as a graph, with what trains, serves and evaluates them: what the monitor's lineage view draws.

Every checkpoint grows from a base model, along its parents (`rollout_train.checkpoints`): its first parent is what it
was trained from, any others what it learned from beside (a merge's). A checkpoint with no parent was trained from its
base model, which is the root its line hangs from. Each checkpoint was made by a step of some run, and says which. A
run that starts from another run's checkpoint forks there. Beside the graph stand each run's trainer with its queue of
steps, the engines and what each serves, and evaluations.

What is read: the checkpoints (each says what its weights are: an adapter, or full weights, which are resharded for
the engines), the runs' steps (which stand for their trainer's queue: a run takes one step at a time; its own trainer
makes what the run's checkpoints are), bookmarks, each checkpoint's reshard (`checkpoints/resharding`,
`checkpoints/resharded`, which `rollout_train.resharding` writes), the `published` notes read from the runners'
heartbeats (which stand for what the run's engines serve), and the suites' versions with how each subject played them
(`evaluations/SUITE/suite`, `evaluations/SUITE/SUBJECT/subject`, `evaluations/SUITE/SUBJECT/results`;
`rollout_train.evals`).
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from pydantic import JsonValue, TypeAdapter

from rollout_train.checkpoints import CHECKPOINTS, RELEASED, Checkpoint, short
from rollout_train.evals import (
    Suite,
    eval_episodes,
    offsets,
    parts_of,
    played_version,
    start_identity,
    suites_among,
    version_id,
    versions_in,
)
from rollout_train.ledger import between
from rollout_train.monitor.statistics import reported

QUEUED, TAKING, MADE, FAILED = "queued", "taking", "made", "failed"
WRITTEN, RESHARDING, RESHARDED, SERVING, SUPERSEDED = "written", "resharding", "resharded", "serving", "superseded"
"""A checkpoint's way to the engines: its files are written (its append), rewritten as the engines load them (where
they need to be), served once its run publishes it, then let go at the next."""

_VERSION = TypeAdapter(Checkpoint)
_RUNS, _EVALUATIONS = "runs/", "evaluations/"


def lineage(
    tables: Mapping[str, Mapping[str, JsonValue]],
    notes: Sequence[Mapping[str, Any]] = (),
    *,
    names: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """The graph a ledger's tables (by name) describe, with the feed's notes (`notes`) and the registry's `names`
    (`rollout_train.registry.names`: the runs' names and the bookmarks)."""
    now = time.time() if now is None else now
    names = dict(names or {"runs": {}, "bookmarks": {}})
    return _Reading(tables, notes, names, now).payload()


class _Reading:
    def __init__(
        self,
        tables: Mapping[str, Mapping[str, JsonValue]],
        notes: Sequence[Any],
        names: Mapping[str, Any],
        now: float,
    ) -> None:
        self.tables, self.names, self.now = tables, names, now
        self.published = [note for note in notes if note.get("kind") == "published" and note.get("adapter")]

    def named(self, before: str, after: str) -> list[str]:
        """What the tables' names hold between `before` and `after`: which runs have them."""
        return sorted({each for table in self.tables if (each := between(table, before, after))})

    def read(self, table: str) -> dict[str, Any]:
        return cast(dict[str, Any], dict(self.tables.get(table, {})))

    def record(self, table: str, key: str) -> dict[str, Any]:
        """A table's record under `key`, or an empty one."""
        return cast(dict[str, Any], self.tables.get(table, {}).get(key) or {})

    def payload(self) -> dict[str, Any]:
        made = self.checkpoints()
        runs = self.runs(made)
        loads = self.loads()
        checkpoints = self.shown(made, runs, loads)
        edges, outside = self.edges(made)
        bases = sorted({checkpoint.base or "the base model" for checkpoint in made.values() if not checkpoint.parents})
        return {
            "now": round(self.now, 1),
            "bases": bases,
            "checkpoints": checkpoints,
            "outside": outside,
            "runs": runs,
            "edges": edges,
            "bookmarks": dict(self.names.get("bookmarks", {})),
            "trainers": self.trainers(runs, made),
            "workers": self.workers(loads),
            "evaluations": self.evaluations(),
        }

    def checkpoints(self) -> dict[str, Checkpoint]:
        """Every checkpoint, by id, oldest first."""
        released = self.read(RELEASED)
        return {
            key: replace(checkpoint, weights=None, state=None, released=float(released[key]["at"]))
            if key in released
            else checkpoint
            for key, record in self.read(CHECKPOINTS).items()
            if (checkpoint := _VERSION.validate_python(record))
        }

    def runs(self, made: Mapping[str, Checkpoint]) -> list[dict[str, Any]]:
        """Every run that decided steps: what it started from, its steps, the checkpoints they made, and its latest
        (what its engines serve, as the feed's `published` notes say)."""
        runs: list[dict[str, Any]] = []
        called: Mapping[str, str] = self.names.get("runs", {})
        for run in self.named(_RUNS, "/steps"):
            failures = self.read(f"{_RUNS}{run}/failures")
            steps = sorted(self.read(f"{_RUNS}{run}/steps").items(), key=lambda item: int(item[0]))
            starts = self.read(f"{_RUNS}{run}/starts")
            begun = starts[max(starts, key=int)].get("from") if starts else None
            listed: list[dict[str, Any]] = []
            for key, step in steps:
                makes = str(step.get("makes"))
                listed.append(
                    {
                        "step": int(key),
                        "makes": makes,
                        "parent": step.get("parent"),
                        "state": FAILED if key in failures else MADE if makes in made else QUEUED,
                        "decided": step.get("decided"),
                        "segments": step.get("segments"),
                        "groups": step.get("groups") or [],
                    }
                )
            fed = [note for note in self.published if note.get("run") == run]
            runs.append(
                {
                    "run": run,
                    "name": called.get(run, run),
                    "from": begun or (listed[0]["parent"] if listed else None),
                    "steps": listed,
                    "checkpoints": [step["makes"] for step in listed if step["state"] == MADE],
                    "latest": str(fed[-1]["adapter"]) if fed else None,
                    "groups": self.waiting(run, listed),
                }
            )
        return runs

    def waiting(self, run: str, steps: list[dict[str, Any]]) -> list[list[float]]:
        """How many of a run's groups waited toward a step, over time: from when each was recorded with something
        to train on to when a step covering it was decided. (A group whose step did not say when is left out.)"""
        results = self.read(f"{_RUNS}{run}/results")
        decided: dict[int, float | None] = {int(group): step["decided"] for step in steps for group in step["groups"]}
        changes: list[tuple[float, int]] = []
        for key, result in results.items():
            if not result.get("segments") or not result.get("time") or (int(key) in decided and not decided[int(key)]):
                continue
            changes.append((float(result["time"]), 1))
            if covered := decided.get(int(key)):
                changes.append((float(covered), -1))
        return _series(changes)

    def shown(
        self,
        made: Mapping[str, Checkpoint],
        runs: list[dict[str, Any]],
        loads: Mapping[str, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Every checkpoint as the graph shows it: where it came from (its parents, its base, the run and step that made
        it), the bookmarks that name it, and where it is on its way to the engines."""
        shorter = short(made)
        by = {run["run"]: run for run in runs}
        marks: dict[str, list[str]] = {}
        for mark, checkpoint in self.names.get("bookmarks", {}).items():
            marks.setdefault(str(checkpoint), []).append(mark)
        latest = {run["latest"]: run["run"] for run in runs if run["latest"]}
        """The checkpoint each run's engines serve."""
        shown: list[dict[str, Any]] = []
        for checkpoint in made.values():
            run = by.get(checkpoint.run or "")
            noted = checkpoint.id in self.read("checkpoints/resharding") or checkpoint.id in self.read(
                "checkpoints/resharded"
            )
            life: dict[str, Any] = {
                "reshard": checkpoint.kind == "full" or noted,
                "resharding": self.record("checkpoints/resharding", checkpoint.id).get("at"),
                "resharded": self.record("checkpoints/resharded", checkpoint.id).get("at"),
                "latest_of": latest.get(checkpoint.id),
                "workers": loads.get(checkpoint.id, {}),
            }
            life["state"] = _state(life)
            shown.append(
                {
                    "id": checkpoint.id,
                    "short": shorter[checkpoint.id],
                    "depth": checkpoint.depth,
                    "parents": list(checkpoint.parents),
                    "base": checkpoint.base,
                    "kind": checkpoint.kind,
                    "made": checkpoint.made,
                    "kept": checkpoint.weights is not None,
                    "released": checkpoint.released,
                    "dataset": checkpoint.dataset,
                    "bookmarks": sorted(marks.get(checkpoint.id, [])),
                    "metrics": {
                        name: checkpoint.metrics[name] for name in ("kl_moved", "loss") if name in checkpoint.metrics
                    },
                    "by": {
                        "run": checkpoint.run,
                        "name": run["name"] if run else self.names.get("runs", {}).get(checkpoint.run, checkpoint.run),
                        "step": checkpoint.step,
                    }
                    if checkpoint.run
                    else None,
                    "life": life,
                }
            )
        return shown

    def edges(self, made: Mapping[str, Checkpoint]) -> tuple[list[dict[str, Any]], list[str]]:
        """What each checkpoint grew from: its first parent (`trained`), or its base model (`base`, from `base:MODEL`);
        and its other parents (`learned`). With the checkpoints they name that this ledger does not have."""
        edges: list[dict[str, Any]] = []
        for checkpoint in made.values():
            if not checkpoint.parents:
                edges.append(
                    {"kind": "base", "from": f"base:{checkpoint.base or 'the base model'}", "to": checkpoint.id}
                )
            for index, parent in enumerate(checkpoint.parents):
                edges.append({"kind": "trained" if index == 0 else "learned", "from": parent, "to": checkpoint.id})
        outside = sorted(
            {edge["from"] for edge in edges if edge["from"] not in made and not edge["from"].startswith("base:")}
        )
        return edges, outside

    def loads(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Each checkpoint's engines: when each loaded it, and let it go (None: it still serves it). A run's engines
        load what the feed says the run published, and let each go at the next."""
        loads: dict[str, dict[str, dict[str, Any]]] = {}
        by_run: dict[str, list[Mapping[str, Any]]] = {}
        for note in self.published:
            by_run.setdefault(f"{note.get('run') or note['channel']} engines", []).append(note)
        for worker, notes in by_run.items():
            for index, note in enumerate(notes):
                until = notes[index + 1]["at"] if index + 1 < len(notes) else None
                loads.setdefault(str(note["adapter"]), {})[worker] = {"since": note["at"], "until": until}
        return loads

    def workers(self, loads: Mapping[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Each run's engines, and the checkpoints they serve now."""
        serving: dict[str, list[str]] = {}
        for checkpoint, spans in loads.items():
            for worker, span in spans.items():
                serving.setdefault(worker, [])
                if span["until"] is None:
                    serving[worker].append(checkpoint)
        return [{"worker": worker, "serving": sorted(serving[worker])} for worker in sorted(serving)]

    def trainers(self, runs: list[dict[str, Any]], made: Mapping[str, Checkpoint]) -> list[dict[str, Any]]:
        """Each run's own trainer, whose queue is the run's steps (one at a time: each is decided once the one before
        is done with)."""
        trainers: list[dict[str, Any]] = []
        for run in runs:
            line = [checkpoint for id, checkpoint in made.items() if id in run["checkpoints"]]
            queue = [
                self.entry({"run": run["run"], "step": step["step"], "makes": step["makes"]} | {"at": step["decided"]},
                           step["decided"], runs, made)
                for step in run["steps"]
            ]  # fmt: skip
            trainers.append(
                {
                    "trainer": f"{run['name']} (the run's own)",
                    "weights": line[-1].kind if line else None,  # (what its steps make, as its newest checkpoint says)
                    "base": next((checkpoint.base for checkpoint in line if checkpoint.base), None),
                    "runs": [run["run"]],
                    "colocated": any("waited_for_requests_seconds" in checkpoint.metrics for checkpoint in line),
                    "queue": queue,
                    "depth": _depth(queue),
                    "groups": run["groups"],
                }
            )
        return trainers

    def entry(
        self, entry: Mapping[str, Any], began: float | None, runs: list[dict[str, Any]], made: Mapping[str, Checkpoint]
    ) -> dict[str, Any]:
        """A queued step, with when it began and ended: its checkpoint made, or its failure written."""
        run = next((each for each in runs if each["run"] == entry["run"]), None)
        step = next((each for each in run["steps"] if each["step"] == int(entry["step"])), None) if run else None
        checkpoint = made.get(str(entry["makes"]))
        failure = self.record(f"{_RUNS}{entry['run']}/failures", str(entry["step"]))
        return {
            "run": entry["run"],
            "step": int(entry["step"]),
            "makes": entry["makes"],
            "queued": entry.get("at"),
            "began": began,
            "done": checkpoint.made if checkpoint else failure.get("at"),
            "state": MADE if checkpoint else FAILED if failure else TAKING if began else QUEUED,
            "segments": step["segments"] if step else None,
        }

    def evaluations(self) -> list[dict[str, Any]]:
        """Each suite: the version its name points to (`version`, by id; where the registry's `names` say, else its
        newest) with that version's starts, every version (`versions`, oldest first, each with its starts), and each
        subject that played it (a checkpoint, or another model), with the version it played and how it did at each
        start (by the start's number in that version) and at each of that version's entries (`entries`)."""
        suites: list[dict[str, Any]] = []
        pointed: Mapping[str, str] = self.names.get("suites", {})
        for suite in suites_among(self.tables):
            versions = versions_in(self.tables, suite)
            if not versions:
                continue
            current = next((each for each in versions if each.id == pointed.get(suite)), versions[-1])
            by_id = {each.id: each for each in versions}
            subjects: list[dict[str, Any]] = []
            for subject in self.named(f"{_EVALUATIONS}{suite}/", "/results"):
                about = self.record(f"{_EVALUATIONS}{suite}/{subject}/subject", "subject")
                version = played_version(about)
                by_start: dict[str, list[dict[str, Any]]] = {}
                episodes = eval_episodes(self.tables, suite, subject)  # (an eval's subject is its run)
                for key, result in self.read(f"{_EVALUATIONS}{suite}/{subject}/results").items():
                    said = reported(episodes.get(key.replace("-", "/", 1))) is not False
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
                        "entries": _entries(by_id.get(version or ""), by_start, parts_of(self.tables, suite, subject)),
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
        "edited_from": version_id(suite.name, suite.edited_from) if suite.edited_from else None,
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


def _state(life: Mapping[str, Any]) -> str:
    """Where a checkpoint is on its way to the engines (`WRITTEN` to `SERVING`, or `SUPERSEDED`)."""
    workers: Mapping[str, Any] = life["workers"]
    now = {worker for worker, span in workers.items() if span["until"] is None}
    if now:
        return SERVING
    if workers:
        return SUPERSEDED
    if life["reshard"]:
        return RESHARDED if life["resharded"] else RESHARDING if life["resharding"] else WRITTEN
    return WRITTEN


def _depth(queue: list[dict[str, Any]]) -> list[list[float]]:
    """A queue's depth over time: `[time, waiting, being taken]` at every change."""
    changes: list[tuple[float, int, int]] = []
    for entry in queue:
        if entry["queued"] is None:
            continue
        began = entry["began"] if entry["began"] is not None else entry["done"]
        changes.append((float(entry["queued"]), 1, 0))
        if began is not None:
            changes.append((float(began), -1, 1))
        if entry["done"] is not None:
            changes.append((float(entry["done"]), 0, -1))
    series: list[list[float]] = []
    waiting = taking = 0
    for at, more, busy in sorted(changes):
        waiting, taking = waiting + more, taking + busy
        if series and series[-1][0] == at:
            series[-1] = [at, waiting, taking]
        else:
            series.append([at, waiting, taking])
    return series


def _series(changes: list[tuple[float, int]]) -> list[list[float]]:
    """A count over time, `[time, count]` at every change, from its changes."""
    series: list[list[float]] = []
    count = 0
    for at, change in sorted(changes):
        count += change
        if series and series[-1][0] == at:
            series[-1][1] = count
        else:
            series.append([at, count])
    return series


def _episodes(subject: Mapping[str, Any]) -> int | None:
    """A subject's episodes of each start, from its record: none where its entries play different numbers of them."""
    if subject.get("episodes") is None and subject.get("parts"):
        return None
    return int(subject.get("episodes") or 1)
