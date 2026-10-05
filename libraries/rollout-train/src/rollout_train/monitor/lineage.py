"""The checkpoints as a graph, with what trains and serves them: what the monitor's lineage view draws.

Every checkpoint grows from a base model, along its parents (`rollout_train.checkpoints`): its first parent is what it
was trained from, any others what it learned from beside (a merge's). A checkpoint with no parent was trained from its
base model, which is the root its line hangs from; a base model that has had an eval is a root too, whether or not
anything was trained from it. Each checkpoint was made by a step of some run, and says which. A
run that starts from another run's checkpoint forks there. Beside the graph stand each run's trainer with its queue of
steps, and the engines and what each serves.

What is read: the checkpoints (each says what its weights are: an adapter, or full weights, which a bridge makes into
the engines' files), the runs' steps (which stand for their trainer's queue: a run takes one step at a time; its own
trainer makes what the run's checkpoints are), bookmarks, each checkpoint's bridges (`checkpoints/resharding`,
`checkpoints/resharded`, keyed `CHECKPOINT@BRIDGE`, which `rollout_train.bridges` writes), the evals' subjects (for the
base models evaluated), the `published` notes read from the runners' heartbeats (which stand for what the run's engines
serve).
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from pydantic import JsonValue, TypeAdapter

from rollout_train.bridges import BRIDGED, BRIDGING, checkpoint_of
from rollout_train.checkpoints import CHECKPOINTS, RELEASED, Checkpoint, short
from rollout_train.evals import EVALUATIONS
from rollout_train.ledger import between
from rollout_train.record import newest_record

QUEUED, TAKING, MADE, FAILED = "queued", "taking", "made", "failed"
WRITTEN, RESHARDING, RESHARDED, SERVING, SUPERSEDED = "written", "resharding", "resharded", "serving", "superseded"
"""A checkpoint's way to the engines: its files are written (its append), rewritten as the engines load them (where
they need to be), served once its run publishes it, then let go at the next."""

_VERSION = TypeAdapter(Checkpoint)
_RUNS = "runs/"


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

    def bridged(self, table: str, checkpoint: str) -> dict[str, Any]:
        """The newest record of a bridges' table (`rollout_train.bridges`) for a checkpoint, or an empty one."""
        found: dict[str, Any] = {}
        for entry, record in self.read(table).items():
            if checkpoint_of(entry) == checkpoint and isinstance(record, dict):
                said = cast(dict[str, Any], record)
                if float(said.get("at") or 0) >= float(found.get("at") or 0):
                    found = said
        return found

    def payload(self) -> dict[str, Any]:
        made = self.checkpoints()
        runs = self.runs(made)
        loads = self.loads()
        checkpoints = self.shown(made, runs, loads)
        edges, outside = self.edges(made)
        trained = sorted(
            {checkpoint.base or "the base model" for checkpoint in made.values() if not checkpoint.parents}
        )
        bases = trained + [each for each in self.evaluated() if each not in trained]
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
        }

    def evaluated(self) -> list[str]:
        """The base models that have had an eval (`rollout_train.evals`: an eval's subject of kind `model`), by name."""
        found: set[str] = set()
        for name, records in self.tables.items():
            if name.startswith(EVALUATIONS) and name.endswith("/subject"):
                about = records.get("subject")
                if isinstance(about, dict) and about.get("kind") == "model" and about.get("model"):
                    found.add(str(about["model"]))
        return sorted(found)

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
            begun = newest_record(starts).get("from")
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
            begun, ended = self.bridged(BRIDGING, checkpoint.id), self.bridged(BRIDGED, checkpoint.id)
            life: dict[str, Any] = {
                "reshard": checkpoint.kind == "full" or bool(begun or ended),
                "resharding": begun.get("at"),
                "resharded": ended.get("at"),
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
