"""The versions as a graph, with what trains, serves and evaluates them: what the monitor's lineage view draws.

Every version grows from a base model, along its parents (`rollout_train.versions`): its first parent is what it was
trained from, any others what it learned from beside (a distillation's teachers). A version with no parent was
trained from its base model, which is the root its line hangs from. Each version was made by a step of some run, and
says which. A run that starts from another run's version forks there. Beside the graph stand the trainers that take
the steps, the inference workers and what each serves, and evaluations.

What a ledger has today is read as it is: the versions, the runs' steps (which stand for their trainer's queue: a run
takes one step at a time), bookmarks, and the `published` notes in the feed (which stand for what the run's engines
serve). The other tables read here are proposed in docs/research/policy-dag.md, and nothing appends them yet:

- `runs/RUN/plan`, `runs/RUN/published`: what a run was set up to do (a distillation's teachers, whose samples it
  trains on, its objective), and the version a request for the run's latest goes to, from when;
- `trainers/NAME/registered`, `trainers/NAME/queue`, `trainers/NAME/taken`: a trainer, the steps queued for it, and
  when it began each;
- `versions/resharding`, `versions/resharded`: a full-weight version's files rewritten as the engines load them;
- `workers/NAME/registered`, `workers/NAME/loaded`, `workers/NAME/unloaded`: an inference worker (what it holds, its
  adapter slots), and each version it loaded and unloaded;
- `evaluations/SUITE/starts`, `evaluations/SUITE/SUBJECT/subject`, `evaluations/SUITE/SUBJECT/results`: a fixed suite
  of starts, and how a version (or another model) played it.

The router's `routing` notes in the feed (requests waiting, by the version they name) are proposed there too.
`SAMPLE` holds such tables and notes as a fixture, so that the view can be seen with them; it is read only when
asked for (`sample=True`), and everything read from it is marked `sample`.
"""

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue, TypeAdapter

from rollout_train.ledger import between
from rollout_train.versions import RELEASED, VERSIONS, Version, short

SAMPLE = Path(__file__).with_name("sample-lineage.json")
"""A fixture of the proposed tables: two more LoRA runs sharing a trainer, a full-weight run with a trainer of its
own, two distillations (one on the student's own samples, one on its teachers'), a queue, resharding and a roll-out in
progress, inference workers with requests waiting, an evaluation suite."""
TIMES = ("made", "decided", "at", "began", "released", "time")
"""Fields that say when: in the fixture, written as seconds before now (negative), so it looks current when shown."""

ON_POLICY, OFF_POLICY, MIXED = "on-policy", "off-policy", "mixed"
SAYS = {
    ON_POLICY: "on-policy: the student samples, and its teachers score every token it sampled",
    OFF_POLICY: "off-policy: the student is trained on its teachers' samples",
    MIXED: "mixed: the student is trained on its own samples and on its teachers'",
}
"""What a distillation's mode means, in words, for whoever reads the graph."""
STUDENT = "student"
"""Whose samples a distillation trains on: the student's own (`data.sampled_by`), else its teachers' by version."""
QUEUED, TAKING, MADE, FAILED = "queued", "taking", "made", "failed"
WRITTEN, RESHARDING, RESHARDED, ROLLING, SERVING, SUPERSEDED = (
    "written",
    "resharding",
    "resharded",
    "rolling out",
    "serving",
    "superseded",
)
"""A version's way to the engines: its files are written (its append), rewritten as the engines load them (where
they need to be), loaded by workers one by one once it is its run's latest (or as requests name it exactly), then
unloaded."""

_VERSION = TypeAdapter(Version)
_RUNS, _TRAINERS, _WORKERS, _EVALUATIONS = "runs/", "trainers/", "workers/", "evaluations/"


def lineage(
    tables: Mapping[str, Mapping[str, JsonValue]],
    notes: Sequence[Mapping[str, Any]] = (),
    *,
    names: Mapping[str, Any] | None = None,
    sample: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """The graph a ledger's tables (by name) describe, with the feed's notes (`notes`) and the registry's `names`
    (`rollout_train.registry.names`: the runs' names and the bookmarks). With `sample`, the fixture's tables, notes
    and names are read beside them (a table or record of the ledger's own is never replaced); what is read from it is
    marked `sample`."""
    now = time.time() if now is None else now
    names = dict(names or {"runs": {}, "bookmarks": {}})
    sampled: set[str] = set()
    if sample:
        fixture: dict[str, Any] = _moved(json.loads(SAMPLE.read_text()), now)
        own = set(tables.get(VERSIONS, {}))
        sampled = (set(fixture["tables"]) - set(tables)) | {
            f"{VERSIONS}/{key}" for key in fixture["tables"][VERSIONS] if key not in own
        }
        tables = {**fixture["tables"], **tables}
        for name in (VERSIONS, RELEASED):
            tables[name] = {**fixture["tables"].get(name, {}), **tables.get(name, {})}
        notes = [*notes, *fixture["notes"]]
        names = {kind: {**fixture["names"].get(kind, {}), **names.get(kind, {})} for kind in ("runs", "bookmarks")}
    return _Reading(tables, sampled, notes, names, now).payload()


def _moved(value: Any, now: float) -> Any:
    """The fixture with its times moved to before `now`."""
    if isinstance(value, dict):
        return {
            key: round(now + each, 1)
            if key in TIMES and isinstance(each, int | float) and each < 0
            else _moved(each, now)
            for key, each in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [_moved(each, now) for each in cast(list[Any], value)]
    return value


def mode(sampled_by: Iterable[str]) -> str:
    """Whether a distillation is on policy or off it, by whose samples it trains on: the student's own (`student`),
    its teachers' (by version), or both."""
    whose = set(sampled_by) or {STUDENT}
    if whose == {STUDENT}:
        return ON_POLICY
    return MIXED if STUDENT in whose else OFF_POLICY


class _Reading:
    def __init__(
        self,
        tables: Mapping[str, Mapping[str, JsonValue]],
        sampled: set[str],
        notes: Sequence[Any],
        names: Mapping[str, Any],
        now: float,
    ) -> None:
        self.tables, self.sampled, self.names, self.now = tables, sampled, names, now
        self.published = [note for note in notes if note.get("kind") == "published" and note.get("adapter")]
        self.routing = [note for note in notes if note.get("kind") == "routing"]

    def named(self, before: str, after: str) -> list[str]:
        """What the tables' names hold between `before` and `after`: which runs, trainers, workers have them."""
        return sorted({each for table in self.tables if (each := between(table, before, after))})

    def read(self, table: str) -> dict[str, Any]:
        return cast(dict[str, Any], dict(self.tables.get(table, {})))

    def record(self, table: str, key: str) -> dict[str, Any]:
        """A table's record under `key`, or an empty one."""
        return cast(dict[str, Any], self.tables.get(table, {}).get(key) or {})

    def payload(self) -> dict[str, Any]:
        made = self.versions()
        runs = self.runs(made)
        loads = self.loads()
        waiting: dict[str, int] = self.routing[-1]["waiting"] if self.routing else {}
        versions = self.shown(made, runs, loads, waiting)
        edges, outside = self.edges(made, runs)
        bases = sorted({version.base or "the base model" for version in made.values() if not version.parents})
        return {
            "now": round(self.now, 1),
            "sample": bool(self.sampled),
            "bases": bases,
            "versions": versions,
            "outside": outside,
            "runs": runs,
            "edges": edges,
            "bookmarks": dict(self.names.get("bookmarks", {})),
            "trainers": self.trainers(runs, made),
            "workers": self.workers(loads),
            "routing": {
                "waiting": waiting,
                "history": [[note["at"], sum(note["waiting"].values())] for note in self.routing],
            },
            "evaluations": self.evaluations(),
        }

    def versions(self) -> dict[str, Version]:
        """Every version, by id, oldest first."""
        released = self.read(RELEASED)
        return {
            key: replace(version, weights=None, state=None, released=float(released[key]["at"]))
            if key in released
            else version
            for key, record in self.read(VERSIONS).items()
            if (version := _VERSION.validate_python(record))
        }

    def runs(self, made: Mapping[str, Version]) -> list[dict[str, Any]]:
        """Every run that decided steps: what it was set up to do (a training run, unless its plan says otherwise),
        what it started from, its steps, the versions they made, and its latest (what a request for the run's latest
        goes to)."""
        runs: list[dict[str, Any]] = []
        called: Mapping[str, str] = self.names.get("runs", {})
        for run in self.named(_RUNS, "/steps"):
            plan = self.record(f"{_RUNS}{run}/plan", "plan")
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
                        "trainer": step.get("trainer"),
                    }
                )
            kind = str(plan.get("kind") or "train")
            data: dict[str, Any] = plan.get("data") or {}
            published = self.read(f"{_RUNS}{run}/published")
            latest = max(published, key=lambda name: float(published[name]["at"])) if published else None
            fed = [note for note in self.published if note.get("run") == run]
            distilled = mode(data.get("sampled_by") or []) if kind == "distill" else None
            runs.append(
                {
                    "run": run,
                    "name": called.get(run, run),
                    "kind": kind,
                    "from": plan.get("from") or begun or (listed[0]["parent"] if listed else None),
                    "teachers": [str(each) for each in cast(list[Any], plan.get("teachers") or [])],
                    "data": data,
                    "objective": plan.get("objective"),
                    "evaluate": plan.get("evaluate"),
                    "mode": distilled,
                    "says": SAYS[distilled] if distilled else None,
                    "steps": listed,
                    "versions": [step["makes"] for step in listed if step["state"] == MADE],
                    "latest": latest or (str(fed[-1]["adapter"]) if fed else None),
                    "groups": self.waiting(run, listed),
                    "sample": f"{_RUNS}{run}/steps" in self.sampled,
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
        made: Mapping[str, Version],
        runs: list[dict[str, Any]],
        loads: Mapping[str, dict[str, dict[str, Any]]],
        waiting: Mapping[str, int],
    ) -> list[dict[str, Any]]:
        """Every version as the graph shows it: where it came from (its parents, its base, the run and step that made
        it), the bookmarks that name it, and where it is on its way to the engines."""
        shorter = short(made)
        by = {run["run"]: run for run in runs}
        marks: dict[str, list[str]] = {}
        for mark, version in self.names.get("bookmarks", {}).items():
            marks.setdefault(str(version), []).append(mark)
        latest = {run["latest"]: run["run"] for run in runs if run["latest"]}
        """The version a request for each run's latest goes to."""
        shown: list[dict[str, Any]] = []
        for version in made.values():
            run = by.get(version.run or "")
            older = {  # workers serving an older version of the same run in this one's place
                worker
                for each in made.values()
                if each.run == version.run and each.depth < version.depth
                for worker, span in loads.get(each.id, {}).items()
                if span["until"] is None
            }
            full = (self.record(f"{_RUNS}{version.run}/plan", "plan").get("weights") or "lora") == "full"
            life: dict[str, Any] = {
                "reshard": full,
                "resharding": self.record("versions/resharding", version.id).get("at"),
                "resharded": self.record("versions/resharded", version.id).get("at"),
                "latest_of": latest.get(version.id),
                "workers": loads.get(version.id, {}),
                "waiting": waiting.get(version.id, 0),
            }
            life["state"] = _state(life, older)
            shown.append(
                {
                    "id": version.id,
                    "short": shorter[version.id],
                    "depth": version.depth,
                    "parents": list(version.parents),
                    "base": version.base,
                    "made": version.made,
                    "kept": version.weights is not None,
                    "released": version.released,
                    "bookmarks": sorted(marks.get(version.id, [])),
                    "metrics": {
                        name: version.metrics[name] for name in ("kl_moved", "loss") if name in version.metrics
                    },
                    "by": {
                        "run": version.run,
                        "name": run["name"] if run else self.names.get("runs", {}).get(version.run, version.run),
                        "kind": run["kind"] if run else None,
                        "step": version.step,
                    }
                    if version.run
                    else None,
                    "life": life,
                    "sample": f"{VERSIONS}/{version.id}" in self.sampled,
                }
            )
        return shown

    def edges(self, made: Mapping[str, Version], runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """What each version grew from: its first parent (`trained`), or its base model (`base`, from `base:MODEL`);
        its other parents (`learned`); and each distillation's teachers and what its student starts from, each with
        what the distillation's mode means. With the versions they name that this ledger does not have."""
        edges: list[dict[str, Any]] = []
        for version in made.values():
            if not version.parents:
                edges.append({"kind": "base", "from": f"base:{version.base or 'the base model'}", "to": version.id})
            for index, parent in enumerate(version.parents):
                edges.append({"kind": "trained" if index == 0 else "learned", "from": parent, "to": version.id})
        for run in runs:
            if run["kind"] != "distill":
                continue
            said = {"mode": run["mode"], "says": run["says"]}
            edges += [{"kind": "teach", "from": teacher, "to": run["run"], **said} for teacher in run["teachers"]]
            if run["from"]:
                edges.append({"kind": "start", "from": run["from"], "to": run["run"], **said})
        outside = sorted(
            {edge["from"] for edge in edges if edge["from"] not in made and not edge["from"].startswith("base:")}
        )
        return edges, outside

    def loads(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Each version's workers: when each loaded it, and unloaded it (None: it still serves it). Where no worker
        is registered, a run's engines stand for one worker that loads what the feed says the run published, and
        lets each go at the next."""
        loads: dict[str, dict[str, dict[str, Any]]] = {}
        for worker in self.named(_WORKERS, "/loaded"):
            for key, record in self.read(f"{_WORKERS}{worker}/loaded").items():
                version = key.rpartition("/")[0] or key  # (keyed `VERSION/N`: a worker may load a version again)
                until = self.record(f"{_WORKERS}{worker}/unloaded", key).get("at")
                loads.setdefault(version, {})[worker] = {"since": record["at"], "until": until}
        by_run: dict[str, list[Mapping[str, Any]]] = {}
        for note in self.published:
            by_run.setdefault(f"{note.get('run') or note['channel']} engines", []).append(note)
        for worker, notes in by_run.items():
            for index, note in enumerate(notes):
                until = notes[index + 1]["at"] if index + 1 < len(notes) else None
                loads.setdefault(str(note["adapter"]), {})[worker] = {"since": note["at"], "until": until}
        return loads

    def workers(self, loads: Mapping[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Each inference worker: what it holds (a base, with slots for adapters, or one full-weight model), and the
        versions it serves now."""
        serving: dict[str, list[str]] = {}
        for version, spans in loads.items():
            for worker, span in spans.items():
                if span["until"] is None:
                    serving.setdefault(worker, []).append(version)
        workers: list[dict[str, Any]] = []
        for worker in sorted({*self.named(_WORKERS, "/registered"), *serving}):
            registrations = self.read(f"{_WORKERS}{worker}/registered")
            about: dict[str, Any] = registrations[max(registrations, key=int)] if registrations else {}
            workers.append(
                {
                    "worker": worker,
                    **about,
                    "registered": bool(registrations),
                    "serving": sorted(serving.get(worker, [])),
                    "sample": f"{_WORKERS}{worker}/registered" in self.sampled
                    or f"{_WORKERS}{worker}/loaded" in self.sampled,
                }
            )
        return workers

    def trainers(self, runs: list[dict[str, Any]], made: Mapping[str, Version]) -> list[dict[str, Any]]:
        """The registered trainers, each with its queue; and for a run whose steps name no trainer, the run's own,
        whose queue is the run's steps (one at a time: each is decided once the one before is done with)."""
        trainers: list[dict[str, Any]] = []
        for name in self.named(_TRAINERS, "/registered"):
            registrations = self.read(f"{_TRAINERS}{name}/registered")
            queue = [
                self.entry(entry, self.record(f"{_TRAINERS}{name}/taken", key).get("began"), runs, made)
                for key, entry in self.read(f"{_TRAINERS}{name}/queue").items()
            ]
            trainers.append(
                {
                    "trainer": name,
                    **registrations[max(registrations, key=int)],
                    "implicit": False,
                    "queue": queue,
                    "depth": _depth(queue),
                    "sample": f"{_TRAINERS}{name}/registered" in self.sampled,
                }
            )
        for run in runs:
            if run["kind"] != "train" or any(step["trainer"] for step in run["steps"]):
                continue
            line = [version for id, version in made.items() if id in run["versions"]]
            queue = [
                self.entry(
                    {"run": run["run"], "step": step["step"], "makes": step["makes"]} | {"at": step["decided"]},
                    step["decided"],
                    runs,
                    made,
                )
                for step in run["steps"]
                if step["decided"]
            ]
            trainers.append(
                {
                    "trainer": f"{run['name']} (the run's own)",
                    "weights": "lora",  # (a channel serves LoRA adapters only)
                    "base": next((version.base for version in line if version.base), None),
                    "runs": [run["run"]],
                    "colocated": any("waited_for_requests_seconds" in version.metrics for version in line),
                    "implicit": True,
                    "queue": queue,
                    "depth": _depth(queue),
                    "groups": run["groups"],
                    "sample": run["sample"],
                }
            )
        return trainers

    def entry(
        self, entry: Mapping[str, Any], began: float | None, runs: list[dict[str, Any]], made: Mapping[str, Version]
    ) -> dict[str, Any]:
        """A queued step, with when it began and ended: its version made, or its failure written."""
        run = next((each for each in runs if each["run"] == entry["run"]), None)
        step = next((each for each in run["steps"] if each["step"] == int(entry["step"])), None) if run else None
        version = made.get(str(entry["makes"]))
        failure = self.record(f"{_RUNS}{entry['run']}/failures", str(entry["step"]))
        return {
            "run": entry["run"],
            "step": int(entry["step"]),
            "makes": entry["makes"],
            "queued": entry.get("at"),
            "began": began,
            "done": version.made if version else failure.get("at"),
            "state": MADE if version else FAILED if failure else TAKING if began else QUEUED,
            "segments": step["segments"] if step else None,
        }

    def evaluations(self) -> list[dict[str, Any]]:
        """Each suite: its starts, and each subject that played it (a version, or another model), with how it did
        at each start."""
        suites: list[dict[str, Any]] = []
        for suite in self.named(_EVALUATIONS, "/starts"):
            starts = self.read(f"{_EVALUATIONS}{suite}/starts")
            subjects: list[dict[str, Any]] = []
            for subject in self.named(f"{_EVALUATIONS}{suite}/", "/results"):
                about = self.record(f"{_EVALUATIONS}{suite}/{subject}/subject", "subject")
                by_start: dict[str, list[dict[str, Any]]] = {}
                for key, result in self.read(f"{_EVALUATIONS}{suite}/{subject}/results").items():
                    by_start.setdefault(key.partition("-")[0], []).append(
                        {"solved": bool(result.get("solved")), "reward": result.get("reward")}
                        | {"run_id": result.get("run_id")}
                    )
                played = [each for listed in by_start.values() for each in listed]
                rewards = [float(each["reward"]) for each in played if each["reward"] is not None]
                subjects.append(
                    {
                        "subject": subject,
                        "kind": about.get("kind", "version"),
                        "version": about.get("version"),
                        "model": about.get("model"),
                        "episodes": int(about.get("episodes") or 1),
                        "asked_by": about.get("asked_by"),
                        "results": by_start,
                        "played": len(played),
                        "solved": sum(each["solved"] for each in played),
                        "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
                        "sample": f"{_EVALUATIONS}{suite}/{subject}/results" in self.sampled,
                    }
                )
            suites.append(
                {
                    "suite": suite,
                    "starts": [
                        {"start": key, **{name: record.get(name) for name in ("task", "title", "seed")}}
                        for key, record in sorted(starts.items(), key=lambda item: int(item[0]))
                    ],
                    "subjects": subjects,
                    "sample": f"{_EVALUATIONS}{suite}/starts" in self.sampled,
                }
            )
        return suites


def _state(life: Mapping[str, Any], older: set[str]) -> str:
    """Where a version is on its way to the engines (`WRITTEN` to `SERVING`, or `SUPERSEDED`). A run's latest rolls
    out until no worker serves an older version of that run in its place (`older`: those that serve one now); any
    other version serves where requests name it exactly."""
    workers: Mapping[str, Any] = life["workers"]
    now = {worker for worker, span in workers.items() if span["until"] is None}
    if life["latest_of"]:
        return SERVING if now and older <= now else ROLLING
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
