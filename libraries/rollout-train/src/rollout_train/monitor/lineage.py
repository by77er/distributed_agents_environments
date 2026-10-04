"""The policies as a graph, with what trains, serves and evaluates them: what the monitor's policies view draws.

A policy is a line of versions (`rollout_train.policies`). Each version was made by a step of some run: a run's
`steps` table names the version each step makes. A version whose parent is another policy's is a fork. Beside the
graph stand the trainers that take the steps, the inference workers and what each serves, and evaluations.

What a ledger has today is read as it is: the policies' versions, the runs' steps (which stand for their trainer's
queue: a run takes one step at a time) and the `published` notes in the feed (which stand for what the run's engines
serve). The other tables read here are proposed in docs/research/policy-dag.md, and nothing appends them yet:

- `runs/RUN/plan`, `runs/RUN/published`: what a run was set up to do (a distillation's teachers, whose samples it
  trains on, its objective), and the version a request for the run's latest goes to, from when;
- `trainers/NAME/registered`, `trainers/NAME/queue`, `trainers/NAME/taken`: a trainer, the steps queued for it, and
  when it began each;
- `policies/NAME/definition`, `policies/NAME/resharding`, `policies/NAME/resharded`: what a policy's weights are,
  and a version's files rewritten as the engines load them;
- `workers/NAME/registered`, `workers/NAME/loaded`, `workers/NAME/unloaded`: an inference worker (what it holds, its
  adapter slots), and each version it loaded and let go;
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
from rollout_train.policies import Version, named, parsed

SAMPLE = Path(__file__).with_name("sample-lineage.json")
"""A fixture of the proposed tables: two more LoRA policies sharing a trainer, a full-weight one with a trainer of its
own, two distillations (off and on policy), a queue, resharding and a roll-out in progress, inference workers with
requests waiting, an evaluation suite."""
TIMES = ("made", "decided", "at", "began", "released", "time")
"""Fields that say when: in the fixture, written as seconds before now (negative), so it looks current when shown."""

ON_POLICY, OFF_POLICY, MIXED = "on-policy", "off-policy", "mixed"
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
let go of."""

_VERSION = TypeAdapter(Version)
_POLICIES, _RUNS, _TRAINERS, _WORKERS, _EVALUATIONS = "policies/", "runs/", "trainers/", "workers/", "evaluations/"


def lineage(
    tables: Mapping[str, Mapping[str, JsonValue]],
    notes: Sequence[Mapping[str, Any]] = (),
    *,
    sample: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """The graph a ledger's tables (by name) describe, with the feed's notes (`notes`). With `sample`, the
    fixture's tables and notes are read beside them (a table of the ledger's own is never replaced)."""
    now = time.time() if now is None else now
    sampled: set[str] = set()
    if sample:
        fixture: dict[str, Any] = _moved(json.loads(SAMPLE.read_text()), now)
        sampled = set(fixture["tables"]) - set(tables)
        tables = {**fixture["tables"], **tables}
        notes = [*notes, *fixture["notes"]]
    return _Reading(tables, sampled, notes, now).payload()


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


def mode(student: str, sampled_by: Iterable[str]) -> str:
    """Whether a distillation is on policy or off it, by whose samples it trains on (policies or versions, by name):
    the student's own, the others', or both."""
    policies = {name.rpartition("@")[0] if "@" in name else name for name in sampled_by}
    if policies == {student}:
        return ON_POLICY
    return MIXED if student in policies else OFF_POLICY


class _Reading:
    def __init__(
        self, tables: Mapping[str, Mapping[str, JsonValue]], sampled: set[str], notes: Sequence[Any], now: float
    ) -> None:
        self.tables, self.sampled, self.now = tables, sampled, now
        self.published = [note for note in notes if note.get("kind") == "published" and "@" in str(note.get("adapter"))]
        self.routing = [note for note in notes if note.get("kind") == "routing"]

    def named(self, before: str, after: str) -> list[str]:
        """What the tables' names hold between `before` and `after`: which policies, runs, trainers have them."""
        return sorted({each for table in self.tables if (each := between(table, before, after))})

    def read(self, table: str) -> dict[str, Any]:
        return cast(dict[str, Any], dict(self.tables.get(table, {})))

    def record(self, table: str, key: str) -> dict[str, Any]:
        """A table's record under `key`, or an empty one."""
        return cast(dict[str, Any], self.tables.get(table, {}).get(key) or {})

    def payload(self) -> dict[str, Any]:
        versions = self.versions()
        made = {version.name: version for line in versions.values() for version in line}
        runs = self.runs(made)
        by = {name: run for run in runs for name in run["versions"]}
        loads = self.loads()
        waiting: dict[str, int] = self.routing[-1]["waiting"] if self.routing else {}
        policies = [self.policy(policy, line, by, runs, loads, waiting) for policy, line in versions.items()]
        edges, outside = self.edges(policies, runs, made)
        return {
            "now": round(self.now, 1),
            "sample": bool(self.sampled),
            "policies": policies,
            "outside": outside,
            "runs": runs,
            "edges": edges,
            "trainers": self.trainers(runs, made),
            "workers": self.workers(loads),
            "routing": {
                "waiting": waiting,
                "history": [[note["at"], sum(note["waiting"].values())] for note in self.routing],
            },
            "evaluations": self.evaluations(),
        }

    def versions(self) -> dict[str, list[Version]]:
        lines: dict[str, list[Version]] = {}
        for policy in self.named(_POLICIES, "/versions"):
            released = self.read(f"{_POLICIES}{policy}/released")
            lines[policy] = [
                replace(version, weights=None, state=None, released=float(released[key]["at"]))
                if key in released
                else version
                for key, record in self.read(f"{_POLICIES}{policy}/versions").items()
                if (version := _VERSION.validate_python(record))
            ]
        return lines

    def runs(self, made: Mapping[str, Version]) -> list[dict[str, Any]]:
        """Every run that decided steps: what it was set up to do (a training run, unless its plan says otherwise),
        its steps, the versions they made, and its latest (what a request for the run's latest goes to)."""
        runs: list[dict[str, Any]] = []
        for run in self.named(_RUNS, "/steps"):
            plan = self.record(f"{_RUNS}{run}/plan", "plan")
            failures = self.read(f"{_RUNS}{run}/failures")
            steps = sorted(self.read(f"{_RUNS}{run}/steps").items(), key=lambda item: int(item[0]))
            policy = str(plan.get("student") or next((step.get("policy") for _, step in steps), "") or "")
            listed: list[dict[str, Any]] = []
            for key, step in steps:
                makes = named(str(step.get("policy") or policy), int(step["number"]))
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
            fed = [note for note in self.published if note.get("job") == run]
            runs.append(
                {
                    "run": run,
                    "kind": kind,
                    "policy": policy,
                    "from": plan.get("from"),
                    "teachers": [str(each) for each in cast(list[Any], plan.get("teachers") or [])],
                    "data": data,
                    "objective": plan.get("objective"),
                    "evaluate": plan.get("evaluate"),
                    "mode": mode(policy, data.get("sampled_by") or [policy]) if kind == "distill" else None,
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

    def policy(
        self,
        policy: str,
        line: list[Version],
        by: Mapping[str, dict[str, Any]],
        runs: list[dict[str, Any]],
        loads: Mapping[str, dict[str, dict[str, Any]]],
        waiting: Mapping[str, int],
    ) -> dict[str, Any]:
        definition = self.record(f"{_POLICIES}{policy}/definition", "definition") or None
        lora = definition is None or definition.get("weights") == "lora"
        latests = sorted(
            (parsed(run["latest"])[1], run["run"]) for run in runs if run["latest"] in {each.name for each in line}
        )
        latest = {named(policy, latests[-1][0]): latests[-1][1]} if latests else {}
        """The newest version a request for some run's latest goes to (several runs may go on with one policy)."""
        versions: list[dict[str, Any]] = []
        for version in line:
            older = {
                worker
                for each in line
                if each.number < version.number
                for worker, span in loads.get(each.name, {}).items()
                if span["until"] is None
            }
            run = by.get(version.name)
            step = next((each for each in run["steps"] if each["makes"] == version.name), None) if run else None
            key = str(version.number)
            life: dict[str, Any] = {
                "reshard": not lora,
                "resharding": self.record(f"{_POLICIES}{policy}/resharding", key).get("at"),
                "resharded": self.record(f"{_POLICIES}{policy}/resharded", key).get("at"),
                "latest_of": latest.get(version.name),
                "workers": loads.get(version.name, {}),
                "waiting": waiting.get(version.name, 0),
            }
            life["state"] = _state(life, older)
            versions.append(
                {
                    "name": version.name,
                    "number": version.number,
                    "parent": version.parent,
                    "made": version.made,
                    "kept": version.weights is not None,
                    "released": version.released,
                    "metrics": {
                        name: version.metrics[name] for name in ("kl_moved", "loss") if name in version.metrics
                    },
                    "by": {"run": run["run"], "kind": run["kind"], "step": step["step"] if step else None}
                    if run
                    else None,
                    "life": life,
                }
            )
        return {
            "policy": policy,
            "definition": definition,
            "fork": line[0].parent if line and line[0].parent and parsed(line[0].parent)[0] != policy else None,
            "head": line[-1].name if line else None,
            "versions": versions,
            "sample": f"{_POLICIES}{policy}/versions" in self.sampled,
        }

    def edges(
        self, policies: list[dict[str, Any]], runs: list[dict[str, Any]], made: Mapping[str, Version]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Forks (a version whose parent is another policy's), and each distillation's teachers and the version its
        student starts from; with the versions they name that this ledger does not have."""
        edges: list[dict[str, Any]] = []
        distilled = {name: run for run in runs if run["kind"] == "distill" for name in run["versions"]}
        for policy in policies:
            for version in policy["versions"]:
                parent = version["parent"]
                if not parent or parsed(parent)[0] == policy["policy"]:
                    continue
                run = distilled.get(version["name"])
                if run is None or run["from"] != parent:  # (a distillation's start is drawn into the distillation)
                    edges.append({"kind": "fork", "from": parent, "to": version["name"]})
        for run in runs:
            if run["kind"] != "distill":
                continue
            edges += [
                {"kind": "teach", "from": teacher, "to": run["run"], "mode": run["mode"]} for teacher in run["teachers"]
            ]
            if run["from"]:
                edges.append({"kind": "start", "from": run["from"], "to": run["run"], "mode": run["mode"]})
        outside = sorted({edge["from"] for edge in edges if edge["from"] not in made})
        return edges, outside

    def loads(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Each version's workers: when each loaded it, and let go of it (None: it still serves it). Where no worker
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
            by_run.setdefault(f"{note.get('job') or note['channel']} engines", []).append(note)
        for worker, notes in by_run.items():
            for index, note in enumerate(notes):
                until = notes[index + 1]["at"] if index + 1 < len(notes) else None
                loads.setdefault(str(note["adapter"]), {})[worker] = {"since": note["at"], "until": until}
        return loads

    def workers(self, loads: Mapping[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Each inference worker: what it holds (a base, with slots for adapters, or one full-weight policy), and the
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
            line = [version for name, version in made.items() if name in run["versions"]]
            queue = [
                self.entry(
                    {"run": run["run"], "step": step["step"], "policy": run["policy"], "makes": step["makes"]}
                    | {"at": step["decided"]},
                    step["decided"],
                    runs,
                    made,
                )
                for step in run["steps"]
                if step["decided"]
            ]
            trainers.append(
                {
                    "trainer": f"{run['run']} (the run's own)",
                    "weights": "lora",  # (a channel serves LoRA adapters only)
                    "base": None,
                    "policies": [run["policy"]],
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
            "policy": entry.get("policy"),
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
                        "version": about.get("version") or (subject if "@" in subject else None),
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
    out until no worker serves an older version of its policy in its place (`older`: those that serve one now); any
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
