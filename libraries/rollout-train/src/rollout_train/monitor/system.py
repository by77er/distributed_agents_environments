"""Where every run of a ledger stands: what the monitor shows.

A ledger (`rollout_train.ledger`) may be shared by many runs. It holds what each training loop decided and what
happened (`rollout_train.record`), where and when each run was started, the policies' versions and the fences. Each
run keeps the rest in its own directory, as an open profile lays it out (`rollout_train.layout`): the jobs' logs (what
was asked for, the episodes that ended, what was acknowledged), the feed (what is happening now) and the episodes'
events. A run's `starts` record says where its directory is and where the monitor on its machine serves: `System`
reads the directory where it is on this machine, asks that monitor otherwise (`System._source` decides which), and
else shows what the ledger alone has. It asks nothing of any run's process: it says the same whether that process is
alive or not, and what it says of a group is what a loop that started now would find.
"""

import asyncio
import json
import lzma
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx
from pydantic import JsonValue

from rollout.contracts import BlobReference, Message, RunEvent, RunEventType
from rollout.harness.blobs import FileBlobStore
from rollout_train.layout import BLOBS, FEED, JOBS, PROCESSES
from rollout_train.ledger import FileLedger, Ledger, of_run, present
from rollout_train.monitor.feed import Appended, FeedReader, plain
from rollout_train.monitor.lineage import lineage
from rollout_train.monitor.statistics import newest, statistics
from rollout_train.policies import Manifest, Version, named, parsed, policies_in, versions_in
from rollout_train.policies import scope as policy_scope
from rollout_train.record import FAILURES, GROUPS, RESULTS, STARTS, STEPS, Result, named_runs, table
from rollout_train.record import scope as run_scope
from rollout_train.rollouts.episodes import Record
from rollout_train.rollouts.jobs import ACKNOWLEDGED, EPISODES, INTERRUPTED, TICKETS

DECIDED = "decided"
"""A group the loop decided to play and has not asked the job for (a loop that starts now asks for it)."""
WAITING = "waiting"
"""Asked for; none of its episodes has started."""
PLAYING = "playing"
ENDED = "ended"
"""Every episode has ended; its result is not written yet (a loop that starts now writes it)."""
DONE = "done"
"""Its result is written. What is trained on it is its step's business: a step of its own stage (`STEPPING`,
`COMMITTED` or `FAILED`), or none yet (it waits toward the next one)."""
STEPPING = "stepping"
"""A step decided that has neither made its version nor failed: the trainer has it, or a loop that starts now takes
it again."""
COMMITTED = "committed"
FAILED = "failed"

RUN_TABLES = (GROUPS, RESULTS, STEPS, FAILURES)
"""A run's tables, as the page reads them."""
ARCHIVED = 8
"""Episodes read back from their events that are kept at a time."""
SHOWN = 240
"""Measurements of each kind in a snapshot: the newest."""

RUNNING, IDLE, GONE = "running", "idle", "ended"
"""A run's state, by when it last wrote anything this reads (its records in the ledger, and its jobs' logs and feed
where those can be read): within `QUIET` seconds it is running, within `SILENT` idle, and after that ended. No process
is asked: a run may be on any machine."""
QUIET = 20 * 60
SILENT = 3 * 3600
FRESH = 5.0
"""Seconds what a monitor elsewhere said is kept before it is asked again."""
UNANSWERED = 30.0
"""Seconds a monitor elsewhere that did not answer is left before it is asked again."""
RELAYED = "x-rollout-monitor-relayed"
"""A header on what one monitor asks another: the one asked answers from its own machine only (so two monitors that
each take a run to be the other's never ask each other in turn)."""


class System:
    def __init__(
        self,
        directory: Path | None = None,
        feed: FeedReader | None = None,
        *,
        ledger: Ledger | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Over a run's `directory` (its ledger, as `rollout_train.ledger.of_run` finds it: every run that shares
        it), or over a `ledger` alone. `feed` reads the directory's feed (by default its `feed`). `client` asks the
        monitors on other machines for their runs' episodes."""
        if directory is None and ledger is None:
            raise ValueError("a run's directory or a ledger")
        self.directory = directory.resolve() if directory is not None else None
        self._ledger = ledger if ledger is not None else of_run(cast(Path, self.directory))
        self.machine = Machine(self.directory or Path.home())
        self.host = socket.gethostname()
        self._client = client or httpx.Client(timeout=2.0, headers={RELAYED: "1"})
        self._places: dict[Path, _Place] = {}
        """The runs' directories on this machine, as they are read."""
        if self.directory is not None:
            self._places[self.directory] = _Place(self.directory, feed)
        self._remotes: dict[str, _Remote] = {}
        """The monitors elsewhere that runs' starts name, by address."""
        self._sources: dict[str, _Place | _Remote | None] = {}
        """Where each run's episodes are, as it was last found."""
        self._archive: dict[str, list[dict[str, Any]]] = {}
        """Episodes read back from their events, the newest few."""

    @property
    def ledger(self) -> str:
        """Where the ledger is: its directory, or its database's URL."""
        if isinstance(self._ledger, FileLedger):
            return str(self._ledger.directory)
        return str(getattr(self._ledger, "url", type(self._ledger).__name__))

    def _source(self, run: str, starts: Mapping[str, Any], relayed: bool = False) -> "_Place | _Remote | None":
        """Where a run's episodes are (its jobs' logs, its feed, its episodes' events): the one place that decides.

        - its directory, where its newest start says, if that is on this machine; for a run that says nothing, the
          directory this was opened on, if the run is its (its job's log is there, or it is named after it);
        - else the monitor at the address its newest start names, which serves its machine's runs (unless this was
          asked by another monitor: then nothing more is asked of others);
        - else nowhere this can read: what is shown of the run is what the ledger has."""
        latest: Mapping[str, Any] = starts[max(starts, key=int)] if starts else {}
        if latest.get("directory"):
            where: Path | None = Path(str(latest["directory"])).expanduser().resolve()
        elif self.directory is not None and ((self.directory / JOBS / run).is_dir() or run == self.directory.name):
            where = self.directory
        else:
            where = None
        found: _Place | _Remote | None = None
        if where is not None and where.is_dir():
            found = self._places.setdefault(where, _Place(where))
        elif latest.get("address") and not relayed:
            address = str(latest["address"]).rstrip("/")
            found = self._remotes.setdefault(address, _Remote(address, self._client))
        self._sources[run] = found
        return found

    async def snapshot(self, relayed: bool = False) -> dict[str, Any]:
        """Where everything stands now: every run (where it is and whether it is running; its groups that are not
        done with and the ones that are), the policies' versions, the jobs, what each channel serves and how fast,
        the machine, and what is kept."""
        tables: dict[str, dict[str, JsonValue]] = {}
        fences: dict[str, int] = {}
        policies: list[dict[str, Any]] = []
        versions: list[Version] = []
        if await asyncio.to_thread(present, self._ledger):
            fences = await self._ledger.fences()
            tables = await self._tables()
            for policy in await policies_in(self._ledger):
                kept = await versions_in(self._ledger, policy)
                versions += kept
                policies.append(_policy(policy, kept, fences.get(policy_scope(policy))))
        return await asyncio.to_thread(self._assembled, tables, fences, policies, versions, relayed)

    async def lineage(self, sample: bool = False) -> dict[str, Any]:
        """The policies as a graph, with what trains, serves and evaluates them (`rollout_train.monitor.lineage`).
        With `sample`, the fixture of the tables proposed for distillation, trainers, workers and evaluations is read
        beside the ledger."""
        tables = await self._tables()
        notes = await asyncio.to_thread(self._notes, tables)
        every = [note for each in notes.values() for note in each]
        return await asyncio.to_thread(lineage, tables, every, sample=sample)

    async def statistics(self) -> dict[str, Any]:
        """Every run of the ledger in figures (`rollout_train.monitor.statistics`), with each run's engines'
        throughput from its feed, and the machine's measurements."""
        tables = await self._tables()
        notes = await asyncio.to_thread(self._notes, tables)
        figures = await asyncio.to_thread(statistics, tables, notes)
        return {**figures, "machine": await asyncio.to_thread(self.machine.shown)}

    async def _tables(self) -> dict[str, dict[str, JsonValue]]:
        """Every table of the ledger, by name (none where there is no ledger: reading makes none)."""
        if not await asyncio.to_thread(present, self._ledger):
            return {}
        return {name: await self._ledger.read(name) for name in await self._ledger.tables()}

    def _notes(self, tables: Mapping[str, Mapping[str, JsonValue]]) -> dict[str, list[dict[str, Any]]]:
        """What each run's job said in its feed, by run (for the runs whose directory is on this machine)."""
        notes: dict[str, list[dict[str, Any]]] = {}
        for run in named_runs(tables):
            found = self._source(run, tables.get(table(run, STARTS), {}), relayed=True)
            if isinstance(found, _Place):
                notes[run] = found.feed.job()
        return notes

    async def group(self, run: str, number: int, relayed: bool = False) -> dict[str, Any] | None:
        """One group: what was decided (the row and its start), its stage, its episodes with what each reported,
        its step and the version it made, and its outcome."""
        if not await asyncio.to_thread(present, self._ledger):
            return None
        tables = {name: await self._ledger.read(table(run, name)) for name in (*RUN_TABLES, STARTS)}
        record: Any = tables[GROUPS].get(str(number))
        if record is None:
            return None
        found = await asyncio.to_thread(self._source, run, tables[STARTS], relayed)
        if isinstance(found, _Remote) and (answer := await asyncio.to_thread(found.group, run, number)) is not None:
            return answer | {"episodes_at": found.address}
        versions = {
            version.name: version
            for policy in await policies_in(self._ledger)
            for version in await versions_in(self._ledger, policy)
        }
        return await asyncio.to_thread(self._group, run, str(number), record, tables, versions, found)

    def _group(
        self,
        run: str,
        number: str,
        record: Mapping[str, Any],
        tables: Mapping[str, Mapping[str, JsonValue]],
        versions: Mapping[str, Version],
        found: "_Place | _Remote | None",
    ) -> dict[str, Any]:
        place = found if isinstance(found, _Place) else None
        job, in_feed = (place.jobs().get(run), place.feed.runs()) if place else (None, [])
        group = _group(number, record, tables, versions, job, in_feed)
        step: Any = group["step"]
        made = versions.get(str(step.get("makes"))) if step else None
        result: Any = tables[RESULTS].get(number)
        return {
            **group,
            "run": run,
            "episodes_at": "here" if place else found.address if isinstance(found, _Remote) else None,
            "parameters": record.get("parameters"),
            "outcome": _done(number, result, record, step, made, group["error"])
            if group["stage"] == DONE and result
            else None,
            "result": _outcome(number, record, result) if result else None,
            "version": _policy(made.policy, [made], None)["versions"][0] if made else None,
        }

    def feeds(self, relayed: bool = False) -> list[dict[str, Any]]:
        """Every episode in the feeds of the runs' directories on this machine (and, unless `relayed`, those the
        monitors elsewhere serve), summarised, newest first."""
        found = [each for place in list(self._places.values()) for each in place.feed.runs()]
        if not relayed:
            found += [each for remote in list(self._remotes.values()) for each in remote.feeds()]
        return sorted(found, key=lambda each: each["started"], reverse=True)

    async def episode(self, run_id: str, after: int = 0, relayed: bool = False) -> dict[str, Any]:
        """One episode: the run's lines from index `after` on (from the feed, or, once the feed has let it go, its
        replies and tool calls from the events the job kept), from which its rollouts (one per model slot) are
        drawn; what it reported when it ended; and where it sits: its job, its group and its labels. An episode of a
        run on another machine is asked of the monitor there."""
        place, ended, summary = await asyncio.to_thread(self._found, run_id)
        if place is None and not relayed:
            for remote in list(self._remotes.values()):
                if (answer := await asyncio.to_thread(remote.episode, run_id, after)) is not None:
                    return answer
        if place is not None and summary is not None:
            source, lines = "feed", await asyncio.to_thread(place.feed.lines, run_id, after)
        elif place is not None and ended is not None and ended.get("events"):
            source, lines = "archive", (await self._archived(place, run_id, ended["events"]))[after:]
        else:
            source, lines = None, []
        labels: Any = (summary or {}).get("labels") or (ended or {}).get("labels") or {}
        return {
            "run_id": run_id,
            "labels": labels,
            "state": (ended or {}).get("state") or (summary or {}).get("state"),
            "ended": {key: value for key, value in ended.items() if key != "events"} if ended else None,
            "source": source,
            "lines": lines,
        }

    def _found(self, run_id: str) -> tuple["_Place | None", dict[str, Any] | None, dict[str, Any] | None]:
        """An episode in a job's log (once it has ended) and in a feed (while the feed keeps it), and the directory
        on this machine that has it."""
        for place in list(self._places.values()):
            jobs = place.jobs()
            ended = next((each for job in jobs.values() for each in job.episodes if each["run_id"] == run_id), None)
            summary = next((run for run in place.feed.runs() if run["run_id"] == run_id), None)
            if ended is not None or summary is not None:
                return place, ended, summary
        return None, None, None

    async def _archived(self, place: "_Place", run_id: str, events: Mapping[str, Any]) -> list[dict[str, Any]]:
        if run_id not in self._archive:
            data = await place.blobs.read(BlobReference.model_validate(events))
            lines = (await asyncio.to_thread(lzma.decompress, data)).decode().splitlines()
            self._archive[run_id] = _replayed([RunEvent.model_validate_json(line) for line in lines])
            while len(self._archive) > ARCHIVED:
                del self._archive[next(iter(self._archive))]
        return self._archive[run_id]

    def _assembled(
        self,
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        policies: list[dict[str, Any]],
        versions: list[Version],
        relayed: bool,
    ) -> dict[str, Any]:
        made = {version.name: version for version in versions}
        blobs = {digest: size for policy in policies for digest, size in policy.pop("blobs")}  # (each kept once)
        now = time.time()
        runs: list[dict[str, Any]] = []
        for run in named_runs(tables):
            starts: Any = tables.get(table(run, STARTS), {})
            found = self._source(run, starts, relayed)
            place = found if isinstance(found, _Place) else None
            listed = _run(
                run,
                {name: tables.get(table(run, name), {}) for name in RUN_TABLES},
                fences.get(run_scope(run)),
                made,
                place.jobs().get(run) if place else None,
                place.feed.runs() if place else [],
            )
            runs.append(listed | self._read(run, starts, found, listed["wrote"], now))
        rank = {RUNNING: 0, IDLE: 1, GONE: 2}
        runs.sort(key=lambda run: (rank[run["state"]], -(run["written"] or 0.0), run["run"]))
        read = list(self._places.values())
        return {
            "at": round(now, 1),
            "name": self.directory.name if self.directory else self.ledger,
            "directory": str(self.directory) if self.directory else None,
            "ledger_at": self.ledger,
            "host": self.host,
            "written": newest(run["written"] for run in runs),
            "processes": _processes(self.directory / PROCESSES) if self.directory else None,
            "runs": runs,
            "policies": policies,
            "jobs": [
                job.counts() | {"directory": str(place.directory)} for place in read for job in place.jobs().values()
            ],
            "channels": [
                channel | {"directory": str(place.directory)}
                for place in read
                for channel in _channels(place.feed.job())
            ],
            "ledger": {"fences": dict(fences), "tables": {name: len(records) for name, records in tables.items()}},
            "machine": self.machine.shown(),
            "kept": {
                "versions": sum(blobs.values()),
                "episodes": sum(job.kept for place in read for job in place.jobs().values()),
            },
        }

    def _read(
        self, run: str, starts: Mapping[str, Any], found: "_Place | _Remote | None", wrote: float | None, now: float
    ) -> dict[str, Any]:
        """What a run's start says (where it is, what started it) and what its episodes' place adds: its groups in
        flight with their episodes, its job and its engines, when it last wrote, and so whether it is running."""
        latest: Mapping[str, Any] = starts[max(starts, key=int)] if starts else {}
        added: dict[str, Any] = {"channels": [], "job": None}
        written = newest([wrote, latest.get("started")])
        if isinstance(found, _Place):
            jobs = found.jobs()
            added = {"channels": _channels(found.feed.job()), "job": jobs[run].counts() if run in jobs else None}
            written = newest([written, found.written()])
        elif isinstance(found, _Remote) and (there := found.run(run)) is not None:
            added = {key: there[key] for key in ("open", "done", "channels", "job") if key in there}
            written = newest([written, there.get("written")])
        quiet = now - written if written is not None else float("inf")
        return {
            **added,
            "state": RUNNING if quiet < QUIET else IDLE if quiet < SILENT else GONE,
            "host": latest.get("host"),
            "address": latest.get("address"),
            "directory": latest.get("directory") or (str(found.directory) if isinstance(found, _Place) else None),
            "episodes_at": "here" if isinstance(found, _Place) else found.address if found else None,
            "reached": found.reached if isinstance(found, _Remote) else None,
            "profile": latest.get("profile"),
            "policy": latest.get("policy"),
            "started": latest.get("started"),
            "starts": len(starts),
            "written": written,
        }


class _Place:
    """A run's directory on this machine, as it is read: its jobs' logs, its feed and its blobs."""

    def __init__(self, directory: Path, feed: FeedReader | None = None) -> None:
        self.directory = directory
        self.feed = feed if feed is not None else FeedReader(directory / FEED)
        self.blobs = FileBlobStore(directory / BLOBS)
        self._jobs: dict[str, _JobLog] = {}
        self._reading = threading.Lock()
        """Held while the jobs' logs are read: requests are answered in threads, and two must not read at once."""

    def jobs(self) -> dict[str, "_JobLog"]:
        """The jobs' logs, each read up to its end, by job (a run's job is named after the run)."""
        with self._reading:
            directory = self.directory / JOBS
            for path in sorted(directory.iterdir()) if directory.is_dir() else []:
                if path.is_dir() and path.name not in self._jobs:
                    self._jobs[path.name] = _JobLog(path)
            for job in self._jobs.values():
                job.refresh()
            return dict(self._jobs)

    def written(self) -> float | None:
        """When anything this reads here was last written: a job's log, or the feed."""
        times: list[Any] = [job.written for job in self.jobs().values()]
        times += [run["updated"] for run in self.feed.runs()]
        job = self.feed.directory / "_job.jsonl"
        times += [job.stat().st_mtime] if job.exists() else []
        return newest(each for each in times if each)


class _Remote:
    """A run's episodes on another machine, as the monitor there serves them (this same page's `/api/system`,
    `/api/groups/...`, `/api/episodes/...` and `/api/runs`), asked with the `RELAYED` header. What it says is kept for
    `FRESH` seconds; a monitor that does not answer is taken to have nothing, and left for `UNANSWERED` seconds."""

    def __init__(self, address: str, client: httpx.Client) -> None:
        self.address = address
        self.reached: bool | None = None
        """Whether it answered when last asked (None: not asked yet)."""
        self._client = client
        self._kept: dict[str, tuple[float, Any]] = {}

    def _get(self, path: str, keep: float = 0.0) -> Any:
        kept = self._kept.get(path)
        if kept is not None and time.time() - kept[0] < (keep if self.reached else max(keep, UNANSWERED)):
            return kept[1]
        try:
            answer = self._client.get(self.address + path, headers={RELAYED: "1"})
            found = answer.json() if answer.status_code == 200 else None
            self.reached = True
        except (httpx.HTTPError, ValueError):
            found, self.reached = None, False
        self._kept[path] = (time.time(), found)
        return found

    def run(self, name: str) -> dict[str, Any] | None:
        """The run as the monitor there has it: its groups in flight with their episodes, its job and engines."""
        system: Any = self._get("/api/system", FRESH)
        return next((run for run in system["runs"] if run["run"] == name), None) if system else None

    def group(self, run: str, number: int) -> dict[str, Any] | None:
        return self._get(f"/api/groups/{quote(run, safe='')}/{number}")

    def episode(self, run_id: str, after: int) -> dict[str, Any] | None:
        """The episode, if the monitor there has it."""
        found: Any = self._get(f"/api/episodes/{quote(run_id, safe='')}?after={after}")
        return found if found and found.get("source") else None

    def feeds(self) -> list[dict[str, Any]]:
        return self._get("/api/runs", FRESH) or []


def _run(
    run: str,
    tables: Mapping[str, Mapping[str, JsonValue]],
    fence: int | None,
    versions: Mapping[str, Version],
    job: "_JobLog | None",
    in_feed: list[dict[str, Any]],
) -> dict[str, Any]:
    """A run: its groups that are not done with, each with its stage, its episodes and its step; the ones that are,
    each with its result and what was done with it; and its steps, each with the groups that went into it. A group
    that gave nothing to train on is listed with the first step decided after it, and until there is one, with
    those the next step will cover (`next`)."""
    groups: Any = tables[GROUPS]
    entries = [_group(number, groups[number], tables, versions, job, in_feed) for number in sorted(groups, key=int)]
    done: list[dict[str, Any]] = []
    for entry in entries:
        if entry["stage"] == DONE:
            step: Any = entry["step"]
            made = versions.get(str(step.get("makes"))) if step else None
            key = str(entry["number"])
            line = _done(key, tables[RESULTS][key], groups[key], step, made, entry["error"])
            shown = ("run_id", "episode", "state", "reward", "solved", "interrupted", "slots", "outcome")
            done.append(line | {"episodes": [{key: each.get(key) for key in shown} for each in entry["episodes"]]})
    steps: Any = tables[STEPS]
    failures: Any = tables[FAILURES]
    listed: list[dict[str, Any]] = [
        {
            "step": int(key),
            "groups": _covers(intent),
            "skipped": [],
            "makes": _makes(intent),
            "parent": intent.get("parent"),
            "segments": intent.get("segments"),
            "decided": intent.get("decided"),
            "state": _state(key, intent, failures, versions),
            "error": failures[key].get("error") if key in failures else None,
        }
        for key, intent in sorted(steps.items(), key=lambda item: int(item[0]))
    ]
    results: Any = tables[RESULTS]

    def ended(number: int) -> float:
        result: Mapping[str, Any] = results.get(str(number)) or {}
        return float(result.get("time") or 0.0)

    def decided(step: Mapping[str, Any]) -> float:  # (a step that did not say when: once its last group had ended)
        return float(step["decided"] or max(map(ended, step["groups"]), default=0.0))

    covered = {number for step in listed for number in step["groups"]}
    upcoming: list[int] = []
    for entry in entries:
        if entry["number"] in covered:
            continue
        after = ended(entry["number"]) if entry["stage"] == DONE else None
        later = next((step for step in listed if after is not None and decided(step) >= after), None)
        (later["skipped"] if later else upcoming).append(entry["number"])
    return {
        "run": run,
        "fence": fence,
        "wrote": newest(
            [group.get("decided") for group in groups.values()]
            + [result.get("time") for result in results.values()]
            + [step["decided"] for step in listed]
            + [versions[step["makes"]].made for step in listed if step["makes"] in versions]
        ),
        "decided": len(groups),
        "open": [entry for entry in entries if entry["stage"] != DONE],
        "done": done,
        "steps": listed,
        "next": upcoming,
    }


def _group(
    number: str,
    group: Mapping[str, Any],
    tables: Mapping[str, Mapping[str, JsonValue]],
    versions: Mapping[str, Version],
    job: "_JobLog | None",
    in_feed: list[dict[str, Any]],
) -> dict[str, Any]:
    """A group: its stage, its episodes (from the feed while they run, from the job's log once they end) and the
    step that covers it, if one does."""
    results: Any = tables[RESULTS]
    failures: Any = tables[FAILURES]
    ticket = job.asked(number) if job else None
    asked = ticket["id"] if ticket else None
    episodes: dict[str, dict[str, Any]] = {}
    for each in reversed(in_feed):  # (oldest first)
        if asked and each["labels"].get("ticket") == asked:
            episodes[each["run_id"]] = {
                "run_id": each["run_id"],
                "episode": each["labels"].get("episode"),
                "state": each["state"],
                "samples": each["samples"],
                "slots": sorted(each["slots"]),
                "reward": next(iter(each["rewards"].values()), None),
                "updated": each["updated"],
                "in_feed": True,
            }
    for ended in job.of(asked) if job and asked else []:
        episodes.setdefault(ended["run_id"], {"samples": None, "updated": None, "in_feed": False}).update(ended)
    counted = [each for each in episodes.values() if "outcome" in each and not each.get("interrupted")]
    key, intent = next(
        ((key, step) for key, step in cast(Mapping[str, Any], tables[STEPS]).items() if int(number) in _covers(step)),
        (None, None),
    )
    result = results.get(number)
    if result is not None:
        stage = DONE
    elif ticket is None:
        stage = DECIDED
    elif len(counted) >= int(ticket["count"]):
        stage = ENDED
    else:
        stage = PLAYING if episodes else WAITING
    step = None
    if intent is not None:
        step = {name: value for name, value in intent.items() if name != "batch"} | {
            "step": int(str(key)),
            "makes": _makes(intent),
            "state": _state(str(key), intent, failures, versions),
        }
    return {
        "number": int(number),
        "task": group.get("task"),
        "title": group.get("title"),
        "decided": group.get("decided"),
        "stage": stage,
        "ticket": asked,
        "count": int(ticket["count"]) if ticket else None,
        "ended": len(counted),
        "episodes": sorted(episodes.values(), key=lambda each: (str(each.get("episode")), each["run_id"])),
        "step": step,
        "error": failures[str(key)].get("error") if key is not None and str(key) in failures else None,
    }


def _done(
    number: str, result: Any, group: Mapping[str, Any], step: Any, made: Version | None, error: str | None
) -> dict[str, Any]:
    """A group that is done with, as the page shows it: its result, and what was done with it (the version its step
    made and the trainer's statistics, or why the step failed), with how long it all took."""
    line = _outcome(number, group, result)
    ended = made.made if made else float(result.get("time") or 0.0)
    began = float(group.get("decided") or result.get("time") or 0.0)
    return {
        **line,
        "adapter": made.name if made else None,
        "step": step.get("step") if step else None,
        "step_state": step.get("state") if step else None,
        "version": made.number if made else None,
        "update": dict(made.metrics) if made else None,
        "segments_trained": int(step.get("segments") or 0) if made and step else 0,
        "error": error,
        "seconds": round(ended - began, 1) if began else None,
    }


def _number(label: Any) -> int | None:
    """A group's number from its label (written `39` or `0039`)."""
    return int(label) if isinstance(label, str) and label.isdigit() else None


def _state(key: str, step: Mapping[str, Any], failures: Mapping[str, Any], versions: Mapping[str, Version]) -> str:
    """Where a step stands: failed, committed (its version made), or stepping."""
    return FAILED if key in failures else COMMITTED if _makes(step) in versions else STEPPING


def _covers(step: Mapping[str, Any]) -> list[int]:
    """The groups a step covers, by their numbers."""
    listed: list[Any] = step.get("groups") or []
    return [int(group) for group in listed]


def _makes(step: Mapping[str, Any]) -> str | None:
    """The version a step's decision names."""
    policy = step.get("policy") or (parsed(step["parent"])[0] if step.get("parent") else None)
    return named(policy, int(step["number"])) if policy else None


def _replayed(events: list[RunEvent]) -> list[dict[str, Any]]:
    """A run's lines as the feed would have had them, from its events: what was sent to a model is kept only as a
    digest, so a sample has its reply and no messages."""
    requested: dict[str, tuple[float, Mapping[str, Any]]] = {}
    lines: list[dict[str, Any]] = []
    for event in events:
        at, payload = event.recorded_at.timestamp(), cast(Mapping[str, Any], event.payload)
        if event.type is RunEventType.EFFECT_REQUESTED and payload.get("kind") == "model.sample":
            requested[str(payload["effect_id"])] = (at, payload)
            continue
        if event.type is RunEventType.EFFECT_COMPLETED and str(payload.get("effect_id")) in requested:
            began, request = requested.pop(str(payload["effect_id"]))
            result: Any = payload.get("payload") or {}
            if "message" in result:
                lines.append(
                    {
                        "kind": "sample",
                        "slot": str(request["payload"]["session_id"]).rsplit("/", 1)[-1],
                        "effect_id": payload["effect_id"],
                        "at": at,
                        "seconds": round(at - began, 2),
                        "messages": [],
                        "tools": list(request["payload"].get("tools") or []),
                        "reply": plain(Message.model_validate(result["message"])),
                        "finish_reason": result.get("finish_reason"),
                    }
                )
                continue
        lines.append({"kind": "event", "seq": event.seq, "type": event.type.value, "at": at, "payload": payload})
    return lines


def _outcome(number: str, group: Mapping[str, Any], line: Mapping[str, Any]) -> dict[str, Any]:
    """A group's result as it is read (with what its record says), its failures said once each and briefly."""
    joined = asdict(Result.from_json(line, int(number), group))
    failures = list(dict.fromkeys(str(failure)[:300] for failure in joined["failures"]))
    return {**joined, "failures": failures}


def _policy(policy: str, versions: list[Version], fence: int | None) -> dict[str, Any]:
    def size(manifest: Manifest | None) -> int:
        return sum(blob.size for blob in manifest.files.values()) if manifest else 0

    def blobs(manifest: Manifest | None) -> list[tuple[str, int]]:
        return [(blob.sha256, blob.size) for blob in manifest.files.values()] if manifest else []

    return {
        "policy": policy,
        "fence": fence,
        "head": versions[-1].name if versions else None,
        "versions": [
            {
                "name": version.name,
                "number": version.number,
                "parent": version.parent,
                "made": version.made,
                "metrics": dict(version.metrics),
                "weights": {"files": len(version.weights.files), "bytes": size(version.weights)}
                if version.weights
                else None,
                "state": {"files": len(version.state.files), "bytes": size(version.state)} if version.state else None,
                "released": version.released,
                "batch": version.batch is not None,
            }
            for version in versions
        ],
        "blobs": [each for version in versions for each in blobs(version.weights) + blobs(version.state)],
    }


def _channels(job: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What each channel serves and how fast, from what the job and the engines said in the feed."""
    channels: dict[str, dict[str, Any]] = {}
    for line in job:
        if line.get("kind") not in ("published", "inference"):
            continue
        channel = channels.setdefault(str(line["channel"]), {"channel": line["channel"], "throughput": []})
        if line["kind"] == "published":
            channel.update(adapter=line.get("adapter"), version=line.get("version"), published=line.get("at"))
        else:
            channel["throughput"].append({key: value for key, value in line.items() if key not in ("kind", "channel")})
    for channel in channels.values():
        channel["throughput"] = channel["throughput"][-SHOWN:]
    return list(channels.values())


def _processes(record: Path) -> dict[str, Any] | None:
    """The run's process and the ones it noted having started, and whether each is there."""
    try:
        noted = json.loads(record.read_text())
        owner, processes = int(noted["owner"]), dict(noted["processes"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    started = [{"pid": int(pid), "name": str(name), "alive": _alive(int(pid))} for pid, name in processes.items()]
    return {"owner": owner, "alive": _alive(owner), "started": started}


def _alive(pid: int) -> bool:
    """Whether a process is there, on this host."""
    return Path(f"/proc/{pid}").exists()


class _JobLog:
    """A job's log, read as it grows: what was asked for, the episodes that ended, what was acknowledged."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.tickets: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self.kept = 0
        """Bytes of the episodes' trajectories and events in the blob store."""
        self.sampled = 0
        self.written = 0.0
        self._tickets, self._episodes = Appended(directory / TICKETS), Appended(directory / EPISODES)

    def refresh(self) -> None:
        self.tickets.extend(self._tickets.more())
        for line in self._episodes.more():
            record = Record.from_json(line)
            episode = record.episode
            self.kept += sum(blob.size for blob in (record.trajectories, record.events) if blob is not None)
            self.sampled += sum(record.sampled.values())
            self.episodes.append(
                {
                    "cursor": episode.cursor,
                    "ticket": episode.ticket,
                    "run_id": episode.run_id,
                    "episode": episode.labels.get("episode"),
                    "state": episode.outcome.value,
                    "outcome": episode.outcome.value,
                    "interrupted": episode.detail == INTERRUPTED,
                    "detail": episode.detail or episode.excluded,
                    "reward": episode.reward,
                    "solved": episode.solved,
                    "sampled": sum(record.sampled.values()),
                    "labels": dict(episode.labels),
                    "slots": sorted(episode.trajectories),
                    "info": dict(episode.info),
                    "events": record.events.model_dump(mode="json") if record.events else None,
                }
            )
        logs = [path for path in (self.directory / TICKETS, self.directory / EPISODES) if path.exists()]
        self.written = max((path.stat().st_mtime for path in logs), default=0.0)

    def asked(self, number: str) -> dict[str, Any] | None:
        """The ticket a group was asked for under, by the group's number (the label `group`)."""
        asked = [ticket for ticket in self.tickets if _number(ticket["labels"].get("group")) == int(number)]
        return asked[-1] if asked else None

    def of(self, ticket: str) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in episode.items() if key != "events"}
            for episode in self.episodes
            if episode["ticket"] == ticket
        ]

    def counts(self) -> dict[str, Any]:
        outcomes: dict[str, int] = {}
        for episode in self.episodes:
            outcome = "interrupted" if episode["interrupted"] else episode["outcome"]
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        acknowledged = self.directory / ACKNOWLEDGED
        return {
            "job": self.directory.name,
            "tickets": len(self.tickets),
            "episodes": len(self.episodes),
            "outcomes": outcomes,
            "last": self.episodes[-1]["cursor"] if self.episodes else 0,
            "acknowledged": int(acknowledged.read_text() or 0) if acknowledged.exists() else 0,
            "sampled": self.sampled,
        }


class Machine:
    """The machine the monitor is on (the run's, when they share one): memory, accelerators, and the disk the run's
    directory is on. It keeps the newest measurements, to show how they moved."""

    def __init__(self, directory: Path, keep: int = SHOWN) -> None:
        self.directory = directory
        self.history: deque[dict[str, Any]] = deque(maxlen=keep)

    def measure(self) -> dict[str, Any]:
        memory = {"available": None, "total": None}
        try:
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            memory = {
                "available": int(fields["MemAvailable"].split()[0]) * 1024,
                "total": int(fields["MemTotal"].split()[0]) * 1024,
            }
        except (OSError, KeyError, ValueError):
            pass
        disk = shutil.disk_usage(self.directory) if self.directory.exists() else None
        measured = {
            "at": round(time.time(), 1),
            "memory": memory,
            "accelerators": _accelerators(),
            "disk": {"free": disk.free, "total": disk.total} if disk else None,
        }
        self.history.append(measured)
        return measured

    def shown(self, within: float = 5.0) -> dict[str, Any]:
        """The newest measurement (taken now, unless one was within `within` seconds) and the ones before it."""
        if not self.history or time.time() - self.history[-1]["at"] > within:
            self.measure()
        return {"now": self.history[-1], "history": list(self.history)}

    async def watch(self, every: float = 15.0) -> None:
        """Measure for as long as this runs, so that there is a history when someone looks."""
        while True:
            await asyncio.to_thread(self.shown, every / 2)
            await asyncio.sleep(every)


def _accelerators() -> list[dict[str, Any]]:
    query = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        listed = subprocess.run(query, capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    accelerators: list[dict[str, Any]] = []
    for line in listed.strip().splitlines():
        name, used, total, busy = (part.strip() for part in line.split(","))
        try:
            accelerators.append(
                {"name": name, "used": int(used) * 2**20, "total": int(total) * 2**20, "busy": int(busy) / 100}
            )
        except ValueError:  # (a field the driver does not report)
            continue
    return accelerators
