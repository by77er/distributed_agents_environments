"""Where every run of a ledger stands: what the monitor shows.

A ledger (`rollout_train.ledger`) may be shared by many runs. It holds what each training loop decided and what happened
(`rollout_train.record`), where and when each run was started, the versions and the fences. Each run's
episodes are in the ledger too, as runners claim, play and record them (`rollout_train.rollouts.scheduler`). Each run
keeps the rest in its own directory, as an open profile lays it out (`rollout_train.layout`): the feed (what is
happening now) and the episodes' events. A run's `starts` record says where its directory is and where the monitor on
its machine serves: `System` reads the directory where it is on this machine, asks that monitor otherwise
(`System._source` decides which), and else shows what the ledger alone has. It asks nothing of any run's process: it
says the same whether that process is alive or not, and what it says of a group is what a loop that started now would
find.
"""

import asyncio
import json
import lzma
import socket
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx
from pydantic import JsonValue

from rollout.contracts import BlobReference, Message, RunEvent, RunEventType
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout_train.launcher import LAUNCHER
from rollout_train.launches import ASKED, OPEN, STOPPED, STOPPING, Asked, Launch, launches_of
from rollout_train.layout import BLOBS, FEED, RUN
from rollout_train.ledger import FileLedger, Ledger, between, of_run, present
from rollout_train.monitor.feed import NOTES, FeedReader, plain
from rollout_train.monitor.lineage import lineage
from rollout_train.monitor.statistics import newest, statistics
from rollout_train.presence import Beat, alive, presence_of
from rollout_train.record import FAILURES, GROUPS, RESULTS, STARTS, STEPS, Result, named_runs, runs_in, table
from rollout_train.record import scope as run_scope
from rollout_train.registry import Bookmark, Entry, Registry, Taken, checked, found, names, registry_of, resolved
from rollout_train.rollouts.episodes import Outcome, Record
from rollout_train.rollouts.scheduler import CLAIMS, EPISODES, INTERRUPTED, runner_scope
from rollout_train.stores import opened
from rollout_train.versions import Manifest, Version, short, versions_in

WAITING = "waiting"
"""Asked for; no runner has claimed any of its episodes."""
PLAYING = "playing"
"""A runner has claimed one of its episodes, at least, and not all have ended."""
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

LOST = "lost"
"""How a launch is shown when its launcher stopped beating before it finished."""
RUN_TABLES = (GROUPS, RESULTS, STEPS, FAILURES, EPISODES, CLAIMS, INTERRUPTED)
"""A run's tables, as the page reads them."""
ARCHIVED = 8
"""Episodes read back from their events that are kept at a time."""
SHOWN = 240
"""Measurements of each kind in a snapshot: the newest."""

RUNNING, IDLE, GONE = "running", "idle", "ended"
"""A run's state, by when it last wrote anything this reads (its records in the ledger, and its feed where that can
be read): within `QUIET` seconds it is running, within `SILENT` idle, and after that ended. No process
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
        self._stores: dict[str, Blobs] = {}
        """The blob stores the runs' starts name, opened once each."""
        self._opened = _run_in(self.directory) if self.directory is not None else None
        """The id of the run in the directory this was opened on."""
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        """Episodes' records, as the page shows them, by run and key (`GROUP/EPISODE`): a record never changes."""

    @property
    def ledger(self) -> str:
        """Where the ledger is: its directory, or its database's URL."""
        if isinstance(self._ledger, FileLedger):
            return str(self._ledger.directory)
        return str(getattr(self._ledger, "url", type(self._ledger).__name__))

    def _source(self, run: str, starts: Mapping[str, Any], relayed: bool = False) -> "_Place | _Remote | None":
        """Where a run's episodes are seen (its feed, its episodes' events): the one place that decides.

        - its directory, where its newest start says, if that is on this machine; for a run that says nothing, the
          directory this was opened on, if the run is its (as its `run.json` says, or by its name);
        - else the monitor at the address its newest start names, which serves its machine's runs (unless this was
          asked by another monitor: then nothing more is asked of others);
        - else nowhere this can read: what is shown of the run is what the ledger has."""
        latest: Mapping[str, Any] = starts[max(starts, key=int)] if starts else {}
        if latest.get("directory"):
            where: Path | None = Path(str(latest["directory"])).expanduser().resolve()
        elif self.directory is not None and run == self._opened:
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
        done with and the ones that are), the versions (each with where it came from and the bookmarks that name it),
        the runners and what they play, what each channel serves and how fast, the machine, and what is kept."""
        tables: dict[str, dict[str, JsonValue]] = {}
        fences: dict[str, int] = {}
        versions: list[Version] = []
        if await asyncio.to_thread(present, self._ledger):
            fences = await self._ledger.fences()
            tables = await self._tables()
            versions = await versions_in(self._ledger)
        called = await names(registry_of(self._ledger))
        beats = await self._beats()
        snapshot = await asyncio.to_thread(self._assembled, tables, fences, versions, called, beats, relayed)
        return snapshot | {"names": called}

    async def rename(self, who: str, name: str) -> Entry:
        """Call the run that `who` is (its id or its name) `name` from now on, in the registry beside the ledger. A
        run from before the registry is registered under its key first. Raises `Taken` for a name it cannot have,
        `KeyError` when there is no such run (or no registry)."""
        registry = self._registry()
        if found(await registry.runs(), who) is None and who in await runs_in(self._ledger):
            await registry.create(who, id=who)
        return await registry.rename(who, name)

    async def bookmark(self, name: str, version: str) -> Bookmark:
        """Make a bookmark name the version `version` says (its id, the start of one, `RUN:STEP`, `RUN` or another
        bookmark), or move it there. Raises `Taken` for a name that cannot be one, `KeyError` for a reference that
        says no version (or no registry)."""
        registry = self._registry()
        found_version = await resolved(self._ledger, registry, version)
        if found_version is None:
            raise KeyError("a bookmark names a version, not the base model")
        return await registry.bookmark(name, found_version)

    async def unbookmark(self, name: str) -> None:
        """Take a bookmark away (the version stays). Raises `KeyError` when there is no such bookmark."""
        await self._registry().unbookmark(name)

    async def launches(self) -> dict[str, Any]:
        """The runs asked for, newest first, and the launchers alive with what each offers (its profiles, with the
        settings a launch may change, its catalogs, and whether it has room). A launch whose launcher stopped beating
        while it was claimed, running or stopping is shown as `lost`: what became of its run is not known."""
        found = launches_of(self._ledger)
        listed = await found.all() if found is not None and await asyncio.to_thread(present, self._ledger) else []
        now = time.time()
        launchers = [
            {"launcher": beat.runner, "at": beat.at, **beat.about}
            for beat in await self._beats()
            if beat.about.get("kind") == LAUNCHER and alive(beat, now)
        ]
        beating = {str(each["launcher"]) for each in launchers}
        shown = [
            asdict(each) | {"state": LOST, "detail": "its launcher stopped beating"}
            if each.state in OPEN and each.state != ASKED and each.launcher not in beating
            else asdict(each)
            for each in listed
        ]
        return {"launches": shown, "launchers": launchers}

    async def launch(self, body: Mapping[str, Any]) -> Launch:
        """Ask for a run (`rollout_train.launches.Asked`'s fields): a launcher alive that offers its profile and its
        catalog starts it. Raises `Taken` for what cannot be asked for (a name taken or no name, a setting the
        profile does not have), `KeyError` for what no launcher offers or a version no reference says."""
        launches, registry = launches_of(self._ledger), self._registry()
        if launches is None:
            raise KeyError("this ledger keeps no launches")
        try:
            asked = Asked(**{key: value for key, value in body.items() if key in Asked.__dataclass_fields__})
        except TypeError as error:
            raise Taken(f"a launch says its profile, its catalog and its name ({error})") from None
        offered = [each for each in (await self.launches())["launchers"] if each.get("playing", 0) is not None]
        profiles = [
            profile for each in offered for profile in each.get("profiles", []) if profile["profile"] == asked.profile
        ]
        if not profiles:
            raise KeyError(f"no launcher alive offers the profile {asked.profile!r}")
        catalogs = {catalog for each in offered for catalog in each.get("catalogs", [])}
        if catalogs and asked.catalog not in catalogs:
            raise KeyError(f"no launcher alive offers the catalog {asked.catalog!r}")
        checked(asked.name, "", await registry.runs())  # (a name another run has, or no name)
        unknown = [
            key for key in asked.settings if key not in profiles[0]["settings"] and not key.startswith("trainer.")
        ]
        if unknown:
            raise Taken(f"the profile {asked.profile!r} has no setting {', '.join(unknown)}")
        if asked.start:
            await resolved(self._ledger, registry, asked.start)  # (raises KeyError for a reference to nothing)
        return await launches.ask(asked)

    async def stop(self, id: str) -> Launch:
        """Ask a launch to stop: one not started yet is stopped at once; a run going is stopped by its launcher, at a
        group boundary. Raises `KeyError` when there is no such launch going."""
        launches = launches_of(self._ledger)
        found = next((each for each in await launches.all() if each.id == id), None) if launches else None
        if launches is None or found is None or found.state not in OPEN:
            raise KeyError(f"there is no launch {id} going")
        return await launches.note(id, state=STOPPED if found.state == ASKED else STOPPING)

    def _registry(self) -> Registry:
        registry = registry_of(self._ledger)
        if registry is None:
            raise KeyError("this ledger has no registry")
        return registry

    async def lineage(self, sample: bool = False) -> dict[str, Any]:
        """The policies as a graph, with what trains, serves and evaluates them (`rollout_train.monitor.lineage`).
        With `sample`, the fixture of the tables proposed for distillation, trainers, workers and evaluations is read
        beside the ledger."""
        tables = await self._tables()
        notes = _noted(await self._beats())
        every = [note for each in notes.values() for note in each]
        called = await names(registry_of(self._ledger))
        return await asyncio.to_thread(lineage, tables, every, names=called, sample=sample)

    async def statistics(self) -> dict[str, Any]:
        """Every run of the ledger in figures (`rollout_train.monitor.statistics`), with each run's engines'
        throughput from its runners' heartbeats, what the runs are called, and the runners' machines."""
        tables = await self._tables()
        beats = await self._beats()
        figures = await asyncio.to_thread(statistics, tables, _noted(beats))
        called = await names(registry_of(self._ledger))
        return {**figures, "names": {"runs": called["runs"]}, "machines": _machines(beats)}

    async def machines(self) -> dict[str, Any]:
        """Every runner's machine, as its heartbeats say: now, and over its recent beats."""
        return {"machines": _machines(await self._beats())}

    async def _beats(self) -> list[Beat]:
        """Every runner's newest heartbeat (none where there is no ledger: reading makes none)."""
        presence = presence_of(self._ledger)
        if presence is None or not await asyncio.to_thread(present, self._ledger):
            return []
        return await presence.beats()

    async def _tables(self) -> dict[str, dict[str, JsonValue]]:
        """Every table of the ledger, by name (none where there is no ledger: reading makes none)."""
        if not await asyncio.to_thread(present, self._ledger):
            return {}
        return {name: await self._ledger.read(name) for name in await self._ledger.tables()}

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
        versions = {version.id: version for version in await versions_in(self._ledger)}
        fences = await self._ledger.fences()
        return await asyncio.to_thread(self._group, run, str(number), record, tables, fences, versions, found)

    def _group(
        self,
        run: str,
        number: str,
        record: Mapping[str, Any],
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        versions: Mapping[str, Version],
        found: "_Place | _Remote | None",
    ) -> dict[str, Any]:
        place = found if isinstance(found, _Place) else None
        played = _Played(run, tables, fences, self._records)
        group = _group(number, record, tables, versions, played, place.feed.runs() if place else [])
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
            "version": _version(made, short(versions)) if made else None,
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
        replies and tool calls from the events its runner kept), from which its rollouts (one per model slot) are
        drawn; what it reported when it ended; and where it sits: its run, its group and its labels. An episode of a
        run on another machine is asked of the monitor there."""
        if run_id not in self._ended_by_id and await asyncio.to_thread(present, self._ledger):
            for name in await self._ledger.tables():
                if (run := between(name, "runs/", f"/{EPISODES}")) is not None:
                    _remembered(run, await self._ledger.read(name), self._records)
        known = self._ended_by_id.get(run_id)
        if known is not None and known[0] not in self._sources:  # (where its run's events are: found once)
            starts = await self._ledger.read(table(known[0], STARTS))
            await asyncio.to_thread(self._source, known[0], starts, relayed)
        place, ended, summary = await asyncio.to_thread(self._found, run_id)
        if place is None and not relayed:
            for remote in list(self._remotes.values()):
                if (answer := await asyncio.to_thread(remote.episode, run_id, after)) is not None:
                    return answer
        store = await self._store(known[0]) if known is not None else None
        store = store or (place.blobs if place is not None else None)
        if place is not None and summary is not None:
            source, lines = "feed", await asyncio.to_thread(place.feed.lines, run_id, after)
        elif store is not None and ended is not None and ended.get("events"):
            source, lines = "archive", (await self._archived(store, run_id, ended["events"]))[after:]
        else:
            source, lines = None, []
        labels: Any = (ended or {}).get("labels") or (summary or {}).get("labels") or {}  # (the record's, once kept)
        return {
            "run_id": run_id,
            "labels": labels,
            "state": (ended or {}).get("state") or (summary or {}).get("state"),
            "ended": {key: value for key, value in ended.items() if key != "events"} if ended else None,
            "source": source,
            "lines": lines,
        }

    @property
    def _ended_by_id(self) -> dict[str, tuple[str, dict[str, Any]]]:
        return {each["run_id"]: (run, each) for (run, _), each in list(self._records.items()) if each["run_id"]}

    def _found(self, run_id: str) -> tuple["_Place | None", dict[str, Any] | None, dict[str, Any] | None]:
        """An episode in the ledger (once it has ended) and in a feed (while the feed keeps it), and the directory on
        this machine that has it (where its run's events are kept)."""
        run, ended = self._ended_by_id.get(run_id, (None, None))
        for place in list(self._places.values()):
            summary = next((each for each in place.feed.runs() if each["run_id"] == run_id), None)
            if summary is not None:
                return place, ended, summary
        found = self._sources.get(run) if run is not None else None
        return (found if isinstance(found, _Place) else None), ended, None

    async def _store(self, run: str) -> Blobs | None:
        """The blob store a run's newest start says its blobs are in (as any machine opens it), if it says."""
        starts: Any = await self._ledger.read(table(run, STARTS))
        where: Any = starts[max(starts, key=int)].get("blobs") if starts else None
        if not where:
            return None
        key = json.dumps(where, sort_keys=True)
        if key not in self._stores:
            self._stores[key] = await asyncio.to_thread(opened, where)
        return self._stores[key]

    async def _archived(self, store: Blobs, run_id: str, events: Mapping[str, Any]) -> list[dict[str, Any]]:
        if run_id not in self._archive:
            data = await store.read(BlobReference.model_validate(events))
            lines = (await asyncio.to_thread(lzma.decompress, data)).decode().splitlines()
            self._archive[run_id] = _replayed([RunEvent.model_validate_json(line) for line in lines])
            while len(self._archive) > ARCHIVED:
                del self._archive[next(iter(self._archive))]
        return self._archive[run_id]

    def _assembled(
        self,
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        versions: list[Version],
        called: Mapping[str, Any],
        beats: list[Beat],
        relayed: bool,
    ) -> dict[str, Any]:
        noted = _noted(beats)
        beaten: dict[str, float] = {}  # each run's runners' newest beat
        for beat in beats:
            if beat.about.get("run"):
                beaten[str(beat.about["run"])] = max(beaten.get(str(beat.about["run"]), 0.0), beat.at)
        made = {version.id: version for version in versions}
        shorter = short(made)
        blobs = {digest: size for version in versions for digest, size in _blobs(version)}  # (each kept once)
        marks: dict[str, list[str]] = {}
        for mark, version in called["bookmarks"].items():
            marks.setdefault(str(version), []).append(mark)
        now = time.time()
        runs: list[dict[str, Any]] = []
        played: dict[str, _Played] = {}
        for run in named_runs(tables):
            starts: Any = tables.get(table(run, STARTS), {})
            found = self._source(run, starts, relayed)
            place = found if isinstance(found, _Place) else None
            own = {name: tables.get(table(run, name), {}) for name in RUN_TABLES}
            played[run] = _Played(run, own, fences, self._records)
            listed = _run(run, own, fences.get(run_scope(run)), made, played[run], place.feed.runs() if place else [])
            seen = self._read(run, starts, found, listed["wrote"], now, noted.get(run, []), beaten.get(run))
            runs.append(listed | seen | {"played": played[run].counts(), "name": called["runs"].get(run, run)})
        rank = {RUNNING: 0, IDLE: 1, GONE: 2}
        runs.sort(key=lambda run: (rank[run["state"]], -(run["written"] or 0.0), run["run"]))
        return {
            "at": round(now, 1),
            "name": self.directory.name if self.directory else self.ledger,
            "directory": str(self.directory) if self.directory else None,
            "ledger_at": self.ledger,
            "host": self.host,
            "written": newest(run["written"] for run in runs),
            "runs": runs,
            "versions": [
                _version(version, shorter) | {"bookmarks": sorted(marks.get(version.id, []))} for version in versions
            ],
            "bookmarks": dict(called["bookmarks"]),
            "runners": _runners(fences, played),
            "channels": [channel | {"run": run} for run, notes in noted.items() for channel in _channels(notes)],
            "ledger": {"fences": dict(fences), "tables": {name: len(records) for name, records in tables.items()}},
            "kept": {
                "versions": sum(blobs.values()),
                "episodes": sum(each.kept for each in played.values()),
            },
        }

    def _read(
        self,
        run: str,
        starts: Mapping[str, Any],
        found: "_Place | _Remote | None",
        wrote: float | None,
        now: float,
        notes: list[dict[str, Any]],
        beaten: float | None,
    ) -> dict[str, Any]:
        """What a run's start says (where it is, what started it), its engines (as its runners' heartbeats say),
        and what its episodes' place adds (its groups in flight with their episodes); when it last wrote or beat,
        and so whether it is running."""
        latest: Mapping[str, Any] = starts[max(starts, key=int)] if starts else {}
        added: dict[str, Any] = {"channels": _channels(notes)}
        written = newest([wrote, latest.get("started"), beaten])
        if isinstance(found, _Place):
            written = newest([written, found.written()])
        elif isinstance(found, _Remote) and (there := found.run(run)) is not None:
            added |= {key: there[key] for key in ("open", "done") if key in there}
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
            "from": latest.get("from"),
            "started": latest.get("started"),
            "starts": len(starts),
            "written": written,
        }


class _Place:
    """A run's directory on this machine, as it is read: its feed and its blobs."""

    def __init__(self, directory: Path, feed: FeedReader | None = None) -> None:
        self.directory = directory
        self.feed = feed if feed is not None else FeedReader(directory / FEED)
        self.blobs = FileBlobStore(directory / BLOBS)

    def written(self) -> float | None:
        """When the feed was last written."""
        times: list[Any] = [run["updated"] for run in self.feed.runs()]
        notes = self.feed.directory / f"{NOTES}.jsonl"
        times += [notes.stat().st_mtime] if notes.exists() else []
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
    played: "_Played",
    in_feed: list[dict[str, Any]],
) -> dict[str, Any]:
    """A run: its groups that are not done with, each with its stage, its episodes and its step; the ones that are,
    each with its result and what was done with it; and its steps, each with the groups that went into it. A group
    that gave nothing to train on is listed with the first step decided after it, and until there is one, with
    those the next step will cover (`next`)."""
    groups: Any = tables[GROUPS]
    entries = [_group(number, groups[number], tables, versions, played, in_feed) for number in sorted(groups, key=int)]
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
    played: "_Played",
    in_feed: list[dict[str, Any]],
) -> dict[str, Any]:
    """A group: its stage, its episodes (from the feed while they run, from the ledger once they end) and the step
    that covers it, if one does."""
    results: Any = tables[RESULTS]
    failures: Any = tables[FAILURES]
    count = group.get("episodes")
    ended = {each["run_id"]: each for each in played.ended.get(number, [])}
    episodes: dict[str, dict[str, Any]] = {}
    for each in reversed(in_feed):  # (oldest first)
        labels = each["labels"]
        if labels.get("run") == played.run and str(labels.get("group")) == number:
            episodes[each["run_id"]] = {
                "run_id": each["run_id"],
                "episode": labels.get("episode"),
                "state": each["state"],
                "samples": each["samples"],
                "slots": sorted(each["slots"]),
                "reward": next(iter(each["rewards"].values()), None),
                "updated": each["updated"],
                "in_feed": True,
                "interrupted": each["state"] == Outcome.CANCELLED.value and each["run_id"] not in ended,
            }
    for run_id, each in ended.items():
        shown = {name: value for name, value in each.items() if name not in ("events", "kept")}
        episodes.setdefault(run_id, {"samples": None, "updated": None, "in_feed": False}).update(shown)
    key, intent = next(
        ((key, step) for key, step in cast(Mapping[str, Any], tables[STEPS]).items() if int(number) in _covers(step)),
        (None, None),
    )
    if results.get(number) is not None:
        stage = DONE
    elif isinstance(count, int) and len(ended) >= count:
        stage = ENDED
    else:
        stage = PLAYING if ended or played.playing.get(number) else WAITING
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
        "count": count if isinstance(count, int) else None,
        "ended": len(ended),
        "playing": played.playing.get(number, []),
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
        "adapter": made.id if made else None,
        "step": step.get("step") if step else None,
        "step_state": step.get("state") if step else None,
        "version": made.depth if made else None,
        "update": dict(made.metrics) if made else None,
        "segments_trained": int(step.get("segments") or 0) if made and step else 0,
        "error": error,
        "seconds": round(ended - began, 1) if began else None,
    }


def _state(key: str, step: Mapping[str, Any], failures: Mapping[str, Any], versions: Mapping[str, Version]) -> str:
    """Where a step stands: failed, committed (its version made), or stepping."""
    return FAILED if key in failures else COMMITTED if _makes(step) in versions else STEPPING


def _covers(step: Mapping[str, Any]) -> list[int]:
    """The groups a step covers, by their numbers."""
    listed: list[Any] = step.get("groups") or []
    return [int(group) for group in listed]


def _makes(step: Mapping[str, Any]) -> str | None:
    """The version a step's decision names, by id."""
    makes = step.get("makes")
    return str(makes) if makes else None


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


def _run_in(directory: Path) -> str:
    """The id of the run in a directory: as its `run.json` says, or (a run from before the registry) its name."""
    path = directory / RUN
    return str(json.loads(path.read_text())["id"]) if path.exists() else directory.name


def _version(version: Version, shorter: Mapping[str, str]) -> dict[str, Any]:
    """A version as the page shows it: where it came from, what it was trained on, and what is kept of it."""

    def size(manifest: Manifest | None) -> int:
        return sum(blob.size for blob in manifest.files.values()) if manifest else 0

    return {
        "id": version.id,
        "short": shorter.get(version.id, version.id),
        "depth": version.depth,
        "parents": list(version.parents),
        "base": version.base,
        "run": version.run,
        "step": version.step,
        "made": version.made,
        "metrics": dict(version.metrics),
        "weights": {"files": len(version.weights.files), "bytes": size(version.weights)} if version.weights else None,
        "state": {"files": len(version.state.files), "bytes": size(version.state)} if version.state else None,
        "released": version.released,
        "batch": version.batch is not None,
    }


def _blobs(version: Version) -> list[tuple[str, int]]:
    return [(blob.sha256, blob.size) for manifest in (version.weights, version.state) if manifest
            for blob in manifest.files.values()]  # fmt: skip


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


class _Played:
    """A run's episodes as the ledger has them: those that ended, by group, and those runners play now (their claims
    that hold: not cut short, made by a runner whose fence is the one it made them under)."""

    def __init__(
        self,
        run: str,
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        records: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        self.run = run
        self.ended: dict[str, list[dict[str, Any]]] = {}
        self.playing: dict[str, list[dict[str, Any]]] = {}
        self.claims: list[dict[str, Any]] = []
        """Every claim, with whether it holds."""
        self.kept = 0
        """Bytes of the episodes' trajectories and events in the blob store."""
        self.sampled = 0
        for key, each in _remembered(run, tables.get(EPISODES, {}), records).items():
            self.ended.setdefault(key.split("/")[0], []).append(each)
            self.kept += each["kept"]
            self.sampled += each["sampled"]
        done, cut = tables.get(EPISODES, {}), tables.get(INTERRUPTED, {})
        for key, line in tables.get(CLAIMS, {}).items():
            claim: Any = line
            group, episode, attempt = key.split("/")
            holds = (
                f"{group}/{episode}" not in done
                and key not in cut
                and fences.get(runner_scope(str(claim["runner"]))) == claim["fence"]
            )
            made = {"group": int(group), "episode": episode, "attempt": int(attempt), "runner": claim["runner"]}
            made["at"] = claim.get("at")
            self.claims.append(made | {"holds": holds})
            if holds:
                self.playing.setdefault(group, []).append(made)
        self.interrupted = len(cut)

    def counts(self) -> dict[str, Any]:
        outcomes: dict[str, int] = {}
        for each in (each for group in self.ended.values() for each in group):
            outcomes[each["outcome"]] = outcomes.get(each["outcome"], 0) + 1
        return {
            "episodes": sum(map(len, self.ended.values())),
            "outcomes": outcomes,
            "playing": sum(map(len, self.playing.values())),
            "interrupted": self.interrupted,
            "sampled": self.sampled,
        }


def _remembered(
    run: str, lines: Mapping[str, JsonValue], records: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """A run's episodes that ended, by key, as the page shows them: each read once into `records`."""
    for key, line in lines.items():
        if (run, key) not in records:
            records[run, key] = _ended(Record.from_json(cast(Mapping[str, Any], line)))
    return {key: records[run, key] for key in lines}


def _ended(record: Record) -> dict[str, Any]:
    """An episode that ended, as the page shows it."""
    episode = record.episode
    return {
        "run_id": episode.run_id,
        "episode": str(episode.number),
        "state": episode.outcome.value,
        "outcome": episode.outcome.value,
        "detail": episode.detail or episode.excluded,
        "reward": episode.reward,
        "solved": episode.solved,
        "sampled": sum(record.sampled.values()),
        "labels": dict(episode.labels),
        "slots": sorted(episode.trajectories),
        "info": dict(episode.info),
        "events": record.events.model_dump(mode="json") if record.events else None,
        "kept": sum(blob.size for blob in (record.trajectories, record.events) if blob is not None),
    }


def _runners(fences: Mapping[str, int], played: Mapping[str, _Played]) -> list[dict[str, Any]]:
    """Every runner that has taken its fence in the ledger: what it plays now, and the claims it has made."""
    found: dict[str, dict[str, Any]] = {}
    for scope, fence in fences.items():
        if (name := between(scope, "runners/", "")) is not None:
            found[name] = {"runner": name, "fence": fence, "playing": [], "claims": 0, "last": None}
    for run, each in played.items():
        for claim in each.claims:
            runner = found.setdefault(
                str(claim["runner"]),
                {"runner": claim["runner"], "fence": None, "playing": [], "claims": 0, "last": None},
            )
            runner["claims"] += 1
            runner["last"] = newest([runner["last"], claim["at"]])
            if claim["holds"]:
                shown = {key: claim[key] for key in ("group", "episode", "attempt", "at")}
                runner["playing"].append({"run": run, **shown})
    return sorted(found.values(), key=lambda runner: (not runner["playing"], -(runner["last"] or 0.0)))


def _machines(beats: list[Beat]) -> list[dict[str, Any]]:
    """Every runner's machine as its heartbeats say: whether it is alive, what it said last, and its recent beats."""
    now = time.time()
    return [
        {"runner": beat.runner, "at": beat.at, "alive": alive(beat, now), **beat.about, "history": beat.history}
        for beat in sorted(beats, key=lambda each: (not alive(each, now), -each.at))
    ]


def _noted(beats: list[Beat]) -> dict[str, list[dict[str, Any]]]:
    """What each run's engines did, by run, as notes read from its runners' recent beats: each channel's
    throughput over each beat (`inference`), and each change in what it serves (`published`)."""
    notes: dict[str, list[dict[str, Any]]] = {}
    for beat in beats:
        run = str(beat.about.get("run") or "")
        if not run:
            continue
        serving: dict[str, Any] = {}
        for point in beat.history:
            listed: Any = point.get("channels") or []
            for channel in listed:
                name = str(channel.get("channel"))
                counts = {key: value for key, value in channel.items() if key not in ("channel", "adapter")}
                if serving.get(name) != channel.get("adapter"):
                    serving[name] = channel.get("adapter")
                    notes.setdefault(run, []).append(
                        {"kind": "published", "run": run, "channel": name, "adapter": channel.get("adapter"),
                         "version": channel.get("version"), "at": point["at"]}
                    )  # fmt: skip
                if channel.get("requests"):
                    notes.setdefault(run, []).append(
                        {"kind": "inference", "at": point["at"], "channel": name, **counts}
                    )
    return notes
