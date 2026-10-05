"""Where every run of a ledger stands: what the monitor shows.

A ledger (`rollout_train.ledger`) may be shared by many runs. It holds what each training loop decided and what happened
(`rollout_train.record`), where and when each run was started, the checkpoints and the fences. Each run's
episodes are in the ledger too, as runners claim, play and record them (`rollout_train.rollouts.scheduler`). Each run
keeps the rest in its own directory, as a run's driver lays it out (`rollout_train.layout`): the feed (what is
happening now) and the episodes' events. A run's `starts` record says where its directory is and where the monitor on
its machine serves: `System` reads the directory where it is on this machine, asks that monitor otherwise
(`System._source` decides which), and else shows what the ledger alone has. It asks nothing of any run's process: it
says the same whether that process is alive or not, and what it says of a group is what a loop that started now would
find.
"""

import asyncio
import contextlib
import json
import lzma
import random
import socket
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import httpx
from pydantic import JsonValue

from rollout.contracts import BlobReference, Message, RunEvent, RunEventType
from rollout.environment import Environment, Start
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.names import named
from rollout_train.checkpoints import Checkpoint, Manifest, checkpoints_in, short
from rollout_train.datasets import where_blobs_are
from rollout_train.evals import (
    DRAWN,
    EVAL,
    EVAL_DATA,
    GIVEN,
    Suite,
    SuiteEntry,
    edit_suite,
    make_suite,
    parsed,
    suite_entry,
    suite_of,
)
from rollout_train.gateway.turns import TurnStore
from rollout_train.inference.remote import ENGINES
from rollout_train.launches import OPEN, TRAIN, Launch, launch_of, launches_of
from rollout_train.launching import Refused as LaunchRefused
from rollout_train.launching import checked as findings_of
from rollout_train.launching import examined, offers, settled
from rollout_train.layout import BLOBS, FEED, RUN
from rollout_train.ledger import FileLedger, Ledger, between, of_run, present
from rollout_train.monitor.environments import Read, described, listed, page_of
from rollout_train.monitor.feed import NOTES, FeedReader, plain
from rollout_train.monitor.lineage import lineage
from rollout_train.monitor.machines import kind_of, machines
from rollout_train.monitor.scores import CHECKPOINT, evals_in, evals_of, history_of, path_of, subjects_in, suites_in
from rollout_train.monitor.statistics import newest, solved_of, statistics, unreported
from rollout_train.presence import STALE, Beat, alive, presence_of
from rollout_train.presets import Preset, presets_of
from rollout_train.published import EnvironmentVersion, environment_versions_of, is_published
from rollout_train.publishing import Importer, Refused, Source, publish
from rollout_train.record import (
    ENDINGS,
    ENDS,
    FAILURES,
    FINISHED,
    GROUPS,
    RESULTS,
    STARTS,
    STEPS,
    Result,
    named_runs,
    newest_record,
    runs_in,
    table,
)
from rollout_train.record import scope as run_scope
from rollout_train.registry import (
    Bookmark,
    Entry,
    Registry,
    Taken,
    names,
    registry_of,
    resolved,
    valid,
)
from rollout_train.resuming import Resumed, pause, resume
from rollout_train.rollouts.episodes import Outcome, Record
from rollout_train.rollouts.scheduler import ADOPTED, CLAIMS, EPISODES, INTERRUPTED, Claims, of_episode
from rollout_train.run_settings import RunSettings
from rollout_train.sandboxes import leases_of
from rollout_train.serving import SERVING
from rollout_train.settings import EVALS_SUITE, TRAINER, Desired, desired_settings_of
from rollout_train.settings import PAUSED as PAUSE
from rollout_train.settings import checked as checked_setting
from rollout_train.stores import opened
from rollout_train.submitting import backend_of, followed, stopped, submit
from rollout_train.validation import with_weights

if TYPE_CHECKING:
    from rollout_train.cluster import Cluster
    from rollout_train.submitting import Backend

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
"""A step decided that has neither made its checkpoint nor failed: the trainer has it, or a loop that starts now takes
it again."""
COMMITTED = "committed"
FAILED = "failed"

LOST = "lost"
"""How a run is shown whose runners beat once and stopped with no word of how it ended."""
RUN_TABLES = (GROUPS, RESULTS, STEPS, FAILURES, EPISODES, CLAIMS, INTERRUPTED, ADOPTED, ENDS)
"""A run's tables, as the page reads them."""
ARCHIVED = 8
"""Episodes read back from their events that are kept at a time."""
SHOWN = 240
"""Measurements of each kind in a snapshot: the newest."""

RUNNING, IDLE, GONE, PAUSED = "running", "idle", "ended", "paused"
"""A run's process is there and writing; there and quiet; not heard from (its runners stopped beating, or never
beat); there and paused, as a runner's beat says (`rollout_train.resuming`). Its runners
beat and stopped with no word of how it ended: lost (it crashed or was killed). A run that said how it ended is in the
state it said (`FINISHED`, `STOPPED`, `FAILED`).

Where its runners beat (`rollout_train.presence`), by their newest beat: one within `STALE` seconds
(by the clock of the store that keeps the beats),
and it is running (idle if it wrote nothing for `QUIET` seconds); none, and its process is gone: ended. An eval that
played every start has ended; one a training run's schedule asked for, and not done, is as that run is. A run whose
runners never beat is running for `STALE` seconds after it started (its first beat is yet to come), and ended after.
No process is asked: a run may be on any machine."""
QUIET = 20 * 60
"""Seconds a run whose runners beat may write nothing before it is idle."""
FRESH = 5.0
"""Seconds what a monitor elsewhere said is kept before it is asked again."""
UNANSWERED = 30.0
"""Seconds a monitor elsewhere that did not answer is left before it is asked again."""
IMPORTS = 20
"""Imports a monitor keeps word of, the newest."""
NO_CLUSTER = "this monitor asks for no runs: start it with the cluster config (`rollout monitor WHERE --cluster`)"
"""Why a monitor started without a cluster config refuses to launch."""
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
        importer: Importer | None = None,
        cluster: "Cluster | None" = None,
        backends: "Mapping[str, Backend] | None" = None,
    ) -> None:
        """Over a run's `directory` (its ledger, as `rollout_train.ledger.of_run` finds it: every run that shares
        it), or over a `ledger` alone. `feed` reads the directory's feed (by default its `feed`). `client` asks the
        monitors on other machines for their runs' episodes. `importer` is where environments imported from git go
        (`import_environment`); none: this monitor imports none."""
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
        self._turn_stores: dict[int, TurnStore] = {}
        """The turns the gateway recorded, read from each blob store (by its identity)."""
        self._opened = _run_in(self.directory) if self.directory is not None else None
        """The id of the run in the directory this was opened on."""
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        """Episodes' records, as the page shows them, by run and key (`GROUP/EPISODE`): a record never changes."""
        self._environments: dict[str, Environment] = {}
        """The environments the suites' forms named, loaded once each."""
        self._reading: dict[str, asyncio.Future[Any]] | None = None
        """What was read within a reading (`one_reading`), by what it is."""
        self._importer = importer
        self._cluster = cluster
        """The cluster config runs are asked for on (`launch`), and their jobs started and read with; none: this
        monitor asks for none."""
        own = {_backend_name(cluster): backend_of(cluster)} if cluster is not None else None
        self._backends = dict(backends) if backends is not None else own
        """Where runs' jobs go, by `Launch.backend`: the cluster config's (made once), or those given (a test's)."""
        self._versions = environment_versions_of(self._ledger)
        self._imports: list[dict[str, Any]] = []
        """The imports this monitor made since it started, newest first (the newest `IMPORTS`): each with its stage."""

    @property
    def ledger(self) -> str:
        """Where the ledger is: its directory, or its database's URL."""
        if isinstance(self._ledger, FileLedger):
            return str(self._ledger.directory)
        return str(getattr(self._ledger, "url", type(self._ledger).__name__))

    def _source(self, run: str, starts: Mapping[str, Any], relayed: bool = False) -> "_Place | _Remote | None":
        """Where a run's episodes are seen (its feed, its episodes' events): the one place that decides.

        - its directory, where its newest start says, if that is on this machine; for a run that says nothing, the
          directory this was opened on, if the run is its (as its `run.json` says);
        - else the monitor at the address its newest start names, which serves its machine's runs (unless this was
          asked by another monitor: then nothing more is asked of others);
        - else nowhere this can read: what is shown of the run is what the ledger has."""
        latest = newest_record(starts)
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
        done with and the ones that are), the checkpoints (each with where it came from and the bookmarks that name it),
        the runners and what they play, what each channel serves and how fast, the machine, and what is kept."""
        tables: dict[str, dict[str, JsonValue]] = {}
        fences: dict[str, int] = {}
        checkpoints: list[Checkpoint] = []
        if await asyncio.to_thread(present, self._ledger):
            fences = await self._ledger.fences()
            tables = await self._tables()
            checkpoints = await self._checkpoints()
        called = await self._names()
        beats = await self._beats()
        snapshot = await asyncio.to_thread(self._assembled, tables, fences, checkpoints, called, beats, relayed)
        store = desired_settings_of(self._ledger)
        for run in snapshot["runs"]:  # (whether it is wanted paused: its process notes it in its beats, `state`)
            wanted = await store.desired(run["run"]) if store is not None and tables else None
            run["pause"] = wanted is not None and wanted.settings.get(PAUSE) is True
        return snapshot | {"names": called}

    async def pause(self, run: str) -> Desired:
        """Pause a run (`rollout_train.resuming.pause`). Raises `KeyError` where there is no such run."""
        return await pause(self._ledger, run)

    async def resume(self, run: str, preset: str | None = None) -> Resumed:
        """Resume a run: in place, or by a launch of its recorded settings (over `preset`'s, for a run whose start
        records no providers: `rollout_train.resuming.resume`). Raises `Taken` for a run that cannot be resumed,
        `KeyError` where there is no such run or this monitor has no cluster config to submit on,
        `rollout_train.launching.Refused` for settings the cluster refuses."""
        backend = self._backends.get(_backend_name(self._cluster)) if self._backends and self._cluster else None
        try:
            return await resume(self._ledger, run, cluster=self._cluster, preset=preset, backend=backend)
        except LaunchRefused:
            raise
        except ValueError as error:
            raise Taken(str(error)) from None

    async def rename(self, who: str, name: str) -> Entry:
        """Call the run that `who` is (its id or its name) `name` from now on, in the registry beside the ledger.
        Raises `Taken` for a name it cannot have, `KeyError` when there is no such run (or no registry)."""
        return await self._registry().rename(who, name)

    async def bookmark(self, name: str, checkpoint: str) -> Bookmark:
        """Make a bookmark name the checkpoint `checkpoint` says (its id, the start of one, `RUN:STEP`, `RUN` or another
        bookmark), or move it there. Raises `Taken` for a name that cannot be one, `KeyError` for a reference that
        says no checkpoint (or no registry)."""
        registry = self._registry()
        found_checkpoint = await resolved(self._ledger, registry, checkpoint)
        if found_checkpoint is None:
            raise KeyError("a bookmark names a checkpoint, not the base model")
        return await registry.bookmark(name, found_checkpoint)

    async def unbookmark(self, name: str) -> None:
        """Take a bookmark away (the checkpoint stays). Raises `KeyError` when there is no such bookmark."""
        await self._registry().unbookmark(name)

    async def launches(self) -> dict[str, Any]:
        """The runs asked for, newest first, each with the job it became and its state: a launch that is going is
        read with its job's status too (`rollout_train.submitting.followed`), so a job that waits says why, and one
        that died without its driver saying so is failed."""
        found = launches_of(self._ledger)
        listed = await found.all() if found is not None and await asyncio.to_thread(present, self._ledger) else []
        if found is not None:
            listed = [
                await followed(each, found, self._cluster, backends=self._backends) if each.state in OPEN else each
                for each in listed
            ]
        return {"launches": [asdict(each) for each in listed], "submits": self._cluster is not None}

    async def offers(self) -> dict[str, Any]:
        """What a run can be asked for here (`rollout_train.launching.offers`); nothing where this monitor was started
        without a cluster config."""
        if self._cluster is None:
            return {"cluster": None, "environments": [], "trainers": [], "inference": [], "pairs": [], "presets": [],
                    "sandboxes": {}, "kinds": [], "capacity": None}  # fmt: skip
        return await offers(self._cluster, self._ledger, await self._beats())

    async def _asked(self, body: Mapping[str, Any]) -> tuple[RunSettings, str | None]:
        """A run's settings as a launch's body says them (`kind`, `name`, `environment`, `settings`, `preset`): the
        preset's, then those given. An eval names its suite (`eval.suite`), whose environment it plays. Raises `Taken`
        for settings that are not a table, `KeyError` for a preset or a suite there is none of."""
        kind = str(body.get("kind") or TRAIN)
        name = str(body.get("name") or "").strip()
        given: Any = body.get("settings") or {}
        if not isinstance(given, dict):
            raise Taken("a launch's settings are a table of run settings, by dotted key")
        settings: dict[str, JsonValue] = dict(cast(dict[str, JsonValue], given))
        if body.get("environment"):
            settings["environment"] = str(body["environment"])
        preset = str(body["preset"]) if body.get("preset") else None
        said, chosen = await settled(kind, name or None, settings, preset=preset, presets=presets_of(self._ledger))
        if kind == EVAL and said.get("environment") is None and isinstance(said.get("eval.suite"), str):
            suite = await suite_of(self._ledger, str(said["eval.suite"]))
            if suite is None:
                raise KeyError(f"there is no suite {said['eval.suite']!r}")
            said = RunSettings({**said.values, "environment": suite.environments[0]})
        return said, chosen

    async def check(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """What a launch's body would be refused for, and the notes beside (each with the setting it is about), on
        this monitor's cluster (`rollout_train.launching.examined`); the settings it would run with, what it trains,
        one step's estimated spend on its metered parts (or why it cannot be estimated yet), and the slots its
        environment's programs declare, where they are known here. Raises `Taken` where this monitor has no cluster
        config, or for a body it cannot read."""
        if self._cluster is None:
            raise Taken(NO_CLUSTER)
        settings, preset = await self._asked(body)
        found = await examined(settings, self._cluster, self._ledger)
        facts = found.environment
        slots = None
        if facts is not None and facts.slots is not None:
            slots = {"slots": sorted(facts.slots), "untrained": sorted(facts.untrained), "judges": sorted(facts.judges)}
        return {
            "refusals": [asdict(each) for each in found.findings if each.refuses],
            "notes": [asdict(each) for each in found.findings if not each.refuses],
            "settings": dict(settings.values), "preset": preset, "weights": found.weights,
            "spend": {"dollars": found.spend.dollars, "parts": dict(found.spend.parts), "why": found.spend.why},
            "environment": slots,
        }  # fmt: skip

    async def presets(self) -> dict[str, Any]:
        """Every preset's newest version, each with how many versions it has (`rollout_train.presets`)."""
        store = presets_of(self._ledger)
        if store is None or not await asyncio.to_thread(present, self._ledger):
            return {"presets": [], "keeps": store is not None}
        listed: list[dict[str, Any]] = []
        for each in await store.all():
            listed.append({**_preset(each), "versions": len(await store.versions(each.name))})
        return {"presets": listed, "keeps": True}

    async def preset(self, name: str) -> dict[str, Any] | None:
        """A preset's versions, oldest first; none where there is no such preset (or it was deleted)."""
        store = presets_of(self._ledger)
        if store is None or await store.get(name) is None:
            return None
        return {"name": name, "versions": [_preset(each) for each in await store.versions(name)]}

    async def save_preset(self, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Save `body`'s settings (`{"settings": {KEY: VALUE}, "note"}`) as a preset's next version. Raises `Taken`
        for a name that cannot be one, settings that are not a table of run settings, or a ledger that keeps no
        presets."""
        store = presets_of(self._ledger)
        if store is None:
            raise Taken("this ledger keeps no presets beside it")
        settings: Any = body.get("settings")
        if not isinstance(settings, dict):
            raise Taken("a preset's settings are a table of run settings, by dotted key")
        try:
            saved = await store.save(name, cast(dict[str, JsonValue], settings), str(body.get("note") or ""))
        except ValueError as error:
            raise Taken(str(error)) from None
        return {"preset": _preset(saved)}

    async def delete_preset(self, name: str) -> dict[str, Any]:
        """Delete a preset (its versions stay readable by number). Raises `KeyError` for one there is none of."""
        store = presets_of(self._ledger)
        if store is None:
            raise KeyError(f"there is no preset {name!r}")
        gone = await store.delete(name)
        return {"deleted": gone.name}

    async def launch(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Ask for a run (`check`'s body), and start its job if nothing refuses it (`rollout_train.submitting
        .submit`): the launch, and the notes beside. Raises `rollout_train.launching.Refused` with the findings that
        refuse it, `Taken` where this monitor has no cluster config or the body says no name."""
        if self._cluster is None:
            raise Taken(NO_CLUSTER)
        settings, preset = await self._asked(body)
        if not settings["name"]:
            raise Taken("a launch says the run's name")
        findings = await findings_of(settings, self._cluster, self._ledger)
        if any(each.refuses for each in findings):
            raise LaunchRefused(findings)
        backend = self._backends.get(_backend_name(self._cluster)) if self._backends else None
        made = await submit(settings, self._cluster, self._ledger, preset=preset, backend=backend)
        return {"launch": asdict(made), "notes": [asdict(each) for each in findings if not each.refuses]}

    async def _version(self, reference: str) -> EnvironmentVersion | None:
        return await self._versions.get(reference) if self._versions is not None else None

    async def _loaded(self, environment: str) -> Environment:
        """An environment, by `module:name`, loaded in this process once. Raises `KeyError` where it does not load."""
        if environment not in self._environments:
            try:
                self._environments[environment] = await asyncio.to_thread(named, environment)
            except Exception as error:  # (whatever loading it raises: it is not here)
                raise KeyError(f"the environment {environment!r} does not load here ({error})") from None
        return self._environments[environment]

    async def environments(self) -> dict[str, Any]:
        """Every environment the system knows of, by `module:name` (`rollout_train.monitor.environments.listed`): those
        the cluster offers, those runs were started on and those suites' versions play; each with a readable
        `name`, the versions of it seen (in runs' starts and suites' entries), whether the cluster offers it
        (`offered`), its training runs and suites, and when a run last started on it (`used`)."""
        return {"environments": await asyncio.to_thread(listed, await self._environments_read())}

    async def environment(self, environment: str) -> dict[str, Any] | None:
        """An environment's page (`rollout_train.monitor.environments.page_of`): what it says of itself where it loads
        in this process (its version, rows, eval data, description and curriculum; else why it does not load) and what
        the ledger has of it (each row played, its runs, suites, evals and newest check). None where it neither loads
        nor is known."""
        if is_published(environment):  # (what it says of itself, as its check found it)
            version = await self._version(environment)
            said: dict[str, Any] | None = dict(version.description) if version is not None else None
            error = None if version is not None else f"there is no published environment {environment}"
        else:
            try:
                said, error = await asyncio.to_thread(described, await self._loaded(environment)), None
            except KeyError as failed:
                said, error = None, str(failed.args[0])
        return await asyncio.to_thread(page_of, await self._environments_read(), environment, said, error)

    async def environment_versions(self) -> dict[str, Any]:
        """Every published environment's version the ledger keeps (`rollout_train.published`), the newest imported
        first."""
        found = await self._versions.all() if self._versions is not None else []
        return {"versions": [each.to_json() for each in found], "importing": self._importer is not None}

    async def environment_version(self, reference: str) -> dict[str, Any] | None:
        """A published environment's version, by its id or `NAME@VERSION`; none where the ledger keeps no such one."""
        found = await self._version(reference)
        return found.to_json() if found is not None else None

    async def imports(self) -> dict[str, Any]:
        """The imports this monitor made since it started, newest first: each with what it imports, its stage
        (`fetching`, `reading`, `packing`, `storing`, `checking`, `recording`, then `done` or `refused`), when it began
        and ended, and the version it made or why it was refused."""
        return {"imports": [dict(each) for each in self._imports], "importing": self._importer is not None}

    async def import_environment(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Import an environment from git (`rollout_train.publishing.publish`): `url`, and optionally `ref`,
        `subdirectory` and `entry_point`. Returns the version (`version`) and whether it was there already
        (`existing`). Raises `Taken` where this monitor imports nothing or the body says no URL, `Refused` saying why
        the import cannot be made."""
        if self._importer is None or self._versions is None:
            raise Taken(
                "this monitor imports no environment: start it with a cluster config (`rollout monitor WHERE "
                "--cluster`) that names a Ray job server and a blob store, over a ledger that keeps published versions"
            )
        said = {key: str(body.get(key) or "").strip() for key in ("url", "ref", "subdirectory", "entry_point")}
        if not said["url"]:
            raise Taken("say the git URL to import from")
        source = Source(said["url"], said["ref"] or None, said["subdirectory"], said["entry_point"] or None)
        noted: dict[str, Any] = {**said, "id": f"import-{len(self._imports)}-{time.time_ns()}", "stage": "fetching"}
        noted |= {"started": round(time.time(), 1), "ended": None, "error": None, "version": None}
        self._imports[:] = [noted, *self._imports][:IMPORTS]

        def stage(name: str) -> None:
            noted["stage"] = name

        importer = self._importer
        try:
            made = await publish(
                source, versions=self._versions, blobs=importer.blobs, jobs=importer.jobs, scratch=importer.scratch,
                said=stage,
            )  # fmt: skip
        except Refused as refused:
            noted |= {"stage": "refused", "error": str(refused), "ended": round(time.time(), 1)}
            raise
        except Exception as error:  # (anything else is why it was refused too: said so, then raised)
            noted |= {"stage": "refused", "error": f"{type(error).__name__}: {error}", "ended": round(time.time(), 1)}
            raise
        noted |= {"stage": "done", "version": made.version.reference, "ended": round(time.time(), 1)}
        return {"version": made.version.to_json(), "existing": made.existing}

    async def _environments_read(self) -> Read:
        """What the environments' sources read: the tables, what the cluster offers, the registry's names, and
        the published versions."""
        published = await self._versions.all() if self._versions is not None else []
        offered = set(self._cluster.environments) if self._cluster is not None else set[str]()
        offered |= {each.reference for each in published} if self._cluster is not None else set[str]()
        return Read(await self._tables(), frozenset(offered), await self._names(), tuple(published))

    async def save_suite(self, name: str, body: Mapping[str, Any]) -> Suite:
        """Make a suite, or its next version, as the page's forms say it (`rollout_train.evals.make_suite`,
        `edit_suite`): its `entries`, each its `environment` (`module:name`), how its starts are `chosen` (`eval data`,
        of the name `eval_data`; `rows and seeds`, `rows` (none: every row) and `seeds`; `starts`, each a row (`task`)
        and its `seed`, drawn as the row's start with that seed unless it says its `parameters`; or, for an edit,
        `same`: those of the edited version's entry of that environment), the `episodes` of each start and the limits
        (`thinking_tokens`, `answer_tokens`; null for the channel's own); and, for an edit, the version it was made from
        (`base`, by number). A body with no `entries` is one entry. Raises `Taken` for what cannot be: a name that is no
        name, no entries, an environment that does not load here or is in two entries, eval data or a row an
        environment does not have, seeds that are no whole numbers, counts below 1, an edit made from another version
        than the newest, or one that changes nothing."""
        name = valid(name)
        current = await suite_of(self._ledger, name)
        listed = body.get("entries")
        if listed is None:  # (one entry: its environment, a new suite's; an edit's, the suite's one)
            sole = current.environments[0] if current is not None and len(current.entries) == 1 else ""
            listed = [{**body, "environment": body.get("environment") or sole}]
        if not isinstance(listed, list) or not listed:
            raise Taken("say the suite's entries: an environment, as module:name, and its starts, each")
        try:
            entries = [await self._entry(cast(dict[str, Any], each), current) for each in cast(list[Any], listed)]
            if current is None:
                return await make_suite(self._ledger, name, entries)
            base = body.get("base")
            return await edit_suite(
                self._ledger, name, entries, base=_whole(base, "base") if base is not None else None
            )
        except (KeyError, ValueError) as error:
            raise Taken(str(error.args[0]) if error.args else str(error)) from None

    async def _entry(self, body: Mapping[str, Any], current: Suite | None) -> SuiteEntry:
        """An entry as a suite's form says it (`save_suite`). Raises `Taken` for what it cannot be."""
        if not isinstance(body, dict):
            raise Taken("an entry says its environment, as module:name, and its starts")
        environment_name = str(body.get("environment") or "")
        if not environment_name:
            raise Taken("say each entry's environment, as module:name")
        try:
            environment = await self._loaded(environment_name)
        except KeyError as error:
            raise Taken(str(error.args[0])) from None
        chosen = str(body.get("chosen") or "")
        counts = {
            "episodes": _whole(body.get("episodes", 1), "episodes"),
            "thinking_tokens": _whole(body.get("thinking_tokens"), "thinking_tokens", optional=True),
            "answer_tokens": _whole(body.get("answer_tokens"), "answer_tokens", optional=True),
        }
        starts: dict[str, Any] = {}
        before = current.entry(environment_name) if current is not None else None
        if chosen == EVAL_DATA:
            starts["eval_data"] = str(body.get("eval_data") or "")
        elif chosen == DRAWN:
            rows, seeds = body.get("rows"), body.get("seeds")
            if rows is not None and not (
                isinstance(rows, list) and all(isinstance(each, str) for each in cast(list[Any], rows))
            ):
                raise Taken("rows are a list of row keys, or null for every row")
            if not isinstance(seeds, list) or not seeds:
                raise Taken("say the seeds, as a list of whole numbers")
            numbers = [_whole(each, "a seed", least=0) for each in cast(list[Any], seeds)]
            starts |= {"rows": cast(list[str], rows) or None, "seeds": numbers}
        elif chosen == GIVEN:
            starts["starts"] = await asyncio.to_thread(_given, environment, body.get("starts"))
        elif chosen == "same" and before is not None:
            starts["starts"] = before.starts
        elif chosen == "same":
            raise Taken(f"the suite has no entry of {environment_name} whose starts it keeps")
        else:
            raise Taken(f"say how its starts are chosen: {EVAL_DATA!r}, {DRAWN!r}, {GIVEN!r} (or, editing, 'same')")
        try:
            return await asyncio.to_thread(lambda: suite_entry(environment_name, environment, **starts, **counts))
        except (KeyError, ValueError) as error:
            raise Taken(str(error.args[0]) if error.args else str(error)) from None

    async def stop(self, id: str) -> Launch:
        """Ask a launch to stop (`rollout_train.submitting.stopped`): one whose job was not made yet stops at once; a
        job going is asked to stop, and its run stops at a group boundary. Raises `KeyError` when there is no such
        launch going."""
        launches = launches_of(self._ledger)
        if launches is None:
            raise KeyError(f"there is no launch {id} going")
        found = await launch_of(launches, id)
        return await stopped(found, launches, self._cluster, backends=self._backends)

    async def settings(self, run: str) -> dict[str, Any] | None:
        """A training run's settings (`rollout_train.settings`): its fixed ones and its changeable ones as its newest
        start says, what is wanted of them now, those its newest step used, and each step that used other settings than
        the one before, with what changed. None where there is no such run."""
        if not await asyncio.to_thread(present, self._ledger):
            return None
        starts: Any = await self._ledger.read(table(run, STARTS))
        if not starts:
            return None
        latest = newest_record(starts)
        said: Mapping[str, Any] = latest.get("run_settings") or latest.get("settings") or {}
        changeable: dict[str, Any] = dict(said.get("changeable") or {})
        steps: Any = await self._ledger.read(table(run, STEPS))
        changes: list[dict[str, Any]] = []
        before: Mapping[str, Any] = changeable
        for key in sorted(steps, key=int):
            used = steps[key].get("settings")
            if not isinstance(used, dict):
                continue
            used = cast(dict[str, Any], used)
            if differ := {name: value for name, value in used.items() if before.get(name) != value}:
                changes.append({"step": int(key), "changed": differ})
            before = used
        store = desired_settings_of(self._ledger)
        desired = await store.desired(run) if store is not None else None
        fixed: dict[str, Any] = dict(said.get("fixed") or {})
        if fixed.get("weights") is None and self._cluster is not None and "trainer.provider" in fixed:
            known = with_weights(RunSettings({"kind": str(latest.get("kind") or TRAIN), **fixed}), self._cluster)
            if known["weights"] is not None:  # (a start from before runs said what they train: its trainer's)
                fixed["weights"] = known["weights"]
        return {
            "run": run,
            "kind": str(latest.get("kind") or "run"),
            "environment": latest.get("environment"),
            "fixed": fixed,
            "changeable": changeable,
            "now": dict(before),
            "desired": dict(desired.settings) if desired else {},
            "changed": desired.changed if desired else None,
            "changes": changes,
        }

    async def want(self, run: str, settings: Mapping[str, Any]) -> Desired:
        """Want these of a run's changeable settings from its next step on. Raises `Taken` for a setting it does not
        have or cannot change, or a value it cannot take (a suite of another environment than the run's, say);
        `KeyError` where there is no such run, or nowhere to keep what is wanted. A suite the ledger does not have is
        taken: the run resolves it from its environment's eval data (`rollout_train.evals.suite_for`), or evaluates
        nothing."""
        found = await self.settings(run)
        store = desired_settings_of(self._ledger)
        if found is None or store is None:
            raise KeyError(f"there is no run {run}" if found is None else "this ledger keeps no settings")
        if not found["changeable"]:
            raise Taken("this run's start says no settings it can change")
        given: dict[str, JsonValue] = {}
        for key, value in settings.items():
            if key not in found["changeable"]:
                fixed = "is fixed" if key in found["fixed"] else "is not one of its settings"
                raise Taken(f"{key} {fixed}: these can change ({', '.join(found['changeable'])})")
            if key.startswith(TRAINER) and not (value is None or isinstance(value, str | int | float | bool)):
                raise Taken(f"{key} is a number, true or false, or text (not {value!r})")
            try:
                given[key] = checked_setting(key, value)
            except ValueError as error:
                raise Taken(str(error)) from None
            if key == EVALS_SUITE and given[key]:  # (one not made yet is the environment's eval data of the name)
                suite = await suite_of(self._ledger, str(given[key]))
                if suite is None and parsed(str(given[key]))[1] not in (None, 1):
                    raise Taken(f"there is no version {given[key]!r} of a suite")
        return await store.want(run, given)

    async def checkpoint_evals(self, checkpoint: str) -> dict[str, Any] | None:
        """Every eval a checkpoint (by its id or the start of it) has had, by hand or by a schedule, newest first
        (`rollout_train.monitor.scores.evals_of`); None where there is no such checkpoint."""
        found = await self._checkpoint(checkpoint)
        if found is None:
            return None
        tables, called = await self._tables(), await self._names()
        return {"checkpoint": found, "evals": await asyncio.to_thread(evals_of, tables, found, called)}

    async def path(self, checkpoint: str) -> dict[str, Any] | None:
        """A checkpoint's line from the base model, with each point's scores at each suite
        (`rollout_train.monitor.scores.path_of`); None where there is no such checkpoint."""
        found = await self._checkpoint(checkpoint)
        if found is None:
            return None
        tables, called = await self._tables(), await self._names()
        made = {each.id: each for each in await self._checkpoints()}
        return await asyncio.to_thread(path_of, tables, made, found, called)

    async def eval_subjects(self) -> dict[str, Any]:
        """Every subject that has had an eval, the one evaluated last first
        (`rollout_train.monitor.scores.subjects_in`)."""
        tables, called = await self._tables(), await self._names()
        made = {each.id: each for each in await self._checkpoints()}
        return {"subjects": await asyncio.to_thread(subjects_in, tables, made, called)}

    async def history(self, kind: str, reference: str) -> dict[str, Any] | None:
        """A subject's history: every eval a checkpoint (by its id or the start of it) or a base model (by name) has had
        (`rollout_train.monitor.scores.history_of`); None where there is no such subject."""
        tables, called = await self._tables(), await self._names()
        made = {each.id: each for each in await self._checkpoints()}
        found = (await self._checkpoint(reference) or reference) if kind == CHECKPOINT else reference
        return await asyncio.to_thread(history_of, tables, made, kind, found, called)

    async def _checkpoint(self, reference: str) -> str | None:
        """A checkpoint's id, by the id or the start of one that no other begins with."""
        if not await asyncio.to_thread(present, self._ledger):
            return None
        ids = [each.id for each in await self._checkpoints()]
        if reference in ids:
            return reference
        starting = [each for each in ids if each.startswith(reference)]
        return starting[0] if len(starting) == 1 else None

    def _registry(self) -> Registry:
        registry = registry_of(self._ledger)
        if registry is None:
            raise KeyError("this ledger has no registry")
        return registry

    async def evals(self) -> dict[str, Any]:
        """Every suite (the version its name points to, with its environment and starts; every version; and each
        subject that played it, with the version it played and how it did at each start) and every eval (its suite, the
        version it played, its checkpoint, how far it has got), newest first (`rollout_train.monitor.scores`)."""
        tables = await self._tables()
        called = await self._names()
        suites = suites_in(tables, called)
        for each in suites:
            current = next(version for version in each["versions"] if version["id"] == each["version"])
            each |= {"environments": current["environments"], "made": current["made"]}
        evals: list[dict[str, Any]] = [
            {key: each[key] for key in ("run", "name", "suite", "version", "checkpoint", "started", "played")}
            | {key: each[key] for key in ("expected", "solved", "done")}
            | {  # (each environment's, for an eval of several)
                "entries": [
                    {key: entry[key] for key in ("environment", "played", "solved", "share", "reward")}
                    for entry in each["entries"]
                ]
                if len(each["entries"]) > 1
                else []
            }
            for each in evals_in(tables, called)
        ]
        evals.sort(key=lambda each: -(each["started"] or 0.0))
        return {"suites": suites, "evals": evals}

    async def lineage(self) -> dict[str, Any]:
        """The checkpoints as a graph, with what trains and serves them (`rollout_train.monitor.lineage`), from every
        base model that has history and every one the cluster offers."""
        tables = await self._tables()
        notes = _noted(await self._beats())
        every = [note for each in notes.values() for note in each]
        called = await self._names()
        return await asyncio.to_thread(lineage, tables, every, names=called, offered=self.offered_models())

    def offered_models(self) -> list[str]:
        """The base models the cluster offers, each once: its inference providers' models and its trainers'."""
        if self._cluster is None:
            return []
        found = [model for each in self._cluster.inference.values() for model in each.models]
        found += [model for each in self._cluster.trainers.values() for model in each.models]
        return list(dict.fromkeys(found))

    async def statistics(self) -> dict[str, Any]:
        """Every run of the ledger in figures (`rollout_train.monitor.statistics`), with each run's engines'
        throughput from its runners' heartbeats, and what the runs are called."""
        tables = await self._tables()
        beats = await self._beats()
        figures = await asyncio.to_thread(statistics, tables, _noted(beats))
        called = await self._names()
        return {**figures, "names": {"runs": called["runs"]}}

    async def machines(self) -> dict[str, Any]:
        """Every machine that beats and the roles on it, as the heartbeats and the ledger say
        (`rollout_train.monitor.machines`): the runners and the episodes their claims hold, the sandbox pools and
        their leases, the engine hosts and how far behind what their run wants each engine is, the drivers that wait
        for what they asked Ray for, and the gateways."""
        beats = await self._beats()
        if not await asyncio.to_thread(present, self._ledger):
            return machines(beats, now=time.time())
        fences = await self._ledger.fences()
        claims = {run: await Claims.read(self._ledger, run) for run in await runs_in(self._ledger)}
        held = leases_of(self._ledger)
        followed = {str(beat.about.get("follows")) for beat in beats if kind_of(beat) == ENGINES}
        serving = {run: await self._ledger.read(table(run, SERVING)) for run in followed}
        return machines(
            beats, now=time.time(), claims=claims, fences=fences, leases=await held.all() if held else [],
            serving=serving,
        )  # fmt: skip

    @contextlib.asynccontextmanager
    async def one_reading(self) -> AsyncGenerator[None]:
        """Within the block, the ledger's tables, the registry's names, the checkpoints and the beats are read once,
        whatever reads them (the hub reads every topic it watches so, once a beat)."""
        self._reading = {}
        try:
            yield
        finally:
            self._reading = None

    def read_afresh(self) -> None:
        """Read the ledger again within a reading (after the monitor itself changed something)."""
        if self._reading is not None:
            self._reading.clear()

    async def _once(self, what: str, read: Callable[[], Awaitable[Any]]) -> Any:
        """What `read` reads, read once within a reading (`one_reading`), however many ask at the same time."""
        if self._reading is None:
            return await read()
        if what not in self._reading:
            self._reading[what] = asyncio.ensure_future(read())
        return await self._reading[what]

    async def _beats(self) -> list[Beat]:
        """Every runner's newest heartbeat (none where there is no ledger: reading makes none)."""

        async def read() -> list[Beat]:
            presence = presence_of(self._ledger)
            if presence is None or not await asyncio.to_thread(present, self._ledger):
                return []
            return await presence.beats()

        return cast(list[Beat], await self._once("beats", read))

    async def _tables(self) -> dict[str, dict[str, JsonValue]]:
        """Every table of the ledger, by name (none where there is no ledger: reading makes none), but the gateway's
        tables of turns (`runs/RUN/turns/RUN_ID`), a row per turn, which nothing here reads."""

        async def read() -> dict[str, dict[str, JsonValue]]:
            if not await asyncio.to_thread(present, self._ledger):
                return {}
            return await self._ledger.read_all(leaving_out="/turns/")

        return cast(dict[str, dict[str, JsonValue]], await self._once("tables", read))

    async def _names(self) -> dict[str, Any]:
        """What the registry names (`rollout_train.registry.names`)."""
        return cast(dict[str, Any], await self._once("names", lambda: names(registry_of(self._ledger))))

    async def _checkpoints(self) -> list[Checkpoint]:
        """Every checkpoint, oldest first."""
        return cast(list[Checkpoint], await self._once("checkpoints", lambda: checkpoints_in(self._ledger)))

    async def group(self, run: str, number: int, relayed: bool = False) -> dict[str, Any] | None:
        """One group: what was decided (the row and its start), its stage, its episodes with what each reported,
        its step and the checkpoint it made, and its outcome."""
        if not await asyncio.to_thread(present, self._ledger):
            return None
        tables = {name: await self._ledger.read(table(run, name)) for name in (*RUN_TABLES, STARTS)}
        record: Any = tables[GROUPS].get(str(number))
        if record is None:
            return None
        found = await asyncio.to_thread(self._source, run, tables[STARTS], relayed)
        if isinstance(found, _Remote) and (answer := await asyncio.to_thread(found.group, run, number)) is not None:
            return answer | {"episodes_at": found.address}
        checkpoints = {checkpoint.id: checkpoint for checkpoint in await self._checkpoints()}
        fences = await self._ledger.fences()
        return await asyncio.to_thread(self._group, run, str(number), record, tables, fences, checkpoints, found)

    def _group(
        self,
        run: str,
        number: str,
        record: Mapping[str, Any],
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        checkpoints: Mapping[str, Checkpoint],
        found: "_Place | _Remote | None",
    ) -> dict[str, Any]:
        place = found if isinstance(found, _Place) else None
        played = _Played(run, tables, fences, self._records)
        group = _group(number, record, tables, checkpoints, played, place.feed.runs() if place else [])
        step: Any = group["step"]
        made = checkpoints.get(str(step.get("makes"))) if step else None
        result: Any = tables[RESULTS].get(number)
        unsaid = number in unreported(tables[EPISODES])
        return {
            **group,
            "run": run,
            "episodes_at": "here" if place else found.address if isinstance(found, _Remote) else None,
            "parameters": record.get("parameters"),
            "outcome": _done(number, result, record, step, made, group["error"], unsaid)
            if group["stage"] == DONE and result
            else None,
            "result": _outcome(number, record, result, unsaid) if result else None,
            "checkpoint": _checkpoint(made, short(checkpoints)) if made else None,
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
        replies and tool calls from the events its runner kept; where neither has a sample, as a harness's are where a
        gateway elsewhere recorded them, the replies of the turns the gateway recorded), from which its rollouts (one
        per model slot) are drawn; what it reported when it ended; and where it sits: its run, its group and its
        labels. An episode of a run on another machine is asked of the monitor there."""
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
        run = known[0] if known is not None else labels.get("run")
        unsampled = (
            (summary or {}).get("samples") == 0
            if source == "feed"
            else source != "archive" or not any(line["kind"] == "sample" for line in self._archive.get(run_id, []))
        )
        turns = await self._turns(store, str(run), run_id) if run and store is not None and unsampled else []
        if turns:  # (a harness's samples, which a gateway elsewhere recorded and the feed never had)
            source, lines = "turns", turns[after:]
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
        try:
            where = await where_blobs_are(self._ledger, run)
        except ValueError:
            return None
        key = json.dumps(where, sort_keys=True)
        if key not in self._stores:
            self._stores[key] = await asyncio.to_thread(opened, where)
        return self._stores[key]

    async def _turns(self, store: Blobs, run: str, run_id: str) -> list[dict[str, Any]]:
        """A run's samples as the gateway recorded them, in the order it recorded them, as the feed would have them:
        each turn's reply (what it was sent is kept only as tokens, so a sample has no messages)."""
        turns = self._turn_stores.setdefault(id(store), TurnStore(self._ledger, store, cached=1024))
        index = await turns.index(run, run_id)
        lines: list[dict[str, Any]] = []
        for effect, entry in index.items():
            reply = await turns.reply(run, run_id, effect, index)
            if reply is None:
                continue
            at: Any = entry.get("at") if isinstance(entry, dict) else None
            seconds: Any = reply.timings.get("seconds") or 0.0
            lines.append(
                {
                    "kind": "sample",
                    "slot": reply.slot,
                    "effect_id": effect,
                    "at": at,
                    "seconds": round(float(seconds), 2),
                    "messages": [],
                    "tools": [],
                    "reply": plain(reply.result.message),
                    "finish_reason": reply.result.finish_reason.value,
                    "checkpoint": reply.checkpoint,
                    "depth": reply.depth,
                }
            )
        return lines

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
        checkpoints: list[Checkpoint],
        called: Mapping[str, Any],
        beats: list[Beat],
        relayed: bool,
    ) -> dict[str, Any]:
        noted = _noted(beats)
        beaten: dict[str, float] = {}  # each run's runners' newest beat
        ages: dict[str, float] = {}  # and how old it is, by the clock of the store that keeps them
        held = {str(run) for beat in beats if alive(beat) for run in cast(list[Any], beat.about.get("paused") or [])}
        for beat in beats:
            if beat.about.get("run"):
                beaten[str(beat.about["run"])] = max(beaten.get(str(beat.about["run"]), 0.0), beat.at)
                ages[str(beat.about["run"])] = min(ages.get(str(beat.about["run"]), beat.age), beat.age)
        made = {checkpoint.id: checkpoint for checkpoint in checkpoints}
        shorter = short(made)
        blobs = {digest: size for checkpoint in checkpoints for digest, size in _blobs(checkpoint)}  # (each kept once)
        marks: dict[str, list[str]] = {}
        for mark, checkpoint in called["bookmarks"].items():
            marks.setdefault(str(checkpoint), []).append(mark)
        now = time.time()
        runs: list[dict[str, Any]] = []
        played: dict[str, _Played] = {}
        finished: set[str] = set()
        """The evals that played every start."""
        for run in named_runs(tables):
            starts: Any = tables.get(table(run, STARTS), {})
            found = self._source(run, starts, relayed)
            place = found if isinstance(found, _Place) else None
            own = {name: tables.get(table(run, name), {}) for name in RUN_TABLES}
            played[run] = _Played(run, own, fences, self._records)
            listed = _run(run, own, fences.get(run_scope(run)), made, played[run], place.feed.runs() if place else [])
            seen = self._read(
                run, starts, found, listed["wrote"], now, noted.get(run, []), beaten.get(run), ages.get(run)
            )
            seen |= _how_it_ended(seen["state"], starts, own[ENDS], beaten.get(run) is not None)
            if seen["state"] in (RUNNING, IDLE) and run in held:  # (its process is there, and a runner says it paused)
                seen["state"] = PAUSED
            begun = newest_record(starts)
            kind = str(begun.get("kind") or "run")
            if kind == EVAL and own[GROUPS] and set(own[GROUPS]) <= set(own[RESULTS]) and seen["state"] not in ENDINGS:
                seen |= {"state": FINISHED, "channels": []}  # (an eval that played every start has ended)
                finished.add(run)
            runs.append(
                listed
                | seen
                | {"played": played[run].counts(), "name": called["runs"].get(run, run), "kind": kind}
                | {"by": begun.get("by"), "by_step": begun.get("step"), "part_of": begun.get("part_of")}
            )
        states = {run["run"]: run["state"] for run in runs}
        for run in runs:  # (an eval a training run's schedule asked for, not done, is played by that run's runner)
            if run["kind"] == EVAL and run["run"] not in finished and run["by"] in states and run["run"] not in beaten:
                by = states[run["by"]]
                run["state"] = PAUSED if by in (RUNNING, IDLE) and run["run"] in held else by
        rank = {RUNNING: 0, PAUSED: 1, IDLE: 1}
        runs.sort(key=lambda run: (rank.get(run["state"], 2), -(run["written"] or 0.0), run["run"]))
        return {
            "at": round(now, 1),
            "name": self.directory.name if self.directory else self.ledger,
            "directory": str(self.directory) if self.directory else None,
            "ledger_at": self.ledger,
            "host": self.host,
            "written": newest(run["written"] for run in runs),
            "runs": runs,
            "checkpoints": [
                _checkpoint(checkpoint, shorter) | {"bookmarks": sorted(marks.get(checkpoint.id, []))}
                for checkpoint in checkpoints
            ],
            "bookmarks": dict(called["bookmarks"]),
            "runners": _runners(fences, played),
            "channels": [channel | {"run": run} for run, notes in noted.items() for channel in _channels(notes)],
            "ledger": {
                "fences": {scope: number for scope, number in fences.items() if not of_episode(scope)},
                "tables": {name: len(records) for name, records in tables.items()},
            },
            "kept": {
                "checkpoints": sum(blobs.values()),
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
        age: float | None,
    ) -> dict[str, Any]:
        """What a run's start says (where it is, what started it), its engines (as its runners' heartbeats say),
        and what its episodes' place adds (its groups in flight with their episodes); when it last wrote or beat
        (`beaten`, and how long ago by the beats' store, `age`), and so whether it is running."""
        latest = newest_record(starts)
        beating = age is not None and age <= STALE
        added: dict[str, Any] = {"channels": _channels(notes) if beating else []}  # (what a gone process served is not)
        written = newest([wrote, latest.get("started"), beaten])
        if isinstance(found, _Place):
            written = newest([written, found.written()])
        elif isinstance(found, _Remote) and (there := found.run(run)) is not None:
            added |= {key: there[key] for key in ("open", "done") if key in there}
            written = newest([written, there.get("written")])
        quiet = now - written if written is not None else float("inf")
        if beaten is not None:  # (its runners beat: whether its process is there is known)
            state = (RUNNING if quiet < QUIET else IDLE) if beating else GONE
        else:
            began = latest.get("started")
            state = RUNNING if isinstance(began, int | float) and now - began <= STALE else GONE
        return {
            **added,
            "state": state,
            "host": latest.get("host"),
            "address": latest.get("address"),
            "directory": latest.get("directory") or (str(found.directory) if isinstance(found, _Place) else None),
            "episodes_at": "here" if isinstance(found, _Place) else found.address if found else None,
            "reached": found.reached if isinstance(found, _Remote) else None,
            **{key: latest.get(key) for key in ("environment", "version", "description")},  # (what its results say)
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
    checkpoints: Mapping[str, Checkpoint],
    played: "_Played",
    in_feed: list[dict[str, Any]],
) -> dict[str, Any]:
    """A run: its groups that are not done with, each with its stage, its episodes and its step; the ones that are,
    each with its result and what was done with it; and its steps, each with the groups that went into it. A group
    that gave nothing to train on is listed with the first step decided after it, and until there is one, with
    those the next step will cover (`next`)."""
    groups: Any = tables[GROUPS]
    entries = [
        _group(number, groups[number], tables, checkpoints, played, in_feed) for number in sorted(groups, key=int)
    ]
    done: list[dict[str, Any]] = []
    unsaid = unreported(tables[EPISODES])
    for entry in entries:
        if entry["stage"] == DONE:
            step: Any = entry["step"]
            made = checkpoints.get(str(step.get("makes"))) if step else None
            key = str(entry["number"])
            line = _done(key, tables[RESULTS][key], groups[key], step, made, entry["error"], key in unsaid)
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
            "state": _state(key, intent, failures, checkpoints),
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
            + [checkpoints[step["makes"]].made for step in listed if step["makes"] in checkpoints]
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
    checkpoints: Mapping[str, Checkpoint],
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
                "reward": sum(each["rewards"].values()) / len(each["rewards"]) if each["rewards"] else None,
                "rewards": each["rewards"],
                "updated": each["updated"],
                "interrupted": each["state"] == Outcome.CANCELLED.value and each["run_id"] not in ended,
            }
    for run_id, each in ended.items():
        shown = {name: value for name, value in each.items() if name not in ("events", "kept")}
        episodes.setdefault(run_id, {"samples": None, "updated": None}).update(shown)
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
            "state": _state(str(key), intent, failures, checkpoints),
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
    number: str,
    result: Any,
    group: Mapping[str, Any],
    step: Any,
    made: Checkpoint | None,
    error: str | None,
    unsaid: bool = False,
) -> dict[str, Any]:
    """A group that is done with, as the page shows it: its result, and what was done with it (the checkpoint its step
    made and the trainer's statistics, or why the step failed), with how long it all took."""
    line = _outcome(number, group, result, unsaid)
    ended = made.made if made else float(result.get("time") or 0.0)
    began = float(group.get("decided") or result.get("time") or 0.0)
    return {
        **line,
        "adapter": made.id if made else None,
        "step": step.get("step") if step else None,
        "step_state": step.get("state") if step else None,
        "depth": made.depth if made else None,
        "update": dict(made.metrics) if made else None,
        "segments_trained": int(step.get("segments") or 0) if made and step else 0,
        "error": error,
        "seconds": round(ended - began, 1) if began else None,
    }


def _state(
    key: str, step: Mapping[str, Any], failures: Mapping[str, Any], checkpoints: Mapping[str, Checkpoint]
) -> str:
    """Where a step stands: failed, committed (its checkpoint made), or stepping."""
    return FAILED if key in failures else COMMITTED if _makes(step) in checkpoints else STEPPING


def _covers(step: Mapping[str, Any]) -> list[int]:
    """The groups a step covers, by their numbers."""
    listed: list[Any] = step.get("groups") or []
    return [int(group) for group in listed]


def _makes(step: Mapping[str, Any]) -> str | None:
    """The checkpoint a step's decision names, by id."""
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


def _outcome(number: str, group: Mapping[str, Any], line: Mapping[str, Any], unsaid: bool = False) -> dict[str, Any]:
    """A group's result as it is read (with what its record says), its failures said once each and briefly; `solved`
    is null for each where none of its episodes said whether it solved its task (`unsaid`)."""
    joined = asdict(Result.from_json(line, int(number), group))
    failures = list(dict.fromkeys(str(failure)[:300] for failure in joined["failures"]))
    return {**joined, "failures": failures, "solved": solved_of(joined["solved"], unsaid)}


def _preset(preset: Preset) -> dict[str, Any]:
    return {"name": preset.name, "version": preset.version, "id": preset.id, "settings": dict(preset.settings),
            "note": preset.note, "saved": preset.saved}  # fmt: skip


def _backend_name(cluster: "Cluster") -> str:
    """Where a cluster's runs' jobs go, by name (`rollout_train.submitting.backend_of`)."""
    return "kubernetes" if cluster.kubernetes is not None else "ray"


def _whole(value: Any, what: str, *, optional: bool = False, least: int = 1) -> Any:
    """A whole number a form gave, `least` at least (none, where it is `optional`); raises `Taken` otherwise."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value or value < least:
        raise Taken(f"{what} is a whole number, {least} at least (not {value!r})")
    return int(value)


def _given(environment: Environment, listed: Any) -> list[Start]:
    """Starts a form gave, each a row (`task`) and a `seed`: the row's start drawn with that seed (`drawn`), unless it
    says its `parameters`. Raises `Taken` for a row the environment does not have, or no starts."""
    if not isinstance(listed, list) or not listed:
        raise Taken("say the starts, each a row (task) and a seed")
    rows = {row.key: row for row in environment.rows()}
    made: list[Start] = []
    for each in cast(list[Any], listed):
        given = cast(dict[str, Any], each) if isinstance(each, dict) else {"task": each}
        if given.get("task") not in rows:
            raise Taken(f"the environment has no row {given.get('task')!r}")
        row, seed = rows[str(given["task"])], _whole(given.get("seed"), "a seed", least=0)
        parameters = given["parameters"] if "parameters" in given else environment.start(row, random.Random(seed))
        made.append(Start(row.key, row.title, seed, parameters))
    return made


def _run_in(directory: Path) -> str | None:
    """The id of the run in a directory, as its `run.json` says (none: it holds no run)."""
    path = directory / RUN
    return str(json.loads(path.read_text())["id"]) if path.exists() else None


def _checkpoint(checkpoint: Checkpoint, shorter: Mapping[str, str]) -> dict[str, Any]:
    """A checkpoint as the page shows it: where it came from, what it was trained on, and what is kept of it."""

    def size(manifest: Manifest | None) -> int:
        return sum(blob.size for blob in manifest.files.values()) if manifest else 0

    return {
        "id": checkpoint.id,
        "short": shorter.get(checkpoint.id, checkpoint.id),
        "depth": checkpoint.depth,
        "parents": list(checkpoint.parents),
        "base": checkpoint.base,
        "kind": checkpoint.kind,
        "run": checkpoint.run,
        "step": checkpoint.step,
        "made": checkpoint.made,
        "metrics": dict(checkpoint.metrics),
        "weights": {"files": len(checkpoint.weights.files), "bytes": size(checkpoint.weights)}
        if checkpoint.weights
        else None,
        "state": {"files": len(checkpoint.state.files), "bytes": size(checkpoint.state)} if checkpoint.state else None,
        "released": checkpoint.released,
        "dataset": checkpoint.dataset,
    }


def _blobs(checkpoint: Checkpoint) -> list[tuple[str, int]]:
    return [(blob.sha256, blob.size) for manifest in (checkpoint.weights, checkpoint.state) if manifest
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
        made = cast(Mapping[str, Mapping[str, Any]], tables.get(CLAIMS, {}))
        claims = Claims(made, set(cut), set(done), set(tables.get(ADOPTED, {})))
        for key, line in made.items():
            claim: Any = line
            group, episode, attempt = key.split("/")
            holds = f"{group}/{episode}" not in done and claims.holds(key, fences, None)
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
        "solved": episode.solved if "solved" in episode.info else None,  # (None: the task did not say)
        "sampled": sum(record.sampled.values()),
        "labels": dict(episode.labels),
        "slots": sorted(episode.trajectories),
        "untrained": sorted(slot for slot, trajectory in episode.trajectories.items() if not trajectory.trained),
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
                counts = {key: value for key, value in channel.items() if key not in ("channel", "adapter", "servers")}
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


def _how_it_ended(state: str, starts: Mapping[str, Any], ends: Mapping[str, Any], beat: bool) -> dict[str, Any]:
    """A run's state as how its newest start ended says, if it said (with what it said, `ending`); else lost, if its
    runners beat once and no longer do; else as its writes and beats say."""
    newest = max(starts, key=int) if starts else None
    said: Any = ends.get(newest) if newest is not None else None
    ending = cast(dict[str, Any], said) if isinstance(said, dict) else {}
    if ending.get("how") in ENDINGS:
        return {"state": str(ending["how"]), "ending": ending, "channels": []}
    if state == GONE and beat:
        return {"state": LOST}
    return {}
