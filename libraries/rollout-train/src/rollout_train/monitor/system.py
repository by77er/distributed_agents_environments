"""Where a run stands, read from its directory: what the monitor's system view shows.

A run's directory, as an open profile lays it out (`rollout_train.layout`), holds everything the run knows: the
ledger (what the training loop decided and what happened, the policies' versions, the fences), the jobs' logs (what
was asked for, the episodes that ended, what was acknowledged) and the feed (what is happening now). `System` reads
them and says where everything stands. It asks nothing of the run's process: it says the same whether that process
is alive or not, and what it says of a group is what a loop that started now would find.
"""

import asyncio
import json
import lzma
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout.contracts import BlobReference, Message, RunEvent, RunEventType
from rollout.harness.blobs import FileBlobStore
from rollout_train.layout import BLOBS, JOBS, LEDGER, PROCESSES
from rollout_train.ledger import FileLedger
from rollout_train.monitor.feed import Appended, FeedReader, plain
from rollout_train.policies import Manifest, Version, named, parsed, policies_in, versions_in
from rollout_train.policies import scope as policy_scope
from rollout_train.record import FAILURES, GROUPS, RESULTS, STEPS, runs_in, table
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


class System:
    def __init__(self, directory: Path, feed: FeedReader) -> None:
        self.directory = directory
        self.feed = feed
        self.machine = Machine(directory)
        self._ledger = FileLedger(directory / LEDGER)
        self._jobs: dict[str, _JobLog] = {}
        self._reading = threading.Lock()
        """Held while the jobs' logs are read: requests are answered in threads, and two must not read at once."""
        self._blobs = FileBlobStore(directory / BLOBS)
        self._archive: dict[str, list[dict[str, Any]]] = {}
        """Episodes read back from their events, the newest few."""

    async def snapshot(self) -> dict[str, Any]:
        """Where everything stands now: the runs' groups that are not done with and the ones that are, the
        policies' versions, the jobs, what each channel serves and how fast, the machine, and what is kept."""
        tables: dict[str, dict[str, JsonValue]] = {}
        fences: dict[str, int] = {}
        runs: list[str] = []
        policies: list[dict[str, Any]] = []
        versions: list[Version] = []
        if await asyncio.to_thread(self._ledger.directory.exists):  # (a reader makes no ledger where none is)
            fences = await self._ledger.fences()
            tables = {name: await self._ledger.read(name) for name in await self._ledger.tables()}
            runs = await runs_in(self._ledger)
            for policy in await policies_in(self._ledger):
                kept = await versions_in(self._ledger, policy)
                versions += kept
                policies.append(_policy(policy, kept, fences.get(policy_scope(policy))))
        return await asyncio.to_thread(self._assembled, tables, fences, runs, policies, versions)

    async def group(self, run: str, number: int) -> dict[str, Any] | None:
        """One group: what was decided (the row and its start), its stage, its episodes with what each reported,
        its step and the version it made, and its outcome."""
        if not await asyncio.to_thread(self._ledger.directory.exists):
            return None
        tables = {name: await self._ledger.read(table(run, name)) for name in RUN_TABLES}
        record: Any = tables[GROUPS].get(str(number))
        if record is None:
            return None
        versions = {
            version.name: version
            for policy in await policies_in(self._ledger)
            for version in await versions_in(self._ledger, policy)
        }
        return await asyncio.to_thread(self._group, run, str(number), record, tables, versions)

    def _group(
        self,
        run: str,
        number: str,
        record: Mapping[str, Any],
        tables: Mapping[str, Mapping[str, JsonValue]],
        versions: Mapping[str, Version],
    ) -> dict[str, Any]:
        jobs = self._job_logs()
        group = _group(number, record, tables, versions, jobs.get(run), self.feed.runs())
        step: Any = group["step"]
        made = versions.get(str(step.get("makes"))) if step else None
        result: Any = tables[RESULTS].get(number)
        return {
            **group,
            "run": run,
            "parameters": record.get("parameters"),
            "outcome": _done(result, record, step, made, group["error"]) if group["stage"] == DONE and result else None,
            "result": _outcome(result) if result else None,
            "version": _policy(made.policy, [made], None)["versions"][0] if made else None,
        }

    async def episode(self, run_id: str, after: int = 0) -> dict[str, Any]:
        """One episode: the run's lines from index `after` on (from the feed, or, once the feed has let it go, its
        replies and tool calls from the events the job kept), from which its rollouts (one per model slot) are
        drawn; what it reported when it ended; and where it sits: its job, its group and its labels."""
        ended, summary = await asyncio.to_thread(self._found, run_id)
        if summary is not None:
            source, lines = "feed", await asyncio.to_thread(self.feed.lines, run_id, after)
        elif ended is not None and ended.get("events"):
            source, lines = "archive", (await self._archived(run_id, ended["events"]))[after:]
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

    def _found(self, run_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """An episode in the jobs' logs (once it has ended) and in the feed (while the feed keeps it)."""
        jobs = self._job_logs()
        ended = next((each for job in jobs.values() for each in job.episodes if each["run_id"] == run_id), None)
        return ended, next((run for run in self.feed.runs() if run["run_id"] == run_id), None)

    async def _archived(self, run_id: str, events: Mapping[str, Any]) -> list[dict[str, Any]]:
        if run_id not in self._archive:
            data = await self._blobs.read(BlobReference.model_validate(events))
            lines = (await asyncio.to_thread(lzma.decompress, data)).decode().splitlines()
            self._archive[run_id] = _replayed([RunEvent.model_validate_json(line) for line in lines])
            while len(self._archive) > ARCHIVED:
                del self._archive[next(iter(self._archive))]
        return self._archive[run_id]

    def _assembled(
        self,
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        runs: list[str],
        policies: list[dict[str, Any]],
        versions: list[Version],
    ) -> dict[str, Any]:
        made = {version.name: version for version in versions}
        blobs = {digest: size for policy in policies for digest, size in policy.pop("blobs")}  # (each kept once)
        in_feed = self.feed.runs()
        jobs = self._job_logs()
        return {
            "at": round(time.time(), 1),
            "name": self.directory.name,
            "directory": str(self.directory),
            "written": self._written(in_feed),
            "processes": _processes(self.directory / PROCESSES),
            "runs": [
                _run(
                    run,
                    {name: tables.get(table(run, name), {}) for name in RUN_TABLES},
                    fences.get(run_scope(run)),
                    made,
                    jobs.get(run),
                    in_feed,
                )
                for run in runs
            ],
            "policies": policies,
            "jobs": [job.counts() for job in jobs.values()],
            "channels": _channels(self.feed.job()),
            "ledger": {"fences": dict(fences), "tables": {name: len(records) for name, records in tables.items()}},
            "machine": self.machine.shown(),
            "kept": {
                "versions": sum(blobs.values()),
                "episodes": sum(job.kept for job in jobs.values()),
            },
        }

    def _job_logs(self) -> dict[str, "_JobLog"]:
        with self._reading:
            return self._refreshed()

    def _refreshed(self) -> dict[str, "_JobLog"]:
        directory = self.directory / JOBS
        for path in sorted(directory.iterdir()) if directory.is_dir() else []:
            if path.is_dir() and path.name not in self._jobs:
                self._jobs[path.name] = _JobLog(path)
        for job in self._jobs.values():
            job.refresh()
        return self._jobs

    def _written(self, in_feed: list[dict[str, Any]]) -> float | None:
        """When the run last wrote anything this reads."""
        times = [path.stat().st_mtime for path in self._ledger.directory.rglob("*.jsonl")]
        times += [job.written for job in self._jobs.values()]
        times += [run["updated"] for run in in_feed]
        return round(max(times), 1) if times else None


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
            line = _done(
                tables[RESULTS][str(entry["number"])], groups[str(entry["number"])], step, made, entry["error"]
            )
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


def _done(result: Any, group: Mapping[str, Any], step: Any, made: Version | None, error: str | None) -> dict[str, Any]:
    """A group that is done with, as the page shows it: its result, and what was done with it (the version its step
    made and the trainer's statistics, or why the step failed), with how long it all took."""
    line = _outcome(result)
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


def _outcome(line: Mapping[str, Any]) -> dict[str, Any]:
    """A group's outcome as it is logged, with its failures said once each and briefly."""
    said: list[Any] = line.get("failures") or []
    failures = list(dict.fromkeys(str(failure)[:300] for failure in said))
    return {**line, "failures": failures}


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
                "weights": {"files": len(version.weights.files), "bytes": size(version.weights)},
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

    def alive(pid: int) -> bool:
        return Path(f"/proc/{pid}").exists()

    started = [{"pid": int(pid), "name": str(name), "alive": alive(int(pid))} for pid, name in processes.items()]
    return {"owner": owner, "alive": alive(owner), "started": started}


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
        """The ticket a group was asked for under, by the group's number (the label `iteration`)."""
        asked = [ticket for ticket in self.tickets if ticket["labels"].get("iteration") == number]
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
