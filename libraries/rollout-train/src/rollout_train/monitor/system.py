"""Where a run stands, read from its directory: what the monitor's system view shows.

A run's directory, as an open profile lays it out (`rollout_train.layout`), holds everything the run knows: the
ledger (what the training loop decided and what happened, the policies' versions, the fences), the jobs' logs (what
was asked for, the episodes that ended, what was acknowledged) and the feed (what is happening now). `System` reads
them and says where everything stands. It asks nothing of the run's process: it says the same whether that process
is alive or not, and what it says of a group is what a loop that started now would find.
"""

import asyncio
import json
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout_train.layout import JOBS, LEDGER, PROCESSES
from rollout_train.ledger import FileLedger
from rollout_train.monitor.feed import Appended, FeedReader
from rollout_train.policies import Manifest, Version, named, parsed, policies_in, versions_in
from rollout_train.policies import scope as policy_scope
from rollout_train.record import GROUPS, ITERATIONS, STEPS, runs_in, table
from rollout_train.record import scope as run_scope
from rollout_train.rollouts.episodes import Record
from rollout_train.rollouts.jobs import ACKNOWLEDGED, EPISODES, INTERRUPTED, TICKETS

DECIDED = "decided"
"""A group the loop decided to play and has not asked the job for (a loop that starts now asks for it)."""
WAITING = "waiting"
"""Asked for; none of its episodes has started."""
PLAYING = "playing"
ENDED = "ended"
"""Every episode has ended; its step is not decided (the loop is busy with a group before it, or is not running)."""
STEPPING = "stepping"
"""Its step is decided and the version it makes is not there: the trainer has it, or a loop that starts now takes
the step again."""
MADE = "made"
"""The version is made; the group's outcome is not written yet."""

SHOWN = 240
"""Measurements of each kind in a snapshot: the newest."""


class System:
    def __init__(self, directory: Path, feed: FeedReader) -> None:
        self.directory = directory
        self.feed = feed
        self.machine = Machine(directory)
        self._ledger = FileLedger(directory / LEDGER)
        self._jobs: dict[str, _JobLog] = {}

    async def snapshot(self) -> dict[str, Any]:
        """Where everything stands now: the runs' groups that are not done with and the ones that are, the
        policies' versions, the jobs, what each channel serves and how fast, the machine, and what is kept."""
        tables: dict[str, dict[str, JsonValue]] = {}
        fences: dict[str, int] = {}
        runs: list[str] = []
        policies: list[dict[str, Any]] = []
        if await asyncio.to_thread(self._ledger.directory.exists):  # (a reader makes no ledger where none is)
            fences = await self._ledger.fences()
            tables = {name: await self._ledger.read(name) for name in await self._ledger.tables()}
            runs = await runs_in(self._ledger)
            for policy in await policies_in(self._ledger):
                versions = await versions_in(self._ledger, policy)
                policies.append(_policy(policy, versions, fences.get(policy_scope(policy))))
        return await asyncio.to_thread(self._assembled, tables, fences, runs, policies)

    def _assembled(
        self,
        tables: Mapping[str, Mapping[str, JsonValue]],
        fences: Mapping[str, int],
        runs: list[str],
        policies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        made = {version["name"] for policy in policies for version in policy["versions"]}
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
                    {name: tables.get(table(run, name), {}) for name in (GROUPS, STEPS, ITERATIONS)},
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
    made: set[str],
    job: "_JobLog | None",
    in_feed: list[dict[str, Any]],
) -> dict[str, Any]:
    """A run: its groups that are not done with, each with its stage and its episodes, and the ones that are."""
    groups: Any = tables[GROUPS]
    steps: Any = tables[STEPS]
    done: Any = tables[ITERATIONS]
    open_groups: list[dict[str, Any]] = []
    for number in sorted((number for number in groups if number not in done), key=int):
        group = groups[number]
        ticket = job.asked(number) if job else None
        asked = ticket["id"] if ticket else None
        episodes: dict[str, dict[str, Any]] = {}
        for each in reversed(in_feed):  # (oldest first)
            if asked and each["labels"].get("ticket") == asked:
                reward = next(iter(each["rewards"].values()), None)
                episodes[each["run_id"]] = {
                    "run_id": each["run_id"],
                    "episode": each["labels"].get("episode"),
                    "state": each["state"],
                    "samples": each["samples"],
                    "reward": reward,
                    "updated": each["updated"],
                    "in_feed": True,
                }
        for ended in job.of(asked) if job and asked else []:
            episodes.setdefault(ended["run_id"], {"samples": None, "updated": None, "in_feed": False}).update(ended)
        counted = [each for each in episodes.values() if "outcome" in each and not each.get("interrupted")]
        step = steps.get(number)
        if step is not None:
            policy = step.get("policy") or (parsed(step["parent"])[0] if step.get("parent") else None)
            name = named(policy, int(step["number"])) if policy else None
            step = {**step, "makes": name}
            step.pop("batch", None)
            stage = MADE if name in made else STEPPING
        elif ticket is None:
            stage = DECIDED
        elif len(counted) >= int(ticket["count"]):
            stage = ENDED
        else:
            stage = PLAYING if episodes else WAITING
        open_groups.append(
            {
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
            }
        )
    iterations = [_iteration(done[number]) for number in sorted(done, key=int)]
    return {"run": run, "fence": fence, "decided": len(groups), "open": open_groups, "iterations": iterations}


def _iteration(line: Mapping[str, Any]) -> dict[str, Any]:
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
        """Bytes of the episodes' traces and events in the blob store."""
        self.sampled = 0
        self.written = 0.0
        self._tickets, self._episodes = Appended(directory / TICKETS), Appended(directory / EPISODES)

    def refresh(self) -> None:
        self.tickets.extend(self._tickets.more())
        for line in self._episodes.more():
            record = Record.from_json(line)
            episode = record.episode
            self.kept += sum(blob.size for blob in (record.traces, record.events) if blob is not None)
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
                }
            )
        logs = [path for path in (self.directory / TICKETS, self.directory / EPISODES) if path.exists()]
        self.written = max((path.stat().st_mtime for path in logs), default=0.0)

    def asked(self, number: str) -> dict[str, Any] | None:
        """The ticket a group was asked for under, by the group's number (the label `iteration`)."""
        asked = [ticket for ticket in self.tickets if ticket["labels"].get("iteration") == number]
        return asked[-1] if asked else None

    def of(self, ticket: str) -> list[dict[str, Any]]:
        return [episode for episode in self.episodes if episode["ticket"] == ticket]

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
