"""What changed, as it changes: the monitor's topics, each read at most once a beat whoever asks, and a version of each
that changes when what it says does.

A topic is a thing the page shows, by name:

- `system`: where every run stands (`System.snapshot`);
- `machines`: every machine that beats and the roles on it (`System.machines`);
- `launches`: the runs asked for, each with its job and state (`System.launches`);
- `offers`: what a run can be asked for on the monitor's cluster (`System.offers`);
- `evals`: the suites and the evals that played them (`System.evals`);
- `environments`: every environment the system knows of (`System.environments`);
- `environment/MODULE:NAME`: one environment's page (`System.environment`; a published one's is
  `environment/NAME@VERSION`);
- `environment-versions`, `environment-version/VERSION`: the published environments' versions, and one
  (`System.environment_versions`, `System.environment_version`);
- `imports`: the imports from git this monitor made, each with its stage (`System.imports`);
- `feeds`: every episode in the runs' feeds, summarised (`System.feeds`);
- `statistics`: every run in figures (`System.statistics`);
- `checkpoints`: the checkpoints as a graph (`System.lineage`);
- `checkpoint-evals/ID`, `path/ID`: every eval a checkpoint has had, and its line with each point's scores
  (`System.checkpoint_evals`, `System.path`);
- `eval-subjects`: every subject (a checkpoint or a base model) that has had an eval (`System.eval_subjects`);
- `history/checkpoint/ID`, `history/model/NAME`: every eval a subject has had (`System.history`);
- `settings/RUN`: a training run's settings, and what is wanted of them (`System.settings`);
- `group/RUN/NUMBER`: one group (`System.group`);
- `episode/RUN_ID`: one episode's lines; its version is how many there are, where from and its state, and the page
  asks for the lines it lacks (`System.episode` with `after`).

`Hub.read` answers a topic with its body (JSON) and its version, read again only when the last reading is older than
a beat; the JSON endpoints answer from it (and a request that names the version it has is told nothing changed).
`Hub.watch` is a stream of each watched topic's version: once at once, then whenever it changes. While anyone watches,
the hub reads the watched topics every beat, reading the ledger once for all of them (`System.one_reading`); nobody
watching, it reads nothing.
"""

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import AsyncGenerator, Collection
from dataclasses import dataclass, field
from typing import Any, cast

from rollout_train.monitor.scores import CHECKPOINT, MODEL
from rollout_train.monitor.system import System

BEAT = 1.5
"""Seconds between readings of a watched topic, and how long a reading is answered from."""
VOLATILE = ("at", "now")
"""What a body says about when it was read: not part of its version (a body read again unchanged keeps it)."""
MISSING = "missing"
"""The version of a topic that names nothing (a group that is not in the ledger)."""


@dataclass(frozen=True)
class Reading:
    body: bytes
    version: str
    at: float


def version_of(payload: Any) -> str:
    """A short hash of what a body says, leaving out when it was read."""
    if isinstance(payload, dict):
        whole = cast(dict[str, Any], payload)
        payload = {key: value for key, value in whole.items() if key not in VOLATILE}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(text.encode(), digest_size=10).hexdigest()


@dataclass
class _Episode:
    """What the hub knows of an episode's lines: how many, from where, and the episode's state."""

    count: int = 0
    source: str | None = None
    state: str | None = None


@dataclass
class Hub:
    system: System
    beat: float = BEAT
    _readings: dict[str, Reading] = field(default_factory=dict[str, Reading])
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict[str, asyncio.Lock])
    _episodes: dict[str, _Episode] = field(default_factory=dict[str, _Episode])
    _watchers: list[tuple[frozenset[str], asyncio.Queue[tuple[str, str]]]] = field(
        default_factory=list[tuple[frozenset[str], asyncio.Queue[tuple[str, str]]]]
    )
    _wake: asyncio.Event = field(default_factory=asyncio.Event)

    async def read(self, topic: str) -> Reading:
        """A topic's body and version: the last reading, if it is younger than a beat; else read now (once, however
        many ask at the same time)."""
        lock = self._locks.setdefault(topic, asyncio.Lock())
        async with lock:
            last = self._readings.get(topic)
            if last is not None and time.monotonic() - last.at < self.beat:
                return last
            payload = await self._payload(topic)
            body = json.dumps(payload, separators=(",", ":"), default=str).encode()
            version = MISSING if payload is None else self._version(topic, payload)
            reading = Reading(body, version, time.monotonic())
            self._readings[topic] = reading
            return reading

    def _version(self, topic: str, payload: Any) -> str:
        if topic.startswith("episode/"):
            known = self._episodes[topic]
            return f"{known.source}:{known.count}:{known.state}"
        return version_of(payload)

    async def _payload(self, topic: str) -> Any:
        system = self.system
        if topic == "system":
            return await system.snapshot()
        if topic == "machines":
            return await system.machines()
        if topic == "launches":
            return await system.launches()
        if topic == "offers":
            return await system.offers()
        if topic == "evals":
            return await system.evals()
        if topic == "environments":
            return await system.environments()
        if topic.startswith("environment/"):
            return await system.environment(topic.removeprefix("environment/"))
        if topic == "environment-versions":
            return await system.environment_versions()
        if topic.startswith("environment-version/"):
            return await system.environment_version(topic.removeprefix("environment-version/"))
        if topic == "imports":
            return await system.imports()
        if topic == "feeds":
            return await asyncio.to_thread(system.feeds)
        if topic == "statistics":
            return await system.statistics()
        if topic == "checkpoints":
            return await system.lineage()
        if topic.startswith("checkpoint-evals/"):
            return await system.checkpoint_evals(topic.removeprefix("checkpoint-evals/"))
        if topic.startswith("path/"):
            return await system.path(topic.removeprefix("path/"))
        if topic == "eval-subjects":
            return await system.eval_subjects()
        if topic.startswith("history/"):
            kind, _, reference = topic.removeprefix("history/").partition("/")
            return await system.history(kind, reference) if kind in (CHECKPOINT, MODEL) and reference else None
        if topic.startswith("settings/"):
            return await system.settings(topic.removeprefix("settings/"))
        if topic.startswith("group/"):
            run, _, number = topic.removeprefix("group/").rpartition("/")
            return await system.group(run, int(number)) if number.isdigit() else None
        if topic.startswith("episode/"):
            return await self._episode(topic)
        raise KeyError(topic)

    async def _episode(self, topic: str) -> dict[str, Any]:
        """What changed of an episode's lines since the hub last looked: their count, where they are read from, its
        state (the page asks for the lines themselves)."""
        run_id = topic.removeprefix("episode/")
        known = self._episodes.setdefault(topic, _Episode())
        answer = await self.system.episode(run_id, after=known.count)
        if answer["source"] != known.source:  # (read from elsewhere now: counted again from the start)
            answer = await self.system.episode(run_id)
            known.count = 0
        known.count += len(answer["lines"])
        known.source, known.state = answer["source"], answer["state"]
        return {"run_id": run_id, "lines": known.count, "source": known.source, "state": known.state}

    def forget(self) -> None:
        """Read every topic afresh, at once (after the monitor itself changed something: a name)."""
        self._readings.clear()
        self.system.read_afresh()
        self._wake.set()

    async def watch(self, topics: Collection[str]) -> AsyncGenerator[tuple[str, str]]:
        """Each topic's version at once, then each time it changes, for as long as the caller reads."""
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        watched = frozenset(topics)
        self._watchers.append((watched, queue))
        self._wake.set()
        try:
            for topic in sorted(watched):
                yield topic, (await self.read(topic)).version
            while True:
                yield await queue.get()
        finally:
            self._watchers.remove((watched, queue))

    async def run(self) -> None:
        """Read the watched topics every beat, and tell whoever watches a topic when its version changes."""
        told: dict[str, str] = {}
        while True:
            topics = {topic for watched, _ in self._watchers for topic in watched}
            if not topics:
                told.clear()
                self._wake.clear()
                await self._wake.wait()
                continue
            async with self.system.one_reading():  # (the ledger read once for every topic)
                for topic in sorted(topics):
                    try:
                        version = (await self.read(topic)).version
                    except Exception as error:  # (a topic that cannot be read now: said so, and read again next beat)
                        version = f"error:{type(error).__name__}"
                    if told.get(topic) != version:
                        if topic in told:
                            for watched, queue in list(self._watchers):
                                if topic in watched:
                                    queue.put_nowait((topic, version))
                        told[topic] = version
            for topic in set(told) - topics:
                del told[topic]
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), self.beat)
            self._wake.clear()
