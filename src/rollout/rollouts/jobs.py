"""Rollout jobs: run rows of a task, many at a time, and read what they produced as one stream of episodes.

The caller, typically whoever trains, decides what to run, how often and how to group it: a job knows no
algorithm. It admits runs as there is room, watches them, and when one ends appends an `Episode` to its log. The log
is read with a cursor, so a caller can train while runs are in flight, on whatever has finished (asynchronous
reinforcement learning is this and nothing more), and can pick up where it left off.

`RolloutJobs` implements this over any `Runner` (one that runs programs in this process, or a durable one over a
database); `rollout.rollouts.service` offers the same job to a caller on another machine.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.core.contracts import RunEvent
from rollout.core.harness.runner import ProgramReference, RunBinding, Runner, RunSpecification, with_row
from rollout.recorder import Epoch
from rollout.rollouts.episodes import Episode, Outcome, assemble


class Recorded(Protocol):
    """What a job needs of the recorder: each run's sequences, and somewhere to publish weights."""

    def sessions(self, run_id: str) -> dict[str, list[Epoch]]: ...
    def forget(self, run_id: str) -> None: ...
    async def publish(self, channel: str, adapter: str, path: str) -> int: ...


class JobHooks:
    """Watch a job at the level its caller thinks at: tickets, episodes, published weights, the caller's own notes."""

    def on_job(self, event: Mapping[str, JsonValue]) -> None:
        """`event["kind"]` is `ticket`, `admitted`, `episode`, `published`, or a kind the caller noted."""


@dataclass(frozen=True)
class Status:
    queued: int
    """Runs waiting for room."""
    running: int
    finished: int
    """Episodes in the log, of every outcome."""
    acknowledged: int
    """The cursor the caller has consumed through."""


class Ticket(Protocol):
    id: str

    async def episodes(self) -> list[Episode]:
        """The ticket's episodes, once every one of its runs has ended (in the order they ended)."""
        ...


class Jobs(Protocol):
    """Where jobs are started: `RolloutJobs` in this process, or `rollout.rollouts.service.RolloutClient` for jobs
    served elsewhere."""

    async def start(
        self, *, program: ProgramReference, binding: RunBinding, in_flight: int, name: str = ""
    ) -> "Job": ...


class Job(Protocol):
    id: str

    async def run(self, parameters: JsonValue, *, labels: Mapping[str, str] | None = None, count: int = 1) -> Ticket:
        """Queue `count` runs of one row. They start together, when there is room for all of them."""
        ...

    def episodes(self, cursor: int = 0) -> AsyncIterator[Episode]:
        """Every episode after `cursor`, then new ones as runs end, until the job is closed."""
        ...

    async def acknowledge(self, cursor: int) -> None:
        """The caller has consumed everything through `cursor`: it need not be kept."""
        ...

    async def publish(self, channel: str, adapter: str, path: str) -> int:
        """Serve new weights on a channel from now on; returns the channel's new version."""
        ...

    async def note(self, kind: str, payload: Mapping[str, JsonValue]) -> None:
        """Put something of the caller's own (an update's statistics, say) where whoever watches the job sees it."""
        ...

    async def status(self) -> Status: ...


@dataclass
class _Ticket:
    id: str
    parameters: JsonValue
    labels: Mapping[str, str]
    count: int
    ended: list[Episode] = field(default_factory=list[Episode])
    done: asyncio.Event = field(default_factory=asyncio.Event)
    failure: BaseException | None = None
    collected: Callable[[], None] = lambda: None

    async def episodes(self) -> list[Episode]:
        await self.done.wait()
        self.collected()
        if self.failure is not None:
            raise self.failure
        return list(self.ended)


class RolloutJob:
    """A job over a runner. Created by `RolloutJobs.start`."""

    def __init__(
        self,
        job_id: str,
        specification: RunSpecification,
        runner: Runner,
        recorder: Recorded,
        *,
        in_flight: int,
        log: Path | None,
        hooks: Sequence[JobHooks],
        guard: Callable[[], None] | None,
    ) -> None:
        self.id = job_id
        self._specification = specification
        self._runner = runner
        self._recorder = recorder
        self._room = in_flight
        self._log = log
        self._hooks = hooks
        self._guard = guard
        self._queue: list[_Ticket] = []
        self.tickets: dict[str, _Ticket] = {}
        """Tickets by id, until their episodes have been collected (`collect`)."""
        self._running = 0
        self._episodes: list[Episode] = []
        self._first = 1  # the cursor of `_episodes[0]`
        self._acknowledged = 0
        self._news = asyncio.Condition()
        self._closed = False
        self._tasks: set[asyncio.Task[None]] = set()
        if log is not None:
            log.mkdir(parents=True, exist_ok=True)
            kept = sorted(log.glob("*.json"))
            self._episodes = [Episode.from_json(json.loads(path.read_text())) for path in kept]
            if self._episodes:
                self._first = self._episodes[0].cursor
                self._acknowledged = self._first - 1

    # For the caller

    async def run(self, parameters: JsonValue, *, labels: Mapping[str, str] | None = None, count: int = 1) -> _Ticket:
        if self._closed:
            raise RuntimeError(f"job {self.id} is closed")
        ticket = _Ticket(f"t_{uuid.uuid4().hex[:10]}", parameters, dict(labels or {}), count)
        ticket.collected = lambda: self.collect(ticket.id)
        self._queue.append(ticket)
        self.tickets[ticket.id] = ticket
        self._tell("ticket", ticket=ticket.id, labels=dict(ticket.labels), count=count)
        self._spawn(self._admit())
        return ticket

    async def episodes(self, cursor: int = 0) -> AsyncIterator[Episode]:
        while True:
            async with self._news:
                await self._news.wait_for(lambda after=cursor: self._closed or self._last > after)
                ready = [episode for episode in self._episodes if episode.cursor > cursor]
            if not ready:
                return  # closed, and nothing is left
            for episode in ready:
                cursor = episode.cursor
                yield episode

    def collect(self, ticket: str) -> None:
        """A ticket's episodes have been read: it need not be kept."""
        self.tickets.pop(ticket, None)

    def after(self, cursor: int) -> list[Episode]:
        """The episodes after `cursor` that are in the log now."""
        return [episode for episode in self._episodes if episode.cursor > cursor]

    async def acknowledge(self, cursor: int) -> None:
        self._acknowledged = max(self._acknowledged, cursor)
        while self._episodes and self._episodes[0].cursor <= cursor:
            done = self._episodes.pop(0)
            self._first = done.cursor + 1
            if self._log is not None:
                (self._log / f"{done.cursor:09d}.json").unlink(missing_ok=True)

    async def publish(self, channel: str, adapter: str, path: str) -> int:
        version = await self._recorder.publish(channel, adapter, path)
        self._tell("published", channel=channel, adapter=adapter, version=version)
        return version

    async def note(self, kind: str, payload: Mapping[str, JsonValue]) -> None:
        self._tell(kind, **payload)

    async def status(self) -> Status:
        queued = sum(ticket.count for ticket in self._queue)
        return Status(queued, self._running, self._last, self._acknowledged)

    async def close(self) -> None:
        """Stop admitting, cancel what is running, and end every `episodes` stream."""
        self._closed = True
        for ticket in self._queue:
            ticket.failure = RuntimeError(f"job {self.id} was closed before the ticket was admitted")
            ticket.done.set()
        self._queue.clear()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        async with self._news:
            self._news.notify_all()

    # Internals

    @property
    def _last(self) -> int:
        return self._first + len(self._episodes) - 1

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _admit(self) -> None:
        """Start the tickets at the head of the queue that there is room for (all of a ticket's runs, or none)."""
        while self._queue and not self._closed:
            ticket = self._queue[0]
            if self._running and self._running + ticket.count > self._room:
                return
            self._queue.pop(0)
            if self._guard is not None:
                try:
                    self._guard()
                except Exception as error:  # refused (no memory, say): the ticket's caller is told
                    ticket.failure = error
                    ticket.done.set()
                    continue
            self._running += ticket.count
            labels = {**ticket.labels, "job": self.id, "ticket": ticket.id}
            started: list[str] = []
            for number in range(1, ticket.count + 1):
                try:
                    handle = await self._runner.start(
                        self._specification_for(ticket), labels={**labels, "episode": str(number)}
                    )
                except Exception as error:  # a run that cannot start is a failed episode like any other
                    await self._ended(ticket, "", [], {}, detail=f"{type(error).__name__}: {error}")
                    continue
                started.append(handle.run_id)
                self._spawn(self._watch(ticket, handle.run_id, handle.events()))
            self._tell("admitted", ticket=ticket.id, runs=list(started))

    def _specification_for(self, ticket: _Ticket) -> RunSpecification:
        program = with_row(self._specification.program, ticket.parameters)
        return self._specification.model_copy(update={"program": program})

    async def _watch(self, ticket: _Ticket, run_id: str, stream: AsyncIterator[RunEvent]) -> None:
        events: list[RunEvent] = []
        try:
            async for event in stream:
                events.append(event)
        finally:
            epochs = self._recorder.sessions(run_id)
            self._recorder.forget(run_id)
            await self._ended(ticket, run_id, events, epochs)

    async def _ended(
        self,
        ticket: _Ticket,
        run_id: str,
        events: Sequence[RunEvent],
        epochs: Mapping[str, list[Epoch]],
        detail: str | None = None,
    ) -> None:
        """A run is over, however it ended: its episode goes into the log, and whoever waits is told."""
        self._running -= 1
        cursor = self._last + 1
        if events:
            episode = assemble(
                events, epochs, cursor=cursor, job=self.id, ticket=ticket.id, parameters=ticket.parameters
            )
        else:  # it never started, or was cancelled before its first event: still an episode, so that counts are exact
            outcome = Outcome.FAILED if detail else Outcome.CANCELLED
            episode = Episode(cursor, self.id, ticket.id, run_id, ticket.labels, ticket.parameters, outcome, detail)
        self._episodes.append(episode)
        if self._log is not None:
            (self._log / f"{episode.cursor:09d}.json").write_text(json.dumps(episode.to_json()))
        ticket.ended.append(episode)
        if len(ticket.ended) == ticket.count:
            ticket.done.set()
        self._tell(
            "episode",
            cursor=episode.cursor,
            ticket=ticket.id,
            run=run_id,
            labels=dict(episode.labels),
            outcome=episode.outcome.value,
            detail=episode.detail,
            reward=episode.reward,
            info=dict(episode.info),
            sampled={slot: sum(epoch.sampled for epoch in trace.epochs) for slot, trace in episode.traces.items()},
        )
        async with self._news:
            self._news.notify_all()
        if not self._closed:
            self._spawn(self._admit())

    def _tell(self, kind: str, **payload: JsonValue) -> None:
        event: dict[str, JsonValue] = {"kind": kind, "job": self.id, "at": round(time.time(), 3), **payload}
        for hook in self._hooks:
            hook.on_job(event)


class RolloutJobs:
    """Starts jobs on a runner. `log` keeps each job's unacknowledged episodes on disk (under `log/JOB`), so that a
    caller that stops can go on from its cursor; `guard` is called before runs are admitted and raises to refuse them
    (a machine out of memory, say)."""

    def __init__(
        self,
        runner: Runner,
        recorder: Recorded,
        *,
        log: Path | None = None,
        hooks: Sequence[JobHooks] = (),
        guard: Callable[[], None] | None = None,
    ) -> None:
        self._runner = runner
        self._recorder = recorder
        self._log = log
        self._hooks = list(hooks)
        self._guard = guard
        self.jobs: dict[str, RolloutJob] = {}

    async def start(
        self, *, program: ProgramReference, binding: RunBinding, in_flight: int, name: str = ""
    ) -> RolloutJob:
        """A job that runs `program` (with each ticket's row as its parameters) under `binding`, at most `in_flight`
        runs at a time. A `name` makes the job's log one a later process finds again."""
        job_id = name or f"j_{uuid.uuid4().hex[:10]}"
        job = RolloutJob(
            job_id,
            RunSpecification(program=program, binding=binding),
            self._runner,
            self._recorder,
            in_flight=in_flight,
            log=self._log / job_id if self._log is not None else None,
            hooks=self._hooks,
            guard=self._guard,
        )
        self.jobs[job_id] = job
        return job

    async def close(self) -> None:
        await asyncio.gather(*(job.close() for job in self.jobs.values()))
