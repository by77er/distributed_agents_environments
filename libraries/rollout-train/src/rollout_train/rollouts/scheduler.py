"""Episodes played by runners: a run asks for each group's episodes in the ledger, and runners claim and play them
(docs/libraries/rollout-train/rollouts.md).

A run records each group with how many episodes it wants (`runs/RUN/groups`, its `episodes`) and how its episodes are
played (`runs/RUN/plans`: its program and binding). A runner, wherever it reaches the ledger, claims episodes nobody is
playing, as many at once as it has places: it appends a claim under `GROUP/EPISODE/ATTEMPT` (`runs/RUN/claims`; the
first append wins, so two runners never play one attempt). It plays the episode, keeps its trajectories and events in
the blob store, and records it under `GROUP/EPISODE` (`runs/RUN/episodes`) when it ends. The run reads a group's
episodes from there once they have all ended.

A claim holds while its runner is the one that made it and is alive: a runner started again takes its fence anew, and
a runner beats every few seconds (`rollout_train.presence`), so one whose machine died stops beating; what either had
claimed is claimed again by whoever has room. An episode its runner cut short by closing is noted
(`runs/RUN/interrupted`) and claimed again too. Several runners, on one machine or many, share the work the same way:
which machine plays a group's episodes is only a matter of where runners are.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.contracts import RunEvent
from rollout.harness.blobs import Blobs
from rollout.harness.runner import ProgramReference, RunBinding, Runner, RunSpecification, with_row
from rollout_train.ledger import Fence, Ledger
from rollout_train.presence import Presence, alive
from rollout_train.record import GROUPS, RESULTS, runs_in, table
from rollout_train.recorder import Segment
from rollout_train.rollouts.episodes import Episode, Outcome, Record, assemble, loaded, stored

PLANS, CLAIMS, EPISODES, INTERRUPTED = "plans", "claims", "episodes", "interrupted"
"""A run's tables beside its groups: how its episodes are played, who plays which, the episodes that ended, and the
attempts cut short."""
CLOSED = "its runner closed"
"""Why an attempt was cut short when its runner closed: it is played again."""


def runner_scope(name: str) -> str:
    """The scope whose fence a runner holds while it plays: a runner started again takes it anew."""
    return f"runners/{name}"


@dataclass(frozen=True)
class Plan:
    """How a run's episodes are played: its program (each group's start is its row) and its binding."""

    program: ProgramReference
    binding: RunBinding

    def to_json(self) -> dict[str, JsonValue]:
        return {"program": self.program.model_dump(mode="json"), "binding": self.binding.model_dump(mode="json")}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Plan":
        return cls(ProgramReference.model_validate(data["program"]), RunBinding.model_validate(data["binding"]))


async def plan(ledger: Ledger, run: str, played: Plan, fence: Fence) -> None:
    """Say how a run's episodes are played, from now on (each start of a run may say it anew)."""
    await ledger.append(table(run, PLANS), str(fence.number), played.to_json(), fence)


async def ended(ledger: Ledger, blobs: Blobs, run: str, group: int, count: int) -> list[Episode] | None:
    """A group's episodes, with their trajectories, once all `count` have ended; until then None."""
    records = await ledger.read(table(run, EPISODES))
    keys = [f"{group}/{number}" for number in range(1, count + 1)]
    if not all(key in records for key in keys):
        return None
    return [await loaded(Record.from_json(_mapping(records[key])), blobs) for key in keys]


async def episodes_of(
    ledger: Ledger, blobs: Blobs, run: str, group: int, count: int, *, every: float = 0.5
) -> list[Episode]:
    """A group's episodes once all `count` have ended, waiting for them."""
    while (found := await ended(ledger, blobs, run, group, count)) is None:  # noqa: ASYNC110 (another process writes)
        await asyncio.sleep(every)
    return found


class Recorded(Protocol):
    """What a runner needs of the recorder: each run's segments, and the channels it serves."""

    channels: Mapping[str, Any]

    def sessions(self, run_id: str) -> dict[str, list[Segment]]: ...
    def forget(self, run_id: str) -> None: ...


class Hooks(Protocol):
    """Watch a runner (episodes as they start and end) or a run (its results and steps)."""

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        """`event["kind"]` is `started` or `ended` (an episode, by a runner), or the run's `result`, `step` or
        `published`."""
        ...


@dataclass(frozen=True)
class Open:
    """An episode a run asks for that nobody plays now: the attempt a claim would make."""

    run: str
    group: int
    number: int
    attempt: int
    decided: float


@dataclass
class EpisodeRunner:
    """Claims the episodes runs ask for in `ledger` and plays them on `runner`, at most `places` at once: those of the
    runs it can serve (whose models its recorder's channels serve and whose imports are among `imports`), and of `runs`
    only, if given. `guard` is called before claiming and raises to wait (a machine short of memory, say). With
    `presence`, it beats every `beating` seconds, with what `about` says of its machine besides its places and how
    many it plays, and a claim holds only while its runner beats."""

    name: str
    ledger: Ledger
    runner: Runner
    recorder: Recorded
    blobs: Blobs
    places: int
    imports: Collection[str] = ()
    runs: Collection[str] | None = None
    hooks: Sequence[Hooks] = ()
    guard: Callable[[], None] | None = None
    presence: Presence | None = None
    about: Callable[[], Mapping[str, JsonValue]] | None = None
    """What the runner says of its machine in each beat (called in a thread: it may measure)."""
    every: float = 0.5
    """Seconds between looks for work while nothing ends."""
    beating: float = 15.0
    """Seconds between beats."""
    _fence: Fence | None = None
    _playing: dict[str, asyncio.Task[None]] = field(default_factory=dict[str, asyncio.Task[None]])
    _news: asyncio.Event = field(default_factory=asyncio.Event)

    async def serve(self) -> None:
        """Claim and play episodes until cancelled; what is playing then is cut short and noted."""
        self._fence = await self.ledger.take(runner_scope(self.name))
        beating: asyncio.Task[None] | None = None
        if self.presence is not None:
            await self._beat()  # (alive before it claims anything)
            beating = asyncio.create_task(self._beats())
        try:
            while True:
                room = self.places - len(self._playing)
                if room > 0 and self._fits():
                    for each in (await self.open())[:room]:
                        await self._claim(each)
                self._news.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._news.wait(), self.every)
        finally:
            playing = [*self._playing.values(), *([beating] if beating else [])]
            for task in playing:
                task.cancel()
            await asyncio.gather(*playing, return_exceptions=True)

    async def _beats(self) -> None:
        while True:
            await asyncio.sleep(self.beating)
            with contextlib.suppress(Exception):  # (a beat missed is noticed only if many are)
                await self._beat()

    async def _beat(self) -> None:
        assert self.presence is not None
        said: Mapping[str, JsonValue] = await asyncio.to_thread(self.about) if self.about is not None else {}
        await self.presence.beat(self.name, {**said, "places": self.places, "playing": len(self._playing)})

    async def open(self) -> list[Open]:
        """The episodes nobody plays now, of the runs this runner serves, oldest group first."""
        fences = await self.ledger.fences()
        beats = {beat.runner: beat for beat in await self.presence.beats()} if self.presence is not None else None
        found: list[Open] = []
        for run in await runs_in(self.ledger):
            if self.runs is not None and run not in self.runs:
                continue
            plans = await self.ledger.read(table(run, PLANS))
            if not plans or not self._serves(Plan.from_json(_mapping(plans[max(plans, key=int)]))):
                continue
            groups = await self.ledger.read(table(run, GROUPS))
            results = await self.ledger.read(table(run, RESULTS))
            done = await self.ledger.read(table(run, EPISODES))
            claims = await self.ledger.read(table(run, CLAIMS))
            cut = await self.ledger.read(table(run, INTERRUPTED))
            attempts: dict[str, list[tuple[int, bool]]] = {}  # each episode's attempts: (attempt, whether it holds)
            for key, claim in claims.items():
                group, number, attempt = key.split("/")
                made = _mapping(claim)
                runner = str(made["runner"])
                holds = key not in cut and fences.get(runner_scope(runner)) == made["fence"]
                holds = holds and (beats is None or runner == self.name or alive(beats.get(runner)))
                attempts.setdefault(f"{group}/{number}", []).append((int(attempt), holds))
            for key, group in groups.items():
                record = _mapping(group)
                count = record.get("episodes")
                if key in results or not isinstance(count, int):
                    continue
                for number in range(1, count + 1):
                    episode = f"{key}/{number}"
                    made = attempts.get(episode, [])
                    if episode in done or any(holds for _, holds in made):
                        continue
                    decided = float(str(record.get("decided") or 0.0))
                    found.append(Open(run, int(key), number, max((a for a, _ in made), default=0) + 1, decided))
        return sorted(found, key=lambda each: (each.decided, each.run, each.group, each.number))

    def _serves(self, played: Plan) -> bool:
        channels = {binding.recorded.channel for binding in played.binding.models.values() if binding.recorded}
        local = {binding.local for binding in played.binding.imports.values() if binding.local}
        return channels <= set(self.recorder.channels) and local <= set(self.imports)

    def _fits(self) -> bool:
        if self.guard is None:
            return True
        try:
            self.guard()
        except Exception:  # (not now: there is not enough of something; looked at again in a while)
            return False
        return True

    async def _claim(self, each: Open) -> None:
        assert self._fence is not None
        key = f"{each.group}/{each.number}/{each.attempt}"
        claim: dict[str, JsonValue] = {"runner": self.name, "fence": self._fence.number, "at": round(time.time(), 1)}
        if not await self.ledger.append(table(each.run, CLAIMS), key, claim, self._fence):
            return  # another runner claimed this attempt first
        task = asyncio.create_task(self._play(each, key))
        self._playing[key] = task
        task.add_done_callback(lambda _: self._done(key))

    def _done(self, key: str) -> None:
        self._playing.pop(key, None)
        self._news.set()

    async def _play(self, each: Open, key: str) -> None:
        assert self._fence is not None
        plans = await self.ledger.read(table(each.run, PLANS))
        played = Plan.from_json(_mapping(plans[max(plans, key=int)]))
        group = _mapping((await self.ledger.read(table(each.run, GROUPS)))[str(each.group)])
        specification = RunSpecification(program=with_row(played.program, group["parameters"]), binding=played.binding)
        labels = {"run": each.run, "group": str(each.group), "episode": str(each.number)}
        try:
            handle = await self.runner.start(specification, labels=labels)
        except Exception as error:  # a run that cannot start is a failed episode like any other
            detail = f"{type(error).__name__}: {error}"
            await self._ended(each, Episode(each.run, each.group, each.number, "", labels, Outcome.FAILED, detail), [])
            return
        self._tell("started", run=each.run, group=each.group, episode=each.number, run_id=handle.run_id)
        events: list[RunEvent] = []
        try:
            async for event in handle.events():
                events.append(event)
        except asyncio.CancelledError:  # the runner is closing: the attempt is noted, and played again by someone
            with contextlib.suppress(Exception):
                await self.runner.cancel(handle.run_id, reason=CLOSED)
            self.recorder.forget(handle.run_id)
            cut: dict[str, JsonValue] = {"why": CLOSED, "at": round(time.time(), 1)}
            with contextlib.suppress(Exception):
                await asyncio.shield(self.ledger.append(table(each.run, INTERRUPTED), key, cut, self._fence))
            raise
        segments = self.recorder.sessions(handle.run_id)
        self.recorder.forget(handle.run_id)
        await self._ended(each, assemble(events, segments, run=each.run, group=each.group, number=each.number), events)

    async def _ended(self, each: Open, episode: Episode, events: Sequence[RunEvent]) -> None:
        assert self._fence is not None
        record = await stored(episode, events, self.blobs)
        key = f"{each.group}/{each.number}"
        await self.ledger.append(table(each.run, EPISODES), key, record.to_json(), self._fence)
        self._tell(
            "ended",
            run=each.run,
            group=each.group,
            episode=each.number,
            run_id=episode.run_id,
            labels=dict(episode.labels),
            outcome=episode.outcome.value,
            detail=episode.detail,
            reward=episode.reward,
            info=dict(episode.info),
            sampled=dict(record.sampled),
        )

    def _tell(self, kind: str, **payload: JsonValue) -> None:
        event: dict[str, JsonValue] = {"kind": kind, "runner": self.name, "at": round(time.time(), 3), **payload}
        for hook in self.hooks:
            hook.on_note(event)


@contextlib.asynccontextmanager
async def playing(runner: EpisodeRunner) -> AsyncGenerator[None]:
    """`async with playing(runner):` — the runner serves while the block runs."""
    task = asyncio.create_task(runner.serve())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _mapping(record: JsonValue) -> dict[str, Any]:
    assert isinstance(record, dict)
    return record
