"""Episodes played by runners: a run asks for each group's episodes in the ledger, and runners claim and play them
(docs/libraries/rollout-train/rollouts.md).

A run records each group with how many episodes it wants (`runs/RUN/groups`, its `episodes`) and how its episodes are
played (`runs/RUN/plans`: its program and binding). A runner, wherever it reaches the ledger, claims episodes nobody is
playing, as many at once as it has places: it appends a claim under `GROUP/EPISODE/ATTEMPT` (`runs/RUN/claims`; the
first append wins, so two runners never play one attempt). It plays the episode, keeps its trajectories and events in
the blob store, and records it under `GROUP/EPISODE` (`runs/RUN/episodes`) when it ends. The run reads a group's
episodes from there once they have all ended.

A claim holds while it is its episode's latest attempt and its runner is the one that made it and is alive: a runner
started again takes its fence anew, and a runner beats every few seconds (`rollout_train.presence`), so one whose
machine died stops beating; what either had claimed is claimed again by whoever has room. An episode its runner cut
short by closing is noted (`runs/RUN/interrupted`) and claimed again too. A claim names the run that plays it. Over a
runner whose runs survive it, a runner started again adopts the runs it finds of its claims that are
still their episodes' latest attempts (`runs/RUN/adopted`, under its new fence), and they play on; one whose claim
lapsed meanwhile is cut short. An adopted run's turns survived with it: the gateway kept them, and answers a sample
asked for again with the turn it recorded, so the episode trains like any other.

Each episode has a fence of its own (`runs/RUN/episodes/GROUP/EPISODE`, `episode_scope`). The runner whose claim was
appended takes it at once, and a runner started again takes it anew for each claim it adopts; the episode's record and
the adoption are appended under it. So whoever took it last shuts out every attempt before: a runner that paused past
its claim's lapse while another claimed the episode again finds its record refused (`Fenced`), and an adoption is
refused once a newer attempt has taken the fence. The runner tells its recorder (`Recorded`: the gateway's endpoints)
which attempt each run plays, under that fence, before the run starts or is adopted; the run's turns are appended under
it (`rollout_train.gateway`), so a stale attempt can record nothing more once a newer one took the fence.

Several runners, on one machine or many, share the work the same way: which machine plays a group's episodes is only
a matter of where runners are.

A runner claims nothing of a paused run (`rollout_train.settings.paused`: the run, or the run it is played for), and
looks at whether each is paused every time it looks for work, whether or not it has room: an episode it plays already
plays out. Each beat says which of the runs it serves are paused (`paused`), and it beats at once when that changes.

A run's sandboxes (`rollout.harness.sandboxes`) are acquired under its claim's key, `RUN/GROUP/EPISODE/ATTEMPT`: a
retried acquire gets the same sandbox, a new attempt a new one, and a sandbox's lease ends with the claim
(`rollout_train.sandboxes`). A runner claims an episode only while the pools of the sandboxes its program declares
have room for them.
"""

import asyncio
import contextlib
import logging
import re
import time
from collections import Counter
from collections.abc import AsyncGenerator, Callable, Collection, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.contracts import RunEvent, new_run_id
from rollout.harness.blobs import Blobs
from rollout.harness.remote import remote_pool
from rollout.harness.runner import (
    ProgramReference,
    RunBinding,
    RunHandle,
    Runner,
    RunSpecification,
    instantiate,
    with_row,
)
from rollout.harness.sandboxes import Pool, PoolBinding, SandboxLost
from rollout_train.gateway.client import Attempt
from rollout_train.ledger import Fence, Fenced, Ledger
from rollout_train.presence import Beat, Presence, alive
from rollout_train.record import GROUPS, RESULTS, mapping, newest_record, runs_in, scope, table
from rollout_train.recorder import Segment
from rollout_train.rollouts.episodes import Episode, Outcome, Record, assemble, loaded, stored
from rollout_train.settings import desired_settings_of, paused

PLANS, CLAIMS, EPISODES, INTERRUPTED, ADOPTED = "plans", "claims", "episodes", "interrupted", "adopted"
"""A run's tables beside its groups: how its episodes are played, who plays which, the episodes that ended, the
attempts cut short, and the attempts a runner started again took up (`GROUP/EPISODE/ATTEMPT/FENCE`)."""
CLOSED = "its runner closed"
"""Why an attempt was cut short when its runner closed: it is played again."""
LAPSED = "its claim lapsed while its runner was stopped"
"""Why a run its runner found on starting again was cut short: it is played again."""
LOST = "its sandboxes did not outlive its runner"
"""Why a run its runner adopted was cut short: what it was playing in is gone, so it is played again."""
SUPERSEDED = "another took its episode's fence: its record was refused"
"""Why an attempt that ended was not recorded (a newer attempt claimed its episode meanwhile, say)."""
RELEASED = "its claim lapsed, and its pool released its sandboxes"
"""Why a pool's keeper cut an attempt short (`rollout_train.sandboxes`): it is played again."""

logger = logging.getLogger(__name__)


def runner_scope(name: str) -> str:
    """The scope whose fence a runner holds while it plays: a runner started again takes it anew."""
    return f"runners/{name}"


def episode_scope(run: str, episode: str) -> str:
    """The scope of an episode's fence (`runs/RUN/episodes/GROUP/EPISODE`): taken by the runner of each attempt once
    its claim is appended (or as it adopts it), and what the attempt's record is appended under, so that the newest
    attempt shuts out the ones before."""
    return f"{scope(run)}/episodes/{episode}"


def of_episode(name: str) -> bool:
    """Whether a scope is an episode's (`episode_scope`)."""
    return _EPISODE_SCOPE.fullmatch(name) is not None


_EPISODE_SCOPE = re.compile(r"runs/.+/episodes/\d+/\d+")


@dataclass(frozen=True)
class Claims:
    """A run's claims as read at one moment, with what says whether each holds: the attempts cut short, the episodes
    that ended, and the adoptions (`KEY/FENCE`)."""

    made: Mapping[str, Mapping[str, Any]]
    cut: Collection[str] = ()
    done: Collection[str] = ()
    adopted: Collection[str] = ()

    @classmethod
    async def read(cls, ledger: Ledger, run: str) -> "Claims":
        claims = await ledger.read(table(run, CLAIMS))
        return cls(
            {key: mapping(claim) for key, claim in claims.items()},
            set(await ledger.read(table(run, INTERRUPTED))),
            set(await ledger.read(table(run, EPISODES))),
            set(await ledger.read(table(run, ADOPTED))),
        )

    @cached_property
    def latest(self) -> dict[str, int]:
        """Each claimed episode's latest attempt, by `GROUP/EPISODE`."""
        found: dict[str, int] = {}
        for key in self.made:
            episode, attempt = key.rsplit("/", 1)
            found[episode] = max(found.get(episode, 0), int(attempt))
        return found

    def is_latest(self, key: str) -> bool:
        """Whether the claim under `key` is its episode's latest attempt."""
        episode, attempt = key.rsplit("/", 1)
        return self.latest.get(episode) == int(attempt)

    def holds(
        self, key: str, fences: Mapping[str, int], beats: Mapping[str, Beat] | None, me: str | None = None
    ) -> bool:
        """Whether the claim made under `key` holds: its episode's latest attempt, not noted as cut short, made (or
        adopted) under its runner's newest fence, and (where runners beat) its runner beating; a runner's own claims
        (`me`) hold for it without its beat."""
        claim = self.made.get(key)
        if claim is None or key in self.cut or not self.is_latest(key):
            return False
        runner = str(claim["runner"])
        newest = fences.get(runner_scope(runner))
        if newest != claim["fence"] and f"{key}/{newest}" not in self.adopted:
            return False
        return beats is None or runner == me or alive(beats.get(runner))

    def holding(self, fences: Mapping[str, int], beats: Mapping[str, Beat] | None, me: str | None = None) -> set[str]:
        """The keys of the claims that hold, of episodes that have not ended."""
        return {
            key for key in self.made if key.rsplit("/", 1)[0] not in self.done and self.holds(key, fences, beats, me)
        }


async def holding(
    ledger: Ledger, run: str, fences: Mapping[str, int], beats: Mapping[str, Beat] | None, me: str | None = None
) -> set[str]:
    """The keys of a run's claims that hold, of episodes that have not ended."""
    return (await Claims.read(ledger, run)).holding(fences, beats, me)


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
    return [await loaded(Record.from_json(mapping(records[key])), blobs) for key in keys]


async def episodes_of(
    ledger: Ledger, blobs: Blobs, run: str, group: int, count: int, *, every: float = 0.5
) -> list[Episode]:
    """A group's episodes once all `count` have ended, waiting for them."""
    while (found := await ended(ledger, blobs, run, group, count)) is None:  # noqa: ASYNC110 (another process writes)
        await asyncio.sleep(every)
    return found


class Recorded(Protocol):
    """What a runner needs of what records its runs' samples (`rollout_train.gateway.GatewayEndpoints`)."""

    def admit(self, run_id: str, attempt: Attempt) -> None:
        """Say which attempt a run plays (its episode's fence), before it starts or is adopted."""
        ...

    async def reaches(self, run: str, binding: RunBinding) -> bool:
        """Whether every recorded model of a run's binding can be sampled now."""
        ...

    async def sessions(self, run: str, run_id: str) -> dict[str, list[Segment]]:
        """What each model slot of a run recorded, by slot."""
        ...

    def forget(self, run_id: str) -> None: ...


class Hooks(Protocol):
    """Watch a runner (episodes as they start and end) or a run (its results and steps)."""

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        """`event["kind"]` is `started` or `ended` (an episode, by a runner: an adopted one is started again), or the
        run's `result`, `step` or `published`."""
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
    runs it can serve (whose models its recorder samples, whose imports are among `imports` and whose local
    pools are among `pools`), and of `runs` only, if given. An episode is claimed only while the pools of its
    sandboxes have room for them, and its run's sandboxes are leased under its claim. `guard` is called before
    claiming and raises to wait (a machine short of memory, say). With `presence`, it beats every `beating` seconds,
    with what `about` says of its machine besides its places, how many it plays and how full its pools are, and a
    claim holds only while its runner beats. Over a runner whose runs survive it (`resumes`), closing leaves its runs
    to be resumed, and starting again adopts them (`prepare`)."""

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
    pools: Mapping[str, Pool] = field(default_factory=dict[str, Pool])
    """The sandbox pools its runner has, by the name a binding gives as `local`."""
    every: float = 0.5
    """Seconds between looks for work while nothing ends."""
    beating: float = 15.0
    """Seconds between beats."""
    _fence: Fence | None = None
    _playing: dict[str, asyncio.Task[None]] = field(default_factory=dict[str, asyncio.Task[None]])
    _news: asyncio.Event = field(default_factory=asyncio.Event)
    _kinds: dict[tuple[str, int], Counter[str]] = field(default_factory=dict[tuple[str, int], Counter[str]])
    """The kinds of sandbox each group's program declares, and how many of each."""
    _adopting: list[tuple[Open, str, RunHandle, Fence]] = field(
        default_factory=list[tuple[Open, str, RunHandle, Fence]]
    )
    _lapsed: list[str] = field(default_factory=list[str])
    """Runs found on starting again whose claims lapsed: cancelled once the runner is serving."""
    _paused: frozenset[str] = frozenset()
    """The runs it serves that were paused when it last looked."""

    @property
    def resumes(self) -> bool:
        """Whether its runner's runs survive it (`Runner.resumes`)."""
        return bool(getattr(self.runner, "resumes", False))

    async def prepare(self) -> None:
        """Take its fence, beat, and adopt what it finds of its runs: over a runner whose runs survive it, call this
        before launching the runner, so that the runs it recovers find their claims holding. `serve` calls it if it
        has not been."""
        if self._fence is not None:
            return
        self._fence = await self.ledger.take(runner_scope(self.name))
        if self.presence is not None:
            await self._beat()  # (alive before it claims or adopts anything)
        if self.resumes:
            await self._adopt()

    async def serve(self) -> None:
        """Claim and play episodes until cancelled; what is playing then is cut short and noted (over a runner whose
        runs survive it, left to be resumed)."""
        await self.prepare()
        beating: asyncio.Task[None] | None = None
        if self.presence is not None:
            beating = asyncio.create_task(self._beats())
        for each, key, handle, fence in self._adopting:
            self._follow(key, self._watch(each, key, handle, fence, adopted=True))
        for run_id in self._lapsed:
            self._follow(f"lapsed/{run_id}", self._cut_short(run_id))
        self._adopting, self._lapsed = [], []
        try:
            while True:
                room = self.places - len(self._playing)
                claiming = room > 0 and self._fits()
                found = await self._look(claiming)  # (looked at whatever its room: which runs are paused)
                if claiming:
                    free: dict[str, int] = {}  # each pool's room, asked once a look
                    plans: dict[str, Plan] = {}  # each run's plan, read once a look
                    for each in found:
                        if room == 0:
                            break
                        needs = await self._needs(each, plans)
                        if not await self._room(needs, free):
                            continue
                        if await self._claim(each):
                            room -= 1
                            for pool, count in needs.items():
                                free[pool] -= count
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

    async def beat(self) -> None:
        """Beat now, beside the beats every `beating` seconds: after what it says of itself changed (a channel serves
        a new checkpoint, say), so that whoever reads the beats does not wait for the next."""
        if self.presence is not None:
            await self._beat()

    async def _beat(self) -> None:
        assert self.presence is not None
        said: Mapping[str, JsonValue] = await asyncio.to_thread(self.about) if self.about is not None else {}
        about: dict[str, JsonValue] = {**said, "places": self.places, "playing": len(self._playing)}
        if self._paused:
            about["paused"] = list[JsonValue](sorted(self._paused))
        if self.pools:
            about["pools"] = {name: (await pool.capacity()).to_json() for name, pool in self.pools.items()}
        await self.presence.beat(self.name, about)

    async def open(self) -> list[Open]:
        """The episodes nobody plays now, of the runs this runner serves that are not paused, oldest group first. Which
        of them are paused is noted, and said at once in a beat when it changed."""
        return await self._look(True)

    async def _look(self, claiming: bool) -> list[Open]:
        """Note which of the runs this runner serves are paused (beating at once when that changed), and, `claiming`,
        find the episodes nobody plays now of the others (`open`)."""
        fences = await self.ledger.fences() if claiming else {}
        beats = {beat.runner: beat for beat in await self.presence.beats()} if self.presence and claiming else None
        desired = desired_settings_of(self.ledger)
        found: list[Open] = []
        halted: set[str] = set()
        for run in await runs_in(self.ledger):
            if self.runs is not None and run not in self.runs:
                continue
            plans = await self.ledger.read(table(run, PLANS))
            if not plans or not await self._serves(run, Plan.from_json(newest_record(plans))):
                continue
            if await paused(self.ledger, run, desired):
                halted.add(run)
                continue
            if not claiming:
                continue
            groups = await self.ledger.read(table(run, GROUPS))
            results = await self.ledger.read(table(run, RESULTS))
            claims = await Claims.read(self.ledger, run)
            held = {key.rsplit("/", 1)[0] for key in claims.holding(fences, beats, self.name)}
            for key, group in groups.items():
                record = mapping(group)
                count = record.get("episodes")
                if key in results or not isinstance(count, int):
                    continue
                for number in range(1, count + 1):
                    episode = f"{key}/{number}"
                    if episode in claims.done or episode in held:
                        continue
                    decided = float(str(record.get("decided") or 0.0))
                    found.append(Open(run, int(key), number, claims.latest.get(episode, 0) + 1, decided))
        if frozenset(halted) != self._paused:
            self._paused = frozenset(halted)
            with contextlib.suppress(Exception):  # (a beat missed: the next says it)
                await self.beat()
        return sorted(found, key=lambda each: (each.decided, each.run, each.group, each.number))

    async def _serves(self, run: str, played: Plan) -> bool:
        local = {binding.local for binding in played.binding.imports.values() if binding.local}
        pools = {binding.local for binding in played.binding.pools.values() if binding.local}
        if not (local <= set(self.imports) and pools <= set(self.pools)):
            return False
        return await self.recorder.reaches(run, played.binding)

    async def _needs(self, each: Open, plans: dict[str, Plan]) -> Counter[str]:
        """The sandboxes an episode's run will acquire, as how many from each pool (by its binding: a local name or
        a URL). `plans` keeps the runs' plans, as read."""
        if each.run not in plans:
            written = await self.ledger.read(table(each.run, PLANS))
            plans[each.run] = Plan.from_json(newest_record(written))
        played = plans[each.run]
        if not played.binding.pools:
            return Counter[str]()
        kinds = self._kinds.get((each.run, each.group))
        if kinds is None:
            group = mapping((await self.ledger.read(table(each.run, GROUPS)))[str(each.group)])
            try:
                program = instantiate(with_row(played.program, group["parameters"]))
                kinds = Counter(spec.kind for spec in program.sandboxes().values())
            except Exception:  # (a program that cannot be made fails when its run starts: an episode like any other)
                kinds = Counter[str]()
            self._kinds[(each.run, each.group)] = kinds
        needs: Counter[str] = Counter()
        for kind, count in kinds.items():
            binding = played.binding.pools.get(kind)
            if binding is not None:
                needs[_named(binding)] += count
        return needs

    async def _room(self, needs: Mapping[str, int], free: dict[str, int]) -> bool:
        """Whether every pool has room for what is needed of it (`free` keeps each pool's room, as asked)."""
        for name, count in needs.items():
            if name not in free:
                pool = self.pools.get(name) or (remote_pool(name) if name.startswith(("http://", "https://")) else None)
                try:
                    free[name] = (await pool.capacity()).free if pool is not None else 0
                except Exception:  # (a pool that cannot be asked has no room now; asked again at the next look)
                    free[name] = 0
            if free[name] < count:
                return False
        return True

    def _fits(self) -> bool:
        if self.guard is None:
            return True
        try:
            self.guard()
        except Exception:  # (not now: there is not enough of something; looked at again in a while)
            return False
        return True

    async def _claim(self, each: Open) -> bool:
        assert self._fence is not None
        key = f"{each.group}/{each.number}/{each.attempt}"
        run_id = new_run_id()  # (named in the claim: a runner started again finds the run by it)
        claim: dict[str, JsonValue] = {
            "runner": self.name,
            "fence": self._fence.number,
            "at": round(time.time(), 1),
            "run_id": run_id,
        }
        appending = asyncio.ensure_future(self.ledger.append(table(each.run, CLAIMS), key, claim, self._fence))
        try:
            if not await asyncio.shield(appending):
                return False  # another runner claimed this attempt first
        except asyncio.CancelledError:  # the runner is closing: a claim it made is noted cut short, as in `_play`
            if not self.resumes:
                with contextlib.suppress(Exception):
                    if await appending:
                        await self._interrupt(each, key, CLOSED)
            raise
        self._follow(key, self._play(each, key, run_id))
        return True

    def _follow(self, key: str, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._playing[key] = task
        task.add_done_callback(lambda _: self._done(key))

    def _done(self, key: str) -> None:
        self._playing.pop(key, None)
        self._news.set()

    async def _adopt(self) -> None:
        """Find the runs of this runner's claims, as its runner recovered them: adopt those whose claims are still
        their episodes' latest attempts, not cut short, of episodes without a record (made under any fence this runner
        held before: one that died before adopting, or whose fence was taken twice, adopts them all the same); cut
        short the others still going. Each episode's fence is taken before its claims are read again and the claim is
        adopted under it: an attempt claimed before the take is seen, and one claimed after shuts the adoption out."""
        assert self._fence is not None
        for run in await runs_in(self.ledger):
            if self.runs is not None and run not in self.runs:
                continue
            claims = await Claims.read(self.ledger, run)
            found: list[tuple[str, RunHandle]] = []
            for key, made in claims.made.items():
                ended = key.rsplit("/", 1)[0] in claims.done
                if made["runner"] != self.name or key in claims.cut or ended or "run_id" not in made:
                    continue
                try:
                    found.append((key, self.runner.run(str(made["run_id"]))))
                except KeyError:
                    continue  # (not a run its runner has: its claim lapses)
            fences = {
                key: await self.ledger.take(episode_scope(run, key.rsplit("/", 1)[0]))
                for key, _ in found
                if claims.is_latest(key)
            }
            now = await Claims.read(self.ledger, run) if fences else claims  # (as it is once the fences are taken)
            for key, handle in found:
                episode, attempt = key.rsplit("/", 1)
                fence = fences.get(key)
                if fence is not None and key not in now.cut and episode not in now.done and now.is_latest(key):
                    noted: dict[str, JsonValue] = {"run_id": handle.run_id, "at": round(time.time(), 1)}
                    try:
                        await self.ledger.append(table(run, ADOPTED), f"{key}/{self._fence.number}", noted, fence)
                    except Fenced:  # (a newer attempt took the episode's fence meanwhile)
                        pass
                    else:
                        group, number = (int(part) for part in episode.split("/"))
                        self.recorder.admit(handle.run_id, Attempt(run, fence, episode, int(attempt)))
                        self._adopting.append((Open(run, group, number, int(attempt), 0.0), key, handle, fence))
                        continue
                if not handle.done:
                    why: dict[str, JsonValue] = {"why": LAPSED, "at": round(time.time(), 1)}
                    await self.ledger.append(table(run, INTERRUPTED), key, why, self._fence)
                    self._lapsed.append(handle.run_id)

    async def _cut_short(self, run_id: str) -> None:
        with contextlib.suppress(Exception):
            await self.runner.cancel(run_id, reason=LAPSED)
        self.recorder.forget(run_id)

    async def _play(self, each: Open, key: str, run_id: str) -> None:
        """Play a claimed attempt to its record. Cut short by the runner closing, anywhere until it is recorded, the
        attempt is noted and played again by someone; over a runner whose runs survive it, the run is left to be
        adopted."""
        try:
            # Only the claim's winner takes its episode's fence: no attempt before this one can record the episode now.
            fence = await self.ledger.take(episode_scope(each.run, f"{each.group}/{each.number}"))
            plans = await self.ledger.read(table(each.run, PLANS))
            played = Plan.from_json(newest_record(plans))
            group = mapping((await self.ledger.read(table(each.run, GROUPS)))[str(each.group)])
            program = with_row(played.program, group["parameters"])
            specification = RunSpecification(program=program, binding=played.binding)
            labels = {"run": each.run, "group": str(each.group), "episode": str(each.number)}
            self.recorder.admit(run_id, Attempt(each.run, fence, f"{each.group}/{each.number}", each.attempt))
            try:
                handle = await self.runner.start(specification, run_id=run_id, labels=labels, lease=f"{each.run}/{key}")
            except Exception as error:  # a run that cannot start is a failed episode like any other
                self.recorder.forget(run_id)
                detail = f"{type(error).__name__}: {error}"
                failed = Episode(each.run, each.group, each.number, "", labels, Outcome.FAILED, detail)
                await self._ended(each, key, failed, [], fence)
                return
            await self._watch(each, key, handle, fence)
        except asyncio.CancelledError:  # the runner is closing
            if self.resumes:  # (the run is resumed when the runner starts again, and adopted)
                raise
            with contextlib.suppress(Exception):
                await self.runner.cancel(run_id, reason=CLOSED)
            self.recorder.forget(run_id)
            await self._interrupt(each, key, CLOSED)  # the attempt is noted, and played again by someone
            raise

    async def _watch(self, each: Open, key: str, handle: RunHandle, fence: Fence, *, adopted: bool = False) -> None:
        """Follow a run to its end, and record its episode under its episode's `fence`, with what the gateway recorded
        of its samples (an adopted run's too, from before its runner stopped). An adopted run whose sandboxes are gone
        is cut short, to be played again."""
        assert self._fence is not None
        self._tell("started", run=each.run, group=each.group, episode=each.number, run_id=handle.run_id)
        events: list[RunEvent] = [event async for event in handle.events()]
        try:
            segments = await self.recorder.sessions(each.run, handle.run_id)
        finally:
            self.recorder.forget(handle.run_id)
        episode = assemble(events, segments, run=each.run, group=each.group, number=each.number)
        if adopted and episode.outcome is Outcome.FAILED and SandboxLost.__name__ in str(episode.detail):
            await self._interrupt(each, key, LOST)
            return
        await self._ended(each, key, episode, events, fence)

    async def _interrupt(self, each: Open, key: str, why: str) -> None:
        assert self._fence is not None
        cut: dict[str, JsonValue] = {"why": why, "at": round(time.time(), 1)}
        with contextlib.suppress(Exception):
            await asyncio.shield(self.ledger.append(table(each.run, INTERRUPTED), key, cut, self._fence))

    async def _ended(self, each: Open, key: str, episode: Episode, events: Sequence[RunEvent], fence: Fence) -> None:
        """Record an episode under its episode's fence. Once another took the fence (a newer attempt) the record is
        refused, and the attempt is noted cut short: its claim holds no more."""
        record = await stored(episode, events, self.blobs)
        try:
            await self.ledger.append(table(each.run, EPISODES), f"{each.group}/{each.number}", record.to_json(), fence)
        except Fenced:
            logger.info("%s/%s ended after another took its episode's fence: it is not recorded", each.run, key)
            await self._interrupt(each, key, SUPERSEDED)
            return
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


def _named(binding: PoolBinding) -> str:
    """A pool as a binding names it: its local name, or its URL."""
    return binding.local or str(binding.url)
