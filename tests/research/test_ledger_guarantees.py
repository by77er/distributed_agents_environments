"""The ledger's guarantees under concurrency and failure (docs/research/ledger-guarantees.md).

Tests that pass show a guarantee holding under real concurrency: processes sharing a ledger of files, threads with
connections of their own to SQLite and to Postgres. Tests marked `xfail(strict=True)` show a guarantee that does not
hold: each reason says what goes wrong, and a fix flips the test.

Where a failure needs a crash or a pause at one exact moment, the test builds the state that moment leaves (a line
half written, a beat that stopped) or runs the other party's step at that moment (a ledger or store wrapped so that a
call runs another writer's step first). Every step is the code's own.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.environment import Start
from rollout.harness import Program, RunContext, SandboxPool, SandboxSpec, register
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout.testing import FakeSandboxes
from rollout_train.checkpoints import Checkpoints, Retention, new_id
from rollout_train.evals import make_suite, suite_of
from rollout_train.launcher import Launcher
from rollout_train.launches import CLAIMED, STOPPED, STOPPING, Asked, FileLaunches, Launch
from rollout_train.ledger import Fence, Fenced, FileLedger, Ledger
from rollout_train.presence import STALE, FilePresence
from rollout_train.record import scope, table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import EpisodeRunner, playing
from rollout_train.rollouts.episodes import Outcome, Record
from rollout_train.rollouts.scheduler import ADOPTED, CLAIMS, EPISODES, INTERRUPTED, holding, runner_scope
from rollout_train.sandboxes import admits, leases_of, sweep
from rollout_train.testing import plain_channel
from tests.rollout_train.test_sandboxes import ask

pytest.importorskip("rollout_durable")
from rollout_train.database import DatabaseLedger

BOX = SandboxSpec(kind="fake")
GATES: dict[str, asyncio.Event] = {}
"""Where `KeyGated` runs wait, by their sandbox's lease key (so by attempt)."""


@pytest.fixture(autouse=True)
def _fresh_gates() -> None:
    GATES.clear()


class KeyGated(Program):
    """Waits at the gate of its sandbox's lease key (one per attempt), then asks its box who it is, and says."""

    def __init__(self, parameters: Mapping[str, JsonValue]) -> None:
        del parameters

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"box": BOX}

    async def main(self, run: RunContext) -> None:
        key = run.sandbox("box").lease.key
        await GATES.setdefault(key, asyncio.Event()).wait()
        await run.sandbox("box").call("describe")
        run.reward(1.0)
        await run.emit("result", {"solved": True, "key": key})


register(KeyGated)


def gate(key: str) -> asyncio.Event:
    return GATES.setdefault(key, asyncio.Event())


def beat_at(directory: Path, runner: str, at: float) -> None:
    """Set a runner's newest beat in a ledger's `presence.json` to `at` (what a beat written then would say)."""
    path = directory / "presence.json"
    beats: list[dict[str, Any]] = json.loads(path.read_text())
    for each in beats:
        if each["runner"] == runner:
            each["at"] = at
    path.write_text(json.dumps(beats))


async def beats_of(presence: FilePresence) -> dict[str, Any]:
    return {beat.runner: beat for beat in await presence.beats()}


async def until(condition: Callable[[], Awaitable[bool]], seconds: float = 10.0) -> None:
    async with asyncio.timeout(seconds):
        while not await condition():  # noqa: ASYNC110 (other tasks write)
            await asyncio.sleep(0.01)


class Hooked:
    """A ledger that runs `after_take[scope]`, `after_read[table]` or `after_append[(table, key)]` once that call
    has returned: another writer's step, at exactly that moment (each hook runs once)."""

    def __init__(self, inner: Ledger) -> None:
        self.inner = inner
        self.after_take: dict[str, Callable[[], Awaitable[None]]] = {}
        self.after_read: dict[str, Callable[[], Awaitable[None]]] = {}
        self.after_append: dict[tuple[str, str], Callable[[], Awaitable[None]]] = {}

    async def take(self, scope: str) -> Fence:
        fence = await self.inner.take(scope)
        if (hook := self.after_take.pop(scope, None)) is not None:
            await hook()
        return fence

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        appended = await self.inner.append(table, key, record, fence)
        if (hook := self.after_append.pop((table, key), None)) is not None:
            await hook()
        return appended

    async def read(self, table: str) -> dict[str, JsonValue]:
        found = await self.inner.read(table)
        if (hook := self.after_read.pop(table, None)) is not None:
            await hook()
        return found

    async def tables(self) -> list[str]:
        return await self.inner.tables()

    async def fences(self) -> dict[str, int]:
        return await self.inner.fences()


# --- Fencing: what holds -------------------------------------------------------------------------------------------

TAKER = """
import asyncio, json, sys
from pathlib import Path
from rollout_train.ledger import FileLedger, Fenced

async def main() -> None:
    ledger = FileLedger(Path(sys.argv[1]))
    me, results = sys.argv[2], []
    for index in range(int(sys.argv[3])):
        fence = await ledger.take("run")
        try:
            appended = await ledger.append("records", f"{me}-{index}", {"by": me}, fence)
        except Fenced:
            appended = None
        results.append([fence.number, appended])
    print(json.dumps(results))

asyncio.run(main())
"""


def test_processes_sharing_a_ledger_of_files_take_distinct_fences_and_stale_appends_are_refused(
    tmp_path: Path,
) -> None:
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", TAKER, str(tmp_path / "ledger"), f"p{index}", "40"],
            stdout=subprocess.PIPE,
            text=True,
        )
        for index in range(4)
    ]
    results = [json.loads(process.communicate(timeout=120)[0]) for process in processes]
    numbers = [number for each in results for number, _ in each]
    assert sorted(numbers) == list(range(1, 161))  # every take a number of its own
    written = [json.loads(line) for line in (tmp_path / "ledger" / "records.jsonl").read_text().splitlines()]
    appended = sum(1 for each in results for _, ok in each if ok)
    assert len(written) == appended
    # An append under fence n succeeded only if no take after n came before it: so the fences of the records
    # appended, in the order they were appended, never go down.
    fences = [each["fence"] for each in written]
    assert fences == sorted(fences)


def _in_threads(count: int, work: Callable[[int], Any]) -> list[Any]:
    results: list[Any] = [None] * count
    barrier = threading.Barrier(count)

    def run(index: int) -> None:
        barrier.wait()
        results[index] = work(index)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def _takes_and_appends(url: str, me: int, rounds: int) -> list[tuple[int, bool | None]]:
    ledger = DatabaseLedger(url)  # a connection pool of its own: as another process would have

    async def work() -> list[tuple[int, bool | None]]:
        done: list[tuple[int, bool | None]] = []
        for index in range(rounds):
            fence = await ledger.take("run")
            try:
                done.append((fence.number, await ledger.append("records", f"{me}-{index}", {"by": me}, fence)))
            except Fenced:
                done.append((fence.number, None))
        return done

    try:
        return asyncio.run(work())
    finally:
        ledger.close()


async def _fences_in_order(url: str) -> list[int]:
    ledger = DatabaseLedger(url)
    try:

        def rows(connection: Any) -> list[tuple[Any, ...]]:
            from rollout_durable.database import fetch_all

            return fetch_all(connection, "SELECT fence FROM ledger_records WHERE name = 'records' ORDER BY position")

        return [int(fence) for (fence,) in ledger.database.read(rows)]
    finally:
        ledger.close()


@pytest.mark.parametrize("kind", ["sqlite", "postgres"])
async def test_writers_on_connections_of_their_own_take_distinct_fences_and_stale_appends_are_refused(
    tmp_path: Path, kind: str, request: pytest.FixtureRequest
) -> None:
    url = f"sqlite:///{tmp_path / 'ledger.db'}" if kind == "sqlite" else request.getfixturevalue("postgres")
    DatabaseLedger(url).close()  # (the tables, made once)
    results = await asyncio.to_thread(_in_threads, 4, lambda me: _takes_and_appends(url, me, 30))
    numbers = [number for each in results for number, _ in each]
    assert sorted(numbers) == list(range(1, 121))
    fences = await _fences_in_order(url)
    assert len(fences) == sum(1 for each in results for _, ok in each if ok)
    assert fences == sorted(fences)  # no append committed under a fence older than one taken before it


# --- Fencing and keys: what does not hold ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="FileLedger: a writer that died mid-line leaves no newline, so the next append is glued to the torn line "
    "and silently lost although append returned True; its key is then free, and a second, different record is "
    "accepted as the first under it",
)
async def test_an_append_after_a_torn_line_is_kept(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("runners/a")
    with (tmp_path / "claims.jsonl").open("a") as file:  # a writer that died mid-line (a large record, a full disk)
        file.write(json.dumps({"key": "1/1/1", "fence": 1, "record": {"runner": "a"}})[:25])
    assert await ledger.append("claims", "1/1/2", {"runner": "a"}, fence)  # acknowledged
    other = await FileLedger(tmp_path).take("runners/b")
    second = await ledger.append("claims", "1/1/2", {"runner": "b"}, other)
    assert not second and await ledger.read("claims") == {"1/1/2": {"runner": "a"}}


@pytest.mark.xfail(
    strict=True,
    reason="FileLedger rewrites fences.json in place (truncate, then write): a crash between the two leaves an "
    "empty file, and every take and append after it raises until someone repairs it by hand",
)
async def test_a_crash_while_taking_a_fence_leaves_the_ledger_usable(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    await ledger.append("groups", "1", {}, fence)
    (tmp_path / "fences.json").write_text("")  # what a crash after the truncate and before the write leaves
    newer = await ledger.take("run")
    assert newer.number > fence.number


def _positions(url: str, name: str) -> list[int]:
    from rollout_durable.database import Database, fetch_all

    database = Database(url)
    try:
        query = "SELECT position FROM ledger_records WHERE name = :name ORDER BY position"
        return [int(position) for (position,) in database.read(lambda c: fetch_all(c, query, {"name": name}))]
    finally:
        database.close()


@pytest.mark.xfail(
    strict=True,
    reason="DatabaseLedger on Postgres: an append computes its position as MAX(position) + 1 under the lock of its "
    "fence's scope, not of its table; two writers of different scopes appending to one table at once (two runners' "
    "claims, two runs' checkpoints) read the same MAX under READ COMMITTED and take the same position, so 'in the "
    "order they were appended' is not defined",
)
async def test_appends_of_two_scopes_to_one_table_get_distinct_positions_on_postgres(postgres: str) -> None:
    first, second = DatabaseLedger(postgres), DatabaseLedger(postgres)
    try:
        a, b = await first.take("runners/a"), await second.take("runners/b")
        write = first.database.write

        def with_another_append_before_commit(change: Callable[[Any], Any], *, exclusive: str | None = None) -> Any:
            def changed(connection: Any) -> Any:
                result = change(connection)
                if exclusive == "ledger:runners/a":  # the other runner appends while this transaction is open
                    done = threading.Thread(target=lambda: asyncio.run(second.append("claims", "1/2/1", {}, b)))
                    done.start()
                    done.join()
                return result

            return write(changed, exclusive=exclusive)

        first.database.write = with_another_append_before_commit  # type: ignore[method-assign]
        await first.append("claims", "1/1/1", {}, a)
        positions = _positions(postgres, "claims")
        assert len(positions) == 2 and len(set(positions)) == 2
    finally:
        first.close()
        second.close()


# --- Claims ---------------------------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="holds() judges a claim by its runner's fence and beat only: a runner that stopped beating (paused, "
    "partitioned) and beats again has its old claim hold again beside the newer attempt another runner claimed "
    "meanwhile, and a pool beside the ledger admits its key again",
)
async def test_a_lapsed_claim_does_not_hold_again_beside_a_newer_attempt(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    await ask(ledger, {1: ({}, 1)})
    a = await ledger.take(runner_scope("a"))
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "a", "fence": a.number}, a)
    await beats.beat("a", {})
    beat_at(ledger.directory, "a", time.time() - STALE - 10)  # a paused: its beats stopped
    b = await ledger.take(runner_scope("b"))
    await beats.beat("b", {})
    assert await holding(ledger, "train", await ledger.fences(), await beats_of(beats)) == set()  # open again
    await ledger.append(table("train", CLAIMS), "1/1/2", {"runner": "b", "fence": b.number}, b)  # b claims it
    await beats.beat("a", {})  # a resumes
    held = await holding(ledger, "train", await ledger.fences(), await beats_of(beats))
    assert held == {"1/1/2"}
    assert not await admits(ledger, beats)("train/1/1/1/box")


@dataclass
class Resumable:
    """A runner whose runs survive it, as `EpisodeRunner._adopt` asks of one: every run it is asked for is going."""

    resumes: bool = True
    asked: list[str] = field(default_factory=list[str])

    def run(self, run_id: str) -> Any:
        self.asked.append(run_id)

        @dataclass
        class Handle:
            run_id: str
            done: bool = False

        return Handle(run_id)


@dataclass
class NoRecorder:
    channels: Mapping[str, Any] = field(default_factory=lambda: {"policy": None})

    def sessions(self, run_id: str) -> dict[str, list[Any]]:
        return {}

    def forget(self, run_id: str) -> None:
        pass


@pytest.mark.xfail(
    strict=True,
    reason="EpisodeRunner._adopt checks 'no later attempt claimed since' and then appends its adoption, with the "
    "claim not holding in between (its fence moved on): a runner that claims the next attempt in that window and "
    "the adoption both stand, and two claims of one episode hold",
)
async def test_an_adoption_and_a_new_attempt_never_both_hold(tmp_path: Path) -> None:
    files = FileLedger(tmp_path / "ledger")
    beats = FilePresence(files.directory)
    await ask(files, {1: ({}, 1)})
    before = await files.take(runner_scope("here"))  # the runner before it stopped: its claim, its run
    claim: JsonValue = {"runner": "here", "fence": before.number, "run_id": "run_1", "at": time.time()}
    await files.append(table("train", CLAIMS), "1/1/1", claim, before)
    await beats.beat("here", {})
    ledger = Hooked(files)
    boxes = SandboxPool(FakeSandboxes())
    other = EpisodeRunner(
        "other", files, LocalRunner(), NoRecorder(), FileBlobStore(tmp_path / "blobs"), places=1, presence=beats,
        pools={"boxes": boxes},
    )  # fmt: skip

    async def the_other_runner_looks_now() -> None:
        (found,) = await other.open()  # the claim does not hold now: its runner's fence moved on
        assert found.attempt == 2
        fence = await files.take(runner_scope("other"))
        mine: JsonValue = {"runner": "other", "fence": fence.number, "run_id": "run_2", "at": time.time()}
        await files.append(table("train", CLAIMS), "1/1/2", mine, fence)  # what `_claim` appends
        await beats.beat("other", {})

    # `_adopt` has read the run's claims (attempt 1 is the latest) and is about to append its adoption.
    ledger.after_read[table("train", ADOPTED)] = the_other_runner_looks_now
    resumable: Any = Resumable()
    again = EpisodeRunner(
        "here", ledger, resumable, NoRecorder(), FileBlobStore(tmp_path / "blobs"), places=1, presence=beats
    )
    await again.prepare()  # takes its fence anew, beats, adopts
    held = await holding(files, "train", await files.fences(), await beats_of(beats))
    assert len(held) == 1, f"two claims of one episode hold: {sorted(held)}"


@pytest.mark.xfail(
    strict=True,
    reason="EpisodeRunner._adopt takes the fence its claims held under to be its new fence less one: a runner that "
    "died after taking its fence and before adopting (or whose take was applied twice, a retried request) finds its "
    "claims made two fences back, judges them lapsed, and cuts its runs short",
)
async def test_a_runner_that_died_while_preparing_adopts_its_runs_when_started_again(tmp_path: Path) -> None:
    files = FileLedger(tmp_path / "ledger")
    beats = FilePresence(files.directory)
    await ask(files, {1: ({}, 1)})
    before = await files.take(runner_scope("here"))
    claim: JsonValue = {"runner": "here", "fence": before.number, "run_id": "run_1", "at": time.time()}
    await files.append(table("train", CLAIMS), "1/1/1", claim, before)
    await beats.beat("here", {})
    await files.take(runner_scope("here"))  # started again, it died before it adopted anything
    resumable: Any = Resumable()
    again = EpisodeRunner(
        "here", files, resumable, NoRecorder(), FileBlobStore(tmp_path / "blobs"), places=1, presence=beats
    )
    await again.prepare()
    assert await files.read(table("train", INTERRUPTED)) == {}
    assert list(await files.read(table("train", ADOPTED))) == ["1/1/1/3"]


@pytest.mark.xfail(
    strict=True,
    reason="a runner whose claim lapsed (it stopped beating) still records its episode under its own fence, which "
    "never moved: the keeper deleted its sandbox when the claim lapsed, so its run fails, and that platform-made "
    "failure is the episode's first record; the newer attempt's completed episode is dropped",
)
async def test_a_runner_whose_claim_lapsed_does_not_record_the_episode(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes, leases=leases_of(ledger), admits=admits(ledger, beats))
    blobs = FileBlobStore(tmp_path / "blobs")
    await ask(ledger, {1: ({}, 1)}, KeyGated)

    def runner(name: str) -> EpisodeRunner:
        recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop")])})
        local = LocalRunner(recorder=recorder, pools={"boxes": pool})
        return EpisodeRunner(
            name, ledger, local, recorder, blobs, places=1, pools={"boxes": pool}, presence=beats, every=0.02,
            beating=1000.0,
        )  # fmt: skip

    async def leased(key: str) -> bool:
        return key in {lease.key for lease in await pool.held()}

    async def recorded() -> bool:
        return "1/1" in await ledger.read(table("train", EPISODES))

    zombie, fresh = runner("zombie"), runner("fresh")
    async with playing(zombie):
        await until(lambda: leased("train/1/1/1/box"))
        beat_at(ledger.directory, "zombie", time.time() - STALE - 10)  # paused (GC, a partition): no beats
        assert await sweep(pool, ledger, beats) == ["train/1/1/1/box"]  # the keeper: its claim lapsed
        async with playing(fresh):
            await until(lambda: leased("train/1/1/2/box"))  # the episode's next attempt
            gate("train/1/1/1/box").set()  # the zombie resumes
            await until(recorded)
            gate("train/1/1/2/box").set()
            await until(lambda: _not(leased("train/1/1/2/box")))
    episode = Record.from_json(_mapping((await ledger.read(table("train", EPISODES)))["1/1"])).episode
    assert episode.outcome is Outcome.COMPLETED, f"recorded {episode.outcome.value}: {episode.detail}"


async def _not(condition: Awaitable[bool]) -> bool:
    return not await condition


def _mapping(record: JsonValue) -> dict[str, Any]:
    assert isinstance(record, dict)
    return record


@pytest.mark.xfail(
    strict=True,
    reason="staleness compares a beat's time, written by the runner's clock, with the reader's clock: a runner whose "
    "clock is behind by more than STALE (90 s) looks dead the moment it beats, so its claims lapse while it plays "
    "(other runners play them again, and the keeper deletes its sandboxes)",
)
async def test_a_live_runner_whose_clock_is_behind_keeps_its_claims(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    await ask(ledger, {1: ({}, 1)})
    fence = await ledger.take(runner_scope("behind"))
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "behind", "fence": fence.number}, fence)
    await beats.beat("behind", {})
    beat_at(ledger.directory, "behind", time.time() - 120)  # it beat just now, by its clock, two minutes behind
    assert await holding(ledger, "train", await ledger.fences(), await beats_of(beats)) == {"1/1/1"}


class SteppedClock:
    """The `time` module as a pool sees it after its machine's clock was stepped forward (NTP, a resumed VM)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def time(self) -> float:
        return time.time() + self.seconds

    def monotonic(self) -> float:
        return time.monotonic()


@pytest.mark.xfail(
    strict=True,
    reason="a sandbox's wall-time limit is kept as a wall-clock time (Lease.ends) and compared with the wall clock: "
    "a clock stepped forward ends a lease early, and the keeper deletes a sandbox its run is still using",
)
async def test_a_clock_stepped_forward_does_not_end_a_lease_early(monkeypatch: pytest.MonkeyPatch) -> None:
    from rollout.harness import sandboxes as module
    from rollout.harness.sandboxes import SandboxLimits

    provider = FakeSandboxes()
    pool = SandboxPool(provider)
    lease = await pool.acquire(SandboxSpec(kind="fake", limits=SandboxLimits(seconds=3600.0)), "train/1/1/1/box")
    monkeypatch.setattr(module, "time", SteppedClock(2 * 3600.0))  # seconds later, the clock jumps two hours
    await pool.sweep()
    assert lease.handle in provider.sandboxes, "deleted after seconds of an hour's limit"


# --- Retention ------------------------------------------------------------------------------------------------------


class DeletingAfter(FileBlobStore):
    """A store of files that runs `before_delete` once, just before its first delete: another run's step then."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.before_delete: Callable[[], Awaitable[None]] | None = None

    async def delete(self, reference: BlobReference) -> None:
        if (hook := self.before_delete) is not None:
            self.before_delete = None
            await hook()
        await super().delete(reference)


@pytest.mark.xfail(
    strict=True,
    reason="Checkpoints.thin reads which blobs checkpoints name, then deletes the others: an add that finds the "
    "blob already stored (content addressing: it is not written again) and appends its checkpoint in between "
    "names a blob that is then deleted",
)
async def test_thin_never_deletes_a_blob_a_concurrent_add_names(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    store = DeletingAfter(tmp_path / "blobs")
    ours, theirs = Checkpoints(ledger, store), Checkpoints(ledger, FileBlobStore(tmp_path / "blobs"))
    x, y = await ledger.take(scope("x")), await ledger.take(scope("y"))

    def weights(name: str, content: bytes) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "adapter.bin").write_bytes(content)
        return directory

    first = await ours.add(x, new_id(), weights=weights("x1", b"shared bytes"), run="x")
    await ours.add(x, new_id(), weights=weights("x2", b"newer bytes"), run="x", parents=[first.id])
    made: list[Any] = []

    async def another_run_adds_now() -> None:
        made.append(await theirs.add(y, new_id(), weights=weights("y1", b"shared bytes"), run="y"))

    store.before_delete = another_run_adds_now
    assert await ours.thin(x, "x", Retention(recent=1, every=0)) == [first.id]
    (checkpoint,) = made
    reference = checkpoint.weights.files["adapter.bin"]
    assert await theirs.blobs.read(reference) == b"shared bytes"


# --- Launches -------------------------------------------------------------------------------------------------------


class Process:
    """A launched run's process that runs until signalled."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.signals: list[int] = []
        self._ended = asyncio.Event()

    def send_signal(self, number: int) -> None:
        self.signals.append(number)
        self._ended.set()

    async def wait(self) -> int:
        await self._ended.wait()
        return 0


def launcher(tmp_path: Path, launches: FileLaunches) -> Launcher:
    made = Launcher(
        "launcher/here", launches, FilePresence(tmp_path / "ledger"), tmp_path / "profiles", [], tmp_path / "runs"
    )
    made._offered = [{"profile": "p", "path": str(tmp_path / "p.toml")}]  # pyright: ignore[reportPrivateUsage]
    return made


async def _launch(launches: FileLaunches, id: str) -> Launch:
    return next(each for each in await launches.all() if each.id == id)


@pytest.mark.xfail(
    strict=True,
    reason="Launches.note overwrites a launch's state unconditionally: a stop asked for while the launcher starts "
    "the run (state STOPPING, over CLAIMED) is overwritten by the launcher's RUNNING, and the run is never stopped",
)
async def test_a_stop_asked_for_while_a_run_starts_stops_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launches = FileLaunches(tmp_path / "ledger")
    asked = await launches.ask(Asked(profile="p", environment="e:e", name="run"))
    starting = launcher(tmp_path, launches)
    process = Process()

    async def started(*command: Any, **options: Any) -> Process:
        await launches.note(asked.id, state=STOPPING)  # the monitor's stop, while the process starts
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", started)
    claimed = await launches.claim(asked.id, starting.name)
    assert claimed is not None and claimed.state == CLAIMED
    await starting._start(claimed)  # pyright: ignore[reportPrivateUsage]
    await starting._step()  # pyright: ignore[reportPrivateUsage]  (a launch asked to stop is signalled here)
    try:
        state = (await _launch(launches, asked.id)).state
        assert process.signals == [signal.SIGINT], f"never signalled: the launch is {state}"
    finally:
        process.send_signal(0)
        await asyncio.gather(*starting._watching, return_exceptions=True)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.xfail(
    strict=True,
    reason="the monitor's stop reads a launch's state and then writes STOPPED (for one asked for) or STOPPING: a "
    "launcher that claims and starts it in between has its run marked STOPPED, a final state, while it runs on, "
    "and never signalled",
)
async def test_a_stop_racing_a_claim_never_marks_a_running_launch_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rollout_train.monitor import system

    launches = FileLaunches(tmp_path / "ledger")
    asked = await launches.ask(Asked(profile="p", environment="e:e", name="run"))
    starting = launcher(tmp_path, launches)
    process = Process()

    async def started(*command: Any, **options: Any) -> Process:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", started)

    class Racing(FileLaunches):
        async def all(self) -> list[Launch]:
            found = await super().all()  # the monitor has read the launch: asked for
            claimed = await launches.claim(asked.id, starting.name)  # the launcher claims and starts it
            assert claimed is not None
            await starting._start(claimed)  # pyright: ignore[reportPrivateUsage]
            return found

    def racing(ledger: Ledger) -> Racing:
        return Racing(tmp_path / "ledger")

    monkeypatch.setattr(system, "launches_of", racing)
    await system.System(ledger=FileLedger(tmp_path / "ledger")).stop(asked.id)
    await starting._step()  # pyright: ignore[reportPrivateUsage]
    state = (await _launch(launches, asked.id)).state
    try:
        assert state != STOPPED or process.signals, f"the launch is {state} and its process runs on, unsignalled"
    finally:
        process.send_signal(0)
        await asyncio.gather(*starting._watching, return_exceptions=True)  # pyright: ignore[reportPrivateUsage]


# --- Suites ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Versioned:
    version: str = "1"


@pytest.mark.xfail(
    strict=True,
    reason="make_suite checks the name is free, takes the suite's fence, then appends its record and each start "
    "under its own key: a second maker that takes the fence midway is refused nothing it appends to new keys, so "
    "the suite in the ledger is the first maker's record and first starts with the second's other starts, which "
    "neither maker made (and the second plays its own list)",
)
async def test_two_makers_of_one_suite_leave_one_of_their_suites(tmp_path: Path) -> None:
    files = FileLedger(tmp_path / "ledger")
    first, second = Hooked(files), Hooked(files)  # two processes making the suite `s` at once
    first_wrote_a_start, second_took = asyncio.Event(), asyncio.Event()

    def starts(seed: int) -> list[Start]:
        return [Start(task="t", title="T", seed=seed + index, parameters={"seed": seed + index}) for index in range(3)]

    async def first_pauses() -> None:
        first_wrote_a_start.set()
        await second_took.wait()

    async def second_pauses() -> None:  # it has found no suite `s`
        await first_wrote_a_start.wait()

    async def second_has_taken() -> None:
        second_took.set()

    first.after_append[("evaluations/s/starts", "1")] = first_pauses
    second.after_read["evaluations/s/starts"] = second_pauses
    second.after_take["suites/s"] = second_has_taken
    by_second = asyncio.create_task(make_suite(second, "s", "e:e", Versioned(), starts=starts(100)))  # type: ignore[arg-type]
    await asyncio.sleep(0)  # (the second looks first)
    with pytest.raises(Fenced):
        await make_suite(first, "s", "e:e", Versioned(), starts=starts(0))  # type: ignore[arg-type]
    assert (await by_second).starts == starts(100)  # what the second maker plays
    found = await suite_of(files, "s")
    assert found is not None
    assert found.starts in (starts(0), starts(100)), [start.seed for start in found.starts]
