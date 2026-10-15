"""The ledger service and `HttpLedger`: the stores beside the ledger through it, retries that do nothing twice, a
database that cannot do it for a moment asked again, records as the stores keep them, `Fenced` as itself, and what a
pod's token may and may not do."""

import asyncio
import json
import math
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import sqlalchemy.exc
from pydantic import JsonValue

from rollout_train.checkpoints import CHECKPOINTS, RELEASED
from rollout_train.database import DatabaseLedger
from rollout_train.launches import Asked
from rollout_train.ledger import Appended, Fence, Fenced
from rollout_train.ledger_service import Conflict, Forbidden, HttpLedger, LedgerUnreachable, app, pod_token, scope_of
from rollout_train.ledger_service.service import READ_ONLY, REPEATED, operations, repeatable, unavailable
from rollout_train.pods.leases import PodLease
from rollout_train.presets import presets_of
from rollout_train.published import environment_versions_of
from rollout_train.record import STARTS, table
from rollout_train.registry import Taken
from rollout_train.serving import SERVING
from rollout_train.testing import LEDGER_TOKEN, served_ledger


def database(directory: Path) -> DatabaseLedger:
    return DatabaseLedger(f"sqlite:///{directory / 'ledger.db'}")


def losing(first: Any, then: Any = None) -> HttpLedger:
    """A client of the service `first` that loses the answer of the first request of each operation (done there), and
    sends the rest to `then` (another replica, or `first` restarted; `first` itself by default)."""
    done, retried = httpx.ASGITransport(app=first), httpx.ASGITransport(app=then or first)
    lost: list[str] = []

    class Losing(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path in lost:
                return await retried.handle_async_request(request)
            lost.append(request.url.path)
            await (await done.handle_async_request(request)).aread()
            raise httpx.ReadTimeout("the answer was lost", request=request)

    client = httpx.AsyncClient(transport=Losing(), base_url="http://ledger.test")
    return HttpLedger("http://ledger.test", token=LEDGER_TOKEN, client=client)


def posted(service: Any) -> Callable[[str, dict[str, JsonValue], str], Awaitable[Any]]:
    """Posts `{"args": args, "request": request}` to an operation of `service` with the platform's token, as JSON."""
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=service), base_url="http://ledger.test")

    async def post(operation: str, args: dict[str, JsonValue], request: str) -> Any:
        said = await client.post(f"/v1/{operation}", json={"args": args, "request": request},
                                 headers={"Authorization": f"Bearer {LEDGER_TOKEN}"})  # fmt: skip
        return said.json()

    return post


async def test_every_store_beside_the_ledger_is_reached_through_it(tmp_path: Path) -> None:
    ledger = served_ledger(database(tmp_path))
    entry = await ledger.registry.create("first")
    with pytest.raises(Taken):
        await ledger.registry.create("first")
    assert [each.name for each in await ledger.registry.runs()] == ["first"]
    assert (await ledger.registry.rename(entry.id, "second")).name == "second"
    with pytest.raises(KeyError):
        await ledger.registry.rename("nobody", "third")
    launch = await ledger.launches.ask(Asked("train", "second", {"environment": "e:env"}), entry.id)
    noted = await ledger.launches.note(launch.id, expect=("asked",), state="submitted", job="j1")
    assert (noted.state, noted.job) == ("submitted", "j1")
    assert [each.id for each in await ledger.launches.all()] == [launch.id]
    assert (await ledger.launches.note(launch.id, expect=("running",), state="ended")).state == "submitted"
    await ledger.presence.beat("runner-1", {"host": "h", "machine": {"memory": 1}})
    beats = await ledger.presence.beats()
    assert [(each.runner, each.about["host"]) for each in beats] == [("runner-1", "h")] and beats[0].age > -1
    presets = presets_of(ledger)
    assert presets is not None
    saved = await presets.save("small", {"groups": 2}, "a note")
    assert (saved.id, (await presets.get("small@1")).settings) == ("small@1", {"groups": 2})  # type: ignore[union-attr]
    assert [each.version for each in await presets.versions("small")] == [1]
    assert (await presets.delete("small")).deleted and await presets.all() == []
    versions = environment_versions_of(ledger)
    assert versions is not None and await versions.all() == [] and await versions.get("none@0") is None
    assert await ledger.now() > 0


async def test_a_take_retried_after_its_answer_was_lost_takes_one_fence(tmp_path: Path) -> None:
    ledger = losing(app(database(tmp_path), lambda: LEDGER_TOKEN))
    fence = await ledger.take("runs/a")
    assert fence.number == 1 and await ledger.fences() == {"runs/a": 1}  # (the retry got the first take's number)
    assert await ledger.append("runs/a/groups", "1", {"group": 1}, fence)  # (its own record, its answer lost)
    assert not await ledger.append("runs/a/groups", "1", {"group": 2}, fence)  # (another's record)


async def test_a_run_a_launch_and_a_pods_lease_made_again_after_a_lost_answer_are_made_once(tmp_path: Path) -> None:
    ledger = database(tmp_path)
    client = losing(*(app(ledger, lambda: LEDGER_TOKEN) for _ in range(2)))  # (the retry reaches another replica)
    entry = await client.registry.create("my-run")
    assert [(each.id, each.name) for each in await ledger.registry.runs()] == [(entry.id, "my-run")]
    launch = await client.launches.ask(Asked("train", "my-run", {"environment": "e:env"}), entry.id)
    assert [(each.id, each.run) for each in await ledger.launches.all()] == [(launch.id, entry.id)]
    lease = PodLease("pod-a", "runpod", 0, "inference", "image", "model", "gpu", 1.0, run=entry.id)
    made = await client.pods.put(lease, expect=None)  # (its retry finds the lease its first attempt wrote)
    assert made.version == 1 and await ledger.pods.get("pod-a") == made
    await client.pods.delete("pod-a", expect=1)  # (and the lease its first attempt deleted gone)
    assert await ledger.pods.get("pod-a") is None
    await client.pods.put(lease, expect=None)
    with pytest.raises(Conflict):  # (another's lease: not its own)
        await client.pods.put(lease, expect=None)
    with pytest.raises(Conflict):
        await client.pods.delete("pod-a", expect=7)


async def test_an_operation_sent_again_while_its_first_attempt_is_being_done_waits_for_its_answer(
    tmp_path: Path,
) -> None:
    done: list[JsonValue] = []

    async def slow(ledger: Any, args: Any, scope: Any) -> JsonValue:
        await asyncio.sleep(0.2)
        done.append(args["name"])
        return len(done)

    post = posted(app(database(tmp_path), lambda: LEDGER_TOKEN, more={"test/slow": (slow, None)}))
    first, again = await asyncio.gather(post("test/slow", {"name": "a"}, "r1"), post("test/slow", {"name": "a"}, "r1"))
    assert first == again == {"result": 1} and done == ["a"]
    assert await post("test/slow", {"name": "a"}, "r1") == {"result": 1} and done == ["a"]  # (its answer, kept)
    assert await post("test/slow", {"name": "a"}, "r2") == {"result": 2}  # (another request: done)


async def test_a_reads_answer_is_not_kept_but_read_again(tmp_path: Path) -> None:
    ledger = database(tmp_path)
    post = posted(app(ledger, lambda: LEDGER_TOKEN))
    assert await post("ledger/read", {"table": "t"}, "r1") == {"result": {}}
    await ledger.append("t", "k", {"v": 1}, await ledger.take("s"))
    assert await post("ledger/read", {"table": "t"}, "r1") == {"result": {"k": {"v": 1}}}
    assert READ_ONLY <= REPEATED <= set(operations())


async def test_a_record_holding_nan_goes_through_the_service_as_it_is_kept(tmp_path: Path) -> None:
    ledger = database(tmp_path)
    fence = await ledger.take("runs/a")
    await ledger.append(CHECKPOINTS, "c1", {"run": "a", "metrics": {"grad_norm": math.nan, "loss": math.inf}}, fence)
    client = losing(app(ledger, lambda: LEDGER_TOKEN))
    said: Any = (await client.read(CHECKPOINTS))["c1"]
    assert math.isnan(said["metrics"]["grad_norm"]) and said["metrics"]["loss"] == math.inf
    assert await client.append("runs/a/steps", "1", {"grad_norm": math.nan}, fence)  # (its own, its answer lost)
    assert math.isnan((await ledger.read("runs/a/steps"))["1"]["grad_norm"])  # type: ignore[index]


def lost_connection() -> sqlalchemy.exc.OperationalError:
    return sqlalchemy.exc.OperationalError("INSERT", {}, Exception("server closed the connection unexpectedly"))


class Coded(Exception):
    """A Postgres driver's error, with its SQLSTATE."""

    def __init__(self, sqlstate: str) -> None:
        super().__init__(f"SQLSTATE {sqlstate}")
        self.sqlstate = sqlstate


def coded(sqlstate: str) -> sqlalchemy.exc.OperationalError:
    return sqlalchemy.exc.OperationalError("UPDATE", {}, Coded(sqlstate))


def test_a_database_error_is_asked_again_where_nothing_was_done_or_it_may_be_done_again() -> None:
    locked = sqlalchemy.exc.OperationalError("BEGIN IMMEDIATE", {}, sqlite3.OperationalError("database is locked"))
    pool = sqlalchemy.exc.TimeoutError("QueuePool limit of size 5 overflow 10 reached")
    for nothing_done in (coded("40001"), coded("40P01"), coded("55P03"), coded("53300"), coded("57P03"), locked, pool):
        assert unavailable(nothing_done, repeated=False)
    for maybe_done in (lost_connection(), coded("08006"), coded("57P01")):
        assert unavailable(maybe_done, repeated=True) and not unavailable(maybe_done, repeated=False)
    disk = sqlalchemy.exc.OperationalError("INSERT", {}, sqlite3.OperationalError("disk I/O error"))
    unique = sqlalchemy.exc.IntegrityError("INSERT", {}, Coded("23505"))
    for failure in (disk, unique, ValueError("not the database's")):
        assert not unavailable(failure, repeated=True)
    assert repeatable("ledger/append", {}) and repeatable("ledger/read", {}) and not repeatable("presets/save", {})
    assert repeatable("ledger/take", {"request": "r1"}) and not repeatable("ledger/take", {})
    assert repeatable("launches/ask", {"id": "launch_1"}) and not repeatable("registry/create", {"id": None})


async def test_a_dropped_database_connection_is_asked_again_only_where_the_operation_may_be_done_again(
    tmp_path: Path,
) -> None:
    failures = [lost_connection(), lost_connection(), coded("40P01")]

    class Dropping(DatabaseLedger):
        async def append_returning(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended:
            if len(failures) == 3:
                raise failures.pop(0)
            return await super().append_returning(table, key, record, fence)

    async def write(ledger: Any, args: Any, scope: Any) -> JsonValue:  # (one that may not be done again)
        if failures:
            raise failures.pop(0)
        return "written"

    ledger = served_ledger(Dropping(f"sqlite:///{tmp_path / 'ledger.db'}"), more={"test/write": (write, None)})
    fence = await ledger.take("runs/a")
    assert await ledger.append("runs/a/groups", "1", {"group": 1}, fence)  # (asked again, and done)
    with pytest.raises(RuntimeError, match="500"):  # (it may have been done: not asked again)
        await ledger.result("test/write")
    assert await ledger.result("test/write") == "written"  # (a deadlock did nothing: asked again)


async def test_fenced_is_its_own_error_and_never_retried(tmp_path: Path) -> None:
    ledger = served_ledger(database(tmp_path))
    old = await ledger.take("runs/a")
    await ledger.take("runs/a")
    with pytest.raises(Fenced):
        await ledger.append("runs/a/groups", "1", {}, old)


async def test_a_service_that_does_not_answer_is_given_up_after_the_deadline() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(refused))
    ledger = HttpLedger("http://ledger.test", token=LEDGER_TOKEN, client=client, deadline=0.2)
    with pytest.raises(LedgerUnreachable):
        await ledger.take("runs/a")


async def test_tokens_are_checked_with_nothing_but_the_platforms_token() -> None:
    token = pod_token(LEDGER_TOKEN, "pod-a", "run_1")
    assert scope_of(LEDGER_TOKEN, LEDGER_TOKEN) is not None and scope_of(LEDGER_TOKEN, LEDGER_TOKEN).platform  # type: ignore[union-attr]
    said = scope_of(token, LEDGER_TOKEN)
    assert said is not None and (said.pod, said.run) == ("pod-a", "run_1")
    assert scope_of(token, "another-platform-token") is None
    assert scope_of(token.replace("rlp1.", "rlp1.x"), LEDGER_TOKEN) is None
    assert scope_of("", LEDGER_TOKEN) is None and scope_of("guess", LEDGER_TOKEN) is None


async def a_ledger_with_two_runs(directory: Path) -> DatabaseLedger:
    """Two runs' records: run_1 made c1 (over c0, made by run_0) and serves it; run_2 made c2."""
    ledger = database(directory)
    fence = await ledger.take("all")
    for id, run, base in (("c0", "run_0", None), ("c1", "run_1", "c0"), ("c2", "run_2", None)):
        await ledger.append(CHECKPOINTS, id, {"id": id, "run": run, "base": base}, fence)
    await ledger.append(RELEASED, "c2", {"at": 1}, fence)
    await ledger.append(table("run_1", SERVING), "1", {"channel": "policy", "checkpoint": "c1"}, fence)
    await ledger.append(table("run_1", STARTS), "1", {"run_settings": {"fixed": {}, "changeable": {}}}, fence)
    await ledger.append(table("run_2", SERVING), "1", {"channel": "policy", "checkpoint": "c2"}, fence)
    await ledger.append(table("run_1", "groups"), "1", {"group": 1}, fence)
    return ledger


async def test_a_pods_token_reads_its_runs_serving_records_and_checkpoints_and_writes_its_beat(tmp_path: Path) -> None:
    pod = served_ledger(await a_ledger_with_two_runs(tmp_path), token=pod_token(LEDGER_TOKEN, "pod-a", "run_1"))
    assert await pod.read(table("run_1", SERVING)) == {"1": {"channel": "policy", "checkpoint": "c1"}}
    assert set(await pod.read(CHECKPOINTS)) == {"c0", "c1"}  # (its run's, and what it was trained over)
    assert await pod.read(RELEASED) == {}
    assert set(await pod.tables()) == {table("run_1", SERVING), table("run_1", STARTS), CHECKPOINTS, RELEASED}
    await pod.presence.beat("pod-a", {"pod": {"name": "pod-a"}})
    for refused in (
        pod.read(table("run_2", SERVING)), pod.read(table("run_1", "groups")), pod.take("runs/run_1"),
        pod.presence.beat("pod-b", {}), pod.presence.beats(), pod.registry.runs(), pod.launches.all(),
        pod.read_all(), pod.fences(), pod.append("runs/run_1/serving", "2", {}, await served_ledger(
            database(tmp_path)).take("x")),
    ):  # fmt: skip
        with pytest.raises(Forbidden):
            await refused


async def test_a_token_that_is_not_the_ledgers_is_refused(tmp_path: Path) -> None:
    stranger = served_ledger(database(tmp_path), token=pod_token("another-platform-token", "pod-a", "run_1"))
    with pytest.raises(RuntimeError, match="401"):
        await stranger.read(CHECKPOINTS)
    nobody: Any = served_ledger(database(tmp_path), token="")
    with pytest.raises(RuntimeError, match="401"):
        await nobody.tables()


async def test_a_pods_token_is_honoured_for_reads_only_while_its_run_holds_it(tmp_path: Path) -> None:
    held = {"run": "run_1"}

    async def honoured(ledger: Any, scope: Any) -> bool:
        return scope.run == held["run"]

    pod = served_ledger(await a_ledger_with_two_runs(tmp_path), token=pod_token(LEDGER_TOKEN, "pod-a", "run_1"),
                        honoured=honoured)  # fmt: skip
    assert await pod.read(table("run_1", SERVING))
    held["run"] = "run_2"  # (released, and taken by another run)
    with pytest.raises(Forbidden):
        await pod.read(table("run_1", SERVING))
    await pod.presence.beat("pod-a", {})  # (it still beats)


def test_the_wire_says_records_as_the_stores_keep_them() -> None:
    from rollout_train.ledger_service.wire import decoded, encoded
    from rollout_train.registry import Entry

    entry = Entry("run_1", "first", 1.5)
    assert decoded(Entry, json.loads(json.dumps(encoded(entry)))) == entry
