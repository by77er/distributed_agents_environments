"""The ledger service and `HttpLedger`: the stores beside the ledger through it, retries that do nothing twice, `Fenced`
as itself, and what a pod's token may and may not do."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rollout_train.checkpoints import CHECKPOINTS, RELEASED
from rollout_train.database import DatabaseLedger
from rollout_train.launches import Asked
from rollout_train.ledger import Fenced
from rollout_train.ledger_service import Forbidden, HttpLedger, LedgerUnreachable, app, pod_token, scope_of
from rollout_train.presets import presets_of
from rollout_train.published import environment_versions_of
from rollout_train.record import STARTS, table
from rollout_train.registry import Taken
from rollout_train.serving import SERVING
from rollout_train.testing import LEDGER_TOKEN, served_ledger


def database(directory: Path) -> DatabaseLedger:
    return DatabaseLedger(f"sqlite:///{directory / 'ledger.db'}")


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
    assert [(each.runner, each.about["host"]) for each in beats] == [("runner-1", "h")] and beats[0].age >= 0
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
    service = app(database(tmp_path), lambda: LEDGER_TOKEN)
    inner = httpx.ASGITransport(app=service)
    lost: list[str] = []

    class Losing(httpx.AsyncBaseTransport):
        """Passes each request on, and loses the answer of the first of each operation."""

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            response = await inner.handle_async_request(request)
            if request.url.path not in lost:
                lost.append(request.url.path)
                await response.aread()
                raise httpx.ReadTimeout("the answer was lost", request=request)
            return response

    client = httpx.AsyncClient(transport=Losing(), base_url="http://ledger.test")
    ledger = HttpLedger("http://ledger.test", token=LEDGER_TOKEN, client=client)
    fence = await ledger.take("runs/a")
    assert fence.number == 1 and await ledger.fences() == {"runs/a": 1}  # (the retry got the first take's number)
    assert await ledger.append("runs/a/groups", "1", {"group": 1}, fence)  # (its own record, its answer lost)
    assert not await ledger.append("runs/a/groups", "1", {"group": 2}, fence)  # (another's record)


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
