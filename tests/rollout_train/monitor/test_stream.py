"""What changed, as it changes: the monitor's topics read at most once a beat, their versions, the answers that say
nothing changed (304), the stream of versions, and a rename that every open page hears of."""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout_train import FileLedger
from rollout_train.checkpoints import Checkpoints, new_id
from rollout_train.layout import BLOBS, FEED, LEDGER
from rollout_train.monitor.feed import FeedReader
from rollout_train.monitor.stream import MISSING, Hub, version_of
from rollout_train.monitor.system import System
from rollout_train.record import GROUPS, STARTS, scope, table
from rollout_train.registry import registry_of

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app
from tests.rollout_train.support import free_port, monitor_client


async def a_run(tmp_path: Path) -> FileLedger:
    """A run that decided one group: what a loop that just started writes."""
    ledger = FileLedger(tmp_path / LEDGER)
    registry = registry_of(ledger)
    assert registry is not None
    await registry.create("train", "train")
    fence = await ledger.take(scope("train"))
    start: JsonValue = {"policy": "miner", "host": "here", "started": 5.0, "directory": str(tmp_path)}
    await ledger.append(table("train", STARTS), str(fence.number), start, fence)
    decided: JsonValue = {"task": "t003", "title": "chests", "decided": 5.0, "episodes": 2}
    await ledger.append(table("train", GROUPS), "1", decided, fence)
    return ledger


def test_a_version_leaves_out_when_a_body_was_read() -> None:
    assert version_of({"at": 1.0, "runs": [1]}) == version_of({"at": 2.0, "runs": [1]})
    assert version_of({"now": 1.0, "runs": [1]}) != version_of({"now": 1.0, "runs": [2]})


async def test_a_topic_is_read_once_a_beat_whoever_asks_and_its_version_changes_with_what_it_says(
    tmp_path: Path,
) -> None:
    ledger = await a_run(tmp_path)
    system = System(tmp_path, FeedReader(tmp_path / FEED))
    reads = 0
    snapshot = system.snapshot

    async def counted(relayed: bool = False) -> dict[str, object]:
        nonlocal reads
        reads += 1
        return await snapshot(relayed)

    system.snapshot = counted  # type: ignore[method-assign]
    hub = Hub(system, beat=60.0)
    first, second = await asyncio.gather(hub.read("system"), hub.read("system"))
    assert reads == 1 and first is second and "machine" not in json.loads(first.body)
    hub.beat = 0.0
    again = await hub.read("system")
    assert reads == 2 and again.version == first.version  # (read again: nothing it says changed)
    fence = await ledger.take(scope("train"))
    await ledger.append(table("train", GROUPS), "2", {"task": "t004", "decided": 6.0, "episodes": 2}, fence)
    assert (await hub.read("system")).version != first.version
    assert (await hub.read("group/train/1")).version != MISSING
    assert (await hub.read("group/train/9")).version == MISSING


async def test_the_ledger_is_read_once_within_a_reading_whatever_topics_read_it(tmp_path: Path) -> None:
    ledger = await a_run(tmp_path)
    system = System(ledger=ledger)
    reads = 0
    read_all = ledger.read_all

    async def counted(*, leaving_out: str | None = None) -> dict[str, dict[str, JsonValue]]:
        nonlocal reads
        reads += 1
        return await read_all(leaving_out=leaving_out)

    ledger.read_all = counted  # type: ignore[method-assign]
    async with system.one_reading():
        await asyncio.gather(system.snapshot(), system.evals(), system.eval_subjects(), system.lineage())
        assert reads == 1
        system.read_afresh()  # (the monitor changed something itself: what it shows is read again)
        await system.snapshot()
        assert reads == 2
    await system.snapshot()
    await system.snapshot()
    assert reads == 4  # (outside a reading, each read reads the ledger)


async def test_a_watcher_hears_each_version_at_once_and_then_each_change(tmp_path: Path) -> None:
    ledger = await a_run(tmp_path)
    hub = Hub(System(tmp_path, FeedReader(tmp_path / FEED)), beat=0.05)
    running = asyncio.create_task(hub.run())
    versions = hub.watch(["system", "group/train/1"])
    try:
        heard = {topic: version for topic, version in [await anext(versions), await anext(versions)]}
        assert set(heard) == {"system", "group/train/1"}
        fence = await ledger.take(scope("train"))
        await ledger.append(table("train", GROUPS), "2", {"task": "t004", "decided": 6.0, "episodes": 2}, fence)
        topic, version = await asyncio.wait_for(anext(versions), 5.0)
        assert topic == "system" and version != heard["system"]  # (the group it watches did not change)
    finally:
        await versions.aclose()
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running


async def test_an_answer_names_its_version_and_one_asked_again_with_it_is_told_nothing_changed(tmp_path: Path) -> None:
    await a_run(tmp_path)
    async with monitor_client(tmp_path, beat=0.0) as client:
        for path in (
            "/api/system",
            "/api/runs",
            "/api/statistics",
            "/api/checkpoints",
            "/api/machines",
            "/api/groups/train/1",
        ):
            answer = await client.get(path)
            etag = answer.headers["etag"]
            assert answer.status_code == 200 and etag.startswith('W/"')
            assert (await client.get(path, headers={"If-None-Match": etag})).status_code == 304
        assert (await client.get("/api/groups/train/9")).status_code == 404
        assert (await client.get("/api/machines")).json()["hosts"] == []  # (nothing has beaten)


async def test_a_run_is_named_again_and_the_page_hears_of_it(tmp_path: Path) -> None:
    await a_run(tmp_path)
    async with monitor_client(tmp_path, beat=60.0) as client:
        before = await client.get("/api/system")
        renamed = await client.post("/api/rename", json={"id": "train", "name": "first-try"})
        entry = renamed.json()["entry"]
        assert renamed.status_code == 200 and entry == {**entry, "id": "train", "name": "first-try"}
        after = await client.get("/api/system", headers={"If-None-Match": before.headers["etag"]})
        assert after.status_code == 200 and after.json()["runs"][0]["name"] == "first-try"  # (read afresh at once)
        assert (await client.get("/api/statistics")).json()["names"]["runs"] == {"train": "first-try"}
        taken = await client.post("/api/rename", json={"id": "train", "name": "has/slash"})
        assert taken.status_code == 409 and taken.json()["error"]
        missing = await client.post("/api/rename", json={"id": "nobody", "name": "other"})
        assert missing.status_code == 404
        assert (await client.post("/api/rename", content=b"not json")).status_code == 400


async def test_a_bookmark_is_made_moved_and_taken_away_and_the_page_hears_of_it(tmp_path: Path) -> None:
    ledger = await a_run(tmp_path)
    fence = await ledger.take(scope("train"))
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / BLOBS))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "adapter.bin").write_text("weights")
    first = await checkpoints.add(fence, new_id(), weights=tmp_path / "weights", run="train", base="model", step=1)
    second = await checkpoints.add(
        fence, new_id(), weights=tmp_path / "weights", run="train", step=2, parents=[first.id]
    )
    async with monitor_client(tmp_path, beat=60.0) as client:

        async def marked() -> dict[str, list[str]]:
            system = (await client.get("/api/system")).json()
            assert system["bookmarks"] == {
                mark: checkpoint for mark, checkpoint in system["names"]["bookmarks"].items()
            }
            return {checkpoint["id"]: checkpoint["bookmarks"] for checkpoint in system["checkpoints"]}

        assert await marked() == {first.id: [], second.id: []}
        made = await client.post("/api/bookmarks", json={"name": "good", "checkpoint": first.id[:6]})
        assert made.status_code == 200 and made.json()["bookmark"]["checkpoint"] == first.id  # (an id's start says it)
        assert await marked() == {first.id: ["good"], second.id: []}
        moved = await client.post("/api/bookmarks", json={"name": "good", "checkpoint": "train"})  # (the run's newest)
        assert moved.status_code == 200 and await marked() == {first.id: [], second.id: ["good"]}
        assert (await client.post("/api/bookmarks", json={"name": "a:b", "checkpoint": first.id})).status_code == 409
        assert (await client.post("/api/bookmarks", json={"name": "x", "checkpoint": "nothing"})).status_code == 404
        assert (await client.post("/api/bookmarks", json={"name": "x", "checkpoint": "base"})).status_code == 404
        assert (await client.delete("/api/bookmarks/good")).status_code == 200
        assert await marked() == {first.id: [], second.id: []}
        assert (await client.delete("/api/bookmarks/good")).status_code == 404


@contextlib.asynccontextmanager
async def serving(tmp_path: Path) -> AsyncGenerator[str]:
    """The monitor over `tmp_path`, served for real (a stream is read as it comes, which needs a server)."""
    import uvicorn

    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(tmp_path, beat=0.05), host="127.0.0.1", port=port, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 (the server says when it is up)
        await asyncio.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


async def test_the_stream_says_each_topics_version_and_then_what_changed(tmp_path: Path) -> None:
    ledger = await a_run(tmp_path)
    async with serving(tmp_path) as address, httpx.AsyncClient(base_url=address, timeout=10) as client:
        held = (await client.get("/api/system")).headers["etag"]
        async with client.stream(
            "GET", "/api/stream", params=[("topic", "system"), ("topic", "group/train/1")]
        ) as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            lines = stream.aiter_lines()

            async def event() -> tuple[str, dict[str, str]]:
                kind = data = ""
                async for line in lines:
                    if line.startswith("event: "):
                        kind = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data = line.removeprefix("data: ")
                    elif not line and kind:
                        return kind, json.loads(data)
                raise AssertionError("the stream ended")

            assert (await event())[0] == "hello"
            first = dict([(said["topic"], said["version"]) for _, said in [await event(), await event()]])
            assert held == f'W/"{first["system"]}"'  # (the version the stream names is the answer's ETag)
            fence = await ledger.take(scope("train"))
            await ledger.append(table("train", GROUPS), "2", {"task": "t004", "decided": 6.0, "episodes": 2}, fence)
            kind, said = await asyncio.wait_for(event(), 5.0)
            assert kind == "version" and said["topic"] == "system" and said["version"] != first["system"]
