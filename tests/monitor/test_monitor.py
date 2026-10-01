"""The monitor: a feed written by hooks as runs happen, and the page's API over it."""

from pathlib import Path

import httpx
import pytest

from rollout.core.contracts import Message, Reasoning, ReasoningScope, Role, Text, ToolCall, ToolResult, ToolResultBlock
from rollout.core.local import LocalRunner
from rollout.monitor import FeedReader, RunFeed, plain
from tests.core.harness.test_hooks import Miner, specification


def test_messages_are_shown_plainly() -> None:
    reply = Message(
        role=Role.ASSISTANT,
        content=[
            Reasoning(scope=ReasoningScope.PORTABLE, text="The ore is close."),
            Text(text="Mining."),
            ToolCall(call_id="c1", name="mine", arguments={"x": 3}),
        ],
    )
    assert plain(reply) == {
        "role": "assistant",
        "text": "Mining.",
        "reasoning": "The ore is close.",
        "calls": [{"id": "c1", "name": "mine", "arguments": {"x": 3}}],
        "results": [],
    }
    result = ToolResult(content=[Text(text="mined deepslate_diamond_ore")])
    answered = Message(role=Role.TOOL, content=[ToolResultBlock(call_id="c1", result=result)])
    assert plain(answered)["results"] == [{"id": "c1", "text": "mined deepslate_diamond_ore", "error": False}]


async def test_the_feed_holds_a_run_as_it_happened_and_the_page_can_ask_for_it(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from rollout.monitor.app import create_app

    feed = RunFeed(tmp_path / "feed")
    runner = LocalRunner(providers={"scripted": lambda model: Miner()}, hooks=[feed])
    handle = await runner.start(specification(), labels={"group": "0001", "task": "t003", "episode": "2"})
    await handle.result()

    (run,) = FeedReader(tmp_path / "feed").runs()
    assert run["run_id"] == handle.run_id and run["state"] == "completed"
    assert run["labels"] == {"group": "0001", "task": "t003", "episode": "2"}
    assert run["rewards"] == {"ada": 2.0} and run["slots"] == ["ada"] and run["samples"] == 1

    transport = httpx.ASGITransport(app=create_app(tmp_path / "feed"))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        assert "Runs monitor" in (await client.get("/")).text
        assert (await client.get("/api/runs")).json()[0]["run_id"] == handle.run_id
        lines = (await client.get(f"/api/runs/{handle.run_id}")).json()["lines"]
        (sample,) = [line for line in lines if line["kind"] == "sample"]
        assert sample["slot"] == "ada" and sample["tools"] == ["mine"]
        assert [message["text"] for message in sample["messages"]] == ["You mine.", "You see a wall."]  # what it saw
        assert sample["reply"]["calls"] == [{"id": "c1", "name": "mine", "arguments": {"x": 3}}]  # what it did
        assert next(line["type"] for line in lines if line["kind"] == "event") == "run.created"
        later = (await client.get(f"/api/runs/{handle.run_id}", params={"after": len(lines)})).json()
        assert later["lines"] == []  # the page asks only for what is new


def test_the_feed_keeps_only_the_newest_runs(tmp_path: Path) -> None:
    import os

    directory = tmp_path / "feed"
    directory.mkdir()
    for index in range(5):
        path = directory / f"old-{index}.jsonl"
        path.write_text("{}\n")
        os.utime(path, (index, index))
    feed = RunFeed(directory, keep=3)
    feed._write("new", {"kind": "event"})  # pyright: ignore[reportPrivateUsage]
    feed.close()
    assert sorted(path.stem for path in directory.glob("*.jsonl")) == ["new", "old-3", "old-4"]
