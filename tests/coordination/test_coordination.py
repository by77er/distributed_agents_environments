import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.coordination import BoardTools, CoordinationStore, Delivery, Relay, SessionTools, register
from rollout.core.contracts import Conflict, Text, ToolResult
from rollout.database import Database


@pytest.fixture
def store(tmp_path: Path, database: str | None) -> Iterator[CoordinationStore]:
    store = CoordinationStore(Database(database) if database else tmp_path / "coordination.sqlite")
    store.write(lambda db: register(db, "lead", None, "Coordinate the migration"))
    yield store
    store.database.close()


def caller_of(effect_id: str) -> str | None:
    """Tests name the caller in the run part of the effect_id: `{name}:{generation}:{ordinal}`."""
    return effect_id.split(":")[0]


def text(result: ToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, Text))


async def call(tools: SessionTools | BoardTools, tool: str, effect_id: str, /, **arguments: JsonValue) -> ToolResult:
    return await tools.call(tool, arguments, effect_id=effect_id, arguments_digest=repr(sorted(arguments.items())))


async def test_sessions_are_created_and_messaged_through_the_outbox(store: CoordinationStore) -> None:
    sessions = SessionTools(store, caller_of, status=lambda name: "waiting")
    created = await call(sessions, "create_session", "lead:0:1", name="worker-1", instructions="Port the parser.")
    assert "Created session 'worker-1'" in text(created)
    again = await call(sessions, "create_session", "lead:0:1", name="worker-1", instructions="Port the parser.")
    assert again == created  # the same effect: the recorded result, nothing created twice
    with pytest.raises(Conflict):
        await call(sessions, "create_session", "lead:0:1", name="worker-2", instructions="Something else.")
    taken = await call(sessions, "create_session", "lead:0:2", name="worker-1", instructions="x")
    assert taken.is_error

    await call(sessions, "send_message", "worker-1:0:5", to="lead", text="Done with the parser.", urgent=True)
    missing = await call(sessions, "send_message", "worker-1:0:6", to="nobody", text="hello")
    assert missing.is_error

    pending = [(d.recipient, d.sender, d.text, d.priority) for d in store.pending()]
    assert pending == [
        ("worker-1", "lead", "Port the parser.", "normal"),
        ("lead", "worker-1", "Done with the parser.", "high"),
    ]
    listing = text(await call(sessions, "list_sessions", "worker-1:0:7"))
    assert "worker-1 (you): waiting; created by lead; Port the parser." in listing
    assert "lead: waiting; created by the operator" in listing


async def test_the_board_fans_out_and_claims_are_exclusive(store: CoordinationStore) -> None:
    store.write(lambda db: register(db, "worker-1", "lead", "w"))
    store.write(lambda db: register(db, "worker-2", "lead", "w"))
    board = BoardTools(store, caller_of)
    await call(board, "subscribe", "worker-1:0:1", channel="migration")
    await call(board, "subscribe", "worker-2:0:1", channel="migration")
    posted = await call(
        board, "post", "lead:0:3", channel="migration", title="Port auth", body="Move auth.", kind="task"
    )
    assert text(posted) == "Posted #1 to migration; 2 subscriber(s) told."
    notices = store.pending()
    assert {d.recipient for d in notices} == {"worker-1", "worker-2"}
    assert all(d.priority == "low" and "claim_task(1)" in d.text for d in notices)

    first = await call(board, "claim_task", "worker-2:0:4", post_id=1)
    second = await call(board, "claim_task", "worker-1:0:4", post_id=1)
    assert not first.is_error and second.is_error and "claimed by worker-2" in text(second)
    wrong = await call(board, "resolve_task", "worker-1:0:5", post_id=1, result="done")
    assert wrong.is_error
    await call(board, "resolve_task", "worker-2:0:5", post_id=1, result="Auth ported in PR 12.")
    assert store.pending()[-1].recipient == "lead"  # the author is told
    board_text = text(await call(board, "read_board", "lead:0:6"))
    assert board_text.startswith("Channels: migration (1)")
    assert "result: Auth ported in PR 12." in board_text


async def test_callers_must_be_participants(store: CoordinationStore) -> None:
    board = BoardTools(store, lambda effect_id: None)
    assert (await call(board, "read_board", "anyone:0:1")).is_error


async def test_the_relay_delivers_the_outbox_once(store: CoordinationStore) -> None:
    delivered: list[Delivery] = []
    failures = [1]

    async def deliver(delivery: Delivery) -> None:
        if failures:
            failures.pop()
            raise ConnectionError("runner unavailable")  # the first attempt fails; it is retried
        delivered.append(delivery)

    sessions = SessionTools(store, caller_of, status=lambda name: "waiting")
    relay = Relay(store, deliver, retry_seconds=0.05)
    relay.start()
    await call(sessions, "create_session", "lead:0:1", name="worker-1", instructions="Go.")
    for _ in range(100):
        if delivered:
            break
        await asyncio.sleep(0.02)
    await relay.stop()
    assert [(d.recipient, d.text) for d in delivered] == [("worker-1", "Go.")]
    assert store.pending() == []


async def test_one_relay_delivers_among_processes_sharing_a_database(postgres: str) -> None:
    """Two stores on separate connections pools stand in for two processes: while one relay delivers, the other
    waits, and takes over when the first stops. Every message is delivered once."""
    stores = [CoordinationStore(Database(postgres)) for _ in range(2)]
    stores[0].write(lambda db: register(db, "lead", None, "Coordinate"))
    delivered: list[tuple[int, str]] = []

    def deliverer(index: int):  # type: ignore[no-untyped-def]
        async def deliver(delivery: Delivery) -> None:
            delivered.append((index, delivery.key))

        return deliver

    relays = [Relay(store, deliverer(index), retry_seconds=0.05) for index, store in enumerate(stores)]
    for relay in relays:
        relay.start()
        await asyncio.sleep(0.2)  # the first one leads
    sessions = [SessionTools(store, caller_of, status=lambda name: "waiting") for store in stores]
    for index in range(4):
        await call(sessions[index % 2], "send_message", f"lead:0:{index}", to="lead", text=f"note {index}")
    for _ in range(100):
        if len(delivered) == 4:
            break
        await asyncio.sleep(0.05)
    assert {relay for relay, _ in delivered} == {0}

    await relays[0].stop()  # the leader stops; the other takes over
    await call(sessions[0], "send_message", "lead:0:9", to="lead", text="after")
    for _ in range(100):
        if len(delivered) == 5:
            break
        await asyncio.sleep(0.05)
    await relays[1].stop()
    assert delivered[-1] == (1, "lead:0:9")
    assert sorted(key for _, key in delivered) == sorted({key for _, key in delivered})  # each once
    for store in stores:
        store.database.close()
