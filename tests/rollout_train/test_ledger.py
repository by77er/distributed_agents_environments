"""The ledger: tables that are only appended to, each key once, and fences that shut a replaced writer out."""

import asyncio
import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rollout_train.ledger import Appended, Fence, Fenced, FileLedger, Ledger, appended


def in_files(directory: Path) -> Ledger:
    return FileLedger(directory)


def in_a_database(directory: Path) -> Ledger:
    pytest.importorskip("rollout_durable")
    from rollout_train.database import DatabaseLedger

    return DatabaseLedger(f"sqlite:///{directory / 'ledger.db'}")


KINDS = pytest.mark.parametrize("opened", [in_files, in_a_database], ids=["files", "database"])


@KINDS
async def test_a_key_is_appended_once_and_tables_are_read_in_order(
    tmp_path: Path, opened: Callable[[Path], Ledger]
) -> None:
    ledger = opened(tmp_path)
    fence = await ledger.take("run")
    assert await ledger.append("groups", "2", {"row": "b"}, fence)
    assert await ledger.append("groups", "1", {"row": "a"}, fence)
    assert not await ledger.append("groups", "2", {"row": "changed"}, fence)  # it is there: the first stands
    assert await ledger.read("groups") == {"2": {"row": "b"}, "1": {"row": "a"}}
    assert await ledger.read("none") == {}
    assert await opened(tmp_path).read("groups") == {"2": {"row": "b"}, "1": {"row": "a"}}  # another process
    assert await ledger.tables() == ["groups"] and await ledger.fences() == {"run": 1}


@KINDS
async def test_a_writer_that_was_replaced_is_refused(tmp_path: Path, opened: Callable[[Path], Ledger]) -> None:
    old_ledger, new_ledger = opened(tmp_path), opened(tmp_path)
    old = await old_ledger.take("policy/a")
    other = await old_ledger.take("policy/b")
    assert await old_ledger.append("checkpoints/a", "1", {}, old)
    new = await new_ledger.take("policy/a")  # a new process takes its place
    assert new.number == old.number + 1
    with pytest.raises(Fenced):
        await old_ledger.append("checkpoints/a", "2", {}, old)
    assert await new_ledger.append("checkpoints/a", "2", {}, new)
    assert await old_ledger.append("checkpoints/b", "1", {}, other)  # another scope's fence is its own
    assert list(await new_ledger.read("checkpoints/a")) == ["1", "2"]


async def test_a_half_written_line_was_never_appended(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    await ledger.append("groups", "1", {"row": "a"}, fence)
    with (tmp_path / "groups.jsonl").open("a") as file:
        file.write(json.dumps({"key": "2", "fence": 1, "record": {"row": "b"}})[:20])  # the writer died here
    assert await ledger.read("groups") == {"1": {"row": "a"}}


async def test_an_append_after_a_line_cut_short_removes_it_and_is_kept(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    for key in ("1", "2", "3"):
        assert await ledger.append("groups", key, {"row": key * 300}, fence)
    path = tmp_path / "groups.jsonl"
    whole = path.read_bytes()
    os.truncate(path, len(whole) - 200)  # the writer of "3" died mid-line (a large record, a full disk)
    again = FileLedger(tmp_path)  # started again
    assert list(await again.read("groups")) == ["1", "2"]
    assert await again.append("groups", "4", {"row": "d"}, fence)
    assert await again.append("groups", "3", {"row": "c"}, fence)  # "3" was never appended: its key is free
    assert await again.read("groups") == {"1": {"row": "1" * 300}, "2": {"row": "2" * 300}, "4": {"row": "d"},
                                          "3": {"row": "c"}}  # fmt: skip
    assert all(json.loads(line) for line in path.read_text().splitlines())  # every line whole
    assert await ledger.read("groups") == await again.read("groups")  # (and the first process sees the same)
    assert not await ledger.append("groups", "4", {"row": "other"}, fence)


async def test_a_record_whose_newline_was_cut_off_stands(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    await ledger.append("groups", "1", {"row": "a"}, fence)
    await ledger.append("groups", "2", {"row": "b"}, fence)
    path = tmp_path / "groups.jsonl"
    os.truncate(path, path.stat().st_size - 1)  # all of "2" but its newline
    assert await ledger.read("groups") == {"1": {"row": "a"}, "2": {"row": "b"}}  # (readers saw it)
    assert not await ledger.append("groups", "2", {"row": "other"}, fence)
    assert await ledger.append("groups", "3", {"row": "c"}, fence)
    assert await FileLedger(tmp_path).read("groups") == {"1": {"row": "a"}, "2": {"row": "b"}, "3": {"row": "c"}}


async def test_a_ledger_of_files_sees_what_others_append_and_a_table_replaced(tmp_path: Path) -> None:
    mine, theirs = FileLedger(tmp_path), FileLedger(tmp_path)
    fence = await mine.take("run")
    await mine.append("groups", "1", {}, fence)
    await theirs.append("groups", "2", {}, fence)  # another process: this one knows only what it appended
    assert not await mine.append("groups", "2", {"other": True}, fence)
    path = tmp_path / "groups.jsonl"
    replaced = tmp_path / "replaced.jsonl"
    replaced.write_text(path.read_text().splitlines()[0] + "\n")  # (a table restored from a copy holding "1")
    os.replace(replaced, path)
    assert await mine.append("groups", "2", {"again": True}, fence)
    assert await theirs.read("groups") == {"1": {}, "2": {"again": True}}


async def test_fences_are_replaced_whole_when_one_is_taken(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    await ledger.take("run")
    before = (tmp_path / "fences.json").stat().st_ino
    await ledger.take("run")
    assert (tmp_path / "fences.json").stat().st_ino != before  # another file, renamed over it: never written in place
    assert await ledger.fences() == {"run": 2}
    assert sorted(os.listdir(tmp_path)) == [".lock", "fences.json"]  # (nothing left beside it)


async def test_a_ledger_of_files_does_not_hold_up_the_event_loop_while_another_holds_its_lock(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    with (tmp_path / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # another process holds the ledger
        appending = asyncio.create_task(ledger.append("groups", "1", {}, fence))
        ticks = 0
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1  # (the loop goes on meanwhile)
        assert ticks == 5 and not appending.done()
        fcntl.flock(lock, fcntl.LOCK_UN)
    assert await appending


@KINDS
async def test_an_append_says_whether_it_wrote_and_what_the_table_holds(
    tmp_path: Path, opened: Callable[[Path], Ledger]
) -> None:
    ledger = opened(tmp_path)
    fence = await ledger.take("run")
    assert await appended(ledger, "groups", "1", {"row": "a"}, fence) == Appended(True, {"row": "a"})
    assert await appended(ledger, "groups", "1", {"row": "b"}, fence) == Appended(False, {"row": "a"})
    assert await ledger.read("groups") == {"1": {"row": "a"}}


async def test_a_ledger_without_appends_that_say_what_is_there_is_read_back(tmp_path: Path) -> None:
    class Plain:  # (a `Ledger` with only the protocol's operations)
        def __init__(self) -> None:
            self.inner = FileLedger(tmp_path)

        async def take(self, scope: str) -> Fence:
            return await self.inner.take(scope)

        async def append(self, table: str, key: str, record: Any, fence: Fence) -> bool:
            return await self.inner.append(table, key, record, fence)

        async def read(self, table: str) -> dict[str, Any]:
            return await self.inner.read(table)

        async def tables(self) -> list[str]:
            return await self.inner.tables()

        async def fences(self) -> dict[str, int]:
            return await self.inner.fences()

    ledger = Plain()
    fence = await ledger.take("run")
    assert await appended(ledger, "groups", "1", 1, fence) == Appended(True, 1)
    assert await appended(ledger, "groups", "1", 2, fence) == Appended(False, 1)


async def test_a_ledger_in_postgres_is_shared_and_fenced_as_in_sqlite(postgres: str) -> None:
    from rollout_train.database import DatabaseLedger

    first, second = DatabaseLedger(postgres), DatabaseLedger(postgres)  # two processes, one database
    old = await first.take("runs/a")
    assert await first.append("runs/a/groups", "1", {"task": "t1"}, old)
    new = await second.take("runs/a")
    with pytest.raises(Fenced):
        await first.append("runs/a/groups", "2", {"task": "t2"}, old)
    assert await second.append("runs/a/groups", "2", {"task": "t2"}, new)
    assert not await second.append("runs/a/groups", "1", {"task": "changed"}, new)
    assert await first.read("runs/a/groups") == {"1": {"task": "t1"}, "2": {"task": "t2"}}
    assert await first.fences() == {"runs/a": 2}


async def test_a_runs_directory_says_where_its_ledger_is(tmp_path: Path) -> None:
    from rollout_train.ledger import LOCATION, of_run, opened, present

    assert isinstance(of_run(tmp_path), FileLedger) and not present(of_run(tmp_path))  # files, by default
    pytest.importorskip("rollout_durable")
    location = {"kind": "rollout_train.database:DatabaseLedger", "url": f"sqlite:///{tmp_path / 'shared.db'}"}
    fence = await opened(location).take("runs/a")
    await opened(location).append("runs/a/groups", "1", {"task": "t1"}, fence)
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-a" / LOCATION).write_text(json.dumps(location))
    found = of_run(tmp_path / "run-a")  # what the monitor and the report open
    assert present(found) and await found.read("runs/a/groups") == {"1": {"task": "t1"}}


async def test_a_ledger_moves_from_files_to_sqlite_to_postgres_whole(tmp_path: Path, postgres: str) -> None:
    from rollout_train.database import DatabaseLedger, copy

    files = FileLedger(tmp_path / "files")
    run, policy = await files.take("runs/a"), await files.take("policies/p")
    await files.take("runs/a")  # (a second writer: the fence is 2 now)
    run = await files.take("runs/a")
    for key in ("2", "1", "3"):
        await files.append("runs/a/groups", key, {"group": key}, run)
    await files.append("policies/p/checkpoints", "1", {"number": 1}, policy)
    sqlite = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    assert await copy(files, sqlite) == 4
    assert await copy(sqlite, DatabaseLedger(postgres)) == 4
    moved = DatabaseLedger(postgres)
    assert list(await moved.read("runs/a/groups")) == ["2", "1", "3"]  # in the order they were appended
    assert await moved.fences() == await files.fences() == {"policies/p": 1, "runs/a": 3}
    assert await moved.tables() == await files.tables()
    with pytest.raises(Fenced):  # a writer of before the move is still shut out
        await moved.append("runs/a/groups", "4", {}, Fence("runs/a", 2))
    assert await moved.append("runs/a/groups", "4", {"group": "4"}, await moved.take("runs/a"))
    with pytest.raises(ValueError, match="already has"):
        await copy(files, moved)  # (copying twice would mix two runs' tables)
