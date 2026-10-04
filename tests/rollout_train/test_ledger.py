"""The ledger: tables that are only appended to, each key once, and fences that shut a replaced writer out."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rollout_train.ledger import Fence, Fenced, FileLedger, Ledger


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
