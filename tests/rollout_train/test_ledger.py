"""The ledger: tables that are only appended to, each key once, and fences that shut a replaced writer out."""

import json
from pathlib import Path

import pytest

from rollout_train.ledger import Fenced, FileLedger


async def test_a_key_is_appended_once_and_tables_are_read_in_order(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    assert await ledger.append("groups", "2", {"row": "b"}, fence)
    assert await ledger.append("groups", "1", {"row": "a"}, fence)
    assert not await ledger.append("groups", "2", {"row": "changed"}, fence)  # it is there: the first stands
    assert await ledger.read("groups") == {"2": {"row": "b"}, "1": {"row": "a"}}
    assert await ledger.read("none") == {}
    assert await FileLedger(tmp_path).read("groups") == {"2": {"row": "b"}, "1": {"row": "a"}}  # another process


async def test_a_writer_that_was_replaced_is_refused(tmp_path: Path) -> None:
    old_ledger, new_ledger = FileLedger(tmp_path), FileLedger(tmp_path)
    old = await old_ledger.take("policy/a")
    other = await old_ledger.take("policy/b")
    assert await old_ledger.append("versions/a", "1", {}, old)
    new = await new_ledger.take("policy/a")  # a new process takes its place
    assert new.number == old.number + 1
    with pytest.raises(Fenced):
        await old_ledger.append("versions/a", "2", {}, old)
    assert await new_ledger.append("versions/a", "2", {}, new)
    assert await old_ledger.append("versions/b", "1", {}, other)  # another scope's fence is its own
    assert list(await new_ledger.read("versions/a")) == ["1", "2"]


async def test_a_half_written_line_was_never_appended(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path)
    fence = await ledger.take("run")
    await ledger.append("groups", "1", {"row": "a"}, fence)
    with (tmp_path / "groups.jsonl").open("a") as file:
        file.write(json.dumps({"key": "2", "fence": 1, "record": {"row": "b"}})[:20])  # the writer died here
    assert await ledger.read("groups") == {"1": {"row": "a"}}
