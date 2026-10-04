"""The registry: a run's id never changes and its name can be chosen and changed; bookmarks name checkpoints; and a
reference to a checkpoint finds one by a bookmark, by the run and step that made it, or by its id."""

import json
from pathlib import Path

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints
from rollout_train.database import DatabaseLedger, copy
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.record import GROUPS, STEPS, scope, table
from rollout_train.registry import BASE, Registry, Taken, found, names, registry_of, resolved, run_of
from rollout_train.settings import desired_settings_of


def ledger_of(tmp_path: Path, kind: str) -> Ledger:
    return FileLedger(tmp_path / "files") if kind == "files" else DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")


def registered(ledger: Ledger) -> Registry:
    registry = registry_of(ledger)
    assert registry is not None
    return registry


@pytest.mark.parametrize("kind", ["files", "database"])
async def test_a_runs_name_is_chosen_changed_and_unique_while_its_id_stays(tmp_path: Path, kind: str) -> None:
    registry = registered(ledger_of(tmp_path, kind))
    first = await registry.create("diamonds")
    assert first.id.startswith("run_") and first.name == "diamonds"
    renamed = await registry.rename("diamonds", "diamonds, unguided")
    assert renamed.id == first.id and found(await registry.runs(), first.id) == renamed
    assert found(await registry.runs(), "diamonds") is None  # the old name is free again
    other = await registry.create("diamonds")
    for taken in ("diamonds, unguided", first.id, " ", "a/b", "v@2", "run:3", BASE):  # another's, or no name at all
        with pytest.raises(Taken):
            await registry.rename(other.id, taken)
    assert (await registry.rename(other.id, "diamonds")).name == "diamonds"  # (its own name again: allowed)
    with pytest.raises(KeyError):
        await registry.rename("nobody", "someone")
    with pytest.raises(Taken):
        await registry.create("anything", first.id)  # an id is had once
    assert [each.name for each in await registry.runs()] == ["diamonds, unguided", "diamonds"]


@pytest.mark.parametrize("kind", ["files", "database"])
async def test_a_bookmark_names_a_version_and_moves(tmp_path: Path, kind: str) -> None:
    registry = registered(ledger_of(tmp_path, kind))
    assert await registry.bookmarks() == []
    first = await registry.bookmark("best", "kkkkkkkk")
    moved = await registry.bookmark("best", "mmmmmmmm")
    await registry.bookmark("alpha", "kkkkkkkk")
    assert first.checkpoint == "kkkkkkkk" and moved.checkpoint == "mmmmmmmm"
    assert [(mark.name, mark.checkpoint) for mark in await registry.bookmarks()] == [
        ("alpha", "kkkkkkkk"),
        ("best", "mmmmmmmm"),
    ]
    with pytest.raises(Taken):
        await registry.bookmark("a:b", "kkkkkkkk")
    await registry.unbookmark("alpha")
    with pytest.raises(KeyError):
        await registry.unbookmark("alpha")
    assert await names(registry) == {"runs": {}, "bookmarks": {"best": "mmmmmmmm"}, "suites": {}}


async def test_a_runs_directory_says_which_run_it_is_whatever_it_is_called(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    registry = registered(ledger)
    directory = tmp_path / "curriculum-10"
    run = await run_of(directory, ledger, registry)
    assert run.name == "curriculum-10" and run.id.startswith("run_")
    assert json.loads((directory / "run.json").read_text()) == {"id": run.id}
    await registry.rename(run.id, "the long one")
    again = await run_of(directory, ledger, registry)  # started again, after a rename
    assert (again.id, again.name) == (run.id, "the long one")
    assert (await run_of(tmp_path / "elsewhere", ledger, registry, "chosen")).name == "chosen"
    alone = await run_of(tmp_path / "unregistered", ledger, None)  # (no registry: the directory's name)
    assert (alone.id, alone.name) == ("unregistered", "unregistered")


async def made(
    ledger: Ledger, tmp_path: Path, run: str, steps: int, after: Checkpoint | None = None
) -> list[Checkpoint]:
    """`steps` checkpoints a run made, each at a step of its own (recorded as its loop records them)."""
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / "blobs"))
    weights = tmp_path / f"weights-{run}.bin"  # (a kept file is the store's: read-only)
    weights.write_text("w")
    fence = await ledger.take(scope(run))
    line: list[Checkpoint] = []
    for step in range(1, steps + 1):
        parent = line[-1] if line else after
        checkpoint = await checkpoints.add(
            fence, f"{run[0] * 4}{'klmnopqrstuvwxyz'[step]}{'z' * 11}", weights=weights, run=run, step=step,
            parents=[parent.id] if parent else [],
        )  # fmt: skip
        await ledger.append(table(run, GROUPS), str(step), {"episodes": 1}, fence)
        await ledger.append(table(run, STEPS), str(step), {"makes": checkpoint.id, "groups": [step]}, fence)
        line.append(checkpoint)
    return line


async def test_a_reference_finds_a_version_by_bookmark_run_and_step_or_id(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    registry = registered(ledger)
    named = await registry.create("scout")
    scout = await made(ledger, tmp_path, named.id, 3)
    fork = await made(ledger, tmp_path, (await registry.create("fork", "nnnn-fork")).id, 1, after=scout[1])
    await registry.bookmark("best", scout[1].id)
    assert await resolved(ledger, registry, BASE) is None
    assert await resolved(ledger, registry, "best") == scout[1].id
    assert await resolved(ledger, registry, "scout:3") == scout[2].id  # by its name
    assert await resolved(ledger, registry, f"{named.id}:1") == scout[0].id  # or its id
    assert await resolved(ledger, registry, "scout") == scout[2].id  # its newest
    assert await resolved(ledger, registry, "nnnn-fork") == fork[0].id  # (by its id)
    assert await resolved(ledger, registry, scout[0].id) == scout[0].id
    assert await resolved(ledger, registry, scout[0].id[:6]) == scout[0].id  # a start no other id shares
    for unknown, says in [
        ("scout:9", "no checkpoint at step 9"),
        (scout[0].id[:4], "more than one checkpoint"),  # every scout checkpoint begins the same
        ("xx", "no checkpoint"),
        (fork[0].id[:2], "too short"),
        ("nothing-like-it", "no checkpoint"),
    ]:
        with pytest.raises(KeyError, match=says):
            await resolved(ledger, registry, unknown)
    await registry.create("idle")
    with pytest.raises(KeyError, match="has made no checkpoint"):
        await resolved(ledger, registry, "idle")


async def test_a_copy_into_a_database_keeps_the_runs_the_bookmarks_the_suites_and_the_wanted_settings(
    tmp_path: Path,
) -> None:
    files, database = ledger_of(tmp_path, "files"), ledger_of(tmp_path, "database")
    registry = registered(files)
    run = await registry.create("copied")
    mark = await registry.bookmark("best", "kkkkkkkk")
    await registry.point_suite("math", "math@2")
    wanted = desired_settings_of(files)
    assert wanted is not None
    await wanted.want(run.id, {"groups_per_step": 8})
    fence = await files.take(scope(run.id))
    await files.append(table(run.id, GROUPS), "1", {"episodes": 1}, fence)
    assert isinstance(database, DatabaseLedger)
    await copy(files, database)
    assert await database.registry.runs() == [run]
    assert [(each.name, each.checkpoint) for each in await database.registry.bookmarks()] == [
        (mark.name, mark.checkpoint)
    ]
    assert [(each.name, each.version) for each in await database.registry.suites()] == [("math", "math@2")]
    copied = await database.desired_settings.desired(run.id)
    assert copied is not None and copied.settings == {"groups_per_step": 8}


def test_the_command_lists_versions_and_moves_bookmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from rollout_train.cli import main

    ledger = FileLedger(tmp_path / "ledger")

    async def set_up() -> list[Checkpoint]:
        named = await registered(ledger).create("scout")
        return await made(ledger, tmp_path, named.id, 2)

    line = asyncio.run(set_up())

    def run(*arguments: str) -> str:
        monkeypatch.setattr("sys.argv", ["rollout", *arguments, "--ledger", str(tmp_path / "ledger")])
        main()
        return capsys.readouterr().out

    assert run("bookmark", "best", "scout:1") == f"best is {line[0].id}\n"
    listed = run("checkpoints").splitlines()
    assert len(listed) == 2 and "scout:1" in listed[1] and "[best]" in listed[1] and "the base model" in listed[1]
    assert "scout:2" in listed[0] and "from " + line[0].id[:5] in listed[0]
    assert "is called scouting" in run("rename", "scout", "scouting")
    assert run("bookmark", "best", "--delete") == "no bookmark best any more\n"
    with pytest.raises(SystemExit, match="says no checkpoint"):
        run("bookmark", "best", "nothing-like-it")
