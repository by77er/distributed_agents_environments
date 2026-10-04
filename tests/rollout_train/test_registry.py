"""The registry: every run and policy has an id that never changes, and a name that can be chosen and changed."""

import json
from pathlib import Path

import pytest

from rollout_train.database import DatabaseLedger, copy
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.record import GROUPS, scope, table
from rollout_train.registry import POLICIES, RUNS, Registry, Taken, find, policy_of, registry_of, run_of


def ledgers(tmp_path: Path) -> list[Ledger]:
    return [FileLedger(tmp_path / "files"), DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")]


@pytest.mark.parametrize("kind", ["files", "database"])
async def test_a_name_is_chosen_changed_and_unique_while_the_id_stays(tmp_path: Path, kind: str) -> None:
    ledger = ledgers(tmp_path)[0 if kind == "files" else 1]
    registry = registry_of(ledger)
    assert registry is not None
    first = await registry.create(RUNS, "diamonds")
    assert first.id.startswith("run_") and first.name == "diamonds"
    renamed = await registry.rename(RUNS, "diamonds", "diamonds, unguided")
    assert renamed.id == first.id and (await find(registry, RUNS, first.id)) == renamed
    assert await find(registry, RUNS, "diamonds") is None  # the old name is free again
    other = await registry.create(RUNS, "diamonds")
    for taken in ("diamonds, unguided", first.id, " ", "a/b", "v@2"):  # another's name or id, or no name at all
        with pytest.raises(Taken):
            await registry.rename(RUNS, other.id, taken)
    assert (await registry.create(POLICIES, "diamonds")).name == "diamonds"  # a policy is named apart from runs
    with pytest.raises(KeyError):
        await registry.rename(RUNS, "nobody", "someone")
    assert [each.name for each in await registry.entries(RUNS)] == ["diamonds, unguided", "diamonds"]


async def test_a_runs_directory_says_which_run_it_is_whatever_it_is_called(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    registry = registry_of(ledger)
    directory = tmp_path / "curriculum-10"
    run = await run_of(directory, ledger, registry)
    assert run.name == "curriculum-10" and run.id.startswith("run_")
    assert json.loads((directory / "run.json").read_text()) == {"id": run.id}
    assert registry is not None
    await registry.rename(RUNS, run.id, "the long one")
    again = await run_of(directory, ledger, registry)  # started again, after a rename
    assert (again.id, again.name) == (run.id, "the long one")
    named = await run_of(tmp_path / "elsewhere", ledger, registry, "chosen")
    assert named.name == "chosen"


async def test_runs_and_policies_from_before_the_registry_keep_their_keys_as_ids(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    registry: Registry | None = registry_of(ledger)
    fence = await ledger.take(scope("curriculum-9"))
    await ledger.append(table("curriculum-9", GROUPS), "1", {"episodes": 4}, fence)
    old = await run_of(tmp_path / "curriculum-9", ledger, registry)
    assert (old.id, old.name) == ("curriculum-9", "curriculum-9")
    policies = FileLedger(tmp_path / "ledger")
    writer = await policies.take("policies/curriculum-9")
    await policies.append("policies/curriculum-9/versions", "1", {"name": "curriculum-9@1"}, writer)
    policy = await policy_of(ledger, registry, "curriculum-9")
    assert (policy.id, policy.name) == ("curriculum-9", "curriculum-9")
    fresh = await policy_of(ledger, registry, "miner")
    assert fresh.id.startswith("policy_") and (await policy_of(ledger, registry, fresh.id)) == fresh


async def test_a_copy_into_a_database_keeps_the_names(tmp_path: Path) -> None:
    files, database = ledgers(tmp_path)
    registry = registry_of(files)
    assert registry is not None
    run = await registry.create(RUNS, "copied")
    fence = await files.take(scope(run.id))
    await files.append(table(run.id, GROUPS), "1", {"episodes": 1}, fence)
    assert isinstance(database, DatabaseLedger)
    await copy(files, database)
    assert await database.registry.entries(RUNS) == [run]
