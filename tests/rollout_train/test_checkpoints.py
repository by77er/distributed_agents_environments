"""Checkpoints: nodes of a graph that are only added, each saying where it came from, with their files kept unpacked."""

from pathlib import Path

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints, Retention, kept, new_id, short
from rollout_train.ledger import Fenced, FileLedger
from rollout_train.record import scope


def checkpoint(directory: Path, text: str) -> Path:
    (directory / "nested").mkdir(parents=True)
    (directory / "adapter_config.json").write_text('{"r": 32}')
    (directory / "nested" / "adapter_model.safetensors").write_text(text)
    return directory


async def test_a_version_says_where_it_came_from_and_where_its_files_are(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / "blobs"))
    fence = await ledger.take(scope("miner"))
    assert await checkpoints.head("miner") is None
    optimizer = tmp_path / "optimizer.pt"
    optimizer.write_text("moments 1")
    first = await checkpoints.add(
        fence, new_id(), weights=checkpoint(tmp_path / "a", "weights 1"), run="miner", base="qwen", step=1,
        state=optimizer,
    )  # fmt: skip
    second = await checkpoints.add(
        fence, new_id(), weights=checkpoint(tmp_path / "b", "weights 2"), run="miner", step=2, parents=[first.id],
        metrics={"kl": 0.01},
    )  # fmt: skip
    assert (first.parents, first.depth, first.base, first.parent) == ((), 1, "qwen", None)
    assert (second.parent, second.depth, second.base, second.step) == (first.id, 2, "qwen", 2)  # its base is its line's
    assert len(first.id) == 16 and set(first.id) <= set("klmnopqrstuvwxyz") and first.id != second.id
    assert [checkpoint.id for checkpoint in await checkpoints.all()] == [first.id, second.id]
    assert await checkpoints.head("miner") == second == await checkpoints.checkpoint(second.id)
    with pytest.raises(KeyError):
        await checkpoints.checkpoint("nothing")
    # Each file is a blob of its own, by its path in the checkpoint: nothing is packed.
    assert first.weights is not None and second.weights is not None
    assert sorted(first.weights.files) == ["adapter_config.json", "nested/adapter_model.safetensors"]
    assert first.state is not None and list(first.state.files) == ["optimizer.pt"]
    assert first.weights.files["adapter_config.json"] == second.weights.files["adapter_config.json"]  # shared

    # Another process, another machine: the files come back whole, from the blobs.
    elsewhere = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    found = (await elsewhere.checkpoint(second.id)).weights
    assert found is not None
    fetched = await elsewhere.files(found, tmp_path / "cache" / second.id)
    assert (fetched / "nested" / "adapter_model.safetensors").read_text() == "weights 2"
    assert not list((tmp_path / "cache").glob(".fetching-*"))


async def test_adding_a_version_again_changes_nothing_and_a_replaced_run_adds_none(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / "blobs"))
    fence = await ledger.take(scope("miner"))
    decided = new_id()
    first = await checkpoints.add(fence, decided, weights=checkpoint(tmp_path / "a", "weights 1"), run="miner")
    again = await checkpoints.add(fence, decided, weights=checkpoint(tmp_path / "b", "other weights"), run="miner")
    assert again == first and len(await checkpoints.all()) == 1  # a step redone after a crash
    await ledger.take(scope("miner"))  # a new loop takes over
    with pytest.raises(Fenced):
        await checkpoints.add(fence, new_id(), weights=tmp_path / "a", run="miner")
    # Another run forks from it: its checkpoint grows from the first, on the same base, and costs no new blobs.
    other = await ledger.take(scope("fork"))
    forked = await checkpoints.add(other, new_id(), weights=tmp_path / "a", run="fork", parents=[first.id])
    assert forked.parent == first.id and forked.depth == 2 and forked.weights == first.weights
    assert await checkpoints.head("fork") == forked and await checkpoints.head("miner") == first
    learned = await checkpoints.add(other, new_id(), weights=tmp_path / "a", run="fork", parents=[forked.id, first.id])
    assert learned.parents == (forked.id, first.id) and learned.depth == 3  # depth follows the first parent


async def test_a_version_is_shown_by_the_shortest_start_of_its_id_that_no_other_shares(tmp_path: Path) -> None:
    assert short(["kkkkmmmm", "kkkkmnnn", "zzzzzzzz"]) == {
        "kkkkmmmm": "kkkkmm",
        "kkkkmnnn": "kkkkmn",
        "zzzzzzzz": "zzzz",
    }
    assert short(["curriculum-9@1", "curriculum-9@12"]) == {"curriculum-9@1": "@1", "curriculum-9@12": "@12"}
    assert short(["a@1", "b@1", "c@2"]) == {"a@1": "a@1", "b@1": "b@1", "c@2": "@2"}  # (two @1s: each whole)
    assert short([]) == {}
    single = tmp_path / "weights.bin"
    single.write_text("w")
    assert list((await kept(single, FileBlobStore(tmp_path / "blobs"))).files) == ["weights.bin"]


async def test_a_runs_saves_thin_out_with_age_and_what_must_stay_stays(tmp_path: Path) -> None:
    blobs = FileBlobStore(tmp_path / "blobs")
    ledger = FileLedger(tmp_path / "ledger")
    checkpoints = Checkpoints(ledger, blobs)
    fence = await ledger.take(scope("miner"))
    made: list[Checkpoint] = []
    for number in range(1, 13):
        here = tmp_path / f"v{number}"
        (here / "weights").mkdir(parents=True)
        (here / "weights" / "adapter.bin").write_text(f"weights {number}")
        (here / "weights" / "config.json").write_text("{}")  # the same bytes in every checkpoint: one blob
        (here / "state").mkdir()
        (here / "state" / "optimizer.bin").write_text(f"moments {number}")
        parents = [made[-1].id] if made else []
        made.append(
            await checkpoints.add(
                fence, new_id(), weights=here / "weights", state=here / "state", run="miner", parents=parents
            )
        )
    other = await ledger.take(scope("other"))
    theirs = await checkpoints.add(other, new_id(), weights=tmp_path / "v1" / "weights", run="other")

    bookmarked = made[2].id  # depth 3: retention would let it go, but something keeps it
    released = await checkpoints.thin(fence, "miner", Retention(recent=3, every=5), keep={bookmarked})
    assert released == [made[depth - 1].id for depth in (1, 2, 4, 6, 7, 8, 9)]
    every = {checkpoint.id: checkpoint for checkpoint in await checkpoints.all()}
    assert [checkpoint.depth for checkpoint in every.values() if checkpoint.run == "miner" and checkpoint.weights] == [
        3,
        5,
        10,
        11,
        12,
    ]
    assert every[theirs.id].weights is not None  # another run's checkpoints are its own to thin
    gone = every[made[0].id]
    assert gone.weights is None and gone.state is None and gone.released is not None
    assert gone.parents == () and (await checkpoints.checkpoint(made[0].id)).weights is None
    stored = {path.name for path in (tmp_path / "blobs").rglob("*") if path.is_file()}
    for checkpoint in every.values():  # the kept checkpoints' files are all there
        for manifest in (checkpoint.weights, checkpoint.state):
            assert manifest is None or {blob.sha256 for blob in manifest.files.values()} <= stored
    assert (
        await checkpoints.thin(fence, "miner", Retention(recent=3, every=5), keep={bookmarked}) == []
    )  # (again: nothing)
