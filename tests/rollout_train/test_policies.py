"""Policies: versions that are only added, named the same everywhere, with their files kept unpacked."""

from pathlib import Path

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.ledger import Fenced, FileLedger
from rollout_train.policies import Policies, kept, named, parsed


def checkpoint(directory: Path, text: str) -> Path:
    (directory / "nested").mkdir(parents=True)
    (directory / "adapter_config.json").write_text('{"r": 32}')
    (directory / "nested" / "adapter_model.safetensors").write_text(text)
    return directory


async def test_versions_are_added_in_order_and_name_where_their_files_are(tmp_path: Path) -> None:
    policies = Policies(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    writer = await policies.writer("swarm")
    assert await policies.head("swarm") is None
    optimizer = tmp_path / "optimizer.pt"
    optimizer.write_text("moments 1")
    first = await policies.add(writer, "swarm", 1, weights=checkpoint(tmp_path / "a", "weights 1"), state=optimizer)
    second = await policies.add(
        writer, "swarm", 2, weights=checkpoint(tmp_path / "b", "weights 2"), parent=first.name, metrics={"kl": 0.01}
    )
    assert (first.name, second.name, second.parent) == ("swarm@1", "swarm@2", "swarm@1")
    assert [version.number for version in await policies.versions("swarm")] == [1, 2]
    assert await policies.head("swarm") == second == await policies.version("swarm@2")
    # Each file is a blob of its own, by its path in the checkpoint: nothing is packed.
    assert sorted(first.weights.files) == ["adapter_config.json", "nested/adapter_model.safetensors"]
    assert first.state is not None and list(first.state.files) == ["optimizer.pt"]
    assert first.weights.files["adapter_config.json"] == second.weights.files["adapter_config.json"]  # shared

    # Another process, another machine: the files come back whole, from the blobs.
    elsewhere = Policies(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    fetched = await elsewhere.files((await elsewhere.version("swarm@2")).weights, tmp_path / "cache" / "swarm@2")
    assert (fetched / "nested" / "adapter_model.safetensors").read_text() == "weights 2"
    assert not list((tmp_path / "cache").glob(".fetching-*"))


async def test_adding_a_version_again_changes_nothing_and_a_replaced_writer_adds_none(tmp_path: Path) -> None:
    policies = Policies(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    writer = await policies.writer("swarm")
    first = await policies.add(writer, "swarm", 1, weights=checkpoint(tmp_path / "a", "weights 1"))
    again = await policies.add(writer, "swarm", 1, weights=checkpoint(tmp_path / "b", "other weights"))
    assert again == first and len(await policies.versions("swarm")) == 1  # a step redone after a crash
    await policies.writer("swarm")  # a new reactor takes over
    with pytest.raises(Fenced):
        await policies.add(writer, "swarm", 2, weights=tmp_path / "a")
    fork = await policies.writer("swarm-fork")
    forked = await policies.add(fork, "swarm-fork", 1, weights=tmp_path / "a", parent=first.name)
    assert forked.parent == "swarm@1" and forked.weights == first.weights  # a fork costs no new blobs


async def test_names_say_the_policy_and_the_number(tmp_path: Path) -> None:
    assert named("curriculum-9", 17) == "curriculum-9@17" and parsed("curriculum-9@17") == ("curriculum-9", 17)
    with pytest.raises(ValueError, match="policy@number"):
        parsed("step-17")
    single = tmp_path / "weights.bin"
    single.write_text("w")
    assert list((await kept(single, FileBlobStore(tmp_path / "blobs"))).files) == ["weights.bin"]


async def test_saves_thin_out_with_age_and_weights_stay(tmp_path: Path) -> None:
    from rollout_train.policies import Retention

    blobs = FileBlobStore(tmp_path / "blobs")
    policies = Policies(FileLedger(tmp_path / "ledger"), blobs)
    writer = await policies.writer("miner")
    for number in range(1, 13):
        here = tmp_path / f"v{number}"
        (here / "weights").mkdir(parents=True)
        (here / "weights" / "adapter.bin").write_text(f"weights {number}")
        (here / "weights" / "config.json").write_text("{}")  # the same bytes in every version: one blob
        (here / "state").mkdir()
        (here / "state" / "optimizer.bin").write_text(f"moments {number}")
        await policies.add(writer, "miner", number, weights=here / "weights", state=here / "state")

    released = await policies.thin(writer, "miner", Retention(recent=3, every=5))
    assert released == [f"miner@{number}" for number in (1, 2, 3, 4, 6, 7, 8, 9)]
    versions = await policies.versions("miner")
    assert [version.number for version in versions if version.state is not None] == [5, 10, 11, 12]
    gone = next(version for version in versions if version.number == 1)
    assert gone.state is None and gone.released is not None and (await policies.version("miner@1")).state is None
    stored = {path.name for path in (tmp_path / "blobs").rglob("*") if path.is_file()}
    for version in versions:  # every version's weights are kept, and only the kept versions' state
        assert {blob.sha256 for blob in version.weights.files.values()} <= stored
        if version.state is not None:
            assert {blob.sha256 for blob in version.state.files.values()} <= stored
    assert len(stored) == 12 + 1 + 4
    assert await policies.thin(writer, "miner", Retention(recent=3, every=5)) == []  # (and again changes nothing)
