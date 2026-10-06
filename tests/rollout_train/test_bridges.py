"""Bridges: the path from a checkpoint's format to what a provider loads, the pairs refused and why, the rank a
provider sees, a checkpoint's format read from its files; and each bridge run once per checkpoint, here and as a Ray
task, noted under `CHECKPOINT@BRIDGE`; the verbatim bridges read no file."""

from pathlib import Path
from typing import Any

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.bridges import (
    BRIDGED,
    BRIDGES,
    BRIDGING,
    Bridge,
    NoBridge,
    bridge_of,
    bridged,
    by_name,
    format_of,
    made,
    on_ray,
    path,
    rank_factor,
)
from rollout_train.checkpoints import Checkpoints, Retention
from rollout_train.ledger import Fence, FileLedger
from rollout_train.record import scope
from rollout_train.stores import FILES
from tests.local_ray import LocalRay


def names(found: tuple[Bridge, ...] | NoBridge) -> list[str]:
    assert not isinstance(found, NoBridge), found
    return [each.name for each in found]


def test_each_pair_takes_its_bridge() -> None:
    assert names(path("tinker", {"tinker"})) == ["none"]
    assert names(path("tinker", {"peft", "full"})) == ["peft-from-tinker"]
    assert names(path("peft", {"peft", "full"})) == ["verbatim"]
    assert names(path("full", {"peft", "full"})) == ["full-reload"]
    assert names(path("peft", {"peft"})) == ["verbatim"]


def test_merge_quantize_only_when_asked() -> None:
    assert isinstance(path("peft", {"full"}), NoBridge)
    assert names(path("peft", {"full"}, wanted="merge-quantize")) == ["merge-quantize"]
    assert names(path("peft", {"peft", "full"}, wanted="merge-quantize")) == ["merge-quantize"]
    assert names(path("tinker", {"full"}, wanted="merge-quantize")) == ["peft-from-tinker", "merge-quantize"]
    refused = path("tinker", {"tinker"}, wanted="merge-quantize")
    assert isinstance(refused, NoBridge) and "goes through merge-quantize" in refused.reason


def test_the_refused_pairs_say_why() -> None:
    for source in ("peft", "full"):
        refused = path(source, {"tinker"})
        assert isinstance(refused, NoBridge)
        assert refused.reason == "Tinker samples only checkpoints Tinker trained: there is no upload"
    refused = path("full", {"peft"})
    assert isinstance(refused, NoBridge) and "not an adapter" in refused.reason
    nothing = path("peft", set())
    assert isinstance(nothing, NoBridge) and "base models only" in nothing.reason
    wrong = path("peft", {"peft"}, wanted="sideways")
    assert isinstance(wrong, NoBridge) and "auto or merge-quantize" in wrong.reason


def test_tinkers_adapters_for_qwen35_triple_their_rank() -> None:
    found = path("tinker", {"peft"})
    assert not isinstance(found, NoBridge)
    assert rank_factor(found, "Qwen/Qwen3.5-9B") == 3 and rank_factor(found, "Qwen/Qwen3-0.6B") == 1
    verbatim = path("peft", {"peft"})
    assert not isinstance(verbatim, NoBridge) and rank_factor(verbatim, "Qwen/Qwen3.5-9B") == 1


def test_bridges_declare_their_needs_and_tasks() -> None:
    by_name = {each.name: each for each in BRIDGES}
    assert by_name["none"].task is None
    assert by_name["peft-from-tinker"].network and by_name["peft-from-tinker"].cpus == 2
    assert by_name["merge-quantize"].explicit and by_name["merge-quantize"].memory_gib == 48
    assert by_name["verbatim"].task == by_name["full-reload"].task == "rollout_train.bridges:verbatim"


def test_a_checkpoints_format_is_read_from_its_files() -> None:
    assert format_of(["weights/tinker.json"]) == {"tinker"}
    assert format_of(["adapter_config.json", "adapter_model.safetensors"]) == {"peft"}
    assert format_of(["config.json", "model-00001-of-00002.safetensors", "tokenizer.json"]) == {"full"}
    both = ["weights/tinker.json", "weights/adapter_config.json", "weights/adapter_model.safetensors"]
    assert format_of(both) == {"tinker", "peft"}
    assert format_of(["adapter_config.json"]) == set()


ADAPTER = {"adapter_config.json": b'{"r": 8}', "adapter_model.safetensors": bytes(64)}


async def a_checkpoint(tmp_path: Path, files: dict[str, bytes] = ADAPTER) -> tuple[Checkpoints, Fence, str]:
    """A ledger and blob store with one checkpoint, its weights `files` (by default an adapter in PEFT's layout)."""
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    fence = await checkpoints.ledger.take(scope("run"))
    weights = tmp_path / "made" / "weights"
    weights.mkdir(parents=True)
    for name, content in files.items():
        (weights / name).write_bytes(content)
    record = await checkpoints.add(fence, "kpqxrmtzwvolxqvu", weights=weights, run="run", step=1, base="tiny")
    return checkpoints, fence, record.id


def locations(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Where a Ray worker finds the ledger and the blob store of `a_checkpoint`."""
    return {"directory": str(tmp_path / "ledger")}, {"kind": FILES, "directory": str(tmp_path / "blobs")}


async def test_a_checkpoint_is_bridged_once_for_each_bridge(tmp_path: Path) -> None:
    checkpoints, fence, checkpoint = await a_checkpoint(tmp_path)
    chain = path("peft", {"peft", "full"})
    assert not isinstance(chain, NoBridge)
    manifest = await bridged(checkpoints, fence, checkpoint, chain, tmp_path / "scratch")
    assert sorted(manifest.files) == ["adapter_config.json", "adapter_model.safetensors"]
    record = await checkpoints.checkpoint(checkpoint)
    assert record.weights is not None
    assert {name: blob.sha256 for name, blob in manifest.files.items()} == {
        name: blob.sha256 for name, blob in record.weights.files.items()
    }  # (verbatim: the same files, so the same blobs)
    entry = f"{checkpoint}@verbatim"
    assert list(await checkpoints.ledger.read(BRIDGING)) == [entry] == list(await checkpoints.ledger.read(BRIDGED))
    assert await made(checkpoints.ledger, checkpoint, "verbatim") == manifest
    assert await bridge_of(checkpoints.ledger, checkpoint) == "verbatim"
    again = await bridged(checkpoints, fence, checkpoint, chain, tmp_path / "scratch")
    assert again == manifest and len(await checkpoints.ledger.read(BRIDGING)) == 1  # not bridged again
    assert not (tmp_path / "scratch").exists()  # (the same files: none was read or written)


async def test_a_released_checkpoint_cannot_be_bridged(tmp_path: Path) -> None:
    checkpoints, fence, checkpoint = await a_checkpoint(tmp_path)
    second = tmp_path / "second"
    second.mkdir()
    (second / "w").write_bytes(b"1")
    later = await checkpoints.add(fence, "zzzzzzzzzzzzzzzz", weights=second, run="run", step=2, parents=[checkpoint])
    await checkpoints.thin(fence, "run", Retention(recent=1, every=0), keep={later.id})
    with pytest.raises(ValueError, match="weights were deleted"):
        await bridged(checkpoints, fence, checkpoint, (by_name("verbatim"),), tmp_path / "scratch")


async def test_a_verbatim_bridge_on_ray_is_noted_with_the_checkpoint_s_own_files(tmp_path: Path) -> None:
    checkpoints, fence, checkpoint = await a_checkpoint(tmp_path)
    ledger_at, blobs_at = locations(tmp_path)
    scratch = str(tmp_path / "worker")
    manifest = await on_ray(ledger_at, blobs_at, fence, checkpoint, (by_name("verbatim"),), scratch=scratch)
    assert manifest == (await checkpoints.checkpoint(checkpoint)).weights  # (the same blobs)
    assert await made(checkpoints.ledger, checkpoint, "verbatim") == manifest  # (noted in the ledger)
    begun: Any = (await checkpoints.ledger.read(BRIDGING))[f"{checkpoint}@verbatim"]
    assert begun["bridge"] == "verbatim"
    assert not (tmp_path / "worker").exists()  # (no task ran: nothing was read to this machine)


async def test_the_path_chosen_is_what_runs(tmp_path: Path, local_ray: LocalRay) -> None:
    pointer = b'{"sampler": "tinker://run/sampler_weights/one", "state": "tinker://run/weights/one"}'
    checkpoints, fence, checkpoint = await a_checkpoint(tmp_path, {"tinker.json": pointer})
    record = await checkpoints.checkpoint(checkpoint)
    assert record.weights is not None and format_of(record.weights.files) == {"tinker"}
    served_as_it_is = path("tinker", {"tinker"})
    assert not isinstance(served_as_it_is, NoBridge) and [each.name for each in served_as_it_is] == ["none"]
    ledger_at, blobs_at = locations(tmp_path)
    assert await on_ray(ledger_at, blobs_at, fence, checkpoint, served_as_it_is) == record.weights
    assert not await checkpoints.ledger.read(BRIDGING)  # (Tinker's sampler reads the pointer: nothing is written)


async def test_refused_pairs_run_nothing_and_a_failed_bridge_makes_no_files(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    from ray.exceptions import RayTaskError

    checkpoints, fence, checkpoint = await a_checkpoint(tmp_path, {"config.json": b"{}", "model.safetensors": b"0"})
    refused = path("full", {"peft"})
    assert isinstance(refused, NoBridge) and "not an adapter" in refused.reason
    assert isinstance(path("full", {"tinker"}), NoBridge)
    ledger_at, blobs_at = locations(tmp_path)
    tinker = by_name("peft-from-tinker")  # (full weights are no Tinker checkpoint: its task fails)
    with pytest.raises(RayTaskError, match="name no sampler checkpoint"):
        await on_ray(ledger_at, blobs_at, fence, checkpoint, (tinker,), scratch=str(tmp_path / "worker"))
    assert list(await checkpoints.ledger.read(BRIDGING)) == [f"{checkpoint}@peft-from-tinker"]
    assert not await checkpoints.ledger.read(BRIDGED)
