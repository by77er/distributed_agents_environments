# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The trainer's processes keep each step's full state after answering it (`rollout_lora.workers`): copied off the
GPU in the step, in the same order in every process, and kept in a blob store from host memory in a thread of each
process's own, one state at a time, while the next step trains. What is kept is the state the step left, whatever the
next step does meanwhile, on one process and on two; an adapter's float32 copy and its optimizer's state go on
bitwise as the processes would from memory; full weights' shares, kept from each process's memory, are read back by
another number of processes; an upload that fails is tried again, and a state never kept is said; a snapshot that host
memory cannot hold is written to disk and kept from there."""

import asyncio
import io
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load, load_file

from rollout.contracts import BlobReference
from rollout.harness.blobs import FileBlobStore
from rollout_lora.full import FullPolicy
from rollout_lora.layers import adapter_tensors
from rollout_lora.policy import Policy
from rollout_lora.resident import Workers
from rollout_lora.settings import LoraSettings
from rollout_lora.workers import MASTER, OPTIMIZER, SEED, SHARDS, SPILLED
from rollout_objectives.step import PolicyStep
from rollout_train.stores import FILES
from rollout_train.trainer import Files, StateLost
from tests.rollout_lora.test_sharded import COMPARED, SETTINGS, batch, close

STORES = "tests.rollout_lora.stores"


def stored(directory: Path, files: dict[str, Any]) -> dict[str, bytes]:
    """The bytes of what the processes kept in the store of files under `directory`, by path within the state."""
    store = FileBlobStore(directory)
    return {path: asyncio.run(store.read(BlobReference.model_validate(each))) for path, each in files.items()}


def same_optimizer(one: dict[str, Any], two: dict[str, Any]) -> None:
    assert one["param_groups"] == two["param_groups"] and sorted(one["state"]) == sorted(two["state"])
    for index, values in one["state"].items():
        for key, value in values.items():
            assert torch.equal(value, two["state"][index][key]), (index, key)


@pytest.mark.parametrize("count", [1, 2])
def test_a_state_kept_after_its_step_is_the_one_the_step_left_while_the_next_trains(
    tiny: str, tmp_path: Path, count: int
) -> None:
    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(SEED)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    # Written before the processes answer: what each step left.
    written = Workers(tiny, settings, "lora", count, device="cpu")
    try:
        asyncio.run(written.step(given, seed=0, parent=None, into=tmp_path / "written" / "first"))
        made = Files(tmp_path / "written" / "first" / "weights", tmp_path / "written" / "first" / "state")
        asyncio.run(written.step(given, seed=1, parent=made, into=tmp_path / "written" / "second"))
    finally:
        written.close()
    # Kept after they answer, in a store whose puts take a second each: the second step trains while the first's
    # state is kept, and takes its own snapshot once that is kept.
    keeping = Workers(tiny, settings, "lora", count, device="cpu")
    keeping.keep = {"kind": f"{STORES}:SlowStore", "directory": str(tmp_path / "blobs"), "delay": 1.0}
    try:
        first = asyncio.run(keeping.step(given, seed=0, parent=None, into=tmp_path / "kept" / "first"))
        assert keeping.keeping("first") and not (tmp_path / "kept" / "first" / "state" / OPTIMIZER).exists()
        made = Files(tmp_path / "kept" / "first" / "weights", tmp_path / "kept" / "first" / "state")
        second = asyncio.run(keeping.step(given, seed=1, parent=made, into=tmp_path / "kept" / "second"))
        kept = [keeping.kept_state("first"), keeping.kept_state("second")]
    finally:
        keeping.close()
    assert first["snapshot_in_memory"] == 1.0 and second["loaded_from_files"] == 0.0
    assert second["state_wait_seconds"] > 0.5  # (the first's state was being kept as the second step ended)
    assert {"train_seconds", "save_adapter_seconds", "snapshot_seconds"} <= set(first)
    for name, files in zip(("first", "second"), kept, strict=True):
        assert sorted(files) == [MASTER, OPTIMIZER]
        found = stored(tmp_path / "blobs", files)
        reference = tmp_path / "written" / name / "state"
        master, wrote = load(found[MASTER]), load_file(str(reference / MASTER))
        assert set(master) == set(wrote) and all(torch.equal(master[key], wrote[key]) for key in wrote), name
        same_optimizer(torch.load(io.BytesIO(found[OPTIMIZER]), weights_only=True),
                       torch.load(reference / OPTIMIZER, weights_only=True))  # fmt: skip


def test_two_processes_go_on_from_the_float32_copy_bitwise_as_from_memory(tiny: str, tmp_path: Path) -> None:
    # (the weights engines load are bfloat16: a step from them would not go on as the processes do)
    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(SEED)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    found: dict[bool, dict[str, torch.Tensor]] = {}
    for kept in (True, False):  # (kept: the second step from memory; else from the first's files)
        workers = Workers(tiny, settings, "lora", 2, device="cpu", kept=kept)
        here = tmp_path / str(kept)
        try:
            asyncio.run(workers.step(given, seed=0, parent=None, into=here / "first"))
            made = Files(here / "first" / "weights", here / "first" / "state")
            second = asyncio.run(workers.step(given, seed=1, parent=made, into=here / "second"))
        finally:
            workers.close()
        assert second["loaded_from_files"] == float(not kept)
        found[kept] = load_file(str(here / "second" / "state" / MASTER))
        served = load_file(str(here / "second" / "weights" / "adapter_model.safetensors"))
        assert all(served[key].dtype == torch.bfloat16 for key in served)
    assert all(torch.equal(found[True][key], found[False][key]) for key in found[True])


def test_every_weight_s_state_kept_from_each_process_s_memory_is_read_by_another_count(
    tiny: str, tmp_path: Path
) -> None:
    changed: dict[str, Any] = {**SETTINGS, "learning_rate": 1e-4}
    settings = LoraSettings(**changed)
    policy = FullPolicy.load(tiny, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    workers = Workers(tiny, settings, "full", 2, device="cpu")
    workers.keep = {"kind": FILES, "directory": str(tmp_path / "blobs")}
    try:
        first = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
        files = workers.kept_state("first")
    finally:
        workers.close()
    assert first["snapshot_in_memory"] == 1.0 and not (tmp_path / "first" / "state" / SHARDS).exists()
    names = [path.removeprefix(f"{SHARDS}/") for path in files]
    assert ".metadata" in names and any(each.startswith("__0_") for each in names)
    assert any(each.startswith("__1_") for each in names)  # (each process kept its own shares)
    state = tmp_path / "first" / "state"
    for path, data in stored(tmp_path / "blobs", files).items():
        (state / path).parent.mkdir(parents=True, exist_ok=True)
        (state / path).write_bytes(data)
    three = Workers(tiny, settings, "full", 3, device="cpu")
    try:
        second = asyncio.run(three.step(given, seed=1, parent=Files(tmp_path / "first" / "weights", state),
                                        into=tmp_path / "second"))  # fmt: skip
    finally:
        three.close()
    assert second["loaded_from_files"] == 1.0
    alone = PolicyStep(policy, settings)
    alone.step(given, seed=0)
    close(
        PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer).step(given, seed=1), second, COMPARED
    )


def test_an_upload_that_fails_is_tried_again_and_a_state_never_kept_is_lost(tiny: str, tmp_path: Path) -> None:
    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(1)
    given = batch(Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu").logprobs)
    workers = Workers(tiny, settings, "lora", 1, device="cpu")
    workers.keep = {"kind": f"{STORES}:FlakyStore", "directory": str(tmp_path / "blobs"), "failures": 2}
    try:
        asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
        assert sorted(workers.kept_state("first")) == [MASTER, OPTIMIZER]  # (refused twice, then kept)
        workers.keep = {"kind": f"{STORES}:NoSuchStore", "directory": str(tmp_path / "blobs")}
        made = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
        asyncio.run(workers.step(given, seed=1, parent=made, into=tmp_path / "second"))
        with pytest.raises(StateLost, match="NoSuchStore"):
            workers.kept_state("second")
        assert workers.holding is not None  # (the processes go on: the next step goes on from what they hold)
    finally:
        workers.close()


def test_a_snapshot_host_memory_cannot_hold_is_written_to_disk_and_kept_from_there(
    tiny: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rollout_lora.workers as workers
    from rollout_lora.workers import Asked

    class One:
        rank, size = 0, 1

        def gathered(self, value: Any) -> list[Any]:
            return [value]

    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(SEED)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    stepping = PolicyStep(policy, settings)
    stepping.step(given, seed=0)
    held: Any = getattr(workers, "_Held")(policy, stepping.optimizer, False)  # noqa: B009
    location = {"kind": FILES, "directory": str(tmp_path / "blobs")}
    asked = Asked(tiny, settings, "lora", given, 0, None, tmp_path / "step", "step:held", keep=location)
    snapshot_of: Any = getattr(workers, "_snapshot")  # noqa: B009
    sent: list[Any] = []
    keeper: Any = getattr(workers, "_Keeper")(sent.append)  # noqa: B009
    for available, in_memory in ((float(2**40), True), (1000.0, False)):
        monkeypatch.setattr(workers, "host_memory", lambda available=available: available)
        snapshot = snapshot_of(asked, held, adapter_tensors(policy.model), One())
        assert snapshot.in_memory == in_memory
        assert all(isinstance(each, Path) for each in snapshot.files.values()) == (not in_memory)
        keeper.keep("step", snapshot, location)
        keeper.wait()
    assert (tmp_path / "step" / SPILLED / OPTIMIZER).exists()  # (written to disk, and kept from there)
    (_, _, from_memory), (_, _, from_disk) = sent
    assert sorted(from_memory["files"]) == sorted(from_disk["files"]) == [MASTER, OPTIMIZER]
    memory, disk = stored(tmp_path / "blobs", from_memory["files"]), stored(tmp_path / "blobs", from_disk["files"])
    assert memory[MASTER] == disk[MASTER]
    same_optimizer(torch.load(io.BytesIO(memory[OPTIMIZER]), weights_only=True),
                   torch.load(io.BytesIO(disk[OPTIMIZER]), weights_only=True))  # fmt: skip


async def test_a_training_pod_answers_with_the_weights_and_completes_the_state_its_trainer_keeps(
    tiny: str, tmp_path: Path
) -> None:
    import httpx

    from rollout_lora.trainer import LoraTrainer
    from rollout_train.checkpoints import Checkpoint, Checkpoints
    from rollout_train.ledger import FileLedger
    from rollout_train.pods import RemoteTrainer
    from rollout_train.pods.training import TrainerService, app
    from rollout_train.trainer import HELD, Budget

    trainer = LoraTrainer(tiny, gpus=1, **SETTINGS)
    trainer._process.device = "cpu"  # pyright: ignore[reportPrivateUsage]  (no GPU here)
    trainer.keep_in({"kind": FILES, "directory": str(tmp_path / "blobs")})
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    service = TrainerService(trainer, checkpoints, tmp_path / "pod")
    torch.manual_seed(1)
    given = batch(Policy.load(tiny, rank=4, alpha=8.0, device="cpu").logprobs)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(service)), base_url="https://pod")
    remote = RemoteTrainer("https://pod", checkpoints, budget=Budget(), client=client, every=0.05)
    try:
        made = await remote.made(given, seed=0, parent=None, into="kmnopqrstuvwxyzk")
        whole = await remote.state("kmnopqrstuvwxyzk")
        parent = Checkpoint("kmnopqrstuvwxyzk", made.weights, state=made.state, state_complete=made.complete)
        second = await remote.made(given, seed=1, parent=parent, into="lmnopqrstuvwxyzl")
    finally:
        await client.aclose()
        trainer.close()
    assert sorted(made.weights.files) == ["adapter_config.json", "adapter_model.safetensors"]
    assert {"step_seconds", "upload_adapter_seconds", "train_seconds", "snapshot_seconds"} <= set(made.metrics)
    assert sorted(whole.files) == sorted([HELD, "minibatches.jsonl", MASTER, OPTIMIZER])
    assert second.metrics["loaded_from_files"] == 0.0  # (from what the processes held: the state was not needed)
    served = load(await checkpoints.blobs.read(made.weights.files["adapter_model.safetensors"]))
    master = load(await checkpoints.blobs.read(whole.files[MASTER]))
    assert all(served[key].dtype == torch.bfloat16 and master[key].dtype == torch.float32 for key in master)
    assert all(torch.equal(served[key], master[key].to(torch.bfloat16)) for key in master)
