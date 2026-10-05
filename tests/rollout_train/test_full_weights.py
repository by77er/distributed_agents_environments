"""Full weights: checkpoints of every weight, served in place of the engines' own; an adapter folded into its base;
and a run built from its settings that starts from a full checkpoint, trained over its files."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout_train import Budget
from rollout_train import testing as support
from rollout_train.checkpoints import Checkpoints
from rollout_train.cluster import Cluster
from rollout_train.jobs import Run, ran, run_directory
from rollout_train.launching import Refused
from rollout_train.ledger import FileLedger
from rollout_train.merging import SCOPE, merge
from rollout_train.record import STARTS, scope, table
from rollout_train.run_settings import RunSettings
from rollout_train.serving import SERVING
from rollout_train.stores import FILES, Stores
from rollout_train.trainer import WEIGHTS, Files, Item, Step
from tests.local_ray import LocalRay
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
from tests.rollout_train.support import files


async def test_a_channel_serves_full_weights_in_place_of_its_engines_and_drops_the_adapters_before() -> None:
    channel = support.plain_channel(always=[("yes\n", "stop")])
    engine = channel.engines[0]
    assert isinstance(engine, support.ScriptedEngine)
    assert await channel.publish("adapter-1", "/files/a1") == 1 and channel.adapter == "adapter-1"
    assert await channel.publish("full-2", "/files/f2", 2, full=True) == 2
    assert (channel.adapter, channel.serving) == (None, "full-2")  # (requests sample the weights the engines hold)
    assert engine.told[-2:] == ["weights /files/f2", "remove adapter-1"]
    assert await channel.publish("full-2", "/files/f2", 2, full=True) == 2  # (what is served: nothing changes)
    assert await channel.publish("adapter-3", "/files/a3", 3) == 3 and channel.adapter == "adapter-3"


async def checkpoints_in(tmp_path: Path) -> Checkpoints:
    return Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))


async def test_what_a_checkpoint_builds_on_follows_its_kind(tmp_path: Path) -> None:
    checkpoints = await checkpoints_in(tmp_path)
    fence = await checkpoints.ledger.take(scope("run"))
    lora = await checkpoints.add(fence, "aaaa" * 4, weights=files(tmp_path, "a"), run="run", base="tiny")
    on_lora = await checkpoints.add(fence, "bbbb" * 4, weights=files(tmp_path, "b"), run="run", parents=[lora.id])
    assert (lora.kind, lora.base, on_lora.base) == ("lora", "tiny", "tiny")
    merged = await checkpoints.add(
        fence, "cccc" * 4, weights=files(tmp_path, "c"), run=None, base="tiny-bf16", kind="full", parents=[on_lora.id]
    )
    assert (merged.kind, merged.base) == ("full", "tiny-bf16")  # (a merge: the model it names)
    stacked = await checkpoints.add(fence, "dddd" * 4, weights=files(tmp_path, "d"), run="run", parents=[merged.id])
    assert stacked.base == merged.id  # (an adapter over full weights builds on them)
    full_on = await checkpoints.add(
        fence, "eeee" * 4, weights=files(tmp_path, "e"), run="run", kind="full", parents=[merged.id]
    )
    assert full_on.base == "tiny-bf16"


def folded(base: str, adapter: Path, into: Path) -> dict[str, int]:
    """A merger for the test: the merged model says what it was made of."""
    into.mkdir(parents=True)
    (into / "model.safetensors").write_text(f"{base} + {(adapter / 'model.safetensors').read_text()}")
    return {"layers": 7, "copied": 3}


async def test_a_merge_makes_a_full_checkpoint_from_an_adapter_and_its_base(tmp_path: Path) -> None:
    checkpoints = await checkpoints_in(tmp_path)
    fence = await checkpoints.ledger.take(scope("run"))
    lora = await checkpoints.add(fence, "aaaa" * 4, weights=files(tmp_path, "a"), run="run", base="tiny")
    merging = await checkpoints.ledger.take(SCOPE)
    merger = "tests.rollout_train.test_full_weights:folded"
    merged = await merge(checkpoints, merging, lora.id, base="tiny-bf16", merger=merger, scratch=tmp_path / "s")
    assert (merged.kind, merged.parents, merged.run, merged.base) == ("full", (lora.id,), None, "tiny-bf16")
    assert merged.metrics == {"merged_layers": 7.0, "merged_copied": 3.0} and merged.depth == lora.depth + 1
    assert merged.weights is not None
    written = await checkpoints.files(merged.weights, tmp_path / "read")
    assert (written / "model.safetensors").read_text() == "tiny-bf16 + a"
    # An adapter stacked on the merged weights merges into those weights, and the root stays the model.
    stacked = await checkpoints.add(fence, "bbbb" * 4, weights=files(tmp_path, "b"), run="run", parents=[merged.id])
    again = await merge(checkpoints, merging, stacked.id, merger=merger, scratch=tmp_path / "s")
    assert again.base == "tiny-bf16" and again.weights is not None
    read = await checkpoints.files(again.weights, tmp_path / "read2")
    assert (read / "model.safetensors").read_text().endswith(" + b")
    assert not list((tmp_path / "s").iterdir())  # (what it fetched and wrote there is gone)


class Full:
    """A trainer of every weight that trains nothing: its files say what they were made from."""

    def __init__(self, model: str, *, segment_tokens: int, segments_per_step: int) -> None:
        self.model = model
        self.budget = Budget(segment_tokens, segments_per_step)
        self.weights = "full"

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "model.safetensors").write_text(f"from {parent.weights.name if parent else self.model}")
        return Step({"segments": float(len(batch))})


class Adapters(Full):
    """A trainer of adapters that trains nothing, and remembers what each step began from."""

    began: list[str | None] = []

    def __init__(self, model: str, *, segment_tokens: int, segments_per_step: int) -> None:
        super().__init__(model, segment_tokens=segment_tokens, segments_per_step=segments_per_step)
        self.weights = "lora"

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        Adapters.began.append(parent.weights.parent.name if parent else None)
        return await super().step(batch, seed=seed, parent=parent, into=into)


TRAINERS = """
[trainers.full]
kind = "full"
implementation = "tests.rollout_train.test_full_weights:Full"
gpus = 0.5
colocate_with = "local"
models = ["tiny"]

[trainers.adapters]
kind = "lora"
implementation = "tests.rollout_train.test_full_weights:Adapters"
gpus = 0.5
colocate_with = "local"
models = ["tiny"]
"""
"""Trainers of every weight and of adapters that train nothing, beside the test cluster's."""


def written(tmp_path: Path, name: str, *, full: bool) -> Path:
    """A checkpoint's files as a trainer of that kind writes them: full weights, or an adapter (as PEFT names them)."""
    directory = tmp_path / "made" / name
    directory.mkdir(parents=True)
    if full:
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_text(name)
    else:
        (directory / "adapter_config.json").write_text("{}")
        (directory / "adapter_model.safetensors").write_text(name)
    return directory


async def seeded(tmp_path: Path) -> tuple[Cluster, dict[str, str]]:
    """The test cluster with the two trainers, and in its ledger a merged checkpoint ("merged"), an adapter over the
    model ("plain") and one over the merged weights ("stacked"), each bookmarked by that name."""
    cluster = a_cluster(tmp_path, more=TRAINERS)
    stores = Stores.open(cluster)
    fence = await stores.ledger.take(scope("elsewhere"))
    add = stores.checkpoints.add
    plain = await add(fence, "pppp" * 4, weights=written(tmp_path, "p", full=False), run="elsewhere", base="tiny")
    merged = await add(fence, "mmmm" * 4, weights=written(tmp_path, "m", full=True), run=None, kind="full",
                       parents=[plain.id], base="tiny")  # fmt: skip
    stacked = await add(fence, "ssss" * 4, weights=written(tmp_path, "s", full=False), run="elsewhere",
                        parents=[merged.id])  # fmt: skip
    made = {"plain": plain.id, "merged": merged.id, "stacked": stacked.id}
    for name, id in made.items():
        await stores.registry.bookmark(name, id)
    return cluster, made


async def a_run(cluster: Cluster, trainer: str, start: str) -> Run:
    """A run of the words on the test cluster, trained by `trainer` from the checkpoint `start` names."""
    stores = Stores.open(cluster)
    name = f"{trainer}-{start}"
    settings: dict[str, JsonValue] = {
        **POLICY, "kind": "train", "name": name, "environment": WORDS, "trainer.provider": trainer, "start": start,
        "groups": 1, "trainer.segment_tokens": 900,
    }  # fmt: skip
    return Run(cluster, stores, RunSettings(settings), await stores.registry.create(name))


async def test_a_run_from_a_full_checkpoint_trains_its_files_and_serves_full_weights(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster, made = await seeded(tmp_path)
    run = await a_run(cluster, "full", "merged")
    await ran(run)
    assert run.origin == made["merged"]
    fetched = run_directory(cluster, run.run.id) / "bases" / made["merged"]
    assert (fetched / "model.safetensors").read_text() == "m"  # (what the trainer is made over)
    mine = [each for each in await run.checkpoints.all() if each.run == run.run.id]
    assert mine and all(each.kind == "full" and each.base == "tiny" for each in mine)
    served = [each for each in (await run.ledger.read(table(run.run.id, SERVING))).values() if each.get("checkpoint")]
    assert served and {each["kind"] for each in served} == {"full"}  # (served as full weights)


async def test_adapters_trained_from_full_weights_begin_new_over_them(tmp_path: Path, local_ray: LocalRay) -> None:
    cluster, made = await seeded(tmp_path)
    run = await a_run(cluster, "adapters", "merged")
    await ran(run)
    mine = [each for each in await run.checkpoints.all() if each.run == run.run.id]
    assert mine and all(each.kind == "lora" and each.base == made["merged"] for each in mine)


async def test_a_run_from_an_adapter_over_full_weights_is_trained_over_those_weights(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster, made = await seeded(tmp_path)
    run = await a_run(cluster, "adapters", "stacked")
    await ran(run)
    assert run.origin == made["stacked"]
    fetched = run_directory(cluster, run.run.id) / "bases" / made["merged"]
    assert (fetched / "model.safetensors").read_text() == "m"


async def test_every_weight_is_not_trained_from_an_adapter_before_it_is_merged(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster, _ = await seeded(tmp_path)
    run = await a_run(cluster, "full", "plain")
    with pytest.raises(Refused) as refused:
        await ran(run)
    assert any(each.key == "start" and "merge" in each.reason for each in refused.value.refusals)


async def test_a_checkpoint_kept_in_another_runs_blob_store_is_read_from_there(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    theirs = Checkpoints(ledger, FileBlobStore(tmp_path / "their-blobs"))
    fence = await ledger.take(scope("theirs"))
    where: JsonValue = {"kind": FILES, "directory": str(tmp_path / "their-blobs")}
    await ledger.append(
        table("theirs", STARTS), str(fence.number), {"blobs": where}, fence
    )  # (as every run's start says)
    made = await theirs.add(fence, "tttt" * 4, weights=files(tmp_path, "t"), run="theirs", base="a-checkpoint")
    assert made.weights is not None
    mine = Checkpoints(ledger, FileBlobStore(tmp_path / "my-blobs"))
    read = await mine.files(made.weights, tmp_path / "read")
    assert (read / "model.safetensors").read_text() == "t"
    with pytest.raises(FileNotFoundError):  # (a file no store has)
        await Checkpoints(FileLedger(tmp_path / "other"), FileBlobStore(tmp_path / "my-blobs")).files(
            made.weights, tmp_path / "again"
        )


async def test_a_checkpoints_files_are_kept_and_fetched_without_a_copy_of_their_bytes(tmp_path: Path) -> None:
    checkpoints = await checkpoints_in(tmp_path)
    fence = await checkpoints.ledger.take(scope("run"))
    made = files(tmp_path, "w")
    added = await checkpoints.add(fence, "wwww" * 4, weights=made, run="run", base="a-checkpoint")
    assert added.weights is not None
    (reference,) = added.weights.files.values()
    blob = Path(reference.uri.removeprefix("file://"))
    assert os.path.samefile(blob, made / "model.safetensors")  # noqa: ASYNC240 (what the trainer wrote is the blob)
    fetched = await checkpoints.files(added.weights, tmp_path / "elsewhere" / "weights")
    assert os.path.samefile(blob, fetched / "model.safetensors")  # noqa: ASYNC240
