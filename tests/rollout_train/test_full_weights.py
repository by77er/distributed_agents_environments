"""Full weights: checkpoints of every weight, served in place of the engines' own; an adapter folded into its base;
and a run that starts from a full checkpoint loading its files as the model it serves and trains."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.environment import binding_for
from rollout.harness.blobs import FileBlobStore
from rollout_train import Budget, train
from rollout_train import testing as support
from rollout_train.checkpoints import Checkpoints
from rollout_train.ledger import FileLedger
from rollout_train.merging import SCOPE, merge
from rollout_train.profile import Profile
from rollout_train.record import STARTS, scope, table
from rollout_train.stores import FILES
from rollout_train.trainer import WEIGHTS, Files, Item, Step
from tests.rollout_train.rollouts.games import words
from tests.rollout_train.support import a_ledger, a_profile, files


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


async def test_a_run_from_a_full_checkpoint_serves_and_trains_its_files_and_publishes_full_weights(
    tmp_path: Path,
) -> None:
    shared, made = await a_ledger(tmp_path)
    support.STARTED.clear()
    async with Profile.load(a_profile(tmp_path, shared, "Full", "merged")).open() as platform:
        fetched = tmp_path / "Full-merged" / "bases" / made["merged"]
        assert platform.origin == made["merged"] and (fetched / "model.safetensors").read_text() == "m"
        trainer = platform.trainer
        assert trainer is not None and trainer.weights == "full"
        assert support.STARTED[0].told[0].startswith(f"started {fetched}")  # (the trained channel's engines)
        assert support.STARTED[2].told[0].startswith("started another-checkpoint")  # (not the other channel's)
        binding = binding_for(words, "policy", platform.tool_bindings)
        await train(
            words, trainer, platform.checkpoints, start=platform.origin, channel="policy",
            directory=tmp_path / "Full-merged" / "checkpoints", publish=platform.publish, run=platform.run.id,
            groups=1, binding=binding, made=platform.made,
        )  # fmt: skip
        mine = [each for each in await platform.checkpoints.all() if each.run == platform.run.id]
        assert mine and all(each.kind == "full" and each.base == "a-checkpoint" for each in mine)
        policy = platform.channels["policy"]
        assert policy.serving == mine[-1].id and policy.adapter is None  # (served as full weights)
        engine = policy.engines[0]
        assert isinstance(engine, support.ScriptedEngine)
        assert any(each.startswith("weights ") for each in engine.told)


async def test_adapters_trained_from_full_weights_begin_new_over_them(tmp_path: Path) -> None:
    shared, made = await a_ledger(tmp_path)
    Adapters.began.clear()
    async with Profile.load(a_profile(tmp_path, shared, "Adapters", "merged")).open() as platform:
        assert platform.trainer is not None
        binding = binding_for(words, "policy", platform.tool_bindings)
        await train(
            words, platform.trainer, platform.checkpoints, start=platform.origin, channel="policy",
            directory=tmp_path / "Adapters-merged" / "checkpoints", publish=platform.publish, run=platform.run.id,
            groups=1, binding=binding, made=platform.made,
        )  # fmt: skip
        mine = [each for each in await platform.checkpoints.all() if each.run == platform.run.id]
    assert Adapters.began[0] is None  # (not the merged weights' files, as if they were an adapter)
    assert mine and all(each.kind == "lora" and each.base == made["merged"] for each in mine)


async def test_a_run_from_an_adapter_over_full_weights_serves_those_weights(tmp_path: Path) -> None:
    shared, made = await a_ledger(tmp_path)
    support.STARTED.clear()
    async with Profile.load(a_profile(tmp_path, shared, "Adapters", "stacked")).open() as platform:
        assert platform.origin == made["stacked"]
        fetched = tmp_path / "Adapters-stacked" / "bases" / made["merged"]
        assert support.STARTED[0].told[0].startswith(f"started {fetched}")


async def test_every_weight_is_not_trained_from_an_adapter_before_it_is_merged(tmp_path: Path) -> None:
    shared, _ = await a_ledger(tmp_path)
    async with Profile.load(a_profile(tmp_path, shared, "Full", "plain")).open() as platform:
        assert platform.trainer is not None
        with pytest.raises(ValueError, match="merge it"):
            await train(
                words, platform.trainer, platform.checkpoints, start=platform.origin, channel="policy",
                directory=tmp_path / "Full-plain" / "checkpoints", publish=platform.publish, run=platform.run.id,
                groups=1, made=platform.made,
            )  # fmt: skip


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
