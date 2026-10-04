# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tinker's adapter in PEFT's layout: the sampler checkpoint's archive downloaded, renamed into PEFT's layout,
Qwen3.5's split q, k and v joined, and the adapter, kept as a checkpoint's files, folded into the model exactly by
`rollout merge`.

`qwen35_archive.json` is a recording: the names and shapes of a Tinker archive for `Qwen/Qwen3.5-4B` (the live test's
496 tensors, at rank 2), the model's configuration and weights' names, and what `tinker_cookbook` 0.5.7's
`build_lora_adapter`, then `fused`, made of it."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from rollout.harness.blobs import FileBlobStore
from rollout_tinker import TinkerTrainer
from rollout_tinker.testing import FakeService, reset
from rollout_tinker.weights import POINTER, peft_adapter, pointer, ranks, renames
from rollout_train.bridges import BRIDGED, bridged, by_name
from rollout_train.checkpoints import Checkpoints
from rollout_train.ledger import FileLedger
from rollout_train.merging import SCOPE, merge
from rollout_train.record import scope
from rollout_train.trainer import WEIGHTS
from tests.rollout_tinker.support import ALPHA, LAYERS, RANK, SPLIT, base_model, segments, tinkers


@pytest.mark.parametrize("shared", [False, True], ids=["an A each", "one A"])
async def test_a_tinker_checkpoint_bridged_to_peft_holds_the_adapter_and_merges_into_the_model_exactly(
    shared: bool, tmp_path: Path
) -> None:
    base = base_model(tmp_path / "base")
    archive = tinkers(shared)
    service = FakeService(vocabulary=24, adapter=archive, rank=RANK, alpha=ALPHA)
    trainer = TinkerTrainer(str(base), service=service, rank=RANK, learning_rate=0.05)
    into = tmp_path / "made"
    await trainer.step(segments(service, 4), seed=1, parent=None, into=into)
    assert [path.name for path in (into / WEIGHTS).iterdir()] == [POINTER]  # (the trainer keeps pointers only)

    # Tinker's bridge: the sampler checkpoint's archive in PEFT's layout, kept as the checkpoint's bridged files.
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    fence = await checkpoints.ledger.take(scope("run"))
    lora = await checkpoints.add(fence, "kkkk" * 4, weights=into / WEIGHTS, run="run", base=str(base))
    settings = {"peft-from-tinker": {"service": "rollout_tinker.testing:fake_service"}}
    chain = (by_name("peft-from-tinker"),)
    reset(service)  # (the service the bridge's settings name)
    try:
        manifest = await bridged(checkpoints, fence, lora.id, chain, tmp_path / "scratch", settings=settings)
    finally:
        reset()  # (afresh, for whatever names the shared fake next)
    peft = await checkpoints.files(manifest, tmp_path / "bridged")
    assert sorted(path.name for path in peft.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    sampler = pointer(into / WEIGHTS, "sampler")
    assert sampler is not None and f"archive {sampler}" in service.calls
    noted: Any = (await checkpoints.ledger.read(BRIDGED))[f"{lora.id}@peft-from-tinker"]
    assert noted["said"] == {"sampler": sampler, "model": str(base), "largest_rank": RANK if shared else 3 * RANK}
    adapter = load_file(str(peft / "adapter_model.safetensors"))
    assert not any("in_proj_q." in key or "in_proj_k." in key or "in_proj_v." in key for key in adapter)
    assert ranks(peft) == (RANK if shared else 3 * RANK)  # (what an engine's max_lora_rank must reach)
    configuration = json.loads((peft / "adapter_config.json").read_text())
    assert "in_proj_qkv" in configuration["target_modules"] and configuration["r"] == RANK
    assert not list((tmp_path / "scratch").iterdir())  # (the download is not kept)

    # `rollout merge`: the bridged adapter folded into the model, each layer W + (alpha / rank) · B · A.
    full = await merge(checkpoints, await checkpoints.ledger.take(SCOPE), lora.id, scratch=tmp_path / "scratch")
    assert full.kind == "full" and full.parents == (lora.id,)
    merged_files = await checkpoints.files(full.weights, tmp_path / "merged") if full.weights else tmp_path
    merged = load_file(str(merged_files / "model.safetensors"))
    original = load_file(str(base / "model.safetensors"))
    tinker = archive(sampler)
    scale = ALPHA / RANK

    def delta(name: str) -> torch.Tensor:
        key = f"base_model.model.model.{name}"
        return scale * tinker[f"{key}.lora_B.weight"] @ tinker[f"{key}.lora_A.weight"]

    for name in LAYERS:
        expected = original[f"model.language_model.{name}.weight"]
        if "in_proj_qkv" in name:  # Q, K and V's updates, one above the other
            expected = expected + torch.cat([delta(f"layers.0.linear_attn.in_proj_{part}") for part in SPLIT])
        else:
            expected = expected + delta(name)
        torch.testing.assert_close(merged[f"model.language_model.{name}.weight"], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(merged["lm_head.weight"], original["lm_head.weight"])  # (not adapted: copied)


def test_a_real_archives_names_become_what_the_cookbook_made_of_them(tmp_path: Path) -> None:
    recorded = json.loads((Path(__file__).parent / "qwen35_archive.json").read_text())
    model = tmp_path / "model"  # (its configuration and its weights' names: the remap reads nothing else)
    model.mkdir()
    (model / "config.json").write_text(json.dumps(recorded["configuration"]))
    index = {"weight_map": dict.fromkeys(recorded["model_weights"], "model.safetensors")}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    generator = torch.Generator().manual_seed(0)
    tensors = {key: torch.randn(shape, generator=generator) for key, shape in recorded["archive"].items()}
    archive = tmp_path / "archive"
    archive.mkdir()
    save_file(tensors, str(archive / "adapter_model.safetensors"))
    (archive / "adapter_config.json").write_text(json.dumps(recorded["archive_configuration"]))

    peft_adapter(archive, tmp_path / "peft", str(model))
    made = load_file(str(tmp_path / "peft" / "adapter_model.safetensors"))
    assert {key: list(tensor.shape) for key, tensor in made.items()} == recorded["peft"]
    said = json.loads((tmp_path / "peft" / "adapter_config.json").read_text())
    assert said == {**recorded["peft_configuration"], "base_model_name_or_path": str(model)}

    def tinkers(key: str) -> str:  # (PEFT's name back to Tinker's)
        return key.replace("base_model.model.model.language_model.", "base_model.model.model.")

    joined = 0
    for key, tensor in made.items():
        if ".in_proj_qkv.lora_A." in key:  # B·A is q's, k's and v's updates, one above the other
            b = made[key.replace("lora_A", "lora_B")]
            parts = [tinkers(key).replace("in_proj_qkv", f"in_proj_{part}") for part in "qkv"]
            expected = torch.cat([tensors[each.replace("lora_A", "lora_B")] @ tensors[each] for each in parts])
            torch.testing.assert_close(b @ tensor, expected, rtol=1e-5, atol=1e-4)
            joined += 1
        elif ".in_proj_qkv." not in key:
            assert torch.equal(tensor, tensors[tinkers(key)])
    assert joined == 24  # (Qwen3.5-4B's linear-attention layers)


@pytest.mark.parametrize(
    ("keys", "unembedding", "layer"),
    [
        ({"model.language_model.embed_tokens.weight"}, "model.language_model.embed_tokens", "model.language_model."),
        ({"model.language_model.embed_tokens.weight", "lm_head.weight"}, "lm_head", "model.language_model."),
        ({"model.embed_tokens.weight", "lm_head.weight"}, "lm_head", "model."),
        ({"model.embed_tokens.weight"}, "model.embed_tokens", "model."),
    ],
    ids=["Qwen3.5, tied", "Qwen3.5, apart", "Qwen3, apart", "Qwen3, tied"],
)
def test_the_unembedding_is_the_head_or_the_embedding_it_is_tied_to(
    keys: set[str], unembedding: str, layer: str
) -> None:
    def renamed(name: str) -> str:
        for old, new in renames({"model_type": "qwen3"}, keys):
            name = name.replace(old, new)
        return name

    assert renamed("base_model.model.model.unembed_tokens.weight") == f"{unembedding}.weight"
    assert renamed("base_model.model.model.layers.2.mlp.up_proj.weight") == f"{layer}layers.2.mlp.up_proj.weight"


def test_adapters_named_otherwise_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="named otherwise"):
        renames({"model_type": "gpt_oss"}, {"model.embed_tokens.weight"})
    model = base_model(tmp_path / "base")
    archive = tmp_path / "archive"
    archive.mkdir()
    experts = "base_model.model.model.layers.0.mlp.experts.w1"
    save_file({f"{experts}.lora_A.weight": torch.zeros(2, 2, 8), f"{experts}.lora_B.weight": torch.zeros(2, 4, 2)},
              str(archive / "adapter_model.safetensors"))  # fmt: skip
    (archive / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 32}))
    with pytest.raises(ValueError, match="experts"):
        peft_adapter(archive, tmp_path / "peft", str(model))
