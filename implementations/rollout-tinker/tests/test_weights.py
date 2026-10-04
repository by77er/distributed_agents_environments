# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""`weights = "peft"`: the sampler checkpoint's archive downloaded, turned into PEFT's layout by the cookbook (on the
platform's transformers), Qwen3.5's split q, k and v joined, and the adapter kept as the checkpoint's files, which
`rollout merge` folds into the model exactly."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from rollout.harness.blobs import FileBlobStore
from rollout_tinker import TinkerTrainer
from rollout_tinker.testing import FakeService
from rollout_tinker.weights import POINTER, pointer, ranks
from rollout_train.checkpoints import Checkpoints
from rollout_train.ledger import FileLedger
from rollout_train.merging import SCOPE, merge
from rollout_train.record import scope
from rollout_train.trainer import WEIGHTS
from tests.support import segments

RANK, ALPHA, HIDDEN = 4, 32.0, 8
LAYERS = {  # a tiny model laid out as Qwen3.5 is: a linear-attention layer, then a full-attention one
    "layers.0.linear_attn.in_proj_qkv": (16, HIDDEN),  # Q 4, K 4, V 8
    "layers.0.linear_attn.in_proj_z": (8, HIDDEN),
    "layers.0.linear_attn.out_proj": (HIDDEN, 8),
    "layers.0.mlp.gate_proj": (12, HIDDEN),
    "layers.0.mlp.down_proj": (HIDDEN, 12),
    "layers.1.self_attn.q_proj": (16, HIDDEN),
    "layers.1.self_attn.o_proj": (HIDDEN, 16),
}
SPLIT = {"q": 4, "k": 4, "v": 8}


def base_model(directory: Path) -> Path:
    """A model's directory as the cookbook and the merge read it: its configuration and its weights."""
    directory.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        f"model.language_model.{name}.weight": torch.randn(shape, generator=generator) for name, shape in LAYERS.items()
    }
    tensors["model.language_model.embed_tokens.weight"] = torch.randn(24, HIDDEN, generator=generator)
    tensors["lm_head.weight"] = torch.randn(24, HIDDEN, generator=generator)
    save_file(tensors, str(directory / "model.safetensors"))
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
    }
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def tinkers(shared: bool):
    """What a Tinker archive holds for the tiny model: Tinker's names, q, k and v of a linear-attention layer apart
    (sharing one A, with `shared`)."""

    def made(path: str) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(len(path))
        tensors: dict[str, torch.Tensor] = {}

        def pair(name: str, out: int, into: int, a: torch.Tensor | None = None) -> None:
            key = f"base_model.model.model.{name}"
            tensors[f"{key}.lora_A.weight"] = a if a is not None else torch.randn(RANK, into, generator=generator)
            tensors[f"{key}.lora_B.weight"] = torch.randn(out, RANK, generator=generator)

        one = torch.randn(RANK, HIDDEN, generator=generator)
        for part, out in SPLIT.items():
            pair(f"layers.0.linear_attn.in_proj_{part}", out, HIDDEN, one.clone() if shared else None)
        for name, (out, into) in LAYERS.items():
            if "in_proj_qkv" not in name:
                pair(name, out, into)
        return tensors

    return made


@pytest.mark.parametrize("shared", [False, True], ids=["an A each", "one A"])
async def test_a_peft_checkpoint_holds_the_adapter_and_merges_into_the_model_exactly(
    shared: bool, tmp_path: Path
) -> None:
    base = base_model(tmp_path / "base")
    archive = tinkers(shared)
    service = FakeService(vocabulary=24, adapter=archive, rank=RANK, alpha=ALPHA)
    trainer = TinkerTrainer(str(base), service=service, rank=RANK, weights="peft", learning_rate=0.05)
    into = tmp_path / "made"
    await trainer.step(segments(service, 4), seed=1, parent=None, into=into)

    weights = into / WEIGHTS
    assert sorted(path.name for path in weights.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        POINTER,
    ]
    sampler = pointer(weights, "sampler")
    assert sampler is not None and f"archive {sampler}" in service.calls
    adapter = load_file(str(weights / "adapter_model.safetensors"))
    assert not any("in_proj_q." in key or "in_proj_k." in key or "in_proj_v." in key for key in adapter)
    assert ranks(weights) == (RANK if shared else 3 * RANK)  # (what an engine's max_lora_rank must reach)
    said = json.loads((weights / "adapter_config.json").read_text())
    assert "in_proj_qkv" in said["target_modules"] and said["r"] == RANK
    assert not (into / "archive").exists()  # (the download is not kept)

    # `rollout merge`: the adapter folded into the model, each layer W + (alpha / rank) · B · A.
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    fence = await checkpoints.ledger.take(scope("run"))
    lora = await checkpoints.add(fence, "kkkk" * 4, weights=weights, run="run", base=str(base))
    assert POINTER in lora.weights.files if lora.weights else False
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
