# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Folding an adapter into the weights it was trained over: W + (alpha / rank) * B @ A, everything else unchanged."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from rollout_lora.bridges import merge_quantize
from rollout_lora.merge import merge
from rollout_train.bridges import Context


def a_model(directory: Path) -> dict[str, torch.Tensor]:
    """A two-file model with one projection to adapt, one not, and its configuration and tokenizer."""
    directory.mkdir(parents=True)
    first = {"model.layers.0.self_attn.q_proj.weight": torch.randn(6, 4, dtype=torch.bfloat16)}
    second = {
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    save_file(first, str(directory / "model-00001-of-00002.safetensors"))
    save_file(second, str(directory / "model-00002-of-00002.safetensors"))
    (directory / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (directory / "tokenizer.json").write_text("{}")
    return first | second


def an_adapter(directory: Path, rank: int, alpha: float, layers: dict[str, tuple[int, int]]) -> dict[str, torch.Tensor]:
    directory.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {}
    for name, (out, inner) in layers.items():
        tensors[f"base_model.model.{name}.lora_A.weight"] = torch.randn(rank, inner)
        tensors[f"base_model.model.{name}.lora_B.weight"] = torch.randn(out, rank)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha}))
    return tensors


def test_an_adapter_is_folded_into_its_layers_and_the_rest_is_copied(tmp_path: Path) -> None:
    base = a_model(tmp_path / "base")
    q = "model.layers.0.self_attn.q_proj"
    adapter = an_adapter(tmp_path / "adapter", rank=2, alpha=4.0, layers={q: (6, 4)})
    said = merge(str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "merged", device="cpu")
    assert said == {"layers": 1, "copied": 2}
    merged = load_file(str(tmp_path / "merged" / "model-00001-of-00002.safetensors"))
    delta = 2.0 * adapter[f"base_model.model.{q}.lora_B.weight"] @ adapter[f"base_model.model.{q}.lora_A.weight"]
    expected = (base[f"{q}.weight"].float() + delta).to(torch.bfloat16)
    assert merged[f"{q}.weight"].dtype == torch.bfloat16 and torch.equal(merged[f"{q}.weight"], expected)
    rest = load_file(str(tmp_path / "merged" / "model-00002-of-00002.safetensors"))
    assert all(torch.equal(rest[key], base[key]) for key in rest)  # (untouched)
    assert (tmp_path / "merged" / "config.json").exists() and (tmp_path / "merged" / "tokenizer.json").exists()


def test_an_adapter_for_layers_the_base_lacks_or_of_another_shape_is_refused(tmp_path: Path) -> None:
    a_model(tmp_path / "base")
    an_adapter(tmp_path / "elsewhere", rank=2, alpha=4.0, layers={"model.layers.9.self_attn.q_proj": (6, 4)})
    with pytest.raises(ValueError, match="1 adapted layers are not in"):
        merge(str(tmp_path / "base"), tmp_path / "elsewhere", tmp_path / "merged", device="cpu")
    an_adapter(tmp_path / "wrong", rank=2, alpha=4.0, layers={"model.layers.0.self_attn.q_proj": (5, 4)})
    with pytest.raises(ValueError, match="the adapter's update is"):
        merge(str(tmp_path / "base"), tmp_path / "wrong", tmp_path / "merged2", device="cpu")


def test_the_merge_bridge_merges_for_a_provider_that_quantizes_as_it_loads(tmp_path: Path) -> None:
    a_model(tmp_path / "base")
    q = "model.layers.0.self_attn.q_proj"
    an_adapter(tmp_path / "adapter", rank=2, alpha=4.0, layers={q: (6, 4)})
    base = str(tmp_path / "base")
    said = merge_quantize(tmp_path / "adapter", tmp_path / "merged", Context("kpqx", model=base, target=base))
    assert said == {"model": base, "layers": 1, "copied": 2}
    assert (tmp_path / "merged" / "model-00001-of-00002.safetensors").exists()
    quantized = Context("kpqx", model=base, target="cyankiwi/Qwen3.5-9B-AWQ-4bit")
    with pytest.raises(ValueError, match="quantized beforehand"):
        merge_quantize(tmp_path / "adapter", tmp_path / "refused", quantized)
