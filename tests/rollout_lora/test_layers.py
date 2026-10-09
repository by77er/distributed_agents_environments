# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""LoRA adapters: saved in PEFT's layout and loaded back exactly (each training step starts from the last one's)."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rollout_lora.layers import LoraLinear, adapter_off, add_lora, load_adapter, lora_parameters, save_adapter


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.q_proj(inputs))


def model() -> nn.Sequential:
    torch.manual_seed(0)
    network = nn.Sequential(Block(), Block())
    add_lora(network, ["q_proj"], rank=2, alpha=4.0, dtype=torch.float32)
    return network


def test_an_adapter_switched_off_gives_the_model_it_was_added_to() -> None:
    trained = model()
    torch.manual_seed(0)
    base = nn.Sequential(Block(), Block())  # (the same weights, without the adapter)
    for parameter in lora_parameters(trained):
        parameter.data.normal_()
    inputs = torch.randn(3, 8)
    with adapter_off(trained):
        torch.testing.assert_close(trained(inputs), base(inputs), rtol=0, atol=0)
    assert not torch.allclose(trained(inputs), base(inputs))  # (and on again after)


def test_an_adapter_round_trips_and_changes_only_its_layers(tmp_path: Path) -> None:
    trained = model()
    assert [type(block.q_proj) for block in trained] == [LoraLinear, LoraLinear]
    assert all(isinstance(block.mlp, nn.Linear) for block in trained)  # not a target
    for parameter in lora_parameters(trained):
        parameter.data.normal_()
    save_adapter(trained, tmp_path, base_model="toy", rank=2, alpha=4.0)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert (config["peft_type"], config["r"], config["target_modules"]) == ("LORA", 2, ["q_proj"])

    fresh = model()
    inputs = torch.randn(3, 8)
    assert not torch.allclose(fresh(inputs), trained(inputs))
    assert load_adapter(fresh, tmp_path) == 2
    assert torch.equal(fresh(inputs), trained(inputs))  # saved in full precision: steps resume exactly


def test_an_adapter_for_another_model_is_refused(tmp_path: Path) -> None:
    save_adapter(model(), tmp_path, base_model="toy", rank=2, alpha=4.0)
    other = nn.Sequential(Block())
    add_lora(other, ["q_proj", "mlp"], rank=2, alpha=4.0, dtype=torch.float32)
    with pytest.raises((ValueError, KeyError)):
        load_adapter(other, tmp_path)


def test_the_adapter_trains_in_float32_and_the_frozen_layer_keeps_its_own_dtype() -> None:
    from rollout_lora.quantized import Int4Linear

    packed = torch.zeros(4, 1, dtype=torch.int32)  # (four outputs, eight inputs, one scale each)
    frozen = Int4Linear(packed, torch.ones(4, 1, dtype=torch.bfloat16), in_features=8, out_features=4)
    network = nn.Module()
    network.add_module("q_proj", frozen)
    add_lora(nn.Sequential(network), ["q_proj"], rank=2, alpha=4.0, dtype=torch.float32)
    lora = network.q_proj
    assert isinstance(lora, LoraLinear) and lora.base is frozen
    assert frozen.weight_scale.dtype == torch.bfloat16 and frozen.weight_packed.dtype == torch.int32  # not copied
    assert lora.lora_A.weight.dtype == lora.lora_B.weight.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("rows", "columns", "group"), [(64, 4096, 128), (3, 72, 8), (5, 1000, 40)])
def test_a_4bit_weight_is_dequantized_on_the_gpu_as_on_the_cpu(
    rows: int, columns: int, group: int, dtype: torch.dtype
) -> None:
    from rollout_lora.quantized import Int4Linear, dequantize

    generator = torch.Generator().manual_seed(rows * columns)
    words = (columns + 7) // 8  # (the last word's high nibbles past `columns` are padding)
    packed = torch.randint(-(2**31), 2**31 - 1, (rows, words), dtype=torch.int32, generator=generator)
    scale = (torch.rand(rows, columns // group, generator=generator) * 0.02 + 1e-4).to(torch.bfloat16)
    on_cpu = dequantize(packed, scale, columns, dtype)
    on_gpu = dequantize(packed.cuda(), scale.cuda(), columns, dtype)
    assert on_gpu.dtype == dtype and torch.equal(on_gpu.cpu(), on_cpu)  # (bit for bit)

    layer = Int4Linear(packed, scale, in_features=columns, out_features=rows)
    inputs = torch.randn(7, columns, generator=generator).to(dtype).requires_grad_(True)
    expected = layer(inputs)
    expected.sum().backward()
    assert inputs.grad is not None
    wanted = inputs.grad.clone()
    layer.cuda()
    inputs = inputs.detach().cuda().requires_grad_(True)
    found = layer(inputs)
    found.sum().backward()
    assert inputs.grad is not None
    torch.testing.assert_close(found.cpu(), expected.detach(), rtol=2e-2, atol=2e-2)  # (the matmuls' own sums)
    torch.testing.assert_close(inputs.grad.cpu(), wanted, rtol=2e-2, atol=2e-2)
