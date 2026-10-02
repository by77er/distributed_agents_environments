# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""LoRA adapters: saved in PEFT's layout and loaded back exactly (each training step starts from the last one's)."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rollout.lora.layers import LoraLinear, add_lora, load_adapter, lora_parameters, save_adapter


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
