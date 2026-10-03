# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration and autograd functions partly untyped.)
"""LoRA adapters over frozen layers, saved in PEFT's format so engines (vLLM) load them as they are."""

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]
from torch import nn


class LoraLinear(nn.Module):
    """`base(x) + scaling · B(A(x))`; only A and B train. B starts at zero, so a new adapter changes nothing."""

    def __init__(self, base: nn.Module, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (
            cast(torch.Tensor, self.base(inputs))
            + self.lora_B(self.lora_A(inputs.to(self.lora_A.weight.dtype))).to(inputs.dtype) * self.scaling
        )


def add_lora(
    model: nn.Module, targets: Sequence[str], *, rank: int, alpha: float, within: str = "", dtype: torch.dtype
) -> list[str]:
    """Wrap every layer whose name ends with one of `targets` (and contains `within`); returns their names."""
    wrapped: list[str] = []
    pattern = re.compile(rf"\.({'|'.join(map(re.escape, targets))})$")
    modules: list[tuple[str, nn.Module]] = list(model.named_modules())
    for name, module in modules:
        if within not in name or not pattern.search(name):
            continue
        layer: Any = module
        in_features, out_features = int(layer.in_features), int(layer.out_features)
        device = next(iter(module.buffers()), next(iter(module.parameters()), None))
        lora = LoraLinear(module, in_features, out_features, rank, alpha)
        for part in (lora.lora_A, lora.lora_B):  # only these: the frozen layer keeps its own dtype and device
            part.to(device=device.device if device is not None else "cpu", dtype=dtype)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, lora)
        wrapped.append(name)
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for name, parameter in model.named_parameters() if ".lora_A." in name or ".lora_B." in name]


def load_adapter(model: nn.Module, directory: Path) -> int:
    """Load an adapter saved by `save_adapter` into the model's LoRA layers; returns how many layers it filled."""
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    filled = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            for part, layer in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
                saved = tensors[f"base_model.model.{name}.{part}.weight"]
                layer.weight.data.copy_(saved.to(layer.weight.device, layer.weight.dtype))
            filled += 1
    if filled == 0 or 2 * filled != len(tensors):
        raise ValueError(f"the adapter in {directory} does not match the model's LoRA layers")
    return filled


def save_adapter(model: nn.Module, directory: Path, *, base_model: str, rank: int, alpha: float) -> Path:
    """Write the adapter in PEFT's layout: `adapter_config.json` and `adapter_model.safetensors`."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    targets: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            # Full precision: each training step resumes from this file, and updates are far smaller than
            # bfloat16 resolves. Engines cast to their own dtype when they load it.
            tensors[f"base_model.model.{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu().contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu().contiguous()
            targets.add(name.rsplit(".", 1)[-1])
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(targets),
        "fan_in_fan_out": False,
        "use_rslora": False,
        "modules_to_save": None,
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return directory
