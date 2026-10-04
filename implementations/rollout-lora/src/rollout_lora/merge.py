# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# (safetensors' loaders are partly untyped.)
"""An adapter folded into the weights it was trained over: a model of its own, with no adapter.

Each adapted layer's weight becomes W + (alpha / rank) * B @ A. The base's files are read one at a time (a 9B model's
are a few GiB each), each delta is computed in float32 and added in the base's own dtype, and every other weight is
copied as it is; the base's configuration and tokenizer come along. An adapter trained over a 4-bit checkpoint is
merged into the same model's unquantized weights (its layers have the same names), which is how such an adapter
becomes a model of its own.
"""

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rollout_lora.models import COPIED, local

_KEY = re.compile(r"^base_model\.model\.(?P<name>.+)\.lora_(?P<part>[AB])\.weight$")


def merge(base: str, adapter: Path, into: Path, *, device: str | None = None) -> dict[str, int]:
    """Fold the adapter in `adapter` (PEFT's layout) into `base` (a model's name or directory), writing the merged
    model to `into`; returns how many layers were merged and how many weights copied. `device` computes the deltas
    (by default the GPU if there is one)."""
    said = json.loads((adapter / "adapter_config.json").read_text())
    scale = float(said["lora_alpha"]) / float(said["r"])
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in load_file(str(adapter / "adapter_model.safetensors")).items():
        found = _KEY.match(key)
        if found is None:
            raise ValueError(f"{key} is not a LoRA weight")
        pairs.setdefault(found["name"], {})[found["part"]] = tensor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = local(base)
    into.mkdir(parents=True, exist_ok=True)
    merged, copied = set[str](), 0
    for file in sorted(source.glob("*.safetensors")):
        tensors = load_file(str(file))
        for key in list(tensors):
            name = key.removesuffix(".weight")
            if not key.endswith(".weight") or name not in pairs:
                copied += 1
                continue
            lora = pairs[name]
            weight = tensors[key]
            delta = scale * (lora["B"].to(device, torch.float32) @ lora["A"].to(device, torch.float32))
            if delta.shape != weight.shape:
                raise ValueError(
                    f"{name}: the adapter's update is {tuple(delta.shape)}, the weight {tuple(weight.shape)}"
                )
            tensors[key] = (weight.to(device, torch.float32) + delta).to(weight.dtype).cpu()
            merged.add(name)
        save_file(tensors, str(into / file.name), metadata={"format": "pt"})
        del tensors
    missing = sorted(set(pairs) - merged)
    if missing:
        raise ValueError(f"{len(missing)} adapted layers are not in {base}: {', '.join(missing[:3])}")
    for name in (*COPIED, "model.safetensors.index.json"):
        if (source / name).exists():
            shutil.copy2(source / name, into / name)
    return {"layers": len(merged), "copied": copied}
