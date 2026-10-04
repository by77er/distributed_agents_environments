# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration partly untyped.)
"""A policy whose every weight is trained: a text model, held in float32 and computed in bfloat16.

The weights stay in float32 between steps (each step starts from the files the one before saved): an update is far
smaller than a bfloat16 weight resolves, and saved in bfloat16 most of it would be lost. Engines cast the files to
their own dtype when they load them. A checkpoint keeps the model's configuration and tokenizer beside its weights,
so that it can also be served, or trained from, as a model of its own.

Its memory: the weights, their gradients and Adam's two moments, four copies in float32 (16 bytes a weight), and the
activations of one segment with gradient checkpointing. Qwen3-0.6B takes about 10 GiB.
"""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rollout_lora.models import COPIED, local, multimodal
from rollout_lora.policy import body, scored


@dataclass
class FullPolicy:
    model: nn.Module
    checkpoint: str
    """Where it was loaded from: a model's name or directory, or a full checkpoint's files."""

    @classmethod
    def load(cls, checkpoint: str, *, gradient_checkpointing: bool = True) -> "FullPolicy":
        from transformers import AutoModelForCausalLM

        if multimodal(checkpoint):
            raise ValueError(f"{checkpoint} is an image-text model: full weights are trained on text models only")
        model = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(str(local(checkpoint)), dtype=torch.float32, device_map={"": "cuda"}),
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if gradient_checkpointing:
            cast(Any, model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(model, checkpoint)

    def parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Logprobs of `tokens[p]` given `tokens[:p]`, for each p in `positions` (all at least 1)."""
        device = next(iter(self.model.parameters())).device
        ids = torch.tensor([list(tokens)], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = body(self.model)(input_ids=ids).last_hidden_state[0]
            return scored(self.model, hidden, ids, positions)

    def save(self, directory: Path) -> Path:
        """The weights (float32, in safetensors) and the model's configuration and tokenizer."""
        directory.mkdir(parents=True, exist_ok=True)
        cast(Any, self.model).save_pretrained(directory, safe_serialization=True, max_shard_size="4GB")
        source = local(self.checkpoint)
        for name in COPIED:
            if (source / name).exists() and not (directory / name).exists():
                shutil.copy2(source / name, directory / name)
        said = json.loads((directory / "config.json").read_text())  # served in bfloat16: the files stay float32
        said |= {key: "bfloat16" for key in ("torch_dtype", "dtype") if key in said}
        (directory / "config.json").write_text(json.dumps(said, indent=2))
        return directory
