# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration partly untyped.)
"""A policy whose every weight is trained: a text model, held in float32 and computed in bfloat16.

The weights stay in float32 between steps (each step starts from the files the one before saved): an update is far
smaller than a bfloat16 weight resolves, and saved in bfloat16 most of it would be lost. Engines cast the files to
their own dtype when they load them. A checkpoint keeps the model's configuration and tokenizer beside its weights,
so that it can also be served, or trained from, as a model of its own.

Its memory: the weights, their gradients and Adam's two moments, four copies in float32 (16 bytes a weight), and the
activations of one segment with gradient checkpointing. Qwen3-0.6B takes about 10 GiB. An objective that reads the
reference needs a frozen copy of the model trained over beside it, in bfloat16 (2 bytes a weight more), which it holds
only when asked (`LoraSettings.frozen_reference`).
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
from rollout_lora.policy import body, scored_with_entropy


@dataclass
class FullPolicy:
    model: nn.Module
    checkpoint: str
    """Where it was loaded from: a model's name or directory, or a full checkpoint's files."""
    frozen: nn.Module | None = None
    """The reference, if it holds one: a frozen copy of the model trained over, in bfloat16."""

    @classmethod
    def load(
        cls, checkpoint: str, *, gradient_checkpointing: bool = True, reference: str | None = None
    ) -> "FullPolicy":
        """The policy from `checkpoint`, and with `reference` (a model's name or directory) a frozen copy of it."""
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
        frozen: nn.Module | None = None
        if reference is not None:
            frozen = cast(
                nn.Module,
                AutoModelForCausalLM.from_pretrained(
                    str(local(reference)), dtype=torch.bfloat16, device_map={"": "cuda"}
                ),
            )
            frozen.eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        return cls(model, checkpoint, frozen)

    def parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Logprobs of `tokens[p]` given `tokens[:p]`, for each p in `positions` (all at least 1)."""
        return self.logprobs_and_entropy(tokens, positions, entropy=False)[0]

    def logprobs_and_entropy(
        self, tokens: Sequence[int], positions: Sequence[int], *, entropy: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`logprobs`, and the entropy of the policy's distribution at each of `positions`."""
        device = next(iter(self.model.parameters())).device
        ids = torch.tensor([list(tokens)], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = body(self.model)(input_ids=ids).last_hidden_state[0]
            return scored_with_entropy(self.model, hidden, ids, positions, entropy=entropy)

    def reference(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """`logprobs` under the frozen copy of the model trained over (no gradient)."""
        if self.frozen is None:
            raise ValueError("this full-weight policy holds no reference (LoraSettings.frozen_reference)")
        device = next(iter(self.frozen.parameters())).device
        ids = torch.tensor([list(tokens)], device=device)
        with torch.no_grad():
            hidden = body(self.frozen)(input_ids=ids).last_hidden_state[0]
            return scored_with_entropy(self.frozen, hidden, ids, positions, entropy=False)[0]

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
