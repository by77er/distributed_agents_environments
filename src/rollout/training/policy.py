# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration and autograd functions partly untyped.)
"""A trainable policy: a 4-bit checkpoint (the same one the engine serves) with LoRA on its linear layers.

`Policy.logprobs` computes the logprobs of sampled tokens. It runs the transformer over the whole sequence but the
output layer (the vocabulary projection, the largest activation by far) only at the positions being scored.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rollout.training.lora import add_lora, lora_parameters, save_adapter
from rollout.training.quantized import replace_compressed_linears

TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
)  # fmt: skip
"""Every attention, linear-attention and MLP projection of Qwen3.5 except the tiny gate projections."""


@dataclass
class Policy:
    model: nn.Module
    checkpoint: str
    rank: int
    alpha: float
    wrapped: list[str]

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        rank: int = 32,
        alpha: float = 64.0,
        device: str = "cuda",
        gradient_checkpointing: bool = True,
    ) -> "Policy":
        from transformers import AutoModelForImageTextToText, CompressedTensorsConfig

        model = cast(
            nn.Module,
            AutoModelForImageTextToText.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                checkpoint,
                dtype=torch.bfloat16,
                device_map={"": device},
                quantization_config=CompressedTensorsConfig(run_compressed=True),
            ),
        )
        replace_compressed_linears(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        wrapped = add_lora(model, TARGETS, rank=rank, alpha=alpha, within="language_model", dtype=torch.float32)
        if gradient_checkpointing:
            enable = cast(Any, model).gradient_checkpointing_enable
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            cast(Any, model).enable_input_require_grads()
        return cls(model, checkpoint, rank, alpha, wrapped)

    def parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.model)

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Logprobs of `tokens[p]` given `tokens[:p]`, for each p in `positions` (all at least 1)."""
        model = cast(Any, self.model)
        device = next(iter(self.model.buffers())).device
        ids = torch.tensor([list(tokens)], device=device)
        hidden = model.model.language_model(input_ids=ids).last_hidden_state[0]
        index = torch.tensor([position - 1 for position in positions], device=device)
        logits = model.lm_head(hidden.index_select(0, index)).float()
        targets = ids[0].index_select(0, torch.tensor(list(positions), device=device))
        return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def save(self, directory: Path) -> Path:
        return save_adapter(self.model, directory, base_model=self.checkpoint, rank=self.rank, alpha=self.alpha)
