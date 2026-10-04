# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration and autograd functions partly untyped.)
"""A trainable policy: the model the engine serves (a 4-bit image-text checkpoint, or a text model in bfloat16) with
LoRA on its linear layers.

`Policy.logprobs` computes the logprobs of sampled tokens. It runs the transformer over the whole sequence but the
output layer (the vocabulary projection, the largest activation by far) only at the positions being scored.

Only what training text needs is kept on the GPU: a vision tower is dropped, and the token embedding table (as
large as the output layer, and used only to look up a sequence's rows) is read from the checkpoint file as needed.
On a 16 GB card that is the difference between turns of 5,000 tokens and turns of about twice that.
"""

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rollout_lora.activations import HostStore, checkpoint_layers
from rollout_lora.layers import add_lora, lora_parameters, save_adapter
from rollout_lora.models import config, local, multimodal, quantized
from rollout_lora.quantized import replace_compressed_linears

TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
)  # fmt: skip
"""Every attention, linear-attention and MLP projection of Qwen3.5 except the tiny gate projections."""


LOGIT_ROWS = 128
"""Positions sent through the output layer at a time (each row of logits is a vocabulary wide)."""


class FileEmbedding:
    """A token embedding table left in its checkpoint file (memory-mapped): only the rows a sequence uses are read."""

    def __init__(self, file: Path, name: str) -> None:
        with file.open("rb") as opened:
            (length,) = struct.unpack("<Q", opened.read(8))
            entry = json.loads(opened.read(length))[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name} is stored as {entry['dtype']}, not bfloat16")
        rows, width = entry["shape"]
        self.table = numpy.memmap(file, dtype=numpy.int16, mode="r", offset=8 + length + entry["data_offsets"][0])
        self.table = self.table[: rows * width].reshape(rows, width)

    def __call__(self, ids: Sequence[int], device: torch.device) -> torch.Tensor:
        """The rows for `ids`, on `device`, as bfloat16."""
        rows = numpy.ascontiguousarray(self.table[numpy.asarray(ids)])
        return torch.from_numpy(rows).view(torch.bfloat16).to(device)

    @classmethod
    def find(cls, checkpoint: str, name: str) -> "FileEmbedding | None":
        """The table named `name` in a checkpoint (a directory, or a model already in the Hugging Face cache)."""
        for file in sorted(local(checkpoint).glob("*.safetensors")):
            with file.open("rb") as opened:
                (length,) = struct.unpack("<Q", opened.read(8))
                if name in json.loads(opened.read(length)):
                    return cls(file, name)
        return None


EMBEDDING = "model.language_model.embed_tokens.weight"
"""The token embeddings of an image-text model's language part (a text model's are `model.embed_tokens.weight`)."""


def body(model: nn.Module) -> Any:
    """The decoder whose last hidden states the output layer reads: an image-text model's language part, or a text
    model's own."""
    inner = cast(Any, model).model
    return getattr(inner, "language_model", inner)


def scored(model: nn.Module, hidden: torch.Tensor, ids: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """Logprobs of `ids[p]` from the hidden state before it, for each p in `positions`, the output layer run a chunk
    of rows at a time (each recomputed for the backward pass, so the peak is one chunk's logits)."""
    device = hidden.device
    index = torch.tensor([position - 1 for position in positions], device=device)
    rows = hidden.index_select(0, index)
    targets = ids[0].index_select(0, torch.tensor(list(positions), device=device))
    head = cast(Any, model).lm_head

    def chunk(rows: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = head(rows).float()
        return -torch.nn.functional.cross_entropy(logits, targets, reduction="none")

    parts = [
        cast(
            torch.Tensor,
            checkpoint(
                chunk, rows[start : start + LOGIT_ROWS], targets[start : start + LOGIT_ROWS], use_reentrant=False
            ),
        )
        for start in range(0, len(positions), LOGIT_ROWS)
    ]
    return torch.cat(parts)


@dataclass
class Policy:
    model: nn.Module
    checkpoint: str
    rank: int
    alpha: float
    embedding: FileEmbedding | None = None
    """The token embeddings, if they are read from the checkpoint file rather than held on the GPU."""

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        rank: int,
        alpha: float,
        device: str = "cuda",
        gradient_checkpointing: bool = True,
        layer_inputs_on_host: bool = False,
        mlp_rows: int | None = None,
    ) -> "Policy":
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, CompressedTensorsConfig

        where = str(local(checkpoint))
        if multimodal(checkpoint):
            options: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": device}}
            if quantized(checkpoint):
                options["quantization_config"] = CompressedTensorsConfig(run_compressed=True)
            model = cast(nn.Module, AutoModelForImageTextToText.from_pretrained(where, **options))  # pyright: ignore[reportUnknownMemberType]
            replace_compressed_linears(model)
        else:
            model = cast(
                nn.Module, AutoModelForCausalLM.from_pretrained(where, dtype=torch.bfloat16, device_map={"": device})
            )  # pyright: ignore[reportUnknownMemberType]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        inner = cast(Any, model).model
        if getattr(inner, "visual", None) is not None:  # agents read text: the vision tower is never run
            inner.visual = None
        tied = bool(config(checkpoint).get("tie_word_embeddings")) or bool(
            config(checkpoint).get("text_config", {}).get("tie_word_embeddings")
        )
        embedding = None if tied else FileEmbedding.find(checkpoint, EMBEDDING)  # (tied: the output layer holds it)
        if embedding is not None:
            body(model).embed_tokens = None
        torch.cuda.empty_cache()
        within = "language_model" if multimodal(checkpoint) else "layers"
        add_lora(model, TARGETS, rank=rank, alpha=alpha, within=within, dtype=torch.float32)
        if gradient_checkpointing:
            enable = cast(Any, model).gradient_checkpointing_enable
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if embedding is None:  # (embeddings read from the file are marked as needing gradients where used)
                cast(Any, model).enable_input_require_grads()
            if layer_inputs_on_host or mlp_rows is not None:
                store = HostStore(pin=device != "cpu") if layer_inputs_on_host else None
                checkpoint_layers(body(model), store=store, rows=mlp_rows)
        return cls(model, checkpoint, rank, alpha, embedding)

    def parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.model)

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Logprobs of `tokens[p]` given `tokens[:p]`, for each p in `positions` (all at least 1)."""
        device = next(iter(self.model.buffers())).device
        ids = torch.tensor([list(tokens)], device=device)
        if self.embedding is None:
            hidden = body(self.model)(input_ids=ids).last_hidden_state[0]
        else:
            embedded = self.embedding(tokens, device).unsqueeze(0).requires_grad_(True)  # (for checkpointing)
            hidden = body(self.model)(inputs_embeds=embedded).last_hidden_state[0]
        return scored(self.model, hidden, ids, positions)

    def save(self, directory: Path) -> Path:
        return save_adapter(self.model, directory, base_model=self.checkpoint, rank=self.rank, alpha=self.alpha)
