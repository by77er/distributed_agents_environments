# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""What the tests share: segments on the fake's bigram, and the same bigram as a policy `PolicyStep` trains; a tiny
model laid out as Qwen3.5 is, with the archive Tinker would make of an adapter of it."""

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy
import torch
from safetensors.torch import save_file

from rollout_tinker.testing import FakeService
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Weighted


class Bigram:
    """The fake's model, as `rollout_lora.step.PolicyStep` trains a policy: the learnable table is its one
    parameter."""

    def __init__(self, service: FakeService, table: torch.Tensor | None = None) -> None:
        self.service = service
        self.model = torch.nn.Module()
        self.table = torch.nn.Parameter(table.clone() if table is not None else torch.zeros_like(service.base))
        self.model.register_parameter("table", self.table)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.table]

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        return self.service.logprobs(self.table, [tokens[p - 1] for p in positions], [tokens[p] for p in positions])


def segments(service: FakeService, count: int, *, seed: int = 0, noise: float = 0.4) -> list[Weighted]:
    """Segments of a few turns each: a prompt, then sampled spans with a token the recorder forced between them,
    behaviour logprobs near the base model's (an engine that computes a little differently), advantages of both
    signs."""
    rng = random.Random(seed)
    vocabulary = service.base.shape[0]
    made: list[Weighted] = []
    for index in range(count):
        tokens = [rng.randrange(vocabulary) for _ in range(rng.randrange(4, 9))]
        spans: list[Span] = []
        for turn in range(rng.randrange(1, 4)):
            start = len(tokens)
            tokens += [rng.randrange(vocabulary) for _ in range(rng.randrange(2, 12))]
            spans.append(Span(start, len(tokens), version=turn))
            tokens += [rng.randrange(vocabulary) for _ in range(rng.randrange(1, 5))]  # forced, then a tool's result
        positions = [p for span in spans for p in range(span.start, span.end)]
        exact = service.logprobs(torch.zeros_like(service.base), [tokens[p - 1] for p in positions],
                                 [tokens[p] for p in positions])  # fmt: skip
        # (in float32's precision: `PolicyStep` reads them as float32 tensors)
        behavior = [float(numpy.float32(float(value) + rng.gauss(0.0, noise))) for value in exact]
        advantage = rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.5)
        made.append(Weighted(Segment(tokens, spans, behavior), advantage, source=f"run/1/{index}/ada/0"))
    return made


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
    """A model's directory as the conversion and the merge read it: its configuration and its weights."""
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


def tinkers(shared: bool) -> Callable[[str], dict[str, torch.Tensor]]:
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


SAMPLER = "tinker://fake-run/sampler_weights/kpqxrmtzwvolxqvu"
"""The sampler checkpoint `archives` holds."""


def archives() -> FakeService:
    """A fake Tinker holding one sampler checkpoint, `SAMPLER`, whose archive is an adapter of the tiny model
    (`tinkers`, with an A each for q, k and v): what a bridge in another process asks, by the name
    `tests.rollout_tinker.support:archives`."""
    service = FakeService(vocabulary=24, adapter=tinkers(shared=False), rank=RANK, alpha=ALPHA)
    service.archived(SAMPLER, base_model="tiny")
    return service
