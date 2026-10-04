# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""What the tests share: segments on the fake's bigram, and the same bigram as a policy `PolicyStep` trains."""

import random
from collections.abc import Sequence

import numpy
import torch

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
