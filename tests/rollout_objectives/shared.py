# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""A toy policy's steps, for `test_shared.py`: each case's step on one process (`steps`), one segment at a time and in
packs, and the same steps shared among the processes `torchrun` starts, the model sharded with FSDP2 over gloo on the
CPU (`python -m torch.distributed.run --nproc-per-node N shared.py OUT`: rank 0 writes what each made to OUT).

The toy is in float64 and computes without mixed precision, so a shared step and a single one differ only by the order
their sums are added in."""

import json
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rollout_objectives.packing import Pack, Scores
from rollout_objectives.ranks import Ranks
from rollout_objectives.settings import StepSettings
from rollout_objectives.step import PolicyStep
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Item, Labelled, Pair, Weighted

VOCABULARY = 12
CASES: dict[str, dict[str, Any]] = {
    "default": {"objective": "default", "max_kl": None},  # (a token mean)
    "stopped": {"objective": "default", "max_kl": 0.002, "learning_rate": 0.2},
    "gspo": {"objective": "gspo", "max_kl": None},  # (a segment mean)
    "dr_grpo": {"objective": "dr_grpo", "max_kl": None},  # (a constant)
    "grpo": {"objective": "grpo", "max_kl": None},  # (a segment mean, and a KL to the reference)
    "reinforce": {"objective": "reinforce", "max_kl": None, "passes": 2},  # (a segment sum, two shuffled passes)
    "sft": {"objective": "sft"},
    "dpo": {"objective": "dpo", "max_kl": None},
    "kto": {"objective": "kto", "max_kl": None},
}
"""Each case's settings, beside `SETTINGS`."""
SETTINGS: dict[str, Any] = {"learning_rate": 0.05, "tokens_per_step": 14}
PREFIX = 34
"""Tokens every segment of a batch starts with: in packs, groups share them."""
PACK_TOKENS = 50
"""A pack's tokens, in the packed cases: two or three segments' after the prefix, so a minibatch takes one pack or
two."""


class Toy(nn.Module):
    """A next-token model over a tiny vocabulary, of layers enough to shard."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, 16)
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.head = nn.Linear(16, VOCABULARY)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return torch.log_softmax(self.head(hidden), -1)


class ToyPolicy:
    """The toy as a policy: one segment at a time, or in packs (`packing`), each pack's row through the model in one
    call (the toy reads each token alone, so a pack gives each segment what it has alone)."""

    def __init__(self, model: nn.Module, frozen: nn.Module, *, packing: bool = False) -> None:
        self.model = model
        self.frozen = frozen
        self.packing = packing

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        return self._scored(self.model, tokens, positions)

    def reference(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        with torch.no_grad():
            return self._scored(self.frozen, tokens, positions)

    def packed(self, pack: Pack, *, entropy: bool = False, candidates: object = None) -> list[Scores]:
        return [Scores(each) for each in self._packed(self.model, pack)]

    def packed_reference(self, pack: Pack) -> list[torch.Tensor]:
        with torch.no_grad():
            return self._packed(self.frozen, pack)

    @staticmethod
    def _packed(model: nn.Module, pack: Pack) -> list[torch.Tensor]:
        rows = model(torch.tensor(pack.tokens))
        found: list[torch.Tensor] = []
        for index in range(len(pack.segments)):
            places, targets = pack.scored(index)
            found.append(rows[places].gather(-1, torch.tensor(targets).unsqueeze(-1)).squeeze(-1))
        return found

    def idle(self, *, gradient: bool = False, reference: bool = False) -> None:
        """A pass of nothing learnt, as a sharded policy takes one (`rollout_objectives.step.SharedPolicy`)."""
        if reference:
            self.reference([0, 0], [1])
            return
        with torch.set_grad_enabled(gradient):
            found = self.logprobs([0, 0], [1])
            if gradient:
                (found.sum() * 0.0).backward()

    def clip_gradients(self, maximum: float, ranks: Ranks) -> float:
        """The gradient clipped by its norm over every process's shard."""
        if not ranks.shared:
            return float(torch.nn.utils.clip_grad_norm_(self.parameters(), maximum))
        from torch.distributed.tensor import DTensor

        gradients = [each.grad for each in self.parameters() if each.grad is not None]
        local = sum(float(cast(DTensor, each).to_local().double().square().sum()) for each in gradients)
        total = math.sqrt(ranks.summed([local])[0])
        coefficient = min(1.0, maximum / (total + 1e-6))
        if coefficient < 1.0:
            for each in gradients:
                each.mul_(coefficient)
        return total

    @staticmethod
    def _scored(model: nn.Module, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        rows = model(ids[[position - 1 for position in positions]])
        return rows.gather(-1, ids[list(positions)].unsqueeze(-1)).squeeze(-1)


def models() -> tuple[Toy, Toy]:
    torch.manual_seed(0)
    model = Toy().double()
    frozen = Toy().double()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return model, frozen


def batch(case: str, policy: ToyPolicy) -> list[Item]:
    """Segments of several lengths after a prefix they all share, sampled at logprobs a little off the policy's:
    weighted ones, or a preference loss's pairs and labelled examples of them."""
    rng = random.Random(1)
    shared = [rng.randrange(VOCABULARY) for _ in range(PREFIX)]
    made: list[Segment] = []
    for index in range(7):
        length = PREFIX + 4 + (index * 5) % 9
        tokens = shared + [rng.randrange(VOCABULARY) for _ in range(length - PREFIX)]
        start = PREFIX + 1 + index % 3
        with torch.no_grad():
            exact = policy.logprobs(tokens, range(start, length))
        behavior = (exact + 0.05 * torch.randn(exact.shape, dtype=exact.dtype)).tolist()
        made.append(Segment(tokens, [Span(start, length, 0)], behavior))
    if case == "dpo":
        return [Pair((made[0],), (made[1],)), Pair((made[2], made[3]), (made[4],)), Pair((made[5],), (made[6],))]
    if case == "kto":
        return [Labelled((each,), index % 2 == 0) for index, each in enumerate(made)]
    return [Weighted(each, 1.0 if index % 3 else -0.7) for index, each in enumerate(made)]


def settings_of(case: str, *, packing: bool = False) -> StepSettings:
    return StepSettings(**{**SETTINGS, **CASES[case]}, pack_tokens=PACK_TOKENS if packing else None)


def steps(case: str, policy: ToyPolicy, ranks: Ranks) -> dict[str, Any]:
    """Two steps of the case on the policy: their metrics and minibatches, and the weights after them."""
    settings = settings_of(case, packing=policy.packing)
    stepping = PolicyStep(policy, settings, ranks=ranks)
    found: dict[str, Any] = {"metrics": [], "minibatches": []}
    for seed in range(2):
        found["metrics"].append(stepping.step(batch(case, policy), seed=seed))
        found["minibatches"].append(list(stepping.minibatches))
        stepping = PolicyStep(policy, settings, fresh=False, ranks=ranks, optimizer_given=stepping.optimizer)
    return found


def nothing_folded(plan: object, settings: object) -> list[Item]:
    """In place of `rollout_objectives.step._first_minibatch`: no first minibatch's start folded into it, every start
    computed in the step's first pass."""
    return []


def whole(model: nn.Module) -> dict[str, list[float]]:
    """A model's weights, gathered where they are sharded."""
    from torch.distributed.tensor import DTensor

    return {
        name: (each.full_tensor() if isinstance(each, DTensor) else each).detach().flatten().tolist()
        for name, each in model.named_parameters()
    }


def main(out: Path) -> None:
    import torch.distributed as distributed
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard

    distributed.init_process_group("gloo")
    rank, size = distributed.get_rank(), distributed.get_world_size()
    mesh = init_device_mesh("cpu", (size,))
    found: dict[str, Any] = {}
    for case, packing in [(case, packing) for case in CASES for packing in (False, True)]:
        model, frozen = models()
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        for module in model.modules():  # (each sharded module's own: the gradients are added up, not averaged)
            if isinstance(module, FSDPModule):
                module.set_gradient_divide_factor(1.0)
                module.set_force_sum_reduction_for_comms(True)
        policy = ToyPolicy(model, frozen, packing=packing)
        ranks = Ranks(rank, size, distributed.new_group(backend="gloo"))
        found[f"{case}+packed" if packing else case] = steps(case, policy, ranks) | {"weights": whole(model)}
    if rank == 0:
        out.write_text(json.dumps(found))
    distributed.destroy_process_group()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
