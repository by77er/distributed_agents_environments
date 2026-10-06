# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""What `test_sharded.py` runs under torchrun on the CPU (gloo), to look inside a sharded step:

    python -m torch.distributed.run --standalone --nproc-per-node N tests/rollout_lora/sharded_alone.py MODE MODEL OUT

- `precision` (one process): an adapter's step sharded with FSDP2 on one process (the frozen model whole, and
  shared), against the step on the policy as it is, from the same adapter and optimizer's state, and against the
  adapter's units gathered in bfloat16; writes how far apart each is from the policy's own step.
- `accumulated` (two processes): one minibatch's gradient of an adapter sharded with its frozen model whole, reduced
  once (`rollout_lora.sharded.gradient_sync`) and after each pass; writes the largest difference of the gradients, and
  how many reductions each took.
"""

import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rollout_lora.layers import LoraLinear, adapter_tensors
from rollout_lora.policy import Policy, layers_of
from rollout_lora.settings import LoraSettings
from rollout_lora.sharded import joined, shard_adapter, summed, whole
from rollout_lora.workers import SEED
from rollout_objectives.step import PolicyStep
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Item

SETTINGS = LoraSettings(rank=4, learning_rate=1e-3, tokens_per_step=24, max_kl=None)


def batch(policy: Policy) -> list[Item]:
    rng = random.Random(0)
    found: list[Item] = []
    for index in range(5):
        length = 10 + 4 * index
        tokens = [rng.randrange(96) for _ in range(length)]
        with torch.no_grad():
            exact = policy.logprobs(tokens, range(length - 6, length)).float()
        found.append(Weighted(Segment(tokens, [Span(length - 6, length, 0)], (exact + 0.05).tolist()),
                              1.0 if index % 2 else -0.5))  # fmt: skip
    return found


def loaded(model: str) -> Policy:
    torch.manual_seed(SEED)
    return Policy.load(model, rank=SETTINGS.rank, alpha=SETTINGS.alpha, device="cpu")


def stepped(policy: Policy, given: list[Item], ranks: Any = None) -> dict[str, torch.Tensor]:
    """Two steps from the same start: the second goes on from the first's optimizer (as a parent's state)."""
    extra: dict[str, Any] = {} if ranks is None else {"ranks": ranks}
    first = PolicyStep(policy, SETTINGS, **extra)
    first.step(given, seed=0)
    PolicyStep(policy, SETTINGS, fresh=False, optimizer_given=first.optimizer, **extra).step(given, seed=1)
    return adapter_tensors(policy.model, whole)


def bfloat16_adapters(policy: Policy, mesh: Any) -> None:
    """The adapter's units gathered in bfloat16, as the frozen model's are."""
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in layers_of(policy.model):
        units: list[nn.Module] = [part for each in layer.modules() if isinstance(each, LoraLinear)
                                  for part in (each.lora_A, each.lora_B)]  # fmt: skip
        fully_shard(units, mesh=mesh, mp_policy=bf16)
        fully_shard(layer, mesh=mesh, mp_policy=bf16, reshard_after_forward=False)
    fully_shard(policy.scorer, mesh=mesh, mp_policy=bf16)
    cast(FSDPModule, policy.scorer).set_reshard_after_backward(False, recurse=False)
    summed(policy.scorer)


def apart(one: dict[str, torch.Tensor], two: dict[str, torch.Tensor]) -> float:
    return max(float((one[key] - two[key]).abs().max()) for key in one)


def precision(model: str) -> dict[str, Any]:
    ranks, device, mesh = joined("cpu")
    plain = loaded(model)
    given = batch(plain)
    alone = stepped(plain, given)

    def whole_base(policy: Policy) -> None:
        shard_adapter(policy, mesh, device, whole_base=True)

    def shared(policy: Policy) -> None:
        shard_adapter(policy, mesh, device, whole_base=False)

    def bfloat16(policy: Policy) -> None:
        bfloat16_adapters(policy, mesh)

    found: dict[str, Any] = {}
    shards: dict[str, Callable[[Policy], None]] = {"whole": whole_base, "shared": shared, "bfloat16": bfloat16}
    for name, shard in shards.items():
        policy = loaded(model)
        shard(policy)
        found[name] = apart(alone, stepped(policy, given, ranks))
    return found


def accumulated(model: str) -> dict[str, Any]:
    import importlib

    group: Any = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_param_group")  # (to count them)
    ranks, device, mesh = joined("cpu")
    reductions = [0]
    reducing = group.foreach_reduce

    def counted(*arguments: Any, **keywords: Any) -> Any:
        reductions[0] += 1
        return reducing(*arguments, **keywords)

    group.foreach_reduce = counted
    found: dict[str, Any] = {}
    gradients: dict[str, list[torch.Tensor]] = {}
    for name in ("once", "each"):
        policy = loaded(model)
        given = batch(policy)
        shard_adapter(policy, mesh, device, whole_base=True)
        if name == "each":
            policy.gradient_sync = None
        settings = LoraSettings(rank=4, learning_rate=1e-3, tokens_per_step=10_000, max_kl=None, pack_tokens=1)
        stepping = PolicyStep(policy, settings, ranks=ranks, optimizer_given=Kept(policy.parameters()))
        reductions[0] = 0
        stepping.step(given, seed=0)
        found[f"reductions_{name}"] = reductions[0]
        gradients[name] = [cast(Any, each.grad).to_local().clone() for each in policy.parameters()]
    found["apart"] = max(float((one - two).abs().max()) for one, two in zip(*gradients.values(), strict=True))
    found["largest"] = max(float(each.abs().max()) for each in gradients["each"])
    return found


class Kept(torch.optim.SGD):
    """An optimizer that keeps the gradient, unapplied."""

    def __init__(self, parameters: list[nn.Parameter]) -> None:
        super().__init__(parameters, lr=0.0)

    def step(self, closure: Any = None) -> None:
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None


def main() -> None:
    import torch.distributed as distributed

    mode, model, out = sys.argv[1:4]
    found = precision(model) if mode == "precision" else accumulated(model)
    if distributed.get_rank() == 0:
        Path(out).write_text(json.dumps(found))
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
