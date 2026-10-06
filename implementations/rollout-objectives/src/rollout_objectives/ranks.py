# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch.distributed's collectives are partly untyped.)
"""The processes a step is shared among: one per GPU (`torch.distributed`), each with its part of every minibatch.

Every process takes the whole batch and makes the same plan of it (`rollout_objectives.step.Plan`, shuffled by the
step's seed), so they agree on each minibatch without being told. Each minibatch's packs (or segments, for a policy
that runs one at a time) are shared out by `shares`, balanced by passes, then by tokens; a process computes its own,
and the sums a minibatch reports (its loss, tokens, the distance moved) are added up across processes before anything
reads them (`Ranks.summed`), so every process takes the same decisions (a minibatch with nothing to train on, the stop
at `max_kl`). One process (`Ranks()`, the default) shares nothing, and the step is computed as on one GPU.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["Ranks", "shares"]


@dataclass(frozen=True)
class Ranks:
    """This process's place among those a step is shared among: its `rank` of `size`. `group` is the process group
    their sums and gathers go through: one on the CPU (gloo), whatever the GPUs' own collectives run on (none: the
    default group)."""

    rank: int = 0
    size: int = 1
    group: Any = None

    @property
    def shared(self) -> bool:
        return self.size > 1

    def summed(self, values: Sequence[float]) -> list[float]:
        """Each of `values` added up across the processes (in float64)."""
        if not self.shared:
            return list(values)
        import torch.distributed as distributed

        tensor = torch.tensor(list(values), dtype=torch.float64)
        distributed.all_reduce(tensor, group=self.group)
        return [float(each) for each in tensor.tolist()]

    def gathered[T](self, value: T) -> list[T]:
        """Every process's `value` (picklable), by rank."""
        if not self.shared:
            return [value]
        import torch.distributed as distributed

        found: list[Any] = [None] * self.size
        distributed.all_gather_object(found, value, group=self.group)
        return found

    def most(self, value: float) -> float:
        """The largest of every process's `value`."""
        return max(self.gathered(value))


def shares(sizes: Sequence[int], count: int) -> list[list[int]]:
    """The indices of `sizes` (each a pass's tokens: a pack's, or a segment's) shared among `count` processes,
    balanced by passes, then by tokens: each process takes at most its even share of the count (rounded up, as the
    processes take their passes together), the largest first, each to the process with the fewest tokens so far of
    those with room (the lower rank on a tie). Each process's indices are in their order in `sizes`. Every process
    computes the same shares from the same sizes."""
    most = -(-len(sizes) // count)
    totals = [0] * count
    found: list[list[int]] = [[] for _ in range(count)]
    for index in sorted(range(len(sizes)), key=lambda each: (-sizes[each], each)):
        room = [rank for rank in range(count) if len(found[rank]) < most]
        least = min(room, key=lambda rank: (totals[rank], rank))
        found[least].append(index)
        totals[least] += sizes[index]
    return [sorted(each) for each in found]
