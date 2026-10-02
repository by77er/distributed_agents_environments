"""Training: the loop, the group algorithm and the curriculum, and what they ask of a trainer.

- `train` (`loop`): the loop, over `Jobs`, a `Catalog`, a `Trainer` and a `Store`.
- `Algorithm`, `Batch`, `Grpo`, `complete_groups` (`algorithm`): what the loop asks of an algorithm, and
  group-relative policy optimisation over episodes.
- `Curriculum`: which row next.
- `Trainer`, `Weighted`, `Budget`, `Step`, `StepFailed` (`trainer`): what a trainer is. `Colocated`: the wrapper for
  one that shares its accelerator with the engines.
- `Iteration`, `iterations`, `Store`, `Directory` (`record`): what a run writes down for each group, and where.
"""

from rollout.training.algorithm import Algorithm, Batch, Grpo, complete_groups, group_advantages
from rollout.training.colocated import Colocated
from rollout.training.curriculum import Curriculum
from rollout.training.loop import train
from rollout.training.record import Directory, Iteration, Store, iterations
from rollout.training.trainer import Budget, Step, StepFailed, Trainer, Weighted

__all__ = [
    "Algorithm",
    "Batch",
    "Budget",
    "Colocated",
    "Curriculum",
    "Directory",
    "Grpo",
    "Iteration",
    "Step",
    "StepFailed",
    "Store",
    "Trainer",
    "Weighted",
    "complete_groups",
    "group_advantages",
    "iterations",
    "train",
]
