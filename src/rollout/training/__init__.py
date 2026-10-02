"""Training: the loop, the group algorithm and the curriculum, and a reference trainer (4-bit checkpoints with LoRA).

- `train` (`loop`): the loop, over `Jobs`, a `Catalog`, a `Trainer` and a `Store`.
- `Grpo`, `complete_groups` (`algorithm`): group-relative policy optimisation over episodes.
- `Curriculum`: which row next.
- `Trainer`, `LoraTrainer`, `Colocated` (`trainer`): what a trainer is, one that trains LoRA adapters in a process of
  its own, and the wrapper for one that shares its accelerator with the engines.
- `grpo`, `policy`, `lora`, `quantized`, `worker`: the trainer's own code (these import torch; the rest does not).
"""

from rollout.training.algorithm import Grpo, complete_groups, group_advantages
from rollout.training.curriculum import Curriculum
from rollout.training.loop import Directory, Store, iterations, train
from rollout.training.trainer import Budget, Colocated, LoraTrainer, Step, Trainer, Weighted

__all__ = [
    "Budget",
    "Colocated",
    "Curriculum",
    "Directory",
    "Grpo",
    "LoraTrainer",
    "Step",
    "Store",
    "Trainer",
    "Weighted",
    "complete_groups",
    "group_advantages",
    "iterations",
    "train",
]
