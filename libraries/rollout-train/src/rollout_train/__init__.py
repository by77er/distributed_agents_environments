"""Training: the loop, the group algorithm and the curriculum, and what they ask of a trainer.

- `train` (`loop`): the loop, over a `Catalog`, a `Trainer` and `Versions`, with runners playing the episodes it asks
  for in the ledger. It can die and be started again.
- `Ledger`, `FileLedger` (`ledger`): append-only tables and fences, the only state the loop has.
- `Versions`, `Version`, `Manifest`, `Retention` (`versions`): the graph of versions, each saying where it came from,
  and their files in a blob store.
- `Algorithm`, `Batch`, `Grpo` (`algorithm`): what the loop asks of an algorithm, and
  group-relative policy optimisation over episodes.
- `Curriculum`: which row next.
- `Trainer`, `Weighted`, `Budget`, `Checkpoint`, `Step`, `StepFailed` (`trainer`): what a trainer is. `Colocated`:
  the wrapper for one that shares its accelerator with the engines.
- `Result`, `results`, `Trained`, `trained` (`record`): how each group of a run went, and what was done with it.
"""

from rollout_train.algorithm import Algorithm, Batch, Grpo, group_advantages
from rollout_train.colocated import Colocated
from rollout_train.curriculum import Curriculum
from rollout_train.ledger import Fence, Fenced, FileLedger, Ledger
from rollout_train.loop import train
from rollout_train.record import Result, Trained, results, trained
from rollout_train.trainer import Budget, Checkpoint, Step, StepFailed, Trainer, Weighted
from rollout_train.versions import Manifest, Retention, Version, Versions

__all__ = [
    "Algorithm",
    "Batch",
    "Budget",
    "Checkpoint",
    "Colocated",
    "Curriculum",
    "Fence",
    "Fenced",
    "FileLedger",
    "Grpo",
    "Ledger",
    "Manifest",
    "Result",
    "Retention",
    "Step",
    "StepFailed",
    "Trained",
    "Trainer",
    "Version",
    "Versions",
    "Weighted",
    "group_advantages",
    "results",
    "train",
    "trained",
]
