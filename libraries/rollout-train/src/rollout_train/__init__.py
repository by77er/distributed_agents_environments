"""Training: the loop, the group algorithm and the curriculum, and what they ask of a trainer.

- `train` (`loop`): the loop, over a `Catalog`, a `Trainer` and `Checkpoints`, with runners playing the episodes it asks
  for in the ledger. It can die and be started again.
- `Ledger`, `FileLedger` (`ledger`): append-only tables and fences, the only state the loop has.
- `Checkpoints`, `Checkpoint`, `Manifest`, `Retention` (`checkpoints`): the graph of checkpoints, each saying where it
  came from, and their files in a blob store.
- `Algorithm`, `Batch`, `Grpo` (`algorithm`): what the loop asks of an algorithm, and
  group-relative policy optimisation over episodes.
- `Curriculum`: which row next.
- `Trainer`, `Weighted`, `Budget`, `Files`, `Step`, `StepFailed` (`trainer`): what a trainer is. `Colocated`:
  the wrapper for one that shares its accelerator with the engines.
- `Result`, `results`, `Trained`, `trained` (`record`): how each group of a run went, and what was done with it.
- `evaluate`, `make_suite`, `suite_of`, `Suite`, `Start`, `Schedule` (`evals`): a frozen suite of starts, an eval
  that plays it with one checkpoint, training nothing, and the evals a training run makes of its checkpoints.
"""

from rollout_train.algorithm import Algorithm, Batch, Grpo, group_advantages
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest, Retention
from rollout_train.colocated import Colocated
from rollout_train.curriculum import Curriculum
from rollout_train.evals import Schedule, Start, Suite, evaluate, make_suite, suite_of
from rollout_train.ledger import Fence, Fenced, FileLedger, Ledger
from rollout_train.loop import train
from rollout_train.record import Result, Trained, results, trained
from rollout_train.trainer import Budget, Files, Step, StepFailed, Trainer, Weighted

__all__ = [
    "Algorithm",
    "Batch",
    "Budget",
    "Checkpoint",
    "Checkpoints",
    "Colocated",
    "Curriculum",
    "Fence",
    "Fenced",
    "FileLedger",
    "Files",
    "Grpo",
    "Ledger",
    "Manifest",
    "Result",
    "Retention",
    "Schedule",
    "Start",
    "Step",
    "StepFailed",
    "Suite",
    "Trained",
    "Trainer",
    "Weighted",
    "evaluate",
    "group_advantages",
    "make_suite",
    "results",
    "suite_of",
    "train",
    "trained",
]
