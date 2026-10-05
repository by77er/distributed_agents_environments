"""Training: the loop and the group algorithm, and what they ask of a trainer.

- `train` (`loop`): the loop, over an `Environment`, a `Trainer` and `Checkpoints`, with runners playing the episodes it
  asks for in the ledger. It can die and be started again.
- `Ledger`, `FileLedger` (`ledger`): append-only tables and fences, the only state the loop has.
- `Checkpoints`, `Checkpoint`, `Manifest`, `Retention` (`checkpoints`): the graph of checkpoints, each saying where it
  came from, and their files in a blob store.
- `Algorithm`, `Batch`, `Grpo`, `Preferences`, `Distillations`, `algorithm_for` (`algorithm`): what the loop asks of an
  algorithm, weighted segments of a group by the objective's advantage components, a group's pairs or labelled
  examples, and its segments with their teachers' scores. Teacher routing and scoring are in `distillation`.
- `Trainer`, `Weighted`, `Pair`, `Labelled`, `Distilled`, `Budget`, `Files`, `Step`, `StepFailed` (`trainer`): what a
  trainer is and what it trains on; `Changeable`, one that takes some of its settings between steps. `Colocated`: the
  wrapper for one that shares its accelerator with the engines. Its objective is declared in `objectives`.
- `Result`, `results`, `Trained`, `trained` (`record`): how each group of a run went, and what was done with it.
- `evaluate`, `make_suite`, `edit_suite`, `suite_entry`, `suite_for`, `suite_of`, `Suite`, `SuiteEntry`, `Schedule`
  (`evals`): a suite, an eval configuration of one or more environments kept in versions, an eval that plays one
  version with one checkpoint, training nothing, and the evals a training run makes of its checkpoints.
- `make_dataset`, `dataset_of`, `Dataset` (`datasets`): examples chosen from runs' episodes by a rule and turn filters,
  made once, which a supervised step (`imitation`) trains on.
- `Serving`, `record_serving`, `wanted` (`serving`): what each run's channel should serve, written down by the loop;
  `Follower` (`following`): keeps a process's channels serving it, wherever the process runs.
"""

from rollout_train.algorithm import (
    Algorithm,
    Batch,
    Distillations,
    Grpo,
    Preferences,
    algorithm_for,
    group_advantages,
)
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest, Retention
from rollout_train.colocated import Colocated
from rollout_train.datasets import Dataset, dataset_of, make_dataset
from rollout_train.evals import (
    Schedule,
    Suite,
    SuiteEntry,
    edit_suite,
    evaluate,
    make_suite,
    suite_entry,
    suite_for,
    suite_of,
)
from rollout_train.following import Follower
from rollout_train.ledger import Fence, Fenced, FileLedger, Ledger
from rollout_train.loop import train
from rollout_train.record import Result, Trained, results, trained
from rollout_train.serving import Serving, record_serving, wanted
from rollout_train.trainer import (
    Budget,
    Changeable,
    Distilled,
    Files,
    Labelled,
    Pair,
    Step,
    StepFailed,
    Trainer,
    Weighted,
)

__all__ = [
    "Algorithm",
    "Batch",
    "Budget",
    "Changeable",
    "Checkpoint",
    "Checkpoints",
    "Colocated",
    "Dataset",
    "Distillations",
    "Distilled",
    "Fence",
    "Fenced",
    "FileLedger",
    "Files",
    "Follower",
    "Grpo",
    "Labelled",
    "Ledger",
    "Manifest",
    "Pair",
    "Preferences",
    "Result",
    "Retention",
    "Schedule",
    "Serving",
    "Step",
    "StepFailed",
    "Suite",
    "SuiteEntry",
    "Trained",
    "Trainer",
    "Weighted",
    "algorithm_for",
    "dataset_of",
    "edit_suite",
    "evaluate",
    "group_advantages",
    "make_dataset",
    "make_suite",
    "record_serving",
    "results",
    "suite_entry",
    "suite_for",
    "suite_of",
    "train",
    "trained",
    "wanted",
]
