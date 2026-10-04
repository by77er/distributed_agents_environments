"""What every test of rollout-train shares."""

import functools

import pytest

from rollout_train import evals as evals_module
from rollout_train import loop as loop_module
from rollout_train.rollouts import episodes_of


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run, and an eval, look for their groups' episodes in the ledger often (a run looks twice a second)."""
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))
    monkeypatch.setattr(evals_module, "episodes_of", functools.partial(episodes_of, every=0.01))
