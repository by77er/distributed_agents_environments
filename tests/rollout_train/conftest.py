"""What every test of rollout-train shares."""

import functools

import pytest

from rollout_train import evals as evals_module
from rollout_train import launching as launching_module
from rollout_train import loop as loop_module
from rollout_train import memory as memory_module
from rollout_train.rollouts import episodes_of


@pytest.fixture(autouse=True)
def no_model_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's check knows no model's size unless a test gives it one: neither the Hugging Face cache of the machine
    the tests run on nor the Hub is the test's (`rollout_train.memory.model_facts`)."""

    def unknown(model: str, **given: object) -> None:
        return None

    monkeypatch.setattr(memory_module, "model_facts", unknown)
    monkeypatch.setattr(launching_module, "_MODEL_FACTS", {})


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run, and an eval, look for their groups' episodes in the ledger often (a run looks twice a second)."""
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))
    monkeypatch.setattr(evals_module, "episodes_of", functools.partial(episodes_of, every=0.01))
