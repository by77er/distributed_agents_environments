"""A step shared among processes moves the policy as one process does: the same batch, two steps of each objective's
case (`shared.py`), on one process and on two (FSDP2 over gloo, on the CPU), give the same losses, minibatches,
gradient norms and weights. And a minibatch's segments are shared out balanced by their tokens."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from rollout_objectives.ranks import Ranks, shares
from tests.rollout_objectives import shared as toy

HERE = Path(__file__).parent


def test_segments_are_shared_out_balanced_by_their_tokens() -> None:
    found = shares([900, 100, 400, 500, 300, 200], 2)
    assert found == [[0, 4], [1, 2, 3, 5]]  # (900 + 300 against 100 + 400 + 500 + 200)
    assert shares([5, 5, 5], 4) == [[0], [1], [2], []]  # (one process with none: it takes idle passes)
    assert shares([], 2) == [[], []]
    assert sorted(each for share in shares([7, 3, 9, 1, 4], 3) for each in share) == [0, 1, 2, 3, 4]  # (each once)


def test_one_process_shares_nothing() -> None:
    alone = Ranks()
    assert not alone.shared and alone.summed([1.5, 2.0]) == [1.5, 2.0] and alone.gathered("x") == ["x"]


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("shared") / "two.json"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2",
               str(HERE / "shared.py"), str(out)]  # fmt: skip
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


def _close(one: Any, two: Any, where: str) -> None:
    if isinstance(one, dict):
        ones, twos = cast(dict[str, Any], one), cast(dict[str, Any], two)
        assert set(ones) == set(twos), where
        for key in ones:
            _close(ones[key], twos[key], f"{where}.{key}")
    elif isinstance(one, list):
        ones, twos = cast(list[Any], one), cast(list[Any], two)
        assert len(ones) == len(twos), where
        for index, (left, right) in enumerate(zip(ones, twos, strict=True)):
            _close(left, right, f"{where}[{index}]")
    elif key_free(where):
        assert two == pytest.approx(one, rel=1e-9, abs=1e-12), where  # (float64: only the order of sums differs)


def key_free(where: str) -> bool:
    """Whether a metric is the step's to compare (its wall-clock seconds are not)."""
    return not where.endswith(("seconds", "start_seconds"))


@pytest.mark.parametrize("case", ["default", "stopped", "gspo", "dr_grpo", "grpo", "reinforce", "sft", "dpo", "kto"])
def test_a_step_shared_by_two_processes_is_the_step_one_process_takes(case: str, shared: dict[str, Any]) -> None:
    model, frozen = toy.models()
    policy = toy.ToyPolicy(model, frozen)
    one: dict[str, Any] = {**toy.steps(case, policy, Ranks()), "weights": toy.whole(model)}
    two = shared[case]
    _close(one, two, case)
    if case == "stopped":
        assert one["metrics"][0]["stopped_at_max_kl"] == 1.0  # (both stopped the pass at the same minibatch)
    assert one["metrics"][0]["optimizer_steps"] >= 1
