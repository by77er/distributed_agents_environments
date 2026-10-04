# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The substitutions are exact: a step through Tinker's losses (the fake computes them as Tinker's documentation
writes them) moves a model exactly as `rollout_lora`'s own step does, for every objective and ratio, with one update or
several, and its metrics are the same."""

from pathlib import Path
from typing import Any

import pytest
import torch

from rollout_lora.settings import LoraSettings
from rollout_lora.step import PolicyStep
from rollout_tinker import TinkerTrainer
from rollout_tinker.testing import FakeService
from rollout_tinker.weights import pointer
from rollout_train.trainer import STATE, WEIGHTS, Files
from tests.support import Bigram, segments

SAME = ("loss", "clip_fraction", "mean_ratio", "kl_floor", "mean_mismatch", "mean_weight", "truncated_fraction",
        "kl_moved", "tokens", "segments", "segments_too_long", "optimizer_steps", "stopped_at_max_kl", "gradient_norm",
        "warmup_updates")  # fmt: skip

CASES = {
    # name: settings, and the Tinker calls a step makes
    "token, one update": ({"tokens_per_step": 10**6}, ["lora 32", "forward_backward cispo", "optim"]),
    "token, several": ({"tokens_per_step": 40}, ["lora 32", "forward", "forward_backward ppo", "optim"]),
    "segment, one update": ({"ratio": "segment", "tokens_per_step": 10**6}, ["lora 32", "custom", "optim"]),
    "segment, several": ({"ratio": "segment", "tokens_per_step": 40}, ["lora 32", "forward", "custom", "optim"]),
    "likelihood": ({"objective": "likelihood", "tokens_per_step": 40}, ["lora 32", "forward_backward cross_entropy"]),
    "untruncated, two passes, warmed up": (
        {"truncate": None, "tokens_per_step": 60, "passes": 2, "warmup_updates": 3},
        ["lora 32", "forward", "forward_backward ppo"],
    ),
    "clipped hard": (
        {"clip_low": 0.01, "clip_high": 0.01, "tokens_per_step": 30, "learning_rate": 0.2},
        ["lora 32", "forward", "forward_backward ppo"],
    ),
}


def ours(settings: dict[str, Any]) -> dict[str, Any]:
    return {"learning_rate": 0.05, "max_kl": None, **settings}


async def stepped(service: FakeService, settings: dict[str, Any], tmp_path: Path, **step: Any) -> tuple[Any, Path]:
    trainer = TinkerTrainer("tiny", service=service, **ours(settings))
    into = tmp_path / "made"
    taken = await trainer.step(segments(service, 12), seed=7, parent=None, into=into, **step)
    return taken, into


@pytest.mark.parametrize("case", list(CASES))
async def test_a_step_on_tinker_moves_the_model_as_the_lora_step_does(case: str, tmp_path: Path) -> None:
    settings, calls = CASES[case]
    service = FakeService(vocabulary=24, seed=3)
    taken, into = await stepped(service, settings, tmp_path)
    policy = Bigram(service)
    expected = PolicyStep(policy, LoraSettings(**ours(settings))).step(segments(service, 12), seed=7)

    state = pointer(into / WEIGHTS, "state")
    assert state is not None and pointer(into / STATE, "state") == state
    moved = service.table(state)
    assert float(moved.abs().max()) > 0.01  # (it trained)
    torch.testing.assert_close(moved, policy.table.detach(), rtol=1e-6, atol=1e-9)
    for key in SAME:
        assert taken.metrics[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7), key
    assert all(call in service.calls for call in calls), service.calls
    assert ("forward" in service.calls) == ("forward" in calls)  # (the pass for where the step starts, if needed)
    assert pointer(into / WEIGHTS, "sampler") in service.saved  # (a sampler checkpoint, named after the version)
    assert (into / STATE / "minibatches.jsonl").read_text().count("\n") == int(taken.metrics["optimizer_steps"])


async def test_one_update_needs_no_pass_for_where_the_step_starts(tmp_path: Path) -> None:
    service = FakeService(vocabulary=24, seed=3)
    await stepped(service, {"tokens_per_step": 10**6}, tmp_path)
    assert "forward" not in service.calls  # (cispo against the behaviour: the update's own pass says where it started)


async def test_a_stop_at_max_kl_comes_where_the_lora_step_stops_and_leaves_the_client_unused(tmp_path: Path) -> None:
    service = FakeService(vocabulary=24, seed=3)
    settings = {"tokens_per_step": 20, "max_kl": 1e-4, "learning_rate": 0.2}
    trainer = TinkerTrainer("tiny", service=service, **ours(settings))
    into = tmp_path / "first"
    taken = await trainer.step(segments(service, 12), seed=7, parent=None, into=into)
    policy = Bigram(service)
    expected = PolicyStep(policy, LoraSettings(**ours(settings))).step(segments(service, 12), seed=7)
    assert taken.metrics["stopped_at_max_kl"] == expected["stopped_at_max_kl"] == 1.0
    assert taken.metrics["optimizer_steps"] == expected["optimizer_steps"] > 0
    state = pointer(into / WEIGHTS, "state")
    assert state is not None
    torch.testing.assert_close(service.table(state), policy.table.detach(), rtol=1e-6, atol=1e-9)
    # Its gradient is still accumulated there: the next step starts a client of its own from the saved state.
    service.calls.clear()
    await trainer.step(segments(service, 4, seed=1), seed=8, parent=Files(into / WEIGHTS, into / STATE),
                       into=tmp_path / "second")  # fmt: skip
    assert service.calls[0] == f"state with optimizer {state}"


async def test_a_step_from_its_parent_goes_on_with_its_optimizer_on_the_client_that_saved_it(tmp_path: Path) -> None:
    service = FakeService(vocabulary=24, seed=3)
    settings = ours({"tokens_per_step": 40})
    trainer = TinkerTrainer("tiny", service=service, **settings)
    first, second = tmp_path / "first", tmp_path / "second"
    await trainer.step(segments(service, 8), seed=1, parent=None, into=first)
    service.calls.clear()
    await trainer.step(segments(service, 8, seed=5), seed=2, parent=Files(first / WEIGHTS, first / STATE), into=second)
    assert not any(call.startswith(("lora", "state")) for call in service.calls)  # (the live client went on)

    # The same as the LoRA step going on with its optimizer's state:
    policy = Bigram(service)
    before = PolicyStep(policy, LoraSettings(**settings))
    before.step(segments(service, 8), seed=1)
    after = PolicyStep(policy, LoraSettings(**settings), fresh=False)
    after.optimizer.load_state_dict(before.optimizer.state_dict())
    after.step(segments(service, 8, seed=5), seed=2)
    state = pointer(second / WEIGHTS, "state")
    assert state is not None
    torch.testing.assert_close(service.table(state), policy.table.detach(), rtol=1e-6, atol=1e-9)

    # A trainer of its own (another process, a run started again) resumes the state, optimizer and all, to the same.
    service.calls.clear()
    again = TinkerTrainer("tiny", service=service, **settings)
    await again.step(segments(service, 8, seed=5), seed=2, parent=Files(first / WEIGHTS, first / STATE),
                     into=tmp_path / "again")  # fmt: skip
    assert service.calls[0] == f"state with optimizer {pointer(first / STATE, 'state')}"
    redone = pointer(tmp_path / "again" / WEIGHTS, "state")
    assert redone is not None
    torch.testing.assert_close(service.table(redone), service.table(state), rtol=0, atol=0)


async def test_a_parent_given_without_its_state_starts_a_fresh_optimizer_from_its_weights(tmp_path: Path) -> None:
    service = FakeService(vocabulary=24, seed=3)
    settings = ours({"objective": "likelihood", "tokens_per_step": 40, "warmup_updates": 2})
    trainer = TinkerTrainer("tiny", service=service, **settings)
    first = tmp_path / "first"
    await trainer.step(segments(service, 8), seed=1, parent=None, into=first)
    parent_state = pointer(first / WEIGHTS, "state")
    assert parent_state is not None
    service.calls.clear()
    taken = await trainer.step(segments(service, 8, seed=5), seed=2, parent=Files(first / WEIGHTS), into=tmp_path / "b")
    assert service.calls[0] == f"state {parent_state}" and taken.metrics["warmup_updates"] == 2.0

    policy = Bigram(service, service.table(parent_state))
    PolicyStep(policy, LoraSettings(**settings), fresh=True).step(segments(service, 8, seed=5), seed=2)
    state = pointer(tmp_path / "b" / WEIGHTS, "state")
    assert state is not None
    torch.testing.assert_close(service.table(state), policy.table.detach(), rtol=1e-6, atol=1e-9)


async def test_a_parent_not_trained_on_tinker_is_refused_not_started_over(tmp_path: Path) -> None:
    from rollout_train.trainer import StepFailed

    elsewhere = tmp_path / "lora" / WEIGHTS
    elsewhere.mkdir(parents=True)
    (elsewhere / "adapter_model.safetensors").write_bytes(b"")
    trainer = TinkerTrainer("tiny", service=FakeService(vocabulary=24))
    with pytest.raises(StepFailed, match="not trained on Tinker"):
        await trainer.step(
            segments(FakeService(vocabulary=24), 2), seed=0, parent=Files(elsewhere), into=tmp_path / "x"
        )


async def test_tinkers_errors_become_a_failed_step_and_the_live_client_is_let_go(tmp_path: Path) -> None:
    from tinker import TinkerError

    from rollout_train.trainer import StepFailed

    service = FakeService(vocabulary=24)
    trainer = TinkerTrainer("tiny", service=service, **ours({}))
    first = tmp_path / "first"
    await trainer.step(segments(service, 4), seed=1, parent=None, into=first)
    service.saved.clear()  # (its checkpoints are gone: resuming fails)

    async def broken(*arguments: Any, **options: Any) -> Any:
        raise TinkerError("the service is down")

    trainer._live = None  # pyright: ignore[reportPrivateUsage]
    service.create_training_client_from_state_with_optimizer_async = broken  # type: ignore[method-assign]
    with pytest.raises(StepFailed, match="tinker: TinkerError: the service is down"):
        await trainer.step(
            segments(service, 4), seed=1, parent=Files(first / WEIGHTS, first / STATE), into=tmp_path / "b"
        )
