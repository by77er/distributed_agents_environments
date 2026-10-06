# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The trainers on several processes, on the CPU (gloo): a tiny random Qwen3 model, its steps taken by two resident
processes (`rollout_lora.resident.Workers`, under torchrun) and by the step on one process, give the same losses and
weights; a step from the checkpoint the processes made last goes on from what they hold; steps on one process and on
several go on from each other's files; and a full-weight trainer's sharded state is read back by another number of
processes. (On the CPU nothing is cast to bfloat16 by the sharding, so the two differ only by the order sums are added
in; the GPU's mixed precision is `test_sharded_on_gpu.py`'s.)"""

import asyncio
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from rollout_lora.full import FullPolicy
from rollout_lora.layers import load_adapter
from rollout_lora.policy import Policy
from rollout_lora.resident import Workers
from rollout_lora.settings import LoraSettings
from rollout_lora.sharded import SHARDS
from rollout_lora.workers import OPTIMIZER, SEED
from rollout_objectives.step import PolicyStep
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import HELD, Files, Item, StepFailed

SETTINGS: dict[str, Any] = {"rank": 4, "learning_rate": 1e-3, "tokens_per_step": 24, "max_kl": None}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A random Qwen3 of two layers, saved as a model's directory."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=False,
                         max_position_embeddings=256)  # fmt: skip
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("tiny")
    model: Any = Qwen3ForCausalLM(config)
    model.to(torch.bfloat16).save_pretrained(directory)
    return str(directory)


def batch(score: Any) -> list[Item]:
    """Segments of several lengths, sampled at logprobs a little off what `score` gives them now."""
    rng = random.Random(0)
    found: list[Item] = []
    for index in range(5):
        length = 10 + 4 * index
        tokens = [rng.randrange(96) for _ in range(length)]
        start = length - 6
        with torch.no_grad():
            exact = score(tokens, range(start, length)).float()
        behavior = (exact + 0.05 * torch.randn(exact.shape)).tolist()
        found.append(Weighted(Segment(tokens, [Span(start, length, 0)], behavior), 1.0 if index % 2 else -0.5))
    return found


def adapter(directory: Path) -> dict[str, torch.Tensor]:
    return load_file(str(directory / "adapter_model.safetensors"))


def close(one: dict[str, float], two: dict[str, float], keys: tuple[str, ...]) -> None:
    for key in keys:
        assert two[key] == pytest.approx(one[key], rel=1e-4, abs=1e-6), key


COMPARED = ("loss", "tokens", "segments", "kl_moved", "mean_ratio", "gradient_norm", "optimizer_steps")


@pytest.mark.parametrize("whole_base", [False, True])
def test_an_adapter_on_two_processes_steps_as_on_one_and_goes_on_from_what_they_hold(
    tiny: str, tmp_path: Path, whole_base: bool
) -> None:
    settings = LoraSettings(**SETTINGS, whole_base=whole_base)
    torch.manual_seed(SEED)  # (the new adapter the processes draw)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)

    workers = Workers(tiny, settings, "lora", 2, device="cpu")
    try:
        first = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
        made = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
        second = asyncio.run(workers.step(given, seed=1, parent=made, into=tmp_path / "second"))
        holding = workers.holding
    finally:
        workers.close()
    assert workers.holding is None  # (closed: the processes, and what they held, are gone)
    assert first["gpus"] == 2.0 and first["whole_base"] == float(whole_base) and first["loaded_from_files"] == 1.0
    assert second["loaded_from_files"] == 0.0  # (its parent is what the processes made last: nothing was loaded)
    assert holding is not None and (tmp_path / "second" / "state" / HELD).read_text() == holding
    assert (tmp_path / "second" / "state" / OPTIMIZER).exists() and second["full_state"] == 1.0

    # One process, from the same start: the same losses, and the same adapter.
    alone = PolicyStep(policy, settings)
    one = alone.step(given, seed=0)
    close(one, first, COMPARED)
    one = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer).step(given, seed=1)
    close(one, second, COMPARED)
    policy.save(tmp_path / "alone")
    saved, ours = adapter(tmp_path / "second" / "weights"), adapter(tmp_path / "alone")
    assert set(saved) == set(ours)
    for key in saved:
        torch.testing.assert_close(saved[key], ours[key], rtol=1e-4, atol=1e-6)

    # A step on one process goes on from the files two wrote: the adapter, and the optimizer's state.
    again = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    load_adapter(again.model, tmp_path / "second" / "weights")
    going = PolicyStep(again, settings, fresh=False)
    going.optimizer.load_state_dict(torch.load(tmp_path / "second" / "state" / OPTIMIZER, weights_only=True))
    third_alone = going.step(given, seed=2)
    third_more = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer).step(given, seed=2)
    close(third_more, third_alone, COMPARED)


def test_processes_that_hold_nothing_load_their_parent_and_go_on_as_one_does(tiny: str, tmp_path: Path) -> None:
    settings = LoraSettings(**SETTINGS)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    alone = PolicyStep(policy, settings)
    alone.step(given, seed=0)
    policy.save(tmp_path / "first" / "weights")
    (tmp_path / "first" / "state").mkdir(parents=True)
    torch.save(alone.optimizer.state_dict(), tmp_path / "first" / "state" / OPTIMIZER)  # (as one GPU's step writes)
    parent = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
    workers = Workers(tiny, settings, "lora", 2, device="cpu")
    try:
        more = asyncio.run(workers.step(given, seed=1, parent=parent, into=tmp_path / "second"))
    finally:
        workers.close()
    assert more["loaded_from_files"] == 1.0
    one = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer).step(given, seed=1)
    close(one, more, COMPARED)


def test_every_weight_on_two_processes_steps_as_on_one_and_its_state_is_read_by_another_count(
    tiny: str, tmp_path: Path
) -> None:
    changed: dict[str, Any] = {**SETTINGS, "learning_rate": 1e-4, "state_every": 2}
    settings = LoraSettings(**changed)
    policy = FullPolicy.load(tiny, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    workers = Workers(tiny, settings, "full", 2, device="cpu")
    try:
        first = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
        made = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
        second = asyncio.run(workers.step(given, seed=1, parent=made, into=tmp_path / "second"))
    finally:
        workers.close()
    assert first["full_state"] == 0.0 and second["full_state"] == 1.0  # (every second step since they loaded)
    assert not (tmp_path / "first" / "state" / SHARDS).exists() and (tmp_path / "second" / "state" / SHARDS).is_dir()
    weights = tmp_path / "second" / "weights"
    assert (weights / "config.json").exists() and (weights / "model.safetensors.index.json").exists()
    served = {key: each for file in weights.glob("*.safetensors") for key, each in load_file(str(file)).items()}
    assert {each.dtype for each in served.values()} == {torch.bfloat16}  # (the serving copy)

    alone = PolicyStep(policy, settings)
    close(alone.step(given, seed=0), first, COMPARED)
    stepping = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer)
    close(stepping.step(given, seed=1), second, COMPARED)
    for name, parameter in policy.model.named_parameters():
        torch.testing.assert_close(served[name], parameter.detach().to(torch.bfloat16), rtol=1e-2, atol=1e-3)

    # Three processes read the full state two wrote (its float32 weights and the optimizer's), and go on as one does.
    parent = Files(weights, tmp_path / "second" / "state")
    workers = Workers(tiny, settings, "full", 3, device="cpu")
    try:
        third = asyncio.run(workers.step(given, seed=2, parent=parent, into=tmp_path / "third"))
    finally:
        workers.close()
    assert third["loaded_from_files"] == 1.0
    close(stepping.step(given, seed=2), third, COMPARED)


def test_a_step_that_fails_in_the_processes_ends_them_and_the_next_starts_them_again(tiny: str, tmp_path: Path) -> None:
    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(SEED)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    broken = Weighted(Segment([1, 2, 3, 4], [Span(2, 4, 0)], [-0.5, float("nan")]), 1.0)  # (no behaviour logprob)
    workers = Workers(tiny, settings, "lora", 2, device="cpu")
    try:
        with pytest.raises(StepFailed, match="no behavior logprob"):
            asyncio.run(workers.step([*given, broken], seed=0, parent=None, into=tmp_path / "broken"))
        assert workers.holding is None
        again = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "again"))
    finally:
        workers.close()
    assert again["loaded_from_files"] == 1.0 and again["segments"] == 5.0
