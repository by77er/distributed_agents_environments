# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The trainers' processes on the CPU (gloo): a tiny random Qwen3 model, its steps taken by two resident processes
(`rollout_lora.resident.Workers`, under torchrun) and by the step on one process, give the same losses and weights; a
step from the checkpoint the processes made last goes on from what they hold, and one after they were ended goes on
from the parent's full state, or is refused where the parent left it out; steps on one process and on several go on
from each other's files; a full-weight trainer's sharded state is read back by another number of processes; one
process kept between steps goes on from memory, and one beside an engine ends after each step; what the processes
hold is dropped before another parent is loaded; an adapter's units sharded on one process step bitwise as the policy
does, and its gradients reduced once a minibatch are those reduced after each pass. (On the CPU full weights are not
cast to bfloat16 by the sharding, so the two differ only by the order sums are added in; the GPU's mixed precision is
`test_sharded_on_gpu.py`'s.)"""

import asyncio
import gc
import json
import os
import random
import signal
import subprocess
import sys
import weakref
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
from rollout_lora.trainer import LoraTrainer
from rollout_lora.workers import OPTIMIZER, SEED, SHARDS, Asked
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


@pytest.mark.parametrize(("whole_base", "objective"), [(False, "default"), (True, "default"), (True, "grpo")])
def test_an_adapter_on_two_processes_steps_as_on_one_and_goes_on_from_what_they_hold(
    tiny: str, tmp_path: Path, whole_base: bool, objective: str
) -> None:
    # (GRPO reads the reference: the adapter switched off, which its layers' unit is not gathered for)
    settings = LoraSettings(**SETTINGS, whole_base=whole_base, objective=objective)
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

    # Three processes read the full state two wrote (its float32 weights and the optimizer's), and go on as one does;
    # the first step's state, which left the full state out, they refuse: no process holds it.
    parent = Files(weights, tmp_path / "second" / "state")
    workers = Workers(tiny, settings, "full", 3, device="cpu")
    try:
        third = asyncio.run(workers.step(given, seed=2, parent=parent, into=tmp_path / "third"))
        with pytest.raises(StepFailed, match="no full state"):
            asyncio.run(workers.step(given, seed=1, parent=made, into=tmp_path / "again"))
        assert workers.holding is not None  # (refused before the processes were asked: they hold the third step)
    finally:
        workers.close()
    assert third["loaded_from_files"] == 1.0
    close(stepping.step(given, seed=2), third, COMPARED)


def test_every_weight_goes_on_from_its_full_state_after_its_processes_end(tiny: str, tmp_path: Path) -> None:
    # (as after a failed step, a lease released or a restart: new processes take the second step from the first's files)
    changed: dict[str, Any] = {**SETTINGS, "learning_rate": 1e-4, "warmup_updates": 4}
    settings = LoraSettings(**changed)
    policy = FullPolicy.load(tiny, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    workers = Workers(tiny, settings, "full", 2, device="cpu")
    try:
        first = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
    finally:
        workers.close()
    assert first["full_state"] == 1.0 and (tmp_path / "first" / "state" / SHARDS).is_dir()  # (every step, by default)
    assert first["warmup_updates"] == 4.0
    workers = Workers(tiny, settings, "full", 2, device="cpu")
    try:
        parent = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
        second = asyncio.run(workers.step(given, seed=1, parent=parent, into=tmp_path / "second"))
    finally:
        workers.close()
    alone = PolicyStep(policy, settings)
    alone.step(given, seed=0)
    going = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer).step(given, seed=1)
    assert second["loaded_from_files"] == 1.0 and second["warmup_updates"] == 0.0  # (its optimizer went on)
    close(going, second, COMPARED)
    served = {key: each for file in (tmp_path / "second" / "weights").glob("*.safetensors")
              for key, each in load_file(str(file)).items()}  # fmt: skip
    for name, parameter in policy.model.named_parameters():
        torch.testing.assert_close(served[name], parameter.detach().to(torch.bfloat16), rtol=1e-2, atol=1e-3)


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


def test_every_weight_with_a_frozen_reference_on_two_processes_steps_as_on_one(tiny: str, tmp_path: Path) -> None:
    changed: dict[str, Any] = {**SETTINGS, "learning_rate": 1e-4, "objective": "grpo", "frozen_reference": True}
    settings = LoraSettings(**changed)
    policy = FullPolicy.load(tiny, reference=tiny, device="cpu")  # (the reference: a second sharded model)
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    workers = Workers(tiny, settings, "full", 2, device="cpu")
    try:
        first = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
        made = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
        second = asyncio.run(workers.step(given, seed=1, parent=made, into=tmp_path / "second"))
    finally:
        workers.close()
    alone = PolicyStep(policy, settings)
    close(alone.step(given, seed=0), first, (*COMPARED, "kl_penalty"))
    following = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer)
    close(following.step(given, seed=1), second, (*COMPARED, "kl_penalty"))
    assert second["kl_penalty"] > 0  # (the second step's policy has moved from its reference)


def test_one_process_kept_goes_on_from_memory_and_one_beside_an_engine_ends_after_each_step(
    tiny: str, tmp_path: Path
) -> None:
    assert LoraTrainer(tiny, gpus=1)._process.kept  # pyright: ignore[reportPrivateUsage]  (a GPU to itself)
    assert not LoraTrainer(tiny, gpus=1, colocated=True)._process.kept  # pyright: ignore[reportPrivateUsage]
    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(SEED)
    policy = Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu")
    torch.manual_seed(1)
    given = batch(policy.logprobs)
    found: dict[bool, list[dict[str, float]]] = {}
    for kept in (True, False):
        workers = Workers(tiny, settings, "lora", 1, device="cpu", kept=kept)
        here = tmp_path / str(kept)
        try:
            first = asyncio.run(workers.step(given, seed=0, parent=None, into=here / "first"))
            running = workers._process is not None  # pyright: ignore[reportPrivateUsage]
            assert (workers.holding is not None) == kept and running == kept
            made = Files(here / "first" / "weights", here / "first" / "state")
            second = asyncio.run(workers.step(given, seed=1, parent=made, into=here / "second"))
        finally:
            workers.close()
        state = here / "second" / "state"
        assert (state / HELD).exists() == kept and (state / OPTIMIZER).exists()
        found[kept] = [first, second]
    assert found[True][1]["loaded_from_files"] == 0.0 and found[False][1]["loaded_from_files"] == 1.0
    assert found[True][0]["gpus"] == 1.0 and found[True][0]["whole_base"] == 1.0
    from_memory = adapter(tmp_path / "True" / "second" / "weights")
    from_files = adapter(tmp_path / "False" / "second" / "weights")
    for key in from_memory:  # (the same step: from memory, and from the files, float32 both)
        assert torch.equal(from_memory[key], from_files[key]), key
    alone = PolicyStep(policy, settings)
    close(alone.step(given, seed=0), found[True][0], COMPARED)
    going = PolicyStep(policy, settings, fresh=False, optimizer_given=alone.optimizer)
    close(going.step(given, seed=1), found[True][1], COMPARED)


def test_what_the_processes_hold_is_nothing_once_torchrun_is_gone(tiny: str, tmp_path: Path) -> None:
    settings = LoraSettings(**SETTINGS)
    torch.manual_seed(1)
    given = batch(Policy.load(tiny, rank=settings.rank, alpha=settings.alpha, device="cpu").logprobs)
    workers = Workers(tiny, settings, "lora", 2, device="cpu")
    try:
        asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "first"))
        assert workers.holding is not None
        process = workers._process  # pyright: ignore[reportPrivateUsage]
        assert process is not None
        os.killpg(process.pid, signal.SIGKILL)  # (torchrun died while idle)
        process.wait(timeout=30)
        assert workers.holding is None
        parent = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
        again = asyncio.run(workers.step(given, seed=1, parent=parent, into=tmp_path / "second"))
    finally:
        workers.close()
    assert again["loaded_from_files"] == 1.0  # (new processes, from the parent's files)


class Held:
    """What a load holds (a policy and its optimizer, on the GPU)."""

    def __init__(self) -> None:
        self.name = ""


def test_what_the_processes_hold_is_dropped_before_another_parent_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rollout_lora.workers as workers

    dropped: list[bool] = []
    previous: list[weakref.ref[Held]] = []

    def loaded(asked: Asked, *_: Any) -> Held:
        dropped.append(all(each() is None for each in previous))  # (whatever was held before is gone)
        made = Held()
        previous.append(weakref.ref(made))
        return made

    def stepped(asked: Asked, held: Held, *_: Any) -> dict[str, float]:
        held.name = asked.held or ""
        (asked.into / "state").mkdir(parents=True, exist_ok=True)
        (asked.into / "state" / HELD).write_text(held.name)
        return {}

    monkeypatch.setattr(workers, "_loaded", loaded)
    monkeypatch.setattr(workers, "_stepped", stepped)

    def freed(device: torch.device) -> None:
        gc.collect()

    monkeypatch.setattr(workers, "_freed", freed)
    settings = LoraSettings(**SETTINGS)

    def asked(name: str, parent: str | None) -> Asked:
        files = Files(tmp_path / parent / "weights", tmp_path / parent / "state") if parent else None
        return Asked("model", settings, "lora", [], 0, files, tmp_path / name, name)

    messages: list[Any] = [("step", asked("a", None)), ("step", asked("b", "a")), ("step", asked("c", None)),
                           ("stop", None)]  # fmt: skip

    class One:
        rank, size = 0, 1

        def gathered(self, value: Any) -> list[Any]:
            return [value]

    sent: list[Any] = []
    serve: Any = getattr(workers, "_serve")  # noqa: B009  (the processes' loop, with its loading and stepping replaced)
    serve(lambda: messages.pop(0), sent.append, One(), torch.device("cpu"), None, 0.0)
    assert [kind for kind, _ in sent] == ["done", "done", "done"]
    assert dropped == [True, True]  # (loaded twice: the second step went on from memory)


def alone(mode: str, model: str, processes: int, out: Path) -> dict[str, Any]:
    """What `sharded_alone.py` finds of `mode` on `processes` processes."""
    script = Path(__file__).with_name("sharded_alone.py")
    environ = {**os.environ, "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={processes}",
               str(script), mode, model, str(out)]  # fmt: skip
    done = subprocess.run(command, env=environ, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(out.read_text())


def test_an_adapters_units_sharded_on_one_process_step_bitwise_as_the_policy_does(tiny: str, tmp_path: Path) -> None:
    found = alone("precision", tiny, 1, tmp_path / "precision.json")
    assert found["whole"] == 0.0 and found["shared"] == 0.0  # (float32 units: what one process computes, exactly)
    assert found["bfloat16"] > 1e-4  # (units gathered in bfloat16 would not be: the comparison sees them)


def test_an_adapters_gradients_reduced_once_a_minibatch_are_those_reduced_after_each_pass(
    tiny: str, tmp_path: Path
) -> None:
    found = alone("accumulated", tiny, 2, tmp_path / "accumulated.json")
    assert found["reductions_once"] == 2 and found["reductions_each"] > found["reductions_once"]  # (a unit a layer)
    assert found["apart"] <= 1e-6 * found["largest"]  # (the same sums, added up in another order)


def test_a_minibatch_out_of_memory_leaves_nothing_in_the_next_ones_gradient(tiny: str, tmp_path: Path) -> None:
    """Full weights sharded on one process: what a backward pass that ran out of memory part way gathered and did not
    reduce (the output layer's gradient, the last norm's, half a layer's) is dropped with it."""
    found = alone("out_of_memory", tiny, 1, tmp_path / "out_of_memory.json")
    assert found["dropped_clean"] == 0 and found["dropped_recovered"] == found["dropped_reset"] == 1
    assert found["recovered"] <= 1e-6 * found["largest"]
    assert found["reset"] > 1e-3 * found["largest"]  # (FSDP's reset alone keeps them: the next minibatch adds them)
