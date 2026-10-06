# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Packs on tiny random models on the CPU, of each kind of attention the trainers pack: Qwen3's and Llama's softmax
attention, and Qwen3.5's hybrid of softmax attention and a gated delta rule (linear attention, a recurrence). A pack
gives each segment the logprobs, entropies and logprobs of given tokens it has alone, with its prefix shared or not;
changing one segment of a pack changes no other's; and a step in packs takes the losses and the gradients a step of
one segment at a time does, for each objective family, with the first minibatch's start folded into it or not. Two
processes (FSDP2 over gloo) step in packs as one does."""

import asyncio
import inspect
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from rollout_lora.layers import add_lora, lora_parameters
from rollout_lora.packing import prepare
from rollout_lora.policy import TARGETS, Policy
from rollout_lora.settings import LoraSettings
from rollout_objectives import step as step_module
from rollout_objectives.packing import Pack, packs
from rollout_objectives.settings import StepSettings
from rollout_objectives.step import PolicyStep, positions
from rollout_train.recorder import Segment, Span, TeacherScores
from rollout_train.trainer import Distilled, Item, Labelled, Pair, Weighted
from tests.rollout_objectives.shared import nothing_folded

VOCABULARY = 96
KINDS = ("qwen3", "qwen3_5", "llama")


@pytest.fixture(autouse=True)
def torch_recurrence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3.5's own forward on the CPU: transformers' recurrence in torch (flash-linear-attention's kernels run on a
    GPU only), the reference a segment alone is compared with."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    reference = inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
    monkeypatch.setattr(modeling_qwen3_5, "torch_chunk_gated_delta_rule", reference)


def config(kind: str) -> Any:
    shape: dict[str, Any] = {"vocab_size": VOCABULARY, "hidden_size": 32, "intermediate_size": 64,
                             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                             "tie_word_embeddings": False, "max_position_embeddings": 512}  # fmt: skip
    if kind == "qwen3":
        from transformers import Qwen3Config

        return Qwen3Config(**shape, num_hidden_layers=2)
    if kind == "llama":
        from transformers import LlamaConfig

        return LlamaConfig(**shape, num_hidden_layers=2)
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(**shape, num_hidden_layers=4, linear_num_key_heads=2, linear_num_value_heads=4,
                             linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                             layer_types=["linear_attention", "full_attention"] * 2)  # fmt: skip


def model_of(kind: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    return cast(nn.Module, AutoModelForCausalLM.from_config(config(kind), dtype=torch.float32))


def policy_of(kind: str) -> Policy:
    """A tiny model with an adapter that has moved (its B is not zero), checkpointed for the backward pass, as
    `Policy.load` makes one."""
    model = model_of(kind)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    add_lora(model, TARGETS, rank=4, alpha=8.0, within="layers", dtype=torch.float32)
    torch.manual_seed(1)
    for name, parameter in model.named_parameters():
        if ".lora_B." in name:
            parameter.data.normal_(0, 0.05)
    cast(Any, model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cast(Any, model).enable_input_require_grads()
    return Policy(model, kind, 4, 8.0, None, prepare(model))


def gridworld(count: int, seed: int = 0) -> list[Segment]:
    """Turns as a gridworld's agents take them: a system prompt every turn shares, an observation some agents share
    the start of, and the sampled reply (its behaviour logprobs filled in later)."""
    rng = random.Random(seed)
    system = [rng.randrange(VOCABULARY) for _ in range(40)]
    views = [[rng.randrange(VOCABULARY) for _ in range(12)] for _ in range(3)]
    found: list[Segment] = []
    for _ in range(count):
        prompt = system + rng.choice(views) + [rng.randrange(VOCABULARY) for _ in range(rng.randrange(4, 20))]
        reply = [rng.randrange(VOCABULARY) for _ in range(rng.randrange(3, 16))]
        tokens = prompt + reply
        found.append(Segment(tokens, [Span(len(prompt), len(tokens), 0)], [0.0] * len(reply)))
    found.append(Segment([rng.randrange(VOCABULARY) for _ in range(30)], [Span(20, 30, 0)], [0.0] * 10))  # (alone)
    return found


def sampled_at(policy: Policy, segments: Sequence[Segment], seed: int = 0) -> list[Segment]:
    """The segments, sampled at logprobs a little off what the policy gives them now."""
    generator = torch.Generator().manual_seed(seed)
    found: list[Segment] = []
    with torch.no_grad():
        for segment in segments:
            exact = policy.logprobs(segment.tokens, positions(segment)).float().cpu()
            noise = 0.05 * torch.randn(exact.shape, generator=generator)
            found.append(Segment(segment.tokens, segment.spans, (exact + noise).tolist()))
    return found


def alone(policy: Policy, pack: Pack, index: int) -> torch.Tensor:
    segment = pack.segments[index]
    return policy.logprobs(segment.tokens, positions(segment))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("share", [False, True])
def test_a_pack_gives_each_segment_what_it_has_alone(kind: str, share: bool) -> None:
    policy = policy_of(kind)
    assert policy.packing
    segments = gridworld(9)
    made = packs(segments, 260, share=share)
    assert len(made) > 1 and sum(len(each.segments) for each in made) == len(segments)
    assert (sum(each.length for each in made) < sum(len(each.tokens) for each in segments)) == share
    with torch.no_grad():
        for pack in made:
            scored = policy.packed(pack, entropy=True)
            references = policy.packed_reference(pack)
            candidates = [torch.randint(0, VOCABULARY, (len(positions(each)), 3)) for each in pack.segments]
            among = policy.packed(pack, candidates=candidates)
            for index, segment in enumerate(pack.segments):
                logprobs, entropy = policy.logprobs_and_entropy(segment.tokens, positions(segment))
                torch.testing.assert_close(scored[index].logprobs, logprobs, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(scored[index].entropy, entropy, rtol=1e-5, atol=1e-5)
                reference = policy.reference(segment.tokens, positions(segment))
                torch.testing.assert_close(references[index], reference, rtol=1e-5, atol=1e-5)
                assert not torch.allclose(reference, logprobs)  # (the adapter moved: the reference differs)
                _, wanted = policy.logprobs_among(segment.tokens, positions(segment), candidates[index])
                torch.testing.assert_close(among[index].among, wanted, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_changing_one_segment_of_a_pack_changes_no_other(kind: str) -> None:
    """Every other segment of a pack changed where it is its own (a branch after the shared prefix, or a whole
    segment alone), the pack laid out as before: the others' logprobs are as they were (no attention, convolution or
    recurrent state crosses a boundary), and the changed ones' are not."""
    policy = policy_of(kind)
    (pack,) = packs(gridworld(7), 10_000)
    parents = {run.parent for run in pack.runs if run.parent is not None}
    assert parents and sum(run.parent is None for run in pack.runs) > len(parents)  # (branches, and one alone)
    shared = {row for index in parents for row in range(pack.runs[index].start, pack.runs[index].end)}
    rng = random.Random(5)
    row = list(pack.tokens)
    segments: list[Segment] = []
    for index, (segment, places) in enumerate(zip(pack.segments, pack.places, strict=True)):
        tokens = list(segment.tokens)
        if index % 2 == 0:
            for position, place in enumerate(places):
                if place not in shared and position > 0:
                    tokens[position] = row[place] = rng.randrange(VOCABULARY)
        segments.append(Segment(tokens, segment.spans, segment.logprobs))
    changed = replace(pack, segments=segments, tokens=row)
    with torch.no_grad():
        before, after = policy.packed(pack), policy.packed(changed)
    for index, (was, now) in enumerate(zip(before, after, strict=True)):
        if index % 2:
            torch.testing.assert_close(now.logprobs, was.logprobs, rtol=0, atol=1e-6)
        else:
            assert not torch.allclose(now.logprobs, was.logprobs, atol=1e-3)
            torch.testing.assert_close(now.logprobs, alone(policy, changed, index), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["qwen3_5", "llama"])
def test_a_recurrent_state_does_not_cross_into_the_next_segment(kind: str) -> None:
    """Two segments side by side in one row: the second's logprobs do not move when the end of the first does (its
    last tokens are what the short convolution and the recurrence would carry over)."""
    policy = policy_of(kind)
    rng = random.Random(3)
    first = [rng.randrange(VOCABULARY) for _ in range(50)]
    second = [rng.randrange(VOCABULARY) for _ in range(40)]
    following = Segment(second, [Span(1, 40, 0)], [0.0] * 39)
    found: list[torch.Tensor] = []
    with torch.no_grad():
        for ending in ([1, 2, 3, 4], [9, 8, 7, 6]):
            leading = Segment(first + ending, [Span(1, 54, 0)], [0.0] * 53)
            (pack,) = packs([leading, following], 200, share=False)
            assert [run.parent for run in pack.runs] == [None, None] and pack.members == [0, 1]
            found.append(policy.packed(pack)[1].logprobs)
        alone_logprobs = policy.logprobs(second, range(1, 40))
    torch.testing.assert_close(found[0], found[1], rtol=0, atol=1e-6)
    torch.testing.assert_close(found[0], alone_logprobs, rtol=1e-5, atol=1e-5)


def items_of(case: str, segments: Sequence[Segment]) -> list[Item]:
    """A batch of each objective family's items."""
    if case in ("dpo", "simpo"):
        return [Pair((segments[i],), (segments[i + 1],)) for i in range(0, len(segments) - 1, 2)]
    if case == "kto":
        return [Labelled((each,), index % 2 == 0) for index, each in enumerate(segments)]
    if case in ("on_policy_distillation", "distillation"):
        rng = random.Random(2)
        found: list[Item] = []
        for segment in segments:
            count = segment.sampled
            tops = [[rng.randrange(VOCABULARY) for _ in range(3)] for _ in range(count)]
            top_logprobs = [sorted((-rng.random() * 4 for _ in range(3)), reverse=True) for _ in range(count)]
            scores = TeacherScores("teacher", [-rng.random() * 3 for _ in range(count)], tops, top_logprobs)
            found.append(Distilled(segment, scores))
        return found
    return [Weighted(each, 1.0 if index % 3 else -0.7) for index, each in enumerate(segments)]


OBJECTIVES: dict[str, Any] = {
    "default": "default",  # (a token mean, an importance weight from where it was sampled)
    "grpo": "grpo",  # (a segment mean, a KL to the reference)
    "gspo": "gspo",  # (a segment's ratio)
    "entropy": {"preset": "default", "entropy.coefficient": 0.05},
    "sft": "sft",  # (a likelihood: no start)
    "dpo": "dpo",  # (pairs, against the reference)
    "simpo": "simpo",
    "kto": "kto",  # (labelled examples)
    "on_policy_distillation": "on_policy_distillation",
    "distillation": {"preset": "distillation", "distillation.top_k": 3},  # (the teacher's top-k tokens)
}


def recorded(stepping: PolicyStep) -> list[list[torch.Tensor]]:
    """Each update's gradients as the step leaves them for its optimizer (a plain gradient descent here, so that a
    gradient's tiny differences stay tiny in the weights)."""
    parameters = stepping.policy.parameters()
    stepping.optimizer = torch.optim.SGD(parameters, lr=0.05)
    found: list[list[torch.Tensor]] = []
    stepped = stepping.optimizer.step

    def step(*arguments: Any, **options: Any) -> Any:
        found.append([cast(torch.Tensor, each.grad).clone() for each in parameters])
        return stepped(*arguments, **options)

    stepping.optimizer.step = step  # type: ignore[method-assign]
    return found


def stepped(
    kind: str, case: str, *, packing: bool, settings: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, float]], list[list[torch.Tensor]]]:
    """A step of the case on a fresh tiny policy: its metrics, its minibatches, and each update's gradients."""
    policy = policy_of(kind)
    segments = sampled_at(policy, gridworld(9))
    policy.packing = packing
    stepping = PolicyStep(policy, StepSettings(objective=OBJECTIVES[case], **settings))
    gradients = recorded(stepping)
    metrics = stepping.step(items_of(case, segments), seed=3)
    return metrics, stepping.minibatches, gradients


COMPARED = ("loss", "tokens", "segments", "kl_moved", "kl_floor", "mean_ratio", "clip_fraction", "kl_penalty",
            "entropy", "gradient_norm", "optimizer_steps", "teacher_gap", "preference_margin")  # fmt: skip


def same(one: tuple[Any, ...], two: tuple[Any, ...]) -> None:
    metrics, lines, gradients = one
    other_metrics, other_lines, other_gradients = two
    for key in COMPARED:
        if key in metrics:
            assert other_metrics[key] == pytest.approx(metrics[key], rel=1e-4, abs=1e-6), key
    assert len(lines) == len(other_lines) and len(gradients) == len(other_gradients) >= 1
    for line, other in zip(lines, other_lines, strict=True):
        for key in line:
            assert other[key] == pytest.approx(line[key], rel=1e-4, abs=1e-6), key
    for update, other in zip(gradients, other_gradients, strict=True):
        for gradient, other_gradient in zip(update, other, strict=True):
            torch.testing.assert_close(other_gradient, gradient, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("case", list(OBJECTIVES))
def test_a_step_in_packs_is_the_step_one_segment_at_a_time(case: str) -> None:
    """Qwen3.5's hybrid, three minibatches over two passes: the same losses, minibatches and gradients."""
    settings = {"learning_rate": 0.05, "tokens_per_step": 40, "passes": 2, "max_kl": None, "pack_tokens": 220}
    packed = stepped("qwen3_5", case, packing=True, settings=settings)
    one_at_a_time = stepped("qwen3_5", case, packing=False, settings=settings)
    same(one_at_a_time, packed)
    assert packed[0]["packed"] == 1.0 and one_at_a_time[0]["packed"] == 0.0
    assert packed[0]["packs"] < one_at_a_time[0]["packs"] and packed[0]["prefix_shared_fraction"] > 0.2
    assert 0 < packed[0]["pack_fill"] <= 1.0 and packed[0]["segment_tokens_per_second"] > 0


@pytest.mark.parametrize("kind", ["qwen3", "llama"])
@pytest.mark.parametrize("case", ["default", "dpo"])
def test_a_step_in_packs_is_the_step_one_segment_at_a_time_on_softmax_attention(kind: str, case: str) -> None:
    settings = {"learning_rate": 0.05, "tokens_per_step": 40, "max_kl": None, "pack_tokens": 220}
    same(stepped(kind, case, packing=False, settings=settings), stepped(kind, case, packing=True, settings=settings))


@pytest.mark.parametrize("case", ["default", "grpo", "distillation"])
def test_the_first_minibatchs_start_folded_into_it_is_the_start_computed_apart(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = {"learning_rate": 0.05, "tokens_per_step": 40, "max_kl": 0.5, "pack_tokens": 220}
    folded = stepped("qwen3_5", case, packing=True, settings=settings)
    monkeypatch.setattr(step_module, "_first_minibatch", nothing_folded)  # (every start computed first)
    apart = stepped("qwen3_5", case, packing=True, settings=settings)
    same(apart, folded)
    assert folded[0]["packs"] < apart[0]["packs"]  # (the first minibatch's segments computed once)
    assert folded[1][0]["kl"] == 0.0  # (on the weights the step starts from: exactly where it started)


@pytest.fixture(scope="module")
def hybrid(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tiny random Qwen3.5 (its text model), saved as a model's directory."""
    directory = tmp_path_factory.mktemp("hybrid")
    cast(Any, model_of("qwen3_5")).save_pretrained(directory)
    return str(directory)


@pytest.mark.parametrize("objective", ["default", "grpo"])
def test_two_processes_step_in_packs_as_one_does(hybrid: str, tmp_path: Path, objective: str) -> None:
    """Shared prefixes grouped before the groups are shared out, each process's packs, and idle packs where one has
    fewer: the losses and the adapter one process makes, packed or one segment at a time."""
    from safetensors.torch import load_file

    from rollout_lora.resident import Workers
    from rollout_lora.workers import SEED

    settings = LoraSettings(rank=4, learning_rate=1e-3, tokens_per_step=60, max_kl=None, objective=objective,
                            pack_tokens=180)  # fmt: skip
    torch.manual_seed(SEED)  # (the new adapter the processes draw)
    policy = Policy.load(hybrid, rank=settings.rank, alpha=settings.alpha, device="cpu")
    assert policy.packing
    torch.manual_seed(1)
    given: list[Item] = [Weighted(each, 1.0 if index % 3 else -0.7)
                         for index, each in enumerate(sampled_at(policy, gridworld(11)))]  # fmt: skip
    workers = Workers(hybrid, settings, "lora", 2, device="cpu")
    try:
        two = asyncio.run(workers.step(given, seed=0, parent=None, into=tmp_path / "two"))
    finally:
        workers.close()
    weights = [each.detach().clone() for each in lora_parameters(policy.model)]
    one = PolicyStep(policy, settings).step(given, seed=0)
    policy.save(tmp_path / "one")
    compared = ("loss", "tokens", "segments", "kl_moved", "mean_ratio", "gradient_norm", "optimizer_steps",
                "prefix_shared_fraction")  # fmt: skip
    for key in compared:
        assert two[key] == pytest.approx(one[key], rel=1e-4, abs=1e-6), key
    assert one["optimizer_steps"] > 1 and one["prefix_shared_fraction"] > 0.2 and two["packs"] >= one["packs"]
    saved, ours = (
        load_file(str(tmp_path / "two" / "weights" / "adapter_model.safetensors")),
        load_file(str(tmp_path / "one" / "adapter_model.safetensors")),
    )
    for key in saved:
        torch.testing.assert_close(saved[key], ours[key], rtol=1e-4, atol=1e-6)
    for parameter, weight in zip(lora_parameters(policy.model), weights, strict=True):  # (one segment at a time)
        parameter.data.copy_(weight)
    policy.packing = False
    unpacked = PolicyStep(policy, settings).step(given, seed=0)
    for key in ("loss", "kl_moved", "gradient_norm"):  # (in bfloat16: packed and alone round differently)
        assert unpacked[key] == pytest.approx(one[key], rel=2e-2, abs=1e-4), key
