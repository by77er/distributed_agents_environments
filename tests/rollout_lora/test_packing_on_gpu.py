# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Packs on the GPU, where Qwen3.5's gated delta rule runs flash-linear-attention's kernels: the runs' boundaries
(`cu_seqlens`), each branch started from the state its prefix ended in, and the gradient back through that state. A
tiny random Qwen3.5 in float32 (bfloat16's rounding would hide a difference): a pack gives each segment its logprobs and
gradients alone, and a step in packs is the step one segment at a time. Run only when asked (`-m live`)."""

from typing import Any, cast

import pytest
import torch

from tests.rollout_lora.test_packing import gridworld, model_of, sampled_at

pytestmark = [pytest.mark.live, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def policy_on_gpu() -> Any:
    from rollout_lora.layers import add_lora
    from rollout_lora.packing import prepare
    from rollout_lora.policy import TARGETS, Policy

    model = model_of("qwen3_5").to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    add_lora(model, TARGETS, rank=4, alpha=8.0, within="layers", dtype=torch.float32)
    torch.manual_seed(1)
    for name, parameter in model.named_parameters():
        if ".lora_B." in name:
            parameter.data.normal_(0, 0.05)
    cast(Any, model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cast(Any, model).enable_input_require_grads()
    return Policy(model, "tiny", 4, 8.0, None, prepare(model))


def test_a_pack_on_the_gpu_gives_each_segment_its_logprobs_and_gradients_alone() -> None:
    from rollout_objectives.packing import packs
    from rollout_objectives.step import positions

    policy = policy_on_gpu()
    segments = gridworld(9)
    (pack,) = packs(segments, 10_000)
    assert any(run.parent is not None for run in pack.runs)
    parameters = policy.parameters()
    found = policy.packed(pack)
    torch.stack([each.logprobs.sum() for each in found]).sum().backward()
    packed = [cast(torch.Tensor, each.grad).clone() for each in parameters]
    for each in parameters:
        each.grad = None
    total = torch.zeros((), device="cuda")
    for index, segment in enumerate(pack.segments):
        alone = policy.logprobs(segment.tokens, positions(segment))
        torch.testing.assert_close(found[index].logprobs.detach(), alone.detach(), rtol=1e-4, atol=1e-4)
        total = total + alone.sum()
    total.backward()
    for gradient, each in zip(packed, parameters, strict=True):
        torch.testing.assert_close(gradient, cast(torch.Tensor, each.grad), rtol=1e-3, atol=1e-3)


def test_a_step_in_packs_on_the_gpu_is_the_step_one_segment_at_a_time() -> None:
    from rollout_objectives.settings import StepSettings
    from rollout_objectives.step import PolicyStep
    from rollout_train.trainer import Weighted

    found: list[dict[str, float]] = []
    for packing in (True, False):
        policy = policy_on_gpu()
        batch = [Weighted(each, 1.0 if index % 3 else -0.7)
                 for index, each in enumerate(sampled_at(policy, gridworld(11)))]  # fmt: skip
        policy.packing = packing
        settings = StepSettings(objective="grpo", tokens_per_step=40, max_kl=None, pack_tokens=220)
        found.append(PolicyStep(policy, settings).step(batch, seed=0))
    packed, alone = found
    for key in ("loss", "kl_moved", "kl_floor", "kl_penalty", "gradient_norm", "optimizer_steps"):
        assert packed[key] == pytest.approx(alone[key], rel=1e-3, abs=1e-5), key
    assert packed["packs"] < alone["packs"] and packed["prefix_shared_fraction"] > 0.2
