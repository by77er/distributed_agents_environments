# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Packs on the GPU, where Qwen3.5's gated delta rule runs flash-linear-attention's kernels: the runs' boundaries
(`cu_seqlens`), each branch started from the state its prefix ended in, and the gradient back through that state. A
tiny random Qwen3.5 in float32 (bfloat16's rounding would hide a difference): a pack gives each segment its logprobs and
gradients alone, and a step in packs is the step one segment at a time. Run only when asked (`-m live`)."""

from typing import Any, cast

import pytest
import torch

from tests.rollout_lora.test_packing import at_the_start, gridworld, policy_of, sampled_at

pytestmark = [pytest.mark.live, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def policy_on_gpu(dtype: torch.dtype = torch.float32) -> Any:
    """Its attention heads 16 wide, the narrowest FlexAttention takes (the branches' attention, on the GPU)."""
    return policy_of("qwen3_5", head=16, dtype=dtype, device="cuda")


def test_a_pack_on_the_gpu_gives_each_segment_its_logprobs_and_gradients_alone() -> None:
    from rollout_objectives.packing import packs, sampled_positions

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
        alone = policy.logprobs(segment.tokens, sampled_positions(segment))
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


@pytest.mark.parametrize("case", ["default", "dpo"])
def test_on_unchanged_weights_every_minibatchs_ratios_are_exactly_1_in_bfloat16_on_the_gpu(case: str) -> None:
    at_the_start(policy_on_gpu(torch.bfloat16), case)
