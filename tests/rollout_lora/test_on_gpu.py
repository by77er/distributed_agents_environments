# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The policy step on the real checkpoint and the GPU: run only when asked (`-m live`), with nothing else on
the card. `ROLLOUT_GPU_MODEL` names the checkpoint (the one-GPU profile's by default)."""

import os
import random
import time

import pytest
import torch

from rollout_lora.settings import LoraSettings
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span

MODEL = os.environ.get("ROLLOUT_GPU_MODEL", "cyankiwi/Qwen3.5-9B-AWQ-4bit")

pytestmark = [pytest.mark.live, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def test_a_step_starts_where_its_policy_is_and_weighs_where_the_tokens_were_sampled() -> None:
    from rollout_lora.policy import Policy
    from rollout_lora.step import PolicyStep, sampled

    settings = LoraSettings(tokens_per_step=2_000, max_kl=None)  # (an update moves random tokens' logprobs a lot)
    policy = Policy.load(MODEL, rank=settings.rank, alpha=settings.alpha)
    rng = random.Random(0)
    segments: list[Weighted] = []
    for index in range(6):  # 3,000 tokens each, the last 1,000 sampled, at logprobs a little off the trainer's
        tokens = [rng.randrange(1_000, 50_000) for _ in range(3_000)]
        spans = [Span(2_000, 3_000, 0)]
        with torch.no_grad():
            exact = policy.logprobs(tokens, range(2_000, 3_000)).float().cpu()
        behavior = (exact + 0.05 * torch.randn(exact.shape)).tolist()
        segments.append(Weighted(Segment(tokens, spans, behavior), 1.0 if index % 2 else -1.0))

    # Without a gradient and with one, the trainer gives a token the same logprob: the ratio starts at 1. (In
    # training mode, as a step is: gradient checkpointing is on only then.)
    policy.model.train()
    first = segments[0]
    with torch.no_grad():
        quiet = policy.logprobs(first.segment.tokens, sampled(first))
    loud = policy.logprobs(first.segment.tokens, sampled(first))
    assert float((quiet - loud.detach()).abs().max()) < 1e-3

    began = time.monotonic()
    stepping = PolicyStep(policy, settings)
    metrics = stepping.step(segments)
    took, start = time.monotonic() - began, metrics["start_seconds"]
    print(f"\nstep of {len(segments)} segments: {took:.1f} s, of which the start {start:.1f} s")
    print({key: round(value, 4) for key, value in metrics.items()})
    assert (
        metrics["segments"] == 6 and metrics["start_out_of_memory"] == 0 and metrics["minibatches_out_of_memory"] == 0
    )
    assert metrics["mean_mismatch"] == pytest.approx(0.04, rel=0.25)  # the noise: E|N(0, 0.05)| is 0.04
    assert stepping.minibatches[0]["clip_fraction"] == 0.0  # before any update nothing is clipped, whatever the noise
