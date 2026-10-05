# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""`VllmEngine` scoring on the GPU: the logprobs it gives a fixed sequence, and the most likely tokens at each position,
against a forward pass of the same checkpoint in transformers (bfloat16, on the GPU), for the small model a student
would be and a larger one a teacher would be. The sequence is longer than the 1,024 positions vLLM computes prompt
logprobs in at a time. A sample's own logprobs and most likely tokens agree with scoring what it sampled. Run only when
asked (`-m live`: the models are downloaded if they are not cached), holding the machine's GPU lock
(`~/.cache/rollout/gpu.lock`). `ROLLOUT_SMALL_MODEL` and `ROLLOUT_TEACHER_MODEL` name the models."""

import asyncio
import contextlib
import fcntl
import gc
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.live, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

ROOT = Path(__file__).resolve().parents[2]
STUDENT = os.environ.get("ROLLOUT_SMALL_MODEL", "Qwen/Qwen3-0.6B")
TEACHER = os.environ.get("ROLLOUT_TEACHER_MODEL", "Qwen/Qwen3-1.7B")
TOP = 20
LENGTH = 1_500
"""Tokens of the sequence scored."""


@contextlib.contextmanager
def gpu() -> Generator[None]:
    """The machine's GPU, held while the block runs."""
    path = Path.home() / ".cache" / "rollout" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def sequence(model: str) -> list[int]:
    """A fixed sequence of natural text: the start of the gateway's page of the docs, `LENGTH` tokens of it."""
    from rollout_qwen import tokenizer_of

    text = (ROOT / "docs" / "libraries" / "rollout-train" / "gateway.md").read_text()
    tokens = tokenizer_of(model).encode(text, add_special_tokens=False)
    assert len(tokens) > LENGTH
    return tokens[:LENGTH]


def reference(model: str, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """What transformers gives each token from the second on, in bfloat16 on the GPU: its logprob, and the `TOP` most
    likely tokens there."""
    from transformers import AutoModelForCausalLM

    loaded: Any = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    loaded = loaded.cuda().eval()
    ids = torch.tensor([tokens], device="cuda")
    with torch.no_grad():
        logprobs = loaded(ids).logits[0, :-1].float().log_softmax(-1)
    own = logprobs.gather(-1, ids[0, 1:, None])[:, 0].cpu()
    top = logprobs.topk(TOP, dim=-1).indices.cpu()
    del loaded, logprobs
    gc.collect()
    torch.cuda.empty_cache()
    return own, top


async def scored(model: str, tokens: list[int]) -> dict[str, float]:
    """`VllmEngine`'s scores of `tokens`, compared with transformers', and a sample's logprobs compared with scoring
    what it sampled."""
    from rollout_vllm import VllmEngine

    own, top = reference(model, tokens)
    engine = VllmEngine(model, gpu_memory_utilization=0.5, max_model_len=2048, max_num_seqs=4, max_logprobs=TOP)
    try:
        scores = await engine.score(tokens, start=1, top=TOP, adapter=None)
        assert (scores.start, scores.end) == (1, len(tokens))
        assert all(len(each) == TOP for each in scores.top_tokens)
        assert all(each == sorted(each, reverse=True) for each in scores.top_logprobs)
        difference = (torch.tensor(scores.logprobs) - own).abs()
        overlap = torch.tensor(
            [len(set(mine) & set(theirs)) / TOP for mine, theirs in zip(scores.top_tokens, top.tolist(), strict=True)]
        )
        later = await engine.score(tokens, start=1_200, end=1_300, top=5, adapter=None)  # (a range within it)
        within = (torch.tensor(later.logprobs) - torch.tensor(scores.logprobs[1_199:1_299])).abs().mean().item()

        prompt = tokens[:200]
        sample = await engine.generate(
            prompt, max_tokens=64, temperature=1.0, top_p=1.0, stop_token_ids=[], adapter=None, top=5
        )
        assert len(sample.top_tokens) == len(sample.tokens) and all(len(each) == 5 for each in sample.top_tokens)
        for token, logprob, ids, values in zip(
            sample.tokens, sample.logprobs, sample.top_tokens, sample.top_logprobs, strict=True
        ):
            assert values == sorted(values, reverse=True)
            if token in ids:  # (sampled from the distribution the most likely tokens are of)
                assert abs(values[ids.index(token)] - logprob) < 1e-6
        rescored = await engine.score([*prompt, *sample.tokens], start=len(prompt), adapter=None)
        sampled = (torch.tensor(rescored.logprobs) - torch.tensor(sample.logprobs)).abs().mean().item()
    finally:
        engine.close()
    return {
        "mean_absolute_difference": difference.mean().item(),
        "largest_difference": difference.max().item(),
        "mean_top_overlap": overlap.mean().item(),
        "least_top_overlap": overlap.min().item(),
        "positions_all_top_shared": (overlap == 1.0).float().mean().item(),
        "range_against_whole": within,
        "sample_against_scoring": sampled,
    }


@pytest.mark.parametrize("model", [STUDENT, TEACHER])
def test_scores_agree_with_a_transformers_forward_pass(model: str) -> None:
    tokens = sequence(model)
    with gpu():
        found = asyncio.run(scored(model, tokens))
    print(f"\n{model}: {found}")
    assert found["mean_absolute_difference"] < 0.05  # (bfloat16 kernels of their own: close, not equal)
    assert found["mean_top_overlap"] > 0.9
    assert found["range_against_whole"] < 0.05  # (scored again in a request of another shape: bfloat16 again)
    assert found["sample_against_scoring"] < 0.05  # (at temperature 1, what was sampled from is the model's own)
