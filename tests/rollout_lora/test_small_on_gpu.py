# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""A small text model on the GPU: an adapter, every weight, and an adapter folded in. Run only when asked
(`-m live`), with nothing else on the card. `ROLLOUT_SMALL_MODEL` names it (Qwen3-0.6B by default)."""

import asyncio
import json
import os
import random
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rollout_lora.settings import LoraSettings
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Files

MODEL = os.environ.get("ROLLOUT_SMALL_MODEL", "Qwen/Qwen3-0.6B")

pytestmark = [pytest.mark.live, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def segments(score: object, count: int = 4, length: int = 600, sampled: int = 200) -> list[Weighted]:
    """Random sequences whose last `sampled` tokens were sampled at the logprobs `score` gives them now."""
    rng = random.Random(0)
    found: list[Weighted] = []
    for index in range(count):
        tokens = [rng.randrange(1_000, 50_000) for _ in range(length)]
        with torch.no_grad():
            exact = score(tokens, range(length - sampled, length)).float().cpu()  # type: ignore[operator]
        found.append(
            Weighted(Segment(tokens, [Span(length - sampled, length, 0)], exact.tolist()), 1.0 if index % 2 else -1.0)
        )
    return found


def test_an_adapter_trains_over_a_text_model() -> None:
    from rollout_lora.policy import Policy
    from rollout_objectives.step import PolicyStep

    settings = LoraSettings(rank=8, tokens_per_step=400, max_kl=None)
    policy = Policy.load(MODEL, rank=settings.rank, alpha=settings.alpha)
    assert policy.embedding is None  # (its embeddings are tied to the output layer: they stay on the GPU)
    metrics = PolicyStep(policy, settings).step(segments(policy.logprobs))
    assert metrics["segments"] == 4 and metrics["minibatches_out_of_memory"] == 0


def test_every_weight_trains_sharded_on_one_gpu_and_leaves_a_serving_copy_and_its_full_state(tmp_path: Path) -> None:
    from rollout_lora.full import FullPolicy
    from rollout_lora.trainer import FullTrainer

    reference = FullPolicy.load(MODEL)
    batch = segments(reference.logprobs)
    before = {name: parameter.detach().clone() for name, parameter in reference.model.named_parameters()}
    del reference
    torch.cuda.empty_cache()
    trainer = FullTrainer(MODEL, learning_rate=1e-5, tokens_per_step=400, max_kl=None, warmup_updates=1)
    assert trainer.weights == "full"
    try:
        step = asyncio.run(trainer.step(batch, seed=0, parent=None, into=tmp_path / "first"))
        assert step.metrics["segments"] == 4 and step.metrics["peak_gpu_gib"] < 14
        weights = tmp_path / "first" / "weights"
        said = json.loads((weights / "config.json").read_text())
        assert said.get("torch_dtype", said.get("dtype")) == "bfloat16" and (weights / "tokenizer.json").exists()
        moved = 0
        for file in weights.glob("*.safetensors"):
            with safe_open(str(file), "pt") as opened:
                for key in opened.keys():  # noqa: SIM118 (a safetensors file is not iterable)
                    saved = opened.get_tensor(key)
                    assert saved.dtype == torch.bfloat16  # (the serving copy)
                    if key in before and not torch.equal(saved.cuda(), before[key].to(torch.bfloat16)):
                        moved += 1
        assert moved > 100  # (every weight is trained: the layers' weights moved)
        assert (tmp_path / "first" / "state" / "shards").is_dir()  # (the float32 weights and the optimizer's state)
        # The next step goes on from those weights and the optimizer's state: from memory, its process kept.
        parent = Files(weights, tmp_path / "first" / "state")
        second = asyncio.run(trainer.step(batch, seed=1, parent=parent, into=tmp_path / "second"))
        assert second.metrics["segments"] == 4 and second.metrics["loaded_from_files"] == 0.0
        assert second.metrics["warmup_updates"] == 0.0  # (its optimizer went on)
        # A step can go on from the weights alone, its optimizer started afresh (as a supervised step does by default).
        fresh = asyncio.run(trainer.step(batch, seed=2, parent=Files(weights, None), into=tmp_path / "fresh"))
    finally:
        trainer.close()
    assert fresh.metrics["segments"] == 4 and fresh.metrics["warmup_updates"] == 1.0  # (its optimizer afresh)
    assert (tmp_path / "fresh" / "state" / "shards").is_dir()


def test_an_adapter_goes_on_from_its_weights_with_its_optimizer_afresh(tmp_path: Path) -> None:
    from rollout_lora.policy import Policy
    from rollout_lora.trainer import LoraTrainer
    from rollout_train.trainer import Files

    reference = Policy.load(MODEL, rank=8, alpha=16.0)
    batch = segments(reference.logprobs)
    del reference
    torch.cuda.empty_cache()
    trainer = LoraTrainer(MODEL, rank=8, learning_rate=1e-4, tokens_per_step=400, max_kl=None, objective="likelihood")
    try:
        asyncio.run(trainer.step(batch, seed=0, parent=None, into=tmp_path / "first"))
        adapter = tmp_path / "first" / "weights"
        fresh = asyncio.run(trainer.step(batch, seed=1, parent=Files(adapter, None), into=tmp_path / "fresh"))
    finally:
        trainer.close()
    assert fresh.metrics["segments"] == 4 and (tmp_path / "fresh" / "weights" / "adapter_model.safetensors").exists()


def test_an_adapter_folded_in_gives_what_the_adapter_gave(tmp_path: Path) -> None:
    from rollout_lora.full import FullPolicy
    from rollout_lora.merge import merge
    from rollout_lora.policy import Policy
    from rollout_objectives.step import PolicyStep

    settings = LoraSettings(rank=8, tokens_per_step=400, max_kl=None, learning_rate=1e-3)
    policy = Policy.load(MODEL, rank=settings.rank, alpha=settings.alpha)
    batch = segments(policy.logprobs)
    PolicyStep(policy, settings).step(batch)  # (an adapter that moved: B is no longer zero)
    policy.save(tmp_path / "adapter")
    probe = batch[0].segment.tokens
    policy.model.eval()
    with torch.no_grad():
        adapted = policy.logprobs(probe, range(400, 600)).float().cpu()
    del policy
    torch.cuda.empty_cache()
    said = merge(MODEL, tmp_path / "adapter", tmp_path / "merged")
    assert said["layers"] == 28 * 7  # (every attention and MLP projection of its 28 layers)
    merged = FullPolicy.load(str(tmp_path / "merged"), gradient_checkpointing=False)
    merged.model.eval()
    with torch.no_grad():
        folded = merged.logprobs(probe, range(400, 600)).float().cpu()
    base = torch.tensor(batch[0].segment.logprobs)  # (what the model gave before the adapter)
    moved, kept = float((adapted - base).abs().mean()), float((adapted - folded).abs().mean())
    print(f"\nthe adapter moved logprobs by {moved:.3f} on average; folded in, they are {kept:.3f} from it")
    # In bfloat16 the merged weight rounds away part of a small update: closer to the adapter than to the base, by far.
    assert kept < 0.3 * moved
