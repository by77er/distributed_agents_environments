# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""A small text model's step sharded with FSDP2 on one GPU, against the step on one GPU as it is otherwise taken (a
fresh process): run only when asked (`-m live`), with nothing else on the card. `ROLLOUT_SMALL_MODEL` names the model
(Qwen3-0.6B by default). The sharded step computes in bfloat16 throughout (the adapter's layers too) and reduces in
float32; the other computes the adapter in float32, so the two agree to bfloat16's precision. Its steps write
gigabytes, so give it `--basetemp` on disk, not `/tmp`."""

import asyncio
import os
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from rollout_lora.resident import Workers
from rollout_lora.settings import LoraSettings
from rollout_lora.trainer import LoraTrainer
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Files, Item

MODEL = os.environ.get("ROLLOUT_SMALL_MODEL", "Qwen/Qwen3-0.6B")
SETTINGS: dict[str, Any] = {"rank": 8, "learning_rate": 1e-4, "tokens_per_step": 400, "max_kl": None}

pytestmark = [pytest.mark.live, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def batch() -> list[Item]:
    """Random sequences whose last 200 tokens were sampled, at the logprobs the model gives them (no adapter)."""
    from rollout_lora.policy import Policy

    policy = Policy.load(MODEL, rank=8, alpha=16.0)
    rng = random.Random(0)
    found: list[Item] = []
    for index in range(6):
        length = 400 + 100 * index
        tokens = [rng.randrange(1_000, 50_000) for _ in range(length)]
        with torch.no_grad():
            exact = policy.logprobs(tokens, range(length - 200, length)).float().cpu()
        found.append(
            Weighted(Segment(tokens, [Span(length - 200, length, 0)], exact.tolist()), 1.0 if index % 2 else -1.0)
        )
    del policy
    torch.cuda.empty_cache()
    return found


def test_a_sharded_step_on_one_gpu_is_the_step_a_fresh_process_takes(tmp_path: Path) -> None:
    given = batch()
    alone = LoraTrainer(MODEL, gpus=1, **SETTINGS)
    asyncio.run(alone.step(given, seed=0, parent=None, into=tmp_path / "first"))
    parent = Files(tmp_path / "first" / "weights", tmp_path / "first" / "state")
    today = asyncio.run(alone.step(given, seed=1, parent=parent, into=tmp_path / "today")).metrics

    workers = Workers(MODEL, LoraSettings(**SETTINGS), "lora", 1)
    try:
        sharded = asyncio.run(workers.step(given, seed=1, parent=parent, into=tmp_path / "sharded"))
        made = Files(tmp_path / "sharded" / "weights", tmp_path / "sharded" / "state")
        held = asyncio.run(workers.step(given, seed=2, parent=made, into=tmp_path / "held"))
    finally:
        workers.close()
    print({key: (round(today[key], 5), round(sharded[key], 5)) for key in sorted(today) if key in sharded})
    assert sharded["gpus"] == 1.0 and sharded["loaded_from_files"] == 1.0 and held["loaded_from_files"] == 0.0
    for key in ("loss", "kl_moved", "mean_ratio", "gradient_norm", "kl_floor", "mean_mismatch"):
        assert sharded[key] == pytest.approx(today[key], rel=0.05, abs=2e-3), key
    assert sharded["tokens"] == today["tokens"] and sharded["optimizer_steps"] == today["optimizer_steps"]
    made_by = (tmp_path / "sharded" / "weights", tmp_path / "today" / "weights", parent.weights)
    ours, theirs, began = (load_file(str(each / "adapter_model.safetensors")) for each in made_by)
    moved = sum(float((ours[key] - began[key]).abs().sum()) for key in ours)
    apart = sum(float((ours[key] - theirs[key]).abs().sum()) for key in ours)
    print(f"the step moved the adapter by {moved:.4f} in all; the two steps' adapters are {apart:.4f} apart")
    assert apart < 0.1 * moved
