# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""A small text model's adapter on one GPU: a step of the trainer that keeps its process between steps (a GPU to
itself) against the step of the trainer whose process ends after each (beside an engine), from the same adapter and
optimizer's state, and the resident trainer's next step, gone on from memory, against the same step taken from its
files. Run only when asked (`-m live`), with nothing else on the card. `ROLLOUT_SMALL_MODEL` names the model
(Qwen3-0.6B by default). The processes compute with PyTorch's deterministic algorithms, and the steps are compared
against the same step taken twice in fresh processes (the noise floor: none, on Qwen3-0.6B, where all four come out
bitwise alike). Its steps write gigabytes, so give it `--basetemp` on disk, not `/tmp`."""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from rollout_lora.trainer import LoraTrainer
from rollout_lora.workers import MASTER
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Files, Item

MODEL = os.environ.get("ROLLOUT_SMALL_MODEL", "Qwen/Qwen3-0.6B")
SETTINGS: dict[str, Any] = {"rank": 8, "learning_rate": 1e-4, "tokens_per_step": 400, "max_kl": None}
DETERMINISTIC = """import torch

torch.use_deterministic_algorithms(True)
"""
"""What every process the trainers start runs first (a `sitecustomize` on their path)."""

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


def files(directory: Path) -> Files:
    return Files(directory / "weights", directory / "state")


def adapter(directory: Path) -> dict[str, torch.Tensor]:
    """A step's adapter in float32, as a step from it loads it (engines load its bfloat16 copy)."""
    return load_file(str(directory / "state" / MASTER))


def apart(one: dict[str, torch.Tensor], two: dict[str, torch.Tensor]) -> float:
    return sum(float((one[key] - two[key]).abs().sum()) for key in one)


def test_a_resident_step_on_one_gpu_is_the_step_a_fresh_process_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "path").mkdir()
    (tmp_path / "path" / "sitecustomize.py").write_text(DETERMINISTIC)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path / "path"), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    given = batch()
    took: dict[str, float] = {}

    def step(trainer: LoraTrainer, name: str, seed: int, parent: Files | None) -> dict[str, float]:
        started = time.monotonic()
        made = asyncio.run(trainer.step(given, seed=seed, parent=parent, into=tmp_path / name)).metrics
        took[name] = time.monotonic() - started
        return dict(made)

    fresh = LoraTrainer(MODEL, gpus=1, colocated=True, **SETTINGS)  # (beside an engine: a process for each step)
    step(fresh, "first", 0, None)
    today = step(fresh, "today", 1, files(tmp_path / "first"))
    step(fresh, "again", 1, files(tmp_path / "first"))
    resident = LoraTrainer(MODEL, gpus=1, **SETTINGS)  # (a GPU to itself: its process kept)
    try:
        kept = step(resident, "kept", 1, files(tmp_path / "first"))
        held = step(resident, "held", 2, files(tmp_path / "kept"))
        assert resident.holding is not None
    finally:
        resident.close()
    from_files = step(fresh, "from_files", 2, files(tmp_path / "kept"))

    began = adapter(tmp_path / "first")
    moved = apart(adapter(tmp_path / "today"), began)
    floor = apart(adapter(tmp_path / "today"), adapter(tmp_path / "again"))
    resident_apart = apart(adapter(tmp_path / "kept"), adapter(tmp_path / "today"))
    memory_apart = apart(adapter(tmp_path / "held"), adapter(tmp_path / "from_files"))
    print(f"\nthe step moved the adapter by {moved:.5f}; the same step twice in fresh processes: {floor:.3e} apart")
    print(
        f"resident against fresh: {resident_apart:.3e}; gone on from memory against from its files: {memory_apart:.3e}"
    )
    print({key: (round(today[key], 6), round(kept[key], 6)) for key in ("loss", "gradient_norm", "kl_moved")})
    print({name: round(seconds, 1) for name, seconds in took.items()})
    assert kept["loaded_from_files"] == 1.0 and held["loaded_from_files"] == 0.0 and kept["gpus"] == 1.0
    assert from_files["loaded_from_files"] == 1.0 and held["full_state"] == 1.0
    allowed = floor + 1e-6 * moved  # (within what the same step taken twice differs by)
    assert resident_apart <= allowed and memory_apart <= allowed
    assert kept["loss"] == pytest.approx(today["loss"], rel=1e-4, abs=1e-6)
