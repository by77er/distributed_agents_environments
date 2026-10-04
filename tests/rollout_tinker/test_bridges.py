# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tinker's bridge as a Ray task: a checkpoint trained on Tinker (its weights a pointer) becomes an adapter in PEFT's
layout, from the sampler checkpoint's archive, noted under `CHECKPOINT@peft-from-tinker`."""

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from rollout.harness.blobs import FileBlobStore
from rollout_train.bridges import BRIDGED, NoBridge, made, on_ray, path, rank_factor
from rollout_train.checkpoints import Checkpoints
from rollout_train.ledger import FileLedger
from rollout_train.record import scope
from rollout_train.stores import FILES
from tests.local_ray import LocalRay
from tests.rollout_tinker.support import RANK, SAMPLER, base_model


async def test_a_tinker_checkpoint_is_bridged_to_peft_on_ray(tmp_path: Path, local_ray: LocalRay) -> None:
    base = base_model(tmp_path / "base")
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    fence = await checkpoints.ledger.take(scope("run"))
    weights = tmp_path / "made" / "weights"
    weights.mkdir(parents=True)
    pointer = {"sampler": SAMPLER, "state": "tinker://fake-run/weights/kpqx", "base_model": str(base), "rank": RANK}
    (weights / "tinker.json").write_text(json.dumps(pointer))
    record = await checkpoints.add(fence, "kpqxrmtzwvolxqvu", weights=weights, run="run", step=1, base=str(base))

    chain = path("tinker", {"peft", "full"})  # (what local vLLM loads)
    assert not isinstance(chain, NoBridge) and [each.name for each in chain] == ["peft-from-tinker"]
    assert rank_factor(chain, "Qwen/Qwen3.5-4B") == 3
    ledger_at, blobs_at = {"directory": str(tmp_path / "ledger")}, {"kind": FILES, "directory": str(tmp_path / "blobs")}
    settings = {"peft-from-tinker": {"service": "tests.rollout_tinker.support:archives"}}
    manifest = await on_ray(
        ledger_at, blobs_at, fence, record.id, chain, settings=settings, scratch=str(tmp_path / "worker")
    )

    assert sorted(manifest.files) == ["adapter_config.json", "adapter_model.safetensors"]
    assert await made(checkpoints.ledger, record.id, "peft-from-tinker") == manifest
    noted: Any = (await checkpoints.ledger.read(BRIDGED))[f"{record.id}@peft-from-tinker"]
    assert noted["said"] == {"sampler": SAMPLER, "model": str(base), "largest_rank": 3 * RANK}
    files = await checkpoints.files(manifest, tmp_path / "served")
    adapter = load_file(str(files / "adapter_model.safetensors"))
    assert any(".in_proj_qkv.lora_A." in key for key in adapter)  # (q, k and v joined, as vLLM adapts them)
    assert not any(".in_proj_q." in key for key in adapter)
    configuration = json.loads((files / "adapter_config.json").read_text())
    assert configuration["base_model_name_or_path"] == str(base) and configuration["rank_pattern"]["in_proj_qkv"] == 12
