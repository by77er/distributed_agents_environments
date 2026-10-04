# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (safetensors' loaders and the cookbook's converter are partly untyped.)
"""A checkpoint's files when its weights are at Thinking Machines: pointers, and the adapter itself if asked.

`weights/tinker.json` names the sampler checkpoint to sample from and the training state a later step starts from;
`state/tinker.json` names the state again (a step given the state goes on with its optimizer, one given the weights
alone starts its optimizer afresh). The pointers are files like any other: the blob store keeps them, and a checkpoint
copied to another machine points at the same remote checkpoints.

With `weights = "peft"` the sampler checkpoint is downloaded too (its archive holds the adapter with Tinker's own
names), turned into PEFT's layout by `tinker_cookbook.weights.build_lora_adapter`, and kept beside the pointer, so that
vLLM serves it, `rollout merge` folds it in, and the blob store holds it. Qwen3.5's linear-attention layers hold one
`in_proj_qkv` where Tinker adapts `in_proj_q`, `in_proj_k` and `in_proj_v` apart, and vLLM adapts only the joined name:
the three are joined into one adapter (`fused`), which is the same update.

    python -m rollout_tinker.weights ARCHIVE_DIRECTORY OUTPUT_DIRECTORY BASE_MODEL   # what `converted` runs
"""

import asyncio
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

POINTER = "tinker.json"


def pointer(directory: Path, key: str) -> str | None:
    """What a checkpoint's pointer file in `directory` says of `key` (`sampler`, `state`), if it has one."""
    path = directory / POINTER
    if not path.is_file():
        return None
    found = json.loads(path.read_text()).get(key)
    return str(found) if found else None


def write_pointer(directory: Path, said: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / POINTER).write_text(json.dumps(said, indent=2) + "\n")


def checkpoint_name(name: str) -> str:
    """A Tinker checkpoint's name for a version's directory name (its id): only letters, digits, `-` and `_`."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


async def downloaded(url: str, into: Path) -> Path:
    """The archive at `url` (a signed URL, or a file's), extracted into `into`."""

    def fetch() -> Path:
        into.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar", dir=into.parent, delete=False) as file:
            archive = Path(file.name)
        try:
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as opened:
                opened.extractall(into, filter="data")  # (no links, nothing outside `into`)
        finally:
            archive.unlink(missing_ok=True)
        found = [path.parent for path in into.rglob("adapter_model.safetensors")]
        if len(found) != 1:
            raise ValueError(f"the archive holds {len(found)} adapters, not one")
        return found[0]

    return await asyncio.to_thread(fetch)


async def converted(archive: Path, into: Path, base_model: str) -> None:
    """The adapter in `archive` (Tinker's names) in PEFT's layout in `into`, over `base_model`. It runs in a process
    of its own, out of sight of the GPU: the converter would otherwise open the GPU in this process."""
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "rollout_tinker.weights", str(archive), str(into), base_model,
        env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )  # fmt: skip
    output, _ = await process.communicate()
    if process.returncode != 0:
        lines = output.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"converting the adapter failed: {lines[-1] if lines else process.returncode}")


def convert(archive: Path, into: Path, base_model: str) -> None:
    """What `converted` runs: the cookbook's conversion into a scratch directory, then `fused`, then into `into`."""
    from tinker_cookbook.weights import build_lora_adapter

    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=into) as scratch:
        made = Path(scratch) / "peft"
        build_lora_adapter(base_model=base_model, adapter_path=str(archive), output_path=str(made))
        fused(made)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.move(made / name, into / name)


_SPLIT = re.compile(r"^(?P<layer>.+)\.in_proj_(?P<part>[qkv])\.lora_(?P<side>[AB])\.weight$")


def fused(directory: Path) -> int:
    """Join each layer's `in_proj_q`, `in_proj_k` and `in_proj_v` adapters (PEFT's layout, in `directory`) into one
    `in_proj_qkv` adapter: A stacked, B block-diagonal, so B·A is the three updates one above the other, Q, K, V, as
    the layer's rows are. Its rank is their sum; `rank_pattern` and `alpha_pattern` keep its scale. Returns how many
    layers were joined."""
    import torch
    from safetensors.torch import load_file, save_file

    path = directory / "adapter_model.safetensors"
    tensors: dict[str, torch.Tensor] = load_file(str(path))
    parts: dict[str, dict[tuple[str, str], torch.Tensor]] = {}
    for key, tensor in tensors.items():
        found = _SPLIT.match(key)
        if found is not None:
            parts.setdefault(found["layer"], {})[(found["part"], found["side"])] = tensor
    if not parts:
        return 0
    said = json.loads((directory / "adapter_config.json").read_text())
    rank, alpha = int(said["r"]), float(said["lora_alpha"])
    joined_rank = 0
    for layer, found in parts.items():
        if len(found) != 6:
            raise ValueError(f"{layer}: in_proj_q, in_proj_k and in_proj_v must all be adapted to be joined")
        a = [found[(part, "A")] for part in "qkv"]
        b = [found[(part, "B")] for part in "qkv"]
        if all(torch.equal(a[0], each) for each in a[1:]):  # one A for the three: B stacked, at the same rank
            joined_a, joined_b = a[0], torch.cat(b)
        else:
            joined_a = torch.cat(a)
            joined_b = torch.zeros(sum(each.shape[0] for each in b), joined_a.shape[0], dtype=b[0].dtype)
            row = column = 0
            for each in b:
                joined_b[row : row + each.shape[0], column : column + each.shape[1]] = each
                row, column = row + each.shape[0], column + each.shape[1]
        joined_rank = max(joined_rank, joined_a.shape[0])
        for part in "qkv":
            for side in "AB":
                del tensors[f"{layer}.in_proj_{part}.lora_{side}.weight"]
        tensors[f"{layer}.in_proj_qkv.lora_A.weight"] = joined_a.contiguous()
        tensors[f"{layer}.in_proj_qkv.lora_B.weight"] = joined_b.contiguous()
    save_file(tensors, str(path), metadata={"format": "pt"})
    targets = [each for each in said.get("target_modules") or [] if each not in ("in_proj_q", "in_proj_k", "in_proj_v")]
    said["target_modules"] = sorted({*targets, "in_proj_qkv"})
    if joined_rank != rank:
        said["rank_pattern"] = {**(said.get("rank_pattern") or {}), "in_proj_qkv": joined_rank}
        said["alpha_pattern"] = {**(said.get("alpha_pattern") or {}), "in_proj_qkv": alpha * joined_rank / rank}
    (directory / "adapter_config.json").write_text(json.dumps(said, indent=2) + "\n")
    return len(parts)


def ranks(directory: Path) -> int:
    """The largest rank of any layer's adapter in `directory` (what an engine's `max_lora_rank` must reach)."""
    from safetensors import safe_open

    with safe_open(str(directory / "adapter_model.safetensors"), framework="pt") as opened:
        return max(opened.get_slice(key).get_shape()[0] for key in opened.keys() if ".lora_A." in key)  # noqa: SIM118


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
