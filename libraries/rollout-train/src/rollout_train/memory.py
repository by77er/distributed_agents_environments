"""What a trainer needs of each of its GPUs' memory, estimated from the model's size, the trainer's settings and how
many GPUs it shards over: what validation's `memory` rule reads (`rollout_train.validation`).

`ModelFacts` are what a model's files say of it (`config.json` and the size of its safetensors), gathered beforehand
(`model_facts`: from the model's directory, the Hugging Face cache, or the Hugging Face Hub). `trainer_memory` is a
pure function of them. Per GPU, for `n` GPUs, `P` parameters, a model of `B` bytes, hidden width `h`, `L` layers, a
vocabulary of `V`, segments of `s` tokens:

- **Weights.** An adapter's frozen model: `B` on one GPU; on several, gathered once and kept beside its shard
  (`B + B/n`) where each GPU holds the whole model (`whole_base`, `holds_whole_base`), else its shard `B/n`, two
  layers gathered at a time and the output layer kept whole; the adapter in float32, `4A/n`, and on several GPUs
  gathered (in float32: `4A`). Every weight: `4P/n` in float32, and (sharded on one GPU too) two layers gathered in
  bfloat16 and the embeddings and output layer with their gradients.
- **Gradients.** `4A/n` of the adapter; `4P/n` of every weight.
- **Optimizer.** Adam's two moments: `8A/n`; `8P/n`.
- **Reference.** None for an adapter (it is switched off); `2P/n` for every weight, where one is held
  (`frozen_reference`).
- **Activations.** Each layer's input kept (`2·s·h·L`), one layer recomputed (`34·s·h`), a chunk of logits
  (`128 · V · 12`); for an adapter, the recomputed layer's inputs cast to float32 for the adapter (`36·s·h`); for a
  model with linear attention (Qwen3.5's), the recomputed layer's state at each chunk of 64 tokens and its gradient,
  in float32 (`2 · s/64 · S · 4`, `S` the state's values: value heads times key width times value width).
- **Allowance.** 3 GiB: the CUDA context, the collectives' buffers, the allocator's fragmentation.

`A`, the adapter's parameters, is about `18 · rank · h · L` (every attention and MLP projection of every layer). The
figures are estimates to tell a shape that fits from one that cannot, not measurements.
"""

import json
import math
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWANCE_GIB",
    "GPU_MEMORY_GIB",
    "SEGMENT_TOKENS",
    "ModelFacts",
    "TrainerMemory",
    "gpu_memory_gib",
    "holds_whole_base",
    "model_facts",
    "trainer_memory",
]

GIB = 2**30
ALLOWANCE_GIB = 3.0
"""Each GPU's memory beyond what the estimate counts: the CUDA context, the collectives' buffers, fragmentation."""
SEGMENT_TOKENS = 8_192
"""The segment an estimate allows activations for where the trainer says no longest one (`segment_tokens`)."""
LOGIT_ROWS = 128
"""Positions sent through the output layer at a time (`rollout_lora.policy.LOGIT_ROWS`)."""
LINEAR_CHUNK = 64
"""Tokens of a linear-attention layer's chunk, at each of which its kernels keep the state (flash-linear-attention's
kernels)."""
GPU_MEMORY_GIB: Mapping[str, float] = {
    "NVIDIA B200": 180, "NVIDIA H200": 141, "NVIDIA H200 NVL": 141, "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 NVL": 94, "NVIDIA H100 PCIe": 80, "NVIDIA A100-SXM4-80GB": 80, "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 96, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 96,
    "NVIDIA L40S": 48, "NVIDIA L40": 48, "NVIDIA RTX 6000 Ada Generation": 48, "NVIDIA RTX A6000": 48,
    "NVIDIA A40": 48, "NVIDIA GeForce RTX 5090": 32, "NVIDIA GeForce RTX 4090": 24, "NVIDIA L4": 24,
}  # fmt: skip
"""Each GPU's memory by RunPod's GPU type id (a type not here says it in its id, `80GB`, or is not known)."""


@dataclass(frozen=True)
class ModelFacts:
    """What a model's files say of its size."""

    file_bytes: int
    """Its weights' files, in all (a quantized model's are its quantized weights)."""
    parameters: float | None
    """Its parameters (none for a quantized model, whose files do not say them)."""
    hidden: int
    layers: int
    vocabulary: int
    tied: bool
    """Whether its output layer is its token embeddings."""
    linear_state: int = 0
    """The values of a linear-attention layer's state (value heads times key width times value width); 0 for a model
    without linear attention."""


@dataclass(frozen=True)
class TrainerMemory:
    """What a trainer needs of each GPU, in GiB, by part (`rollout_train.memory`)."""

    gpus: int
    weights: float
    gradients: float
    optimizer: float
    reference: float
    activations: float
    allowance: float = ALLOWANCE_GIB
    whole_base: bool = False
    """For an adapter on several GPUs: whether the estimate has each hold the whole frozen model."""

    @property
    def total(self) -> float:
        return self.weights + self.gradients + self.optimizer + self.reference + self.activations + self.allowance

    def said(self) -> str:
        """In words: `52 GiB a GPU (weights 18, gradients 9, ...)`."""
        parts = [(name, getattr(self, name)) for name in ("weights", "gradients", "optimizer", "reference",
                                                          "activations", "allowance")]  # fmt: skip
        listed = ", ".join(f"{name} {value:.1f}" for name, value in parts if value >= 0.05)
        return f"{self.total:.1f} GiB a GPU ({listed})"


def trainer_memory(
    model: ModelFacts,
    *,
    weights: str,
    gpus: int,
    rank: int = 32,
    segment_tokens: int | None = None,
    frozen_reference: bool = False,
    whole_base: bool | None = None,
    gpu_gib: float | None = None,
) -> TrainerMemory:
    """What a trainer of `weights` (`lora` or `full`) over `model` on `gpus` GPUs needs of each
    (`rollout_train.memory`). `whole_base` is the adapter's setting (none: whole where the model takes at most half of
    `gpu_gib`: `holds_whole_base`)."""
    n = max(1, gpus)
    h, layers, vocabulary = model.hidden, model.layers, model.vocabulary
    s = segment_tokens or SEGMENT_TOKENS
    activations = 2 * s * h * layers + 34 * s * h + LOGIT_ROWS * vocabulary * 12
    activations += 2 * -(-s // LINEAR_CHUNK) * model.linear_state * 4
    heads = vocabulary * h * (1 if model.tied else 2)
    """The token embeddings and the output layer, which the root of a sharded model holds."""
    if weights == "lora":
        base = model.file_bytes
        adapter = 18 * rank * h * layers
        whole = n == 1 or (whole_base if whole_base is not None else holds_whole_base(base, gpu_gib))
        if n == 1:
            held = base
        elif whole:
            held = base + base / n
        else:
            held = base / n + 2 * base / layers + 2 * heads
        held += 4 * adapter / n + (4 * adapter if n > 1 else 0)
        activations += 36 * s * h  # (the recomputed layer's inputs, cast to float32 for the adapter)
        return TrainerMemory(n, held / GIB, 4 * adapter / n / GIB, 8 * adapter / n / GIB, 0.0, activations / GIB,
                             whole_base=whole and n > 1)  # fmt: skip
    parameters = model.parameters if model.parameters is not None else model.file_bytes / 2
    reference = 2 * parameters / n if frozen_reference else 0.0
    # Sharded on any number of GPUs: two layers gathered in bfloat16, the root's embeddings and output layer with
    # their gradients.
    held = 4 * parameters / n + 2 * 2 * (parameters - heads) / layers + 6 * heads
    reference += 2 * 2 * (parameters - heads) / layers if frozen_reference else 0.0
    return TrainerMemory(n, held / GIB, 4 * parameters / n / GIB, 8 * parameters / n / GIB, reference / GIB,
                         activations / GIB)  # fmt: skip


def holds_whole_base(file_bytes: float, gpu_gib: float | None) -> bool:
    """Whether each GPU holds an adapter's whole frozen model where the settings do not say (`whole_base`): where the
    model's files take at most half of a GPU's memory (`gpu_gib`, as `gpu_memory_gib` gives it; none: not known). The
    estimate decides by it, and so does the trainer, from its GPUs' names."""
    return gpu_gib is not None and file_bytes <= gpu_gib * GIB / 2


def gpu_memory_gib(gpu_types: tuple[str, ...]) -> float | None:
    """The least memory of any of RunPod's GPU types (`GPU_MEMORY_GIB`, or the `80GB` its id says); none where one is
    not known."""
    found: list[float] = []
    for each in gpu_types:
        said = re.search(r"(\d+)\s?GB", each)
        memory = GPU_MEMORY_GIB.get(each, float(said.group(1)) if said else None)
        if memory is None:
            return None
        found.append(memory)
    return min(found) if found else None


def _local(model: str, environ: Mapping[str, str]) -> Path | None:
    """A model's directory: `model` itself, or its snapshot in the Hugging Face cache."""
    directory = Path(model).expanduser()
    if directory.is_dir():
        return directory
    home = environ.get("HF_HOME")
    cache = environ.get("HF_HUB_CACHE") or (str(Path(home) / "hub") if home else None)
    root = Path(cache).expanduser() if cache else Path.home() / ".cache" / "huggingface" / "hub"
    cached = root / f"models--{model.replace('/', '--')}"
    reference = cached / "refs" / "main"
    if not reference.exists():
        return None
    snapshot = cached / "snapshots" / reference.read_text().strip()
    return snapshot if (snapshot / "config.json").exists() else None


def _header_bytes(file: Path) -> tuple[int, int]:
    """A safetensors file's tensor bytes and parameters, from its header."""
    with file.open("rb") as opened:
        (length,) = struct.unpack("<Q", opened.read(8))
        header: dict[str, Any] = json.loads(opened.read(length))
    tensors = [each for name, each in header.items() if name != "__metadata__"]
    total = sum(int(each["data_offsets"][1]) - int(each["data_offsets"][0]) for each in tensors)
    return total, sum(math.prod(int(size) for size in each["shape"]) for each in tensors)


def _facts(config: Mapping[str, Any], file_bytes: int, parameters: float | None) -> ModelFacts | None:
    text: Mapping[str, Any] = config.get("text_config") or config
    try:
        hidden, layers, vocabulary = int(text["hidden_size"]), int(text["num_hidden_layers"]), int(text["vocab_size"])
    except (KeyError, TypeError, ValueError):
        return None
    tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    quantized = "quantization_config" in config or "quantization_config" in text
    linear = 0
    sizes = [text.get(key) for key in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")]
    if all(isinstance(each, int) for each in sizes) and "linear_attention" in (text.get("layer_types") or []):
        linear = math.prod(int(each) for each in sizes if isinstance(each, int))
    return ModelFacts(file_bytes, None if quantized else parameters, hidden, layers, vocabulary, tied, linear)


def model_facts(model: str, *, environ: Mapping[str, str] | None = None, patience: float = 5.0) -> ModelFacts | None:
    """What a model's files say of its size: read from its directory or the Hugging Face cache, else asked of the
    Hugging Face Hub, for at most `patience` seconds (`config.json` and the safetensors index; `HF_TOKEN` for a gated
    model; never with `HF_HUB_OFFLINE`); none where neither says. It blocks: call it in a thread."""
    environ = os.environ if environ is None else environ
    found = _local(model, environ)
    if found is not None:
        try:
            config = json.loads((found / "config.json").read_text())
            sizes = [_header_bytes(each) for each in sorted(found.glob("*.safetensors"))]
        except (OSError, ValueError, KeyError, struct.error):
            return None
        if sizes:
            return _facts(config, sum(each for each, _ in sizes), float(sum(each for _, each in sizes)))
    if environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        return None
    return _from_hub(model, environ, patience)


def _from_hub(model: str, environ: Mapping[str, str], patience: float) -> ModelFacts | None:
    import httpx

    if "/" not in model or Path(model).expanduser().exists():
        return None
    base = f"{environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{model}/resolve/main"
    token = environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=patience, follow_redirects=True, headers=headers) as client:
            config = client.get(f"{base}/config.json").raise_for_status().json()
            index = client.get(f"{base}/model.safetensors.index.json")
            if index.status_code == 200:
                total = int(index.json()["metadata"]["total_size"])
                dtype = str((config.get("text_config") or config).get("dtype") or config.get("torch_dtype") or "")
                per = 4 if dtype in ("float32", "fp32") else 2
                return _facts(config, total, total / per)
            head = client.get(f"{base}/model.safetensors", headers={"Range": "bytes=0-7"})
            (length,) = struct.unpack("<Q", head.raise_for_status().content[:8])
            said = client.get(f"{base}/model.safetensors", headers={"Range": f"bytes=8-{7 + length}"})
            header: dict[str, Any] = json.loads(said.raise_for_status().content)
    except (httpx.HTTPError, ValueError, KeyError, struct.error):
        return None
    tensors = [each for name, each in header.items() if name != "__metadata__"]
    total = sum(int(each["data_offsets"][1]) - int(each["data_offsets"][0]) for each in tensors)
    return _facts(config, total, float(sum(math.prod(int(size) for size in each["shape"]) for each in tensors)))
