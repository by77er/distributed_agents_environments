# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (safetensors' loaders are partly untyped.)
"""A checkpoint's files when its weights are at Thinking Machines: pointers, and the adapter in PEFT's layout.

`weights/tinker.json` names the sampler checkpoint to sample from and the training state a later step starts from;
`state/tinker.json` names the state again (a step given the state goes on with its optimizer, one given the weights
alone starts its optimizer afresh). The pointers are files like any other: the blob store keeps them, and a checkpoint
copied to another machine points at the same remote checkpoints.

The sampler checkpoint's archive holds the adapter with Tinker's own names. Tinker's bridge
(`rollout_tinker.bridges`) downloads it and turns it into PEFT's layout (`peft_adapter`), so that vLLM serves it,
`rollout merge` folds it in, and the blob store holds it. Tinker names each adapted weight `base_model.model.` and its
name in a plain text model; renaming makes it the model's own name (Qwen3.5's weights are under
`model.language_model.`). Qwen3.5's linear-attention layers hold one `in_proj_qkv` where Tinker adapts `in_proj_q`,
`in_proj_k` and `in_proj_v` apart, and vLLM adapts only the joined name: the three are joined into one adapter
(`fused`), which is the same update.
"""

import asyncio
import json
import re
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


_RENAMED_OTHERWISE = ("gpt_oss", "nemotron_h", "deepseek", "kimi")
"""Model types whose adapters Tinker names in other ways than renaming alone undoes (their own prefixes, projections
fused differently): refused."""


def peft_adapter(archive: Path, into: Path, base_model: str) -> None:
    """The adapter in `archive` (Tinker's names) in PEFT's layout in `into`, over `base_model` (a model's name or
    directory): each tensor renamed to the layer it adapts in the model's own weights (`peft_names`), q, k and v
    joined (`fused`), and `adapter_config.json` written. It reads the model's configuration and its weights' names,
    never its weights, and computes on the CPU."""
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(archive / "adapter_model.safetensors"))
    said = json.loads((archive / "adapter_config.json").read_text())
    for key in ("lora_alpha", "r"):
        if key not in said:
            raise ValueError(f"the adapter's configuration has no {key!r}")
    configuration, keys = model_names(base_model)
    renamed, targets = peft_names(tensors, configuration, keys)
    made = peft_configuration(said, base_model, targets)
    fused(renamed, made)
    into.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(into / "adapter_model.safetensors"), metadata={"format": "pt"})
    (into / "adapter_config.json").write_text(json.dumps(made, indent=2) + "\n")


def local(model: str) -> Path:
    """A model's directory: `model` itself if it is one, else its snapshot in the Hugging Face cache (as the cache's
    `refs/main` names it). Raises `FileNotFoundError` where it is neither."""
    directory = Path(model).expanduser()
    if directory.is_dir():
        return directory
    from huggingface_hub.constants import HF_HUB_CACHE

    cached = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
    reference = cached / "refs" / "main"
    if not reference.exists():
        raise FileNotFoundError(f"{model} is neither a directory nor in the Hugging Face cache")
    return cached / "snapshots" / reference.read_text().strip()


def model_names(base_model: str) -> tuple[dict[str, Any], set[str]]:
    """A model's configuration and the names of its weights: from its directory, or its snapshot in the Hugging Face
    cache, or else the Hub (its configuration and its safetensors' headers, not its weights)."""
    try:
        directory = local(base_model)
    except FileNotFoundError:
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        configuration = json.loads(Path(hf_hub_download(base_model, "config.json")).read_text())
        return configuration, set(get_safetensors_metadata(base_model).weight_map)
    configuration = json.loads((directory / "config.json").read_text())
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        return configuration, set(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    keys: set[str] = set()
    for file in sorted(directory.glob("*.safetensors")):
        with safe_open(str(file), framework="pt") as opened:
            keys.update(opened.keys())
    if not keys:
        raise FileNotFoundError(f"{directory} holds no safetensors")
    return configuration, keys


def renames(configuration: dict[str, Any], keys: set[str]) -> list[tuple[str, str]]:
    """What turns Tinker's name for an adapted weight into the model's, in order: Tinker's `base_model.model.` prefix
    dropped; its `unembed_tokens` the model's `lm_head`, or its `embed_tokens` where the two are tied; and the
    `model.language_model.` prefix of a model whose text is the language part of a larger one (Qwen3.5)."""
    model_type = str(configuration.get("model_type", ""))
    if model_type.startswith(_RENAMED_OTHERWISE):
        raise ValueError(f"adapters of {model_type} models are named otherwise: only renaming is done here")
    language = any(key.startswith("model.language_model.") for key in keys)
    head = any(key == "lm_head.weight" or key.startswith("lm_head.") for key in keys)
    return [
        ("base_model.model.", ""),
        ("model.unembed_tokens", "lm_head" if head else "model.embed_tokens"),
        *([("model.", "model.language_model.")] if language else []),
    ]


def peft_names(
    tensors: dict[str, Any], configuration: dict[str, Any], keys: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Tinker's adapter tensors under PEFT's names (`base_model.model.` + the adapted weight's name in the model, then
    `.lora_A.weight` or `.lora_B.weight`), and the adapted modules' short names (PEFT's `target_modules`)."""
    replacements = renames(configuration, keys)
    renamed: dict[str, Any] = {}
    targets: set[str] = set()
    for key in tensors:
        if ".lora_A" not in key:
            continue
        name = key.replace(".lora_A", "")
        if ".experts" in name:
            raise ValueError(f"{name}: experts' adapters (a mixture of experts) are not converted here")
        target = name
        for old, new in replacements:
            target = target.replace(old, new)
        module = target.removesuffix(".weight")
        for side in "AB":
            peft = f"base_model.model.{module}.lora_{side}.weight"
            if peft in renamed:
                raise ValueError(f"two of the adapter's weights are named {peft}")
            renamed[peft] = tensors[name.replace(".weight", f".lora_{side}.weight")]
        targets.add(module.rsplit(".", 1)[-1])
    if not renamed:
        raise ValueError("the archive holds no LoRA weights")
    return renamed, sorted(targets)


def peft_configuration(said: dict[str, Any], base_model: str, targets: list[str]) -> dict[str, Any]:
    """PEFT's `adapter_config.json` for an adapter of Tinker's configuration `said`."""
    return {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": said["lora_alpha"],
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": said["r"],
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }


_SPLIT = re.compile(r"^(?P<layer>.+)\.in_proj_(?P<part>[qkv])\.lora_(?P<side>[AB])\.weight$")


def fused(tensors: dict[str, Any], said: dict[str, Any]) -> int:
    """Join each layer's `in_proj_q`, `in_proj_k` and `in_proj_v` adapters (PEFT's names, in `tensors`, with PEFT's
    configuration `said`) into one `in_proj_qkv` adapter: A stacked, B block-diagonal, so B·A is the three updates
    one above the other, Q, K, V, as the layer's rows are. Its rank is their sum; `rank_pattern` and `alpha_pattern`
    keep its scale. Both are changed in place; returns how many layers were joined."""
    import torch

    parts: dict[str, dict[tuple[str, str], torch.Tensor]] = {}
    for key, tensor in tensors.items():
        found = _SPLIT.match(key)
        if found is not None:
            parts.setdefault(found["layer"], {})[(found["part"], found["side"])] = tensor
    if not parts:
        return 0
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
    targets = [each for each in said.get("target_modules") or [] if each not in ("in_proj_q", "in_proj_k", "in_proj_v")]
    said["target_modules"] = sorted({*targets, "in_proj_qkv"})
    if joined_rank != rank:
        said["rank_pattern"] = {**(said.get("rank_pattern") or {}), "in_proj_qkv": joined_rank}
        said["alpha_pattern"] = {**(said.get("alpha_pattern") or {}), "in_proj_qkv": alpha * joined_rank / rank}
    return len(parts)


def ranks(directory: Path) -> int:
    """The largest rank of any layer's adapter in `directory` (what an engine's `max_lora_rank` must reach)."""
    from safetensors import safe_open

    with safe_open(str(directory / "adapter_model.safetensors"), framework="pt") as opened:
        return max(opened.get_slice(key).get_shape()[0] for key in opened.keys() if ".lora_A." in key)  # noqa: SIM118
