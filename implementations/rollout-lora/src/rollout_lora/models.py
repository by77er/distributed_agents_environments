"""Where a model's files are on this machine, what kind of model they hold, and where its decoder is once loaded."""

import json
from pathlib import Path
from typing import Any


def local(model: str) -> Path:
    """A model's directory: `model` itself if it is one, else its snapshot in the Hugging Face cache (as the cache's
    `refs/main` names it: a snapshot missing files a model does not need, such as its README, is still found)."""
    directory = Path(model).expanduser()
    if directory.is_dir():
        return directory
    from huggingface_hub.constants import HF_HUB_CACHE

    cached = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
    reference = cached / "refs" / "main"
    if not reference.exists():
        raise FileNotFoundError(f"{model} is neither a directory nor in the Hugging Face cache")
    return cached / "snapshots" / reference.read_text().strip()


def fetched(model: str) -> Path:
    """A model's directory, downloaded into the Hugging Face cache first where it is neither a directory nor cached (a
    training pod starts with an empty cache; each process may ask, and the hub's file locks keep it to one copy)."""
    try:
        return local(model)
    except FileNotFoundError:
        from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

        snapshot_download(model)  # pyright: ignore[reportUnknownMemberType]
        return local(model)


def config(model: str) -> dict[str, Any]:
    return json.loads((local(model) / "config.json").read_text())


def multimodal(model: str) -> bool:
    """Whether a model's text is the language part of a larger model (an image-text one, such as Qwen3.5)."""
    return any("ConditionalGeneration" in each for each in config(model).get("architectures", []))


def quantized(model: str) -> bool:
    found = config(model)
    return "quantization_config" in found or "quantization_config" in found.get("text_config", {})


def body(model: Any) -> Any:
    """A loaded model's decoder, whose last hidden states the output layer reads: an image-text model's language part,
    or a text model's own."""
    inner = model.model
    return getattr(inner, "language_model", inner)


COPIED = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "special_tokens_map.json", "vocab.json", "merges.txt", "preprocessor_config.json")  # fmt: skip
"""A model's files beside its weights, which a checkpoint made from it keeps too (engines and renderers read them)."""
