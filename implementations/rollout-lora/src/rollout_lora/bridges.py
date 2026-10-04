"""The LoRA trainer's bridge: an adapter merged into the weights it was trained over, for a provider that serves full
weights.

`merge_quantize` is the task of the `merge-quantize` bridge (`rollout_train.bridges`), which a run asks for by name
(`channels.NAME.bridge`). It folds the adapter into its base (`rollout_lora.merge.merge`, on the CPU) and writes the
full weights in the base's own dtype. A provider that quantizes as it loads (vLLM's `quantization = "fp8"`) serves
them so; a provider whose model is a checkpoint quantized beforehand (AWQ, GPTQ) is refused, since the merged weights
are not quantized here.
"""

from pathlib import Path

from pydantic import JsonValue

from rollout_lora.merge import merge
from rollout_train.bridges import Context

__all__ = ["merge_quantize"]


def merge_quantize(weights: Path, into: Path, context: Context) -> dict[str, JsonValue]:
    """The adapter in `weights` (PEFT's layout) folded into the model it is over, as full weights in `into`."""
    if context.model is None:
        raise ValueError(f"{context.checkpoint} says no model its adapter is over: there is nothing to merge it into")
    if context.target is not None and context.target != context.model:
        raise ValueError(
            f"the provider serves {context.target}, a copy of {context.model} quantized beforehand: merged weights are "
            "not quantized here (serve the adapter as it is, through `verbatim`)"
        )
    said = merge(context.model, weights, into, device="cpu")
    return {"model": context.model, **said}
