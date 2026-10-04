"""A LoRA checkpoint folded into the weights it was trained over: a full checkpoint of its own.

The merged checkpoint's parent is the LoRA checkpoint, its kind `full`, and its base the model its line began from;
it was made by no run. A run started from it (`[trainer] start`) loads its files as the model its engines serve and
its trainer trains: new adapters stack on the merged weights, or every weight goes on being trained.

What folds an adapter in is named as `module:name` (`rollout_lora.merge:merge` by default) and called with the base
(a model's name or directory), the adapter's directory and where to write the merged model. The adapter is the
checkpoint's weights when they are in PEFT's layout, else what a bridge made of them in it (a Tinker checkpoint's
adapter, once `peft-from-tinker` has bridged it: `rollout_train.bridges`).
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from rollout.names import named
from rollout_train.bridges import BRIDGES, format_of, made
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest, new_id
from rollout_train.ledger import Fence

MERGER = "rollout_lora.merge:merge"
SCOPE = "merges"
"""The scope whose fence a merge appends its checkpoint under."""


async def merge(
    checkpoints: Checkpoints,
    fence: Fence,
    lora: str,
    *,
    base: str | None = None,
    merger: str = MERGER,
    scratch: Path,
) -> Checkpoint:
    """Fold the LoRA checkpoint `lora` (by id) into `base` (by default what it was trained over: its base model, or
    the full checkpoint it builds on) and add the full checkpoint it makes. `base` names another copy of the same
    model where the adapter was trained over a quantized one (`Qwen/Qwen3.5-9B` for `…-AWQ-4bit`, say). `scratch`
    holds the files on this machine while it works."""
    made = await checkpoints.checkpoint(lora)
    if made.kind != "lora":
        raise ValueError(f"{lora} is a full checkpoint: there is no adapter to merge")
    if made.weights is None:
        raise ValueError(f"{lora} was released: its weights were deleted")
    await asyncio.to_thread(scratch.mkdir, parents=True, exist_ok=True)
    work = Path(await asyncio.to_thread(tempfile.mkdtemp, dir=scratch, prefix=f"merge-{lora}-"))
    try:
        over = root = base or made.base  # (the merged weights are that model's, whatever the adapter trained over)
        if over is None:
            raise ValueError(f"{lora} names no base to merge into: say one")
        known = {each.id: each for each in await checkpoints.all()}
        if over in known:  # an adapter over a full checkpoint: that checkpoint's files are what it merges into
            under = known[over]
            if under.weights is None:
                raise ValueError(f"{over} was released: its weights were deleted")
            root = under.base
            over = str(await checkpoints.files(under.weights, work / "base"))
        adapter = await checkpoints.files(await _adapter(checkpoints, lora, made.weights), work / "adapter")
        into = work / "merged"
        said = await asyncio.to_thread(named(merger), over, adapter, into)
        metrics = {f"merged_{key}": float(value) for key, value in dict(said).items()}
        return await checkpoints.add(
            fence, new_id(), weights=into, run=None, base=root, kind="full", parents=[lora], metrics=metrics
        )
    finally:
        await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)


async def _adapter(checkpoints: Checkpoints, lora: str, weights: Manifest) -> Manifest:
    """The files of a LoRA checkpoint's adapter: its weights, or, where they are in a format a bridge turns into PEFT's
    layout (a Tinker checkpoint's pointer), what that bridge made of them."""
    formats = format_of(weights.files)
    bridges = [each for each in BRIDGES if each.target == "peft" and each.source in formats]
    if "peft" in formats or not bridges:
        return weights
    for bridge in bridges:
        if (found := await made(checkpoints.ledger, lora, bridge.name)) is not None:
            return found
    said = ", ".join(sorted(formats))
    raise ValueError(f"{lora}'s weights are not an adapter in PEFT's layout ({said}): bridge them to PEFT first")
