"""Tinker's bridge: a checkpoint trained on Tinker, as an adapter in PEFT's layout that engines here load.

`peft` is the task of the `peft-from-tinker` bridge (`rollout_train.bridges`). The checkpoint's weights are a pointer
(`rollout_tinker.weights`) to its sampler checkpoint at Thinking Machines; the bridge asks Tinker for that checkpoint's
archive, downloads it, and writes the adapter in PEFT's layout (`rollout_tinker.weights.peft_adapter`: Tinker's names
made the model's own, and Qwen3.5's q, k and v joined into one projection), on the CPU. It reads the model's
configuration and its weights' names, never its weights.

Its settings (`Context.settings`): `service`, `module:name` of what makes the service it asks (by default a session
with Tinker, which finds its key as the trainer's does), and `project`, a Tinker project's id.
"""

import asyncio
import shutil
from pathlib import Path

from pydantic import JsonValue

from rollout_tinker.service import Service, service_of
from rollout_tinker.weights import downloaded, peft_adapter, pointer, ranks
from rollout_train.bridges import Context

__all__ = ["converted", "peft"]


def peft(weights: Path, into: Path, context: Context) -> dict[str, JsonValue]:
    """The adapter of the Tinker checkpoint `weights` points at, in PEFT's layout in `into`."""
    service, project = context.settings.get("service"), context.settings.get("project")
    asked = service_of(str(service) if service else None, str(project) if project else None)
    return asyncio.run(converted(weights, into, context.model, asked))


async def converted(weights: Path, into: Path, model: str | None, service: Service) -> dict[str, JsonValue]:
    """`peft`, asking `service`: the sampler checkpoint's archive downloaded beside `into` and converted over the
    model the pointer names (else `model`); the download is not kept. Says the sampler checkpoint and the adapter's
    largest rank (what an engine's `max_lora_rank` must reach)."""
    sampler = pointer(weights, "sampler")
    if sampler is None:
        raise ValueError("the checkpoint was not trained on Tinker: its weights name no sampler checkpoint")
    base = pointer(weights, "base_model") or model
    if base is None:
        raise ValueError("the checkpoint says no model its adapter is over")
    archive = await service.create_rest_client().get_checkpoint_archive_url_from_tinker_path_async(sampler)
    scratch = into.parent / f"{into.name}.archive"
    try:
        found = await downloaded(archive.url, scratch)
        await asyncio.to_thread(peft_adapter, found, into, base)
    finally:
        await asyncio.to_thread(shutil.rmtree, scratch, ignore_errors=True)
    return {"sampler": sampler, "model": base, "largest_rank": await asyncio.to_thread(ranks, into)}
