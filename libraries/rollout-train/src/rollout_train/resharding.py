"""Resharding: a version's files rewritten into the layout its engines load, as a task of its own.

A trainer writes a version's weights in its own layout; an engine may need another (its tensor-parallel division, a
merged checkpoint, a format of its own). A layout is a function, named as `module:name`, that writes the engines'
files from the trainer's: `layout(weights, into)` returns what it says about the files it wrote. `verbatim` is the
layout for engines that load the trainer's files as they are, such as vLLM with a LoRA adapter.

`reshard` notes in the ledger when it begins (`versions/resharding`) and what it made (`versions/resharded`: the
layout and a manifest of the files, in the blob store), under the fence of the run that made the version. A version
resharded before is not resharded again. With Ray, `on_ray` runs it as a Ray task on whichever node has room; without,
it runs in this process.
"""

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.names import named
from rollout_train.ledger import Fence, Ledger
from rollout_train.ray_cluster import prepare
from rollout_train.trainer import WEIGHTS
from rollout_train.versions import Manifest, Versions, kept

RESHARDING, RESHARDED = "versions/resharding", "versions/resharded"
"""The ledger's tables of reshards begun, and of what each made, by version id."""
VERBATIM = "rollout_train.resharding:verbatim"

_MANIFEST = TypeAdapter(Manifest)


def verbatim(weights: Path, into: Path) -> dict[str, JsonValue]:
    """The engines load the trainer's files as they are: each is linked (or copied) into `into`."""
    for each in sorted(weights.rglob("*")):
        if each.is_file():
            target = into / each.relative_to(weights)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(each, target)
            except OSError:
                shutil.copy2(each, target)
    return {"kind": "verbatim"}


async def resharded(ledger: Ledger, version: str) -> Manifest | None:
    """What a version was resharded into, if it was."""
    record: Any = (await ledger.read(RESHARDED)).get(version)
    return _MANIFEST.validate_python(record["files"]) if record else None


async def reshard(versions: Versions, fence: Fence, version: str, layout: str, scratch: Path) -> Manifest:
    """Reshard a version into `layout` (`module:name`), unless it was: the manifest of the engines' files. `scratch`
    is where the trainer's files are read to and the engines' written, on this machine, for the while it takes."""
    if (done := await resharded(versions.ledger, version)) is not None:
        return done
    made = await versions.version(version)
    if made.weights is None:
        raise ValueError(f"{version} was released: its weights were deleted")
    begun: JsonValue = {"at": round(time.time(), 1), "layout": layout, "host": os.uname().nodename}
    await versions.ledger.append(RESHARDING, version, begun, fence)
    await asyncio.to_thread(scratch.mkdir, parents=True, exist_ok=True)
    work = Path(await asyncio.to_thread(tempfile.mkdtemp, dir=scratch, prefix=f"{version}-"))
    try:
        weights = await versions.files(made.weights, work / WEIGHTS)
        into = work / "resharded"
        said = await asyncio.to_thread(named(layout), weights, into)
        manifest = await kept(into, versions.blobs)
    finally:
        await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)
    record: dict[str, Any] = {
        "at": round(time.time(), 1),
        "layout": layout,
        "said": said,
        "files": _MANIFEST.dump_python(manifest, mode="json"),
    }
    if not await versions.ledger.append(RESHARDED, version, record, fence):
        return await resharded(versions.ledger, version) or manifest
    return manifest


def _on_worker(
    ledger_at: Mapping[str, Any], blobs_at: Mapping[str, Any], scope: str, number: int, version: str, layout: str
) -> dict[str, Any]:
    """`reshard` in a Ray worker: it opens the ledger and the blob store from where they are, and returns the
    manifest as JSON (what crosses between processes)."""
    from rollout_train.ledger import opened
    from rollout_train.stores import opened as store_at

    versions = Versions(opened(ledger_at), store_at(blobs_at))
    scratch = Path.home() / ".cache" / "rollout" / "resharding"  # (on disk: a machine's /tmp may be memory)
    manifest = asyncio.run(reshard(versions, Fence(scope, number), version, layout, scratch))
    return _MANIFEST.dump_python(manifest, mode="json")


def connect(address: str) -> None:
    """Connect this process to a Ray cluster (`auto`: the one this machine is part of), as `ray_cluster` says."""
    prepare()
    import ray

    ray.init(address=address, log_to_driver=False)  # pyright: ignore[reportUnknownMemberType]


def disconnect() -> None:
    import ray

    ray.shutdown()  # pyright: ignore[reportUnknownMemberType]


async def on_ray(
    ledger_at: Mapping[str, Any], blobs_at: Mapping[str, Any], fence: Fence, version: str, layout: str
) -> Manifest:
    """`reshard` as a Ray task (one CPU), on the cluster this process is connected to (`ray.init`)."""
    import ray

    task = ray.remote(num_cpus=1)(_on_worker)
    reference = task.remote(dict(ledger_at), dict(blobs_at), fence.scope, fence.number, version, layout)
    return _MANIFEST.validate_python(await asyncio.wrap_future(reference.future()))
