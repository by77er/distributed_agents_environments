"""What a pod's processes read from their environment: the stores, the pod's name, and where to serve on the pod's
loopback interface.

- `ROLLOUT_POD_NAME`: the pod's name, as its starter gave it; its certificate's identity is
  `spiffe://rollout/pod/NAME`.
- `ROLLOUT_LEDGER`: where the ledger is, as JSON (`{"kind": "rollout_train.database:DatabaseLedger", "url": "…"}`).
- `ROLLOUT_BLOBS`: where the blob store is, as JSON (`{"kind": "rollout_s3:S3BlobStore", "bucket": "…"}`); its
  credentials come from its own variables (`AWS_*`).
- `ROLLOUT_CERT_SERIAL_FILE`: a file holding the serial of the pod's certificate now (default
  `/certs/current/serial`), said in its beats.
- `ROLLOUT_BLOB_CACHE`: a directory on the pod's disk that keeps a copy of every blob the pod's processes put or read
  (`CachedBlobs`): on a host pod, the checkpoints the training service makes are loaded by the follower from it, with
  no round trip through the bucket (each is in the bucket too).

The ledger is opened with `rollout_train.ledger.opened`, so any implementation of `Ledger` can be named, by
`module:name` and its settings.
"""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout_train.ledger import Ledger, opened
from rollout_train.stores import opened as blobs_at

PORT = 8443
"""The port the pod's proxy listens on, which the provider maps to a public one."""


def required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def stores(environ: Mapping[str, str]) -> tuple[Ledger, Blobs]:
    """The ledger and the blob store the environment names (with a copy on the pod's disk, `ROLLOUT_BLOB_CACHE`)."""
    blobs = blobs_at(location(environ, "ROLLOUT_BLOBS"))
    if cache := environ.get("ROLLOUT_BLOB_CACHE"):
        blobs = CachedBlobs(blobs, FileBlobStore(Path(cache)))
    return opened(location(environ, "ROLLOUT_LEDGER")), blobs


class CachedBlobs:
    """A blob store with a copy of every blob that passes through it in a store of files on this machine: a put goes to
    both, a read is answered from the copy where it has the blob, else from the store (and kept)."""

    def __init__(self, store: Blobs, cache: FileBlobStore) -> None:
        self.store = store
        self.cache = cache

    def holds(self, reference: BlobReference) -> bool:
        holds = getattr(self.store, "holds", None)
        return bool(holds(reference)) if callable(holds) else False

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        reference = await self.store.put(data, media_type)
        await self.cache.put(data, media_type)
        return reference

    async def read(self, reference: BlobReference) -> bytes:
        try:
            return await self.cache.read(reference)
        except (OSError, ValueError):  # (not here, or not whole: read from the store)
            data = await self.store.read(reference)
            await self.cache.put(data, reference.media_type)
            return data

    async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None:
        await self.store.delete(reference, unused_for=unused_for)


def serial(path: Path | None) -> str | None:
    """The serial of the pod's certificate now, as the file the certificates' renewal writes says (None: unknown)."""
    if path is None:
        return None
    try:
        return path.read_text().strip().removeprefix("serial=") or None
    except OSError:
        return None


def serial_file(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("ROLLOUT_CERT_SERIAL_FILE", "/certs/current/serial"))


def listening(environ: Mapping[str, str], name: str, default: str) -> tuple[str, int]:
    """Where to serve, from `HOST:PORT` in variable `name`: on the loopback interface unless it says otherwise."""
    host, _, port = environ.get(name, default).rpartition(":")
    return host or "127.0.0.1", int(port)


async def served(app: Any, host: str, port: int) -> None:
    """Serve `app` at `host:port` until cancelled."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off"))
    try:
        await server.serve()
    except asyncio.CancelledError:
        server.should_exit = True
        raise


def location(environ: Mapping[str, str], name: str) -> dict[str, Any]:
    """Where a store is, as the JSON object in variable `name` says (`ROLLOUT_BLOBS`, `ROLLOUT_LEDGER`)."""
    try:
        said: Any = json.loads(required(environ, name))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{name} is not JSON: {error.msg}") from None
    if not isinstance(said, dict):
        raise SystemExit(f"{name} is not a JSON object")
    return cast(dict[str, Any], said)
