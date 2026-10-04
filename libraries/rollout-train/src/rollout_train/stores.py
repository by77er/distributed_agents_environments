"""Where a run's blobs are, said so that any process can open the same store: an open profile notes it in the run's
`starts` record, and the monitor reads finished episodes from there, wherever it runs."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.names import named

FILES = "rollout.harness.blobs:FileBlobStore"
"""The store of files in a directory (`{"kind": FILES, "directory": …}`)."""
SECRET = ("secret", "password", "token", "credential", "access_key")
"""Settings that are never written down: a store's credentials come from its environment."""


def location(store: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Where a profile's blob store is (`store`: its `[blobs]` table, `kind` and the store's settings; empty: files
    under `directory`), without any setting that looks like a credential."""
    if not store:
        return {"kind": FILES, "directory": str(directory)}
    return {key: value for key, value in store.items() if not any(word in key.lower() for word in SECRET)}


def opened(where: Mapping[str, Any]) -> Blobs:
    """The blob store a location names."""
    settings = dict(where)
    kind = str(settings.pop("kind"))
    if kind == FILES:
        return FileBlobStore(Path(str(settings["directory"])).expanduser())
    return named(kind)(**settings)
