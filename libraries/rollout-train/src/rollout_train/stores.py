"""The ledger and the blob store a process works with, opened from the cluster config (`Stores.open`), and where a
run's blobs are, said so that any process can open the same store.

`Stores.open(cluster)` opens the ledger its `[ledger]` names (a database: `sqlite:///…` or `postgresql://…`, the URL
read on this node from the environment variable or file the config names, where it names one) and the blob store its
`[blobs]` names (files in a directory, or `module:name` called with the store's settings, such as
`rollout_s3:S3BlobStore` with a `bucket`; the store's credentials come from its own environment). The stores beside
the ledger (checkpoints, the registry, presets) are reached through it.

A run's `starts` record notes where its blobs are (`location`, `Stores.location`), and the monitor reads finished
episodes from there, wherever it runs (`opened`). A process the run starts elsewhere (an engine host, a bridge's task)
is told where the ledger is as `ledger_at`: the cluster config itself, so that it reads the ledger's URL on its own
node (`cluster_ledger`), and no secret is handed on.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.names import named

if TYPE_CHECKING:
    from rollout_train.checkpoints import Checkpoints
    from rollout_train.cluster import Cluster
    from rollout_train.ledger import Ledger
    from rollout_train.presets import Presets
    from rollout_train.registry import Registry

__all__ = ["FILES", "Stores", "blobs_at", "cluster_ledger", "ledger_at", "ledger_url", "location", "opened"]

FILES = "rollout.harness.blobs:FileBlobStore"
"""The store of files in a directory (`{"kind": FILES, "directory": …}`)."""
SECRET = ("secret", "password", "token", "credential", "access_key")
"""Settings that are never written down: a store's credentials come from its environment."""
DATABASES = ("sqlite:", "postgresql:", "postgresql+")
"""What a ledger's URL begins with."""


def location(store: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Where a blob store is (`store`: a `[blobs]` table, `kind` and the store's settings; empty: files under
    `directory`), without any setting that looks like a credential."""
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


def ledger_url(cluster: "Cluster", environ: Mapping[str, str] | None = None) -> str:
    """The URL of the ledger a cluster names: its `[ledger] url`, or the value of the secret it names, read now on
    this node. Raises `ClusterError` where the secret is not set here, or the URL is not a database's (saying where
    it came from, never the URL: it may hold a password)."""
    from rollout_train.cluster import ClusterError

    named_by = cluster.ledger.url_secret
    url = cluster.ledger.url if named_by is None else named_by.resolve(environ)
    if url is None:
        raise ClusterError(f"[ledger] names its URL as {named_by}, which is not set on this node")
    if not url.startswith(DATABASES):
        said = repr(url) if named_by is None else f"the URL {named_by} holds"
        raise ClusterError(f"[ledger] url is a database's (sqlite:///… or postgresql://…), and {said} is not")
    return url


def cluster_ledger(cluster: Mapping[str, Any]) -> "Ledger":
    """The ledger a cluster config names (given as JSON, `Cluster.described`), opened on this node: what a ledger's
    location `ledger_at` gives names (`rollout_train.ledger.opened`)."""
    from rollout_train.cluster import parsed
    from rollout_train.database import DatabaseLedger

    return DatabaseLedger(ledger_url(parsed(cluster)))


def ledger_at(cluster: "Cluster") -> dict[str, JsonValue]:
    """Where a cluster's ledger is, as a ledger's location (`rollout_train.ledger.opened`) that names the cluster
    config: whoever opens it reads the URL, and any secret it is behind, on its own node."""
    return {"kind": "rollout_train.stores:cluster_ledger", "cluster": dict(cluster.described)}


def blobs_at(cluster: "Cluster") -> dict[str, JsonValue]:
    """Where a cluster's blob store is (its `[blobs]`), as `opened` opens it: a directory of files made absolute."""
    settings = dict(cluster.blobs.settings)
    if cluster.blobs.kind == "files":
        directory = Path(str(settings["directory"])).expanduser().absolute()
        return {"kind": FILES, "directory": str(directory)}
    return {"kind": cluster.blobs.kind, **settings}


@dataclass(frozen=True)
class Stores:
    """The ledger and the blob store, opened, and where the blob store is (`location`: as any process opens it, for a
    run's `starts` record)."""

    ledger: "Ledger"
    blobs: Blobs
    location: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])

    @classmethod
    def open(cls, cluster: "Cluster", environ: Mapping[str, str] | None = None) -> "Stores":
        """The stores a cluster's config names, opened on this node: the ledger from `[ledger]` (its URL read from the
        secret it names, where it names one), the blob store from `[blobs]`. Raises `ClusterError` where the ledger's
        URL is not set here or is not a database's."""
        from rollout_train.database import DatabaseLedger

        ledger = DatabaseLedger(ledger_url(cluster, environ))
        where = blobs_at(cluster)
        return cls(ledger, opened(where), where)

    @property
    def checkpoints(self) -> "Checkpoints":
        from rollout_train.checkpoints import Checkpoints

        return Checkpoints(self.ledger, self.blobs)

    @property
    def registry(self) -> "Registry":
        """Run names, bookmarks, and dataset and suite names, beside the ledger."""
        from rollout_train.registry import registry_of

        registry = registry_of(self.ledger)
        assert registry is not None, "a database ledger has a registry beside it"
        return registry

    @property
    def presets(self) -> "Presets":
        from rollout_train.presets import presets_of

        presets = presets_of(self.ledger)
        assert presets is not None, "a database ledger has presets beside it"
        return presets
