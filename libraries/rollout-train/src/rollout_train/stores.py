"""The ledger and the blob store a process works with, opened from the cluster config (`Stores.open`), and where a
run's blobs are, said so that any process can open the same store.

`Stores.open(cluster)` opens the ledger its `[ledger]` names (a database: `sqlite:///…` or `postgresql://…`, the URL
read on this node from the environment variable or file the config names, where it names one; or the ledger service,
`http(s)://…`, with the platform's token) and the blob store its
`[blobs]` names (files in a directory, or `module:name` called with the store's settings, such as
`rollout_s3:S3BlobStore` with a `bucket`; the store's credentials come from its own environment). The stores beside
the ledger (checkpoints, the registry, presets) are reached through it.

A cluster may name blob stores beside the default (`[stores.NAME]`: an R2 bucket that RunPod's pods reach, say). A run
writes to one store (`Stores.open(cluster, store=NAME)`: a run whose trainer or servers are RunPod's writes to the
store its RunPod providers name), and its `starts` record notes where that is (`location`, `Stores.location`), with
the names of the variables its key is read from, never the key. Every blob reference says its store in its URI
(`s3://BUCKET/PREFIX…`), and whoever reads a checkpoint's files finds each in the store that holds it: its own, or
the store a run's start names (`rollout_train.checkpoints.Checkpoints.files`). A process that reads such references
holds the keys of every store. The monitor reads finished episodes from where a run's start says, wherever it runs
(`opened`). A pod gets a store's location and a key of its own (`for_pods`): the read-only key (the store's `reader`)
for one that only reads, the store's own for one that writes. A process the run starts elsewhere (an engine host, a
bridge's task) is told where the ledger is as `ledger_at`: the cluster config itself, so that it reads the ledger's URL
on its own node (`cluster_ledger`), and no secret is handed on. A page says a store as `described` says it: its name
(`store_named` finds it from a location), its kind, and its bucket and prefix or its directory.
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
    from rollout_train.providers import Secret
    from rollout_train.registry import Registry

__all__ = [
    "FILES",
    "Stores",
    "blobs_at",
    "cluster_ledger",
    "described",
    "for_pods",
    "ledger_at",
    "ledger_of",
    "ledger_url",
    "location",
    "opened",
    "opened_ledger",
    "store_named",
]

FILES = "rollout.harness.blobs:FileBlobStore"
"""The store of files in a directory (`{"kind": FILES, "directory": …}`)."""
SECRET = ("secret", "password", "token", "credential", "access_key")
"""Settings that are never written down: a store's credentials come from its environment."""
DATABASES = ("sqlite:", "postgresql:", "postgresql+")
"""What a database ledger's URL begins with."""
SERVICES = ("http://", "https://")
"""What the ledger service's URL begins with."""


def location(store: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Where a blob store is (`store`: a `[blobs]` table, `kind` and the store's settings; empty: files under
    `directory`), without any setting that looks like a credential (the names of the variables one is read from,
    `…_env`, are kept)."""
    if not store:
        return {"kind": FILES, "directory": str(directory)}
    return {key: value for key, value in store.items()
            if key.endswith(("_env", "_file")) or not any(word in key.lower() for word in SECRET)}  # fmt: skip


def opened(where: Mapping[str, Any]) -> Blobs:
    """The blob store a location names."""
    settings = dict(where)
    kind = str(settings.pop("kind"))
    if kind == FILES:
        return FileBlobStore(Path(str(settings["directory"])).expanduser())
    return named(kind)(**settings)


def described(where: Mapping[str, Any], name: str | None = None) -> dict[str, JsonValue]:
    """A blob store as a page says it, from its location (`blobs_at`, `location`): its `name` (`[stores.NAME]`; none:
    `[blobs]`), its `kind` (`files`, or `module:name`), and its `bucket` and `prefix` or its `directory`; never a key,
    nor the variables one is read from."""
    kind = str(where.get("kind") or FILES)

    def said(key: str) -> str | None:
        value = where.get(key)
        return str(value) if isinstance(value, str) and value else None

    return {"name": name, "kind": "files" if kind == FILES else kind, "bucket": said("bucket"),
            "prefix": said("prefix"), "directory": said("directory")}  # fmt: skip


def store_named(cluster: "Cluster", where: Mapping[str, Any]) -> str | None:
    """The name of the cluster's store a location is (`[stores.NAME]`), where it is one of them; none for `[blobs]`
    or a store the cluster does not name."""
    for name in cluster.stores:
        if blobs_at(cluster, name) == dict(where):
            return name
    return None


def ledger_url(cluster: "Cluster", environ: Mapping[str, str] | None = None) -> str:
    """The URL of the ledger a cluster names: its `[ledger] url`, or the value of the secret it names, read now on
    this node. Raises `ClusterError` where the secret is not set here, or the URL is not a database's (saying where
    it came from, never the URL: it may hold a password)."""
    from rollout_train.cluster import ClusterError

    named_by = cluster.ledger.url_secret
    url = cluster.ledger.url if named_by is None else named_by.resolve(environ)
    if url is None:
        raise ClusterError(f"[ledger] names its URL as {named_by}, which is not set on this node")
    if not url.startswith((*DATABASES, *SERVICES)):
        said = repr(url) if named_by is None else f"the URL {named_by} holds"
        raise ClusterError(
            f"[ledger] url is a database's (sqlite:///… or postgresql://…) or the ledger service's (https://…), and "
            f"{said} is neither"
        )
    return url


def opened_ledger(url: str, token: "Secret | None" = None) -> "Ledger":
    """The ledger at `url`: a database (`sqlite:///…`, `postgresql://…`), or the ledger service (`http(s)://…`) with
    the token `token` names."""
    if url.startswith(SERVICES):
        from rollout_train.ledger_service import HttpLedger

        return HttpLedger(url, token_env=token.env if token else None, token_file=token.file if token else None)
    from rollout_train.database import DatabaseLedger

    return DatabaseLedger(url)


def ledger_of(cluster: "Cluster", environ: Mapping[str, str] | None = None) -> "Ledger":
    """The ledger a cluster's config names, opened on this node (`ledger_url`, `opened_ledger`)."""
    return opened_ledger(ledger_url(cluster, environ), cluster.ledger.token)


def cluster_ledger(cluster: Mapping[str, Any]) -> "Ledger":
    """The ledger a cluster config names (given as JSON, `Cluster.described`), opened on this node: what a ledger's
    location `ledger_at` gives names (`rollout_train.ledger.opened`)."""
    from rollout_train.cluster import parsed

    return ledger_of(parsed(cluster))


def ledger_at(cluster: "Cluster") -> dict[str, JsonValue]:
    """Where a cluster's ledger is, as a ledger's location (`rollout_train.ledger.opened`) that names the cluster
    config: whoever opens it reads the URL, and any secret it is behind, on its own node."""
    return {"kind": "rollout_train.stores:cluster_ledger", "cluster": dict(cluster.described)}


def blobs_at(cluster: "Cluster", store: str | None = None) -> dict[str, JsonValue]:
    """Where a cluster's blob store is (its `[blobs]`, or the store `[stores.NAME]` names), as `opened` opens it: a
    directory of files made absolute. Raises `KeyError` for a store the cluster does not name."""
    section = cluster.blobs if store is None else cluster.stores[store]
    settings = dict(section.settings)
    if section.kind == "files":
        directory = Path(str(settings["directory"])).expanduser().absolute()
        return {"kind": FILES, "directory": str(directory)}
    return {"kind": section.kind, **settings}


def for_pods(
    cluster: "Cluster", store: str | None, *, writes: bool, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    """Where a pod finds a store, and the variables it is given with the store's key: the store's own key for a pod
    that writes (a trainer's), its read-only key (`reader`) for one that only reads, where it names one. The key is
    read here (`environ`, by default this process's environment); the location names the variables the pod reads it
    from. Raises `ValueError` where the key is named and not set here."""
    import os

    from rollout_train.cluster import CREDENTIALS

    said = blobs_at(cluster, store)
    section = cluster.blobs if store is None else cluster.stores[store]
    names = dict(section.reader) if not writes and section.reader else {
        key: str(said[key]) for key in CREDENTIALS if isinstance(said.get(key), str)
    }  # fmt: skip
    environment = os.environ if environ is None else environ
    given: dict[str, str] = {}
    for key, variable in names.items():
        value = environment.get(variable, "")
        if not value:
            raise ValueError(f"the key of store {store or 'blobs'} is not set here: {variable}")
        said[key] = variable
        given[variable] = value
    return said, given


@dataclass(frozen=True)
class Stores:
    """The ledger and the blob store, opened, and where the blob store is (`location`: as any process opens it, for a
    run's `starts` record)."""

    ledger: "Ledger"
    blobs: Blobs
    location: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])

    @classmethod
    def open(
        cls, cluster: "Cluster", environ: Mapping[str, str] | None = None, *, store: str | None = None
    ) -> "Stores":
        """The stores a cluster's config names, opened on this node: the ledger from `[ledger]` (its URL read from the
        secret it names, where it names one), the blob store from `[blobs]` (or `[stores.NAME]`, with `store`).
        Raises `ClusterError` where the ledger's URL is not set here or is neither a database's nor the ledger
        service's."""
        ledger = ledger_of(cluster, environ)
        where = blobs_at(cluster, store)
        return cls(ledger, opened(where), where)

    def writing_to(self, cluster: "Cluster", store: str | None) -> "Stores":
        """The same ledger, with blobs written to `store` (`[stores.NAME]`; none: the default)."""
        where = blobs_at(cluster, store)
        return Stores(self.ledger, opened(where), where)

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
