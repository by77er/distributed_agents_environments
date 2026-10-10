"""What a pod needs to serve a kind of sandbox, as data: the provider's source (`SandboxSource`), made by the run's
driver and given to the pod in its lease's settings (`settings["sandboxes"]`, by kind), which the pod's sandbox host
reads (`rollout_train.pods.sandboxes`). The pod's image holds no environment's code: it resolves each source into a
Python environment of its own, cached on its volume.

A source is:

- **the provider**, `module:name`, its settings (the `[sandboxes.KIND]` section's, under its `on_pods.settings`), and
  what a pool of it is sized by (`size`, `cpus`, `memory_gib`, `share`);
- **the code**: projects' zips in the pods' blob store, packed as an imported environment is packed
  (`rollout_train.publishing.packed`: the same files are the same bytes, so a zip is stored once per content), each
  installed editable, so a project finds the files beside its package (a plugin's sources, a harness);
- **the requirements**: pins (`NAME==VERSION`) of every other distribution those projects need, at the versions the
  platform's Python holds, installed as they are with nothing resolved (`--no-deps`): the pod's environment runs what
  the platform tested, and nothing is fetched by a name alone;
- **the Python**: the version the pod's environment is made with (`PYTHON`), a uv-managed one, whatever the pod's own.

Where the code comes from (`shipped`): every project a provider needs that the platform holds as a directory of its own
is shipped as a zip, every other distribution is pinned, and one the platform holds neither way is refused: a pod never
installs a distribution by name alone, so a name taken on PyPI cannot stand in for a project of the platform's.

- **A cluster's own kind** (`[sandboxes.KIND] provider`, no `on_pods.version`): the provider's project (found from its
  module), the projects it depends on, and `rollout` with its `http` extra, which serves the pool. For
  `minecraft_horizons.worlds:worlds`: environments/minecraft-horizons, environments/minecraft and libraries/rollout.
- **A published version** (`on_pods.version = "NAME@VERSION"`): the version's zip (its `published` record's blob),
  read from the store versions are published to and put in the pods' store, never the platform's project of its name;
  beside it the projects its dependencies name that the platform holds as directories, and `rollout`. The provider is
  not imported here: the platform need not hold it.

A source holds nothing secret: provider settings are a cluster config's, which holds no secret values.
"""

import hashlib
import json
import logging
import re
import tomllib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs

if TYPE_CHECKING:
    from rollout_train.cluster import Cluster
    from rollout_train.published import EnvironmentVersions

log = logging.getLogger(__name__)

__all__ = [
    "PYTHON",
    "LocalProject",
    "Project",
    "SandboxSource",
    "local_projects",
    "shipped",
    "sources_in",
    "sources_of",
]

PYTHON = "3.13"
"""The Python a pod's sandbox environments are made with: the platform's own."""
KIND = re.compile(r"[a-z0-9-]+")
SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!_-]+")
PYTHON_VERSION = re.compile(r"3\.[0-9]+")
SERVER = ("rollout", ("http",))
"""What serves a pool in its own process (`rollout.harness.pool_server`), and the extra it needs."""


@dataclass(frozen=True)
class Project:
    """A project's zip in the blob store, installed editable with `extras`."""

    name: str
    blob: Mapping[str, JsonValue]
    """Its `BlobReference`, as JSON."""
    extras: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        return str(self.blob["sha256"])


@dataclass(frozen=True)
class SandboxSource:
    """Everything a pod needs to serve a kind of sandbox (the module's docstring)."""

    kind: str
    provider: str
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    projects: tuple[Project, ...] = ()
    pins: tuple[str, ...] = ()
    """`NAME==VERSION` pins of everything else the projects need: installed as they are, nothing resolved."""
    python: str = PYTHON
    size: int | None = None
    cpus: float = 1.0
    memory_gib: float = 1.0
    share: float | None = None
    """The part of the pod's spare CPUs and memory it may take (none: what the kinds before it left)."""

    @property
    def digest(self) -> str:
        """What its Python environment is made from, hashed: its projects' contents and extras, its pins and
        its Python. Sources of the same digest share an environment."""
        made: dict[str, Any] = {
            "projects": [[each.name, each.sha256, sorted(each.extras)] for each in self.projects],
            "pins": sorted(self.pins), "python": self.python,
        }  # fmt: skip
        return hashlib.sha256(json.dumps(made, sort_keys=True).encode()).hexdigest()

    def checked(self) -> None:
        """Raise `ValueError` where what becomes a path or an argument on the pod is not well formed."""
        if not KIND.fullmatch(self.kind):
            raise ValueError(f"{self.kind!r} is not a kind's name")
        for each in self.projects:
            if not SHA256.fullmatch(each.sha256):
                raise ValueError(f"the zip of {each.name} is not named by a SHA-256")
            if not all(NAME.fullmatch(extra) for extra in each.extras):
                raise ValueError(f"the extras of {each.name} are not names")
        if not PYTHON_VERSION.fullmatch(self.python):
            raise ValueError(f"{self.python!r} is not a Python version")
        if not all(PIN.fullmatch(each) for each in self.pins):
            raise ValueError("a pin is not NAME==VERSION")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind, "provider": self.provider, "settings": dict(self.settings),
            "projects": [{"name": each.name, "blob": dict(each.blob), "extras": list(each.extras)}
                         for each in self.projects],
            "pins": list(self.pins), "python": self.python, "size": self.size, "cpus": self.cpus,
            "memory_gib": self.memory_gib, "share": self.share,
        }  # fmt: skip

    @classmethod
    def from_json(cls, said: Mapping[str, Any]) -> "SandboxSource":
        projects = tuple(
            Project(str(each["name"]), dict(each["blob"]), tuple(str(extra) for extra in each.get("extras") or ()))
            for each in said.get("projects") or ()
        )
        size = said.get("size")
        share = said.get("share")
        return cls(
            kind=str(said["kind"]), provider=str(said["provider"]), settings=dict(said.get("settings") or {}),
            projects=projects, pins=tuple(str(each) for each in said.get("pins") or ()),
            python=str(said.get("python") or PYTHON), size=int(size) if size is not None else None,
            cpus=float(said.get("cpus") or 1.0), memory_gib=float(said.get("memory_gib") or 1.0),
            share=float(share) if share is not None else None,
        )  # fmt: skip


@dataclass(frozen=True)
class LocalProject:
    """A project the platform holds as a directory of its own (installed editable from it)."""

    name: str
    directory: Path
    extras: tuple[str, ...] = ()


def _canonical(name: str) -> str:
    from packaging.utils import canonicalize_name

    return str(canonicalize_name(name))


def _directory_of(name: str) -> Path | None:
    """Where the platform's distribution `name` is installed from, if it is a directory (a project of its own)."""
    try:
        said = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not said:
        return None
    url = str(json.loads(said).get("url") or "")
    return Path(url.removeprefix("file://")) if url.startswith("file://") else None


def _project_of_module(module: str) -> LocalProject:
    """The project a module's package is in: the nearest directory above it with a `pyproject.toml`."""
    found = find_spec(module)
    if found is None or found.origin is None:
        raise ValueError(f"the platform does not hold {module}: it cannot pack its project")
    for directory in Path(found.origin).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            said = tomllib.loads((directory / "pyproject.toml").read_text())
            if "workspace" in said.get("tool", {}).get("uv", {}):  # (a workspace's root: no project of its own)
                break
            return LocalProject(_canonical(str(said["project"]["name"])), directory)
    raise ValueError(f"{module} is in no project the platform holds as a directory: it cannot pack it")


def _holding(lines: Collection[str], extras: Collection[str]) -> list[Any]:
    """The requirements among `lines` that hold here, with `extras`."""
    from packaging.requirements import Requirement

    found: list[Any] = []
    for line in lines:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker is None or any(marker.evaluate({"extra": extra}) for extra in ("", *extras)):
            found.append(requirement)
    return found


def _requirements(name: str, extras: Collection[str]) -> list[Any]:
    """The requirements of the platform's distribution `name` that hold here, with `extras`."""
    try:
        listed = metadata.distribution(name).requires or []
    except metadata.PackageNotFoundError:
        return []
    return _holding(listed, extras)


def shipped(
    roots: Sequence[tuple[str, tuple[str, ...]]], own: Mapping[str, Sequence[str]] | None = None
) -> tuple[list[LocalProject], list[str]]:
    """What `roots` (names, and the extras asked of each) need, all of it: the projects the platform holds as
    directories of its own (shipped as zips, by name), and pins (`NAME==VERSION`) of every other distribution, at the
    version the platform holds. `own` names projects shipped from elsewhere (a published version's), with their
    dependencies: those are walked, never replaced by the platform's. Raises `ValueError` for a distribution that is
    needed and the platform holds neither as a project nor installed: it could be neither shipped nor pinned, and
    nothing a pod installs is resolved by name."""
    given = {_canonical(name): list(lines) for name, lines in (own or {}).items()}
    projects: dict[str, LocalProject] = {}
    pins: dict[str, str] = {}
    wanted = [(_canonical(name), tuple(extras)) for name, extras in roots]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while wanted:
        name, asked = wanted.pop()
        if (name, asked) in seen:
            continue
        seen.add((name, asked))
        if name in given:
            needs = _holding(given[name], asked)
        elif (directory := _directory_of(name)) is not None:
            held = projects.get(name)
            projects[name] = LocalProject(name, directory, tuple(sorted({*(held.extras if held else ()), *asked})))
            needs = _requirements(name, asked)
        else:
            try:
                pins[name] = f"{name}=={metadata.version(name)}"
            except metadata.PackageNotFoundError:
                raise ValueError(
                    f"{name} is needed, and the platform holds it neither as a project of its own nor installed: a "
                    "pod could only fetch it by name, which it does not"
                ) from None
            needs = _requirements(name, asked)
        wanted += [(_canonical(each.name), tuple(sorted(each.extras))) for each in needs]
    return sorted(projects.values(), key=lambda each: each.name), sorted(pins.values())


def local_projects(provider: str) -> tuple[list[LocalProject], list[str]]:
    """The projects a cluster's own provider's code is in (its own, found from its module; those of its dependencies
    the platform holds as directories; and `rollout` with what serves a pool), by name; and pins of every other
    distribution they need (`shipped`)."""
    own = _project_of_module(provider.partition(":")[0])
    found, pins = shipped([(own.name, ()), SERVER])
    projects = {each.name: each for each in found}
    projects[own.name] = LocalProject(
        own.name, own.directory, projects[own.name].extras if own.name in projects else ()
    )
    return sorted(projects.values(), key=lambda each: each.name), pins


async def _stored(blobs: Blobs, directory: Path) -> Mapping[str, JsonValue]:
    """A project's directory, packed as an imported environment is, in the blob store (once per content)."""
    import asyncio

    from rollout_train.publishing import packed

    data = await asyncio.to_thread(packed, directory)
    return (await blobs.put(data, "application/zip")).model_dump(mode="json")


async def sources_of(
    cluster: "Cluster",
    kinds: Collection[str],
    blobs: Blobs,
    versions: "EnvironmentVersions | None" = None,
    *,
    published: Blobs | None = None,
) -> dict[str, SandboxSource]:
    """The sources of the kinds among `kinds` the cluster serves from pods (`on_pods`), their code stored in `blobs`
    (the pods' store), by kind. A published version's zip is read from `published` (the store versions are published
    to, the cluster's own; by default `blobs`) and put in `blobs`."""
    found: dict[str, SandboxSource] = {}
    for kind in sorted(kinds):
        section = cluster.sandboxes.get(kind)
        if section is None or section.on_pods is None or section.provider is None:
            continue
        on = section.on_pods
        if on.version is not None:
            if versions is None:
                raise ValueError(f"[sandboxes.{kind}] on_pods version {on.version}: this ledger keeps no versions")
            version = await versions.get(on.version)
            if version is None:
                raise ValueError(f"[sandboxes.{kind}] on_pods version {on.version}: there is no such version")
            data = await (published or blobs).read(BlobReference.model_validate(version.blob))
            code = (await blobs.put(data, "application/zip")).model_dump(mode="json")
            name = _canonical(version.name)
            local, pins = shipped([(name, ()), SERVER], own={name: version.dependencies})
            projects = (
                *[Project(each.name, await _stored(blobs, each.directory), each.extras) for each in local],
                Project(name, code),
            )
        else:
            local, pins = local_projects(section.provider)
            projects = tuple([Project(each.name, await _stored(blobs, each.directory), each.extras) for each in local])
        found[kind] = SandboxSource(
            kind=kind, provider=section.provider, settings={**section.settings, **on.settings}, projects=projects,
            pins=tuple(pins), size=on.size, cpus=on.cpus, memory_gib=on.memory_gib, share=on.share,
        )  # fmt: skip
    return found


def sources_in(settings: Mapping[str, JsonValue]) -> dict[str, SandboxSource]:
    """The sources a lease's settings give its pod, by kind (none where they give none). One that is not well formed (a
    kind not named as a route names it, a zip not named by its SHA-256, an extra that is no name) is left out, and
    said: what a lease says becomes paths on the pod."""
    said = settings.get("sandboxes")
    if not isinstance(said, dict):
        return {}
    found: dict[str, SandboxSource] = {}
    for kind, each in cast(dict[str, Any], said).items():
        try:
            source = SandboxSource.from_json(cast(dict[str, Any], each))
            if source.kind != kind:
                raise ValueError(f"it says it is of the kind {source.kind}")
            source.checked()
        except (ValueError, KeyError, TypeError) as error:
            log.error("the lease gives a source of %s that is not well formed: %s", kind, error)
            continue
        found[kind] = source
    return found
