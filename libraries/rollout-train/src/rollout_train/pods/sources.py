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
  platform's Python holds, as constraints: the pod's environment runs what the platform tested;
- **the Python**: the version the pod's environment is made with (`PYTHON`), a uv-managed one, whatever the pod's own.

Where the code comes from:

- **A cluster's own kind** (`[sandboxes.KIND] provider`, no `on_pods.version`): the provider's project, the projects
  it depends on that the platform holds as projects of its own (installed from a directory: the workspace's), and
  `rollout` with its `http` extra, which serves the pool. For `minecraft_horizons.worlds:worlds`: environments/
  minecraft-horizons, environments/minecraft and libraries/rollout.
- **A published version** (`on_pods.version = "NAME@VERSION"`): the version's zip (its `published` record's blob),
  copied into the pods' store, beside `rollout`; its dependencies pinned where the platform holds them, the rest
  resolved by uv.

A source holds nothing secret: provider settings are a cluster config's, which holds no secret values.
"""

import hashlib
import json
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

__all__ = [
    "PYTHON",
    "LocalProject",
    "Project",
    "SandboxSource",
    "local_projects",
    "pinned",
    "sources_in",
    "sources_of",
]

PYTHON = "3.13"
"""The Python a pod's sandbox environments are made with: the platform's own."""
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
    constraints: tuple[str, ...] = ()
    """`NAME==VERSION` pins of what the projects need."""
    python: str = PYTHON
    size: int | None = None
    cpus: float = 1.0
    memory_gib: float = 2.4
    share: float | None = None
    """The part of the pod's spare CPUs and memory it may take (none: what the kinds before it left)."""

    @property
    def digest(self) -> str:
        """What its Python environment is made from, hashed: its projects' contents and extras, its constraints and
        its Python. Sources of the same digest share an environment."""
        made: dict[str, Any] = {
            "projects": [[each.name, each.sha256, sorted(each.extras)] for each in self.projects],
            "constraints": sorted(self.constraints), "python": self.python,
        }  # fmt: skip
        return hashlib.sha256(json.dumps(made, sort_keys=True).encode()).hexdigest()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind, "provider": self.provider, "settings": dict(self.settings),
            "projects": [{"name": each.name, "blob": dict(each.blob), "extras": list(each.extras)}
                         for each in self.projects],
            "constraints": list(self.constraints), "python": self.python, "size": self.size, "cpus": self.cpus,
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
            projects=projects, constraints=tuple(str(each) for each in said.get("constraints") or ()),
            python=str(said.get("python") or PYTHON), size=int(size) if size is not None else None,
            cpus=float(said.get("cpus") or 1.0), memory_gib=float(said.get("memory_gib") or 2.4),
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


def _requirements(name: str, extras: Collection[str]) -> list[Any]:
    """The requirements of the platform's distribution `name` that hold here, with `extras`."""
    from packaging.requirements import Requirement

    try:
        listed = metadata.distribution(name).requires or []
    except metadata.PackageNotFoundError:
        return []
    found: list[Any] = []
    for line in listed:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker is None or any(marker.evaluate({"extra": extra}) for extra in ("", *extras)):
            found.append(requirement)
    return found


def local_projects(provider: str) -> tuple[list[LocalProject], list[str]]:
    """The projects a provider's code is in (its own, those of its dependencies the platform holds as directories, and
    `rollout` with what serves a pool), by name; and pins of every other distribution they need, at the platform's
    versions."""
    own = _project_of_module(provider.partition(":")[0])
    server, extras = SERVER
    projects: dict[str, LocalProject] = {own.name: own}
    pins: dict[str, str] = {}
    wanted: list[tuple[str, tuple[str, ...]]] = [(own.name, ()), (server, extras)]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while wanted:
        name, asked = wanted.pop()
        if (name, asked) in seen:
            continue
        seen.add((name, asked))
        if name not in projects and (directory := _directory_of(name)) is not None:
            projects[name] = LocalProject(name, directory)
        if name in projects and asked:
            projects[name] = LocalProject(name, projects[name].directory,
                                          tuple(sorted({*projects[name].extras, *asked})))  # fmt: skip
        if name not in projects:
            try:
                pins[name] = f"{name}=={metadata.version(name)}"
            except metadata.PackageNotFoundError:
                continue
        for requirement in _requirements(name, asked):
            wanted.append((_canonical(requirement.name), tuple(sorted(requirement.extras))))
    return sorted(projects.values(), key=lambda each: each.name), sorted(pins.values())


def pinned(dependencies: Sequence[str]) -> list[str]:
    """Pins of `dependencies` and what they need, where the platform holds them (each `NAME==VERSION`), through the
    projects the platform holds as directories (which are not pinned)."""
    from packaging.requirements import Requirement

    pins: dict[str, str] = {}
    wanted = [(_canonical(Requirement(line).name), tuple(sorted(Requirement(line).extras))) for line in dependencies]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while wanted:
        name, asked = wanted.pop()
        if (name, asked) in seen:
            continue
        seen.add((name, asked))
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        if _directory_of(name) is None:  # (a project of the platform's own is installed from its zip, not pinned)
            pins[name] = f"{name}=={version}"
        wanted += [(_canonical(each.name), tuple(sorted(each.extras))) for each in _requirements(name, asked)]
    return sorted(pins.values())


async def _stored(blobs: Blobs, directory: Path) -> Mapping[str, JsonValue]:
    """A project's directory, packed as an imported environment is, in the blob store (once per content)."""
    import asyncio

    from rollout_train.publishing import packed

    data = await asyncio.to_thread(packed, directory)
    return (await blobs.put(data, "application/zip")).model_dump(mode="json")


async def sources_of(
    cluster: "Cluster", kinds: Collection[str], blobs: Blobs, versions: "EnvironmentVersions | None" = None
) -> dict[str, SandboxSource]:
    """The sources of the kinds among `kinds` the cluster serves from pods (`on_pods`), their code stored in `blobs`
    (the pods' store), by kind."""
    found: dict[str, SandboxSource] = {}
    for kind in sorted(kinds):
        section = cluster.sandboxes.get(kind)
        if section is None or section.on_pods is None or section.provider is None:
            continue
        on = section.on_pods
        server = next(each for each in local_projects(section.provider)[0] if each.name == SERVER[0])
        if on.version is not None:
            if versions is None:
                raise ValueError(f"[sandboxes.{kind}] on_pods version {on.version}: this ledger keeps no versions")
            version = await versions.get(on.version)
            if version is None:
                raise ValueError(f"[sandboxes.{kind}] on_pods version {on.version}: there is no such version")
            data = await blobs.read(BlobReference.model_validate(version.blob))
            code = (await blobs.put(data, "application/zip")).model_dump(mode="json")
            projects = (Project(SERVER[0], await _stored(blobs, server.directory), SERVER[1]),
                        Project(_canonical(version.name), code))  # fmt: skip
            constraints = tuple(pinned([f"{SERVER[0]}[{','.join(SERVER[1])}]", *version.dependencies]))
        else:
            local, pins = local_projects(section.provider)
            projects = tuple([Project(each.name, await _stored(blobs, each.directory), each.extras) for each in local])
            constraints = tuple(pins)
        found[kind] = SandboxSource(
            kind=kind, provider=section.provider, settings={**section.settings, **on.settings}, projects=projects,
            constraints=constraints, size=on.size, cpus=on.cpus, memory_gib=on.memory_gib, share=on.share,
        )  # fmt: skip
    return found


def sources_in(settings: Mapping[str, JsonValue]) -> dict[str, SandboxSource]:
    """The sources a lease's settings give its pod, by kind (none where they give none)."""
    said = settings.get("sandboxes")
    if not isinstance(said, dict):
        return {}
    return {str(kind): SandboxSource.from_json(cast(dict[str, Any], each)) for kind, each in said.items()}
