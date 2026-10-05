"""The cluster config: one TOML file per cluster, the only description of its infrastructure.

It says where the ledger and the blob store are, the node-local scratch directory, the cluster's certificate
authority, the gateway, the monitor, the launcher, the runners and the memory guards; the inference providers and
trainers it offers (`rollout_train.providers`); its sandbox pools, tool sets served elsewhere, the environments it
offers and the Python each runs in; and where roles run (placement) and what bridges need. Nothing about a run is in
it: that is the run's settings (`rollout_train.run_settings`).

A process finds the file (`find`) by `--cluster PATH` or `--cluster NAME` (`~/.config/rollout/clusters/NAME.toml`),
else the `ROLLOUT_CLUSTER` environment variable (a path or a name), else `~/.config/rollout/cluster.toml`. `load`
reads and checks it into a `Cluster`: an unknown key is an error, every kind is known, every trainer's
`colocate_with` names a `vllm` provider, a provider reached with no auth is on this machine.

A secret appears only as a reference (`rollout_train.providers.Secret`): a key `NAME_env` (an environment variable's
name) or `NAME_file` (a file's path), resolved where it is used, at the moment it is needed. A key that looks like a
secret holding a value is refused, and so is a URL with a password in it. So the parsed `Cluster` holds no secret, and
is safe to show and to hand on as JSON (`Cluster.described`, which `parsed` reads back). `inspect` says, on this node,
which secret references resolve (by name, never by value) and which environments' projects have no lock.
"""

import copy
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from pydantic import JsonValue

from rollout_train.providers import (
    INFERENCE_KINDS,
    TRAINER_KINDS,
    Auth,
    InferenceProvider,
    ModelOffer,
    Secret,
    SharedPool,
    Tls,
    TrainerProvider,
    is_local,
)

__all__ = [
    "BlobsSection",
    "BridgeSection",
    "Cluster",
    "ClusterError",
    "EnvironmentSection",
    "GatewaySection",
    "GuardsSection",
    "LauncherSection",
    "LedgerSection",
    "MonitorSection",
    "RaySection",
    "RunnersSection",
    "SandboxesSection",
    "ToolsSection",
    "auth_problem",
    "find",
    "inspect",
    "load",
    "parsed",
]

HOME = Path("~/.config/rollout")
ROLES = ("gateway", "monitor", "launcher", "runners", "pools", "engines", "trainers", "workers", "bridges")
"""The roles `[placement.ROLE]` may steer."""
SCRATCH = "~/.cache/rollout/scratch"
"""Where a node keeps its working files unless the config says (`[scratch] directory`): on disk, since a machine's /tmp
may be memory."""
SECRET_WORDS = ("secret", "password", "token", "credential", "access_key", "api_key", "private_key")
"""A key with one of these in its name holds a secret: it is named (`_env`, `_file`), never written."""


class ClusterError(ValueError):
    """A cluster config that cannot be used, and why."""


@dataclass(frozen=True)
class RaySection:
    address: str = "auto"
    """The head this machine runs (`auto`), or `ray://host:port`."""
    jobs: str = "http://127.0.0.1:8265"
    """The job server."""
    temp_dir: str = "~/.cache/ray"
    """On disk: /tmp may be memory."""
    memory_threshold: float = 0.85
    """Ray's memory monitor kills a task past this share of the machine's memory."""
    python: str = "platform"
    """The interpreter platform actors run in."""


@dataclass(frozen=True)
class LedgerSection:
    url: str | None = None
    """`sqlite:///…` on one machine, `postgresql://…` for several (with no password: that is `url_env`'s)."""
    url_secret: Secret | None = None
    """The URL, named, where it holds a password (`url_env`, `url_file`)."""


@dataclass(frozen=True)
class BlobsSection:
    kind: str = "files"
    """`files`, or `module:name` of the store."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The store's settings (a `directory` for files), none of them a credential: those are the store's own, from its
    environment."""


@dataclass(frozen=True)
class GatewaySection:
    url: str = "http://127.0.0.1:8830"
    """How runners and harnesses reach it."""
    listen: str = "127.0.0.1:8830"
    """`host:port` a replica serves on."""
    replicas: int = 1
    keys: Secret | None = None
    """The secrets keys are signed with (`keys_file`, `keys_env`)."""
    lifetime: float = 21600.0
    """Seconds a key minted for a slot is good for."""


@dataclass(frozen=True)
class MonitorSection:
    listen: str = "127.0.0.1:8765"
    feed_episodes: int = 80
    """Episodes kept in a run's live feed."""


@dataclass(frozen=True)
class LauncherSection:
    at_once: int = 1
    """Runs it plays at once, beside what the cluster's resources allow."""


@dataclass(frozen=True)
class RunnersSection:
    places: int = 8
    """Episodes one runner plays at once."""


@dataclass(frozen=True)
class GuardsSection:
    runs_gib: float = 0.0
    """System memory a node must have free before a runner claims an episode."""
    training_gib: float = 0.0
    """And before a colocated step starts."""


@dataclass(frozen=True)
class SandboxesSection:
    """A pool of sandboxes of one kind, which environments declare they need (`[sandboxes.KIND]`)."""

    kind: str
    provider: str
    """`module:name` of what makes them."""
    python: str = "platform"
    """`platform`, or the name of an environment whose Python the provider is in."""
    size: int = 1
    cpus: float = 1
    memory_gib: float = 1
    pools: int = 1
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The provider's own settings."""


@dataclass(frozen=True)
class ToolsSection:
    """A tool set served elsewhere, by name (`[tools.NAME]`)."""

    name: str
    url: str
    auth: Auth = field(default_factory=Auth)


@dataclass(frozen=True)
class EnvironmentSection:
    """The Python an environment runs in (`[environments."NAME"]`): the platform's, or a uv project's lock built into a
    cached virtualenv."""

    environment: str
    python: str | None = "platform"
    project: str | None = None
    """A uv project's directory (relative paths are from the config file's)."""


@dataclass(frozen=True)
class BridgeSection:
    """What a bridge's task asks for, where the cluster says more than the bridge declares (`[bridges."NAME"]`)."""

    bridge: str
    cpus: float | None = None
    memory_gib: float | None = None


@dataclass(frozen=True)
class Cluster:
    """A cluster, as its config describes it. It holds no secret, only references to secrets."""

    name: str
    """What runs record as where they ran; the Ray namespace is `rollout-NAME`."""
    ledger: LedgerSection
    blobs: BlobsSection = field(default_factory=BlobsSection)
    scratch: str = SCRATCH
    """Node-local: checkpoints in use, fetched bases, bridge work, built Pythons."""
    ray: RaySection = field(default_factory=RaySection)
    tls: Tls | None = None
    gateway: GatewaySection = field(default_factory=GatewaySection)
    monitor: MonitorSection = field(default_factory=MonitorSection)
    launcher: LauncherSection = field(default_factory=LauncherSection)
    runners: RunnersSection = field(default_factory=RunnersSection)
    guards: GuardsSection = field(default_factory=GuardsSection)
    inference: Mapping[str, InferenceProvider] = field(default_factory=dict[str, InferenceProvider])
    trainers: Mapping[str, TrainerProvider] = field(default_factory=dict[str, TrainerProvider])
    sandboxes: Mapping[str, SandboxesSection] = field(default_factory=dict[str, SandboxesSection])
    tools: Mapping[str, ToolsSection] = field(default_factory=dict[str, ToolsSection])
    environments: Mapping[str, EnvironmentSection] = field(default_factory=dict[str, EnvironmentSection])
    placement: Mapping[str, Mapping[str, float]] = field(default_factory=dict[str, Mapping[str, float]])
    """Custom resources each role asks for, by role."""
    bridges: Mapping[str, BridgeSection] = field(default_factory=dict[str, BridgeSection])
    described: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue], repr=False, compare=False)
    """The config as it was read (relative paths made absolute): what is handed on as JSON, and `parsed` reads back."""

    @property
    def namespace(self) -> str:
        return f"rollout-{self.name}"

    def secrets(self) -> dict[str, Secret]:
        """Every secret the config names, by where (`inference.tinker.auth.key`)."""
        found: dict[str, Secret] = {}

        def note(where: str, secret: Secret | None) -> None:
            if secret is not None:
                found[where] = secret

        note("ledger.url", self.ledger.url_secret)
        note("gateway.keys", self.gateway.keys)
        for kind, providers in (("inference", self.inference), ("trainers", self.trainers)):
            for name, provider in providers.items():
                note(f"{kind}.{name}.auth.token", provider.auth.token)
                note(f"{kind}.{name}.auth.key", provider.auth.key)
                for key, secret in provider.secrets.items():
                    note(f"{kind}.{name}.{key}", secret)
        for name, tool in self.tools.items():
            note(f"tools.{name}.auth.token", tool.auth.token)
        return found


def find(given: str | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """The cluster config's file: `given` (`--cluster`: a path, or a name under `~/.config/rollout/clusters`), else
    `ROLLOUT_CLUSTER` (the same), else `~/.config/rollout/cluster.toml`. Raises `ClusterError` where the file it
    names is not there, saying where it looked."""
    environ = os.environ if environ is None else environ
    for said, source in ((given, "--cluster"), (environ.get("ROLLOUT_CLUSTER"), "ROLLOUT_CLUSTER")):
        if said:
            path = _named(said)
            if not path.is_file():
                raise ClusterError(f"{source} names {said!r}, and there is no cluster config at {path}")
            return path
    default = (HOME / "cluster.toml").expanduser()
    if not default.is_file():
        raise ClusterError(
            f"no cluster config: pass --cluster PATH or NAME, set ROLLOUT_CLUSTER, or write {default} "
            "(deploy/clusters/example.toml is one for a single machine)"
        )
    return default


def _named(said: str) -> Path:
    if "/" in said or said.endswith(".toml"):
        return Path(said).expanduser()
    return (HOME / "clusters" / f"{said}.toml").expanduser()


def load(path: Path) -> Cluster:
    """The cluster a config file describes, checked. Raises `ClusterError` saying what is wrong and where."""
    try:
        described = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ClusterError(f"{path}: {error}") from error
    return parsed(described, relative_to=path.parent)


def parsed(described: Mapping[str, Any], *, relative_to: Path | None = None) -> Cluster:
    """The cluster a config's table describes, checked; relative paths of environments' projects are from
    `relative_to`."""
    table = _Table(copy.deepcopy(dict(described)), "the cluster config")  # (reading takes keys: not from the caller's)
    _refuse_secret_values(described, "")
    name = table.text("name")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ClusterError(f"name is lowercase letters, digits and '-', for the Ray namespace (not {name!r})")
    ledger = table.section("ledger")
    url, url_secret = ledger.text("url", None), ledger.secret("url")
    if (url is None) == (url_secret is None):
        raise ClusterError("[ledger] has url or url_env (or url_file): one of them")
    if url is not None and urlsplit(str(url)).password:
        raise ClusterError("[ledger] url holds a password: put the URL in an environment variable and name it url_env")
    ledger.done()
    blobs = table.section("blobs")
    blob_kind = blobs.text("kind", "files")
    blob_settings = blobs.rest()
    if blob_kind == "files" and set(blob_settings) - {"directory"}:
        raise ClusterError(f"[blobs] of files has only a directory (not {', '.join(sorted(blob_settings))})")
    if blob_kind != "files" and ":" not in blob_kind:
        raise ClusterError(f"[blobs] kind is files or module:name, not {blob_kind!r}")
    if blob_kind == "files":
        blob_settings.setdefault("directory", "~/.cache/rollout/blobs")
    scratch = table.section("scratch")
    scratch_directory = scratch.text("directory", SCRATCH)
    scratch.done()
    ray = table.section("ray")
    ray_said = RaySection(
        address=ray.text("address", "auto"),
        jobs=ray.text("jobs", RaySection.jobs),
        temp_dir=ray.text("temp_dir", RaySection.temp_dir),
        memory_threshold=ray.number("memory_threshold", RaySection.memory_threshold),
        python=ray.text("python", "platform"),
    )
    ray.done()
    tls: Tls | None = None
    if "tls" in table.table:
        said = table.section("tls")
        tls = Tls(
            ca=said.text("ca", None),
            certificate=said.text("certificate", None),
            key=said.text("key", None),
            identity=said.text("identity", Tls.identity),
        )
        said.done()
    gateway = table.section("gateway")
    gateway_said = GatewaySection(
        url=gateway.text("url", GatewaySection.url),
        listen=gateway.text("listen", GatewaySection.listen),
        replicas=gateway.whole("replicas", 1, least=1),
        keys=gateway.secret("keys"),
        lifetime=gateway.number("lifetime", GatewaySection.lifetime),
    )
    gateway.done()
    monitor = table.section("monitor")
    monitor_said = MonitorSection(
        monitor.text("listen", MonitorSection.listen), monitor.whole("feed_episodes", 80, least=1)
    )
    monitor.done()
    launcher = table.section("launcher")
    launcher_said = LauncherSection(launcher.whole("at_once", 1, least=1))
    launcher.done()
    runners = table.section("runners")
    runners_said = RunnersSection(places=runners.whole("places", 8, least=1))
    runners.done()
    guards = table.section("guards")
    guards_said = GuardsSection(guards.number("runs_gib", 0.0), guards.number("training_gib", 0.0))
    guards.done()
    inference = {name: _inference(name, each, tls) for name, each in table.tables("inference").items()}
    trainers = {name: _trainer(name, each) for name, each in table.tables("trainers").items()}
    for trainer in trainers.values():
        if trainer.colocate_with is not None:
            shared = inference.get(trainer.colocate_with)
            if shared is None or shared.kind != "vllm":
                raise ClusterError(
                    f"[trainers.{trainer.name}] colocate_with names {trainer.colocate_with!r}, which is not a vllm "
                    "provider of this cluster: a trainer shares only the GPU of engines this cluster starts"
                )
    sandboxes = {kind: _sandboxes(kind, each) for kind, each in table.tables("sandboxes").items()}
    tools = {name: _tool(name, each) for name, each in table.tables("tools").items()}
    environments: dict[str, EnvironmentSection] = {}
    for environment, each in table.tables("environments").items():
        said = _Table(each, f'[environments."{environment}"]')
        project = said.text("project", None)
        python = said.text("python", None if project else "platform")
        said.done()
        if (python is None) == (project is None):
            raise ClusterError(f'[environments."{environment}"] has python = "platform" or a project, one of them')
        if python not in (None, "platform"):
            raise ClusterError(f'[environments."{environment}"] python is "platform" (or say a project)')
        if project is not None:
            where = Path(project).expanduser()
            if not where.is_absolute() and relative_to is not None:
                where = relative_to / where
            project = str(where)
        environments[environment] = EnvironmentSection(environment, python, project)
    for kind, pool in sandboxes.items():
        if pool.python not in ("platform", *environments):
            raise ClusterError(f"[sandboxes.{kind}] python is platform or an environment of this cluster")
    placement: dict[str, Mapping[str, float]] = {}
    for role, each in table.tables("placement").items():
        if role not in ROLES:
            raise ClusterError(f"[placement.{role}]: the roles are {', '.join(ROLES)}")
        said = _Table(each, f"[placement.{role}]")
        resources = said.take("resources", {})
        said.done()
        numbers = _numbers(resources)
        if numbers is None:
            raise ClusterError(f"[placement.{role}] resources is a table of numbers")
        placement[role] = numbers
    bridges: dict[str, BridgeSection] = {}
    for bridge, each in table.tables("bridges").items():
        said = _Table(each, f'[bridges."{bridge}"]')
        bridges[bridge] = BridgeSection(bridge, said.number("cpus", None), said.number("memory_gib", None))
        said.done()
    table.done()
    return Cluster(
        name=name,
        ledger=LedgerSection(url, url_secret),
        blobs=BlobsSection(blob_kind, blob_settings),
        scratch=scratch_directory,
        ray=ray_said,
        tls=tls,
        gateway=gateway_said,
        monitor=monitor_said,
        launcher=launcher_said,
        runners=runners_said,
        guards=guards_said,
        inference=inference,
        trainers=trainers,
        sandboxes=sandboxes,
        tools=tools,
        environments=environments,
        placement=placement,
        bridges=bridges,
        described=_handed_on(described, environments),
    )


def auth_problem(where: str, auth: Auth, endpoints: Sequence[str]) -> str | None:
    """Why a provider reached as `auth` at `endpoints` may not be, if it may not: with no auth, only on this
    machine."""
    if auth.kind == "none" and (away := [each for each in endpoints if not is_local(each)]):
        return (
            f"{where} is reached with no auth at {', '.join(away)}, which is not this machine: auth none is only for "
            "localhost; say auth = mtls or bearer"
        )
    return None


def _inference(name: str, described: dict[str, Any], tls: Tls | None) -> InferenceProvider:
    where = f"[inference.{name}]"
    said = _Table(described, where)
    kind_name = said.text("kind")
    kind = INFERENCE_KINDS.get(kind_name)
    if kind is None:
        raise ClusterError(f"{where} kind is one of {', '.join(INFERENCE_KINDS)}, not {kind_name!r}")
    auth = _auth(said, where, kind.auths, kind.auth)
    if auth.kind == "mtls" and (tls is None or tls.ca is None or tls.certificate is None):
        raise ClusterError(f"{where} is reached over mutual TLS: the cluster needs [tls] ca, certificate and key")
    gpus = said.number("gpus", 1 if kind_name == "vllm" else 0)
    replicas = said.whole("replicas", 1, least=1)
    models: dict[str, ModelOffer] = {}
    for model, each in said.tables("models").items():
        offer = _Table(each, f'{where}.models."{model}"')
        options = offer.take("options", {})
        if not isinstance(options, dict):
            raise ClusterError(f'{where}.models."{model}" options is a table')
        options = cast(dict[str, JsonValue], options)
        if "max_logprobs" in options:
            raise ClusterError(
                f'{where}.models."{model}" options: max_logprobs is the provider\'s ({where} max_logprobs)'
            )
        cost = _numbers(offer.take("cost", {}))
        if cost is None:
            raise ClusterError(f'{where}.models."{model}" cost is a table of dollars per million tokens (or per hour)')
        rank = options.get("max_lora_rank")
        models[model] = ModelOffer(
            model=model,
            context=offer.whole("context", least=1),
            base=offer.text("base", None),
            max_lora_rank=int(rank) if isinstance(rank, int) else None,
            cost=cost,
            options=options,
        )
        offer.done()
    if not models:
        raise ClusterError(f'{where} offers no model: add [inference.{name}.models."MODEL"] with its context')
    secrets = {each: secret for each in kind.secrets if (secret := said.secret(each)) is not None}
    pool: SharedPool | None = None
    if "pool" in said.table:
        if not kind.shared:
            raise ClusterError(f"{where}: a {kind_name} provider is not a pool runs share")
        shape = _Table(said.take("pool"), f"{where} pool")
        pool = SharedPool(shape.whole("adapter_slots", None, least=1), shape.whole("max_runs", None, least=1))
        shape.done()
    elif kind.shared:
        pool = SharedPool()
    capabilities = kind.capabilities
    if "max_logprobs" in said.table:
        capabilities = replace(capabilities, top_logprobs=said.whole("max_logprobs", least=0))
    settings = said.rest()
    if unknown := sorted(set(settings) - set(kind.fields)):
        fields = ("kind", "auth", "gpus", "replicas", "models", *kind.fields)
        raise ClusterError(f"{where} has no {', '.join(unknown)} (a {kind_name} provider has {', '.join(fields)})")
    endpoints: tuple[str, ...] = ()
    if kind_name == "vllm":
        endpoints = (str(settings.get("listen", "127.0.0.1")),)
    elif kind_name == "vllm-servers":
        addresses = settings.get("addresses")
        if not isinstance(addresses, list) or not addresses or not all(isinstance(each, str) for each in addresses):
            raise ClusterError(f"{where} addresses is a list of its servers' URLs")
        endpoints = tuple(str(each) for each in addresses) + ((str(settings["via"]),) if "via" in settings else ())
    elif kind_name == "api" and "endpoint" not in settings:
        raise ClusterError(f'{where} names its endpoint (endpoint = "module:name")')
    if (problem := auth_problem(where, auth, endpoints)) is not None:
        raise ClusterError(problem)
    return InferenceProvider(
        name, kind_name, capabilities, models, auth, gpus, replicas, pool, endpoints, settings, secrets
    )


def _trainer(name: str, described: dict[str, Any]) -> TrainerProvider:
    where = f"[trainers.{name}]"
    said = _Table(described, where)
    kind_name = said.text("kind")
    kind = TRAINER_KINDS.get(kind_name)
    if kind is None:
        hint = " (a trainer is named by its kind, not module:name)" if ":" in kind_name else ""
        raise ClusterError(f"{where} kind is one of {', '.join(TRAINER_KINDS)}, not {kind_name!r}{hint}")
    auth = _auth(said, where, kind.auths, kind.auth)
    models: object = said.take("models", None)
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(each, str) for each in cast(list[object], models))
    ):
        raise ClusterError(f"{where} models is a list of the models it trains here")
    cost = _numbers(said.take("cost", {}))
    if cost is None:
        raise ClusterError(f"{where} cost is a table of dollars per million tokens trained (train) or per hour")
    costs: dict[str, Mapping[str, float]] = {}
    said_costs: object = said.take("costs", {})
    if not isinstance(said_costs, dict):
        raise ClusterError(f"{where} costs is a table of each model's cost")
    for model, priced in cast(dict[str, object], said_costs).items():
        numbers = _numbers(priced)
        if numbers is None or model not in cast(list[str], models):
            raise ClusterError(f'{where}.costs."{model}" is a model it trains, priced as cost is')
        costs[model] = numbers
    secrets = {each: secret for each in kind.secrets if (secret := said.secret(each)) is not None}
    provider = TrainerProvider(
        name=name,
        kind=kind_name,
        capabilities=kind.capabilities,
        models=tuple(str(each) for each in cast(list[str], models)),
        auth=auth,
        segment_tokens=said.whole("segment_tokens", None, least=1),
        gpus=said.number("gpus", 0.0),
        colocate_with=said.text("colocate_with", None),
        cost=cost,
        costs=costs,
        secrets=secrets,
    )
    settings = said.rest()
    if unknown := sorted(set(settings) - set(kind.fields)):
        fields = ("kind", "auth", "models", "segment_tokens", "gpus", "colocate_with", "cost", "costs", *kind.fields)
        raise ClusterError(f"{where} has no {', '.join(unknown)} (a {kind_name} trainer has {', '.join(fields)})")
    if kind_name == "runpod-trainer":
        runs = settings.get("trainer", "lora")
        if runs not in ("lora", "full"):
            raise ClusterError(f"{where} trainer is lora or full, the trainer its pods run (not {runs!r})")

    provider = replace(provider, settings=settings)
    return replace(provider, capabilities=provider.runs.capabilities)


def _sandboxes(kind: str, described: dict[str, Any]) -> SandboxesSection:
    said = _Table(described, f"[sandboxes.{kind}]")
    return SandboxesSection(
        kind=kind,
        provider=said.text("provider"),
        python=said.text("python", "platform"),
        size=said.whole("size", 1, least=1),
        cpus=said.number("cpus", 1.0),
        memory_gib=said.number("memory_gib", 1.0),
        pools=said.whole("pools", 1, least=1),
        settings=said.rest(),
    )


def _tool(name: str, described: dict[str, Any]) -> ToolsSection:
    where = f"[tools.{name}]"
    said = _Table(described, where)
    url = said.text("url")
    auth = _auth(said, where, ("none", "bearer", "mtls"), Auth("none"))
    said.done()
    if (problem := auth_problem(where, auth, (url,))) is not None:
        raise ClusterError(problem)
    return ToolsSection(name, url, auth)


def _auth(said: "_Table", where: str, allowed: Sequence[str], default: Auth | None) -> Auth:
    given = said.take("auth", None)
    if given is None:
        if default is None:
            raise ClusterError(f"{where} says how it is reached: auth = one of {', '.join(allowed)}")
        return default
    if not isinstance(given, str | dict):
        raise ClusterError(f"{where} auth is a kind ({', '.join(allowed)}) or a table with one")
    table = _Table({"kind": given} if isinstance(given, str) else dict(cast(dict[str, Any], given)), f"{where} auth")
    kind = table.text("kind")
    if kind not in allowed:
        raise ClusterError(f"{where} auth is one of {', '.join(allowed)}, not {kind!r}")
    try:
        auth = Auth(
            kind=kind,
            token=table.secret("token"),
            key=table.secret("key") or (default.key if default is not None and kind == default.kind else None),
            trust=table.text("trust", None),
            identity=table.text("identity", default.identity if default is not None and kind == default.kind else None),
        )
    except ValueError as error:
        raise ClusterError(f"{where} auth: {error}") from error
    table.done()
    return auth


def _refuse_secret_values(table: Mapping[str, Any], where: str) -> None:
    for key, value in table.items():
        here = f"{where}.{key}" if where else key
        if isinstance(value, dict):
            if key != "secrets":  # (a RunPod provider's console secrets, by name)
                _refuse_secret_values(cast(dict[str, Any], value), here)
            continue
        named = key.endswith(("_env", "_file"))
        if not named and isinstance(value, str) and any(word in key.lower() for word in SECRET_WORDS):
            raise ClusterError(
                f"{here} looks like a secret written down: name it instead, {key}_env (an environment variable) or "
                f"{key}_file (a file)"
            )


def _handed_on(described: Mapping[str, Any], environments: Mapping[str, EnvironmentSection]) -> dict[str, Any]:
    """The config as JSON, with each environment's project as it was resolved."""
    handed: dict[str, Any] = _json(described)
    for name, python in environments.items():
        if python.project is not None:
            handed["environments"][name]["project"] = python.project
    return handed


def _numbers(value: object) -> dict[str, float] | None:
    """A table of numbers, as one; none for anything else."""
    if not isinstance(value, dict):
        return None
    table = cast(dict[str, object], value)
    if not all(_is_number(each) for each in table.values()):
        return None
    return {str(key): float(cast(float, each)) for key, each in table.items()}


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(each) for key, each in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_json(each) for each in cast(list[Any], value)]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)  # (TOML's dates and times)


_REQUIRED: Any = object()


class _Table:
    """A table of the config, read key by key: each read takes its key, and `done` refuses what is left."""

    def __init__(self, table: dict[str, Any], where: str) -> None:
        self.table = table
        self.where = where

    def take(self, key: str, default: Any = _REQUIRED) -> Any:
        if key in self.table:
            return self.table.pop(key)
        if default is _REQUIRED:
            raise ClusterError(f"{self.where} needs {key}")
        return default

    def text(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self.take(key, default)
        if value is not None and value is not default and not isinstance(value, str):
            raise ClusterError(f"{self.where} {key} is text, not {value!r}")
        return value

    def number(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self.take(key, default)
        if value is not None and value is not default and not _is_number(value):
            raise ClusterError(f"{self.where} {key} is a number, not {value!r}")
        return float(value) if value is not None else None

    def whole(self, key: str, default: Any = _REQUIRED, *, least: int | None = None) -> Any:
        value = self.take(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or (least is not None and value < least):
            floor = f", {least} at least" if least is not None else ""
            raise ClusterError(f"{self.where} {key} is a whole number{floor}, not {value!r}")
        return value

    def flag(self, key: str, default: bool) -> bool:
        value = self.take(key, default)
        if not isinstance(value, bool):
            raise ClusterError(f"{self.where} {key} is true or false, not {value!r}")
        return value

    def secret(self, key: str) -> Secret | None:
        env, file = self.take(f"{key}_env", None), self.take(f"{key}_file", None)
        if env is not None and file is not None:
            raise ClusterError(f"{self.where} names {key} once: {key}_env or {key}_file")
        if env is None and file is None:
            return None
        if not isinstance(env or file, str):
            raise ClusterError(f"{self.where} {key}_env or {key}_file is a name, not a value")
        return Secret(env=env, file=file)

    def section(self, key: str) -> "_Table":
        value = self.take(key, {})
        if not isinstance(value, dict):
            raise ClusterError(f"[{key}] is a table")
        return _Table(dict(cast(dict[str, Any], value)), f"[{key}]")

    def tables(self, key: str) -> dict[str, dict[str, Any]]:
        value = self.take(key, {})
        if not isinstance(value, dict) or not all(
            isinstance(each, dict) for each in cast(dict[str, object], value).values()
        ):
            raise ClusterError(f"[{key}.NAME] are tables")
        return {str(name): dict(each) for name, each in cast(dict[str, dict[str, Any]], value).items()}

    def rest(self) -> dict[str, JsonValue]:
        rest = dict(self.table)
        self.table.clear()
        return rest

    def done(self) -> None:
        if self.table:
            raise ClusterError(f"{self.where} has no {', '.join(sorted(self.table))}")


def inspect(cluster: Cluster, environ: Mapping[str, str] | None = None) -> list[str]:
    """What is wrong with the cluster on this node, in words: each secret reference that does not resolve (by name,
    never by value), and each environment's project with no `uv.lock`. Empty: nothing."""
    problems: list[str] = []
    for where, secret in cluster.secrets().items():
        if secret.resolve(environ) is None:
            problems.append(f"{where}: {secret} is not set on this node")
    for name, python in cluster.environments.items():
        if python.project is not None and not (Path(python.project) / "uv.lock").is_file():
            problems.append(f'environments."{name}": {python.project} has no uv.lock to build its Python from')
    return problems
