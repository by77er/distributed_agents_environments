"""The cluster config: one TOML file per cluster, the only description of its infrastructure.

It says where the ledger and the blob store are, the node-local scratch directory, the cluster's certificate
authority, the Ray cluster runs' jobs are submitted to (or, with `[kubernetes]`, the RayJob each run's job is made
from), the gateway, the monitor, the runners and the memory guards; the inference providers and trainers it offers
(`rollout_train.providers`); its sandbox pools, tool sets served elsewhere, the environments it offers and the Python
each runs in; where roles run (placement), what bridges need, and the most it schedules for one run (capacity).
Nothing about a run is in it: that is the run's settings (`rollout_train.run_settings`).

A process finds the file (`find`) by `--cluster PATH` or `--cluster NAME` (`~/.config/rollout/clusters/NAME.toml`),
else the `ROLLOUT_CLUSTER` environment variable (a path or a name), else `~/.config/rollout/cluster.toml`. A run's job
is handed the config it was submitted with as JSON (`ROLLOUT_CLUSTER_JSON`, which `located` reads first). `load`
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
    ALLOCATIONS,
    INFERENCE_KINDS,
    POD_FIELDS,
    RUNPOD,
    TRAINER_KINDS,
    Allocation,
    Auth,
    InferenceProvider,
    ModelOffer,
    Secret,
    Tls,
    TrainerProvider,
    is_local,
    pod_table,
)

__all__ = [
    "BlobsSection",
    "BridgeSection",
    "CapacitySection",
    "Cluster",
    "ClusterError",
    "EnvironmentSection",
    "GatewaySection",
    "GuardsSection",
    "KubernetesSection",
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
    "located",
    "parsed",
]

HOME = Path("~/.config/rollout")
ROLES = ("gateway", "monitor", "runners", "pools", "engines", "trainers", "workers", "bridges")
"""The roles `[placement.ROLE]` may steer."""
SECRET_ROLES = ("run", "gateway", "monitor", "ledger", "pool", "reaper")
"""The roles whose secrets `Cluster.secrets_of` says: a run's job, the gateway, a monitor, the ledger service, a sandbox
pool served from a pod of its own, and the reaper of RunPod's pods."""
KEYED_BY_THE_GATEWAY = ("api", "vllm", "vllm-servers")
"""The kinds of inference provider the gateway samples with a key or a token of theirs
(`rollout_train.gateway.directory`); it reaches RunPod's pods, which it samples too, with its certificate."""
SCRATCH = "~/.cache/rollout/scratch"
"""Where a node keeps its working files unless the config says (`[scratch] directory`): on disk, since a machine's /tmp
may be memory."""
HANDED = "ROLLOUT_CLUSTER_JSON"
"""The environment variable a run's job is handed its cluster config in, as JSON (`Cluster.described`)."""
SECRET_WORDS = ("secret", "password", "token", "credential", "access_key", "api_key", "private_key")
"""A key with one of these in its name holds a secret: it is named (`_env`, `_file`), never written."""


class ClusterError(ValueError):
    """A cluster config that cannot be used, and why."""


@dataclass(frozen=True)
class RaySection:
    address: str = "auto"
    """The Ray cluster's address (its GCS, `host:port`), which a run's driver joins; `auto`: the one Ray finds (in a
    job, the cluster the job runs on)."""
    jobs: str = "http://127.0.0.1:8265"
    """The job server."""
    temp_dir: str = "~/.cache/ray"
    """On disk: /tmp may be memory."""
    memory_threshold: float = 0.85
    """Ray's memory monitor kills a task past this share of the machine's memory."""
    python: str = "platform"
    """The interpreter a run's job starts in: `platform`, the `python` on the job's `PATH` (the platform's), or a
    path."""


@dataclass(frozen=True)
class KubernetesSection:
    """Where each run's job is a RayJob with a Ray cluster of its own (`[kubernetes]`): the namespace RayJobs are made
    in, the template each is made from (a RayJob's YAML: its Ray cluster, image, volumes and retries; relative paths
    are from the config file's), and the API server (by default the one of the cluster the process runs in, reached
    with its service account)."""

    namespace: str
    rayjob: str
    api: str = "https://kubernetes.default.svc"
    queue: str | None = None
    """The Kueue LocalQueue in `namespace` that admits runs' RayJobs: each is made suspended, with the label
    `kueue.x-k8s.io/queue-name`, and starts when Kueue admits it whole. None: each starts when it is made."""


@dataclass(frozen=True)
class LedgerSection:
    url: str | None = None
    """`sqlite:///…` on one machine, `postgresql://…` for several (with no password: that is `url_env`'s), or the
    ledger service's `http(s)://…` (`rollout_train.ledger_service.HttpLedger`, with `token`)."""
    url_secret: Secret | None = None
    """The URL, named, where it holds a password (`url_env`, `url_file`)."""
    token: Secret | None = None
    """The platform's token for the ledger service (`token_env`, `token_file`): what its roles send when `url` is the
    service's, what the service takes as the platform's, and what pods' tokens are signed with."""
    public: str | None = None
    """Where processes outside the cluster (RunPod's pods) reach the ledger service: `https://…`."""


@dataclass(frozen=True)
class BlobsSection:
    """A blob store: the cluster's default (`[blobs]`), or another, by name (`[stores.NAME]`)."""

    kind: str = "files"
    """`files`, or `module:name` of the store."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The store's settings (a `directory` for files), none of them a credential: those come from its environment, or
    from the variables it names (`access_key_id_env`, `secret_access_key_env`)."""
    reader: Mapping[str, str] = field(default_factory=dict[str, str])
    """The variables a read-only key is read from (`access_key_id_env`, `secret_access_key_env`), for whoever only
    reads it (inference pods): a named store's `reader`."""


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
    token: Secret | None = None
    """The monitor's token (`token_env`, `token_file`), which its page and every client of its API present
    (`rollout_train.monitor.access`); none: `ROLLOUT_MONITOR_TOKEN`."""


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
    """A pool of sandboxes of one kind, which environments declare they need (`[sandboxes.KIND]`): made in each run's
    driver from its provider, or, with `url`, served elsewhere (`rollout pool --kind KIND`), where runs reach it. What
    its sandboxes run and hold is the pool's business: a run's demand counts none of it."""

    kind: str
    provider: str | None = None
    """`module:name` of what makes them."""
    python: str = "platform"
    """`platform`, or the name of an environment whose Python the provider is in."""
    size: int = 1
    url: str | None = None
    """Where the pool is served (`rollout.harness.remote.serve_pool`): runs acquire from it there."""
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
    interpreter: str | None = None
    """The interpreter a run on it starts in, where it is not the platform's: by default a project's
    `PROJECT/.venv/bin/python`."""

    @property
    def runs_in(self) -> str | None:
        """The interpreter a run on it starts in; none: the platform's."""
        if self.interpreter is not None:
            return self.interpreter
        return str(Path(self.project) / ".venv" / "bin" / "python") if self.project is not None else None


@dataclass(frozen=True)
class BridgeSection:
    """What a bridge's task asks for, where the cluster says more than the bridge declares (`[bridges."NAME"]`)."""

    bridge: str
    cpus: float | None = None
    memory_gib: float | None = None


@dataclass(frozen=True)
class CapacitySection:
    """The most the cluster can schedule for one run (`[capacity]`): on Kubernetes with Kueue, its queue's quota. A run
    whose demand exceeds it is refused (`rollout_train.demand`). Each is unbounded where it is not said."""

    cpus: float | None = None
    memory_gib: float | None = None
    gpus: float | None = None


@dataclass(frozen=True)
class Cluster:
    """A cluster, as its config describes it. It holds no secret, only references to secrets."""

    name: str
    """What runs record as where they ran; the Ray namespace is `rollout-NAME`."""
    ledger: LedgerSection
    blobs: BlobsSection = field(default_factory=BlobsSection)
    stores: Mapping[str, BlobsSection] = field(default_factory=dict[str, BlobsSection])
    """Blob stores beside the default, by name (`[stores.NAME]`): an R2 bucket that RunPod's pods reach, say."""
    scratch: str = SCRATCH
    """Node-local: checkpoints in use, fetched bases, bridge work, built Pythons."""
    ray: RaySection = field(default_factory=RaySection)
    kubernetes: KubernetesSection | None = None
    tls: Tls | None = None
    gateway: GatewaySection = field(default_factory=GatewaySection)
    monitor: MonitorSection = field(default_factory=MonitorSection)
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
    capacity: CapacitySection | None = None
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
        note("ledger.token", self.ledger.token)
        for name, store in (("blobs", self.blobs), *((f"stores.{each}", said) for each, said in self.stores.items())):
            for key in CREDENTIALS:
                if isinstance(env := store.settings.get(key), str):
                    note(f"{name}.{key.removesuffix('_env')}", Secret(env=env))
                if key in store.reader:
                    note(f"{name}.reader.{key.removesuffix('_env')}", Secret(env=store.reader[key]))
        note("gateway.keys", self.gateway.keys)
        note("monitor.token", self.monitor.token)
        for kind, providers in (("inference", self.inference), ("trainers", self.trainers)):
            for name, provider in providers.items():
                note(f"{kind}.{name}.auth.token", provider.auth.token)
                note(f"{kind}.{name}.auth.key", provider.auth.key)
                for key, secret in provider.secrets.items():
                    note(f"{kind}.{name}.{key}", secret)
        for name, tool in self.tools.items():
            note(f"tools.{name}.auth.token", tool.auth.token)
        return found

    def secrets_of(self, role: str = "run") -> dict[str, Secret]:
        """The secrets a role reads (`SECRET_ROLES`), by where, as `secrets` says them: a run's job reads every one but
        the monitor's token; the gateway the ledger's, its keys and those of the providers it samples (a hosted API's
        key, a server's token); a monitor the ledger's, the blob stores' (but their read-only keys, which only pods
        read) and its own token; the ledger service the ledger's and the platform's token; a sandbox pool the ledger's;
        the reaper the ledger's and RunPod's keys. What each role is given on Kubernetes is this
        (docs/deploy/helm.md)."""
        if role not in SECRET_ROLES:
            raise ValueError(f"a role is one of {', '.join(SECRET_ROLES)}, not {role!r}")
        every = self.secrets()
        if role == "run":
            return {where: secret for where, secret in every.items() if where != "monitor.token"}

        def provider(where: str) -> InferenceProvider | TrainerProvider | None:
            kind, _, rest = where.partition(".")
            name = rest.split(".")[0]
            if kind == "inference":
                return self.inference.get(name)
            return self.trainers.get(name) if kind == "trainers" else None

        def wanted(where: str) -> bool:
            if where == "ledger.url":
                return True
            if role == "ledger":
                return where == "ledger.token"
            if role == "gateway":
                found = provider(where) if where.startswith("inference.") else None
                keyed = found is not None and found.kind in KEYED_BY_THE_GATEWAY
                return where == "gateway.keys" or (where.startswith("blobs.") and ".reader." not in where) or keyed
            if role == "monitor":
                stored = where.startswith(("blobs.", "stores.")) and ".reader." not in where
                return stored or where == "monitor.token"
            if role == "reaper":
                found = provider(where)
                return found is not None and found.kind in RUNPOD and where.endswith(".api_key")
            return False  # (a pool: the ledger's alone)

        return {where: secret for where, secret in every.items() if wanted(where)}


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


def located(given: str | None = None, environ: Mapping[str, str] | None = None) -> Cluster:
    """The cluster config this process works with: the one its job was handed (`ROLLOUT_CLUSTER_JSON`), else the file
    `find` finds, read and checked. Raises `ClusterError` saying what is wrong."""
    import json

    environ = os.environ if environ is None else environ
    if given is None and (handed := environ.get(HANDED)):
        try:
            return parsed(json.loads(handed))
        except json.JSONDecodeError as error:
            raise ClusterError(f"{HANDED} is not JSON: {error}") from error
    return load(find(given, environ))


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
    ledger_token, public = ledger.secret("token"), ledger.text("public", None)
    if url is not None and str(url).startswith(("http://", "https://")) and ledger_token is None:
        raise ClusterError("[ledger] url is the ledger service's: name the platform's token, token_env or token_file")
    if public is not None and not str(public).startswith(("http://", "https://")):
        raise ClusterError(f"[ledger] public is the ledger service's address outside the cluster, http(s)://… (not "
                           f"{public!r})")  # fmt: skip
    ledger.done()
    blobs_said = _blobs(table.section("blobs"), "[blobs]")
    stores = {name: _blobs(_Table(each, f"[stores.{name}]"), f"[stores.{name}]", named=True)
              for name, each in table.tables("stores").items()}  # fmt: skip
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
    kubernetes: KubernetesSection | None = None
    if "kubernetes" in table.table:
        said = table.section("kubernetes")
        template = Path(said.text("rayjob")).expanduser()
        if not template.is_absolute() and relative_to is not None:
            template = relative_to / template
        kubernetes = KubernetesSection(
            namespace=said.text("namespace"),
            rayjob=str(template),
            api=said.text("api", KubernetesSection.api),
            queue=said.text("queue", None),
        )
        said.done()
    capacity: CapacitySection | None = None
    if "capacity" in table.table:
        said = table.section("capacity")
        capacity = CapacitySection(
            said.number("cpus", None), said.number("memory_gib", None), said.number("gpus", None)
        )
        said.done()
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
        monitor.text("listen", MonitorSection.listen), monitor.whole("feed_episodes", 80, least=1),
        monitor.secret("token"),
    )  # fmt: skip
    monitor.done()
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
            if trainer.kind == "runpod-trainer" and (shared is None or shared.kind != "runpod-host"):
                raise ClusterError(
                    f"[trainers.{trainer.name}] colocate_with names {trainer.colocate_with!r}, which is not a "
                    "runpod-host provider of this cluster: a runpod-trainer takes its steps on a host's pods"
                )
            if trainer.kind != "runpod-trainer" and (shared is None or shared.kind != "vllm"):
                raise ClusterError(
                    f"[trainers.{trainer.name}] colocate_with names {trainer.colocate_with!r}, which is not a vllm "
                    "provider of this cluster: a trainer shares only the GPU of engines this cluster starts"
                )
    for where, provider in [*(("inference", each) for each in inference.values()),
                            *(("trainers", each) for each in trainers.values())]:  # fmt: skip
        store = provider.settings.get("store")
        if isinstance(store, str) and store not in stores:
            raise ClusterError(f"[{where}.{provider.name}] store names {store!r}, which is no [stores.NAME] here")
    sandboxes = {kind: _sandboxes(kind, each) for kind, each in table.tables("sandboxes").items()}
    tools = {name: _tool(name, each) for name, each in table.tables("tools").items()}
    environments: dict[str, EnvironmentSection] = {}
    for environment, each in table.tables("environments").items():
        said = _Table(each, f'[environments."{environment}"]')
        project = said.text("project", None)
        python = said.text("python", None if project else "platform")
        interpreter = said.text("interpreter", None)
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
        environments[environment] = EnvironmentSection(environment, python, project, interpreter)
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
        ledger=LedgerSection(url, url_secret, ledger_token, public),
        blobs=blobs_said,
        stores=stores,
        scratch=scratch_directory,
        ray=ray_said,
        kubernetes=kubernetes,
        tls=tls,
        gateway=gateway_said,
        monitor=monitor_said,
        runners=runners_said,
        guards=guards_said,
        inference=inference,
        trainers=trainers,
        sandboxes=sandboxes,
        tools=tools,
        environments=environments,
        placement=placement,
        bridges=bridges,
        capacity=capacity,
        described=_handed_on(described, environments, kubernetes),
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
    allocation, concurrency = _allocation(said, where, kind.allocation, gpus)
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
    capabilities = kind.capabilities
    if "max_logprobs" in said.table:
        capabilities = replace(capabilities, top_logprobs=said.whole("max_logprobs", least=0))
    settings = said.rest()
    if unknown := sorted(set(settings) - set(kind.fields)):
        fields = ("kind", "auth", "gpus", "replicas", "allocation", "concurrency", "models", *kind.fields)
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
    elif kind_name in RUNPOD:
        _pods(where, kind_name, settings, secrets)
    if (problem := auth_problem(where, auth, endpoints)) is not None:
        raise ClusterError(problem)
    return InferenceProvider(
        name,
        kind_name,
        capabilities,
        models,
        auth,
        gpus,
        replicas,
        endpoints,
        settings,
        secrets,
        allocation=allocation,
        concurrency=concurrency,
    )


def _allocation(said: "_Table", where: str, default: Allocation, gpus: float) -> tuple[Allocation, int | None]:
    """A provider's or trainer's allocation (its kind's unless said) and concurrency: one with GPUs here is scheduled,
    and only a metered one has a concurrency."""
    allocation = said.text("allocation", default)
    if allocation not in ALLOCATIONS:
        raise ClusterError(f"{where} allocation is metered or scheduled, not {allocation!r}")
    if allocation == "metered" and gpus:
        raise ClusterError(f"{where} asks for GPUs of the cluster's, so it is scheduled, not metered")
    concurrency = said.whole("concurrency", None, least=1)
    if concurrency is not None and allocation != "metered":
        raise ClusterError(f"{where} concurrency bounds a metered provider; a scheduled one is bounded by its capacity")
    return allocation, concurrency


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
    gpus = said.number("gpus", 0.0)
    allocation, concurrency = _allocation(said, where, kind.allocation, gpus)
    provider = TrainerProvider(
        name=name,
        kind=kind_name,
        capabilities=kind.capabilities,
        models=tuple(str(each) for each in cast(list[str], models)),
        auth=auth,
        segment_tokens=said.whole("segment_tokens", None, least=1),
        gpus=gpus,
        colocate_with=said.text("colocate_with", None),
        cost=cost,
        costs=costs,
        secrets=secrets,
        allocation=allocation,
        concurrency=concurrency,
    )
    settings = said.rest()
    if unknown := sorted(set(settings) - set(kind.fields)):
        fields = ("kind", "auth", "models", "segment_tokens", "gpus", "colocate_with", "cost", "costs", "allocation",
                  "concurrency", *kind.fields)  # fmt: skip
        raise ClusterError(f"{where} has no {', '.join(unknown)} (a {kind_name} trainer has {', '.join(fields)})")
    memory = settings.get("gpu_memory_gib")
    if memory is not None and (isinstance(memory, bool) or not isinstance(memory, int | float) or memory <= 0):
        raise ClusterError(f"{where} gpu_memory_gib is the memory of each of its GPUs, in GiB (not {memory!r})")
    if gpus > 1 and gpus != int(gpus):
        raise ClusterError(f"{where} gpus above one is a whole number: a trainer steps on whole GPUs (not {gpus:g})")
    if gpus > 1 and provider.colocate_with is not None:
        raise ClusterError(f"{where} shares {provider.colocate_with}'s GPU (colocate_with), so it steps on one: a "
                           "trainer on several keeps its processes and model on them between steps")  # fmt: skip
    if kind_name == "runpod-trainer":
        runs = settings.get("trainer", "lora")
        if runs not in ("lora", "full"):
            raise ClusterError(f"{where} trainer is lora or full, the trainer its pods run (not {runs!r})")
        if provider.colocate_with is None:
            _pods(where, kind_name, settings, secrets)
            provider = replace(provider, secrets=secrets)
        elif said_pods := sorted(set(settings) & set(POD_FIELDS)):
            raise ClusterError(f"{where} takes its steps on {provider.colocate_with}'s pods, which say what they are: "
                               f"it says no {', '.join(said_pods)}")  # fmt: skip

    provider = replace(provider, settings=settings)
    return replace(provider, capabilities=provider.runs.capabilities)


def _pods(where: str, kind: str, settings: Mapping[str, JsonValue], secrets: dict[str, Secret]) -> None:
    """A RunPod kind's table, checked (`pod_table`); its API key is `RUNPOD_API_KEY`'s unless it names another."""
    try:
        pod_table(kind, settings)
    except ValueError as error:
        raise ClusterError(f"{where} {error}") from None
    secrets.setdefault("api_key", Secret(env="RUNPOD_API_KEY"))


CREDENTIALS = ("access_key_id_env", "secret_access_key_env")
"""What names a store's key: the variables it is read from."""


def _blobs(said: "_Table", where: str, *, named: bool = False) -> BlobsSection:
    kind = said.text("kind", "files")
    reader: dict[str, str] = {}
    if named and "reader" in said.table:
        given = said.take("reader")
        if not isinstance(given, dict) or set(cast(dict[str, Any], given)) != set(CREDENTIALS):
            raise ClusterError(f"{where} reader names a read-only key: {' and '.join(CREDENTIALS)}")
        reader = {str(key): str(value) for key, value in cast(dict[str, Any], given).items()}
    settings = said.rest()
    if kind == "files" and set(settings) - {"directory"}:
        raise ClusterError(f"{where} of files has only a directory (not {', '.join(sorted(settings))})")
    if kind != "files" and ":" not in kind:
        raise ClusterError(f"{where} kind is files or module:name, not {kind!r}")
    if (CREDENTIALS[0] in settings) != (CREDENTIALS[1] in settings):
        raise ClusterError(f"{where} names its key whole: {' and '.join(CREDENTIALS)}")
    if kind == "files":
        settings.setdefault("directory", "~/.cache/rollout/blobs")
    return BlobsSection(kind, settings, reader)


def _sandboxes(kind: str, described: dict[str, Any]) -> SandboxesSection:
    where = f"[sandboxes.{kind}]"
    said = _Table(described, where)
    provider, url = said.text("provider", None), said.text("url", None)
    if provider is None and url is None:
        raise ClusterError(f"{where} names its provider (made in each run), or the url it is served at, or both")
    if url is not None and not url.startswith(("http://", "https://")):
        raise ClusterError(f"{where} url is an http:// or https:// URL")
    return SandboxesSection(
        kind=kind,
        provider=provider,
        python=said.text("python", "platform"),
        size=said.whole("size", 1, least=1),
        url=url,
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


def _handed_on(
    described: Mapping[str, Any],
    environments: Mapping[str, EnvironmentSection],
    kubernetes: KubernetesSection | None = None,
) -> dict[str, Any]:
    """The config as JSON, with each environment's project and the RayJob template as they were resolved."""
    handed: dict[str, Any] = _json(described)
    for name, python in environments.items():
        if python.project is not None:
            handed["environments"][name]["project"] = python.project
    if kubernetes is not None:
        handed["kubernetes"]["rayjob"] = kubernetes.rayjob
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


def inspect(cluster: Cluster, environ: Mapping[str, str] | None = None, role: str = "run") -> list[str]:
    """What is wrong with the cluster on this node, in words: each secret reference `role` reads
    (`Cluster.secrets_of`) that does not resolve (by name, never by value), and each environment's project with no
    `uv.lock`. Empty: nothing."""
    problems: list[str] = []
    for where, secret in cluster.secrets_of(role).items():
        if secret.resolve(environ) is None:
            problems.append(f"{where}: {secret} is not set on this node")
    for name, python in cluster.environments.items():
        if python.project is not None and not (Path(python.project) / "uv.lock").is_file():
            problems.append(f'environments."{name}": {python.project} has no uv.lock to build its Python from')
    return problems
