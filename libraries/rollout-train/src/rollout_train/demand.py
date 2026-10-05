# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray's placement groups are partly untyped.)
"""What a run's scheduled parts need, and the placement group that reserves them together.

A run's parts are metered or scheduled, as each provider's and trainer's `allocation` says. A metered part (Tinker's
trainer and sampler, a hosted API) is bounded by spend, rate limits and its concurrency, and is not in the run's
demand: a metered trainer's actor asks Ray for nothing and runs on the driver's node beside it. A scheduled part is
capacity the run must be given: its trainer, its engine hosts, the bridge that turns each checkpoint into what its
engines load, and its driver. `demand(settings, cluster)` says what they need, from the run's settings and the cluster
config alone:

- **The driver** (the job's entrypoint): 1 CPU for the loop and its gateway, 1 CPU for each runner of `[runners]
  places` episodes (`episodes_at_once`), and each sandbox pool's `size` times its `cpus`; 2 GiB of memory, and each
  pool's `size` times its `memory_gib`.
- **The trainer** (training and imitate runs on a scheduled trainer), in a bundle on the driver's node: 1 CPU and the
  trainer's `gpus` (half of them where it shares the trained channel's card, `colocate_with`).
- **Each engine host** (per replica of a channel on a scheduled `vllm` provider), in a bundle of its own, or in the
  trainer's where they share a card: 1 CPU, its replica's GPUs (`host_spec`), and `[placement.engines]`.
- **The bridge** (training runs, and evals of a checkpoint), in a bundle of its own: the largest bridge of the chain
  the run may run, its `cpus` and `memory_gib` or what `[bridges."NAME"]` says.

Bridges run one at a time (a checkpoint is bridged before the next is served, and a chain's bridges in turn), so one
bundle the size of the largest holds every bridge the run runs. Channels on servers elsewhere (`vllm-servers`, RunPod
pods, which are scheduled where they run) and on Tinker ask the run's Ray cluster for nothing, nor does a trainer on
RunPod's pods (`runpod-trainer`): the run leases its pods (`rollout_train.pods.leasing`), outside Kueue's quota.

The driver reserves its parts as one placement group (`reserve`) before it starts any of them, so a run starts only
with all of it reserved and never waits half-placed. The group is `PACK`: Ray puts its bundles on as few nodes as hold
them, all on one node where one has room, and spreads engine replicas over nodes only where no one node holds them.
The trainer's bundle is pinned to the driver's node (a step's files are handed to it by path); a trainer that shares
its engines' card shares their bundle. Each actor and task asks for exactly what its part counted, in its bundle.

On Kubernetes the run's Ray cluster is sized from the same demand (`rollout_train.submitting.rendered`): its pods ask
for `Demand.total` and room for Ray's own processes (`HEADROOM`); and Kueue admits the RayJob only when its whole
request fits the queue's quota.
"""

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from rollout_train.bridges import AUTO, FORMATS, Bridge, NoBridge, path
from rollout_train.cluster import Cluster

if TYPE_CHECKING:
    from rollout_train.run_settings import RunSettings

__all__ = [
    "BRIDGE",
    "HEADROOM",
    "SUBMITTER",
    "TRAINER",
    "Bundle",
    "Demand",
    "Part",
    "Pod",
    "Resources",
    "bridge_asks",
    "colocating",
    "demand",
    "placed",
    "played_channel",
    "pods",
    "requested",
    "reserve",
]

DRIVER_CPUS = 1.0
"""The driver's own: the loop and the gateway in its process."""
DRIVER_MEMORY_GIB = 2.0
RUNNER_CPUS = 1.0
"""Each runner of `[runners] places` episodes, in the driver's process."""
TRAINER_CPUS = 1.0
TRAINER = "trainer"
"""The trainer's part, by name (an engine host's is `engine/CHANNEL/N`)."""
BRIDGE = "bridge"
"""The bridge's part, by name."""
PACK = "PACK"
GIB = 2**30


@dataclass(frozen=True)
class Resources:
    """CPUs, memory, GPUs and custom resources (`[placement.ROLE]`), as Ray counts them."""

    cpus: float = 0.0
    memory_gib: float = 0.0
    gpus: float = 0.0
    custom: Mapping[str, float] = field(default_factory=dict[str, float])

    def __add__(self, other: "Resources") -> "Resources":
        custom = dict(self.custom)
        for key, value in other.custom.items():
            custom[key] = custom.get(key, 0.0) + value
        return Resources(self.cpus + other.cpus, self.memory_gib + other.memory_gib, self.gpus + other.gpus, custom)

    def most(self, other: "Resources") -> "Resources":
        """Each resource the larger of the two."""
        custom = dict(self.custom)
        for key, value in other.custom.items():
            custom[key] = max(custom.get(key, 0.0), value)
        return Resources(
            max(self.cpus, other.cpus), max(self.memory_gib, other.memory_gib), max(self.gpus, other.gpus), custom
        )

    def beyond(self, room: "Resources", *, known: Collection[str] = ("cpus", "memory_gib", "gpus")) -> list[str]:
        """What of this exceeds `room` (only the resources in `known`, and the custom ones `room` names), each as
        `CPUs (9 for 8)`."""
        over: list[str] = []
        for name, called in (("gpus", "GPUs"), ("cpus", "CPUs"), ("memory_gib", "memory")):
            if name in known and getattr(self, name) > getattr(room, name) + 1e-9:
                unit = " GiB" if name == "memory_gib" else ""
                over.append(f"{called} ({_number(getattr(self, name))}{unit} for {_number(getattr(room, name))}{unit})")
        for key, value in self.custom.items():
            if key in room.custom and value > room.custom[key] + 1e-9:
                over.append(f"{key} ({_number(value)} for {_number(room.custom[key])})")
        return over

    def bundle(self) -> dict[str, float]:
        """As a Ray placement group bundle (memory in bytes), leaving out what is zero. A share of GPUs above one is
        rounded up to whole GPUs: Ray takes fractions of one GPU only."""
        gpus = self.gpus if self.gpus <= 1 else math.ceil(self.gpus - 1e-9)
        said = {"CPU": self.cpus, "GPU": gpus, "memory": float(int(self.memory_gib * GIB))}
        said |= dict(self.custom)
        return {key: value for key, value in said.items() if value > 0}

    def said(self) -> str:
        """In words: `1 GPU, 2 CPUs, 1 GiB`."""
        parts = [_amount(name, getattr(self, name)) for name in ("gpus", "cpus", "memory_gib") if getattr(self, name)]
        parts += [f"{_number(value)} {key}" for key, value in self.custom.items() if value]
        return ", ".join(parts) or "nothing"

    def to_json(self) -> dict[str, JsonValue]:
        return {"cpus": self.cpus, "memory_gib": self.memory_gib, "gpus": self.gpus, "custom": dict(self.custom)}


def _number(value: float) -> str:
    return f"{round(value, 3):g}"


def _amount(name: str, value: float) -> str:
    if name == "memory_gib":
        return f"{_number(value)} GiB"
    unit = {"gpus": "GPU", "cpus": "CPU"}[name]
    return f"{_number(value)} {unit}{'s' if name == 'cpus' and value != 1 else ''}"


@dataclass(frozen=True)
class Part:
    """One actor or task of a run, by name (`trainer`, `engine/CHANNEL/N`, `bridge`), and what it asks Ray for."""

    name: str
    asks: Resources


@dataclass(frozen=True)
class Bundle:
    """One bundle of a run's placement group: the parts placed in it, and whether it is on the driver's node."""

    parts: tuple[Part, ...]
    on_driver: bool = False

    @property
    def resources(self) -> Resources:
        total = Resources()
        for part in self.parts:
            total = total + part.asks
        return total


@dataclass(frozen=True)
class Demand:
    """What a run's scheduled parts need: its driver's (the job's entrypoint), and its placement group's bundles."""

    driver: Resources
    bundles: tuple[Bundle, ...] = ()
    strategy: str = PACK

    @property
    def reserved(self) -> Resources:
        """What its placement group reserves."""
        total = Resources()
        for bundle in self.bundles:
            total = total + bundle.resources
        return total

    @property
    def total(self) -> Resources:
        """The driver's and the placement group's."""
        return self.driver + self.reserved

    @property
    def parts(self) -> dict[str, Part]:
        return {part.name: part for bundle in self.bundles for part in bundle.parts}

    def index(self, name: str) -> int | None:
        """The bundle a part is placed in, by its name; none for a part the demand does not count."""
        return next((index for index, bundle in enumerate(self.bundles) if any(each.name == name for each in
                                                                                bundle.parts)), None)  # fmt: skip

    def asks(self, name: str) -> Resources | None:
        """What a part asks Ray for, by its name."""
        found = self.parts.get(name)
        return found.asks if found is not None else None

    def to_json(self) -> dict[str, JsonValue]:
        bundles: list[JsonValue] = [
            {"parts": {part.name: part.asks.to_json() for part in each.parts}, "on_driver": each.on_driver}
            for each in self.bundles
        ]
        return {"driver": self.driver.to_json(), "bundles": bundles, "strategy": self.strategy,
                "total": self.total.to_json()}  # fmt: skip


HEADROOM = Resources(cpus=1.0, memory_gib=2.0)
"""Room a pod of a run's Ray cluster keeps beyond what Ray schedules: Ray's own processes (its GCS, raylet, dashboard
and object store) and the memory its GPU processes hold outside Ray's count."""
SUBMITTER = Resources(cpus=0.5, memory_gib=0.2)
"""The pod KubeRay starts to submit a RayJob's job (its default requests), which Kueue counts with the job's."""


@dataclass(frozen=True)
class Pod:
    """One kind of pod of a run's Ray cluster on Kubernetes: the head (`head`) or a worker group (`engines-N`), what
    Ray schedules on each (whole CPUs and GPUs, as Ray starts a node with), and how many there are."""

    group: str
    ray: Resources
    replicas: int = 1

    @property
    def requests(self) -> Resources:
        """What each pod asks Kubernetes for: what Ray schedules on it, and `HEADROOM`."""
        return self.ray + HEADROOM


def _whole(resources: Resources) -> Resources:
    return Resources(math.ceil(resources.cpus - 1e-9), resources.memory_gib, math.ceil(resources.gpus - 1e-9),
                     dict(resources.custom))  # fmt: skip


def pods(asked: Demand, most: Resources | None = None, *, known: Collection[str] = ()) -> tuple[Pod, ...]:
    """The pods of a run's Ray cluster: one head pod for all of it where its requests fit `most` (a pod's upper bound,
    in the resources `known`: the template's limits, one node's worth); else the head holds the driver, the trainer's
    bundle and the bundles with no GPU (the bridge's), and each other bundle (an engine host's) is a worker pod,
    grouped by size (`engines-0`, `engines-1`, ...)."""
    one = Pod("head", _whole(asked.total))
    if most is None or not one.requests.beyond(most, known=known):
        return (one,)
    head = asked.driver
    sizes: list[Resources] = []
    counts: list[int] = []
    for each in asked.bundles:
        if each.on_driver or not each.resources.gpus:
            head = head + each.resources
            continue
        sized = _whole(each.resources)
        if sized in sizes:
            counts[sizes.index(sized)] += 1
        else:
            sizes.append(sized)
            counts.append(1)
    workers = [Pod(f"engines-{index}", size, count) for index, (size, count) in
               enumerate(zip(sizes, counts, strict=True))]  # fmt: skip
    return (Pod("head", _whole(head)), *workers)


def requested(asked: Demand, *, kubernetes: bool = False, most: Resources | None = None,
              known: Collection[str] = ()) -> Resources:  # fmt: skip
    """What a run's Ray cluster asks for in all: each pod's requests (`pods`), and on Kubernetes the pod that submits
    its job."""
    total = SUBMITTER if kubernetes else Resources()
    for pod in pods(asked, most, known=known):
        for _ in range(pod.replicas):
            total = total + pod.requests
    return total


def played_channel(settings: "RunSettings") -> str:
    """The channel a run trains or plays: the trained one, else the first its settings name."""
    trained = settings["trainer.channel"]
    channels = settings.channels
    return str(trained) if trained in channels or not channels else channels[0]


def colocating(settings: "RunSettings", cluster: Cluster) -> bool:
    """Whether the run's trainer shares the GPU of its trained channel's engine hosts (`colocate_with`)."""
    provider = cluster.trainers.get(str(settings["trainer.provider"]))
    trained = settings.trained
    if provider is None or trained is None or provider.colocate_with is None:
        return False
    return provider.colocate_with in settings.providers(trained)


def bridge_asks(bridge: Bridge, cluster: Cluster) -> Resources:
    """What a bridge's task asks Ray for: its declared CPUs and memory, or what `[bridges."NAME"]` says."""
    said = cluster.bridges.get(bridge.name)
    cpus = said.cpus if said is not None and said.cpus is not None else bridge.cpus
    memory = said.memory_gib if said is not None and said.memory_gib is not None else bridge.memory_gib
    return Resources(cpus=cpus, memory_gib=memory)


def demand(settings: "RunSettings", cluster: Cluster, *, sandboxes: Collection[str] = ()) -> Demand:
    """What a run with these settings needs on this cluster (the module's docstring); `sandboxes` are the kinds of
    sandbox its environment's programs declare. Parts the cluster does not offer are left out (validation refuses
    them)."""
    from rollout_train.run_settings import KINDS

    train, evaluate, imitate, check = KINDS
    kind = settings.kind
    provider = cluster.trainers.get(str(settings["trainer.provider"])) if kind in (train, imitate) else None
    hosts = _hosts(settings, cluster)
    bundles: list[Bundle] = []
    if provider is not None and provider.allocation == "scheduled" and provider.kind != "runpod-trainer":
        shared = provider.colocate_with if colocating(settings, cluster) else None
        trainer = Part(TRAINER, Resources(cpus=TRAINER_CPUS, gpus=provider.gpus / 2 if shared else provider.gpus))
        with_it = next((part for channel, name, part in hosts
                        if channel == settings.trained and name == shared), None)  # fmt: skip
        bundles.append(Bundle((trainer, *((with_it,) if with_it is not None else ())), on_driver=True))
        hosts = [each for each in hosts if each[2] is not with_it]
    bundles += [Bundle((part,)) for _, _, part in hosts]
    bridge = _bridge(settings, cluster) if kind in (train, evaluate) else None
    if bridge is not None:
        bundles.append(Bundle((Part(BRIDGE, bridge),)))
    driver = Resources(cpus=DRIVER_CPUS, memory_gib=DRIVER_MEMORY_GIB)
    if kind in (train, evaluate, check):
        at_once = settings["episodes_at_once"]
        runners = math.ceil(int(at_once) / cluster.runners.places) if isinstance(at_once, int) else 1
        driver = driver + Resources(cpus=runners * RUNNER_CPUS)
        for each in sorted(set(sandboxes)):
            pool = cluster.sandboxes.get(each)
            if pool is not None:
                driver = driver + Resources(cpus=pool.size * pool.cpus, memory_gib=pool.size * pool.memory_gib)
    return Demand(driver, tuple(bundles))


def _hosts(settings: "RunSettings", cluster: Cluster) -> list[tuple[str, str, Part]]:
    """Each engine host the run starts, with its channel and provider: one per replica of each channel on a `vllm`
    provider, as `rollout_train.jobs.Run.hosted` starts them."""
    from rollout_train.inference.hosts import host_spec

    found: list[tuple[str, str, Part]] = []
    for channel in settings.channels:
        model = settings.get(f"channels.{channel}.model")
        if model is None or settings.get(f"channels.{channel}.renderer") is None:
            continue
        for name in settings.providers(channel):
            offered = cluster.inference.get(name)
            if offered is None or offered.kind != "vllm" or offered.allocation != "scheduled":
                continue
            try:
                spec = host_spec(cluster, name, str(model), settings=settings)
            except ValueError:  # (a model it does not serve: validation refuses it)
                continue
            asked = settings[f"channels.{channel}.replicas"]
            count = asked if isinstance(asked, int) else offered.replicas
            for index in range(count):
                asks = Resources(cpus=spec.cpus, gpus=spec.gpus, custom=dict(spec.resources))
                found.append((channel, name, Part(f"engine/{channel}/{index}", asks)))
    return found


def _bridge(settings: "RunSettings", cluster: Cluster) -> Resources | None:
    """The largest bridge the run may run: of the chain from its trainer's format (a training run) or from any format
    a checkpoint may be in (an eval of one) to what its channel's first provider loads; none where no bridge writes
    anything."""
    channel = played_channel(settings)
    providers = settings.providers(channel)
    offered = cluster.inference.get(providers[0]) if providers else None
    if offered is None:
        return None
    if settings.kind == "train":
        trainer = cluster.trainers.get(str(settings["trainer.provider"]))
        formats: Collection[str] = (trainer.capabilities.format,) if trainer is not None else ()
    else:
        formats = FORMATS if isinstance(settings["start"], str) else ()  # (an eval of a base model bridges nothing)
    wanted = str(settings.get(f"channels.{channel}.bridge") or AUTO)
    largest: Resources | None = None
    for each in formats:
        chain = path(each, offered.capabilities.loads, wanted=wanted)
        if isinstance(chain, NoBridge):
            continue
        for bridge in chain:
            if bridge.task is not None:
                asks = bridge_asks(bridge, cluster)
                largest = asks if largest is None else largest.most(asks)
    return largest


def _node_resource(node: str) -> str | None:
    """The resource Ray gives a node by its address (`node:IP`), which a bundle names to be placed there."""
    import ray

    for each in ray.nodes():
        if each.get("NodeID") == node:
            for key in each.get("Resources", {}):
                if key.startswith("node:") and not key.startswith("node:__internal"):
                    return str(key)
    return None


def reserve(asked: Demand, name: str) -> Any:
    """A placement group for a run's demand on the Ray cluster this process is connected to, named `name`: its bundles
    in order, the trainer's pinned to this process's node. Ray reserves every bundle at once or none (`ready()` says
    when); none where the demand has no bundle."""
    import ray
    from ray.util.placement_group import placement_group

    if not asked.bundles:
        return None
    here = _node_resource(ray.get_runtime_context().get_node_id())
    bundles: list[dict[str, float]] = []
    for each in asked.bundles:
        bundle = each.resources.bundle()
        if each.on_driver and here is not None:
            bundle[here] = 0.001
        bundles.append(bundle)
    return placement_group(bundles, strategy=asked.strategy, name=name)


def placed(group: Any, asked: Demand | None, part: str) -> dict[str, Any]:
    """The options that place a part in its bundle of `group` (none where there is no group, or the demand does not
    count the part)."""
    if group is None or asked is None:
        return {}
    index = asked.index(part)
    if index is None:
        return {}
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    return {"scheduling_strategy": PlacementGroupSchedulingStrategy(group, placement_group_bundle_index=index)}
