"""What asking for a run takes, wherever it is asked: its settings in layers, the facts validation reads, and what a
cluster offers.

A run is asked for with its kind, its name, its environment, its settings and the preset they start from (`settled`:
the preset's settings, then those given). Before it is recorded, and again by its driver before it claims anything,
its settings are checked against the cluster config (`checked`, over `rollout_train.validation.check`) with the facts
gathered here: the environment's (`environment_facts`: whether it loads, the sandboxes and slots its programs declare;
a published one's from its version's record) and the ledger's (`ledger_facts`: the checkpoints the settings name, the
suites, the names taken, the GPUs the caller knows of, and what Ray has free: `ray_free`). A finding that refuses names
the setting it is about; one that does not is a note (the run waits for something).

`offers` is what the New run form chooses from: the cluster config's environments (built-in, and every published
version beside the ledger), trainers with their settings, inference providers with their capabilities and models (each
with the renderer families that render it, among those named so far), each trainer's and provider's allocation and the
weights it takes (`lora`, `full`), a RunPod provider's pods (their GPU type and hourly price, cloud, regions, most at
once), whether a trainer can train apart from what samples its checkpoints (`separate`), each trainer and provider
pair's bridge or why there is none and whether the pair shares one machine (`together`), sandbox pools, presets,
the GPUs the heartbeats say are free, the objective's families, presets and components, and the keys a training run
takes (the schema). Each environment carries the renderer families runs and presets on it named for their trained
channel (its model family, as far as is known).

`checkpoints_at` says where a training run's checkpoints go, which the check (`examined`) answers with: the blob store
its trainer writes them to (the one its RunPod providers name, `[stores.NAME]`, else the cluster's `[blobs]`), whether
Tinker keeps the weights (the store then holds pointers to Tinker's archive), and the bridges that write converted
copies for its trained channel's provider, with the store those go to (the cluster's `[blobs]`).
"""

import asyncio
import contextlib
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout_train.bridges import AUTO, VERBATIM, NoBridge, format_of, path
from rollout_train.cluster import Cluster
from rollout_train.demand import Resources, played_channel
from rollout_train.ledger import Ledger
from rollout_train.presence import Beat, alive
from rollout_train.presets import Presets
from rollout_train.providers import settings_of
from rollout_train.published import environment_versions_of, is_published
from rollout_train.recorder.renderers import renderers_for
from rollout_train.registry import registry_of, resolved
from rollout_train.run_settings import TRAINING, WEIGHTS, RunSettings, layered
from rollout_train.stores import blobs_at, described, store_named
from rollout_train.validation import (
    CheckpointFacts,
    EnvironmentFacts,
    Finding,
    LedgerFacts,
    Spend,
    SuiteEntryFacts,
    SuiteFacts,
    check,
    completed,
    serves,
    spend_of,
    weights_of,
)

if TYPE_CHECKING:
    from rollout.environment import Environment

__all__ = [
    "Examined",
    "Refused",
    "capacity_of",
    "checked",
    "checkpoints_at",
    "declared",
    "environment_facts",
    "examined",
    "free_name",
    "ledger_facts",
    "offers",
    "ray_free",
    "settled",
]


class Refused(ValueError):
    """A run whose settings are refused: the findings that refuse it (each with the setting it is about), and the notes
    beside them."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings = list(findings)
        refusing = [each for each in self.findings if each.refuses]
        super().__init__("; ".join(f"{each.key}: {each.reason}" for each in refusing) or "refused")

    @property
    def refusals(self) -> list[Finding]:
        return [each for each in self.findings if each.refuses]


async def settled(
    kind: str,
    name: str | None,
    settings: Mapping[str, JsonValue],
    *,
    preset: str | None = None,
    presets: Presets | None = None,
) -> tuple[RunSettings, str | None]:
    """A run's settings: the preset's (`NAME` or `NAME@N`: those a run of its kind takes, so that a training run's
    preset serves an eval of the same channels), then `settings`, then its kind and name; and the preset's version
    (`NAME@N`). Raises `KeyError` for a preset there is none of."""
    from rollout_train.run_settings import TRAINING, is_trainers, key_of

    def taken(key: str) -> bool:
        found = key_of(key)
        return kind in found.kinds if found is not None else is_trainers(key) and kind in TRAINING

    chosen = None
    if preset:
        chosen = await presets.get(preset) if presets is not None else None
        if chosen is None:
            raise KeyError(f"there is no preset {preset!r}")
    kept = {key: value for key, value in chosen.settings.items() if taken(key)} if chosen else None
    said = layered(kept, settings, {"kind": kind, "name": name})
    return said, chosen.id if chosen else None


def declared(environment: "Environment") -> tuple[frozenset[str], EnvironmentFacts]:
    """The sandbox kinds an environment's first program declares, and what validation reads of it."""
    from rollout.environment import first_program
    from rollout.harness import instantiate
    from rollout_train.slots import Declared

    program = instantiate(first_program(environment))
    kinds = frozenset(spec.kind for spec in program.sandboxes().values())
    slots = Declared.of(program.model_slots())
    return kinds, EnvironmentFacts(
        "", sandboxes=kinds, slots=slots.names, untrained=slots.untrained, judges=slots.judges
    )


async def environment_facts(
    environment: str | None, cluster: Cluster, ledger: Ledger, *, loaded: "Environment | None" = None
) -> EnvironmentFacts | None:
    """What validation reads of an environment: imported here (`loaded`, where the caller has it), its sandboxes, slots
    and what an episode samples (its description's `turns`, `samples_per_turn`, `prompt_tokens`); a published one's
    sandboxes and what an episode samples as its version recorded them, or that there is no such version. None where it
    is not known here: an environment whose Python is a project of its own, which this process does not import."""
    if not environment:
        return None
    if is_published(environment):
        versions = environment_versions_of(ledger)
        version = await versions.get(environment) if versions is not None else None
        if version is None:
            return EnvironmentFacts(environment, loads=False, why=f"there is no published environment {environment}")
        if loaded is not None:
            return _facts(environment, loaded)
        listed = version.description.get("sandboxes")
        kinds = (
            frozenset(str(each) for each in cast(list[Any], listed)) if isinstance(listed, list) else frozenset[str]()
        )
        said = version.description.get("description")
        return EnvironmentFacts(environment, sandboxes=kinds, **_sampling(said if isinstance(said, dict) else {}))
    python = cluster.environments.get(environment)
    if loaded is None and (python is None or python.project is not None):
        return None
    if loaded is None:
        from rollout.names import named

        try:
            imported: Environment = await asyncio.to_thread(named, environment)
        except Exception as error:  # (whatever importing it raises: it does not load)
            return EnvironmentFacts(environment, loads=False, why=f"{type(error).__name__}: {error}")
        return _facts(environment, imported)
    return _facts(environment, loaded)


def _facts(name: str, environment: "Environment") -> EnvironmentFacts:
    try:
        _, facts = declared(environment)
    except Exception as error:  # (a program that cannot be made: it does not load)
        return EnvironmentFacts(name, loads=False, why=f"{type(error).__name__}: {error}")
    return EnvironmentFacts(name, sandboxes=facts.sandboxes, slots=facts.slots, untrained=facts.untrained,
                            judges=facts.judges, **_sampling(environment.description.to_json()))  # fmt: skip


def _sampling(description: Mapping[str, Any]) -> dict[str, Any]:
    """What an episode samples, as an environment's description says it (`Description.to_json`): what a step's spend
    is estimated from."""
    turns, samples, prompt = (description.get(key) for key in ("turns", "samples_per_turn", "prompt_tokens"))
    number = (int, float)
    return {
        "turns_per_episode": float(turns) if isinstance(turns, number) else None,
        "samples_per_turn": float(samples) if isinstance(samples, number) else 1.0,
        "prompt_tokens": int(prompt) if isinstance(prompt, number) else None,
    }


async def ledger_facts(
    settings: RunSettings,
    ledger: Ledger,
    *,
    own: str | None = None,
    gpus: float | None = None,
    free: Resources | None = None,
    cluster: Cluster | None = None,
) -> LedgerFacts:
    """What validation reads of the ledger: each checkpoint the settings name (the start, a fixed channel's), the
    suites their evals name, the names other runs have (`own`, the run's id, is left out), the GPUs and free resources
    the caller knows of, and, for a run with pods on `cluster`'s RunPod providers, how long a step took here lately."""
    from rollout_train.evals import suite_of, versions_of
    from rollout_train.pods.leasing import needs_of

    try:
        pods = bool(needs_of(settings, cluster)) if cluster is not None else False
    except ValueError:
        pods = False

    registry = registry_of(ledger)
    references = [settings["start"], *(value for key, value in settings.values.items()
                                       if key.startswith("channels.") and key.endswith(".checkpoint"))]  # fmt: skip
    checkpoints: dict[str, CheckpointFacts] = {}
    for reference in references:
        if not isinstance(reference, str) or reference in checkpoints:
            continue
        try:
            id = await resolved(ledger, registry, reference)
        except KeyError:
            checkpoints[reference] = CheckpointFacts(reference, exists=False)
            continue
        if id is None:
            continue
        from rollout_train.checkpoints import Checkpoints

        made = Checkpoints(ledger, cast(Any, None))
        record = await made.checkpoint(id)
        files: Mapping[str, Any] = record.weights.files if record.weights is not None else {}
        model, seen = record.base, {record.id}
        while model is not None and model not in seen:  # (an adapter over full weights: the model those are over)
            seen.add(model)
            try:
                model = (await made.checkpoint(model)).base
            except KeyError:
                break
        checkpoints[reference] = CheckpointFacts(
            reference, released=record.weights is None, formats=format_of(files), model=model
        )
    suites: dict[str, SuiteFacts] = {}
    for key in ("evals.suite", "eval.suite"):
        named = settings.get(key)
        if not isinstance(named, str):
            continue
        name = named.partition("@")[0]
        versions = await versions_of(ledger, name)
        newest = await suite_of(ledger, name)
        if newest is not None and versions:
            environments = frozenset(each for version in versions for each in version.environments)
            try:
                played = await suite_of(ledger, named)
            except (KeyError, ValueError):
                played = None
            entries = tuple(
                SuiteEntryFacts(each.environment, len(each.starts), each.episodes, each.thinking_tokens,
                                each.answer_tokens)
                for each in (played.entries if played is not None else ())
            )  # fmt: skip
            suites[name] = SuiteFacts(name, max(version.number for version in versions), environments, entries)
    runs = await registry.runs() if registry is not None else []
    taken = frozenset(each.name for each in runs if each.id != own)
    seconds = await _step_seconds(settings, ledger) if pods else None
    return LedgerFacts(checkpoints=checkpoints, suites=suites, names_taken=taken, gpus=gpus, free=free,
                       step_seconds=seconds)  # fmt: skip


async def _step_seconds(settings: RunSettings, ledger: Ledger) -> float | None:
    """How long a step of the run's trainer and model took here lately: the median time between the checkpoints of the
    newest run of the same trainer provider and model that made three or more (none: no such run)."""
    from rollout_train.checkpoints import checkpoints_in
    from rollout_train.record import recorded_settings

    provider, model = settings.get("trainer.provider"), settings.trainer_model
    if settings.kind != "train" or provider is None:
        return None
    made: dict[str, list[float]] = {}
    for each in await checkpoints_in(ledger):
        if each.run is not None:
            made.setdefault(each.run, []).append(each.made)
    for run, times in sorted(made.items(), key=lambda item: -max(item[1])):
        if len(times) < 3:
            continue
        said = await recorded_settings(ledger, run) or {}
        trained = said.get("trainer.model") or said.get(f"channels.{said.get('trainer.channel') or 'policy'}.model")
        if said.get("trainer.provider") != provider or trained != model:
            continue
        times.sort()
        gaps = sorted(later - earlier for earlier, later in itertools.pairwise(times))
        return gaps[len(gaps) // 2]
    return None


async def checked(
    settings: RunSettings,
    cluster: Cluster,
    ledger: Ledger,
    *,
    loaded: "Environment | None" = None,
    own: str | None = None,
    gpus: float | None = None,
    free: Resources | None = None,
) -> list[Finding]:
    """Everything wrong with a run's settings on this cluster (`rollout_train.validation.check`), with the facts
    gathered now."""
    return (await examined(settings, cluster, ledger, loaded=loaded, own=own, gpus=gpus, free=free)).findings


@dataclass(frozen=True)
class Examined:
    """A run's settings, checked: the findings, what is known of its environment, its estimated spend (one step's, or
    an eval's), what it trains and where its checkpoints go (`checkpoints_at`)."""

    findings: list[Finding]
    environment: EnvironmentFacts | None
    spend: Spend
    weights: str | None
    checkpoints: dict[str, JsonValue] | None = None


async def examined(
    settings: RunSettings,
    cluster: Cluster,
    ledger: Ledger,
    *,
    loaded: "Environment | None" = None,
    own: str | None = None,
    gpus: float | None = None,
    free: Resources | None = None,
) -> Examined:
    """A run's settings checked on this cluster with the facts gathered now (`checked`), with those facts' environment,
    its estimated spend (`rollout_train.validation.spend_of`), what it trains (`weights_of`) and where its checkpoints
    go (`checkpoints_at`)."""
    settings = completed(settings, cluster)
    environment = settings.get("environment")
    facts = await environment_facts(str(environment) if environment else None, cluster, ledger, loaded=loaded)
    known = await ledger_facts(settings, ledger, own=own, gpus=gpus, free=free, cluster=cluster)
    return Examined(
        check(settings, cluster, facts, known),
        facts,
        spend_of(settings, cluster, facts, known),
        weights_of(settings, cluster),
        checkpoints_at(settings, cluster),
    )


def checkpoints_at(
    settings: RunSettings, cluster: Cluster, *, written: Mapping[str, Any] | None = None
) -> dict[str, JsonValue] | None:
    """Where a training run's checkpoints go (none for a run that makes none): `store`, the blob store its trainer
    writes them to (`rollout_train.stores.described`: the store its RunPod providers name, else `[blobs]`; or the one
    `written` says, where a run's start recorded it); `tinker`, whether Tinker keeps the weights (the store holds
    pointers to Tinker's archive); `bridges`, the bridges that write converted copies for the trained channel's first
    provider (none where its files are served as they are), and `bridged`, the store those copies go to (`[blobs]`)."""
    from rollout_train.pods.leasing import pods_store

    if settings.kind not in TRAINING:
        return None
    if written is not None:
        store: dict[str, JsonValue] = described(written, store_named(cluster, written))
    else:
        try:
            named = pods_store(settings, cluster)
        except ValueError:  # (a trainer on a host's pods whose trained channel is elsewhere: validation refuses it)
            named = None
        store = described(blobs_at(cluster, named), named)
    trainer = cluster.trainers.get(str(settings["trainer.provider"]))
    format = trainer.capabilities.format if trainer is not None else None
    copies: list[str] = []
    channel = played_channel(settings)
    providers = settings.providers(channel)
    provider = cluster.inference.get(providers[0]) if providers else None
    if format is not None and provider is not None:
        wanted = str(settings.get(f"channels.{channel}.bridge") or AUTO)
        chain = path(format, provider.capabilities.loads, wanted=wanted)
        if not isinstance(chain, NoBridge):
            copies = [each.name for each in chain if each.task is not None and each.task != VERBATIM]
    return {
        "store": store, "tinker": format == "tinker", "bridges": cast(JsonValue, copies),
        "bridged": described(blobs_at(cluster), None) if copies else None,
    }  # fmt: skip


def ray_free() -> Resources | None:
    """What the Ray cluster this process is connected to has free now: CPUs, memory, GPUs and custom resources (none:
    not connected). Its total is not said: an autoscaled cluster has more than its nodes now."""
    import ray

    if not ray.is_initialized():
        return None
    said = cast(dict[str, float], ray.available_resources())  # pyright: ignore[reportUnknownMemberType]
    custom = {key: float(value) for key, value in said.items() if key not in ("CPU", "GPU", "memory",
              "object_store_memory") and not key.startswith("node:") and not key.startswith("bundle_")}  # fmt: skip
    return Resources(float(said.get("CPU", 0.0)), float(said.get("memory", 0.0)) / 2**30, float(said.get("GPU", 0.0)),
                     custom)  # fmt: skip


def capacity_of(beats: Sequence[Beat]) -> dict[str, JsonValue] | None:
    """The GPUs the machines that beat now have, and those of them idle (under a twentieth of their memory used), by
    machine; none where no beat says."""
    machines: dict[str, list[dict[str, Any]]] = {}
    for beat in beats:
        if not alive(beat):
            continue
        measured = beat.about.get("machine")
        host = beat.about.get("host")
        if not isinstance(measured, dict) or not isinstance(host, str):
            continue
        listed = cast(dict[str, Any], measured).get("accelerators")
        if isinstance(listed, list) and listed:
            machines[host] = cast(list[dict[str, Any]], listed)
    if not machines:
        return None
    total = sum(len(each) for each in machines.values())
    idle = sum(
        1 for each in machines.values() for gpu in each if gpu.get("total") and gpu.get("used", 0) < gpu["total"] / 20
    )
    return {"gpus": total, "gpus_free": idle, "machines": {host: len(each) for host, each in machines.items()}}


def free_name(wanted: str, taken: set[str]) -> str:
    """A name no run has: the one wanted, else it with the first number after it that no run has."""
    wanted = re.sub(r"[^\w .@:()/-]+", "-", wanted).strip() or "run"
    if wanted not in taken:
        return wanted
    number = 2
    while f"{wanted} ({number})" in taken:
        number += 1
    return f"{wanted} ({number})"


def _family(renderer: str) -> str:
    return renderer.partition(":")[0]


async def offers(cluster: Cluster, ledger: Ledger, beats: Sequence[Beat] = ()) -> dict[str, Any]:
    """What a run can be asked for here (the module's docstring), as JSON."""
    from rollout_train.presets import presets_of

    environments: list[dict[str, JsonValue]] = [
        {"environment": name, "published": False, "python": "project" if each.project else "platform"}
        for name, each in cluster.environments.items()
    ]
    versions = environment_versions_of(ledger)
    for version in await versions.all() if versions is not None else []:
        listed = version.description.get("sandboxes")
        environments.append({
            "environment": version.reference, "published": True, "name": version.name, "source": version.source,
            "commit": version.commit, "imported": version.imported,
            "sandboxes": cast(JsonValue, listed if isinstance(listed, list) else []),
        })  # fmt: skip
    _, families = await _renderers(ledger)
    for each in environments:
        each["families"] = cast(JsonValue, families.get(str(each["environment"]), []))
    trainers: list[dict[str, Any]] = []
    for name, trainer in cluster.trainers.items():
        try:
            specs = [asdict(each) for each in settings_of(trainer.runs)]
        except ImportError:  # (its package is not installed where this runs)
            specs = []
        trainers.append({
            "name": name, "kind": trainer.kind, "produces": trainer.capabilities.produces,
            "format": trainer.capabilities.format, "models": list(trainer.models), "gpus": trainer.gpus,
            "colocate_with": trainer.colocate_with, "segment_tokens": trainer.segment_tokens,
            "cost": dict(trainer.cost), "families": sorted(trainer.capabilities.families), "settings": specs,
            "allocation": trainer.allocation, "concurrency": trainer.concurrency,
            "weights": [trainer.capabilities.produces], "pods": _pods_offered(cluster, name),
            "separate": not (trainer.kind == "runpod-trainer" and trainer.colocate_with is not None),
        })  # fmt: skip
    inference: list[dict[str, Any]] = []
    for name, provider in cluster.inference.items():
        capabilities = asdict(provider.capabilities)
        capabilities["loads"] = sorted(provider.capabilities.loads)
        capabilities["unchecked"] = sorted(provider.capabilities.unchecked)
        models: list[dict[str, Any]] = []
        for model, offer in provider.models.items():
            # (the renderers that say they render the model; a hosted API renders messages itself)
            rendering = renderers_for(model, offer.base) if provider.capabilities.token_exact else []
            models.append({
                "model": model, "context": offer.context, "base": offer.base, "max_lora_rank": offer.max_lora_rank,
                "cost": dict(offer.cost), "renderers": rendering,
                "families": sorted({_family(each) for each in rendering}),
            })  # fmt: skip
        inference.append({
            "name": name, "kind": provider.kind, "gpus": provider.gpus, "replicas": provider.replicas,
            "allocation": provider.allocation, "concurrency": provider.concurrency, "capabilities": capabilities,
            "models": models, "weights": [each for each in WEIGHTS if serves(provider, each) is None],
            "pods": _pods_offered(cluster, name),
        })  # fmt: skip
    pairs: list[dict[str, Any]] = []
    for trainer_name, trainer in cluster.trainers.items():
        for name, provider in cluster.inference.items():
            found = path(trainer.capabilities.format, provider.capabilities.loads)
            unserved = serves(provider, trainer.capabilities.produces)
            together = trainer.colocate_with == name  # (they share one machine)
            pair: dict[str, Any] = {"trainer": trainer_name, "inference": name, "together": together}
            if isinstance(found, NoBridge):
                pairs.append({**pair, "bridge": None, "refused": found.reason})
            elif unserved is not None:
                pairs.append({**pair, "bridge": None, "refused": unserved})
            else:
                pairs.append({**pair, "bridge": [each.name for each in found]})
    presets = presets_of(ledger)
    listed = await presets.all() if presets is not None else []
    kept = [
        {"name": each.name, "version": each.version, "id": each.id, "settings": dict(each.settings), "note": each.note,
         "saved": each.saved}
        for each in listed
    ]  # fmt: skip
    return {
        "cluster": cluster.name,
        "kinds": ["train", "eval", "imitate", "check"],
        "submits": "kubernetes" if cluster.kubernetes is not None else "ray",
        "environments": environments,
        "trainers": trainers,
        "inference": inference,
        "pairs": pairs,
        "sandboxes": {kind: {"size": each.size, "provider": each.provider} for kind, each in cluster.sandboxes.items()},
        "presets": kept,
        "capacity": capacity_of(beats),
        "objectives": _objectives(),
        "schema": _schema(),
    }


def _pods_offered(cluster: Cluster, name: str) -> dict[str, JsonValue] | None:
    """What a RunPod provider's pods are, as the form shows them: the GPU type (the first of those it asks for, and
    all of them), the hourly price a pod is reckoned at, the cloud tier, the regions, the most pods at once, and how
    long a released pod stays warm; none for another kind (a trainer on a host's pods: the host's)."""
    from rollout_train.providers import RUNPOD, pod_table

    provider = cluster.inference.get(name) or cluster.trainers.get(name)
    if provider is None or provider.kind not in RUNPOD:
        return None
    host = getattr(provider, "colocate_with", None)
    if host is not None:
        return _pods_offered(cluster, host)
    table = pod_table(provider.kind, provider.settings)
    return {
        "gpu": table.gpu_types[0], "gpu_types": list(table.gpu_types), "gpu_count": table.gpu_count,
        "price": table.price, "cloud": table.cloud, "regions": list(table.regions), "max_pods": table.max_pods,
        "idle_stop": table.idle_stop, "host": host,
    }  # fmt: skip


def _objectives() -> dict[str, Any]:
    """The objective's families, its presets (each with its family, source and components) and the components, each
    with the families that accept it (`rollout_train.objectives`)."""
    from rollout_train.objectives import COMPONENTS, FAMILIES, PRESETS

    return {
        "families": list(FAMILIES),
        "presets": [
            {"name": name, "family": each.objective.family, "source": each.source, "says": each.says,
             "components": each.objective.components()}
            for name, each in PRESETS.items()
        ],
        "components": [
            {"key": each.key, "types": list(each.types), "families": sorted(each.families),
             "changeable": each.changeable, "says": each.says, "choices": list(each.choices), "least": each.least,
             "above": each.above}
            for each in COMPONENTS
        ],
    }  # fmt: skip


def _schema() -> list[dict[str, Any]]:
    """The keys a training run takes (`rollout_train.run_settings.KEYS`), each with its types, default and whether it
    is changeable."""
    from rollout_train.run_settings import KEYS

    return [
        {"key": each.pattern, "types": list(each.types), "default": each.default, "changeable": each.changeable,
         "says": each.says, "choices": list(each.choices), "least": each.least}
        for each in KEYS
        if "train" in each.kinds and not each.pattern.startswith("objective.")
    ]  # fmt: skip


async def _renderers(ledger: Ledger) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The renderers runs and presets named for each model so far, by model; and the renderer families they named for
    each environment's trained channel, by environment."""
    from rollout_train.presets import presets_of
    from rollout_train.record import recorded_settings, runs_in

    found: dict[str, set[str]] = {}
    played: dict[str, set[str]] = {}

    def note(said: Mapping[str, Any]) -> None:
        for key, value in said.items():
            if key.startswith("channels.") and key.endswith(".model") and isinstance(value, str):
                renderer = said.get(key.removesuffix(".model") + ".renderer")
                if isinstance(renderer, str):
                    found.setdefault(value, set()).add(renderer)
        environment = said.get("environment")
        trained = said.get(f"channels.{said.get('trainer.channel') or 'policy'}.renderer")
        if isinstance(environment, str) and isinstance(trained, str):
            played.setdefault(environment, set()).add(_family(trained))

    presets = presets_of(ledger)
    for each in await presets.all() if presets is not None else []:
        note(each.settings)
    with contextlib.suppress(Exception):  # (a ledger that cannot list its runs: none named)
        for run in await runs_in(ledger):
            recorded = await recorded_settings(ledger, run)
            if recorded is not None:
                note(recorded)
    by_model = {model: sorted(each) for model, each in found.items()}
    return by_model, {environment: sorted(each) for environment, each in played.items()}
