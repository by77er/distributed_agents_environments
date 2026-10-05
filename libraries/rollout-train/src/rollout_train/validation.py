"""Validating a run's settings against a cluster, in one pure function.

`check(settings, cluster, environment, ledger)` says everything wrong with a run's settings (`Finding`s, each with the
rule it breaks, the key it is about and a reason a person can act on), or nothing. It reads no file, opens no
connection and starts nothing: what it needs to know beyond the settings and the cluster is gathered beforehand, as
facts: the environment's (`EnvironmentFacts`: whether it loads, the sandboxes and tool sets it needs, its slots, how
long its episodes run) and the ledger's (`LedgerFacts`: the checkpoints the settings name and their formats, the
suites, the names taken, the shared pools' use, the cluster's capacity).

A finding refuses the run unless it says it does not (`refuses`): a run that only waits (for a GPU, a pool's slots)
is told so and not refused. `RULES` lists the rules in the order findings are reported (docs/guide/cluster.md says
when each refuses): settings, providers, auth, capabilities, bridge, weights, models, rank, segment, start, objective,
evals, distillation, environment, capacity, pools, spend, name.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from rollout_train.algorithm import needs_of
from rollout_train.bridges import AUTO, Bridge, NoBridge, path, rank_factor
from rollout_train.cluster import Cluster, auth_problem
from rollout_train.distillation import routes_of
from rollout_train.objectives import DEFAULT, POLICY_GRADIENT, Objective, composed
from rollout_train.providers import InferenceProvider, SettingSpec, TrainerProvider, settings_of
from rollout_train.registry import Taken, valid
from rollout_train.run_settings import KINDS, TRAINING, RunSettings, is_trainers, key_of, objective_in
from rollout_train.slots import Declared
from rollout_train.slots import problems as slot_problems

__all__ = [
    "RULES",
    "CheckpointFacts",
    "EnvironmentFacts",
    "Finding",
    "LedgerFacts",
    "PoolUse",
    "Rule",
    "SuiteFacts",
    "check",
    "estimated_spend",
    "refusals",
]


@dataclass(frozen=True)
class Finding:
    """One thing wrong with a run's settings: the rule, the key it is about (the field a form marks), and why."""

    rule: str
    key: str
    reason: str
    refuses: bool = True
    """False: the run may go (it waits, or something could not be known); the finding is a note."""


@dataclass(frozen=True)
class Rule:
    name: str
    refuses: str
    """When it refuses a run, in words."""


RULES: tuple[Rule, ...] = (
    Rule("settings", "a key the kind does not take, a wrong type or range, a required key missing, contradictions"),
    Rule("providers", "the trainer or a channel's provider is not offered"),
    Rule("auth", "a provider reached with no auth away from this machine"),
    Rule(
        "capabilities",
        "the trained channel's provider is not token-exact (a policy gradient), or lacks sampled logprobs and honoured "
        "sampling (an importance correction)",
    ),
    Rule("bridge", "no bridge from the checkpoint's format to what the provider loads"),
    Rule("weights", "adapters for a provider without adapters, full weights for one without full reload"),
    Rule("models", "a model not offered, or not the one trained"),
    Rule("rank", "the adapter's rank, as the provider sees it, above its highest"),
    Rule("segment", "segments longer than the trainer or the context takes"),
    Rule("start", "the start does not exist, was released, or is in a format the trainer cannot start from"),
    Rule(
        "objective",
        "a component its family does not accept, a combination that means nothing, a family the trainer or the kind of "
        "run does not take, a reference, an entropy or logprobs of tokens not sampled that the trainer cannot give",
    ),
    Rule("evals", "a suite that does not exist, or whose environment is not offered"),
    Rule(
        "distillation",
        "no teacher for a route or for the environment played, a teacher without the logprobs distillation reads or "
        "whose logprobs are unchecked, or of another renderer family",
    ),
    Rule("environment", "not offered, does not load, or needs sandboxes or tool sets the cluster lacks"),
    Rule("capacity", "more GPUs than the cluster has"),
    Rule("pools", "more adapter slots than a shared pool has"),
    Rule("spend", "a spend limit below one step's estimated cost"),
    Rule("name", "not a name, or taken"),
)
"""Every rule `check` applies, in the order it reports them."""


@dataclass(frozen=True)
class EnvironmentFacts:
    """What the environment's worker says of it, asked beforehand."""

    name: str
    loads: bool = True
    why: str = ""
    """Why it does not load, where it does not."""
    sandboxes: frozenset[str] = frozenset()
    """The sandbox kinds its programs need."""
    tool_sets: frozenset[str] = frozenset()
    """The tool sets its programs import by name that are served elsewhere (`[tools.NAME]`)."""
    slots: frozenset[str] | None = None
    """Its programs' slots (none: not known)."""
    untrained: frozenset[str] = frozenset()
    """Those of its slots that are not trained (a judge, a fixed opponent): each must be bound by name."""
    judges: frozenset[str] = frozenset()
    """Those of its slots that judge: bound to a channel serving the run's own checkpoints only with `self_judging`."""
    episodes_per_group: int | None = None
    turns_per_episode: float | None = None
    prompt_tokens: int | None = None
    """Prompt tokens of a turn, on average: with the two above, what a step's spend is estimated from."""


@dataclass(frozen=True)
class CheckpointFacts:
    """A checkpoint a run's settings name (its start, a fixed channel's), as the ledger has it."""

    reference: str
    exists: bool = True
    released: bool = False
    """Its weights were deleted."""
    formats: frozenset[str] = frozenset()
    """The formats its files are in (`rollout_train.bridges.format_of`)."""
    model: str | None = None
    """The model it was trained over."""


@dataclass(frozen=True)
class SuiteFacts:
    name: str
    newest: int
    """Its newest version's number."""
    environments: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PoolUse:
    """A shared pool's use by the runs bound to it now."""

    runs: int = 0
    slots: int = 0
    """Adapter slots they hold."""
    shares: float = 0.0
    """The sum of their shares."""


@dataclass(frozen=True)
class LedgerFacts:
    """What the ledger, and the launchers' offers beside it, say, asked beforehand."""

    checkpoints: Mapping[str, CheckpointFacts] = field(default_factory=dict[str, CheckpointFacts])
    """Every checkpoint reference the settings name, looked up (one not here does not exist)."""
    suites: Mapping[str, SuiteFacts] = field(default_factory=dict[str, SuiteFacts])
    names_taken: frozenset[str] = frozenset()
    pools: Mapping[str, PoolUse] = field(default_factory=dict[str, PoolUse])
    """By provider."""
    gpus: float | None = None
    """GPUs the cluster has in all (none: not known)."""
    gpus_free: float | None = None


def refusals(findings: list[Finding]) -> list[Finding]:
    """The findings that refuse the run."""
    return [each for each in findings if each.refuses]


@dataclass
class _Run:
    """What the rules share: the settings, the cluster, the facts, and what earlier rules worked out."""

    settings: RunSettings
    cluster: Cluster
    environment: EnvironmentFacts | None
    ledger: LedgerFacts
    found: list[Finding] = field(default_factory=list[Finding])
    trainer: TrainerProvider | None = None
    specs: tuple[SettingSpec, ...] | None = None
    chains: dict[tuple[str, str], tuple[Bridge, ...]] = field(default_factory=dict[tuple[str, str], tuple[Bridge, ...]])
    """The bridges chosen, by channel and provider."""

    def refuse(self, rule: str, key: str, reason: str) -> None:
        self.found.append(Finding(rule, key, reason))

    def note(self, rule: str, key: str, reason: str) -> None:
        self.found.append(Finding(rule, key, reason, refuses=False))

    @property
    def kind(self) -> str:
        return self.settings.kind

    def providers(self, channel: str) -> list[tuple[str, InferenceProvider]]:
        """The channel's providers the cluster offers."""
        return [(name, self.cluster.inference[name]) for name in self.settings.providers(channel)
                if name in self.cluster.inference]  # fmt: skip

    def setting(self, key: str) -> JsonValue:
        """A trainer's own setting as the run has it: given, else its trainer's default."""
        if key in self.settings.values:
            return self.settings.values[key]
        spec = next((each for each in self.specs or () if each.key == key), None)
        return spec.default if spec is not None else None

    def serving(self) -> list[str]:
        """The channels that serve the run's own checkpoints: the trained one, and those that follow it."""
        trained = self.settings.trained
        if trained is None:
            return []
        return [name for name in self.settings.channels if self.root(name) == trained]

    def root(self, channel: str) -> str | None:
        seen: set[str] = set()
        while channel not in seen:
            seen.add(channel)
            mode = self.settings.mode(channel)
            if mode == "trained":
                return channel
            if mode != "follows":
                return None
            channel = str(self.settings[f"channels.{channel}.follows"])
        return None


def check(
    settings: RunSettings,
    cluster: Cluster,
    environment: EnvironmentFacts | None = None,
    ledger: LedgerFacts | None = None,
) -> list[Finding]:
    """Everything wrong with a run's settings on this cluster, given what is known of its environment and the ledger;
    empty when nothing is. A finding whose `refuses` is false is a note: the run may go."""
    run = _Run(settings, cluster, environment, ledger or LedgerFacts())
    if settings.kind not in KINDS:
        run.refuse("settings", "kind", f"kind is one of {', '.join(KINDS)}, not {settings.kind!r}")
        return run.found
    trainer_name = settings["trainer.provider"]
    if settings.kind in TRAINING and isinstance(trainer_name, str):
        run.trainer = cluster.trainers.get(trainer_name)
        if run.trainer is not None:
            try:
                run.specs = settings_of(run.trainer.runs)
            except ImportError as error:
                run.note(
                    "settings",
                    "trainer.provider",
                    f"the {trainer_name} trainer's settings could not be read here ({error}): its trainer.* keys are "
                    "checked where it runs",
                )
    for rule in _RULES.values():
        rule(run)
    order = {each.name: index for index, each in enumerate(RULES)}
    return sorted(run.found, key=lambda each: order[each.rule])


def _settings(run: _Run) -> None:
    settings, kind = run.settings, run.kind
    for key, value in settings.values.items():
        if is_trainers(key):
            if kind not in TRAINING:
                run.refuse("settings", key, f"{key} is a trainer's setting, and a {kind} run trains nothing")
            elif run.trainer is not None and run.specs is not None:
                spec = next((each for each in run.specs if each.key == key), None)
                field_name = key.removeprefix("trainer.")
                if spec is None:
                    who = run.trainer.runs.not_settings.get(field_name)
                    reason = f": {who}" if who else ""
                    run.refuse("settings", key, f"the {run.trainer.name} trainer takes no {key}{reason}")
                elif not spec.accepts(value):
                    run.refuse("settings", key, f"{key} is {' or '.join(spec.types)}, not {value!r}")
            continue
        found = key_of(key)
        if found is None:
            run.refuse("settings", key, f"{key} is not a run setting")
        elif kind not in found.kinds:
            run.refuse("settings", key, f"{key} is not a setting a {kind} run takes")
        elif (problem := found.problem(value)) is not None:
            run.refuse("settings", key, f"{key} {problem}")
    required = {
        "train": ["environment", "trainer.provider"],
        "eval": ["environment", "eval.suite"],
        "imitate": ["trainer.provider", "imitation.dataset"],
        "check": ["environment"],
    }[kind]
    for key in required:
        if settings[key] is None:
            run.refuse("settings", key, f"a {kind} run needs {key}")
    if kind == "train" and (trained := settings.trained) is not None:
        if not settings.providers(trained):
            run.refuse("settings", f"channels.{trained}.provider", f"the trained channel {trained} needs a provider")
        if settings.get(f"channels.{trained}.model") is None:
            run.refuse("settings", f"channels.{trained}.model", f"the trained channel {trained} needs a model")
    for channel in settings.channels:
        _channel(run, channel)
    for key, value in settings.values.items():
        if key.startswith("slots.") and isinstance(value, str):
            slot = key.removeprefix("slots.")
            if value not in settings.channels:
                run.refuse("settings", key, f"slot {slot} samples channel {value}, which the settings do not describe")
            known = run.environment.slots if run.environment is not None else None
            if known is not None and slot not in known:
                run.refuse(
                    "settings", key, f"the environment's programs have no slot {slot} ({', '.join(sorted(known))})"
                )
    if run.environment is not None and run.environment.slots is not None:  # (rollout_train.slots)
        declared = Declared(run.environment.slots, run.environment.untrained, run.environment.judges)
        for key, reason in slot_problems(settings, declared, run.serving()):
            run.refuse("settings", key, reason)


def _channel(run: _Run, channel: str) -> None:
    settings, prefix = run.settings, f"channels.{channel}."
    several = settings[prefix + "providers"]
    if several is not None and settings[prefix + "provider"] is not None:
        run.refuse(
            "settings", prefix + "providers", f"channel {channel} names its providers once: provider or providers"
        )
    if isinstance(several, list) and (not several or not all(isinstance(each, str) for each in several)):
        run.refuse(
            "settings", prefix + "providers", f"channel {channel}'s providers is a list of at least one provider"
        )
    providers = settings.providers(channel)
    weights = settings[prefix + "weights"]
    if settings[prefix + "routing"] == "weighted":
        if (
            not isinstance(weights, dict)
            or set(weights) != set(providers)
            or not all(
                isinstance(each, int | float) and not isinstance(each, bool) and each > 0 for each in weights.values()
            )
        ):
            run.refuse(
                "settings", prefix + "weights",
                f"channel {channel} shares turns by weight: give each of its providers ({', '.join(providers)}) a "
                "weight above 0",
            )  # fmt: skip
    elif weights is not None:
        run.refuse(
            "settings", prefix + "weights", f"weights are for routing = weighted (channel {channel} spills over)"
        )
    mode = settings[prefix + "mode"]
    if channel == settings.trained and mode is not None:
        run.refuse("settings", prefix + "mode", f"{channel} is the trained channel: it serves what the run trains")
        return
    follows = settings[prefix + "follows"]
    if mode == "follows":
        if follows is None or follows not in settings.channels or follows == channel:
            run.refuse("settings", prefix + "follows", f"channel {channel} follows another channel of the run: name it")
        elif run.root(channel) is None:
            run.refuse(
                "settings", prefix + "follows", f"channel {channel} follows a channel that serves no checkpoints"
            )
    elif follows is not None or settings[prefix + "lag"] != 0:
        run.refuse("settings", prefix + "follows", f"follows and lag are for mode = follows (channel {channel})")
    if settings[prefix + "checkpoint"] is not None and settings.mode(channel) != "fixed":
        run.refuse("settings", prefix + "checkpoint", f"a checkpoint is what a fixed channel serves ({channel})")


def _providers(run: _Run) -> None:
    cluster = run.cluster
    trainer = run.settings["trainer.provider"]
    if run.kind in TRAINING and isinstance(trainer, str) and run.trainer is None:
        offered = ", ".join(cluster.trainers) or "none"
        run.refuse("providers", "trainer.provider", f"the cluster offers no trainer {trainer} (it offers {offered})")
    for channel in run.settings.channels:
        for name in run.settings.providers(channel):
            if name not in cluster.inference:
                key = f"channels.{channel}.provider"
                offered = ", ".join(cluster.inference) or "none"
                run.refuse("providers", key, f"the cluster offers no inference provider {name} (it offers {offered})")


def _auth(run: _Run) -> None:
    if run.trainer is not None:
        problem = auth_problem(f"trainer {run.trainer.name}", run.trainer.auth, ())
        if problem:
            run.refuse("auth", "trainer.provider", problem)
    for channel in run.settings.channels:
        for name, provider in run.providers(channel):
            if problem := auth_problem(f"provider {name}", provider.auth, provider.endpoints):
                run.refuse("auth", f"channels.{channel}.provider", problem)


def _objective_of(run: _Run) -> Objective | None:
    """The objective the run's settings ask for, if it can be made (what is wrong with it is `_objective`'s to say)."""
    try:
        return objective_in(run.settings)
    except ValueError:
        return None


def _capabilities(run: _Run) -> None:
    trained = run.settings.trained
    if trained is None:
        return
    objective = _objective_of(run) or DEFAULT
    needs = needs_of(objective)
    if not needs:  # (a preference loss and a likelihood read neither exact tokens nor behaviour logprobs)
        return
    weighs = "sampled_logprobs" in needs
    for name, provider in run.providers(trained):
        offered = provider.capabilities
        key = f"channels.{trained}.provider"
        if weighs and not offered.token_exact and not offered.sampled_logprobs:
            run.refuse(
                "capabilities", key,
                f"channel {trained} is trained, and provider {name} ({provider.kind}) returns text, not the sampled "
                "token ids and their logprobs, which the importance weight needs",
            )  # fmt: skip
            continue
        if not weighs:
            if not offered.token_exact:
                run.refuse(
                    "capabilities", key,
                    f"channel {trained} is trained by a policy gradient, and provider {name} ({provider.kind}) does "
                    "not return the exact tokens it sampled",
                )  # fmt: skip
            continue
        lacks = [
            what
            for what, has in (
                ("token-exact output", offered.token_exact),
                ("sampled-token logprobs", offered.sampled_logprobs),
                ("honoured temperature and top-p", offered.honours_sampling),
            )
            if not has
        ]
        if lacks:
            run.refuse(
                "capabilities", key,
                f"channel {trained} is trained, and provider {name} ({provider.kind}) lacks {' and '.join(lacks)}: "
                "the importance weight needs the behaviour logprob of each exact sampled token",
            )  # fmt: skip


def _sources(run: _Run, channel: str) -> tuple[frozenset[str], str] | None:
    """The formats a channel's provider must load, and whose they are: the trainer's for a channel serving the run's
    checkpoints, a checkpoint's for a fixed channel serving one (or an eval's subject)."""
    if channel in run.serving():
        if run.trainer is None:
            return None
        return frozenset({run.trainer.capabilities.format}), f"the {run.trainer.name} trainer's"
    reference = run.settings[f"channels.{channel}.checkpoint"]
    if reference is None and run.kind == "eval":
        reference = run.settings["start"]
    if reference is None:
        return None
    facts = run.ledger.checkpoints.get(str(reference))
    if facts is None or not facts.exists or not facts.formats:
        return None
    return facts.formats, f"checkpoint {reference}'s"


def _bridge(run: _Run) -> None:
    for channel in run.settings.channels:
        sources = _sources(run, channel)
        if sources is None:
            continue
        formats, whose = sources
        wanted = str(run.settings[f"channels.{channel}.bridge"] or AUTO)
        for name, provider in run.providers(channel):
            found: tuple[Bridge, ...] | NoBridge = NoBridge("", "", "")
            for each in sorted(formats):
                found = path(each, provider.capabilities.loads, wanted=wanted)
                if not isinstance(found, NoBridge):
                    break
            if isinstance(found, NoBridge):
                loads = ", ".join(sorted(provider.capabilities.loads)) or "no checkpoint"
                run.refuse(
                    "bridge", f"channels.{channel}.provider",
                    f"no bridge from {whose} {' or '.join(sorted(formats))} checkpoints to what provider {name} loads "
                    f"({loads}): {found.reason}",
                )  # fmt: skip
            else:
                run.chains[(channel, name)] = found


def _weights(run: _Run) -> None:
    for (channel, name), chain in run.chains.items():
        provider = run.cluster.inference[name]
        target = chain[-1].target
        if target in ("peft", "tinker") and not provider.capabilities.adapters:
            run.refuse("weights", f"channels.{channel}.provider", f"provider {name} serves no adapters, and {channel} "
                       "serves adapters")  # fmt: skip
        if target == "full" and not provider.capabilities.full_reload:
            run.refuse("weights", f"channels.{channel}.provider", f"provider {name} cannot reload full weights, and "
                       f"{channel} serves full weights")  # fmt: skip


def _models(run: _Run) -> None:
    settings, trainer = run.settings, run.trainer
    trainer_model = settings.trainer_model
    if trainer is not None and trainer_model is not None and trainer_model not in trainer.models:
        run.refuse("models", "trainer.model", f"the {trainer.name} trainer does not train {trainer_model} here "
                   f"(it trains {', '.join(trainer.models)})")  # fmt: skip
    serving = run.serving()
    for channel in settings.channels:
        model = settings.get(f"channels.{channel}.model")
        if model is None:
            continue
        for name, provider in run.providers(channel):
            offer = provider.models.get(str(model))
            if offer is None:
                run.refuse("models", f"channels.{channel}.model", f"provider {name} does not serve {model} "
                           f"(it serves {', '.join(provider.models)})")  # fmt: skip
            elif channel in serving and trainer_model is not None and trainer_model not in (offer.model, offer.base):
                run.refuse(
                    "models", f"channels.{channel}.model",
                    f"channel {channel} serves the run's checkpoints, trained over {trainer_model}, and {model} is "
                    "neither that model nor quantized from it",
                )  # fmt: skip
    start = settings["start"]
    facts = run.ledger.checkpoints.get(str(start)) if start is not None else None
    if run.kind in TRAINING and facts is not None and facts.model and trainer_model and facts.model != trainer_model:
        run.refuse("models", "start", f"the start {start} was trained over {facts.model}, and this run trains "
                   f"{trainer_model}")  # fmt: skip


def _rank(run: _Run) -> None:
    trainer = run.trainer
    if trainer is None or trainer.capabilities.produces != "lora":
        return
    rank = run.setting("trainer.rank")
    model = run.settings.trainer_model
    if not isinstance(rank, int) or model is None:
        return
    for (channel, name), chain in run.chains.items():
        served = run.settings.get(f"channels.{channel}.model")
        offer = run.cluster.inference[name].models.get(str(served))
        if offer is None or offer.max_lora_rank is None or chain[-1].target != "peft":
            continue
        factor = rank_factor(chain, model)
        if rank * factor > offer.max_lora_rank:
            times = f" times {factor} (the bridge's rank factor for {model})" if factor != 1 else ""
            run.refuse(
                "rank", "trainer.rank",
                f"trainer.rank {rank}{times} is {rank * factor}, above provider {name}'s max_lora_rank "
                f"{offer.max_lora_rank} for {served}",
            )  # fmt: skip


def _segment(run: _Run) -> None:
    trainer = run.trainer
    if trainer is None:
        return
    said = run.setting("trainer.segment_tokens")
    if isinstance(said, int) and trainer.segment_tokens is not None and said > trainer.segment_tokens:
        run.refuse("segment", "trainer.segment_tokens", f"trainer.segment_tokens {said} is above the "
                   f"{trainer.segment_tokens} the {trainer.name} trainer takes here")  # fmt: skip
    longest = said if isinstance(said, int) else None  # (unsaid: the trainer takes what the context leaves)
    trained = run.settings.trained
    if longest is None or trained is None:
        return
    model = run.settings.get(f"channels.{trained}.model")
    for name, provider in run.providers(trained):
        offer = provider.models.get(str(model))
        if offer is not None and longest > offer.context:
            run.refuse("segment", "trainer.segment_tokens", f"segments of up to {longest} tokens are longer than "
                       f"{model}'s context on provider {name} ({offer.context})")  # fmt: skip


def _start(run: _Run) -> None:
    start = run.settings["start"]
    if start is None:
        return
    facts = run.ledger.checkpoints.get(str(start))
    if facts is None or not facts.exists:
        run.refuse("start", "start", f"there is no checkpoint {start}")
        return
    if facts.released:
        run.refuse("start", "start", f"{start} was released: its weights were deleted")
        return
    trainer = run.trainer
    if run.kind not in TRAINING or trainer is None or facts.formats & trainer.capabilities.starts_from:
        return
    if trainer.capabilities.produces == "full" and "peft" in facts.formats:
        reason = f"a full-weight trainer starts from full weights: merge this adapter first (`rollout merge {start}`)"
    elif trainer.capabilities.format == "tinker":
        reason = f"Tinker trains only from checkpoints Tinker made, and {start} is not one: there is no upload"
    elif "tinker" in facts.formats:
        reason = f"{start} is Tinker's: bridge it to PEFT first, or start a new line"
    else:
        reason = f"the {trainer.name} trainer cannot start from {start}'s files ({', '.join(sorted(facts.formats))})"
    run.refuse("start", "start", reason)


def _objective(run: _Run) -> None:
    if run.kind not in TRAINING:
        return
    settings = run.settings
    preset = settings["objective.preset"]
    given = {
        key.removeprefix("objective."): value
        for key, value in settings.values.items()
        if key.startswith("objective.") and key != "objective.preset" and value is not None
    }
    try:
        objective, problems = composed(str(preset), given)
    except ValueError:  # (a preset or a value the schema does not take: the settings rule said so)
        return
    for key, reason in problems:
        run.refuse("objective", f"objective.{key}", reason)
    if run.kind == "imitate" and objective.family == POLICY_GRADIENT and preset != DEFAULT.preset:
        run.refuse("objective", "objective.preset", "an imitate run trains on a dataset's examples, which have no "
                   f"advantages: a likelihood or preference preset, not {preset}")  # fmt: skip
    trainer = run.trainer
    if trainer is None:
        return
    offered = trainer.capabilities
    if objective.family not in offered.families:
        run.refuse("objective", "objective.preset", f"the {trainer.name} trainer does not take a {objective.family} "
                   f"objective (it takes {', '.join(sorted(offered.families))})")  # fmt: skip
    if objective.needs_reference:
        key = "objective.reference"
        if offered.reference == "no":
            sdk = " (Tinker's SDK offers prompt logprobs from a sampler of the base model, not yet confirmed by a live "
            sdk += "test)"
            why = sdk if offered.format == "tinker" else ""
            run.refuse("objective", key, f"the {trainer.name} trainer gives no reference logprobs{why}, and the "
                       f"objective reads them ({_reads(objective)})")  # fmt: skip
        elif offered.reference == "asked" and run.setting("trainer.frozen_reference") is not True:
            run.refuse("objective", key, f"the {trainer.name} trainer holds a reference only when asked, and the "
                       f"objective reads it ({_reads(objective)}): trainer.frozen_reference = true keeps a frozen copy "
                       "of the model beside the policy")  # fmt: skip
    if objective.needs_entropy and not offered.entropy:
        run.refuse("objective", "objective.entropy.coefficient", f"the {trainer.name} trainer gives no entropies, and "
                   "an entropy bonus reads them")  # fmt: skip
    if objective.needs_distribution and not offered.distribution:
        run.refuse("objective", "objective.distillation.form", f"the {trainer.name} trainer gives the logprobs of the "
                   "sampled tokens only, and the top_k form reads the student's logprobs of the teacher's top-k "
                   "tokens: distillation.form = policy_gradient")  # fmt: skip


def _reads(objective: Objective) -> str:
    return "a KL to the reference" if objective.family == POLICY_GRADIENT else f"the {objective.preference.loss} loss"


def _evals(run: _Run) -> None:
    for key in ("evals.suite", "eval.suite"):
        reference, schema = run.settings[key], key_of(key)
        if not isinstance(reference, str) or schema is None or run.kind not in schema.kinds:
            continue
        name, _, number = reference.partition("@")
        suite = run.ledger.suites.get(name)
        if suite is None:
            run.refuse("evals", key, f"there is no suite {name}: make it with `rollout suite make {name} …` (a name "
                       "never becomes a suite by itself)")  # fmt: skip
            continue
        if number and (not number.isdigit() or not 1 <= int(number) <= suite.newest):
            run.refuse("evals", key, f"suite {name} has versions 1 to {suite.newest}, not {number}")
        for environment in sorted(suite.environments - set(run.cluster.environments)):
            run.refuse("evals", key, f"suite {name} plays {environment}, which this cluster does not offer")


def _distillation(run: _Run) -> None:
    """A training run whose objective distills: a teacher channel for each route, covering the environment it plays,
    each served by providers that return the logprobs it reads, in the student's renderer family. (A step on a dataset
    of teacher samples reads the scores its segments carry, and asks no teacher.)"""
    objective = _objective_of(run)
    if run.kind != "train" or objective is None or not objective.distills:
        return
    key = "objective.distillation.teachers"
    teachers = objective.distillation.teachers
    if not teachers:
        run.refuse("distillation", key, "a distillation needs a teacher channel for each route it plays: "
                   'objective.distillation.teachers = {"*" = "teacher"}, say')  # fmt: skip
        return
    environment = run.settings["environment"]
    if isinstance(environment, str):
        covered = routes_of(teachers, environment)
        if covered == "none":
            run.refuse("distillation", key, f"no route names {environment}, which the run plays: route it, or every "
                       "environment (`*`)")  # fmt: skip
        elif covered == "some":
            run.note("distillation", key, f"only some rows of {environment} are routed: an episode of another row is "
                     "not scored, and trains nothing")  # fmt: skip
    if run.trainer is not None and not run.trainer.capabilities.scores:
        run.refuse("distillation", key, f"the {run.trainer.name} trainer does not score tokens, which distillation "
                   "needs")  # fmt: skip
    top_k = objective.needs_top
    student = run.settings.get(f"channels.{run.settings.trained}.renderer")
    for teacher in sorted(set(teachers.values())):
        if teacher not in run.settings.channels or not run.settings.providers(teacher):
            run.refuse("distillation", key, f"the teacher's channel {teacher} has no provider")
            continue
        for name, provider in run.providers(teacher):
            offered = provider.capabilities
            unchecked = sorted(offered.unchecked & {"prompt_logprobs", "top_logprobs"})
            if unchecked:
                which = " and ".join(each.replace("_", " ") for each in unchecked)
                run.refuse("distillation", key, f"provider {name}'s {which} are declared by its SDK but not yet "
                           "confirmed by a live test: it cannot teach until they are")  # fmt: skip
                continue
            if not offered.prompt_logprobs:
                run.refuse("distillation", key, f"the teacher's provider {name} ({provider.kind}) does not return "
                           "prompt logprobs, to score the student's tokens")  # fmt: skip
            elif top_k and offered.top_logprobs < top_k:
                run.refuse("distillation", key, f"the teacher's provider {name} ({provider.kind}) returns at most "
                           f"{offered.top_logprobs} top logprobs a position, and the objective reads {top_k} (its "
                           "max_logprobs)")  # fmt: skip
        renderer = run.settings.get(f"channels.{teacher}.renderer")
        if isinstance(student, str) and isinstance(renderer, str) and _family(student) != _family(renderer):
            run.refuse("distillation", f"channels.{teacher}.renderer", f"the teacher renders as {renderer}, of another "
                       f"family than the student's {student}: their tokens do not compare")  # fmt: skip


def _family(renderer: str) -> str:
    return renderer.partition(":")[0]


def _environment(run: _Run) -> None:
    environment = run.settings["environment"]
    if not isinstance(environment, str) or run.kind == "imitate":
        return
    if environment not in run.cluster.environments:
        offered = ", ".join(run.cluster.environments) or "none"
        run.refuse("environment", "environment", f"this cluster does not offer {environment} (it offers {offered})")
        return
    facts = run.environment
    if facts is None:
        return
    if not facts.loads:
        run.refuse("environment", "environment", f"{environment} does not load: {facts.why or 'its worker says so'}")
    for kind in sorted(facts.sandboxes - set(run.cluster.sandboxes)):
        run.refuse("environment", "environment", f"{environment} needs sandboxes of kind {kind}, and this cluster has "
                   "no pool of them")  # fmt: skip
    for name in sorted(facts.tool_sets - set(run.cluster.tools)):
        run.refuse("environment", "environment", f"{environment} imports the tool set {name}, which this cluster does "
                   "not serve ([tools])")  # fmt: skip


def _capacity(run: _Run) -> None:
    total, free = run.ledger.gpus, run.ledger.gpus_free
    if total is None:
        return
    engines: dict[str, float] = {}
    for channel in run.settings.channels:
        for name, provider in run.providers(channel):
            if provider.kind != "vllm" or run.ledger.pools.get(name, PoolUse()).runs > 0:
                continue  # (a pool already serving other runs needs no more GPUs)
            replicas = run.settings[f"channels.{channel}.replicas"]
            count = replicas if isinstance(replicas, int) else provider.replicas
            engines[name] = max(engines.get(name, 0.0), count * provider.gpus)
    trainer = run.trainer.gpus if run.trainer is not None else 0.0
    shared = run.trainer.colocate_with if run.trainer is not None else None
    if shared in engines:
        engines[shared] = max(engines[shared], trainer)
        trainer = 0.0
    needs = trainer + sum(engines.values())
    if needs > total:
        run.refuse("capacity", "trainer.provider", f"the run needs {needs:g} GPUs, and the cluster has {total:g}: it "
                   "would never start")  # fmt: skip
    elif free is not None and needs > free:
        run.note("capacity", "trainer.provider", f"the run needs {needs:g} GPUs, and {free:g} are free: it waits")


def _slots(run: _Run, channel: str) -> int:
    """The adapter slots a channel holds on a shared pool: `max_lag + 1` for the trained channel, 2 for one that follows
    another (what it serves, and the next), 1 for a fixed checkpoint (an eval's subject among them), none for the base
    model."""
    mode = run.settings.mode(channel)
    if mode == "trained":
        lag = run.settings["max_lag"]
        return (lag if isinstance(lag, int) else 1) + 1
    if mode == "follows":
        return 2
    fixed = run.settings[f"channels.{channel}.checkpoint"]
    return 1 if fixed is not None or (run.kind == "eval" and run.settings["start"] is not None) else 0


def _pools(run: _Run) -> None:
    needs: dict[str, int] = {}
    for channel in run.settings.channels:
        want = _slots(run, channel)
        if not want:
            continue
        for name, provider in run.providers(channel):
            if provider.pool is not None:
                needs[name] = needs.get(name, 0) + want
    for name, want in needs.items():
        pool = run.cluster.inference[name].pool
        assert pool is not None
        use = run.ledger.pools.get(name, PoolUse())
        key = "max_lag"
        if pool.adapter_slots is not None and want > pool.adapter_slots:
            run.refuse("pools", key, f"the run needs {want} adapter slots on {name} (max_lag + 1 for the trained "
                       f"channel, 2 for one following it, 1 for a fixed checkpoint), and the pool has "
                       f"{pool.adapter_slots}")  # fmt: skip
        elif pool.adapter_slots is not None and want > pool.adapter_slots - use.slots:
            run.note("pools", key, f"{name} has {pool.adapter_slots - use.slots} adapter slots free, and the run needs "
                     f"{want}: it waits")  # fmt: skip
        if pool.max_runs is not None and use.runs >= pool.max_runs:
            run.note("pools", key, f"{name} serves its most runs ({pool.max_runs}): the run waits")
        if run.trainer is not None and run.trainer.capabilities.produces == "full" and use.runs > 0:
            run.note("pools", "trainer.provider", f"full weights need {name}'s servers to themselves, and it serves "
                     f"{use.runs} runs: the run waits")  # fmt: skip
        share = run.settings["share"]
        if use.runs > 0 and isinstance(share, int | float):
            run.note("pools", "share", f"with share {share:g} it gets {share / (use.shares + share):.0%} of {name}'s "
                     "turns while all its runs are busy")  # fmt: skip


def estimated_spend(settings: RunSettings, cluster: Cluster, environment: EnvironmentFacts | None) -> float | None:
    """Dollars one step is estimated to cost, at most: every token trained (each turn's prompt and its sampled tokens,
    every turn filling its budgets) times the trainer's cost for the model, plus the sampled tokens times the dearest
    provider's cost to sample them and the prompts' tokens times its cost to read them, uncached. None where it cannot
    be estimated: budgets or the environment's numbers unknown, or a provider or trainer that bills by the hour."""
    trained = settings.trained
    trainer = cluster.trainers.get(str(settings["trainer.provider"]))
    if trained is None or trainer is None or environment is None:
        return None
    if environment.episodes_per_group is None or environment.turns_per_episode is None:
        return None
    thinking, answer = settings[f"channels.{trained}.thinking_tokens"], settings[f"channels.{trained}.answer_tokens"]
    if not isinstance(thinking, int) and not isinstance(answer, int):
        return None
    groups = settings["groups_per_step"]
    turns = (groups if isinstance(groups, int) else 4) * environment.episodes_per_group * environment.turns_per_episode
    sampled = turns * ((thinking if isinstance(thinking, int) else 0) + (answer if isinstance(answer, int) else 0))
    prompts = turns * (environment.prompt_tokens or 0)
    model = str(settings.get(f"channels.{trained}.model"))
    priced = trainer.cost_of(str(settings.get("trainer.model") or model))
    if "hour" in priced:
        return None
    spend = (sampled + prompts) * priced.get("train", 0.0) / 1e6
    dearest = 0.0
    for name in settings.providers(trained):
        provider = cluster.inference.get(name)
        offer = provider.models.get(model) if provider is not None else None
        if offer is None:
            continue
        if "hour" in offer.cost or (provider is not None and provider.capabilities.bills == "hours"):
            return None
        dearest = max(dearest, (sampled * offer.cost.get("output", 0.0) + prompts * offer.cost.get("input", 0.0)) / 1e6)
    return spend + dearest


def _spend(run: _Run) -> None:
    limit = run.settings["limits.spend"]
    if not isinstance(limit, int | float) or isinstance(limit, bool) or run.kind != "train":
        return
    estimate = estimated_spend(run.settings, run.cluster, run.environment)
    if estimate is None:
        run.note("spend", "limits.spend", "one step's spend could not be estimated before the run (budgets, the "
                 "environment's numbers, or an hourly price unknown): the run still ends once its spend reaches "
                 f"${limit:g}")  # fmt: skip
    elif estimate > limit:
        run.refuse("spend", "limits.spend", f"one step is estimated at up to ${estimate:.2f}, above limits.spend "
                   f"${limit:g}: the run would stop before its first step")  # fmt: skip


def _name(run: _Run) -> None:
    name = run.settings["name"]
    if not isinstance(name, str):
        return
    try:
        name = valid(name)
    except Taken as error:
        run.refuse("name", "name", str(error))
        return
    if name in run.ledger.names_taken:
        run.refuse("name", "name", f"another run is called {name!r}")


_RULES: Mapping[str, Callable[[_Run], None]] = {
    "settings": _settings,
    "providers": _providers,
    "auth": _auth,
    "capabilities": _capabilities,
    "bridge": _bridge,
    "weights": _weights,
    "models": _models,
    "rank": _rank,
    "segment": _segment,
    "start": _start,
    "objective": _objective,
    "evals": _evals,
    "distillation": _distillation,
    "environment": _environment,
    "capacity": _capacity,
    "pools": _pools,
    "spend": _spend,
    "name": _name,
}
assert tuple(_RULES) == tuple(each.name for each in RULES)
