"""Validating a run's settings against a cluster, in one pure function.

`check(settings, cluster, environment, ledger)` says everything wrong with a run's settings (`Finding`s, each with the
rule it breaks, the key it is about and a reason a person can act on), or nothing. It reads no file, opens no
connection and starts nothing: what it needs to know beyond the settings and the cluster is gathered beforehand, as
facts: the environment's (`EnvironmentFacts`: whether it loads, the sandboxes and tool sets it needs, its slots, how
long its episodes run) and the ledger's (`LedgerFacts`: the checkpoints the settings name and their formats, the
suites, the names taken, the GPUs the cluster has and what it has free).

A finding refuses the run unless it says it does not (`refuses`): a run that only waits (for a GPU) is told so and
not refused. `RULES` lists the rules in the order findings are reported (docs/guide/cluster.md says
when each refuses): settings, providers, auth, capabilities, bridge, weights, models, rank, segment, start, objective,
evals, distillation, environment, capacity, spend, name.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from pydantic import JsonValue

from rollout_train.algorithm import algorithm_for, needs_of
from rollout_train.bridges import AUTO, Bridge, NoBridge, path, rank_factor
from rollout_train.cluster import Cluster, auth_problem
from rollout_train.demand import HEADROOM, Demand, Resources, demand, played_channel, requested
from rollout_train.distillation import routes_of
from rollout_train.objectives import DEFAULT, POLICY_GRADIENT, Objective, composed
from rollout_train.providers import InferenceProvider, SettingSpec, TrainerProvider, settings_of
from rollout_train.published import is_published
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
    "Rule",
    "Spend",
    "SuiteEntryFacts",
    "SuiteFacts",
    "check",
    "completed",
    "estimated_spend",
    "refusals",
    "renderers_of",
    "serves",
    "spend_of",
    "weights_of",
    "with_renderers",
    "with_weights",
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
    Rule("providers", "the trainer or a channel's provider is not offered, or a hosted API shares a channel"),
    Rule("auth", "a provider reached with no auth away from this machine"),
    Rule(
        "capabilities",
        "the trained channel's provider is a hosted API (no exact tokens or behaviour logprobs, whatever the "
        "objective), is not token-exact (a policy gradient), or lacks sampled logprobs and honoured sampling (an "
        "importance correction)",
    ),
    Rule("bridge", "no bridge from the checkpoint's format to what the provider loads"),
    Rule(
        "weights",
        "a trainer that makes the other kind of weights than the run trains; a LoRA on a provider without adapters, "
        "full weights on one without full reload",
    ),
    Rule("models", "a model not offered, or not the one trained"),
    Rule(
        "renderer",
        "a channel sampling tokens whose model no renderer renders, that several do with none said, or "
        "a renderer said that says it renders other models",
    ),
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
    Rule(
        "environment",
        "not offered, does not load, needs sandboxes or tool sets the cluster lacks, or, on Kubernetes, sandboxes "
        "whose pool is not served from pods of its own",
    ),
    Rule(
        "capacity",
        "more than the cluster schedules for one run ([capacity]), or more GPUs than it has, counting "
        "the run's scheduled parts",
    ),
    Rule("spend", "a training run's spend limit below one step's estimated cost"),
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
    samples_per_turn: float = 1.0
    """Samples a turn takes: one for each model slot that samples in it (each agent of a team)."""
    prompt_tokens: int | None = None
    """Prompt tokens of a sample, on average: with the three above, what a step's spend is estimated from."""


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
class SuiteEntryFacts:
    """One entry of the version of a suite the settings name: what an eval of it plays."""

    environment: str
    starts: int
    episodes: int
    """Of each start, unless the eval says another number."""
    thinking_tokens: int | None = None
    answer_tokens: int | None = None


@dataclass(frozen=True)
class SuiteFacts:
    name: str
    newest: int
    """Its newest version's number."""
    environments: frozenset[str] = frozenset()
    entries: tuple[SuiteEntryFacts, ...] = ()
    """The entries of the version the settings name (its newest, where they name none)."""


@dataclass(frozen=True)
class LedgerFacts:
    """What the ledger and the live cluster say, asked beforehand."""

    checkpoints: Mapping[str, CheckpointFacts] = field(default_factory=dict[str, CheckpointFacts])
    """Every checkpoint reference the settings name, looked up (one not here does not exist)."""
    suites: Mapping[str, SuiteFacts] = field(default_factory=dict[str, SuiteFacts])
    names_taken: frozenset[str] = frozenset()
    gpus: float | None = None
    """GPUs the cluster has in all (none: not known)."""
    free: Resources | None = None
    """What the cluster has free now (none: not known)."""
    step_seconds: float | None = None
    """How long a step of the run's trainer and model took here lately (none: not known): what its pods' hours per
    step are reckoned from."""


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

    @property
    def weights(self) -> str | None:
        """What the run trains (`weights_of`)."""
        return weights_of(self.settings, self.cluster) if self.kind in TRAINING else None

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
        providers = run.settings.providers(channel)
        for name in providers:
            if name not in cluster.inference:
                key = f"channels.{channel}.provider"
                offered = ", ".join(cluster.inference) or "none"
                run.refuse("providers", key, f"the cluster offers no inference provider {name} (it offers {offered})")
        hosted = [name for name in providers if name in cluster.inference and cluster.inference[name].kind == "api"]
        if hosted and len(providers) > 1:
            run.refuse("providers", f"channels.{channel}.providers", f"channel {channel} is on the hosted API "
                       f"{hosted[0]}, which shares a channel with no other provider")  # fmt: skip


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
    for name, provider in run.providers(trained):
        offered = provider.capabilities
        if not offered.token_exact and not offered.sampled_logprobs:  # (a hosted API: nothing it samples is trained on)
            run.refuse(
                "capabilities", f"channels.{trained}.provider",
                f"channel {trained} is trained, and provider {name} ({provider.kind}) returns text, not the exact "
                "tokens it sampled or their behaviour logprobs: what it samples is never trained on. A hosted API "
                "serves evals and slots that are not trained, such as judges",
            )  # fmt: skip
    objective = _objective_of(run) or DEFAULT
    needs = needs_of(objective)
    if not needs:  # (a preference loss and a likelihood read neither exact tokens nor behaviour logprobs)
        return
    weighs = "sampled_logprobs" in needs
    for name, provider in run.providers(trained):
        offered = provider.capabilities
        key = f"channels.{trained}.provider"
        if not offered.token_exact and not offered.sampled_logprobs:
            continue  # (refused above)
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


_WEIGHTS = {"lora": "a LoRA", "full": "full weights"}


def weights_of(settings: RunSettings, cluster: Cluster) -> str | None:
    """What a run trains: its `weights`, else what its trainer makes (`lora` or `full`); none where neither is known."""
    said = settings["weights"]
    if isinstance(said, str):
        return said
    trainer = cluster.trainers.get(str(settings["trainer.provider"]))
    return trainer.capabilities.produces if trainer is not None else None


def with_weights(settings: RunSettings, cluster: Cluster) -> RunSettings:
    """A training run's settings with what it trains said (`weights_of`), as its start records them."""
    said = weights_of(settings, cluster)
    if settings.kind not in TRAINING or settings["weights"] is not None or said is None:
        return settings
    return RunSettings({**settings.values, "weights": said})


def renders_tokens(settings: RunSettings, cluster: Cluster, channel: str) -> bool:
    """Whether a channel's provider samples tokens, so the channel needs a renderer: not a hosted API, which takes
    messages and renders them itself."""
    provider = cluster.inference.get(str(settings.get(f"channels.{channel}.provider")))
    return provider is None or provider.capabilities.token_exact


def renderers_of(settings: RunSettings, cluster: Cluster, channel: str) -> list[str]:
    """The renderers that render a channel's model, as `module:name` (`rollout_train.recorder.renderers.renderers_for`):
    over the model its provider says a quantized model was made from, where none renders the model itself."""
    from rollout_train.recorder.renderers import renderers_for

    model = settings.get(f"channels.{channel}.model")
    if not isinstance(model, str):
        return []
    provider = cluster.inference.get(str(settings.get(f"channels.{channel}.provider")))
    offer = provider.models.get(model) if provider is not None else None
    return renderers_for(model, offer.base if offer is not None else None)


def with_renderers(settings: RunSettings, cluster: Cluster) -> RunSettings:
    """A run's settings with each channel's renderer said where it names a model and no renderer, and exactly one
    renderer renders that model (`renderers_of`)."""
    filled: dict[str, JsonValue] = {}
    for key, model in settings.values.items():
        if not (key.startswith("channels.") and key.endswith(".model") and isinstance(model, str)):
            continue
        channel = key.removeprefix("channels.").removesuffix(".model")
        if settings.get(f"channels.{channel}.renderer") is not None or not renders_tokens(settings, cluster, channel):
            continue
        found = renderers_of(settings, cluster, channel)
        if len(found) == 1:
            filled[f"channels.{channel}.renderer"] = found[0]
    return RunSettings({**settings.values, **filled}) if filled else settings


def completed(settings: RunSettings, cluster: Cluster) -> RunSettings:
    """A run's settings with what follows from them said: what it trains (`with_weights`) and each channel's renderer
    (`with_renderers`), as its start records them."""
    return with_renderers(with_weights(settings, cluster), cluster)


def serves(provider: InferenceProvider, weights: str) -> str | None:
    """Why `provider` cannot serve a run's checkpoints of `weights` (`lora`: adapters by name; `full`: full weights
    reloaded in place), if it cannot."""
    offered = provider.capabilities
    if weights == "lora" and not offered.adapters:
        return f"provider {provider.name} ({provider.kind}) serves no adapters, and the run trains a LoRA"
    if weights == "full" and not offered.full_reload:
        sampler = ": Tinker's sampler serves only checkpoints Tinker trained" if provider.kind == "tinker" else ""
        return (
            f"provider {provider.name} ({provider.kind}) cannot reload full weights in place, and the run trains full "
            f"weights{sampler}"
        )
    return None


def _weights(run: _Run) -> None:
    said, trainer = run.settings["weights"], run.trainer
    if run.kind in TRAINING and trainer is not None and isinstance(said, str) and said in _WEIGHTS:
        makes = trainer.capabilities.produces
        if makes != said:
            run.refuse("weights", "trainer.provider", f"the {trainer.name} trainer trains {_WEIGHTS[makes]} only, and "
                       f"the run trains {_WEIGHTS[said]}")  # fmt: skip
    weights, serving = run.weights, run.serving()
    for channel in serving if weights is not None else ():
        merged = run.settings[f"channels.{channel}.bridge"] == "merge-quantize"  # (a LoRA served as full weights)
        for name, provider in run.providers(channel):
            reason = serves(provider, "full" if merged else str(weights))
            if reason is None:
                continue
            if merged and weights == "lora":
                reason = f"provider {name} ({provider.kind}) cannot reload full weights in place, and channel "
                reason += f"{channel} serves the run's LoRA merged into them (bridge = merge-quantize)"
            run.refuse("weights", f"channels.{channel}.provider", reason)
    for (channel, name), chain in run.chains.items():
        if weights is not None and channel in serving:
            continue
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


def _renderer(run: _Run) -> None:
    from rollout.names import named
    from rollout_train.recorder.renderers import rendered

    for key, model in run.settings.values.items():
        if not (key.startswith("channels.") and key.endswith(".model") and isinstance(model, str)):
            continue
        channel = key.removeprefix("channels.").removesuffix(".model")
        if not renders_tokens(run.settings, run.cluster, channel):
            continue
        said = run.settings.get(f"channels.{channel}.renderer")
        found = renderers_of(run.settings, run.cluster, channel)
        where = f"channels.{channel}.renderer"
        if said is None:
            if not found:
                run.refuse("renderer", where, f"no renderer here says it renders {model}: name one")
            elif len(found) > 1:
                run.refuse("renderer", where, f"several renderers render {model} ({', '.join(found)}): name one")
            continue
        try:
            factory = named(str(said))
        except Exception:
            continue  # (whether it imports is the settings' and the job's to say)
        provider = run.cluster.inference.get(str(run.settings.get(f"channels.{channel}.provider")))
        offer = provider.models.get(model) if provider is not None else None
        base = offer.base if offer is not None else None
        if rendered(factory, model) is False and (base is None or rendered(factory, base) is False):
            better = f"; {', '.join(found)} does" if found else ""
            run.refuse("renderer", where, f"{said} says it renders other models than {model}{better}")


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
    _uncorrected(run, objective)
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


def _uncorrected(run: _Run, objective: Objective) -> None:
    """A note for a training run whose objective makes no importance correction, while its turns may begin behind the
    newest checkpoint: they are trained on as if on-policy."""
    if run.kind != "train" or not objective.takes_importance or objective.importance.correction != "none":
        return
    lag = run.settings["max_lag"]
    if not isinstance(lag, int) or lag <= 0:
        return
    exact = objective.importance.paper_exact
    key = "objective.importance.paper_exact" if exact else "objective.importance.correction"
    fix = "importance.paper_exact = false" if exact else "importance.correction = truncate"
    run.note("objective", key, f"the objective makes no importance correction, and max_lag is {lag}: turns that began "
             f"up to {lag} checkpoint{'' if lag == 1 else 's'} behind the newest, sampled by an engine that computes "
             f"slightly differently from the trainer, are trained on as if on-policy. {fix} weighs them; max_lag = 0 "
             "keeps them nearer")  # fmt: skip


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
        offered = set(run.cluster.environments)
        for environment in sorted(each for each in suite.environments - offered if not is_published(each)):
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
    facts = run.environment
    if is_published(environment):
        if facts is not None and not facts.loads:
            run.refuse("environment", "environment", facts.why or f"there is no published environment {environment}")
            return
    elif environment not in run.cluster.environments:
        offered = ", ".join(run.cluster.environments) or "none"
        run.refuse("environment", "environment", f"this cluster does not offer {environment} (it offers {offered})")
        return
    if facts is None:
        return
    if not facts.loads:
        run.refuse("environment", "environment", f"{environment} does not load: {facts.why or 'its worker says so'}")
    for kind in sorted(facts.sandboxes - set(run.cluster.sandboxes)):
        run.refuse("environment", "environment", f"{environment} needs sandboxes of kind {kind}, and this cluster has "
                   "no pool of them")  # fmt: skip
    if run.cluster.kubernetes is not None:
        for kind in sorted(facts.sandboxes & set(run.cluster.sandboxes)):
            if run.cluster.sandboxes[kind].url is None:
                run.refuse("environment", "environment", f"{environment} needs sandboxes of kind {kind}, whose pool "
                           "this cluster makes in each run's pod, where Kubernetes accounts nothing of what they hold: "
                           f"serve it from pods of its own ([sandboxes.{kind}] url)")  # fmt: skip
    for name in sorted(facts.tool_sets - set(run.cluster.tools)):
        run.refuse("environment", "environment", f"{environment} imports the tool set {name}, which this cluster does "
                   "not serve ([tools])")  # fmt: skip


def _capacity(run: _Run) -> None:
    asked = demand(run.settings, run.cluster)
    needs = asked.total
    key = "trainer.provider" if run.kind in TRAINING else f"channels.{played_channel(run.settings)}.provider"
    capacity = run.cluster.capacity
    if capacity is not None:
        known = [name for name in ("cpus", "memory_gib", "gpus") if getattr(capacity, name) is not None]
        room = Resources(capacity.cpus or 0.0, capacity.memory_gib or 0.0, capacity.gpus or 0.0)
        pods = requested(asked, kubernetes=run.cluster.kubernetes is not None)
        if over := pods.beyond(room, known=known):
            most = Resources(capacity.cpus or 0.0, capacity.memory_gib or 0.0, capacity.gpus or 0.0).said()
            run.refuse("capacity", key, f"the run needs {pods.said()} ({_parts(asked)}, and room for Ray's own "
                       f"processes, {HEADROOM.said()}); the cluster schedules at most {most} for one run "
                       f"([capacity]): it asks for more {', '.join(over)}, and would never start")  # fmt: skip
    total, free = run.ledger.gpus, run.ledger.free
    gpus = needs.gpus + _elsewhere(run)
    if total is not None and gpus > total:
        run.refuse("capacity", key, f"the run needs {gpus:g} GPUs, and the cluster has {total:g}: it would never "
                   "start")  # fmt: skip
    elif free is not None and (short := _short(needs, free)):
        run.note("capacity", key, f"the run needs {short}: it waits")
    _pods(run, key)


def _pods(run: _Run, key: str) -> None:
    """A run's pods on RunPod (none of them the cluster's capacity): no more of a provider than its `max_pods`, and
    what the pods need of the cluster (the ledger service, which pods reach at `[ledger] public` with tokens signed
    with the platform's, and step-ca for their certificates)."""
    from rollout_train.pods.leasing import needs_of
    from rollout_train.providers import pod_table

    try:
        needs = needs_of(run.settings, run.cluster)
    except ValueError as error:
        run.refuse("capacity", key, str(error))
        return
    counted: dict[str, int] = {}
    for need in needs:
        counted[need.provider] = counted.get(need.provider, 0) + need.count
    for name, count in counted.items():
        provider = run.cluster.inference.get(name) or run.cluster.trainers.get(name)
        if provider is None:
            continue
        table = pod_table(provider.kind, provider.settings)
        where = f"[{'inference' if name in run.cluster.inference else 'trainers'}.{name}]"
        if count > table.max_pods:
            run.refuse("capacity", key, f"the run needs {count} pods of {name}, which has at most {table.max_pods} "
                       f"(max_pods): it would never start")  # fmt: skip
        if not table.step_ca:
            run.refuse("capacity", key, f"{name}'s pods get their certificates from step-ca: say {where} step_ca")
        if table.store is None and run.cluster.blobs.kind == "files":
            run.refuse("capacity", key, f"{name}'s pods cannot read a store of files: name a bucket they reach in "
                       f"{where} store ([stores.NAME])")  # fmt: skip
    ledger = run.cluster.ledger
    if counted and (
        ledger.token is None or not (ledger.public or (ledger.url or "").startswith(("http://", "https://")))
    ):
        run.refuse("capacity", key, "pods reach the ledger service at [ledger] public, with tokens signed with the "
                   "platform's ([ledger] token_env): the cluster config says neither")  # fmt: skip


def _elsewhere(run: _Run) -> float:
    """The GPUs of the run's scheduled providers whose servers are outside its Ray cluster and are the cluster's own
    (`vllm-servers`; RunPod's pods are not), by provider: those a channel's replicas take."""
    from rollout_train.providers import RUNPOD

    taken: dict[str, float] = {}
    for channel in run.settings.channels:
        for name, provider in run.providers(channel):
            if provider.allocation == "scheduled" and provider.kind != "vllm" and provider.kind not in RUNPOD:
                replicas = run.settings[f"channels.{channel}.replicas"]
                count = replicas if isinstance(replicas, int) else provider.replicas
                taken[name] = max(taken.get(name, 0.0), count * provider.gpus)
    return sum(taken.values())


def _parts(asked: Demand) -> str:
    """Each part of a run's demand, in words: `its driver (2 CPUs, 2 GiB), trainer (1 CPU), …`."""
    said = [f"its driver ({asked.driver.said()})"]
    said += [f"{part.name} ({part.asks.said()})" for bundle in asked.bundles for part in bundle.parts]
    return ", ".join(said)


def _short(needs: Resources, free: Resources) -> str:
    """What of `needs` the cluster does not have free, each as `N GPUs, and M are free`; empty where it has it all."""
    units = (("gpus", "GPUs"), ("cpus", "CPUs"), ("memory_gib", "GiB of memory"))
    return "; ".join(
        f"{getattr(needs, name):g} {unit}, and {getattr(free, name):g} are free" for name, unit in units
        if getattr(needs, name) > getattr(free, name) + 1e-9
    )  # fmt: skip


@dataclass(frozen=True)
class Spend:
    """A run's estimated spend on its metered parts, at most. For a training run, one step's (`per` is `step`): every
    token trained times the trainer's cost for the model, and the sampled and prompt tokens times the dearest metered
    provider's costs to sample and read them, uncached. For an eval, the whole eval's (`per` is `eval`): every
    episode of the suite's starts, each turn's thinking and answer budgets sampled and its prompt read at the
    dearest metered provider of the channel it plays. Scheduled parts are capacity the run is placed on, not spent."""

    dollars: float | None
    """None where it cannot be estimated (`why`)."""
    parts: Mapping[str, float] = field(default_factory=dict[str, float])
    """Each metered part's dollars, by provider or trainer."""
    why: str = ""
    """Why it cannot be estimated."""
    per: str = "step"
    """What it is the spend of: one step of a training run (`step`), or a whole eval (`eval`)."""


def _priced(
    metered: list[tuple[str, InferenceProvider]], model: str, sampled: float, prompts: float
) -> tuple[str, float] | Spend | None:
    """The dearest of the metered providers to sample `sampled` tokens and read `prompts` tokens of `model`, uncached
    (none: none of them serves it); a `Spend` that says why not where one is priced by the hour."""
    dearest: tuple[str, float] | None = None
    for name, provider in metered:
        offer = provider.models.get(model)
        if offer is None:
            continue
        if "hour" in offer.cost or provider.capabilities.bills == "hours":
            return Spend(None, why=f"provider {name} is priced by the hour")
        cost = (sampled * offer.cost.get("output", 0.0) + prompts * offer.cost.get("input", 0.0)) / 1e6
        if dearest is None or cost > dearest[1]:
            dearest = (name, cost)
    return dearest


def _eval_spend(
    settings: RunSettings, cluster: Cluster, environment: EnvironmentFacts | None, ledger: LedgerFacts | None
) -> Spend:
    """An eval's estimated spend (`Spend`, `per` eval), or why it cannot be estimated."""
    channel = played_channel(settings)
    metered = [(name, cluster.inference[name]) for name in settings.providers(channel)
               if name in cluster.inference and cluster.inference[name].allocation == "metered"]  # fmt: skip
    if not metered:
        return Spend(0.0, per="eval")
    reference = settings["eval.suite"]
    suite = ledger.suites.get(str(reference).partition("@")[0]) if ledger is not None and reference else None
    if suite is None or not suite.entries:
        return Spend(None, why="the suite's starts are not known here", per="eval")
    if environment is None or environment.turns_per_episode is None:
        return Spend(None, why="the environment does not say how many turns an episode takes", per="eval")
    asked = settings["eval.episodes"]
    thinking, answer = settings[f"channels.{channel}.thinking_tokens"], settings[f"channels.{channel}.answer_tokens"]
    sampled = prompts = 0.0
    for entry in suite.entries:
        episodes = entry.starts * (asked if isinstance(asked, int) else entry.episodes)
        turns = episodes * environment.turns_per_episode * environment.samples_per_turn  # (each a sample)
        think = entry.thinking_tokens if entry.thinking_tokens is not None else thinking
        reply = entry.answer_tokens if entry.answer_tokens is not None else answer
        if not isinstance(think, int) and not isinstance(reply, int):
            return Spend(None, why=f"channel {channel} has no thinking or answer budget", per="eval")
        sampled += turns * ((think if isinstance(think, int) else 0) + (reply if isinstance(reply, int) else 0))
        prompts += turns * (environment.prompt_tokens or 0)
    found = _priced(metered, str(settings.get(f"channels.{channel}.model")), sampled, prompts)
    if isinstance(found, Spend):
        return replace(found, per="eval")
    if found is None:
        return Spend(0.0, per="eval")
    return Spend(found[1], {found[0]: found[1]}, per="eval")


def spend_of(
    settings: RunSettings,
    cluster: Cluster,
    environment: EnvironmentFacts | None,
    ledger: LedgerFacts | None = None,
) -> Spend:
    """A run's estimated spend (`Spend`), or why it cannot be estimated: one step of a training run, or a whole eval
    (from the suite's starts in `ledger`); not for other runs; for a training run, not without a trainer; not where the
    environment's numbers or the budgets are unknown, or a metered part is priced by the hour. Episodes a group are
    the run's `group_size`, else the environment's, else the objective's group size; each turn of an episode is as many
    samples as the environment says (`samples_per_turn`: every agent of a team samples each turn)."""
    if settings.kind == "eval":
        return _eval_spend(settings, cluster, environment, ledger)
    trained = settings.trained
    trainer = cluster.trainers.get(str(settings["trainer.provider"]))
    if trained is None:
        return Spend(None, why="only a training run's or an eval's spend is estimated")
    if trainer is None:
        return Spend(None, why="it has no trainer")
    model = str(settings.get(f"channels.{trained}.model"))
    metered = [(name, cluster.inference[name]) for name in settings.providers(trained)
               if name in cluster.inference and cluster.inference[name].allocation == "metered"]  # fmt: skip
    pods = _pod_hours(settings, cluster, ledger)
    if isinstance(pods, Spend):
        return pods
    if trainer.allocation != "metered" and not metered:
        return Spend(sum(pods.values()), pods)
    if environment is None:
        return Spend(None, why="the environment's numbers are not known here")
    size = settings["group_size"]
    episodes = size if isinstance(size, int) else environment.episodes_per_group
    if episodes is None:
        try:
            episodes = algorithm_for(objective_in(settings)).group_size
        except ValueError:
            return Spend(None, why="its objective is not one")
    if environment.turns_per_episode is None:
        return Spend(None, why="the environment does not say how many turns an episode takes")
    thinking, answer = settings[f"channels.{trained}.thinking_tokens"], settings[f"channels.{trained}.answer_tokens"]
    if not isinstance(thinking, int) and not isinstance(answer, int):
        return Spend(None, why=f"channel {trained} has no thinking or answer budget")
    groups = settings["groups_per_step"]
    episodes_per_step = (groups if isinstance(groups, int) else 4) * episodes
    samples = episodes_per_step * environment.turns_per_episode * environment.samples_per_turn
    sampled = samples * ((thinking if isinstance(thinking, int) else 0) + (answer if isinstance(answer, int) else 0))
    prompts = samples * (environment.prompt_tokens or 0)
    parts: dict[str, float] = {}
    if trainer.allocation == "metered":
        priced = trainer.cost_of(str(settings.get("trainer.model") or model))
        if "hour" in priced:
            return Spend(None, why=f"the {trainer.name} trainer is priced by the hour")
        parts[trainer.name] = (sampled + prompts) * priced.get("train", 0.0) / 1e6
    dearest = _priced(metered, model, sampled, prompts)
    if isinstance(dearest, Spend):
        return dearest
    if dearest is not None:
        parts[dearest[0]] = dearest[1]
    parts |= pods
    return Spend(sum(parts.values()), parts)


def _hourly(run: _Run) -> list[str]:
    """The RunPod providers a run's pods are of."""
    from rollout_train.pods.leasing import needs_of

    try:
        return sorted({need.provider for need in needs_of(run.settings, run.cluster)})
    except ValueError:
        return []


def _pod_hours(settings: RunSettings, cluster: Cluster, ledger: LedgerFacts | None) -> dict[str, float] | Spend:
    """What one step of a training run costs on its pods on RunPod, by provider: each pod's hourly price (its table's
    `price`) for as long as a step of its trainer and model took here lately (`LedgerFacts.step_seconds`), a host's
    pod once for its trainer and its channel. A `Spend` that says why not where it cannot be reckoned."""
    from rollout_train.pods.leasing import needs_of
    from rollout_train.providers import pod_table

    try:
        needs = needs_of(settings, cluster)
    except ValueError:
        return {}
    parts: dict[str, float] = {}
    for need in needs:
        provider = cluster.inference.get(need.provider) or cluster.trainers.get(need.provider)
        if provider is None:
            continue
        table = pod_table(provider.kind, provider.settings)
        if table.price is None:
            return Spend(None, why=f"{need.provider}'s pods are paid by the hour, and its table says no price")
        seconds = ledger.step_seconds if ledger is not None else None
        if seconds is None:
            return Spend(None, why=f"{need.provider}'s pods are paid by the hour (${table.price:g} each), and how long "
                         "a step takes here is not known yet")  # fmt: skip
        parts[need.provider] = parts.get(need.provider, 0.0) + need.count * table.price * seconds / 3600
    return parts


def estimated_spend(
    settings: RunSettings,
    cluster: Cluster,
    environment: EnvironmentFacts | None,
    ledger: LedgerFacts | None = None,
) -> float | None:
    """Dollars one step (of an eval: the eval) is estimated to cost on the run's metered parts, at most
    (`spend_of`); none where it cannot be estimated."""
    return spend_of(settings, cluster, environment, ledger).dollars


def _spend(run: _Run) -> None:
    if run.kind == "eval":
        _eval_limit(run)
        return
    if run.kind != "train":
        return
    limit = run.settings["limits.spend"]
    if not isinstance(limit, int | float) or isinstance(limit, bool):
        metered = sorted({name for channel in run.settings.channels for name, provider in run.providers(channel)
                          if provider.allocation == "metered"}
                         | ({run.trainer.name} if run.trainer is not None and run.trainer.allocation == "metered"
                            else set()))  # fmt: skip
        if metered:
            run.note("spend", "limits.spend", f"{', '.join(metered)} {'is' if len(metered) == 1 else 'are'} metered, "
                     "and no limits.spend bounds what the run spends")  # fmt: skip
        if hourly := _hourly(run):
            run.note("spend", "limits.spend", f"{', '.join(hourly)}'s pods are paid by the hour, and no limits.spend "
                     "bounds what the run spends (limits.hours bounds how long)")  # fmt: skip
        return
    spend = spend_of(run.settings, run.cluster, run.environment, run.ledger)
    if spend.dollars is None:
        run.note("spend", "limits.spend", f"one step's spend cannot be estimated yet ({spend.why}): the run still ends "
                 f"once its spend reaches ${limit:g}")  # fmt: skip
    elif spend.dollars > limit:
        run.refuse("spend", "limits.spend", f"one step is estimated at up to ${spend.dollars:.2f}, above limits.spend "
                   f"${limit:g}: the run would stop before its first step")  # fmt: skip


def _eval_limit(run: _Run) -> None:
    """An eval on metered providers: noted where no limit bounds it, or where it is estimated above its limit (it ends
    once it spends the limit, before every start is played)."""
    channel = played_channel(run.settings)
    metered = sorted(name for name, provider in run.providers(channel) if provider.allocation == "metered")
    if not metered:
        return
    limit = run.settings["limits.spend"]
    if not isinstance(limit, int | float) or isinstance(limit, bool):
        run.note("spend", "limits.spend", f"{', '.join(metered)} {'is' if len(metered) == 1 else 'are'} metered, and "
                 "no limits.spend bounds what the eval spends")  # fmt: skip
        return
    spend = spend_of(run.settings, run.cluster, run.environment, run.ledger)
    if spend.dollars is None:
        run.note("spend", "limits.spend", f"the eval's spend cannot be estimated yet ({spend.why}): it still ends "
                 f"once it spends ${limit:g}")  # fmt: skip
    elif spend.dollars > limit:
        run.note("spend", "limits.spend", f"the eval is estimated at up to ${spend.dollars:.2f}, above limits.spend "
                 f"${limit:g}: it ends once it spends ${limit:g}, before every start is played")  # fmt: skip


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
    "renderer": _renderer,
    "rank": _rank,
    "segment": _segment,
    "start": _start,
    "objective": _objective,
    "evals": _evals,
    "distillation": _distillation,
    "environment": _environment,
    "capacity": _capacity,
    "spend": _spend,
    "name": _name,
}
assert tuple(_RULES) == tuple(each.name for each in RULES)
