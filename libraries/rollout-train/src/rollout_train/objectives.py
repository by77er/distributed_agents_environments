"""Objectives, declared: a family, the primary selector, and components, orthogonal settings that compose its loss;
presets, the literature's objectives as a family and component values; and resolving a preset and overrides into a
full objective. Nothing here computes a loss (`rollout_objectives` does, in torch).

A **family** fixes what a batch item is and the core term of the loss (`FAMILIES`):

- `policy_gradient`: a segment with an advantage (`rollout_train.trainer.Weighted`), the advantage times the policy's
  logprob of each sampled token, under the ratio, clipping and importance components;
- `preference`: a pair, chosen and rejected over a shared context (`rollout_train.trainer.Pair`), or an example
  labelled desirable or undesirable (`rollout_train.trainer.Labelled`), a function of each side's log-likelihood
  ratio to the reference (or of its likelihood alone, for a loss without one);
- `likelihood`: a segment with a weight, the weighted log-likelihood of its sampled tokens (supervised fine-tuning,
  imitation);
- `distillation`: a segment with a teacher's logprobs of its sampled tokens, and optionally the teacher's top-k tokens
  and logprobs at each (`rollout_train.trainer.Distilled`), a divergence between the teacher's next-token distribution
  and the policy's (`Distillation`).

A **component** (`COMPONENTS`) has a dotted key under `objective.` in a run's settings (`objective.clip.low`), the
values it takes, the families that accept it, and whether it may change between steps (the numbers may; what decides
the shape of the loss may not). `Objective` holds every component; `Objective.components` the family's.

A **preset** (`PRESETS`) is a family and component values, with the paper it comes from. `resolved(preset,
overrides)` is the objective a run asks for: the preset with each override in its place, refused (`ValueError`) for a
component the family does not accept or a combination that means nothing (`problems`). A run's start records the
resolved objective, so what a run trains with never depends on what a preset means later.

Where a component that follows from another is not given, it follows: a KL to the reference reads the base model
(`reference = base`), a preference loss with a reference reads it and one without does not, and an odds ratio is of
length-normalized likelihoods.

A policy gradient can add a distillation term with a coefficient (`distillation.coefficient`), as a preference loss can
add a likelihood term: its batch items are then distilled segments, each with its episode's advantage.

A trainer's own settings can name an objective too (`from_trainer_settings`): `objective = "policy_gradient"` is the
`default` preset and `"likelihood"` is `sft`, and `ratio`, `clip_low`, `clip_high`, `segment_clip_low`,
`segment_clip_high` and `truncate` are its components.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, cast

from pydantic import JsonValue

__all__ = [
    "COMPONENTS",
    "DEFAULT",
    "FAMILIES",
    "LEGACY",
    "PRESETS",
    "Advantage",
    "Clip",
    "Component",
    "Distillation",
    "Entropy",
    "Importance",
    "Kl",
    "Likelihood",
    "Objective",
    "Preference",
    "Preset",
    "component",
    "composed",
    "from_trainer_settings",
    "objective_of",
    "problems",
    "resolved",
]

POLICY_GRADIENT, PREFERENCE, LIKELIHOOD = "policy_gradient", "preference", "likelihood"
DISTILLATION = "distillation"
FAMILIES = (POLICY_GRADIENT, PREFERENCE, LIKELIHOOD, DISTILLATION)
"""Every family, the primary selector of an objective."""


@dataclass(frozen=True)
class Advantage:
    """How a policy-gradient segment's advantage is made from its group's scores (`rollout_train.algorithm`)."""

    baseline: str = "group_mean"
    """`group_mean`: each score less the group's mean. `leave_one_out`: less the mean of the others' (RLOO). `none`:
    the score itself."""
    scale: str = "none"
    """`none`, or `group_std`: divided by the standard deviation of the group's scores (GRPO)."""
    filter: str = "equal_scores"
    """`equal_scores`: a group whose scores are all equal is skipped (DAPO's dynamic sampling). `none`: kept."""
    tiebreak: float = 0.0
    """What the shortest episodes (`Episode.duration`) of a group whose every episode saturated its task score more,
    before the baseline. 0, every preset's: nothing, so how long an episode took never changes its score."""


@dataclass(frozen=True)
class Clip:
    """How the update's movement is bounded, through the ratio of each token's logprob now to the step's start."""

    kind: str = "ratio"
    """`none`; `ratio`: PPO's, the smaller of the ratio and the clipped ratio, each times the advantage; `weight`: the
    clipped ratio as a weight with no gradient, times the logprob (CISPO); `dual`: PPO's, and for a negative advantage
    no less than `dual` times it (dual-clip PPO)."""
    low: float = 0.2
    high: float = 0.28
    """The ratio is clipped to 1 - `low` .. 1 + `high` (asymmetric: DAPO's clip-higher)."""
    dual: float = 3.0
    """For `dual`: the bound for a negative advantage, in times it (above 1)."""


@dataclass(frozen=True)
class Importance:
    """The correction for where each token was sampled: the weight of its logprob at the step's start against the one
    the engine recorded (`old / behavior`), a constant with no gradient."""

    correction: str = "truncate"
    """`none`; `untruncated`: the weight (importance sampling); `truncate`: the weight, at most `cap` (truncated
    importance sampling); `mask`: the weight, and a token
    whose weight is outside `floor` .. `cap` is dropped (masked importance sampling)."""
    level: str = "token"
    """`token`: a weight for each token. `segment`: one for the segment, the geometric mean of its tokens'."""
    cap: float = 2.0
    floor: float = 0.0
    """For `mask`: the lowest weight kept."""


@dataclass(frozen=True)
class Kl:
    """A penalty for the policy's divergence from a target, estimated on the sampled tokens."""

    target: str = "none"
    """`none`; `reference`: the reference model (`Objective.reference`); `old`: the policy at the step's start."""
    estimator: str = "k3"
    """With `log_r` the target's logprob less the policy's: `k1` is `-log_r`, `k2` is `log_r² / 2`, `k3` is
    `exp(log_r) - 1 - log_r` (Schulman's estimators)."""
    placement: str = "loss"
    """`loss`: added to each token's loss, with its gradient. `reward`: taken from each token's advantage, with none."""
    coefficient: float = 0.0


@dataclass(frozen=True)
class Entropy:
    coefficient: float = 0.0
    """Each sampled position's entropy, times this, is taken from its loss (a bonus for keeping the policy spread)."""


@dataclass(frozen=True)
class Preference:
    """The preference loss, of each side's log-likelihood ratio to the reference (`rho`: the sum over its sampled
    tokens of the policy's logprob less the reference's, or their mean with `length_normalized`)."""

    loss: str = "sigmoid"
    """For a pair, of `h = rho_chosen - rho_rejected` (`sigmoid(x)` the logistic function): `sigmoid`, `-log
    sigmoid(beta·h)` (DPO); `hinge`, `max(0, 1 - beta·h)`; `square`, `(h - 1/(2·beta))²` (IPO, beta as its tau);
    `margin`, `-log sigmoid(beta·h - margin)` of the likelihoods alone (SimPO); `odds_ratio`, `-beta·log sigmoid(log
    odds_chosen - log odds_rejected)` of the length-normalized likelihoods, beta weighing the term (ORPO). For a
    labelled example: `kto`, `desirable·(1 - sigmoid(beta·(rho - z)))` or `undesirable·(1 - sigmoid(beta·(z - rho)))`,
    `z` the mean of the minibatch's `rho` (no less than 0, no gradient)."""
    beta: float = 0.1
    margin: float = 0.0
    length_normalized: bool = False
    desirable: float = 1.0
    undesirable: float = 1.0
    """KTO's weights of desirable and undesirable examples."""


@dataclass(frozen=True)
class Likelihood:
    coefficient: float = 0.0
    """A likelihood term beside a preference loss: the chosen side's mean negative logprob, times this (ORPO)."""


@dataclass(frozen=True)
class Distillation:
    """A divergence between a teacher's next-token distribution and the policy's at each sampled token
    (`rollout_objectives.distillation`)."""

    divergence: str = "reverse_kl"
    """`reverse_kl`, KL(policy || teacher); `forward_kl`, KL(teacher || policy); `jsd`, GKD's generalized
    Jensen-Shannon divergence, `beta·KL(teacher || m) + (1 - beta)·KL(policy || m)` with `m = beta·teacher + (1 -
    beta)·policy`."""
    form: str = "policy_gradient"
    """`policy_gradient`: each sampled token's advantage is the teacher's logprob less the policy's at the step's start
    (no gradient), clipped to `advantage_clip`, and its loss the advantage times the policy's logprob (the reverse KL's
    gradient, estimated from the sampled tokens alone). `top_k`: the divergence over the teacher's top-k tokens at each
    sampled position: for `reverse_kl`, of the probabilities as they are there (MOPD's top-k form); for `forward_kl`
    and `jsd`, of both distributions renormalized over those tokens, at `temperature` (the rest of the vocabulary is
    dropped)."""
    top_k: int = 0
    """How many of the teacher's most likely tokens each sampled position carries (0: its logprob of the sampled token
    alone)."""
    temperature: float = 1.0
    """For a renormalized top-k divergence: both sides' logprobs are divided by it before renormalizing, and the
    divergence is multiplied by its square (Hinton et al., 2015)."""
    advantage_clip: float = 0.0
    """For the `policy_gradient` form: the advantage is clipped to -this .. this (0: not clipped)."""
    beta: float = 0.5
    """For `jsd`: the teacher's weight in the mixture, between 0 and 1."""
    teachers: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)
    """The teacher channel that scores each episode, by route: an environment (`module:name`), one of its rows
    (`module:name/ROW`), or `*` for any other (`rollout_train.distillation.teacher_for`). One teacher scores each
    segment; teachers are never combined."""
    coefficient: float = 0.0
    """For a policy gradient: a distillation term beside it, times this (0: none)."""


@dataclass(frozen=True)
class Objective:
    """An objective, every component of it (those its family does not accept keep their defaults and mean nothing)."""

    family: str = POLICY_GRADIENT
    advantage: Advantage = field(default_factory=Advantage)
    ratio: str = "token"
    """`token`: each token's logprob now against the step's start. `segment`: one for the segment, the geometric mean
    of its tokens' (GSPO), its gradient spread over them. `none`: no ratio, the logprob itself (REINFORCE)."""
    clip: Clip = field(default_factory=Clip)
    importance: Importance = field(default_factory=Importance)
    kl: Kl = field(default_factory=Kl)
    entropy: Entropy = field(default_factory=Entropy)
    aggregate: str = "token_mean"
    """How a minibatch's per-token losses become one: `token_mean`, over its sampled tokens; `segment_mean`, over each
    segment's tokens, then its segments; `segment_sum`, summed over each segment's tokens, then a mean over its segments
    (a sequence's logprob, as REINFORCE and RLOO take it); `constant`, summed over each segment's tokens and divided by
    `constant_tokens`, then a mean over its segments (Dr. GRPO). A preference loss is a mean over pairs or examples."""
    constant_tokens: int = 1024
    """For `constant`: the fixed token count (the turn's token budget)."""
    reference: str = "none"
    """The model the KL to the reference and a preference loss compare with: `base`, the model trained over (an
    adapter switched off; a frozen copy for a full-weight trainer); `none`."""
    preference: Preference = field(default_factory=Preference)
    likelihood: Likelihood = field(default_factory=Likelihood)
    distillation: Distillation = field(default_factory=Distillation)
    preset: str = "default"
    """The preset it was resolved from (what it says, not what it is: the components are)."""

    def components(self) -> dict[str, JsonValue]:
        """The components its family accepts, by dotted key, with their values."""
        return {each.key: self.get(each.key) for each in COMPONENTS if self.family in each.families}

    def get(self, key: str) -> JsonValue:
        """A component's value, by dotted key (`clip.low`)."""
        value: Any = self
        for part in key.split("."):
            value = getattr(value, part)
        return cast(JsonValue, dict(cast(Mapping[str, str], value)) if isinstance(value, Mapping) else value)

    def to_json(self) -> dict[str, JsonValue]:
        """What a run's start records: the preset, the family and the family's components."""
        return {"preset": self.preset, "family": self.family, **self.components()}

    @classmethod
    def from_json(cls, said: Mapping[str, JsonValue]) -> "Objective":
        """An objective as `to_json` recorded it."""
        family = str(said.get("family", POLICY_GRADIENT))
        given = {key: value for key, value in said.items() if key not in ("preset", "family")}
        return _with(replace(cls(), family=family, preset=str(said.get("preset", "default"))), given)

    def changed(self, changes: Mapping[str, JsonValue]) -> "Objective":
        """With `changes` (dotted keys, each a component that may change between steps), checked as any objective is
        (else `ValueError`)."""
        for key in changes:
            found = component(key)
            if found is None or not found.changeable:
                raise ValueError(f"objective.{key} cannot change between steps")
        made = _with(self, changes)
        if made.distills != self.distills:  # (what a batch item is would change)
            raise ValueError("objective.distillation.coefficient can change between steps, but not to or from 0")
        if said := problems(made, changes):
            raise ValueError("; ".join(reason for _, reason in said))
        return made

    @property
    def needs_reference(self) -> bool:
        """Whether its loss reads the reference's logprobs."""
        return self.reference != "none"

    @property
    def needs_behaviour(self) -> bool:
        """Whether its loss reads the logprobs the engine recorded (an importance correction)."""
        return self.family in (POLICY_GRADIENT, DISTILLATION) and self.importance.correction != "none"

    @property
    def distills(self) -> bool:
        """Whether its loss reads a teacher's logprobs: a distillation, or a policy gradient with a distillation term
        (its batch items are `rollout_train.trainer.Distilled`)."""
        return self.family == DISTILLATION or (self.family == POLICY_GRADIENT and self.distillation.coefficient != 0)

    @property
    def needs_top(self) -> int:
        """How many of the teacher's most likely tokens each sampled position must carry (0: none)."""
        return self.distillation.top_k if self.distills else 0

    @property
    def needs_distribution(self) -> bool:
        """Whether its loss reads the policy's logprobs of tokens other than the sampled ones: those of the teacher's
        top-k (the `top_k` form of distillation)."""
        return self.distills and self.distillation.form == "top_k"

    @property
    def needs_entropy(self) -> bool:
        return self.family == POLICY_GRADIENT and self.entropy.coefficient != 0.0

    @property
    def labelled(self) -> bool:
        """Whether its batch items are labelled examples (KTO), rather than pairs."""
        return self.family == PREFERENCE and self.preference.loss == "kto"


@dataclass(frozen=True)
class Component:
    """One component: its dotted key under `objective.`, the JSON types and strings it takes, the families that accept
    it, and whether a running run takes it from its next step on."""

    key: str
    types: tuple[str, ...]
    families: frozenset[str]
    changeable: bool
    says: str
    choices: tuple[str, ...] = ()
    least: float | None = None
    above: bool = False
    """`least` itself is not taken."""


_PG = frozenset({POLICY_GRADIENT})
_ADVANTAGED = frozenset({POLICY_GRADIENT, LIKELIHOOD})
_PREFERENCE = frozenset({PREFERENCE})
_DISTILLED = frozenset({POLICY_GRADIENT, DISTILLATION})
_AGGREGATED = frozenset({POLICY_GRADIENT, LIKELIHOOD, DISTILLATION})
_S, _F, _I, _B, _T = ("str",), ("float",), ("int",), ("bool",), ("table",)
COMPONENTS: tuple[Component, ...] = (
    Component("advantage.baseline", _S, _ADVANTAGED, False, "What a score is measured against",
              ("group_mean", "leave_one_out", "none")),
    Component("advantage.scale", _S, _ADVANTAGED, False, "What an advantage is divided by", ("none", "group_std")),
    Component("advantage.filter", _S, _ADVANTAGED, False, "Which groups are skipped", ("none", "equal_scores")),
    Component("advantage.tiebreak", _F, _ADVANTAGED, False, "What the shortest of a saturated group scores more; 0: "
              "nothing", least=0),
    Component("ratio", _S, _PG, False, "The ratio to the step's start", ("token", "segment", "none")),
    Component("clip.kind", _S, _PG, False, "How the ratio is clipped", ("none", "ratio", "weight", "dual")),
    Component("clip.low", _F, _PG, True, "The ratio's lower bound, below 1", least=0),
    Component("clip.high", _F, _PG, True, "The ratio's upper bound, above 1", least=0),
    Component("clip.dual", _F, _PG, True, "Dual clipping's bound, in times a negative advantage", least=1, above=True),
    Component("importance.correction", _S, _DISTILLED, False, "The correction for where tokens were sampled",
              ("none", "untruncated", "truncate", "mask")),
    Component("importance.level", _S, _DISTILLED, False, "A weight per token or per segment", ("token", "segment")),
    Component("importance.cap", _F, _DISTILLED, True, "The largest weight", least=0, above=True),
    Component("importance.floor", _F, _DISTILLED, True, "The smallest weight a mask keeps", least=0),
    Component("kl.target", _S, _DISTILLED, False, "What the KL penalty measures against", ("none", "reference", "old")),
    Component("kl.estimator", _S, _DISTILLED, False, "How the KL is estimated", ("k1", "k2", "k3")),
    Component("kl.placement", _S, _DISTILLED, False, "Where the KL penalty goes", ("loss", "reward")),
    Component("kl.coefficient", _F, _DISTILLED, True, "The KL penalty's weight", least=0),
    Component("entropy.coefficient", _F, _PG, True, "The entropy bonus's weight"),
    Component("aggregate", _S, _AGGREGATED, False, "How per-token losses become one",
              ("token_mean", "segment_mean", "segment_sum", "constant")),
    Component("constant_tokens", _I, _AGGREGATED, False, "The token count `constant` divides by", least=1),
    Component("reference", _S, frozenset({POLICY_GRADIENT, PREFERENCE, DISTILLATION}), False, "The reference model",
              ("none", "base")),
    Component("preference.loss", _S, _PREFERENCE, False, "The preference loss",
              ("sigmoid", "hinge", "square", "margin", "odds_ratio", "kto")),
    Component("preference.beta", _F, _PREFERENCE, True, "The preference loss's scale", least=0, above=True),
    Component("preference.margin", _F, _PREFERENCE, True, "SimPO's target margin"),
    Component("preference.length_normalized", _B, _PREFERENCE, False, "Each side's mean logprob, not its sum"),
    Component("preference.desirable", _F, _PREFERENCE, True, "KTO's weight of desirable examples", least=0),
    Component("preference.undesirable", _F, _PREFERENCE, True, "KTO's weight of undesirable examples", least=0),
    Component("likelihood.coefficient", _F, _PREFERENCE, True, "A likelihood term beside the preference loss", least=0),
    Component("distillation.divergence", _S, _DISTILLED, False, "The divergence from the teacher",
              ("reverse_kl", "forward_kl", "jsd")),
    Component("distillation.form", _S, _DISTILLED, False, "From the sampled tokens, or over the teacher's top-k",
              ("policy_gradient", "top_k")),
    Component("distillation.top_k", _I, _DISTILLED, False, "The teacher's most likely tokens at each position",
              least=0),
    Component("distillation.temperature", _F, _DISTILLED, True, "A renormalized top-k divergence's temperature",
              least=0, above=True),
    Component("distillation.advantage_clip", _F, _DISTILLED, True, "The advantage's bound either side; 0: none",
              least=0),
    Component("distillation.beta", _F, _DISTILLED, True, "The teacher's weight in the JSD's mixture", least=0,
              above=True),
    Component("distillation.teachers", _T, _DISTILLED, False, "The teacher channel of each route"),
    Component("distillation.coefficient", _F, _PG, True, "A distillation term beside the policy gradient", least=0),
)  # fmt: skip
"""Every component, by dotted key under `objective.`."""
_BY_KEY = {each.key: each for each in COMPONENTS}


def component(key: str) -> Component | None:
    """The component a dotted key names (`clip.low`, without `objective.`)."""
    return _BY_KEY.get(key)


@dataclass(frozen=True)
class Preset:
    name: str
    objective: Objective
    source: str
    """The paper it comes from."""
    says: str


def _preset(name: str, source: str, says: str, family: str = POLICY_GRADIENT, **values: Any) -> Preset:
    made = Objective(family=family, preset=name)
    for key, value in values.items():
        made = replace(made, **{key: value})
    return Preset(name, made, source, says)


_NO_CLIP = Clip(kind="none")
_NO_IMPORTANCE = Importance(correction="none")
_GROUP_NORMALIZED = Advantage(scale="group_std", filter="none")
PRESETS: Mapping[str, Preset] = {
    each.name: each
    for each in (
        _preset(
            "default",
            "this platform: Liu et al., 2025 (Dr. GRPO) for the advantage, Yu et al., 2025 (DAPO) for clip-higher and "
            "the token mean, Yao et al., 2025 for truncated importance sampling",
            "Dr. GRPO's advantages, DAPO's clip-higher and token mean, truncated importance sampling at 2",
        ),
        _preset(
            "reinforce",
            "Williams, 1992",
            "the score times each segment's logprob: no baseline, ratio, clipping or correction",
            advantage=Advantage(baseline="none", filter="none"),
            ratio="none",
            clip=_NO_CLIP,
            importance=_NO_IMPORTANCE,
            aggregate="segment_sum",
        ),
        _preset(
            "rloo",
            "Ahmadian et al., 2024 (Back to Basics)",
            "REINFORCE with a leave-one-out baseline",
            advantage=Advantage(baseline="leave_one_out", filter="none"),
            ratio="none",
            clip=_NO_CLIP,
            importance=_NO_IMPORTANCE,
            aggregate="segment_sum",
        ),
        _preset(
            "ppo_clip",
            "Schulman et al., 2017",
            "the token ratio clipped at 0.2 either side; advantages normalized within the group (no critic)",
            advantage=_GROUP_NORMALIZED,
            clip=Clip(low=0.2, high=0.2),
            importance=_NO_IMPORTANCE,
        ),
        _preset(
            "grpo",
            "Shao et al., 2024 (DeepSeekMath)",
            "group mean and standard deviation; the token ratio clipped at 0.2; KL to the reference by k3 in the "
            "loss, at 0.04; a mean over each segment's tokens, then segments",
            advantage=_GROUP_NORMALIZED,
            clip=Clip(low=0.2, high=0.2),
            importance=_NO_IMPORTANCE,
            kl=Kl(target="reference", estimator="k3", placement="loss", coefficient=0.04),
            aggregate="segment_mean",
            reference="base",
        ),
        _preset(
            "dr_grpo",
            "Liu et al., 2025 (Understanding R1-Zero-Like Training)",
            "the group mean without the standard deviation; summed over tokens and divided by a constant; no KL",
            advantage=Advantage(filter="none"),
            clip=Clip(low=0.2, high=0.2),
            importance=_NO_IMPORTANCE,
            aggregate="constant",
            constant_tokens=3000,
        ),
        _preset(
            "dapo",
            "Yu et al., 2025 (DAPO)",
            "clip-higher (0.2, 0.28); the token mean; groups of equal scores skipped; no KL",
            advantage=Advantage(scale="group_std", filter="equal_scores"),
            clip=Clip(low=0.2, high=0.28),
            importance=_NO_IMPORTANCE,
        ),
        _preset(
            "gspo",
            "Zheng et al., 2025 (GSPO)",
            "the segment ratio, the geometric mean of its tokens', clipped to (3e-4, 4e-4); a mean over segments",
            advantage=_GROUP_NORMALIZED,
            ratio="segment",
            clip=Clip(low=3e-4, high=4e-4),
            importance=Importance(correction="none", level="segment"),
            aggregate="segment_mean",
        ),
        _preset(
            "cispo",
            "MiniMax, 2025 (MiniMax-M1)",
            "the importance weight clipped above, with its gradient stopped, times the logprob: no update clipping",
            advantage=_GROUP_NORMALIZED,
            clip=Clip(kind="weight", low=1.0, high=3.0),
            importance=_NO_IMPORTANCE,
        ),
        _preset(
            "sft",
            "supervised fine-tuning",
            "the sampled tokens' log-likelihood, each segment weighted by its advantage (1 for a dataset's)",
            family=LIKELIHOOD,
        ),
        _preset(
            "dpo",
            "Rafailov et al., 2023",
            "the sigmoid loss over pairs, against the reference, at beta 0.1",
            family=PREFERENCE,
            reference="base",
            preference=Preference(loss="sigmoid", beta=0.1),
        ),
        _preset(
            "ipo",
            "Azar et al., 2023",
            "the square loss over pairs, against the reference",
            family=PREFERENCE,
            reference="base",
            preference=Preference(loss="square", beta=0.1, length_normalized=True),
        ),
        _preset(
            "simpo",
            "Meng et al., 2024",
            "the length-normalized margin loss, with no reference",
            family=PREFERENCE,
            preference=Preference(loss="margin", beta=2.0, margin=1.0, length_normalized=True),
        ),
        _preset(
            "kto",
            "Ethayarajh et al., 2024",
            "desirable and undesirable examples, unpaired, against the reference",
            family=PREFERENCE,
            reference="base",
            preference=Preference(loss="kto", beta=0.1),
        ),
        _preset(
            "orpo",
            "Hong et al., 2024",
            "an odds-ratio term at 0.1 beside the chosen side's likelihood, with no reference",
            family=PREFERENCE,
            preference=Preference(loss="odds_ratio", beta=0.1, length_normalized=True),
            likelihood=Likelihood(coefficient=1.0),
        ),
        _preset(
            "on_policy_distillation",
            "Agarwal et al., 2024 (GKD); Thinking Machines, 2025 (On-Policy Distillation)",
            "the reverse KL on the student's own samples, from the teacher's logprob of each sampled token: its "
            "advantage the teacher's logprob less the student's",
            family=DISTILLATION,
            importance=_NO_IMPORTANCE,
            distillation=Distillation(divergence="reverse_kl", form="policy_gradient"),
        ),
        _preset(
            "distillation",
            "Hinton et al., 2015; Kim and Rush, 2016",
            "the forward KL to the teacher's top-20 logprobs, renormalized over them, on the teacher's samples",
            family=DISTILLATION,
            importance=_NO_IMPORTANCE,
            distillation=Distillation(divergence="forward_kl", form="top_k", top_k=20),
        ),
        _preset(
            "mopd",
            "Ma et al., 2026, MOPD: Multi-Teacher On-Policy Distillation (MiMo)",
            "each sampled token's advantage the teacher's logprob less the student's, clipped at 5; a mean over each "
            "segment's tokens; one teacher for each domain",
            family=DISTILLATION,
            importance=_NO_IMPORTANCE,
            distillation=Distillation(divergence="reverse_kl", form="policy_gradient", advantage_clip=5.0),
            aggregate="segment_mean",
        ),
        _preset(
            "mopd_top_k",
            "Ma et al., 2026, MOPD: Multi-Teacher On-Policy Distillation (MiMo)",
            "MOPD's top-k form: the reverse KL over the teacher's top-64 tokens; a mean over each segment's tokens",
            family=DISTILLATION,
            importance=_NO_IMPORTANCE,
            distillation=Distillation(divergence="reverse_kl", form="top_k", top_k=64),
            aggregate="segment_mean",
        ),
    )
}
"""Every preset, by name."""
DEFAULT = PRESETS["default"].objective


def _with(objective: Objective, overrides: Mapping[str, JsonValue]) -> Objective:
    """`objective` with each override (dotted keys) in its place, checked against its component's type and values (else
    `ValueError`)."""
    made = objective
    for key, value in overrides.items():
        found = component(key)
        if found is None:
            raise ValueError(f"objective.{key} is no component of an objective")
        if (problem := _problem(found, value)) is not None:
            raise ValueError(f"objective.{key} {problem}")
        typed: Any = float(value) if "float" in found.types and isinstance(value, int | float) else value
        if isinstance(value, dict):
            typed = dict(value)
        made = _replaced(made, key.split("."), typed)
    return made


def _replaced(at: Any, parts: Sequence[str], value: Any) -> Any:
    if len(parts) == 1:
        return replace(at, **{parts[0]: value})
    return replace(at, **{parts[0]: _replaced(getattr(at, parts[0]), parts[1:], value)})


def _problem(found: Component, value: JsonValue) -> str | None:
    if "table" in found.types:  # (a table of text by text: the only table a component takes)
        entries = cast(dict[Any, Any], value).items() if isinstance(value, dict) else None
        if entries is None or not all(isinstance(key, str) and isinstance(each, str) for key, each in entries):
            return f"is a table of text by text, not {json.dumps(value)}"
        return None
    if isinstance(value, bool):
        ok = "bool" in found.types
    elif isinstance(value, int):
        ok = "int" in found.types or "float" in found.types
    elif isinstance(value, float):
        ok = "float" in found.types or ("int" in found.types and value.is_integer())
    else:
        ok = isinstance(value, str) and "str" in found.types
    if not ok:
        spoken = {"str": "text", "float": "a number", "int": "a whole number", "bool": "true or false"}
        return f"is {' or '.join(spoken[each] for each in found.types)}, not {json.dumps(value)}"
    if isinstance(value, str) and found.choices and value not in found.choices:
        return f"is one of {', '.join(found.choices)}, not {value!r}"
    number = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
    if (
        number is not None
        and found.least is not None
        and (number < found.least or (found.above and number == found.least))
    ):
        return f"is {'above' if found.above else 'at least'} {found.least:g}, not {value:g}"
    return None


def problems(objective: Objective, overrides: Mapping[str, JsonValue] | None = None) -> list[tuple[str, str]]:
    """What is wrong with an objective, as (dotted key, reason): an override of a component its family does not accept,
    and combinations that mean nothing."""
    found: list[tuple[str, str]] = []
    family = objective.family
    for key in overrides or {}:
        each = component(key)
        if each is not None and family not in each.families:
            accepted = ", ".join(sorted(each.families))
            found.append((key, f"objective.{key} is not a component of a {family} objective (it is of {accepted})"))
    if family == POLICY_GRADIENT:
        if objective.ratio == "none" and objective.clip.kind != "none":
            found.append(("clip.kind", f"clip.kind {objective.clip.kind} clips a ratio, and ratio is none"))
        if objective.clip.kind in ("ratio", "dual") and objective.clip.low >= 1:
            found.append(("clip.low", "clip.low is below 1: the ratio's lower bound is 1 - clip.low, above 0"))
        if objective.distillation.coefficient == 0:
            for key in overrides or {}:
                if key.startswith("distillation.") and key != "distillation.coefficient":
                    found.append(
                        (key, f"objective.{key} shapes a distillation term, and distillation.coefficient is 0")
                    )
    if family in (POLICY_GRADIENT, DISTILLATION):
        if objective.importance.correction == "mask" and objective.importance.floor >= objective.importance.cap:
            found.append(("importance.floor", "importance.floor is below importance.cap: a mask keeps what is between"))
        if objective.kl.target == "reference" and objective.reference == "none":
            found.append(("reference", "kl.target reference needs a reference: reference = base"))
        if objective.kl.target == "none" and objective.kl.coefficient != 0:
            found.append(("kl.coefficient", "kl.coefficient weighs a KL penalty, and kl.target is none"))
        if objective.kl.target != "reference" and objective.reference != "none":
            found.append(("reference", f"a {family} objective reads the reference only for kl.target = reference"))
    if objective.distills:
        found += _distillation_problems(objective)
    if family == PREFERENCE:
        loss = objective.preference.loss
        free = loss in ("margin", "odds_ratio")
        if free and objective.reference != "none":
            found.append(("reference", f"the {loss} loss compares likelihoods alone: reference = none"))
        if not free and objective.reference == "none":
            found.append(("reference", f"the {loss} loss compares with a reference: reference = base"))
        if loss == "odds_ratio" and not objective.preference.length_normalized:
            found.append(("preference.length_normalized", "an odds ratio is of length-normalized likelihoods"))
        if loss == "kto" and objective.likelihood.coefficient != 0:
            found.append(("likelihood.coefficient", "a labelled example has no chosen side for a likelihood term"))
    return found


def _distillation_problems(objective: Objective) -> list[tuple[str, str]]:
    """What is wrong with a distillation, or a policy gradient's distillation term."""
    said = objective.distillation
    found: list[tuple[str, str]] = []
    renormalized = said.form == "top_k" and said.divergence != "reverse_kl"
    if said.form == "policy_gradient" and said.divergence != "reverse_kl":
        found.append(("distillation.divergence", "the policy_gradient form estimates the reverse KL from the sampled "
                      "tokens alone: distillation.divergence = reverse_kl, or distillation.form = top_k"))  # fmt: skip
    if said.form == "top_k" and said.top_k < 1:
        found.append(("distillation.top_k", "the top_k form is over the teacher's top-k tokens: distillation.top_k is "
                      "at least 1"))  # fmt: skip
    if said.form == "top_k" and said.advantage_clip != 0:
        found.append(("distillation.advantage_clip", "distillation.advantage_clip bounds the policy_gradient form's "
                      "advantage, and the form is top_k: 0"))  # fmt: skip
    if said.temperature != 1 and not renormalized:
        found.append(("distillation.temperature", "distillation.temperature softens the renormalized top-k "
                      "distributions of forward_kl and jsd in the top_k form: 1 otherwise"))  # fmt: skip
    if said.divergence == "jsd" and said.beta >= 1:
        found.append(("distillation.beta", "distillation.beta is the teacher's weight in the mixture, below 1"))
    top_k_alone = objective.family == DISTILLATION and said.form == "top_k"
    if top_k_alone and objective.kl.placement == "reward" and objective.kl.target != "none":
        found.append(("kl.placement", "the top_k form has no advantage to take a KL penalty from: kl.placement = loss"))
    return found


def resolved(preset: str, overrides: Mapping[str, JsonValue] | None = None) -> Objective:
    """A preset with `overrides` (dotted keys under `objective.`) in place. Raises `ValueError` for a preset that does
    not exist, an override that is no component or not of the family, a value it does not take, or a combination that
    means nothing (`problems`)."""
    made, said = composed(preset, overrides)
    if said:
        raise ValueError("; ".join(reason for _, reason in said))
    return made


def composed(preset: str, overrides: Mapping[str, JsonValue] | None = None) -> tuple[Objective, list[tuple[str, str]]]:
    """A preset with `overrides` in place, and what is wrong with it (`problems`), by dotted key. Raises `ValueError`
    for a preset that does not exist, or an override that is no component or a value it does not take."""
    if preset not in PRESETS:
        raise ValueError(f"objective.preset is one of {', '.join(PRESETS)}, not {preset!r}")
    given = dict(overrides or {})
    made = _followed(_with(PRESETS[preset].objective, given), given)
    return made, problems(made, given)


def _followed(objective: Objective, given: Mapping[str, JsonValue]) -> Objective:
    """The components that follow from others where they were not given: the reference, from the KL's target or the
    preference loss; an odds ratio's length normalization."""
    made = objective
    if "reference" not in given and ("kl.target" in given or "preference.loss" in given):
        if made.family in (POLICY_GRADIENT, DISTILLATION):
            made = replace(made, reference="base" if made.kl.target == "reference" else "none")
        elif made.family == PREFERENCE:
            made = replace(made, reference="none" if made.preference.loss in ("margin", "odds_ratio") else "base")
    if (
        made.family == PREFERENCE
        and made.preference.loss == "odds_ratio"
        and "preference.length_normalized" not in given
    ):
        made = replace(made, preference=replace(made.preference, length_normalized=True))
    return made


LEGACY = ("objective", "ratio", "clip_low", "clip_high", "segment_clip_low", "segment_clip_high", "truncate")
"""The trainer settings that once said the objective, which still say it (`from_trainer_settings`)."""
_ALIASES = {POLICY_GRADIENT: "default", LIKELIHOOD: "sft"}


def from_trainer_settings(said: Mapping[str, Any]) -> tuple[str | None, dict[str, JsonValue]]:
    """The preset and overrides that a trainer's settings named the objective by (`LEGACY`), where they name it:
    `objective = "policy_gradient"` is the `default` preset and `"likelihood"` is `sft`; `ratio = "segment"` is a
    segment ratio and weight clipped to `segment_clip_low` and `segment_clip_high` (3e-4, 4e-4 by default), a mean over
    segments; `clip_low` and `clip_high` bound a token ratio; `truncate` is the importance weight's cap (none: the
    weight untruncated). A name other than those two is a preset's."""
    kind = said.get("objective")
    preset = _ALIASES.get(kind, kind) if isinstance(kind, str) else None
    overrides: dict[str, JsonValue] = {}
    family = PRESETS[preset].objective.family if preset is not None and preset in PRESETS else POLICY_GRADIENT
    if family != POLICY_GRADIENT:
        return preset, overrides
    if said.get("ratio") == "segment":
        overrides |= {"ratio": "segment", "importance.level": "segment", "aggregate": "segment_mean"}
        overrides["clip.low"] = float(said.get("segment_clip_low", 3e-4))
        overrides["clip.high"] = float(said.get("segment_clip_high", 4e-4))
    else:
        for name, key in (("clip_low", "clip.low"), ("clip_high", "clip.high")):
            if said.get(name) is not None:
                overrides[key] = float(said[name])
    if "truncate" in said:
        truncate = said["truncate"]
        overrides |= (
            {"importance.correction": "untruncated"} if truncate is None else {"importance.cap": float(truncate)}
        )
    return preset, overrides


def objective_of(given: Any = None, legacy: Mapping[str, Any] | None = None) -> Objective:
    """The objective a trainer is given: an `Objective`; a preset's name (or `policy_gradient`, `likelihood`); a table
    of `preset` and overrides (dotted keys, or tables of them), or one recorded by `Objective.to_json` (it has
    `family`); none for `default`. `legacy` are the trainer's other settings that once named the objective (`LEGACY`),
    taken as overrides before the table's. Raises `ValueError` for one that is wrong."""
    if isinstance(given, Objective):
        _, over = from_trainer_settings({**(legacy or {}), "objective": "default"})
        if not over or given.family != POLICY_GRADIENT:
            return given
        made = _with(given, over)
        if said := problems(made, over):
            raise ValueError("; ".join(reason for _, reason in said))
        return made
    named, overrides = from_trainer_settings(
        {**(legacy or {}), **({"objective": given} if isinstance(given, str) else {})}
    )
    if isinstance(given, Mapping):
        table = _flat(cast(Mapping[str, Any], given))
        if "family" in table:
            return Objective.from_json(table)
        named = str(table.pop("preset", named or "default"))
        overrides |= table
    return resolved(named or "default", overrides)


def _flat(table: Mapping[str, Any], prefix: str = "") -> dict[str, JsonValue]:
    flat: dict[str, JsonValue] = {}
    for key, value in table.items():
        found = component(f"{prefix}{key}")
        if isinstance(value, Mapping) and not (found is not None and "table" in found.types):
            flat |= _flat(cast(Mapping[str, Any], value), f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = cast(JsonValue, value)
    return flat


assert {field.name for field in fields(Objective)} >= {each.key.split(".")[0] for each in COMPONENTS}
