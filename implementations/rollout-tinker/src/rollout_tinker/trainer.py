# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""`TinkerTrainer`: a `Trainer` whose weights live at Thinking Machines.

A step resumes its parent's training state there (or starts a LoRA run on the model), takes the minibatches of
`rollout_objectives.step.PolicyStep` (its `Plan`: the same filter, shuffle and cut), sends each as a Tinker loss, saves
a training state and a sampler checkpoint named after the version, and leaves pointers to them
(`rollout_tinker.weights`).

**The objective, exactly** (`rollout_objectives.terms`). Tinker's built-in losses are faster than a custom one, and
each takes one logprob per token to compare with. Where the objective, with the importance weight, the aggregation's
scale and the minibatch's units folded into the advantages, is one of them in value and gradient, that loss is sent
(`route`):

| Objective | Updates | Tinker | Compared with; advantage |
|---|---|---|---|
| likelihood | any | `cross_entropy` | weights A·s/U |
| no ratio (REINFORCE), no correction | any | `cross_entropy` | weights A·s/U |
| no ratio, a weight | several | a forward pass for `old`, then `cross_entropy` | weights A·w·s/U |
| token ratio, no correction | one | `cross_entropy` | weights A·s/U |
| token ratio or none, a weight (truncated or not) | one | `cispo`, clipped to 0 .. the cap | behaviour; A·s/U |
| token ratio, clipped (PPO) | several | a forward pass for `old`, then `ppo` | old; A·w·s/U |
| token ratio, weight clipped (CISPO) | several | a forward pass for `old`, then `cispo` | old; A·w·s/U |
| token ratio, unclipped | several | a forward pass for `old`, then `importance_sampling` | old; A·w·s/U |
| anything else | any | (a forward pass for `old` if several) then a custom loss | — |

Anything else is a segment ratio, dual clipping, a mask, a KL penalty, a preference loss or a distillation (alone, or as
a policy gradient's term: the teacher's scores are in the distilled items, and the student's logprobs are the sampled
tokens'), and its custom loss is the objective itself. `ppo` and `cispo` are clipped to 1 - `clip.low` .. 1 +
`clip.high`. `w` is the importance weight, `s` the aggregation's scale of each token (1 for a token mean or a sum, one
over the segment's tokens for a segment mean, one over `constant_tokens` for `constant`) and `U` the minibatch's units.
With one update `old` is the logprob now: a ratio is 1 and unclipped, so every clipped surrogate's gradient is the
weighted advantage's, the forward-backward's own output is `old`, and no forward pass is needed. The custom loss (Tinker
computes logprobs, the objective is computed here with its gradient, and Tinker takes a pass on a linear stand-in with
that gradient) costs a forward pass more than a built-in loss. Tinker gives no reference logprobs, no entropies and no
logprobs of tokens other than the sampled ones here, so an objective that reads any of them (the last: the top-k form
of distillation) is refused (validation says so before a run starts).

Each minibatch's statistics are the objective's terms of the logprobs the forward-backward returns (the policy before
that update), so a step's metrics are `PolicyStep`'s (`rollout_objectives.step.metrics`). It says how far a step has got
as `PolicyStep` does (`rollout_objectives.step.StepProgress`, `rollout_train.trainer.Progressing`), counting each call
to Tinker as a pack: the forward pass for `old`, then each minibatch's forward-backward. A minibatch that finds the
policy further than `max_kl` from where the step began, by the k3 estimate the LoRA step reads
(`rollout_objectives.terms.moved_kl`), stops the pass; its gradient has been accumulated where no call clears it, so
that client is not used again.
"""

import asyncio
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import JsonValue
from tinker import AdamParams, Datum, ForwardBackwardOutput

from rollout_objectives.distillation import distilled
from rollout_objectives.step import (
    GRADIENT_WORK,
    MINIBATCHES,
    Plan,
    StepProgress,
    from_sampler,
    line,
    metrics,
    preference_terms,
)
from rollout_objectives.terms import SUMS, Terms, moved_kl, tally, terms, units
from rollout_tinker.data import datum, rows
from rollout_tinker.service import Service, Trainable, Unpaid, said, service_of, unpaid
from rollout_tinker.settings import TinkerSettings
from rollout_tinker.weights import checkpoint_name, pointer, write_pointer
from rollout_train.objectives import LIKELIHOOD, PREFERENCE, Objective
from rollout_train.recorder import Segment
from rollout_train.trainer import (
    STATE,
    WEIGHTS,
    Budget,
    Distilled,
    Files,
    Item,
    Progress,
    Step,
    StepFailed,
    Weighted,
    segments_of,
)

__all__ = ["TinkerTrainer", "route"]

CUSTOM = "custom"
"""The route of a loss Tinker has not built in: the objective, computed here."""
BEHAVIOUR_CISPO = "cispo against the behaviour"
"""`cispo` compared with the behaviour logprobs, clipped to 0 .. the cap (`UNTRUNCATED` for none): the importance
weight, truncated."""
UNTRUNCATED = 1e9
"""`cispo`'s upper clip when the importance weight is not truncated."""


def route(objective: Objective, single: bool) -> str:
    """How a minibatch of `objective` is sent to Tinker (`single`: the step makes one optimizer update): one of
    Tinker's losses (`cross_entropy`, `importance_sampling`, `ppo`, `cispo`), `BEHAVIOUR_CISPO`, or `CUSTOM`."""
    if objective.family == LIKELIHOOD:
        return "cross_entropy"
    if objective.family == PREFERENCE or objective.distills:
        return CUSTOM
    if objective.kl.target != "none" or objective.entropy.coefficient != 0.0:
        return CUSTOM
    correction = objective.importance.correction
    if objective.ratio == "segment" or objective.clip.kind == "dual" or correction == "mask":
        return CUSTOM
    if objective.ratio == "none" or single:  # the gradient is the weighted advantage's: a ratio of 1 is not clipped
        if correction in ("truncate", "untruncated") and single:
            return BEHAVIOUR_CISPO
        return "cross_entropy"
    return {"ratio": "ppo", "weight": "cispo", "none": "importance_sampling"}[objective.clip.kind]


@dataclass
class _Segment:
    """A segment of the step, as Tinker sees it."""

    segment: Segment
    rows: list[int]
    behavior: torch.Tensor
    old: torch.Tensor | None = None


class TinkerTrainer:
    """Trains a LoRA adapter over `model` at Thinking Machines, one step at a time: a step starts from the training
    state its parent names (its optimizer too, if it is given the parent's state) and leaves pointers to the new
    checkpoints (which Tinker's bridge, `rollout_tinker.bridges`, turns into an adapter engines here load). A client
    from the step before is used again when the parent is the state it saved. `service` is what calls Tinker: by
    default a session the SDK opens with the key it finds; `module:name` of what makes another (a test names a fake
    one so). `settings` are `TinkerSettings`' (its `objective` among them); those in `CHANGEABLE`, and the changeable
    components of its objective, it takes between steps (`rollout_train.trainer.Changeable`). Raises `ValueError` for
    an objective that reads the reference, the entropy or the top-k form's logprobs, which Tinker does not give here."""

    weights = "lora"

    def __init__(self, model: str, *, service: "Service | str | None" = None, **settings: Any) -> None:
        self.settings = TinkerSettings(**settings)
        refused(self.settings.loss)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        self.model = model
        self._service = service_of(service, self.settings.project)
        self._live: tuple[str, Trainable] | None = None
        """The training state the last step saved, and the client that saved it."""
        self._lock = asyncio.Lock()
        self._told: Callable[[Progress], None] | None = None

    @property
    def objective(self) -> Objective:
        return self.settings.loss

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return self.settings.changeable()

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        self.settings = self.settings.changed(settings)  # (the live client goes on)

    def watch(self, told: Callable[[Progress], None] | None) -> None:
        """Have `told` told how far each step has got."""
        self._told = told

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        async with self._lock:
            started = time.monotonic()
            try:
                client, fresh = await self._client(parent, seed)
                metrics, minibatch_lines, clean = await self._passes(client, batch, seed, fresh)
                name = checkpoint_name(into.name)
                state = (await (await client.save_state_async(name, overwrite=True))).path
                sampler = (await (await client.save_weights_for_sampler_async(name))).path
            except StepFailed:
                self._live = None
                raise
            except Exception as error:  # (Tinker's errors, and a batch it cannot take: the run goes on)
                self._live = None
                if unpaid(error):  # (but not unpaid: every step after would be refused too)
                    raise Unpaid(f"tinker refused the step for billing: {said(error)}") from error
                raise StepFailed(f"tinker: {said(error)}") from error
            except BaseException:
                self._live = None
                raise
            try:
                await asyncio.to_thread(self._write, into, sampler, state, minibatch_lines)
            except OSError as error:  # (Tinker holds the checkpoints all the same)
                self._live = None
                raise StepFailed(f"the step's files could not be written: {error}") from error
            self._live = (state, client) if clean else None
            metrics["seconds"] = time.monotonic() - started
            return Step(metrics)

    async def _client(self, parent: Files | None, seed: int) -> tuple[Trainable, bool]:
        """The training client a step starts from, and whether its optimizer starts afresh."""
        if parent is None:
            client = await self._service.create_lora_training_client_async(
                base_model=self.model, rank=self.settings.rank, seed=seed, train_unembed=False
            )  # (an adapter of attention and MLP layers: what engines here and the merge take most simply)
            return client, True
        state = pointer(parent.state, "state") if parent.state is not None else None
        if state is not None:  # its optimizer too
            if self._live is not None and self._live[0] == state:
                return self._live[1], False
            return await self._service.create_training_client_from_state_with_optimizer_async(state), False
        weights = pointer(parent.weights, "state")
        if weights is None:
            raise StepFailed("the parent was not trained on Tinker: its weights name no Tinker checkpoint")
        return await self._service.create_training_client_from_state_async(weights), True

    async def _passes(
        self, client: Trainable, batch: Sequence[Item], seed: int, fresh: bool
    ) -> tuple[dict[str, float], list[dict[str, float]], bool]:
        """The step's passes over its items on `client`: its metrics, each minibatch's line, and whether the client
        is clean (no gradient left behind) to go on with."""
        started = time.monotonic()
        settings, objective = self.settings, self.settings.loss
        plan = Plan.of(batch, settings, seed)
        segments: dict[int, _Segment] = {}
        for segment in plan.segments:
            behavior = torch.tensor(segment.logprobs, dtype=torch.float64)
            if objective.needs_behaviour and not bool(torch.isfinite(behavior).all()):
                raise ValueError("a sampled token has no behavior logprob")
            segments[id(segment)] = _Segment(segment, rows(segment), behavior)
        updates = plan.minibatches(settings)
        single = len(updates) == 1
        way = route(objective, single)
        billed = 0.0
        # Where the step starts, when more than one update needs it: a segment sampled on these weights at the
        # logprobs the engine recorded (`old_logprobs`), and every other one's computed in a forward pass.
        sampled: set[int] = set()
        if objective.family != LIKELIHOOD and not single:
            sampled = {id(segments_of(item)[0]) for item in plan.items if from_sampler(item, settings)}
            for key in sampled:
                segments[key].old = segments[key].behavior
        starting = [part for key, part in segments.items() if key not in sampled]
        starts = objective.family != LIKELIHOOD and not single and bool(starting)
        stops = objective.family != LIKELIHOOD and settings.max_kl is not None
        tracked = StepProgress(self._told, max_kl=settings.max_kl if stops else None)
        sizes = [float(sum(len(each.tokens) for each in _segments(minibatch))) for minibatch in updates]
        whole = float(sum(len(part.segment.tokens) for part in starting))
        tracked.plan((1, whole) if starts else (0, 0.0), [(1, each * GRADIENT_WORK) for each in sizes])

        if starts:
            parts = starting
            data = [datum(part.segment.tokens, [], {"weights": []}) for part in parts]
            out = await (await client.forward_async(data, "cross_entropy"))
            billed += sum(len(part.segment.tokens) - 1 for part in parts)
            for part, found in zip(parts, out.loss_fn_outputs, strict=True):
                part.old = _logprobs(found)[part.rows]
            tracked.ran(1, whole, whole, gradient=False)
        started_pass = time.monotonic()

        totals = dict.fromkeys(SUMS, 0.0)
        moved = 0.0
        optimizer: dict[str, list[float]] = {}
        lines: list[dict[str, float]] = []
        stopped = False
        for update, minibatch in enumerate(updates):
            tracked.begin(update + 1)
            parts = [segments[id(each)] for each in _segments(minibatch)]
            if objective.family == PREFERENCE:
                count = float(len(minibatch))
            else:
                count = sum(
                    units(objective, item.segment.sampled)
                    for item in minibatch
                    if isinstance(item, Weighted | Distilled)
                )
            rate = settings.rate(len(lines), fresh=fresh)
            pending = await self._sent(client, minibatch, parts, count, way)
            billed += sum(len(part.segment.tokens) - 1 for part in parts) * (2 if way == CUSTOM else 1)
            if stops and update > 0:  # the distance, before the update
                sums = self._measured(minibatch, parts, await pending)
                tracked.ran(1, sizes[update], sizes[update], gradient=True)
                if sums["moved"] / max(sums["tokens"], 1.0) > (settings.max_kl or 0.0):
                    stopped = True
                    break
                stepped = await (await client.optim_step_async(self._adam(rate)))
            else:  # the update sent at once, beside the forward-backward
                optimizing = await client.optim_step_async(self._adam(rate))
                sums = self._measured(minibatch, parts, await pending)
                tracked.ran(1, sizes[update], sizes[update], gradient=True)
                stepped = await optimizing
            moved = sums["moved"] / max(sums["tokens"], 1.0)
            for key, value in sums.items():
                totals[key] += value
            reported = {key: float(value) for key, value in (stepped.metrics or {}).items()}
            for key, value in reported.items():
                optimizer.setdefault(key, []).append(value)
            lines.append({**line(sums, count, rate), **{f"optimizer_{key}": value for key, value in reported.items()}})
            kl = moved if objective.family != LIKELIHOOD else None
            loss, clipped = totals["loss"] / max(totals["units"], 1.0), totals["clipped"] / max(totals["tokens"], 1.0)
            tracked.stepped(loss=loss, kl=kl, clip_fraction=clipped)

        found = metrics(
            totals,
            [
                (part.behavior, part.old)
                for key, part in segments.items()
                if part.old is not None and key not in sampled
            ],
            plan=plan,
            given=len(batch),
            moved=moved,
            updates=len(lines),
            settings=settings,
            fresh=fresh,
            stopped=stopped,
            start_seconds=started_pass - started,
            from_sampler=len(sampled),
        )
        found["billed_tokens"] = billed
        norms = next((values for key, values in optimizer.items() if "grad" in key and "norm" in key), None)
        if norms:  # (if Tinker reports it: before clipping, mean over the updates)
            found["gradient_norm"] = sum(norms) / len(norms)
        return found, lines, not stopped

    async def _sent(
        self, client: Trainable, minibatch: list[Item], parts: list[_Segment], count: float, way: str
    ) -> Any:
        """A minibatch's forward-backward, sent: what awaits its output."""
        objective = self.settings.loss
        if way == CUSTOM:
            return await client.forward_backward_custom_async(
                [self._datum(part) for part in parts], self._custom(minibatch, parts, count)
            )
        weighted = [item for item in minibatch if isinstance(item, Weighted)]
        if objective.family == LIKELIHOOD:
            data = [
                self._datum(part, weights=[item.advantage * _scale(objective, part) / count] * len(part.rows))
                for item, part in zip(weighted, parts, strict=True)
            ]
            return await client.forward_backward_async(data, "cross_entropy")
        data: list[Datum] = []
        for item, part in zip(weighted, parts, strict=True):
            scale = item.advantage * _scale(objective, part) / count
            if way == BEHAVIOUR_CISPO:  # its clipped ratio is the truncated weight, at the step's start
                data.append(self._datum(part, logprobs=part.behavior.tolist(), advantages=[scale] * len(part.rows)))
                continue
            weight = _weight(objective, part)
            if way == "cross_entropy":
                data.append(self._datum(part, weights=(weight * scale).tolist()))
            else:
                assert part.old is not None
                data.append(self._datum(part, logprobs=part.old.tolist(), advantages=(weight * scale).tolist()))
        if way == "cross_entropy":
            return await client.forward_backward_async(data, "cross_entropy")
        if way == BEHAVIOUR_CISPO:
            config = {"clip_low_threshold": 0.0, "clip_high_threshold": _cap(objective)}
            return await client.forward_backward_async(data, "cispo", config)
        if way == "importance_sampling":
            return await client.forward_backward_async(data, "importance_sampling")
        clip = objective.clip
        config = {"clip_low_threshold": 1 - clip.low, "clip_high_threshold": 1 + clip.high}
        return await client.forward_backward_async(data, way, config)

    def _custom(self, minibatch: list[Item], parts: list[_Segment], count: float) -> Any:
        """The loss of a minibatch, as Tinker's custom loss takes it: the objective's terms of each datum's sampled
        rows (the logprob at the step's start being, with one update, the logprob now)."""
        objective = self.settings.loss

        def loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
            now = {
                id(part.segment): found[part.rows].to(torch.float64)
                for part, found in zip(parts, logprobs, strict=True)
            }
            if objective.family == PREFERENCE:
                found_terms = preference_terms(objective, minibatch, now, {})
                return torch.stack([each.loss for _, each in found_terms]).sum() / count, {}
            total = torch.zeros((), dtype=torch.float64)
            for item, part in zip(_segmented(minibatch), parts, strict=True):
                logprob = now[id(part.segment)]
                old = part.old if part.old is not None else logprob.detach()
                total = total + _terms(objective, item, logprob, old, part.behavior).loss / count
            return total, {}

        return loss

    def _measured(self, minibatch: list[Item], parts: list[_Segment], out: ForwardBackwardOutput) -> dict[str, float]:
        """A minibatch's `SUMS`, from the logprobs its forward-backward returned (the policy before its update)."""
        objective = self.settings.loss
        sums = dict.fromkeys(SUMS, 0.0)
        now: dict[int, torch.Tensor] = {}
        for part, found in zip(parts, out.loss_fn_outputs, strict=True):
            logprob = _logprobs(found)[part.rows]
            if part.old is None and objective.family != LIKELIHOOD:
                part.old = logprob.clone()  # (one update: where the step starts is what the first pass found)
            now[id(part.segment)] = logprob
        with torch.no_grad():
            if objective.family == PREFERENCE:
                for item, found in preference_terms(objective, minibatch, now, {}):
                    tally(sums, found, objective, segments=float(len(segments_of(item))))
                sums["moved"] = sum(
                    float(moved_kl(part.old, now[id(part.segment)]).sum()) for part in parts if part.old is not None
                )
                sums["tokens"] = sum(float(now[id(part.segment)].numel()) for part in parts)
                return sums
            for item, part in zip(_segmented(minibatch), parts, strict=True):
                logprob = now[id(part.segment)]
                old = part.old if part.old is not None else logprob
                tally(sums, _terms(objective, item, logprob, old, part.behavior), objective)
        return sums

    def _datum(self, part: _Segment, **values: Sequence[float]) -> Datum:
        return datum(part.segment.tokens, part.rows, values)

    def _adam(self, rate: float) -> AdamParams:
        """Adam as torch's AdamW is in the LoRA step (Tinker's own defaults are 0.95 and 1e-12)."""
        return AdamParams(
            learning_rate=rate,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            grad_clip_norm=self.settings.max_gradient_norm,
        )

    def _write(self, into: Path, sampler: str, state: str, lines: list[dict[str, float]]) -> None:
        """The step's files: the pointers, and what each minibatch did."""
        said = {"sampler": sampler, "state": state, "base_model": self.model, "rank": self.settings.rank}
        write_pointer(into / WEIGHTS, said)
        write_pointer(into / STATE, {"state": state, "sampler": sampler, "sdk": _sdk()})
        (into / STATE / MINIBATCHES).write_text("".join(json.dumps(each) + "\n" for each in lines))


def refused(objective: Objective) -> None:
    """Raises `ValueError` for an objective Tinker cannot take here: one that reads the reference (Tinker's SDK offers
    prompt logprobs from a sampler of the base model, not yet confirmed by a live test), the entropy, or the logprobs
    of tokens not sampled (the top-k form of distillation; Tinker returns the sampled tokens' logprobs only)."""
    if objective.needs_reference:
        raise ValueError(
            "Tinker gives no reference logprobs here (its SDK's prompt logprobs from a sampler of the base model are "
            "not yet confirmed by a live test): reference = none"
        )
    if objective.needs_entropy:
        raise ValueError("Tinker returns the sampled tokens' logprobs only, not the entropy: entropy.coefficient = 0")
    if objective.needs_distribution:
        raise ValueError(
            "Tinker returns the sampled tokens' logprobs only, not the student's logprobs of the teacher's top-k "
            "tokens: distillation.form = policy_gradient"
        )


def _segmented(minibatch: Sequence[Item]) -> list[Weighted | Distilled]:
    """A minibatch's weighted or distilled segments, in order."""
    return [each for each in minibatch if isinstance(each, Weighted | Distilled)]


def _terms(
    objective: Objective, item: Weighted | Distilled, logprobs: torch.Tensor, old: torch.Tensor, behavior: torch.Tensor
) -> Terms:
    """A weighted or distilled segment's terms, of its sampled tokens' logprobs."""
    if isinstance(item, Distilled):
        return distilled(objective, item, logprobs, old, behavior)
    return terms(objective, logprobs, item.advantage, old, behavior)


def _segments(minibatch: Sequence[Item]) -> list[Segment]:
    """A minibatch's segments, each once, in order."""
    return list({id(each): each for item in minibatch for each in segments_of(item)}.values())


def _scale(objective: Objective, part: _Segment) -> float:
    """What the aggregation scales each of a segment's tokens by, before the minibatch's units."""
    if objective.aggregate == "segment_mean":
        return 1.0 / max(len(part.rows), 1)
    if objective.aggregate == "constant":
        return 1.0 / objective.constant_tokens
    return 1.0


def _cap(objective: Objective) -> float:
    """The importance weight's cap, as `cispo`'s upper clip."""
    return UNTRUNCATED if objective.importance.correction == "untruncated" else objective.importance.cap


def _weight(objective: Objective, part: _Segment) -> torch.Tensor:
    """A segment's importance weights, each a constant, as the objective's correction makes them (ones for none)."""
    importance = objective.importance
    if importance.correction == "none":
        return torch.ones(len(part.rows), dtype=torch.float64)
    assert part.old is not None
    log_weight = part.old - part.behavior
    if importance.level == "segment":
        log_weight = log_weight.mean().expand_as(part.old)
    raw = torch.exp(log_weight)
    return raw if importance.correction == "untruncated" else raw.clamp(max=importance.cap)


def _logprobs(found: Any) -> torch.Tensor:
    """A datum's `logprobs` output, one per row, in float64."""
    data = found["logprobs"].data
    values = torch.tensor([float(each) for each in data], dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Tinker returned a logprob that is not finite")
    return values


def _sdk() -> str:
    import tinker

    return str(tinker.__version__)
