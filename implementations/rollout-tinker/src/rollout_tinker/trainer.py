# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""`TinkerTrainer`: a `Trainer` whose weights live at Thinking Machines.

A step resumes its parent's training state there (or starts a LoRA run on the model), takes the same minibatches as
`rollout_lora.step.PolicyStep` (the same filter, shuffle and cut), sends each as Tinker's loss, saves a training
state and a sampler checkpoint named after the version, and leaves pointers to them (`rollout_tinker.weights`).

**The objective, exactly.** Tinker's built-in losses take one reference logprob per token; ours (`rollout_lora.
objectives`) has two, the logprob at the step's start (`old`) and the one it was sampled at (`behavior`). Folding the
importance weight and the minibatch's units into the advantages makes each loss ours, in value and gradient:

| Objective, ratio | Optimizer steps | Tinker | Reference, advantage |
|---|---|---|---|
| `policy_gradient`, `token` | one | `cispo`, clipped to 0 .. `truncate` | behaviour; A/U |
| `policy_gradient`, `token` | several | a forward pass for `old`, then `ppo`, clipped to 1 ± clip | old; A·w/U |
| `policy_gradient`, `segment` | any | (a forward pass for `old` if several) then a custom loss: `terms` itself | — |
| `likelihood` | any | `cross_entropy`, weights A/U | — |

With one optimizer step `old` is the logprob now, so the forward-backward's own output is `old`, and no forward pass is
needed. The custom loss (Tinker computes logprobs, we compute the loss and its gradient here, and Tinker takes a pass
on a linear stand-in with that gradient) costs a forward pass more than a built-in loss.

Each minibatch's statistics are `terms` of the logprobs the forward-backward returns (the policy before that update),
so a step's metrics are `PolicyStep`'s. A minibatch that finds the policy further than `max_kl` from where the step
began stops the pass; its gradient has been accumulated where no call clears it, so that client is not used again.
"""

import asyncio
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from pydantic import JsonValue
from tinker import AdamParams, Datum, ForwardBackwardOutput

from rollout_lora.objectives import Terms, terms
from rollout_lora.step import minibatches, sampled
from rollout_tinker.data import datum, rows
from rollout_tinker.service import Service, Trainable, Unpaid, said, service_of, unpaid
from rollout_tinker.settings import CHANGEABLE, TinkerSettings
from rollout_tinker.weights import checkpoint_name, pointer, write_pointer
from rollout_train.trainer import STATE, WEIGHTS, Budget, Files, Step, StepFailed, Weighted

__all__ = ["MINIBATCHES", "TinkerTrainer"]

MINIBATCHES = "minibatches.jsonl"
"""In a step's state: what each of its minibatches did, one line each (as `LoraTrainer` writes it)."""
UNTRUNCATED = 1e9
"""`cispo`'s upper clip when the importance weight is not truncated."""


@dataclass
class _Segment:
    """A segment of the step, as Tinker sees it."""

    weighted: Weighted
    rows: list[int]
    behavior: torch.Tensor
    old: torch.Tensor | None = None


class TinkerTrainer:
    """Trains a LoRA adapter over `model` at Thinking Machines, one step at a time: a step starts from the training
    state its parent names (its optimizer too, if it is given the parent's state) and leaves pointers to the new
    checkpoints (which Tinker's bridge, `rollout_tinker.bridges`, turns into an adapter engines here load). A client
    from the step before is used again when the parent is the state it saved. `service` is what calls Tinker: by
    default a session the SDK opens with the key it finds; `module:name` of what makes another (a profile names a fake
    one so). `settings` are `TinkerSettings`'; those in `CHANGEABLE` it takes between steps
    (`rollout_train.trainer.Changeable`)."""

    weights = "lora"

    def __init__(self, model: str, *, service: "Service | str | None" = None, **settings: Any) -> None:
        self.settings = TinkerSettings(**settings)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        self.model = model
        self._service = service_of(service, self.settings.project)
        self._live: tuple[str, Trainable] | None = None
        """The training state the last step saved, and the client that saved it."""
        self._lock = asyncio.Lock()

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        """The settings it takes between steps (`rollout_train.trainer.Changeable`), with their values now."""
        said = asdict(self.settings)
        return {name: said[name] for name in CHANGEABLE}

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        if unknown := sorted(set(settings) - set(CHANGEABLE)):
            raise ValueError(f"{', '.join(unknown)} cannot change between steps (these can: {', '.join(CHANGEABLE)})")
        self.settings = replace(self.settings, **settings)  # (checked as any settings are; the live client goes on)

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
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
        self, client: Trainable, batch: Sequence[Weighted], seed: int, fresh: bool
    ) -> tuple[dict[str, float], list[dict[str, float]], bool]:
        """The step's passes over its segments on `client`: its metrics, each minibatch's line, and whether the
        client is clean (no gradient left behind) to go on with."""
        started = time.monotonic()
        settings, objective = self.settings, self.settings.loss
        longest = settings.segment_tokens
        order = [
            weighted
            for weighted in batch
            if (longest is None or len(weighted.segment.tokens) <= longest) and sampled(weighted)
        ]
        too_long = sum(1 for weighted in batch if longest is not None and len(weighted.segment.tokens) > longest)
        shuffled = random.Random(seed)
        shuffled.shuffle(order)
        segments: dict[int, _Segment] = {}
        for weighted in order:
            behavior = torch.tensor(weighted.segment.logprobs, dtype=torch.float64)
            if objective.reads_old and not bool(torch.isfinite(behavior).all()):
                raise ValueError("a sampled token has no behavior logprob")
            segments[id(weighted)] = _Segment(weighted, rows(weighted), behavior)
        passes = [order] + [shuffled.sample(order, len(order)) for _ in range(settings.passes - 1)]
        plan = [each for one in passes for each in minibatches(one, settings.tokens_per_step)]
        single = len(plan) == 1
        billed = 0.0

        # Where the step starts, when more than one update needs it: each sampled token's logprob on these weights.
        if objective.reads_old and not single and order:
            data = [datum(each.segment.tokens, [], {"weights": []}) for each in order]
            out = await (await client.forward_async(data, "cross_entropy"))
            billed += sum(len(each.segment.tokens) - 1 for each in order)
            for weighted, found in zip(order, out.loss_fn_outputs, strict=True):
                part = segments[id(weighted)]
                part.old = _logprobs(found)[part.rows]
        started_pass = time.monotonic()

        totals: dict[str, float] = dict.fromkeys(
            ("loss", "units", "clipped", "truncated", "tokens", "ratio", "weight", "segments"), 0.0
        )
        moved: list[float] = []
        optimizer: dict[str, list[float]] = {}
        lines: list[dict[str, float]] = []
        stopped = False
        for update, minibatch in enumerate(plan):
            parts = [segments[id(weighted)] for weighted in minibatch]
            units = sum(objective.units(part.weighted.segment.sampled) for part in parts)
            rate = settings.rate(len(lines), fresh=fresh)
            pending = await self._sent(client, parts, units, single)
            billed += sum(len(part.weighted.segment.tokens) - 1 for part in parts) * (
                2 if objective.kind == "policy_gradient" and objective.ratio == "segment" else 1
            )
            if objective.reads_old and settings.max_kl is not None and update > 0:  # the distance, before the update
                out = await pending
                sums, distance = self._measured(parts, out)
                if distance > settings.max_kl:
                    stopped = True
                    break
                stepped = await (await client.optim_step_async(self._adam(rate)))
            else:  # the update sent at once, beside the forward-backward
                optimizing = await client.optim_step_async(self._adam(rate))
                out = await pending
                stepped = await optimizing
                sums, distance = self._measured(parts, out)
            moved.append(distance)
            for key, value in sums.items():
                totals[key] += value
            for key, value in (stepped.metrics or {}).items():
                optimizer.setdefault(key, []).append(float(value))
            lines.append(
                {
                    "segments": sums["segments"],
                    "tokens": sums["tokens"],
                    "loss": sums["loss"] / units,
                    "clip_fraction": sums["clipped"] / max(sums["tokens"], 1.0),
                    "kl": distance,
                    "learning_rate": rate,
                    **{f"optimizer_{key}": float(value) for key, value in (stepped.metrics or {}).items()},
                }
            )

        tokens = max(totals["tokens"], 1.0)
        starts = [part for part in segments.values() if part.old is not None]
        start_tokens = max(sum(float(part.old.numel()) for part in starts if part.old is not None), 1.0)
        floor = sum(float((part.behavior - part.old).sum()) for part in starts if part.old is not None)
        mismatch = sum(float((part.behavior - part.old).abs().sum()) for part in starts if part.old is not None)
        metrics = {
            "loss": totals["loss"] / max(totals["units"], 1.0),
            "clip_fraction": totals["clipped"] / tokens,
            "mean_ratio": totals["ratio"] / tokens,
            "kl_floor": floor / start_tokens,
            "mean_mismatch": mismatch / start_tokens,
            "mean_weight": totals["weight"] / tokens,
            "truncated_fraction": totals["truncated"] / tokens,
            "kl_moved": moved[-1] if moved else 0.0,
            "tokens": totals["tokens"],
            "segments": totals["segments"],
            "segments_given": float(len(batch)),
            "segments_too_long": float(too_long),
            "longest_segment_tokens": float(max((len(weighted.segment.tokens) for weighted in order), default=0)),
            "optimizer_steps": float(len(lines)),
            "passes": float(settings.passes),
            "learning_rate": settings.learning_rate,
            "warmup_updates": float(settings.warmup_updates if fresh else 0),
            "stopped_at_max_kl": float(stopped),
            "billed_tokens": billed,
            "start_seconds": started_pass - started,
        }
        norms = next((values for key, values in optimizer.items() if "grad" in key and "norm" in key), None)
        if norms:  # (if Tinker reports it: before clipping, mean over the updates)
            metrics["gradient_norm"] = sum(norms) / len(norms)
        return metrics, lines, not stopped

    async def _sent(self, client: Trainable, parts: list[_Segment], units: float, single: bool) -> Any:
        """A minibatch's forward-backward, sent: what awaits its output."""
        objective = self.settings.loss
        if objective.kind == "likelihood":
            data = [self._datum(part, weights=[part.weighted.advantage / units] * len(part.rows)) for part in parts]
            return await client.forward_backward_async(data, "cross_entropy")
        if objective.ratio == "segment":
            return await client.forward_backward_custom_async(
                [self._datum(part) for part in parts], self._custom(parts, units)
            )
        if single:  # cispo, against the behaviour: its clipped ratio is our truncated weight, at the step's start
            data = [
                self._datum(
                    part,
                    logprobs=part.behavior.tolist(),
                    advantages=[part.weighted.advantage / units] * len(part.rows),
                )
                for part in parts
            ]
            ceiling = objective.truncate if objective.truncate is not None else UNTRUNCATED
            config = {"clip_low_threshold": 0.0, "clip_high_threshold": ceiling}
            return await client.forward_backward_async(data, "cispo", config)
        data: list[Datum] = []
        for part in parts:
            assert part.old is not None
            weight = torch.exp(part.old - part.behavior)
            if objective.truncate is not None:
                weight = weight.clamp(max=objective.truncate)
            data.append(
                self._datum(
                    part, logprobs=part.old.tolist(), advantages=(weight * part.weighted.advantage / units).tolist()
                )
            )
        config = {"clip_low_threshold": 1 - objective.clip_low, "clip_high_threshold": 1 + objective.clip_high}
        return await client.forward_backward_async(data, "ppo", config)

    def _custom(self, parts: list[_Segment], units: float) -> Any:
        """The loss of a minibatch of segment ratios, as Tinker's custom loss takes it: `terms` of each datum's
        sampled rows (the logprob at the step's start being, with one update, the logprob now)."""
        objective = self.settings.loss

        def loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
            total = torch.zeros((), dtype=torch.float64)
            for part, found in zip(parts, logprobs, strict=True):
                now = found[part.rows].to(torch.float64)
                old = part.old if part.old is not None else now.detach()
                total = total + terms(objective, now, part.weighted.advantage, old, part.behavior).loss / units
            return total, {}

        return loss

    def _measured(self, parts: list[_Segment], out: ForwardBackwardOutput) -> tuple[dict[str, float], float]:
        """A minibatch's sums from the logprobs its forward-backward returned (the policy before its update), and
        how far it found the policy from where the step began (nats per token)."""
        objective = self.settings.loss
        sums: dict[str, float] = dict.fromkeys(
            ("loss", "units", "clipped", "truncated", "tokens", "ratio", "weight", "segments"), 0.0
        )
        distance = 0.0
        for part, found in zip(parts, out.loss_fn_outputs, strict=True):
            now = _logprobs(found)[part.rows]
            if part.old is None and objective.reads_old:
                part.old = now.clone()  # (one update: where the step starts is what the first pass found)
            with torch.no_grad():
                said: Terms = terms(objective, now, part.weighted.advantage, part.old, part.behavior)
            sums["loss"] += float(said.loss)
            sums["units"] += objective.units(int(said.tokens))
            sums["clipped"] += said.clipped
            sums["truncated"] += said.truncated
            sums["tokens"] += said.tokens
            sums["ratio"] += said.ratio
            sums["weight"] += said.weight
            sums["segments"] += 1
            distance += said.moved
        return sums, distance / max(sums["tokens"], 1.0)

    def _datum(self, part: _Segment, **values: Sequence[float]) -> Datum:
        return datum(part.weighted.segment.tokens, part.rows, values)

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
