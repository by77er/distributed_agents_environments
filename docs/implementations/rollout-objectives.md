# Objectives in torch

Code: `rollout_objectives` · See [objectives](../libraries/rollout-train/training.md#objectives), [LoRA trainer](rollout-lora.md),
[Tinker trainer](rollout-tinker.md)

`rollout_objectives` computes what [`rollout_train.objectives`](../libraries/rollout-train/training.md#objectives)
declares: an objective's loss composed from its components, the step a local trainer takes over a batch, the plan of
minibatches every trainer of the step shares, and the step's statistics. The [LoRA and full-weight
trainers](rollout-lora.md) take its step on the GPU; the [Tinker trainer](rollout-tinker.md) takes its plan, and its
loss as a custom loss where Tinker has no built-in one. It needs torch and nothing else beyond the workspace's
libraries; it is installed with either the `gpu` or the `tinker` extra.

| Module | What it holds |
|---|---|
| `settings` | `StepSettings`: a step's settings, which the LoRA, full-weight and Tinker trainers take alike. Importing it does not load torch |
| `terms` | The loss of one weighted segment (`policy_gradient`, `likelihood`) or of preference items (`pair`, `labelled`), the KL estimators, aggregation, and what a minibatch's terms add up to (`SUMS`, `tally`) |
| `step` | `PolicyStep`, a step over a batch on a local policy; `Plan`, which items a step takes and in which minibatches; `metrics` and `line`, a step's and a minibatch's statistics |

## Settings

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank |
| `learning_rate` | AdamW's learning rate. Adam moves a weight by at most this much per optimizer step |
| `tokens_per_step` | Sampled tokens per optimizer step. How far a step moves the policy is set by how many optimizer steps its tokens make |
| `max_kl` | The pass stops when the policy has moved this far from where the step began, in nats per token (`None`: never). A likelihood step does not stop |
| `max_gradient_norm` | Gradients are clipped to this norm before each optimizer step |
| `passes` | Passes a step takes over its items, each shuffled anew and cut into minibatches of its own (1) |
| `warmup_updates` | When a step's optimizer starts afresh, its rate rises linearly over its first this many updates, from `learning_rate / warmup_updates` (0: none) |
| `segment_tokens` | The longest segment a step can hold (`None`: any) |
| `segments_per_step` | How many segments a step can afford (`None`: any number) |
| `objective` | The objective: an `Objective`, a preset's name, or a table of `preset` and component overrides. A run's `objective.*` settings give it |

A trainer's own settings can also name the objective by `ratio`, `clip_low`, `clip_high`, `segment_clip_low`,
`segment_clip_high` and `truncate`, and `objective = "policy_gradient"` (the `default` preset) or `"likelihood"`
(`sft`): they are taken as the components they say (`rollout_train.objectives.from_trainer_settings`).

Between steps a trainer takes `learning_rate`, `tokens_per_step`, `max_kl`, `max_gradient_norm` and its objective's
numbers, by their run settings' keys (`objective.clip.low`, `objective.kl.coefficient`): `changeable` lists them for
the objective's family.

## The loss

Four logprobs of each sampled token meet in it:

| Logprob | Computed by | When |
|---|---|---|
| behavior | the engine | while sampling, under whichever checkpoint was served then (recorded in the segment) |
| old | the trainer, without a gradient | at the start of the step, on the weights the step starts from |
| now | the trainer, with a gradient | in each minibatch, as the step updates the weights |
| reference | the trainer, without a gradient | under the reference model: the base with the adapter switched off, or a frozen copy |

Behavior and old differ because the data came from elsewhere: an older checkpoint, and the engine computing
differently from the trainer. Old and now differ by how far the step has moved the policy. Now and reference differ by
how far training has moved it from the base.

### A policy gradient

For each sampled token of a segment with advantage `A`:

1. **The importance weight** `w`, a constant (`importance.*`): none; `old / behavior` (`untruncated`); that, at most
   `cap` (`truncate`); or that, with a token whose weight is outside `floor` .. `cap` dropped (`mask`). At the
   `segment` level the weight is one for the segment, the geometric mean of its tokens'.
2. **The ratio** `r = now / old` (`ratio = token`), one for the segment (`segment`: `exp(mean(now - old))`, its
   gradient spread over the tokens as GSPO's token form spreads it), or none (`none`: the logprob itself).
3. **The surrogate**, by `clip.kind`: `r·A` (`none`); `min(r·A, clip(r)·A)` with `clip(r)` bounded to
   `1 - clip.low .. 1 + clip.high` (`ratio`); that, and for a negative advantage no less than `clip.dual · A`
   (`dual`); `sg(clip(r))·A·now` (`weight`, CISPO: every token keeps a gradient). With no ratio, `A·now`.
4. **A KL penalty** (`kl.*`), `k` of `log_r = target - now` with the target the reference or old: `k1 = -log_r`,
   `k2 = log_r²/2`, `k3 = exp(log_r) - 1 - log_r`. In the `loss` it adds `coefficient · k` to each token's loss, with
   its gradient; in the `reward` it takes `coefficient · k` from each token's advantage, with none.
5. **An entropy bonus**: `entropy.coefficient` times each position's entropy is taken from its loss.

Each token's loss is `-w · surrogate`, with the penalty and the bonus. `aggregate` reduces a segment's tokens: a sum
over the minibatch's tokens (`token_mean`, the segment's units its tokens); a mean over its tokens, then over the
minibatch's segments (`segment_mean`); a sum, then a mean over segments (`segment_sum`); a sum divided by
`constant_tokens`, then a mean over segments (`constant`). The minibatch's loss is the sum of its segments' over its
units.

### A likelihood

`-A · now` for each sampled token, reduced by `aggregate`: the sampled tokens' log-likelihood, each segment weighted by
its advantage (1 for a dataset's). No logprob but now is read.

### A preference loss

Each side of an item is one or more segments (an episode's turns); its log-likelihood is the sum of its sampled tokens'
logprobs, or their mean (`length_normalized`), and `rho` is that less the reference's (or the likelihood alone, for a
loss without a reference). For a pair, with `h = rho_chosen - rho_rejected`:

| `preference.loss` | Loss |
|---|---|
| `sigmoid` (DPO) | `-log sigmoid(beta·h)` |
| `hinge` | `max(0, 1 - beta·h)` |
| `square` (IPO) | `(h - 1/(2·beta))²` |
| `margin` (SimPO) | `-log sigmoid(beta·h - margin)`, of the length-normalized likelihoods |
| `odds_ratio` (ORPO) | `-beta·log sigmoid(log odds_chosen - log odds_rejected)`, the odds of `p = exp(mean logprob)` |
| `kto` (labelled examples) | `desirable·(1 - sigmoid(beta·(rho - z)))`, or `undesirable·(1 - sigmoid(beta·(z - rho)))` |

`likelihood.coefficient` adds the chosen side's mean negative logprob (ORPO's supervised term). KTO's reference point
`z` is the mean `rho` of the minibatch's examples, no less than 0, with no gradient: the paper estimates it from
mismatched pairs of the microbatch (one example's prompt with another's answer), which a multi-turn episode does not
have. A minibatch's loss is the mean over its items.

## The step

`PolicyStep(policy, settings).step(items, seed=...)` takes a step on a policy that gives `logprobs(tokens, positions)`
(with a gradient), `reference(tokens, positions)` when the objective reads the reference, and
`logprobs_and_entropy(tokens, positions)` when it has an entropy bonus. Which items, and with what advantages, is the
[algorithm's](../libraries/rollout-train/training.md) business.

1. **The plan** (`Plan.of`). Items with a segment longer than `segment_tokens`, or with nothing sampled (a pair with a
   side that sampled nothing), are left out and counted. The rest are shuffled with the step's `seed` and cut into
   minibatches of about `tokens_per_step` sampled tokens; a last minibatch of less than half that joins the one
   before. With `passes` above 1, each further pass shuffles them anew.
2. **Where the step starts.** For a policy gradient or a preference loss, every sampled token's logprob on the weights
   the step starts from (old), without a gradient, and the reference's where it is read. A behaviour logprob that is
   not finite fails the step only where an importance correction reads it. A segment that does not fit the GPU here is
   left out and counted.
3. **Each minibatch.** A weighted segment's loss is computed and its gradient accumulated one segment at a time. A
   preference loss is a function of each side's whole likelihood, so its gradient is taken in two parts that also
   hold one segment's activations at a time: the loss of the logprobs computed without a gradient (on the weights the
   minibatch steps from; before any update, old) gives each logprob's gradient, and each segment's logprobs, computed
   again with a gradient, are moved by it. The gradient is the loss's.
4. **The stop.** Before a minibatch's optimizer step, its estimate of KL(old ‖ now) on the sampled tokens is compared
   with `max_kl`; if it is more, the pass stops without that step.
5. **The update.** Gradients are clipped to `max_gradient_norm` and AdamW steps, with no weight decay, at the
   minibatch's (warmed-up) rate.

## Metrics

A step returns these; a trainer adds its own (`peak_gpu_gib`, `billed_tokens`).

| Metric | Meaning |
|---|---|
| `loss` | The loss, over the units (per token, per segment, or per item) |
| `kl_floor`, `mean_mismatch` | KL(behavior ‖ old) estimated on the sampled tokens, and the mean absolute difference: how far the data is from the policy the step starts from (tokens without a finite behaviour logprob left out) |
| `mean_weight`, `truncated_fraction` | The mean importance weight, and the share of tokens whose weight was truncated or masked |
| `clip_fraction`, `mean_ratio` | The share of tokens whose ratio was outside the clip's bounds, and the mean ratio |
| `kl_moved` | KL(old ‖ now) as the last stepped minibatch found it: how far the step moved the policy |
| `kl_penalty`, `entropy` | The mean KL penalty's estimate and the mean entropy per token, where the objective has them |
| `stopped_at_max_kl` | 1 if the pass stopped at `max_kl` |
| `items`, `preference_accuracy`, `preference_margin` | For a preference loss: items trained on, the share whose chosen side's log ratio is above the rejected's (a desirable example's above `z`, an undesirable one's below), and the mean `h` (an example's distance from `z` on its label's side) |
| `chosen_log_ratio`, `rejected_log_ratio` | For pairs: the mean `rho` of each side |
| `gradient_norm` | Before clipping, the mean over optimizer steps |
| `optimizer_steps`, `passes`, `learning_rate`, `warmup_updates` | Minibatches stepped on, and the settings the step took |
| `tokens`, `segments` | Sampled tokens and segments trained on |
| `segments_given`, `segments_too_long`, `longest_segment_tokens` | Items the batch held, how many were left out for their length, and the longest segment kept |
| `minibatches_out_of_memory`, `start_out_of_memory` | Minibatches dropped, and items left out of the first pass |
| `start_seconds`, `seconds` | The first pass, and the whole step |

`state/minibatches.jsonl`, which a trainer writes, has a line per minibatch stepped on (`line`): segments, tokens,
loss, clip fraction, KL estimate and learning rate, and the trainer's gradient norm.

## Tests

`tests/rollout_objectives/` needs torch. `test_presets.py` pins every preset's values to its paper and compares its
composed loss and gradient with a direct transcription of the paper's formula on a fixed batch.
`test_components.py` covers each component alone (clipping by kind, the importance corrections, the KL estimators and
their placement, the entropy bonus, aggregation), the default against the step's loss as it was written before
components (to the last bit), and the step over pairs and labelled examples (its two-part gradient equal to the
gradient of the whole loss, the direction of DPO's, SimPO's and KTO's updates, the stop at `max_kl`). `test_step.py`
covers the step on a toy policy: the direction of an update, minibatches, warmup, forced tokens, segments left out, the
KL stop, a missing logprob, the likelihood objective, the importance weight and its truncation, the token clip and the
segment ratio.
