# Objectives in torch

For whoever changes or adds an objective: how the local trainers compute an objective's loss from its components, the
step, and its metrics.

**Read first:** [objectives](../libraries/rollout-train/training.md#objectives). **Next:** [Tinker trainer and
engine](rollout-tinker.md).

Code: `rollout_objectives` · See [objectives](../libraries/rollout-train/training.md#objectives), [LoRA
trainer](rollout-lora.md),
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
| `terms` | The loss of one weighted segment (`policy_gradient`, `likelihood`) or of preference items (`pair`, `labelled`), the importance weight, the Kullback-Leibler (KL) divergence estimators, aggregation, and what a minibatch's terms add up to (`SUMS`, `tally`) |
| `distillation` | The loss of one distilled segment (`distillation`, and `distilled` for either family): the policy-gradient form, the top-k divergences (`top_k_divergence`), and a policy gradient's distillation term |
| `packing` | `Pack`: segments laid out in one row of a model's input, a prefix several share once; `packs`, segments cut into packs (`grouped`, then `packed`) |
| `step` | `PolicyStep`, a step over a batch on a local policy; `Plan`, which items a step takes and in which minibatches; `metrics` and `line`, a step's and a minibatch's statistics |

## Settings

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank |
| `learning_rate` | AdamW's learning rate. Adam moves a weight by at most this much per optimizer step |
| `tokens_per_step` | Sampled tokens per optimizer step. How far a step moves the policy is set by how many optimizer steps its tokens make |
| `max_kl` | The pass stops when the policy has moved this far from where the step began, in nats per token by the k3 estimate of KL(old ‖ now) on the sampled tokens (`None`: never). A likelihood step does not stop |
| `max_gradient_norm` | Gradients are clipped to this norm before each optimizer step |
| `passes` | Passes a step takes over its items, each shuffled anew and cut into minibatches of its own (1) |
| `warmup_updates` | When a step's optimizer starts afresh, its rate rises linearly over its first this many updates, from `learning_rate / warmup_updates` (0: none) |
| `segment_tokens` | The longest segment a step can hold (`None`: any) |
| `segments_per_step` | How many segments a step can afford (`None`: any number) |
| `pack_tokens` | The most tokens one forward and backward pass runs: segments are packed into rows of up to this many (`None`: `segment_tokens`, or 8,192 where that is none, what the trainer's memory estimate allows for). A segment longer than it has a pack of its own |
| `share_prefixes` | Whether segments of a pack that start with the same tokens share them (`True`) |
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
| behavior | the engine | while sampling, under whichever [checkpoint](../libraries/rollout-train/checkpoints.md) was served then (recorded in the segment) |
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
   `segment` level the weight is one for the segment, the geometric mean of its tokens'. Every policy-gradient preset
   truncates it at 2; `importance.paper_exact` takes it away (none), for samples that are on-policy.
2. **The ratio** `r = now / old` (`ratio = token`), one for the segment (`segment`: `exp(mean(now - old))`, its
   gradient spread over the tokens as GSPO's token form spreads it), or none (`none`: the logprob itself).
3. **The surrogate**, by `clip.kind`: `r·A` (`none`); `min(r·A, clip(r)·A)` with `clip(r)` bounded to
   `1 - clip.low .. 1 + clip.high` (`ratio`); that, and for a negative advantage no less than `clip.dual · A`
   (`dual`); `sg(clip(r))·A·now` (`weight`, CISPO: every token keeps a gradient). With no ratio, `A·now`.
4. **A KL penalty** (`kl.*`), `k` of `log_r = target - now` with the target the reference or old: `k1 = -log_r`,
   `k2 = log_r²/2`, `k3 = exp(log_r) - 1 - log_r`. In the `loss` it adds `coefficient · k` to each token's loss, with
   its gradient; in the `reward` it takes `coefficient · k` from each token's advantage, with none. `k1` is refused in
   the loss: its gradient is the logprob's alone, whose mean over the policy's own samples is 0, so it would add noise
   and no pull toward the target.
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

Each side of an item is one or more segments (an [episode](../libraries/rollout-train/episodes.md)'s turns); its
log-likelihood is the sum of its sampled tokens' logprobs, or their mean (`length_normalized`), and `rho` is that less
the reference's (or the likelihood alone, for a loss without a reference). For a pair, with `h = rho_chosen -
rho_rejected`:

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

### A distillation

A distilled segment carries the teacher's logprob `T(y_t)` of each sampled token `y_t` and, for the top-k form, the
teacher's top-k tokens `V_t` at each with their logprobs. With `w` the importance weight (as a policy gradient's, from
old and behavior; NeMo-RL's `icepop` is `mask`), each sampled token's loss is, by `distillation.form`:

| Form | Divergence | Each token's loss |
|---|---|---|
| `policy_gradient` | `reverse_kl` | `-w · A · now`, with `A = clip(T(y) - old(y), -A_max, A_max)` (no gradient; `A_max = advantage_clip`, unclipped at 0): the reverse KL's gradient, estimated from the sampled tokens alone (MOPD's Eq. 3 and 4) |
| `top_k` | `reverse_kl` | `w · sum over v in V of [p(v) log(p(v)/q(v)) - p(v) + q(v)]`, `p` the policy's probabilities now and `q` the teacher's, as they are (MOPD's Eq. 5): each term at least 0, and 0 where the two agree |
| `top_k` | `forward_kl` | `w · tau² · sum over v in V of q'(v) log(q'(v)/p'(v))`, both renormalized over `V` at temperature `tau` (`p'(v) ∝ p(v)^(1/tau)`) |
| `top_k` | `jsd` | `w · tau² · [beta KL(q' ‖ m) + (1 - beta) KL(p' ‖ m)]`, `m = beta q' + (1 - beta) p'` (GKD) |

The student's logprob in the policy-gradient form's advantage is old, the step's start, as NeMo-RL's MOPD takes
`prev_logprobs`; at the first minibatch it is the logprob now. The forward KL and the JSD renormalize over the teacher's
top k: the teacher's mass outside them is dropped rather than spread, so the student is fitted to the teacher's top-k
distribution, which is the teacher's own where its top k hold nearly all its mass; at k = the vocabulary each is the full
divergence. The reverse KL of the top-k form needs no renormalization (its terms are those of the full reverse KL's sum
over `V`, with `q - p` keeping each at least 0). A token the teacher did not score (beyond its context) adds nothing,
and counts in the mean. A KL penalty is added to each token's loss, or, in the policy-gradient form, taken from its
advantage (`kl.placement = reward`). `aggregate` reduces a segment's tokens as a policy gradient's.

A policy gradient with a distillation term (`distillation.coefficient`) is the policy gradient's loss of a distilled
segment (its episode's advantage) plus the coefficient times the distillation term, reduced the same way, weighed by the
same importance weight and with no KL penalty of its own.

## The step

`PolicyStep(policy, settings).step(items, seed=...)` takes a step on a policy that gives `logprobs(tokens, positions)`
(with a gradient), `reference(tokens, positions)` when the objective reads the reference,
`logprobs_and_entropy(tokens, positions)` when it has an entropy bonus, and `logprobs_among(tokens, positions,
candidates)` (the sampled tokens' logprobs and those of a row of candidate tokens at each position, with a gradient) for
the top-k form of distillation. A policy that runs packs (`PackingPolicy`: `packing` true, as the LoRA and full-weight
policies are on the models `rollout_lora.packing` readies) gives the same of every segment of a pack in one pass:
`packed(pack, entropy=..., candidates=...)` and `packed_reference(pack)`; any other is run one segment at a time.
Which items, and with what advantages, is the [algorithm's](../libraries/rollout-train/training.md) business.

1. **The plan** (`Plan.of`). Items with a segment longer than `segment_tokens`, or with nothing sampled (a pair with a
   side that sampled nothing), are left out and counted. The rest are shuffled with the step's `seed` and cut into
   minibatches of about `tokens_per_step` sampled tokens; a last minibatch of less than half that joins the one
   before. With `passes` above 1, each further pass shuffles them anew. What would fail the step (an item of the
   wrong kind for the objective, a behaviour logprob that is not finite where an importance correction reads it) is
   raised here, before anything is computed.
2. **Where the step starts.** For a policy gradient, a distillation or a preference loss, every sampled token's
   logprob on the weights the step starts from (old), without a gradient, and the reference's where it is read. The
   first minibatch of a policy gradient or a distillation runs on those weights, so its segments' old is what it
   computes itself (with a gradient, detached) rather than a pass of their own; the result is the same. Each other
   minibatch's start is computed in the packs that minibatch's own pass makes, so that on the weights the step starts
   from its ratios are exactly 1 (in other packs bfloat16 rounds a segment's logprobs a little differently). A
   behaviour logprob that is not finite fails the step only where an importance correction reads it. A pack that does
   not fit the GPU here runs again a segment at a time, and a segment that does not fit alone is left out and counted.
3. **Each minibatch.** A weighted or distilled segment's loss is computed and its gradient accumulated a pack at a
   time (for the top-k form, with the policy's logprobs of the teacher's top-k tokens at each position). A
   preference loss is a function of each side's whole likelihood, so its gradient is taken in two parts that also
   hold one pack's activations at a time: the loss of the logprobs computed without a gradient (on the weights the
   minibatch steps from; before any update, old) gives each logprob's gradient, and each segment's logprobs, computed
   again with a gradient, are moved by it. The gradient is the loss's. A minibatch that does not fit is dropped and
   counted, and with it what its pass left of the gradient (a sharded policy's `recover`: the gradients of the weights
   it gathered and did not reduce, and the sharded model's state of the pass).
4. **The stop.** Before a minibatch's optimizer step, its estimate of KL(old ‖ now) on the sampled tokens is compared
   with `max_kl`; if it is more, the pass stops without that step. The estimate is the mean over its sampled tokens of
   k3, `(r - 1) - log r` with `log r = now - old` (`moved_kl`): never below 0, and with the KL's mean on tokens sampled
   where the step began. (The k1 estimate, `old - now`, has the same mean and either sign: a step that makes every
   sampled token likelier, as REINFORCE without a baseline or a distillation whose teacher is surer than the student
   does, reads near 0 or below by it however far the policy moved.)
5. **The update.** Gradients are clipped to `max_gradient_norm` and AdamW steps, with no weight decay, at the
   minibatch's (warmed-up) rate.

### Packs

Every pass over segments (the start, the reference, each minibatch) runs them in packs (`rollout_objectives.packing`),
as many to a pass as fit in `pack_tokens`:

- **Groups.** Sorted by their tokens, each segment joins the group before it, under the prefix they all share, where
  that prefix is at least `SHARED_PREFIX` (32) tokens, the group fits in a pack, and the group with it puts fewer
  tokens in the row than the two apart: a turn's system prompt, tools and rules, which every turn of a run repeats, are
  shared, and two environments' turns that share only a chat template's header are not grouped. Without
  `share_prefixes`, each segment is a group alone.
- **Packs.** Groups are placed first-fit-decreasing by the tokens each puts in the row: its prefix once, then each
  segment's rest.
- **A row.** Roots first (each segment alone, or each shared prefix), then branches (each segment's rest after its
  prefix). Positions restart at each root and go on from the prefix in a branch. Each token sees only the tokens of
  its own segment before it: a root's own, a branch's own and its prefix's. The policy (`rollout_lora.packing`)
  keeps them apart in each kind of layer: softmax attention runs each root on its own and every branch token over its
  prefix and its own branch, nothing copied for a branch, and Qwen3.5's gated delta rule starts each root from a zero
  state and each branch from the state its prefix ended in. A pack's activations are those of a segment as long as its
  row. A sampled token is scored from the tokens before it: one at position 0 is an error.
- **The same loss.** Each segment's logprobs are those it has alone, so its terms, their normalisation (per token,
  segment, item or group) and the minibatch's units are the unpacked step's; a minibatch's gradient is accumulated a
  pack at a time. A prefix's backward pass adds up every branch's gradient.

### A step on several GPUs

`PolicyStep(policy, settings, ranks=Ranks(rank, size, group))` shares a step among processes, one per GPU, over a model
sharded among them (`rollout_lora.sharded`); one process (`Ranks()`, the default) shares nothing. The policy is a
`SharedPolicy`: its gradients are added up across the processes, not averaged; it takes an idle pass (`idle`) and clips
its gradient by the norm over every process's shard (`clip_gradients`, an error for a gradient outside the sharded
model, which no process adds up). One may reduce a minibatch's gradient once, in its last pass (`gradient_sync`, told
before each of a minibatch's gradient passes whether it is the last: a sharded adapter keeps the others' gradients in
each process).

- **The same plan.** Every process takes the whole batch and makes the same plan of it, shuffled by the step's seed, so
  they agree on every minibatch without being told. What would fail the step is raised before any process waits on
  another, by every process alike.
- **Shares balanced by passes.** Each pass over segments is packed as on one process (the same packs, whatever the
  number of processes, so a segment is computed alike and a prefix and its branches are one pack's; the price is
  balance, since packs are shared out whole rather than made to each process's measure), and the packs are
  shared out by `rollout_objectives.ranks.shares`: no process takes more than its even share of them, rounded up (the
  processes take their passes together, so the most any takes is what the pass costs), the largest first, each to the
  process with the fewest tokens so far. A policy that runs one segment at a time shares its segments likewise. Where
  the step starts, each process computes its packs of the segments' start (and reference) logprobs, and the processes
  gather them all; a first minibatch's start, which its own pass computes, is gathered likewise.
- **The same normalisation.** Each item's loss is divided by its minibatch's units counted over the whole minibatch
  (its tokens for a token mean, its segments for a segment mean, `constant_tokens` for a constant, its items for a
  preference loss), and the sharded model sums the gradients across processes rather than averaging them. The update
  is the one a single process makes of the same minibatch, whatever the number of processes.
- **Sums added up.** A minibatch's sums (its loss, tokens, the distance moved) are added up across processes before
  anything reads them, so every process takes the same decisions: a minibatch with nothing to train on, the stop at
  `max_kl`. A preference loss's logprobs are gathered, and every process computes the loss of the whole minibatch;
  each moves its own segments by it. The gradient's norm is that of every shard's.
- **Idle passes.** A sharded model's layers are gathered by every process at once, so each takes as many passes as the
  others: a process with fewer packs takes the policy's idle passes (a two-token sequence, its loss times zero) for the
  rest.
- **No minibatch left out.** A minibatch or a start's pack that runs out of memory fails the step (the others wait on
  its passes).

On the CPU over gloo, two processes take two steps of each objective's case, one segment at a time and in packs with a
shared prefix, to within about 1e-15 of one process's, in float64 (`test_shared.py`).

## Metrics

A step returns these; a trainer adds its own (`peak_gpu_gib`, `billed_tokens`).

| Metric | Meaning |
|---|---|
| `loss` | The loss, over the units (per token, per segment, or per item) |
| `kl_floor`, `mean_mismatch` | KL(behavior ‖ old) estimated on the sampled tokens, and the mean absolute difference: how far the data is from the policy the step starts from (tokens without a finite behaviour logprob left out) |
| `mean_weight`, `truncated_fraction` | The mean importance weight, and the share of tokens whose weight was truncated or masked |
| `clip_fraction`, `mean_ratio` | The share of tokens whose ratio was outside the clip's bounds, and the mean ratio |
| `kl_moved` | KL(old ‖ now) by the k3 estimate the stop reads, as the last stepped minibatch found it: how far the step moved the policy (never below 0) |
| `kl_penalty`, `entropy` | The mean KL penalty's estimate and the mean entropy per token, where the objective has them |
| `stopped_at_max_kl` | 1 if the pass stopped at `max_kl` |
| `items`, `preference_accuracy`, `preference_margin` | For a preference loss: items trained on, the share whose chosen side's log ratio is above the rejected's (a desirable example's above `z`, an undesirable one's below), and the mean `h` (an example's distance from `z` on its label's side) |
| `chosen_log_ratio`, `rejected_log_ratio` | For pairs: the mean `rho` of each side |
| `teacher_gap` | For distilled segments: the mean, over the tokens the teacher scored, of the policy's logprob less the teacher's, as each minibatch found it before its update. On the policy's own samples, an estimate of KL(policy ‖ teacher) per token; it is also in each minibatch's line |
| `teacher_divergence` | The mean top-k divergence per scored token (the top-k form) |
| `advantage_clip_fraction`, `unscored_fraction` | The share of scored tokens whose advantage was clipped (the policy-gradient form), and the share of sampled tokens the teacher did not score |
| `gradient_norm` | Before clipping, the mean over optimizer steps |
| `optimizer_steps`, `passes`, `learning_rate`, `warmup_updates` | Minibatches stepped on, and the settings the step took |
| `tokens`, `segments` | Sampled tokens and segments trained on |
| `segments_given`, `segments_too_long`, `longest_segment_tokens` | Items the batch held, how many were left out for their length, and the longest segment kept |
| `minibatches_out_of_memory`, `start_out_of_memory` | Minibatches dropped, and items left out of the first pass |
| `packed` | 1 where the policy ran packs, 0 where it ran one segment at a time |
| `packs`, `pack_fill` | Passes over packs the step made (every pass over the batch counted; on several GPUs, every process's), and the mean share of `pack_tokens` their rows held |
| `prefix_shared_fraction` | The share of the segments' tokens that shared prefixes spared computing |
| `segment_tokens_per_second` | The segments' tokens the step ran through the model (every pass counted), a second of the step |
| `start_seconds`, `seconds` | The first pass, and the whole step |

`state/minibatches.jsonl`, which a trainer writes, has a line per minibatch stepped on (`line`): segments, tokens,
loss, clip fraction, KL estimate (as `kl_moved`) and learning rate, and the trainer's gradient norm.

## Tests

`tests/rollout_objectives/` needs torch. `test_presets.py` pins every preset's values to its paper and compares its
composed loss and gradient with a direct transcription of the paper's formula on a fixed batch: with
`importance.paper_exact`, or on samples taken where the step starts (every weight 1), the paper's loss itself; as the
preset composes it on samples taken elsewhere, the paper's loss with each token weighed by its truncated importance
weight.
`test_components.py` covers each component alone (clipping by kind, the importance corrections, the KL estimators and
their placement, the entropy bonus, aggregation), the default against the step's loss as it was written before
components (to the last bit), and the step over pairs and labelled examples (its two-part gradient equal to the
gradient of the whole loss, the direction of DPO's, SimPO's and KTO's updates, the stop at `max_kl`). `test_step.py`
covers the step on a toy policy: the direction of an update, minibatches, warmup, forced tokens, segments left out, the
KL stop (also where every advantage pushes the same way, REINFORCE's and a policy-gradient distillation's, whose k1
estimate falls below 0 as the policy moves), a missing logprob, the likelihood objective, the importance weight and its
truncation, the token clip and the segment ratio. `test_distillation.py` covers distillation's terms against their definitions (each top-k divergence over
the whole vocabulary equal to the full one: GKD's JSD as TRL writes it, Hinton's softened KL, the reverse KL), tokens
the teacher did not score, a teacher that gave fewer than k tokens, the importance mask, a KL in the reward and in the
loss, a policy gradient's distillation term, and the step on a toy policy, whose every distillation preset moves it
toward its teacher on its own samples. `test_shared.py` takes each objective's case on two processes under torchrun,
a toy model sharded with FSDP2 over gloo, one segment at a time and in packs, against one process, and checks how
passes are shared out. `test_packing.py` covers packs' layout (first-fit-decreasing, a shared prefix once, each
segment's tokens at its own positions), a segment grouped only where that spares tokens (two environments' turns, each
under its own prompt), a token at position 0 refused, a pack that runs out of memory running again a segment at a time,
the start computed in each minibatch's own packs, and what fails a step raised before it computes anything; packs on
real models are `tests/rollout_lora/test_packing.py`'s
([LoRA trainer](rollout-lora.md#packs)).
