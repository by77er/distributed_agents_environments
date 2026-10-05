# Objectives design: families, components, presets, distillation and judges

Status: proposed. It builds on the step code in `rollout_lora.step` and `rollout_lora.objectives` and on the
declarations in `rollout_train.providers`, `rollout_train.run_settings` and `rollout_train.validation`.

## Goals

- An objective is chosen by its **family**, the primary selector, and composed from **components**. These are
  orthogonal settings (the advantage, the ratio, clipping, importance correction, KL, entropy, aggregation, the
  reference), so a combination is a configuration rather than another hard-coded loss.
- The objectives of the literature ship as **presets**: each one a family plus component values, cited, and tested
  against a direct transcription of the paper's formula. A run names a preset and overrides any component.
- Two more sources of supervision: **distillation** from a teacher's logprobs, on-policy and off-policy (not black-box
  distillation from text alone), and **LLM judges** as a source of rewards and preferences.
- One definition serves every training provider. Local LoRA and full-weight trainers compute it in torch. Tinker runs
  it through its custom-loss path, or through its built-in loss when the composition is one Tinker has built in.

## Families

A family fixes what a batch item is and what the core term of the loss is. Everything else is a component.

| Family | Batch item | Core term | Needs |
|---|---|---|---|
| `policy_gradient` | a segment with an advantage | the advantage times the policy's logprob of each sampled token, under the ratio and clipping components | behaviour logprobs when importance correction is on |
| `preference` | a pair (chosen, rejected) over a shared context, or a single example labelled desirable or undesirable | a function of the policy-to-reference log-likelihood ratio of each side | a reference, unless the loss is reference-free |
| `distillation` | a segment with the teacher's logprobs for its tokens | a divergence between the teacher's and the policy's next-token distributions | a teacher channel that returns logprobs |
| `likelihood` | a segment with a weight | the weighted log-likelihood of its tokens (supervised fine-tuning, imitation) | nothing more |

A run may add terms from another family with a coefficient: a policy gradient plus a distillation term, or a
preference loss plus a likelihood term (as ORPO does).

## Components

Each component has a small set of values. A family accepts only the components that mean something for it, and
validation refuses the rest.

| Component | Values | Families |
|---|---|---|
| `advantage.baseline` | `group_mean`, `leave_one_out`, `none` | policy_gradient |
| `advantage.scale` | `none`, `group_std`, `batch_std` | policy_gradient |
| `advantage.filter` | `none`, `equal_scores` (DAPO's dynamic sampling) | policy_gradient |
| `ratio` | `token`, `segment` (the geometric mean of its tokens' ratios, as GSPO) | policy_gradient |
| `clip.kind` | `none`, `ratio` (PPO), `weight` (clip the importance weight and stop its gradient, as CISPO), `dual` (with a lower bound for negative advantages) | policy_gradient |
| `clip.low`, `clip.high` | numbers; asymmetric for clip-higher | policy_gradient |
| `importance.correction` | `none`, `truncate` (TIS), `mask` (drop tokens outside the bounds) | policy_gradient, distillation |
| `importance.level` | `token`, `segment` | policy_gradient, distillation |
| `importance.cap` | a number, or low and high bounds for `mask` | policy_gradient, distillation |
| `kl.target` | `none`, `reference`, `old` | policy_gradient, distillation |
| `kl.estimator` | `k1`, `k2`, `k3` | policy_gradient, distillation |
| `kl.placement` | `loss`, `reward` | policy_gradient |
| `kl.coefficient` | a number | policy_gradient, distillation |
| `entropy.coefficient` | a number | policy_gradient |
| `aggregate` | `token_mean` (over the minibatch's tokens), `segment_mean` (over each segment's tokens, then segments), `constant` (summed and divided by a fixed token count, as Dr. GRPO) | every family |
| `reference` | `base` (the adapter switched off), `checkpoint` (a named one), `none` | preference, and policy_gradient or distillation with a KL to the reference |
| `preference.loss` | `sigmoid` (DPO), `hinge`, `square` (IPO), `margin` (SimPO), `kto` | preference |
| `preference.beta`, `preference.margin` | numbers | preference |
| `preference.length_normalized` | true or false | preference |
| `preference.weights` | the desirable and undesirable weights | preference with `kto` |
| `distillation.divergence` | `reverse_kl`, `forward_kl`, `jsd` (with its mixing weight) | distillation |
| `distillation.top_k` | how many of the teacher's logprobs each position carries | distillation |
| `distillation.temperature` | a number | distillation |

The step's existing controls stay as they are: `max_kl` (stop a pass when the policy has moved too far), the
gradient norm, the learning rate and warmup, `passes`, and `tokens_per_step`.

## Presets

A preset is a family and component values, with the paper it comes from. Its test transcribes the paper's loss
directly on a fixed batch and compares the composed objective with it. Values below are the papers' defaults, and the
test pins each one to its source.

| Preset | Family | What sets it apart | Source |
|---|---|---|---|
| `reinforce` | policy_gradient | no ratio, no clipping, no importance correction | Williams, 1992 |
| `rloo` | policy_gradient | leave-one-out baseline | Ahmadian et al., 2024 |
| `ppo_clip` | policy_gradient | token ratio clipped symmetrically at 0.2; advantages from the group (no critic) | Schulman et al., 2017 |
| `grpo` | policy_gradient | group mean and standard deviation; token ratio clipped at 0.2; KL to the reference by k3 in the loss; mean over each segment's tokens, then segments | Shao et al., 2024 (DeepSeekMath) |
| `dr_grpo` | policy_gradient | group mean without the standard deviation; constant aggregation; no KL | Liu et al., 2025 |
| `dapo` | policy_gradient | clip-higher (0.2, 0.28); token mean; dynamic sampling; no KL | Yu et al., 2025 |
| `gspo` | policy_gradient | segment ratio clipped at (3e-4, 4e-4); segment mean | Zheng et al., 2025 |
| `cispo` | policy_gradient | the importance weight clipped with its gradient stopped; no update clipping | MiniMax, 2025 (MiniMax-M1) |
| `dpo` | preference | sigmoid loss over pairs, against the reference | Rafailov et al., 2023 |
| `ipo` | preference | square loss | Azar et al., 2023 |
| `simpo` | preference | length-normalized margin loss, no reference | Meng et al., 2024 |
| `kto` | preference | unpaired desirable and undesirable examples | Ethayarajh et al., 2024 |
| `orpo` | preference plus likelihood | an odds-ratio term added to supervised fine-tuning, no reference | Hong et al., 2024 |
| `on_policy_distillation` | distillation | reverse KL on the student's own samples, scored by the teacher | Agarwal et al., 2024 (GKD); Thinking Machines, 2025 |
| `distillation` | distillation | forward KL to the teacher's top-k logprobs on the teacher's samples | Hinton et al., 2015; Kim and Rush, 2016 |
| `sft` | likelihood | supervised fine-tuning | — |

Importance correction is a modifier any policy-gradient preset can take. `truncate` follows truncated importance
sampling and `mask` follows masked importance sampling, for the gap between the sampler and the trainer (Yao et al.,
2025).

The platform's present default becomes a preset of its own, `default`: Dr. GRPO's advantages, DAPO's clip-higher and
token mean, and truncated importance sampling at 2, with the `max_kl` stop. Existing runs read it from their recorded
settings, unchanged.

A run says `objective.preset = "dapo"` and may override components (`objective.kl.target = "reference"`,
`objective.kl.coefficient = 0.01`). Its start records the fully resolved objective, so a run's objective never depends
on what a preset means later. Users' own combinations are saved as run-settings presets (`NAME@N`), as today.
Coefficients can change between steps; the family and the structural components cannot.

## Where it lives

- **The declaration**, `Objective` (family, components, presets, and which components each family accepts), lives in
  `rollout_train`, free of torch. The run settings, validation and the New run form read it there.
- **The torch implementation** is a small package of its own: the composed per-segment terms, aggregation, the step
  plan and the step statistics. The local LoRA and full-weight trainers and Tinker's custom-loss path all use it.
  rollout-tinker then no longer depends on rollout-lora and the heavy packages it brings.
- **Tinker's built-in losses** (`ppo`, `cispo`, `importance_sampling`, `cross_entropy`) are faster than a custom loss.
  When a resolved objective is exactly one of them, the Tinker provider uses it, and tests check that both paths
  agree number for number.
- **The algorithm** (`rollout_train.algorithm`) reads the advantage components and builds each family's batch items:
  weighted segments, pairs, labelled examples, or segments with teacher logprobs.

## Distillation

- **On-policy.** The student samples as it does for reinforcement learning. The teacher channel scores each sampled
  token with its own logprobs (prompt logprobs over the student's tokens, with the top k at each position), and the
  loss is the reverse KL per token. It needs a teacher provider that returns prompt logprobs, with the same tokenizer
  as the student, which the validation's renderer-family rule already checks. Scoring is built: `Engine.score` gives
  the logprobs of given tokens with the top k at each position, on `VllmEngine` (vLLM's prompt logprobs),
  `RemoteEngine` and engine hosts, and the gateway's `POST /v1/scores` asks a run's channel for them, recording each
  request as a turn that is never trained on ([scoring tokens](../libraries/rollout-train/gateway.md#scoring-tokens)).
- **Off-policy, with logprobs.** The teacher samples, with its top-k logprobs at each position (built:
  `generate(…, top=K)`), and the student fits them by forward KL. A dataset of teacher samples (the dataset module, with a `teacher` supervision) serves as
  well as live sampling. Sampling stays exact as far as it goes: the teacher's turns are recorded through the gateway
  like any other.
- Either kind can be mixed with a policy gradient by a coefficient. Black-box distillation (from a teacher's text
  alone) is left out.

## LLM judges

A judge is a channel serving a fixed model, or one following a checkpoint, with no training of its own. It is
reached through the gateway, so every judge call is recorded, counted in spend, and visible in the monitor.

- **What a judge gives:**
  - a **score** against a rubric, used as a reward or as part of one;
  - a **comparison** of two transcripts, which feeds the preference family directly, or ranks a group's episodes for
    a policy gradient (Bradley–Terry scores within the group).
- **Rubrics** belong to the environment, or to a suite's entry for evals, and are versioned with it, so a score
  always says which rubric and which judge gave it.
- **Robustness:**
  - comparisons are asked both ways round, with the order of the two transcripts swapped;
  - a judge may be asked several times and its answers combined;
  - answers are cached by a hash of the transcript and the rubric.
- **Validation** refuses a judge that is the trained channel itself, unless the run says so (self-judging), and
  checks that the judge's provider honours the sampling settings the rubric asks for.
- **Evals:** judge-scored environments are ordinary suite entries, so a suite can mix verifiable and judged scores,
  each reported on its own.

### What is built

- **A judge is a model slot that is not trained.** A program declares it `ModelSlot(trained=False, judge=True)`; the
  binding, the slot's key and every turn record it (`trained`), its segments are kept in the episode marked untrained,
  counted in what the episode sampled and shown in the monitor's episode view, and the algorithm and datasets never
  train on them. An episode's reward is its trained slots'.
- **Each slot samples the channel the run binds it to** (`slots.SLOT`; `rollout_train.slots`). The channel serves a
  fixed model (the base model, or a pinned checkpoint) or follows another channel `lag` serving records behind
  (`rollout_train.serving.source_of`), whatever serves it: a gateway's `ChannelDirectory` builds every channel a run's
  start names over its providers' servers, followers load what a following or pinned channel serves, and a run over a
  profile binds its slots to the profile's channels.
- **Validation** refuses an untrained slot left unbound, a judge's channel without a provider or a model, a provider
  that does not offer the model, a judge bound to the trained channel or one following it without `self_judging`, and
  a `follows` channel that names no channel of the run.
- **Rubric scores**, in the [judging environment](../products/judging.md): versioned rubrics in the environment's
  code, a strictly read JSON verdict asked for again once, the normalized score as the reward, and the judge asked
  twice and averaged on request. The result records the rubric's version, every reply and the score.

### What remains

- **Comparisons** of two transcripts, asked both ways round, ranking a group's episodes (Bradley–Terry) or feeding
  the preference family as pairs; they wait for that family.
- **Judges that build datasets**: comparisons and scores kept as preference or SFT data.
- **Caching** answers by a hash of the transcript and the rubric.
- **Sampling settings a rubric asks for** (a judge at temperature 0, say), checked against what the provider honours.
- **Spend limits** that count a judge's paid tokens: its turns are recorded with their token counts, and
  `estimated_spend` still estimates the trained channel's alone.
- **Channels built from a run's start beyond vLLM's API**: the directory serves providers whose servers answer vLLM's
  API at the cluster config's endpoints; a judge on Tinker's sampler or a frontier API through it waits for the
  gateway's samplers of those kinds.

## Order of work

1. **The composable objective.** Build the declaration, the shared torch package, and the presets. Make the
   current behaviour reproduce exactly as `default`, `dapo` and `gspo`, and add `grpo`, `dr_grpo`, `rloo`, `cispo` and
   `reinforce`. Wire `objective.preset` and component overrides into the run settings and validation.
2. **The preference family**: `dpo`, `ipo`, `simpo`, `kto`, `orpo`, with pairs from a group's best and worst
   episodes and from datasets. Use the adapter-off reference for LoRA.
3. **Judges**: judge channels and rubric scores (built: [what is built](#what-is-built)), then comparisons, and
   comparisons feeding preferences and group rankings.
4. **Distillation**: prompt logprobs in `VllmEngine` (built: scoring through the engines and the gateway), then
   on-policy distillation, then off-policy distillation with teacher datasets.
