# Objectives design: families, components, presets, distillation and judges

**Status: in progress.** The composable objective, the preference family and the distillation family are built
(steps 1, 2 and 4 of the [order of work](#order-of-work)): the declaration is `rollout_train.objectives`, the torch
implementation `rollout_objectives` ([objectives in torch](../implementations/rollout-objectives.md),
[objectives](../libraries/rollout-train/training.md#objectives)). Distillation is library code: the loop does not yet
ask teachers to score its episodes. Judges are proposed. Each section says which of its parts are built. A design
note: see [Design notes](README.md) for the others.

## Goals

- An objective is chosen by its **family**, the primary selector, and composed from **components**. These are orthogonal
  settings (the advantage, the ratio, clipping, importance correction, Kullback-Leibler (KL) divergence, entropy,
  aggregation, the reference), so a combination is a configuration rather than another hard-coded loss. *Built.*
- The objectives of the literature ship as **presets**: each one a family plus component values, cited, and tested
  against a direct transcription of the paper's formula. A run names a preset and overrides any component. *Built.*
- Two more sources of supervision: **distillation** from a teacher's logprobs, on-policy and off-policy (not black-box
  distillation from text alone), *built* as library code; and **LLM judges** as a source of rewards and preferences,
  *proposed*.
- One definition serves every training provider. Local LoRA (low-rank adaptation) and full-weight trainers compute it in
  torch. Tinker runs it through its custom-loss path, or through its built-in loss when the composition is one Tinker
  has built in. *Built.*

## Families

A family fixes what a batch item is and what the core term of the loss is. Everything else is a component.

| Family | Batch item | Core term | Needs | Status |
|---|---|---|---|---|
| `policy_gradient` | a segment with an advantage | the advantage times the policy's logprob of each sampled token, under the ratio and clipping components | behaviour logprobs when importance correction is on | built |
| `preference` | a pair (chosen, rejected) over a shared context, or a single example labelled desirable or undesirable | a function of the policy-to-reference log-likelihood ratio of each side | a reference, unless the loss is reference-free | built |
| `distillation` | a segment with the teacher's logprob of each sampled token, and for the top-k form its top-k tokens and logprobs there (`Distilled`) | a divergence between the teacher's and the policy's next-token distributions | a teacher channel that returns logprobs; for the top-k form, a trainer that gives the policy's logprobs of tokens it did not sample | built |
| `likelihood` | a segment with a weight | the weighted log-likelihood of its tokens (supervised fine-tuning, imitation) | nothing more | built |

A preference loss can add a likelihood term with a coefficient (`likelihood.coefficient`, as ORPO does), and a policy
gradient a distillation term (`distillation.coefficient`, its batch items then distilled segments with their episode's
advantage): both built.

## Components

Each component has a small set of values. A family accepts only the components that mean something for it, and
validation refuses the rest. Every component below is built but those marked proposed.

| Component | Values | Families |
|---|---|---|
| `advantage.baseline` | `group_mean`, `leave_one_out`, `none` | policy_gradient, likelihood |
| `advantage.scale` | `none`, `group_std` (the sample's standard deviation); `batch_std` proposed | policy_gradient, likelihood |
| `advantage.filter` | `none`, `equal_scores` (DAPO's dynamic sampling) | policy_gradient, likelihood |
| `advantage.tiebreak` | a number, 0 in every preset: what the shortest episodes of a group whose every episode saturated its task score more ([Minecraft rewards](minecraft-rewards.md)) | policy_gradient, likelihood |
| `ratio` | `token`, `segment` (the geometric mean of its tokens' ratios, as GSPO), `none` (the logprob itself, as REINFORCE) | policy_gradient |
| `clip.kind` | `none`, `ratio` (PPO), `weight` (clip the importance weight and stop its gradient, as CISPO), `dual` (with a lower bound for negative advantages) | policy_gradient |
| `clip.low`, `clip.high`, `clip.dual` | numbers; asymmetric for clip-higher; `dual` in times a negative advantage | policy_gradient |
| `importance.correction` | `none`, `untruncated`, `truncate` (TIS), `mask` (drop tokens outside the bounds, keeping the weight of those within; NeMo-RL's `icepop`) | policy_gradient, distillation |
| `importance.level` | `token`, `segment` | policy_gradient, distillation |
| `importance.cap`, `importance.floor` | numbers: the cap, and the lowest weight a mask keeps | policy_gradient, distillation |
| `kl.target` | `none`, `reference`, `old` | policy_gradient, distillation |
| `kl.estimator` | `k1`, `k2`, `k3` | policy_gradient, distillation |
| `kl.placement` | `loss`, `reward` (taken from each token's advantage, with no gradient, after the group's baseline) | policy_gradient; distillation (`reward` in the policy-gradient form) |
| `kl.coefficient` | a number | policy_gradient, distillation |
| `entropy.coefficient` | a number | policy_gradient |
| `aggregate` | `token_mean` (over the minibatch's tokens), `segment_mean` (over each segment's tokens, then segments), `segment_sum` (summed over each segment's tokens, then a mean over segments, as REINFORCE and RLOO), `constant` (summed and divided by a fixed token count, `constant_tokens`, as Dr. GRPO (group relative policy optimization)) | policy_gradient, likelihood, distillation; a preference loss is a mean over its items |
| `reference` | `base` (the adapter switched off, or a frozen copy for a full-weight trainer), `none`; `checkpoint` (a named one) proposed | preference, and policy_gradient or distillation with a KL to the reference |
| `preference.loss` | `sigmoid` (DPO), `hinge`, `square` (IPO), `margin` (SimPO), `odds_ratio` (ORPO), `kto` | preference |
| `preference.beta`, `preference.margin` | numbers (for `odds_ratio`, beta weighs the term, as TRL's ORPO) | preference |
| `preference.length_normalized` | true or false | preference |
| `preference.desirable`, `preference.undesirable` | KTO's weights | preference with `kto` |
| `likelihood.coefficient` | the chosen side's mean negative logprob beside the preference loss | preference |
| `distillation.divergence` | `reverse_kl`, `forward_kl`, `jsd` (GKD's, with its mixing weight `distillation.beta`) | distillation; policy_gradient's distillation term |
| `distillation.form` | `policy_gradient` (a per-token advantage from the teacher-student gap; reverse KL only), `top_k` (a divergence over the teacher's top-k tokens) | distillation; policy_gradient's term |
| `distillation.top_k` | how many of the teacher's most likely tokens each position carries (0: none) | distillation; policy_gradient's term |
| `distillation.temperature` | a number: for the renormalized `forward_kl` and `jsd` of the top-k form | distillation; policy_gradient's term |
| `distillation.advantage_clip` | a number: the policy-gradient form's advantage within ± it (0: unclipped) | distillation; policy_gradient's term |
| `distillation.beta` | the teacher's weight in the JSD's mixture, between 0 and 1 | distillation; policy_gradient's term |
| `distillation.teachers` | a table: the teacher channel of each route (an environment, one of its rows, or `*`); one teacher a sample, never an ensemble | distillation; policy_gradient's term |
| `distillation.coefficient` | a number: the distillation term beside a policy gradient (0: none); it changes between steps, but not to or from 0 | policy_gradient |

Where a component that follows from another is not given, it follows: a KL to the reference reads `reference = base`,
a preference loss with a reference reads it and one without reads none, and an odds ratio is length-normalized.

The step's existing controls stay as they are: `max_kl` (stop a pass when the policy has moved too far), the
gradient norm, the learning rate and warmup, `passes`, and `tokens_per_step`. They are trainer settings, not
components: every preset runs with the trainer's `max_kl` (0.02 by default), which a run sets to none for the papers'
unbounded steps.

## Presets

A preset is a family and component values, with the paper it comes from. Its test transcribes the paper's loss
directly on a fixed batch and compares the composed objective with it, in value and gradient
(`tests/rollout_objectives/test_presets.py`, which also pins each value to its source). *Built.* A distillation
preset's transcription reads the full distributions (the student's logits and the teacher's logprobs over the
vocabulary), and its composed loss only what a distilled segment carries and the step computes.

| Preset | Family | What sets it apart | Source |
|---|---|---|---|
| `default` | policy_gradient | Dr. GRPO's advantages (no standard deviation), DAPO's clip-higher (0.2, 0.28) and token mean, groups of equal scores skipped, truncated importance sampling at 2 | this platform's |
| `reinforce` | policy_gradient | no baseline, ratio, clipping or importance correction; each segment's logprob summed | Williams, 1992 (Eq. 11) |
| `rloo` | policy_gradient | leave-one-out baseline; each segment's logprob summed, the whole completion one action | Ahmadian et al., 2024 (Sec. 2.3) |
| `ppo_clip` | policy_gradient | token ratio clipped symmetrically at 0.2; advantages normalized within the group (no critic); token mean | Schulman et al., 2017 (Sec. 3, Table 1) |
| `grpo` | policy_gradient | group mean and standard deviation; token ratio clipped at 0.2; KL to the reference by k3 in the loss at 0.04; mean over each segment's tokens, then segments | Shao et al., 2024, DeepSeekMath (Eq. 3, 4; Sec. 4.2) |
| `dr_grpo` | policy_gradient | group mean without the standard deviation; summed over tokens and divided by 3,000 (their code's generation budget); clip 0.2; no KL | Liu et al., 2025 (Sec. 3.2, Listing 1, Table 6) |
| `dapo` | policy_gradient | clip-higher (0.2, 0.28); token mean; dynamic sampling; advantages normalized by the standard deviation; no KL | Yu et al., 2025 (Eq. 8, 9, 11; Sec. 4.1) |
| `gspo` | policy_gradient | segment ratio clipped at (3e-4, 4e-4); advantages normalized by the standard deviation; segment mean | Zheng et al., 2025 (Eq. 5 to 7; Sec. 5.1) |
| `cispo` | policy_gradient | the importance weight clipped above at 4, with its gradient stopped, and no lower bound; no update clipping; advantages normalized by the standard deviation; token mean | MiniMax, 2025, MiniMax-M1 (Eq. 4, 5) |
| `sft` | likelihood | the sampled tokens' log-likelihood, each segment weighted by its advantage (1 for a dataset's) | — |
| `dpo` | preference | sigmoid loss over pairs at beta 0.1, against the reference, sums over tokens | Rafailov et al., 2023 (Eq. 7, Appendix B) |
| `ipo` | preference | square loss at tau 0.1 of token-mean log ratios | Azar et al., 2023 (Eq. 17) |
| `simpo` | preference | length-normalized margin loss at beta 2.0, margin 1.0, no reference | Meng et al., 2024 (Eq. 6, Table 8) |
| `kto` | preference | unpaired desirable and undesirable examples at beta 0.1, both weights 1 | Ethayarajh et al., 2024 (Eq. 8, Sec. 4.2) |
| `orpo` | preference plus likelihood | an odds-ratio term at 0.1 beside the chosen side's likelihood, odds of the mean token logprob, no reference | Hong et al., 2024 (Eq. 3, 6, 7; Sec. 6.1) |
| `on_policy_distillation` | distillation | reverse KL on the student's own samples, from the teacher's logprob of each sampled token: the advantage `log T(y) - log pi_old(y)`, unclipped, its loss the advantage times the logprob; token mean; no importance correction | Agarwal et al., 2024, GKD (Sec. 3, on-policy with the reverse KL); Thinking Machines, 2025 (On-Policy Distillation) |
| `distillation` | distillation | forward KL to the teacher's top-20 logprobs, renormalized over them, on the teacher's samples; temperature 1; token mean | Hinton et al., 2015; Kim and Rush, 2016 (word-level KD) |
| `mopd` | distillation | the policy-gradient form: `A = clip(sg[log T(y) - log pi_old(y)], -5, 5)`, loss `-1/|y| sum_t A log pi(y)` (a mean over each segment's tokens, then segments); one rollout a prompt (the algorithm's group of 1); each domain routed to its teacher (`distillation.teachers`); no importance correction, no KL | Ma et al., 2026, MOPD: Multi-Teacher On-Policy Distillation (MiMo, ICML 2026; Eq. 3, 4; Sec. 4) |
| `mopd_top_k` | distillation | MOPD's top-k form: `1/|y| sum_t sum over the teacher's top-64 v of [p log(p/q) - p + q]`, the probabilities as they are | Ma et al., 2026 (Eq. 5; k = 64) |

Values the papers do not state, and where they come from instead:

- **GRPO's clip.** DeepSeekMath takes one update a step, so its clip never acts; 0.2, as its implementations (TRL).
- **The standard deviation** (GRPO, DAPO, GSPO, CISPO, `ppo_clip`) is the sample's, as TRL's and oat's are; the papers
  write `std` alone.
- **Dr. GRPO's constant** is "the maximum number of generation tokens"; its code passes 3,000. A run sets
  `objective.constant_tokens` to its own turn budget.
- **CISPO's upper bound** is tuned in MiniMax-M1 and not given; 4 is Tinker's default for its `cispo` loss, and among the
  bounds ScaleRL found equal (4, 5, 8).
- **IPO's tau** has no default in the paper (its experiments are bandits); 0.1 and token-mean log ratios are TRL's, which
  it says the authors confirmed.
- **SimPO's margin** is the paper's for Llama-3-Base (beta 2.0, gamma 1.0); the repository's later configurations give
  gamma as a ratio of beta (0.5 here).
- **KTO's reference point** `z0` is estimated in the paper from mismatched pairs of the microbatch (one example's prompt
  with another's answer), clamped at 0 and without a gradient. A multi-turn episode has no such mismatched pair, so
  here `z0` is the mean log ratio of the minibatch's own examples, clamped and without a gradient.
- **RLOO's KL.** The paper puts a KL penalty in the reward before the baseline (beta 0.03 to 0.10). `kl.placement =
  reward` takes it from each token's advantage after the baseline, which is not the same estimator, so `rloo` carries
  no KL; a verifiable reward needs none.
- **The student's logprob in a distillation advantage** is the step's start (`old`), as NeMo-RL's MOPD takes
  `prev_logprobs`; MOPD's paper writes `log pi_theta` under a stop-gradient, the same at the first update. Thinking
  Machines' cookbook takes the sampler's logprob and Tinker's `importance_sampling` loss (the ratio to the sampler);
  on-policy, sampler and start agree up to the engine's numerical gap, which `importance.correction` can weigh.
- **`distillation`'s top 20** is vLLM's default most logprobs a position; Hinton et al. and Kim and Rush match the whole
  vocabulary. The tail outside the top k is dropped and the rest renormalized; at k = the vocabulary the loss is theirs.
  Hinton's temperature (they report 20 for MNIST, and scale by its square) is a component (`distillation.temperature`);
  Kim and Rush use 1, the preset's.
- **MOPD's values** are the paper's for Qwen3-30B-A3B: A_max 5, k 64 for the top-k form, one rollout a prompt (against 8
  for its per-domain RL), temperature 1 for sampling, no dynamic sampling, domain ratios 0.35 : 0.35 : 0.3 (Math, IF,
  SWE) and a batch of 2,048 prompts. Its learning rate is not stated. The paper reports the two forms equal (Sec. 4.4.1).
  The ratios and the batch belong to a run's mixture of environments, not to the objective.
- **GKD's JSD** (`distillation.divergence = jsd`) is `beta·KL(T ‖ m) + (1 - beta)·KL(pi ‖ m)` with `m = beta·T + (1 -
  beta)·pi`, as TRL writes it; `beta` 0.5 by default, TRL's. It is a component of the top-k form, over the
  renormalized top k; no preset takes it.

Importance correction is a modifier any policy-gradient preset can take. `truncate` follows truncated importance
sampling (Yao et al., 2025; the cap of 2 is their code's) and `mask` follows masked importance sampling, which keeps
a weight within the bounds and drops the token outside them (Liu, Li et al., 2025), for the gap between the sampler and
the trainer. The papers' presets carry none, as the papers do; `default` truncates at 2. A distillation takes it too:
NeMo-RL's MOPD runs with `icepop`, which is `mask`.

The platform's present default is a preset of its own, `default`, with the `max_kl` stop. A trainer's own settings
that named the objective before (`objective = "policy_gradient"`, `ratio`, the clips, `truncate`) say `default` and its
components (`likelihood` says `sft`).

A run says `objective.preset = "dapo"` and may override components (`objective.kl.target = "reference"`,
`objective.kl.coefficient = 0.01`). Its start records the fully resolved objective, and a run started again trains
with what its start recorded, so a run's objective never depends on what a preset means later. Users' own combinations
are saved as run-settings presets (`NAME@N`), as today. Coefficients can change between steps; the family and the
structural components cannot. *Built.*

## Where it lives

- **The declaration**, `Objective` (family, components, presets, and which components each family accepts), lives in
  `rollout_train.objectives`, free of torch. The run settings and validation read it there. *Built*; the New run form
  reads the run settings' schema.
- **The torch implementation** is a small package of its own, `rollout_objectives`: the composed per-segment terms,
  the preference losses, aggregation, the step plan and the step statistics. The local LoRA and full-weight trainers
  and Tinker's custom-loss path all use it, and rollout-tinker depends on it, not on rollout-lora. *Built.*
- **Tinker's built-in losses** (`ppo`, `cispo`, `importance_sampling`, `cross_entropy`) are faster than a custom loss.
  When a resolved objective is one of them, the Tinker provider uses it, and tests check that both paths agree, to the
  float32 precision Tinker takes its inputs in. *Built.*
- **The algorithm** (`rollout_train.algorithm`) reads the advantage components and builds each family's batch items:
  weighted segments (`Grpo`), pairs or labelled examples (`Preferences`), segments with their teacher's scores
  (`Distillations`, and `Grpo` for a policy gradient's distillation term). Datasets of pairs and labelled examples come
  from `rollout_train.datasets` (`best-and-worst`, `above-and-below`), and datasets of teacher samples from the same
  rules, of `teacher` supervision. *Built.*
- **Teacher routing and scoring** (`rollout_train.distillation`): which teacher an episode goes to, the positions a
  segment is scored at, the teacher's `Scores` aligned with its sampled tokens, and `taught`, which asks a scorer and
  returns the segment carrying its scores. *Built*; the loop's call to it after each episode is not wired yet.
- **The reference** is the base with the adapter switched off for LoRA (a no-gradient pass), a frozen copy for the
  full-weight trainer only when asked (`trainer.frozen_reference`), and none on Tinker: its SDK offers prompt
  logprobs from a sampler of the base model, unconfirmed by a live test, so validation refuses an objective that
  reads one there. *Built.*

## Distillation

*Built* as library code, and [checked on Qwen3](#checked-on-qwen3); the loop does not yet ask teachers.

- **On-policy.** The student samples as it does for reinforcement learning. A teacher channel scores each sampled
  token with its own logprobs (prompt logprobs over the student's tokens, with the top k at each position where the
  objective reads them), and the loss is the reverse KL: from the sampled tokens alone (the policy-gradient form,
  `on_policy_distillation`, `mopd`), or over the teacher's top k (`mopd_top_k`). Scoring: `Engine.score` gives the
  logprobs of given tokens with the top k at each position, on `VllmEngine` (vLLM's prompt logprobs), `RemoteEngine`
  and engine hosts, and the gateway's `POST /v1/scores` asks a run's channel for them, recording each request as a
  turn that is never trained on ([scoring tokens](../libraries/rollout-train/gateway.md#scoring-tokens)).
  `rollout_train.distillation.taught` scores a segment (from its first sampled token to its last, within the teacher's
  context) and keeps the scores on it (`Segment.teacher`); `Distillations` makes the batch of distilled items.
- **Several teachers** (MOPD). `distillation.teachers` routes each episode, by its environment and row, to one teacher
  channel; each segment is scored by that one teacher, never an ensemble. A run over several environments with mixing
  ratios is the run's business; validation checks that the routes cover the environment a run plays.
- **Off-policy, with logprobs.** The teacher's own samples, each segment scored by the teacher with its top-k logprobs
  at each sampled position, are a dataset of `teacher` supervision (`rollout_train.datasets`), and the student fits
  them by forward KL over the top k (`distillation`). Sampling stays exact as far as it goes: the teacher's turns are
  recorded through the gateway like any other.
- **Mixing.** A policy gradient adds a distillation term with `distillation.coefficient`. Black-box distillation (from
  a teacher's text alone) is left out.
- **Validation** refuses a distillation with no teacher for a route, or no route for the environment played (routes
  for some of its rows only are a note); a teacher whose provider gives no prompt logprobs, fewer top logprobs than
  `distillation.top_k`, or logprobs not yet confirmed by a live test (Tinker's); a teacher of another renderer family
  than the student; and the top-k form on a trainer that gives the sampled tokens' logprobs only (Tinker).
- **Tinker** trains the policy-gradient form (alone, or as a policy gradient's term) through its custom loss: the
  teacher's scores are in the items, and the student's logprobs are the sampled tokens', which Tinker returns. The
  top-k form needs the student's logprobs of the teacher's top tokens, which Tinker's loss functions do not give, so it
  is refused there.

**The wiring left for runs.** After each episode, each trained segment is routed (`teacher_for`, by the run's
environment and the episode's row), scored by `taught` with a scorer that calls `Gateway.score` for its teacher's
channel, with `top = distillation.top_k` and the teacher's context (its `max_model_len` less the one token vLLM
generates), and kept in the episode's trajectories before the algorithm reads them. The teacher channels are fixed
channels of the run (`channels.NAME`), each bound for scoring like a judge.

## LLM judges

*Proposed.*

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

## Order of work

1. **The composable objective.** Build the declaration, the shared torch package, and the presets. Make the
   current behaviour reproduce exactly as `default`, `dapo` and `gspo`, and add `grpo`, `dr_grpo`, `rloo`, `cispo` and
   `reinforce`. Wire `objective.preset` and component overrides into the run settings and validation. *Built*
   (`default` reproduces the step to the last bit; `dapo` and `gspo` are the papers' presets, which differ from it in
   their advantages' scale and importance correction).
2. **The preference family**: `dpo`, `ipo`, `simpo`, `kto`, `orpo`, with pairs from a group's best and worst
   episodes and from datasets. Use the adapter-off reference for LoRA. *Built.*
3. **Judges**: judge channels, rubric scores and comparisons, and comparisons feeding preferences and group
   rankings.
4. **Distillation**: prompt logprobs in `VllmEngine` (scoring through the engines and the gateway), on-policy
   distillation, MOPD with routing by teacher, and off-policy distillation with teacher datasets. *Built* as library
   code; asking teachers after each episode is wired where runs are built.

## Checked on Qwen3-0.6B

On 2026-10-04, at commit `94146d7` (the code of this branch, run before it was rebased onto main's scoring and
untrained-slot changes), every preset took a short real run on one RTX 5080 (16 GB): `rollout train` over GSM8K
(`rollout_verifiers.environments:gsm8k`), sampled by local vLLM (`gpu_memory_utilization` 0.35, sleeping while the
trainer steps) and trained by the local LoRA trainer on the same card, with the scratch SQLite ledger and file blobs.
GSM8K because its reward is verifiable (`math-verify` compares the final number) and Qwen3-0.6B solves part of it:
57.5% of the 480 episodes, so most groups' outcomes vary.

Each run: 8 groups of 4 episodes, a step every 2 groups that have something to train on, 384 thinking and 256 answer
tokens a turn, LoRA rank 16 at a rate of 1e-4, `tokens_per_step` 1,024 (two to four updates a step), `max_kl` 0.5 (no
step stopped at it), gradients clipped to norm 1. Groups whose scores are all equal give nothing for a policy gradient
with a baseline and no pair; `reinforce` (no baseline) trains on every group. Per step, `/` between steps:

| Preset | Groups (trained) | Steps | Loss finite | Loss | KL moved | Mean ratio | Clip fraction | Truncated | Preference accuracy, margin | Step s (trainer; with sleep and wake) | Peak GPU GiB (trainer; card) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 8 (4) | 2 | yes | -0.0285 / 0.0117 | 0.014 / 0.017 | 0.9998 / 0.9976 | 0.029 / 0.041 | 0 / 0.0011 | — | 6.3 / 6.2 (14 / 13) | 1.67; 8.7 |
| `reinforce` | 8 (8) | 4 | yes | 166 / 215 / 122 / 186 | 0.0035 / 0.013 / 0 / 0 | 1 / 0.9997 / 1 / 1 | 0 | 0 | — | 3.5 / 4.8 / 3.2 / 2.8 (11 / 13 / 12 / 9) | 1.68; 8.8 |
| `rloo` | 8 (3) | 2 | yes | 2.9 / -6.12 | 0.029 / 0.0058 | 1.005 / 0.9991 | 0 | 0 | — | 5.3 / 3.2 (11 / 9.9) | 1.67; 9.0 |
| `ppo_clip` | 8 (6) | 3 | yes | 0.00963 / 0.013 / -0.0036 | 0.012 / 0.005 / 0.0059 | 0.9979 / 0.9988 / 0.9994 | 0.04 / 0.029 / 0.023 | 0 | — | 5.4 / 5.3 / 4.4 (14 / 14 / 10) | 1.67; 9.1 |
| `grpo` | 8 (5) | 3 | yes | 0.00511 / 0.00337 / 0.00123 | 0.017 / 0.0031 / 0.00044 | 1.002 / 1.003 / 1.001 | 0.054 / 0.026 / 0.011 | 0 | — | 6.2 / 5.4 / 3.7 (17 / 15 / 10) | 1.68; 9.3 |
| `dr_grpo` | 8 (5) | 3 | yes | 0.000715 / -0.00249 / -0.00816 | 0.011 / 0.0077 / 0.0055 | 1 / 1 / 0.9994 | 0.057 / 0.045 / 0.0066 | 0 | — | 4.5 / 4.5 / 3.3 (12 / 11 / 9.6) | 1.68; 9.0 |
| `dapo` | 8 (5) | 3 | yes | 0.0322 / -0.0111 / -0.00181 | 0.023 / 0.0074 / 0.0022 | 0.9989 / 1 / 0.9998 | 0.055 / 0.012 / 0.0055 | 0 | — | 4.9 / 4.5 / 2.8 (12 / 13 / 8.7) | 1.67; 8.7 |
| `gspo` | 8 (5) | 3 | yes | 0.0108 / 0.000374 / 0.00145 | 0.034 / 0.011 / 0.0086 | 0.9856 / 0.9964 / 0.9958 | **0.63 / 0.6 / 0.49** | 0 | — | 4.4 / 4.6 / 3 (12 / 13 / 9.3) | 1.67; 8.7 |
| `cispo` | 8 (7) | 4 | yes | -0.00225 / -0.0752 / -0.0744 / -0.0277 | 0.016 / 0.036 / 0.0052 / 0.021 | 0.9995 / 0.9917 / 0.9991 / 0.9949 | 0.00025 / 0 / 0.00026 / 0 | 0 | — | 4.6 / 4.7 / 4.6 / 2.7 (13 / 13 / 12 / 8.8) | 1.68; 8.8 |
| `sft` | 8 (6) | 3 | yes | -0.0169 / -0.00348 / 0.00121 | n/a | n/a | n/a | n/a | — | 4.8 / 4 / 4.4 (12 / 11 / 11) | 1.67; 8.7 |
| `dpo` | 8 (5) | 3 | yes | 0.761 / 0.434 / 1.44 | 0.0038 / 0 / 0 | n/a | n/a | n/a | 0.00, -1.27 / 1.00, 6.19 / 0.00, -11.7 | 2.9 / 2.8 / 1.8 (9.6 / 9.5 / 7.5) | 1.73; 8.7 |
| `ipo` | 8 (5) | 3 | yes | **25 / 25 / 25.1** | -0.0024 / 0 / 0 | n/a | n/a | n/a | 0.00, -0.0015 / 1.00, 0.0033 / 0.00, -0.0139 | 3 / 3 / 2.1 (10 / 10 / 8.1) | 1.67; 8.8 |
| `simpo` | 8 (7) | 4 | yes | 1.27 / 1.33 / 1.09 / 1.12 | 0.0011 / 0 / 0.0077 / 0 | n/a | n/a | n/a | 1.00, 0.028 / 0.50, -0.0076 / 1.00, 0.163 / 1.00, 0.139 | 2.7 / 2.6 / 2.8 / 1.7 (9.4 / 10 / 9.5 / 7.5) | 1.67; 8.8 |
| `kto` | 8 (6) | 3 | yes | 0.525 / 0.615 / 0.384 | 0.037 / 0.015 / 0.034 | n/a | n/a | n/a | 0.25, -1.06 / 0.38, -7.12 / 0.62, 5.93 | 5.4 / 7.2 / 6.3 (13 / 18 / 19) | 1.75; 8.7 |
| `orpo` | 8 (6) | 3 | yes | 0.493 / 0.504 / 0.594 | 0.0012 / 0 / 0.0042 | n/a | n/a | n/a | 0.50, -0.033 / 0.00, -0.019 / 0.50, 0.091 | 3.7 / 4.5 / 3.1 (13 / 15 / 9.2) | 1.67; 8.8 |

Nothing failed: every run played its groups, every loss was finite, and the policy moved in every run that measures
it. *KL moved* is KL(start ‖ now) as a step's last stepped minibatch found it; a step of one minibatch reads 0 there
by construction (a preference step of one or two pairs, a REINFORCE step whose groups are few), and such runs moved
between steps instead: `grpo`'s KL penalty to the reference grew from 0.011 to 0.016 nats a token, `dpo`'s chosen side
left the reference by 0.75 nats by its second step, and `simpo`'s and `orpo`'s margins rose. A likelihood reads no
`old`, so `sft` has none of the ratio's numbers (n/a). The trainer's peak is its process's on the card while the engine
sleeps; the card's is everything nvidia-smi saw (the desktop's 2.3 GiB and the engine's 5.6 GiB among it). The engine's
sleep and wake add 6 to 12 seconds to each step.

What looks wrong, and is reported rather than tuned away:

- **`gspo` clips half its tokens or more** (0.49 to 0.63). Its bounds hold a segment's mean log ratio within
  -3e-4 .. 4e-4 nats a token, and a rate of 1e-4 moves a segment past them in one update; the paper reports GSPO
  clipping far more tokens than GRPO, so the direction is the paper's, the size this rate's.
- **`ipo`'s loss sits at 25.** With tau 0.1 and token-mean log ratios its target margin is 1/(2·tau) = 5 nats a token,
  while three steps move the margin by thousandths: the loss is (h - 5)² ≈ 25 by the formula, not a fault, but the
  preset's tau (TRL's) asks far more than a few steps give.
- **Gradient norms before clipping** are 276 to 588 for `reinforce` and `rloo` (raw rewards times summed logprobs),
  22 to 110 for `dpo` (sums over hundreds of tokens), 13 to 22 for `ipo`, about 10 for `kto`, and 0.06 for `dr_grpo`
  (divided by 3,000): clipping at 1 makes the summed presets' updates normalized steps here, and `dr_grpo`'s small
  gradient is taken up by Adam.
- **`dpo`'s margins swing from -11.7 to 6.2** across steps: each step's pairs are new episodes, and a log ratio summed
  over a few hundred tokens moves by many nats under an adapter that has trained two steps. Accuracy on one or two
  fresh pairs a step says little.
- **`kl_moved` can be negative** (`ipo`, -0.0024): it is the k1 estimate on the sampled tokens, which is unbiased but
  not bounded below.

## Checked on Qwen3

On 2026-10-04, at commit `3d8f100`, the distillation presets took a few steps each on one RTX 5080 (16 GB), without
the run wiring: a script drove `PolicyStep` directly. The student, Qwen/Qwen3-0.6B with a fresh LoRA adapter (rank 16,
alpha 32), sampled answers to GSM8K training questions (chat template without thinking, temperature 1, at most 192
tokens, one sample a prompt, by transformers' `generate` on the trainer's own model); each sample was scored through
`rollout_train.distillation.taught` by a teacher served by `VllmEngine.score`; `PolicyStep` stepped on the distilled
items (learning rate 2e-4 unless said, `tokens_per_step` 512, two or three updates a step, `max_kl` none, gradients
clipped to 1). *Gap* is the mean, over sampled tokens, of the student's logprob less the teacher's: on the student's
own samples, an estimate of KL(student ‖ teacher) per token, in nats. Each step's gap is on that step's fresh samples
before its update (`teacher_gap`); the held-out gap is on fixed prompts the student never trained on, one sample each,
before the first step and after the last.

**One teacher**, Qwen/Qwen3-1.7B (`max_logprobs` 64). 8 prompts a step, 8 steps; held-out: 16 prompts.

| Preset | Held-out gap, before → after | Step gaps, `/` between steps | Advantage clipped | Top-k divergence, first → last step |
|---|---|---|---|---|
| `on_policy_distillation` | 0.730 → 0.529 | 0.585 / 0.362 / 0.515 / 0.395 / 0.456 / 0.468 / 0.464 / 0.351 | — | — |
| `mopd` | 0.730 → 0.503 | 0.586 / 0.383 / 0.510 / 0.544 / 0.560 / 0.467 / 0.540 / 0.653 | 3.0 to 4.7% | — |
| `mopd_top_k`, k = 20 | 0.730 → 0.408 | 0.586 / 0.259 / 0.388 / 0.318 / 0.432 / 0.479 / 0.307 / 0.397 | — | 0.511 → 0.350 |
| `mopd_top_k`, k = 64 | 0.730 → 0.444 | 0.586 / 0.354 / 0.350 / 0.326 / 0.432 / 0.365 / 0.567 / 0.342 | — | 0.560 → 0.305 |

Every preset lowered the held-out gap, by 0.20 to 0.32 nats a token; the top-k forms the most. A step took 17 to 19
seconds (sampling most of it), about 1,450 sampled tokens; no token went unscored. The trainer process's allocated peak
was 1.6 GiB in the first run and 2.8 GiB over the script; the card's peak was 12.0 GiB, the desktop's 3.4 GiB and the
teacher engine's 30% share among it.

**Two teachers**, LoRA adapters over Qwen/Qwen3-0.6B, each made by 8 supervised steps (`sft`, rate 3e-4, three passes
over 32 examples a step): *math* on GSM8K's reference solutions (calculator annotations and a final `#### N`), *caps*
on Dolly's answers written in capitals. One engine served both side by side (`max_loras` 2). Sampled by the engine,
the math teacher ended 7 of 8 held-out math answers with `####` (the base: none) and the caps teacher wrote 85% of the
letters of its caps answers in capitals (99% on math questions; the base: 6%). The student trained by `mopd` with
`distillation.teachers = {"check:gsm8k" = "math", "check:caps" = "caps"}`, each item routed by `teacher_for`: 4 GSM8K
and 4 Dolly prompts a step, rate 1e-4, 40 steps; held-out: 8 prompts of each task.

| Held-out gap | Before | After 10 steps | After 40 steps |
|---|---|---|---|
| math prompts, to the math teacher (its own task) | 0.589 | 0.386 | 0.312 |
| math prompts, to the caps teacher | 0.318 | 0.262 | 0.301 |
| caps prompts, to the caps teacher (its own task) | 0.569 | 0.518 | 0.252 |
| caps prompts, to the math teacher | 0.340 | 0.310 | 0.168 |

On each task the gap to that task's teacher fell the most: on math 0.28 nats against 0.02 to the other teacher, on caps
0.32 against 0.17. (The 10- and 40-step columns are separate runs from the same start.)

What looks wrong, and is reported rather than tuned away:

- **Routing moved the gaps, not the teachers' visible behaviour.** After 40 steps the student wrote 4.7% of its
  caps answers' letters in capitals (the base 4.4%, the caps teacher 85%) and ended no math answer with `####`. Reverse
  KL from the student's own samples fits the teacher where the student already writes; the teachers' distinct modes
  (capitals from the first token, GSM8K's annotations) are sequences the student almost never samples, so a few hundred
  samples do not reach them. The gap falls by matching word choice and length within the student's own modes.
- **A first two-teacher run diverged.** With teachers of 4 supervised steps each and a student rate of 2e-4, every gap
  rose from the fifth step (the step gap from 0.28 at the fourth to 1.24 and 2.11; held out, after 10 steps 0.71 to
  0.99 against 0.24 to 0.62 before) and the student's capitals fell to 1.8%. The runs above use teachers of 8 steps and a rate of 1e-4.
- **Answers got shorter.** In the 10-step run the mean sampled length fell from 153 to 40 to 105 tokens from the fourth
  step, its steps then one update each (`kl_moved` reads 0 by construction); in the 40-step run it moved between 66 and
  141. Both teachers' training answers are short, and `segment_mean` weighs each token of a short segment more.
- **Step gaps are noisy.** Each step's gap is on 8 new prompts, so it moves by ±0.1 from step to step: `mopd`'s last
  step read 0.653, above its first, while its held-out gap fell from 0.730 to 0.503. The held-out gaps are one sample
  of 16 (or 8) prompts each.
- **The policy-gradient form's loss is negative** (-0.94 to -0.16): it is the surrogate `-A log pi`, with `A` mostly
  negative where the student is more confident than the teacher; only its gradient means anything.
