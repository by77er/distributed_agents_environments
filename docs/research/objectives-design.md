# Objectives design: families, components, presets, distillation and judges

Status: the composable objective and the preference family are built (steps 1 and 2 of the [order of
work](#order-of-work)): the declaration is `rollout_train.objectives`, the torch implementation `rollout_objectives`
([objectives in torch](../implementations/rollout-objectives.md), [objectives](../libraries/rollout-train/training.md#objectives)).
Judges and distillation are proposed, but for teacher scoring, which is built. Each section says which of its parts are built.

## Goals

- An objective is chosen by its **family**, the primary selector, and composed from **components**. These are
  orthogonal settings (the advantage, the ratio, clipping, importance correction, KL, entropy, aggregation, the
  reference), so a combination is a configuration rather than another hard-coded loss. *Built.*
- The objectives of the literature ship as **presets**: each one a family plus component values, cited, and tested
  against a direct transcription of the paper's formula. A run names a preset and overrides any component. *Built.*
- Two more sources of supervision: **distillation** from a teacher's logprobs, on-policy and off-policy (not black-box
  distillation from text alone), and **LLM judges** as a source of rewards and preferences. *Proposed.*
- One definition serves every training provider. Local LoRA and full-weight trainers compute it in torch. Tinker runs
  it through its custom-loss path, or through its built-in loss when the composition is one Tinker has built in.
  *Built.*

## Families

A family fixes what a batch item is and what the core term of the loss is. Everything else is a component.

| Family | Batch item | Core term | Needs | Status |
|---|---|---|---|---|
| `policy_gradient` | a segment with an advantage | the advantage times the policy's logprob of each sampled token, under the ratio and clipping components | behaviour logprobs when importance correction is on | built |
| `preference` | a pair (chosen, rejected) over a shared context, or a single example labelled desirable or undesirable | a function of the policy-to-reference log-likelihood ratio of each side | a reference, unless the loss is reference-free | built |
| `distillation` | a segment with the teacher's logprobs for its tokens | a divergence between the teacher's and the policy's next-token distributions | a teacher channel that returns logprobs | proposed |
| `likelihood` | a segment with a weight | the weighted log-likelihood of its tokens (supervised fine-tuning, imitation) | nothing more | built |

A preference loss can add a likelihood term with a coefficient (`likelihood.coefficient`, as ORPO does): built. A
policy gradient plus a distillation term is proposed, with distillation.

## Components

Each component has a small set of values. A family accepts only the components that mean something for it, and
validation refuses the rest. Every component below is built but those marked proposed.

| Component | Values | Families |
|---|---|---|
| `advantage.baseline` | `group_mean`, `leave_one_out`, `none` | policy_gradient, likelihood |
| `advantage.scale` | `none`, `group_std` (the sample's standard deviation); `batch_std` proposed | policy_gradient, likelihood |
| `advantage.filter` | `none`, `equal_scores` (DAPO's dynamic sampling) | policy_gradient, likelihood |
| `ratio` | `token`, `segment` (the geometric mean of its tokens' ratios, as GSPO), `none` (the logprob itself, as REINFORCE) | policy_gradient |
| `clip.kind` | `none`, `ratio` (PPO), `weight` (clip the importance weight and stop its gradient, as CISPO), `dual` (with a lower bound for negative advantages) | policy_gradient |
| `clip.low`, `clip.high`, `clip.dual` | numbers; asymmetric for clip-higher; `dual` in times a negative advantage | policy_gradient |
| `importance.correction` | `none`, `untruncated`, `truncate` (TIS), `mask` (drop tokens outside the bounds, keeping the weight of those within) | policy_gradient; distillation proposed |
| `importance.level` | `token`, `segment` | policy_gradient; distillation proposed |
| `importance.cap`, `importance.floor` | numbers: the cap, and the lowest weight a mask keeps | policy_gradient; distillation proposed |
| `kl.target` | `none`, `reference`, `old` | policy_gradient; distillation proposed |
| `kl.estimator` | `k1`, `k2`, `k3` | policy_gradient; distillation proposed |
| `kl.placement` | `loss`, `reward` (taken from each token's advantage, with no gradient, after the group's baseline) | policy_gradient |
| `kl.coefficient` | a number | policy_gradient; distillation proposed |
| `entropy.coefficient` | a number | policy_gradient |
| `aggregate` | `token_mean` (over the minibatch's tokens), `segment_mean` (over each segment's tokens, then segments), `segment_sum` (summed over each segment's tokens, then a mean over segments, as REINFORCE and RLOO), `constant` (summed and divided by a fixed token count, `constant_tokens`, as Dr. GRPO) | policy_gradient, likelihood; a preference loss is a mean over its items |
| `reference` | `base` (the adapter switched off, or a frozen copy for a full-weight trainer), `none`; `checkpoint` (a named one) proposed | preference, and policy_gradient with a KL to the reference; distillation proposed |
| `preference.loss` | `sigmoid` (DPO), `hinge`, `square` (IPO), `margin` (SimPO), `odds_ratio` (ORPO), `kto` | preference |
| `preference.beta`, `preference.margin` | numbers (for `odds_ratio`, beta weighs the term, as TRL's ORPO) | preference |
| `preference.length_normalized` | true or false | preference |
| `preference.desirable`, `preference.undesirable` | KTO's weights | preference with `kto` |
| `likelihood.coefficient` | the chosen side's mean negative logprob beside the preference loss | preference |
| `distillation.divergence` | `reverse_kl`, `forward_kl`, `jsd` (with its mixing weight) | distillation, proposed |
| `distillation.top_k` | how many of the teacher's logprobs each position carries | distillation, proposed |
| `distillation.temperature` | a number | distillation, proposed |

Where a component that follows from another is not given, it follows: a KL to the reference reads `reference = base`,
a preference loss with a reference reads it and one without reads none, and an odds ratio is length-normalized.

The step's existing controls stay as they are: `max_kl` (stop a pass when the policy has moved too far), the
gradient norm, the learning rate and warmup, `passes`, and `tokens_per_step`. They are trainer settings, not
components: every preset runs with the trainer's `max_kl` (0.02 by default), which a run sets to none for the papers'
unbounded steps.

## Presets

A preset is a family and component values, with the paper it comes from. Its test transcribes the paper's loss
directly on a fixed batch and compares the composed objective with it, in value and gradient
(`tests/rollout_objectives/test_presets.py`, which also pins each value to its source). *Built*, but the distillation
presets.

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
| `on_policy_distillation` | distillation | reverse KL on the student's own samples, scored by the teacher | Agarwal et al., 2024 (GKD); Thinking Machines, 2025. *Proposed* |
| `distillation` | distillation | forward KL to the teacher's top-k logprobs on the teacher's samples | Hinton et al., 2015; Kim and Rush, 2016. *Proposed* |

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

Importance correction is a modifier any policy-gradient preset can take. `truncate` follows truncated importance
sampling (Yao et al., 2025; the cap of 2 is their code's) and `mask` follows masked importance sampling, which keeps
a weight within the bounds and drops the token outside them (Liu, Li et al., 2025), for the gap between the sampler and
the trainer. The papers' presets carry none, as the papers do; `default` truncates at 2.

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
  weighted segments (`Grpo`), pairs or labelled examples (`Preferences`); segments with teacher logprobs are proposed.
  Datasets of pairs and labelled examples come from `rollout_train.datasets` (`best-and-worst`, `above-and-below`).
  *Built.*
- **The reference** is the base with the adapter switched off for LoRA (a no-gradient pass), a frozen copy for the
  full-weight trainer only when asked (`trainer.frozen_reference`), and none on Tinker: its SDK offers prompt
  logprobs from a sampler of the base model, unconfirmed by a live test, so validation refuses an objective that
  reads one there. *Built.*

## Distillation

*Proposed*, but teacher scoring, which is built.

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
4. **Distillation**: prompt logprobs in `VllmEngine` (built: scoring through the engines and the gateway), then
   on-policy distillation, then off-policy distillation with teacher datasets.
