# 0009 — Reuse stale KV across weight updates; correct with importance sampling

Status: **Accepted** · Date: 2026-09-26

## Context

Recomputing KV for every active context after each weight update costs ~9B prefill tokens at the design point,
which would cap update frequency. PipelineRL reports that retaining stale KV for mixed-policy sequences adds only
slightly more divergence than recomputing.

## Decision

- Keep the prefix cache across weight updates. Tokens sampled over stale KV come from a different but fully
  specified conditional distribution μ(yₜ | prefix); the recorded behavior logprob is exactly that distribution.
- Correct for all off-policyness (stale weights, stale KV, in-flight updates, inference/training numeric
  mismatch) with importance sampling against recorded behavior logprobs (P10). The estimator is the trainer's
  choice.
- Bound KV age per channel (`max_kv_age` versions): shared prefixes would otherwise stay hot, and stale, forever.
  Initially enforced by staggered whole-cache flushes; version-tagged eviction is a desirable upstream patch.
- KV provenance is recorded best-effort as diagnostics only; correctness never depends on it (engines evict and
  recompute silently).
- Trainable channels require logprobs of the processed distribution (e.g. vLLM `processed_logprobs`) and restrict
  sampling transforms the learner cannot reproduce.
- A feedback loop (IS diagnostics by weight lag × KV age) lets the rollout controller tune `max_kv_age` and
  throttling to hold a target effective sample size.

## Consequences

- Weight updates can happen every trainer step with ~no re-prefill cost.
- Off-policyness is measured and controlled, not guessed.
- Correctness hinges on honest logprobs — enforced by the recorder, not by convention.

## Alternatives considered

- **Always recompute KV** (`max_kv_age = 0`): cleanest data, prohibitive at scale; remains available as a setting.
- **Infrequent updates with fresh KV**: trades stale KV for stale weights, usually worse.

## Amendment ([0015](0015-rollout-interface.md))

The adaptive feedback loop (importance-sampling diagnostics tuning `max_kv_age` and throttling) is deferred.
`max_kv_age` is an operator setting per channel; staleness is bounded by the rollout buffer and `max_turns`, and
trainers mask outliers by per-token `weights_version`. Temperature-only sampling on trainable channels is decided.
