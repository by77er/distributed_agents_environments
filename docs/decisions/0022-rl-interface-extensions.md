# 0022 — RL interface extensions

Status: **Proposed**; routing replay for mixture-of-experts is **Accepted** · Date: 2026-09-27 · Extends [0015](0015-rollout-interface.md)

## Context

A survey of RL environment specifications and training frameworks ([research](../research/interfaces-rl-environments.md))
confirmed the core RL design and found data and control gaps. Mixture-of-experts training is a requirement.

## Decision

- **`Sample`** gains turn spans, routed experts, `outcome`, `truncation_reason`, `mask_reasons`
  (`run.exclude_from_training`), media references, an optional transcript, `root_run_id`, and optional
  enrichments.
- **Routing replay** (accepted): engines must return routed experts for every position, including prefix-cache hits;
  the recorder stores them per span; trainers for mixture-of-experts policies must support routing replay.
- **Rollout API** gains cancellation by run or label, per-ticket sample streams, gang admission
  (`admit_together`), several trainable channels per job, and `publish(channel, WeightsSource, phase)` with staged
  transfer (`STAGE` / `COMMIT`) and in-process, checkpoint, delta, LoRA and distributed (trainer-participating)
  sources.
- **Swarm rewards**: child runs inherit the job; a run may reward its descendants; samples are assembled when the
  root ends.
- **`score_tokens`** (prefill-only logprobs) on the engine adapter, for teacher and reference logprobs.

## Consequences

- Trainer adapters grow slightly; the rollout side still knows no algorithm.
- Distributed weight transfer couples trainer and inference placement (same region and zone).

## Deferred

The first trainer to integrate; samples from runs outside rollout jobs.
