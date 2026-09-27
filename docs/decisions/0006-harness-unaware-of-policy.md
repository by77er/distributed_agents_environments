# 0006 — The harness is unaware of the model and policy changes

Status: **Accepted** · Date: 2026-09-26

## Context

The policy must be changeable mid-rollout (R5): weights versions, sampling parameters, or the model itself.
Making the harness resolve policies per turn would put tokenization, versions and provenance into every harness
and into the run log.

## Decision

The harness depends only on the [model endpoint contract](../contracts/model-endpoint.md): canonical content in,
canonical content out, plus a capability contract and a usage signal. Policy identity, versions, sampling
parameters, tokens and engines are all below that contract. Swaps are invisible exactly when they preserve the
channel's capability contract; contract-breaking changes require a different channel (a visible RunBinding change).

## Consequences

- Harnesses are simple and reusable across RL, evals and production.
- The run log holds canonical content only; RL data lives in the recorder.
- Prompts cannot adapt to the specific model (quality concern, not correctness); per-policy prompt adaptation, if
  needed, belongs in the recorder's renderer.
- Debugging behavior changes requires joining run logs with recorder data (by `effect_id` / `session_id`).

## Alternatives considered

- **Harness-visible policy events** (`policy.changed` in the run log, continuation handles, provenance blobs):
  functional, but couples every harness to RL concerns. Superseded during design.
