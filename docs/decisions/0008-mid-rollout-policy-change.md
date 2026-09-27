# 0008 — Policy may change mid-rollout; split generations by abort-and-resubmit

Status: **Accepted** (requirement R5) · mechanism **Proposed** · Date: 2026-09-26

## Context

Long agentic rollouts would otherwise accumulate unbounded staleness; weight updates must reach in-flight runs,
including mid-generation. No engine we surveyed reports which weights version produced each token.

## Decision

- Runs reference **channels**; the Policy Registry moves channels as versions are published and ready.
- The Weight Update Controller guarantees **abort-before-update**: pause admission, abort in-flight requests,
  load weights, resume.
- The recorder receives aborted partials, records them as a span at the old version, and resubmits
  `input + partial` tokens-in at the new version. Each engine response is therefore single-version, and per-token
  version spans are exact without engine patches.
- The harness sees one ordinary message.

## Consequences

- Mid-generation switching is engine-agnostic, requiring only tokens-in and abort-with-partials.
- Staleness can be bounded per token rather than per trajectory.
- Each abort costs a resubmission; with stale-KV reuse (ADR-0009) the prefix is a cache hit.

## Alternatives considered

- **Turn-boundary switching only**: simpler, but staleness bounded by turn length (minutes for reasoning turns).
- **Engine `pause(keep)` without abort**: cheaper, but the recorder cannot see where versions changed unless the
  engine reports it (upstream patch).
