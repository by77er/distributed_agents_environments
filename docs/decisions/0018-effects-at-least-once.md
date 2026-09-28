# 0018 — Effects are at-least-once; receivers deduplicate with argument digests

Status: **Accepted** · Date: 2026-09-27 · Supersedes [0003](0003-log-is-the-outbox.md)

## Context

Every durable substrate evaluated records an effect's result after performing it (or can redeliver it), so a crash
can re-execute an effect. Separately, deriving `effect_id` from position means a database failover that lost an
acknowledged record could reuse an identifier for a *different* request.

## Decision

- `effect_id = {run_id}:{generation}:{ordinal}`, deterministic across re-executions, plus an **argument digest**.
- Receivers deduplicate with three states (absent → execute; in progress → join; done → recorded result) and
  **reject a known `effect_id` with a different digest** (`CONFLICT`).
- Tools whose receivers cannot deduplicate are guarded by an **attempt marker** committed before dispatch; a
  re-execution that finds one reports `OUTCOME_UNKNOWN`.
- The durable runner's database uses synchronous replication with zero data loss on failover.

## Consequences

- P5 is restated from "persist before effect" to "at-least-once, made effectively-once by receivers".
- Receiver deduplication becomes mandatory for the recorder and the environment system.
- `OUTCOME_UNKNOWN` is a normal, model-visible result for non-deduplicating tools after a crash.

## Alternatives considered

- **Persist before dispatch** (ADR-0003): not available on the chosen substrate without a second journal.
