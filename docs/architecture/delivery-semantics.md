# Delivery semantics

Status: **Proposed** · See [ADR-0018](../decisions/0018-effects-at-least-once.md) · Defines what effects and messages
guarantee under failure. Component documents MUST NOT weaken these; they MAY strengthen them.

## Summary

| Mechanism | Guarantee |
|---|---|
| Deterministic effect identity | Every re-execution of an effect carries the same `effect_id` and argument digest |
| Durable steps (durable runner) | Every effect a run requested is performed at least once; its first recorded result is final |
| Receiver deduplication | Duplicate executions of one `effect_id` happen at most once where the receiver deduplicates |
| Argument digests | A reused `effect_id` with different arguments is rejected, never answered from a cache |
| Attempt markers | Where the receiver cannot deduplicate, a possible duplicate is reported as `OUTCOME_UNKNOWN`, never silently retried |
| Ownership fencing | A deposed executor cannot record results; at most the steps it had started are duplicated |

Together: **effectively-once** for deduplicating receivers, **at-most-once-or-flagged** for the rest. The
`LocalRunner` gives none of this: a crash loses in-flight runs.

## Receiver deduplication

Every receiver that can keys on `(effect_id, arguments_digest)`:

| State of `effect_id` | Same digest | Different digest |
|---|---|---|
| absent | execute; record *in progress* | — |
| in progress | join the running execution and return its result | reject (`CONFLICT`) |
| done | return the recorded result | reject (`CONFLICT`) |

- Retention must cover the longest recovery window (default 24 h).
- Receivers that implement this: the recorder, environment services (required of the environment system), tool
  bindings of kind `agent` and `human`, and message delivery (by `message_id`).
- Third-party HTTP and MCP tools receive `effect_id` (`Idempotency-Key`, `_meta.idempotency_key`) and may or may not
  honor it; their `retry_class` decides.

## Retry classes

| Class | After a crash, the effect is… |
|---|---|
| `PURE`, `IDEMPOTENT` | re-executed |
| `SIDE_EFFECTING` with a deduplicating receiver | re-executed with the same identity |
| `SIDE_EFFECTING` / `UNKNOWN` without | guarded by an attempt marker: `OUTCOME_UNKNOWN` if the marker exists |

Precedence (lowest → highest): tool annotations from trusted servers → binding default → operator configuration →
`RunBinding` override.

## Why digests and zero-data-loss replication

`effect_id` is derived from position. If a database failover lost an acknowledged record, replay could reach the same
position with *different* arguments (a different message arrived first, for example), and a receiver would return the
cached result of the old request. So: the durable runner's database uses synchronous replication with zero data loss
on failover, and receivers compare argument digests.

## Messages

- Delivery into a conversation's or run's mailbox is idempotent by `message_id`; order is first-in, first-out per
  (sender, conversation).
- Delivering to an idle conversation and scheduling its next activation happen in one transaction, so no message is
  delivered without a consumer being scheduled, and parking re-checks the mailbox so none is stranded.

## Deadlines and cancellation

- Every effect carries an absolute deadline; on expiry the runner records a failure (`DEADLINE`) and sends a
  best-effort cancel. A late completion is dropped: the first recorded result wins.
- Run cancellation is cooperative: `run.cancel_requested` is delivered at the next turn boundary and `teardown`
  runs. Hard cancellation skips cleanup, so a reaper destroys what the run still owns.
