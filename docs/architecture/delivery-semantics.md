# Delivery semantics

Status: **Proposed** · Defines exactly what each boundary guarantees under failure. Component docs MUST NOT
weaken these; they MAY strengthen them.

## Summary

| Mechanism | Guarantee |
|---|---|
| Fenced append | At most one worker advances a run at a time; a deposed worker cannot commit |
| Persist-then-dispatch | No effect exists that the log does not know was requested (durable runs) |
| Run log as outbox | Every committed `*.requested` event is dispatched at least once |
| Receiver idempotency | Duplicate dispatches of one `effect_id` execute at most once *if the receiver can dedupe* |
| Retry classes | When a receiver cannot dedupe, recovery is explicit (`outcome_unknown`), never a silent retry |

Together: **effectively-once** for dedupe-capable receivers, **at-most-once-or-flagged** for the rest.

## Fencing

- Leases are held per **partition**, not per run. A lease row holds `(partition, owner, epoch, expires_at)`.
- Acquiring a lease increments `epoch`. Every append includes the caller's epoch; the store checks it **in the
  same transaction** as the insert and rejects on mismatch (`FENCED`).
- Every append also carries `expected_seq`; a mismatch is `SEQ_CONFLICT` (indicates a bug or a race during
  takeover — the worker MUST evict and reload the run).
- A worker that receives `FENCED` for any run in a partition MUST drop the entire partition immediately.

## The run log is the outbox

A committed `*.requested` event *is* the outbox entry. There is no separate outbox table on the hot path.

```
commit [input, …, X.requested]   fenced
  ok     → dispatch X            eager, in-process
  FENCED → drop; never dispatch
takeover → pending = requested − completed − failed   (derived by replay)
         → re-dispatch each pending effect per retry_class
```

### Three delivery patterns

| Pattern | Effects | Mechanism |
|---|---|---|
| **Eager dispatch** | `model.request`, `tool.request` | Owner dispatches after commit; recovery scan on takeover |
| **Same-transaction write** | intra-cell `send`, `spawn`, timer creation | Written into the target's inbox / `timers` table in the same transaction. Only a best-effort nudge remains |
| **Relayed outbox** | cross-cell `send`/`spawn`, due timers, outbound webhooks | A relay polls with `SKIP LOCKED` and delivers; used where no shared transaction exists or dispatch is deferred |

## Receiver idempotency

Every effect receiver keys on `effect_id` and implements three states:

| State | Behavior on receipt |
|---|---|
| absent | execute; record *in progress* |
| **in progress** | **join** the running execution and return its result |
| done | return the cached result |

- Retention MUST cover the maximum takeover window: lease TTL + max effect deadline + margin. Default: 24 h.
- Receivers that implement this: **envd**, **recorder**, **Environment Manager**, **tool router agent/human bindings**,
  **Run Store** (inbox/timer writes are keyed).
- Receivers that may not: third-party `http` and `mcp` bindings. The tool router forwards `effect_id` (header or
  `_meta.idempotency_key`) and relies on `retry_class`.

## Retry classes

Every tool has a `retry_class`, resolved at tool-resolution time and pinned in `tools.resolved`:

| Class | On pending-at-takeover | On transport error |
|---|---|---|
| `pure` | re-dispatch | retry |
| `idempotent` | re-dispatch | retry |
| `side_effecting` + receiver dedupes | re-dispatch same key | retry same key |
| `side_effecting` / `unknown`, no dedupe | append `tool.outcome_unknown` | append `tool.outcome_unknown` |

Precedence (lowest → highest): MCP annotation hints (only from trusted servers) → binding default → operator
provider config → RunBinding override.

`model.request` is always `idempotent`: the recorder dedupes; direct provider adapters may re-sample, which costs
tokens but not correctness (non-trainable path).

## Completion routing

| Effect duration | Route |
|---|---|
| Short (≤ deadline the worker can await; default 15 min) | Owner awaits the call; completion enters the mailbox in-process |
| Long (human policy, long tools, child runs) | Receiver acknowledges; completion is later written to the run's **inbox** (keyed by `effect_id`) and the owner is nudged |

Late or duplicate completions are dropped by dedupe on `effect_id` against the log.

## Deadlines, timeouts, cancellation

- Every effect carries an absolute `deadline`. On expiry the owner appends `*.failed{class: DEADLINE}` and
  issues a best-effort `Cancel(effect_id)` to the receiver.
- A completion arriving after a `failed{DEADLINE}` is dropped (dedupe) — the log's first terminal event wins.
- Run cancellation: `run.cancel_requested` input → harness step decides wind-down (MAY be immediate) → runtime
  cancels all pending effects → `run.cancelled`.

## Run durability tiers

| | `durable` | `best_effort` |
|---|---|---|
| Append timing | before every dispatch | buffered; flushed periodically and at completion |
| Dispatch | after commit | immediately |
| Worker crash | resume from log | run marked `crashed`; owner (e.g. rollout controller) resamples |
| Use | coding agents, swarms, long runs | short RL rollouts |
| Hazard | write load | crash-correlated sampling bias (long episodes crash more) — monitor `crashed` rate by episode length |
