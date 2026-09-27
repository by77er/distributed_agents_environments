# 0003 — The run log is the outbox; eager dispatch

Status: **Proposed** · Date: 2026-09-26

## Context

Effects (tool calls, model samples) must never happen without the log knowing they were requested, and must be
delivered at least once despite worker crashes. The classic transactional outbox adds a relay that polls a table.

## Decision

A committed `*.requested` event **is** the outbox entry. The owning worker dispatches it eagerly right after its
own commit succeeds; the log is only read back for recovery on takeover. Intra-cell `send`/`spawn`/timers are
written in the same transaction as the step. A relayed outbox is used only where no shared transaction exists
(cross-cell) or dispatch is deferred (due timers, webhooks). Receivers dedupe on `effect_id` with three states
(absent / in progress → join / done).

## Consequences

- No polling latency or relay load on the hot path.
- A deposed worker cannot dispatch: it cannot commit, and dispatch requires commit.
- A worker that committed just before losing its lease may dispatch a duplicate; receivers absorb it.
- Pending effects are derived by replay; a `pending_effects` index table can be added later for cross-run ops
  queries at the cost of one write per effect.

## Alternatives considered

- **Polling relay for all effects**: adds latency (poll interval) to every half-turn and load proportional to
  effect rate.
- **Dispatch before persist**: loses track of side effects on crash (outcome unknown on every crash).
