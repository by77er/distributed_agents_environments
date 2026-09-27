# 0004 — Harness is a pure step function hosted by a durable actor runtime

Status: **Proposed**, authoring model superseded by [0012](0012-task-agent-loop.md) and [0013](0013-replay-durability.md) · Date: 2026-09-26

## Context

Runs must be durable (R1) and hosted by the hundreds of thousands (N1). Most of a run's life is spent waiting on
model samples (seconds) and tool calls. A harness that performs its own model calls holds a slot while waiting.

## Decision

Model the harness as `step(state, input) → events` plus `apply(state, event) → state`, with no I/O. Model
samples and tool calls are effects performed by the runtime after commit. The runtime is a durable actor host:
partition leases, per-run serial mailboxes, commit-then-dispatch. Build it rather than adopt a general workflow
engine.

## Consequences

- Workers never block; thousands of runs per worker.
- Replay equivalence by construction: live and replay paths both use `apply`.
- Harness code cannot corrupt the log (runtime validates emitted events).
- Harness authors write in an event-sourced style — an SDK must make this pleasant.
- Harnesses that cannot be expressed this way run as unmanaged runs (trainable, not durable).

## Alternatives considered

- **Temporal / Restate / DBOS**: solve general workflows; we have one fixed workflow shape whose log schema is our
  product (read by RL tooling). Temporal's history limits and per-step overhead also conflict with long agent runs
  and short RL episodes.
- **Harness calls the model directly, runtime checkpoints**: simpler to write harnesses, but blocks worker slots
  and makes durability depend on harness discipline.

## Amendment ([0012](0012-task-agent-loop.md), [0013](0013-replay-durability.md))

The runtime half of this decision stands: partition leases, serial mailboxes, commit-then-dispatch, and a
`HarnessHost` boundary. The authoring model changes: instead of explicit `step` / `apply` functions, task and agent
code is ordinary `async` Python run under deterministic replay by a task host, which implements `HarnessHost`
(`Step` advances code speculatively; a failed commit evicts and replays).
