# 0013 — Durability of task and agent code by deterministic replay

Status: **Proposed** · Date: 2026-09-27

## Context

Task and agent code must survive worker and host crashes, suspend for hours cheaply, and scale to thousands of runs
per process. The initial idea was to pickle class members between calls. CPython cannot pickle suspended
generators or coroutines, so resuming in the middle of a method requires re-running it.

## Decision

- Task and agent code is `async` Python driven like a generator by a task host: each awaited handle call yields an
  `Effect` to the driver; no asyncio event loop is involved.
- **Durability comes from the log.** Effects are committed before dispatch and their completions after. Recovery
  replays the code, resolving the *k*-th requested effect from the *k*-th committed request; divergence fails the
  run (`NON_DETERMINISM`).
- **Snapshots are disposable optimizations**: pickled task and agent state at resumable points (after each hook and
  turn) bounds replay and enables forking. `@checkpoint` memoizes units of work inside hooks.
- **Determinism is enforced**, not requested: task hosts run in a no-network sandbox; time and randomness come
  from `run.now()` and `run.random`; foreign awaitables are rejected; hash seeds are fixed.
- Runs are pinned to code versions; long runs cross deploys with `run.patched(change_id)`.
- Group setup is forked by pickling the task and snapshotting its environments.

## Consequences

- Authors write straight-line code with loops, branches, and `try` / `finally`; nothing else to learn except the
  determinism rules.
- One worker hosts thousands of runs; suspended runs cost nothing and can be evicted.
- Replay is CPU-only and fast; snapshots keep it bounded.
- The sandbox needed for determinism is also the security boundary for tenant-authored task code.
- Costs: determinism rules can be violated in subtle ways (caught by divergence detection); pickle is fragile
  across code changes (mitigated by pinning and by snapshots being disposable).

## Alternatives considered

- **Pickle state at every await, no replay**: impossible without frames; would require every step to contain at
  most one effect.
- **Thread per run with blocking calls**: simplest code, but cannot host hundreds of thousands of mostly-idle runs
  or evict them while suspended.
- **Temporal / Restate SDKs**: the same replay model, but each brings its own server and history model that
  duplicates our run log.
- **Explicit state machines**: see ADR-0012.

## Amendment ([0014](0014-no-forks-template-recipes.md))

Group forks are removed. Snapshots never move between runs; they only bound replay.
