# Determinism rules for durable code

Status: **Proposed** · Layer: core · See [ADR-0013](../../decisions/0013-replay-durability.md), [ADR-0020](../../decisions/0020-deterministic-event-loop.md)

Task, agent and program code is ordinary `async` Python. Under the `LocalRunner` it simply runs. Under a durable
runner it must also survive crashes and suspend for days without holding compute. CPython cannot serialize a
suspended coroutine, so a durable runner resumes code by **replay**: it re-runs the code and feeds back the recorded
results of the operations it already performed. This document states what code must do for replay to be correct.
How a runner implements replay is in [durability/task-host.md](../../durability/task-host.md).

## What is recorded

Every operation that reaches outside the code is an **effect** whose result is recorded:

- model samples (`run.model.sample`), imported tool calls, environment operations;
- messages (`run.send`, `WaitFor` / `resume`), child runs (`run.spawn`), sleeps and timeouts;
- `run.now()` (the time of the latest recorded input) and `run.random` (seeded from `run_id`).

## Effect identity

Within one run generation, the code's effects form a deterministic sequence. The *k*-th effect requested has

```
effect_id = {run_id}:{generation}:{k}
```

The identifier is the same on every re-execution, so receivers that deduplicate on it (the recorder, environment
services, well-behaved tools) perform each effect at most once. Every effect also carries a digest of its arguments;
a receiver that sees a known `effect_id` with a different digest rejects the request
([effects](../../contracts/effects.md)).

## Rules

| Rule | Why |
|---|---|
| No I/O except through `run` and handles | replay must see the same inputs |
| No wall clock or ambient randomness (`time.time`, `uuid4`, `os.urandom`, `random`) — use `run.now()`, `run.random` | same |
| No threads; concurrency only through `asyncio` tasks on the run's loop (`run.gather`, `asyncio.gather`, `create_task`) | the runner's event loop orders tasks deterministically |
| No ordering that depends on `hash()` or `id()` (e.g. iterating a `set` of strings) | differs between processes |
| Code changes that alter the effects, observations or rewards of existing runs use `run.patched(change_id)` or wait for the next generation | replay must reproduce what was recorded |

Durable runners enforce these rules where they can (a sandbox with no network; a deterministic event loop that
rejects real I/O and patches clocks and randomness) and detect the rest.

## Divergence detection

On replay, each requested effect is compared with the recorded one (kind + argument digest), and each recorded
observation and reward with the replayed one (digest). A mismatch fails the run with `NON_DETERMINISM` and
quarantines it, rather than continuing on a history it did not produce.

## Generations

A long-running run hands over to a new **generation** at a resumable point (after a hook or a turn) when its replay
budget (turns, recorded bytes, wall time) is exceeded or a deploy asks it to drain. The new generation starts from
exported task and agent state, so replay cost and recorded history stay bounded. Conversations get the same effect
from `WaitFor` and `End(continue_as=…)`: each episode is its own run.

## Versioning

- A run is pinned to the code references it started with. A new generation, and a conversation's next run, may
  move to the deployment's current version.
- `run.patched(change_id) -> bool` returns `True` for executions that reach it on new code and `False` while
  replaying history recorded before the change. Under the `LocalRunner` it always returns `True`.
