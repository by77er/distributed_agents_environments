# Determinism

Status: **Working** (2026-10-02) · Code: `rollout.durable.context`, `rollout.core.local.context`

Task, agent and program code is ordinary `async` Python. Under the `LocalRunner` it runs once. Under the
`DurableRunner` it can run several times, because the runner resumes a run by **replay**. This page says what replay
is and what code must do for it to be correct.

## Replay

A durable run is a DBOS workflow. Every effect is a recorded step, and every message the run takes from its inbox is
a recorded receive. To resume a run, the runner runs its program again from the start: it creates the task and the
agent again from the run's specification, and each step and receive that already happened returns its recorded
result instead of being performed. The program takes the same path to the point where it stopped and continues from
there. DBOS matches recorded results to steps by their order in the run.

A run is replayed when:

- its runner starts again after a crash or a shutdown;
- another runner on the same database takes it over, after the first runner's heartbeat stopped;
- it was unloaded from memory while it waited, and a message, the wait's timeout or a cancellation wakes it
  ([evicting idle runs](../../durability/eviction.md)).

Replay records the run's events again with the same `seq` and content; the store keeps the first copy of each.

## What code sees on replay

| Operation | On replay |
|---|---|
| An effect that completed: a model sample, an imported tool call, an environment operation, `run.emit` | returns the recorded result; nothing is performed |
| An effect that a crash interrupted | is performed again with the same `effect_id`; a guarded one raises `OutcomeUnknown` instead |
| A wait for a message, and the messages merged at a turn boundary | the same messages, in the same order |
| `run.now()` | the same time: that of the latest recorded input (the run's start, the last effect's completion, the last message's sending, or the end of a wait that timed out) |
| `run.random` | the same sequence: a `random.Random` seeded from `run_id` |
| `await run.gather(*awaitables)` | awaits concurrently and returns results in argument order, as `asyncio.gather` does |
| `run.patched(change_id)` | `True`, as on the first execution |
| Everything else: hooks, `@tool` bodies, `__init__` of the task and the agent, `run.blobs.put`, `run.reward` | runs again |

Guarded effects are commands in an environment (`Environment.execute`) and calls to side-effecting imported tools
whose tool set does not deduplicate ([tools](../../guide/tools.md#after-a-crash)).

Under the `LocalRunner` nothing is replayed: `run.now()` is the wall clock in UTC, and `run.random`, `run.gather`
and `run.patched` behave as above.

## Effect identity

The k-th effect a run requests has `effect_id = {run_id}:0:{k}`, and carries a digest of its arguments
([identifiers](../../guide/runs-and-events.md#effects-and-their-identity)). Replay requests the same effects in the
same order, so every identifier is the same on every execution. Receivers use it to perform an effect once: the
recorder returns the recorded reply for an `effect_id` it has seen, and an environment's id is derived from the
`effect_id` of its creation, so a repeated creation finds the same environment.

## Rules

| Rule | Why |
|---|---|
| Reach outside the code only through `run` and the handles it gives: models, tools, environments, `emit` | Only effects are recorded. Other input and output happens again on replay and can give a different answer. |
| Use `run.now()` and `run.random`, never the wall clock, `uuid4`, `os.urandom` or the `random` module | Replay must take the same path. |
| Keep `__init__` of tasks and agents deterministic, and keep state on `self` | Replay creates both again and rebuilds their state by running the hooks. |
| Request effects in an order that does not depend on timing | Effects are numbered in the order they are requested. Concurrent branches that request one effect each, as the default `respond` does for a reply's tool calls, are numbered in the order the branches start. |
| Do not let ordering depend on `hash()` or `id()`, for example by iterating a `set` of strings | They differ between processes, and replay can happen in another one. |
| Put work that must happen once in an effect, not in a hook or a `@tool` body | Hooks and tool bodies run again. |
| Change code under live runs only in ways that keep the effects each run already requested, and their order | Replay runs the code the runner has loaded, against the results recorded for the old code. |
