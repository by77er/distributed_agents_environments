# Determinism

Code: `rollout.local.context` · See [effects](contracts/effects.md)

Task, agent and program code is ordinary `async` Python, and a runner runs it once. What it reaches outside itself
goes through `run`, as effects, each with an identity. This page says what code sees of time, randomness and effects,
and the rules that keep a run's effects identified and its draws reproducible.

## What code sees

| Operation | What it gives |
|---|---|
| An effect: a model sample, an imported tool call, an environment or sandbox operation, `run.emit` | its result; the effect is recorded as run events ([effects](contracts/effects.md)) |
| A wait for a message, and the messages merged at a turn boundary | the messages, in the order they were delivered ([conversations](../../guide/conversations.md)) |
| `run.now()` | the wall clock, in UTC |
| `run.random` | a `random.Random` seeded from `run_id`: a run with the same id draws the same sequence |
| `await run.gather(*awaitables)` | awaits concurrently and returns results in argument order, as `asyncio.gather` does |
| Everything else: task hooks, `@tool` bodies, `__init__` of the task and the agent, `run.blobs.put`, `run.reward` | runs as written; none of it is an effect |

## Effect identity

The k-th effect a run requests is `{run_id}:0:{k}` ([effects](contracts/effects.md#identity)). Receivers use it to
perform an effect once ([receivers that deduplicate](contracts/effects.md#receivers-that-deduplicate)): `Model.sample`
retries a failing endpoint under one `effect_id`, and the gateway answers a sample asked for again under an
`effect_id` it has recorded with the turn it recorded.

## Rules

| Rule | Why |
|---|---|
| Reach outside the code only through `run` and the handles it gives: models, tools, environments, sandboxes, `emit` | Only effects are recorded and identified. Other input and output leaves no trace in the run's events. |
| Use `run.now()` and `run.random`, never the wall clock, `uuid4`, `os.urandom` or the `random` module | A run's draws then follow from its id, and its times are the ones its events carry. |
| Request effects in an order that does not depend on timing | Effects are numbered in the order they are requested. Concurrent branches that request one effect each, as the default `respond` does for a reply's tool calls, are numbered in the order the branches start. |
| Put work that must be recorded in an effect, not in a task hook or a `@tool` body | Task hooks and tool bodies are not effects: what they do is not in the run's events. |
