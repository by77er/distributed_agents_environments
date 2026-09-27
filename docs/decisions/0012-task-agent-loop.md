# 0012 — Tasks, agents, and one framework-owned loop

Status: **Accepted** · Date: 2026-09-27

## Context

Authors need one convenient way to define what an agent works on: compute environments, tools, lifecycle,
scoring — written in Python, with environment operations like `workspace.put(...)` and `workspace.destroy()`. Two
earlier shapes were rejected in design discussion: an explicit step-list state machine with `goto` transitions (a
poor interface), and two separate patterns for tool-driven agents and environment-driven (multi-step) tasks.

## Decision

- A **Task** is the environment in the reinforcement-learning sense. It owns compute environments (as Python
  handles), defines tools (`@tool` methods; imported external tools; environment-provided tools), and implements
  fixed-name lifecycle hooks: `group_setup`, `setup`, `start`, `respond`, `score`, `teardown`.
- An **Agent** is the policy side: `select_context(history)` and `act(...)`, defaulting to one sample.
- **One loop**, owned by the framework: `start` → repeat {`agent.act` → `task.respond`} until an `Ending` →
  `score` → `teardown`. `respond` returns an `Observation` (messages, reward, ending, info). Its default executes
  the reply's tool calls; environment-driven tasks override it.
- Hook bodies are ordinary `async` Python with ordinary control flow; no step lists or transitions.
- Rewards: an observation's reward binds to the reply it answers; `run.reward(...)` assigns out-of-band and
  episode-level rewards. Endings are `TERMINATED` or `TRUNCATED`.
- Multiple model slots per task (e.g. a frozen simulated user); orchestration of several agents is composition of
  runs (`run.spawn`).

## Consequences

- Tool-driven, environment-driven and hybrid tasks share one interface; the framework understands each hook's
  meaning (fork group setup, attribute setup failures, attach scores, always tear down).
- The rollout controller's environment preparation and the separate verifier disappear into the task
  (`group_setup`, `score`).
- Tools defined by tasks run in the task host; the tool router shrinks to imported external tools.
- The agent — not the task — sees history and chooses context; tasks may only hint (`ContextHints`).
- Task code sees environments; agent code does not (P4 is scoped accordingly).

## Alternatives considered

- **Explicit state machine** (`@step` list + `goto`): durable at step boundaries without replay, but an awkward
  interface; replay makes it unnecessary.
- **Free-form `run` method as the main extension point**: maximal flexibility, but two patterns (agent loop vs
  hand-written loop) and no framework-understood phases.
- **Pure step functions** (ADR-0004's original authoring model): correct, but authors must write state machines.

## Amendment ([0014](0014-no-forks-template-recipes.md))

The `group_setup` hook is removed. Setup cost is paid by content-addressed environment templates; the hooks are
`setup`, `start`, `respond`, `score`, `teardown`.
