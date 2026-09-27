# Harness

Status: **Proposed** · Boundary B4 · See [ADR-0012](../../decisions/0012-task-agent-loop.md), [ADR-0013](../../decisions/0013-replay-durability.md), [ADR-0006](../../decisions/0006-harness-unaware-of-policy.md)

## Purpose

The harness is the **framework-owned rollout loop**. It drives a **Task** — the environment the agent acts in:
compute environments, tools, lifecycle, responses to each model turn, scoring — with an **Agent** — the policy
side: what the model sees and how it acts. Task and agent code are written in Python as ordinary `async` code and
run in a **task host** process, which implements the runtime's `HarnessHost` protocol by driving that code under
deterministic replay.

| Document | Defines |
|---|---|
| this document | the loop, `RunSpecification`, the `HarnessHost` protocol (B4) |
| [task.md](task.md) | the Task authoring interface: hooks, `Observation`, tools, environments, `RunContext` |
| [agent.md](agent.md) | the Agent interface: context selection, acting |
| [durability.md](durability.md) | replay driver, checkpoints, snapshots, determinism rules, versioning |

There is exactly **one** interaction pattern. Tool-driven tasks, environment-driven (multi-step) tasks and hybrids
differ only in how `Task.respond` answers a model turn.

## The loop (normative)

```python
async def rollout(task: Task, agent: Agent, run: RunContext) -> None:
    await task.setup(run)
    try:
        observation = await task.start(run)
        run.history.record_start(observation)
        while observation.end is None:
            if task.max_turns is not None and run.turn >= task.max_turns:
                observation = End(truncated=True)
                run.history.record_end(observation)
                break
            reply = await agent.act(run, run.history, task.tools_for_turn(run))
            observation = await task.respond(run, reply)        # validated; see task.md
            run.history.record(reply, observation)              # observation.reward binds to `reply`
        episode_reward = await task.score(run)
        if episode_reward is not None:
            run.reward(episode_reward)
    finally:
        await task.teardown(run)
```

- `setup` runs before anything else; `teardown` runs whenever `setup` began, on success, failure and cancellation.
- `score` runs only when the loop ends with an `Ending` (not when a hook raised).
- Every `(reply, observation)` pair is committed as a `model.completed` + `observation.recorded` event pair; the
  history the agent sees is derived from the log, never pickled.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Sequencing hooks; turn limits; history | Persistence, dispatch, retries (Runtime) |
| Binding observations and rewards to replies | Tokens, rendering, policy versions (Recorder, inference) |
| Validating observations (roles, tool-result completeness) | Environment lifecycle implementation (Environment Manager) |
| Translating `await` on handles into effects | What the task and agent decide |
| Replay, checkpoints, snapshots, divergence detection | |

## Boundaries

| Direction | Counterpart | What |
|---|---|---|
| provides → | Runtime (B4) | `HarnessHost` protocol |
| consumes | nothing directly | every interaction with the outside world is an effect handed to the runtime (P3) |

## Interface: `HarnessHost` (B4)

```proto
service HarnessHost {
  rpc Load(LoadRequest)   returns (Empty);              // {run_id, code_reference, snapshot?, events[]}: replay to the end of events
  rpc Step(StepRequest)   returns (StepResult);         // {run_id, input: RunEvent} → events to commit
  rpc Evict(RunReference) returns (Empty);              // discard all state for the run
  rpc Snapshot(RunReference) returns (SnapshotBlob);    // pickled task + agent state and driver position (at resumable points)
}

message StepResult {
  repeated RunEvent events = 1;   // only task-emittable types; seq/epoch assigned by the runtime
}
```

- `Step` resumes the run's suspended coroutines with the input and runs them until every one is blocked on an
  effect or finished. It returns the resulting events (effect requests, observations, rewards, checkpoints,
  terminal events).
- `Step` **advances host state speculatively**. If the runtime's append fails (`FENCED`, `SEQ_CONFLICT`), the
  runtime MUST call `Evict`; the next input triggers `Load`, which replays from the log.
- `Load` replays: effects the code requests are matched against committed `*.requested` events in order and
  resolved with their committed completions. Effects still pending in the log are left pending (the runtime
  re-dispatches them). A mismatch is a [non-determinism error](durability.md#divergence-detection).

**Inputs** passed to `Step`: `run.created`, `model.completed`, `model.failed`, `environment.completed`,
`environment.failed`, `tool.completed`, `tool.failed`, `tool.outcome_unknown`, `child.completed`,
`signal.received`, `timer.fired`, `run.cancel_requested`.

**Task-emittable events**: `model.requested`, `environment.requested`, `tool.requested`, `spawn.requested`,
`message.sent`, `timer.requested`, `observation.recorded`, `reward.assigned`, `checkpoint.completed`,
`tools.changed`, `patch.marked`, `run.suspended`, `run.completed`, `run.failed`. Anything else is rejected.
See [run-events](../../contracts/run-events.md).

## RunSpecification

What a client submits to create a run. Portable parts (task, agent) are separate from deployment-specific parts
(binding).

```proto
message RunSpecification {
  TaskReference  task    = 1;
  AgentReference agent   = 2;
  RunBinding     binding = 3;
}

message TaskReference  { string code_reference = 1; string class_name = 2; Json parameters = 3; }     // parameters: e.g. a dataset row
message AgentReference { string code_reference = 1; string class_name = 2; Json configuration = 3; }

message RunBinding {
  map<string, ModelBinding>    models           = 1;  // model slot declared by the task → endpoint
  map<string, ToolBinding>     imports          = 2;  // import name declared by the task → external tool binding (tool-router)
  map<string, SecurityProfile> security_classes = 3;  // security class → concrete profile (grants, allowlists); default: operator profile
  EnvironmentPlacement environment_placement    = 4;  // optional driver / zone preferences
  Durability durability                         = 5;  // DURABLE | BEST_EFFORT
}

message ModelBinding {
  oneof binding {
    Recorded recorded = 1;   // {channel, mode: ACTIVE | PASSIVE, sampling: SamplingParameters} → recorder session
    Direct   direct   = 2;   // {provider, model, sampling: SamplingParameters}                → direct adapter
  }
}

```

`code_reference` is `{package}@{content_hash}`; task and agent may come from the same package. A run is pinned to
the code references it was created with (see [versioning](durability.md#versioning)).

## Failure modes

| Failure | Result |
|---|---|
| Unhandled exception in a task hook | `run.failed{TASK_ERROR}` (after `teardown`) |
| Exception inside a `@tool` body | returned to the model as a `tool_result` with `is_error = true` |
| Invalid observation (roles, missing tool results) | `run.failed{INVALID_OBSERVATION}` |
| Replay divergence | `run.failed{NON_DETERMINISM}`; the run is quarantined for inspection |
| Task host crash | runtime evicts the host's runs and reloads them elsewhere |

## Scale envelope

One task host process per core; each hosts hundreds to thousands of suspended runs. `Step` SHOULD complete in
≤ 5 ms p99 excluding task and agent compute. Task host memory per suspended run: coroutine frames + task and agent
attributes (history lives in the log and is materialized lazily).

## Build vs buy

Build the Python SDK and task host. Design references: Temporal's Python workflow sandbox and deterministic
event loop; DBOS / Restate step memoization; Inspect AI's task / solver / scorer split; multi-turn environment
libraries (e.g. verifiers).

## Open questions

- Context shape: agent-owned with task hints (current design), or may a task require a shape?
- Should `score` also run after a hook raised, to score partial work?
- Branching partway through an episode (tree search, Monte Carlo value estimates) is not supported
  ([ADR-0014](../../decisions/0014-no-forks-template-recipes.md)); add it only when an algorithm needs it.
