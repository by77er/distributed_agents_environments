# Harness

Status: **Proposed** · Layer: core · See [ADR-0012](../../decisions/0012-task-agent-loop.md), [ADR-0006](../../decisions/0006-harness-unaware-of-policy.md), [ADR-0019](../../decisions/0019-conversations-and-priority-delivery.md)

## Purpose

The harness is the **framework-owned rollout loop**. It drives a **Task** — the environment the agent acts in, in
the reinforcement-learning sense: tools, lifecycle, responses to each model turn, scoring — with an **Agent** — the
policy side: what the model sees and how it acts. Task and agent code are ordinary `async` Python.

A **`Runner`** executes runs. The core ships `LocalRunner` (in-process, no persistence). The
[durability layer](../../durability/README.md) provides `DurableRunner`, which executes the same code so that it
survives crashes.

| Document | Defines |
|---|---|
| this document | the loop, `Program`, `RunSpecification`, deployments, `Runner` |
| [task.md](task.md) | `Task`, `Observation`, `WaitFor`, tools, `RunContext` |
| [agent.md](agent.md) | `Agent` |
| [conversations.md](conversations.md) | conversations, messages, priorities, delivery modes |
| [determinism.md](determinism.md) | rules for code that runs under a durable runner |

There is exactly **one** interaction pattern. Tool-driven tasks, environment-driven (multi-step) tasks, single-turn
tasks and conversations differ only in how `Task.start`, `Task.respond` and `Task.resume` answer.

## The loop (normative)

```python
async def rollout(task: Task, agent: Agent, run: RunContext) -> None:
    await task.setup(run)
    try:
        observation = await task.start(run)
        run.history.record_start(observation)
        while True:
            if isinstance(observation, WaitFor):
                envelope = await run.wait_for_message(observation)            # suspends; see conversations.md
                observation = (await task.resume(run, envelope)) if envelope else observation.on_timeout
                run.history.record_resume(envelope, observation)
                continue
            if observation.end is not None:
                break
            if task.max_turns is not None and run.turn >= task.max_turns:
                run.history.record_end(End(truncated=True))
                break
            try:
                reply = await agent.act(run, run.history, task.tools_for_turn(run))
            except Interrupted as interruption:                                    # a message with mode INTERRUPT
                observation = await task.resume(run, interruption.envelope)
                run.history.record_interrupted(interruption, observation)
                continue
            observation = await task.respond(run, reply)                           # validated; see task.md
            if isinstance(observation, Observation):
                observation = await task.steer(run, run.take_steering_messages(), observation)
            run.history.record(reply, observation)                                 # observation.reward binds to reply
        episode_reward = await task.score(run)
        if episode_reward is not None:
            run.reward(episode_reward)
    finally:
        await task.teardown(run)
```

- `setup` runs first; `teardown` runs whenever `setup` began (success, failure, cancellation).
- `score` runs only when the loop ends with an `Ending`.
- A `WaitFor` suspends the run until a message of the requested kind arrives or the timeout passes
  ([conversations](conversations.md)). Under a durable runner, a suspended run holds no compute.
- Messages delivered with mode `STEER` are merged into the next observation by `Task.steer` (default: appended as
  USER content). Mode `INTERRUPT` during a model sample cancels it (`agent.act` raises `Interrupted`) and the loop
  calls `Task.resume` with the message; during tool execution the tools finish and the message is merged like
  `STEER`. Messages delivered with mode `QUEUE` wait for the next `WaitFor`.
- A task with no tools and one turn (`start` → one reply → `respond` returns `End`) is a complete task. No
  environment is involved unless the task creates one.

## Program

The loop is one `Program`. Plain durable workflows are others.

```python
class Program:
    async def main(self, run: RunContext) -> None: ...

class AgentProgram(Program):                        # the loop above
    def __init__(self, task: Task, agent: Agent): ...
    async def main(self, run: RunContext) -> None:
        await rollout(self.task, self.agent, run)
```

A `Program` that hosts a third-party agent framework (e.g. an OpenAI Agents SDK agent whose model calls go through
`run.model`) is how foreign loops run on the platform; see [research: interoperability](../../research/interfaces-agent-frameworks.md#8-interoperability-adapters).

## RunSpecification and deployments

```python
@dataclass(frozen=True)
class RunSpecification:
    program: ProgramReference            # {code_reference, class_name, parameters} — e.g. AgentProgram(task, agent)
    binding: RunBinding

@dataclass(frozen=True)
class RunBinding:
    models: Mapping[str, ModelBinding]             # model slot → recorded channel or direct provider
    imports: Mapping[str, ToolBinding] = field(default_factory=dict)   # import name → external tool binding
    environments: EnvironmentBinding | None = None # opaque to the core; interpreted by the environment layer
    delivery: DeliveryPolicy = DeliveryPolicy()    # priority → delivery mode (conversations.md)

@dataclass(frozen=True)
class ModelBinding:
    recorded: RecordedModel | None = None          # {channel, sampling: SamplingParameters} → recorder
    direct: DirectModel | None = None              # {provider, model, sampling}           → direct adapter

@dataclass(frozen=True)
class Deployment:                                   # a named, addressable agent
    name: str                                       # e.g. "acme/support-bot"
    specification: RunSpecification                # program + default binding
```

`code_reference` is `{package}@{content_hash}`. A run is pinned to the code references it started with; a
conversation's next run after a `WaitFor` timeout or `End(continue_as=…)` takes the deployment's current version.

## Runner

```python
class Runner(Protocol):
    async def start(self, specification: RunSpecification, *, run_id: str | None = None,
                    conversation: ConversationKey | None = None, labels: Mapping[str, str] = {}) -> RunHandle: ...
    async def send(self, to: Address, envelope: Envelope, *, priority: Priority = Priority.NORMAL,
                   idempotency_key: str | None = None) -> None: ...       # starts the conversation's run if none is live
    async def cancel(self, run_id: str, *, reason: str) -> None: ...

class RunHandle(Protocol):
    run_id: str
    async def result(self) -> RunOutcome: ...
    def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]: ...
```

| Implementation | Layer | Behaviour |
|---|---|---|
| `LocalRunner` | core | Runs the program on the current asyncio loop. Nothing persists; a process crash loses in-flight runs. Waits are in memory. |
| `DurableRunner` | [durability](../../durability/README.md) | Runs the program in a sandboxed task host driven by a DBOS workflow; effects are durable steps; runs resume after crashes; suspended runs hold no compute. |

## Failure modes

| Failure | Result |
|---|---|
| Unhandled exception in a task hook | `run.failed{TASK_ERROR}` (after `teardown`) |
| Exception inside a `@tool` body | returned to the model as a `tool_result` with `is_error = true` |
| Invalid observation (roles, missing tool results) | `run.failed{INVALID_OBSERVATION}` |
| Process crash | `LocalRunner`: runs are lost. `DurableRunner`: runs resume ([durability](../../durability/README.md)) |

## Open questions

- Should `score` also run after a hook raised, to score partial work?
- Branching partway through an episode (tree search, Monte Carlo value estimates) is not supported
  ([ADR-0014](../../decisions/0014-no-forks-template-recipes.md)); add it only when an algorithm needs it.
