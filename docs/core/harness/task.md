# Task

Status: **Proposed** · Layer: core · See [ADR-0012](../../decisions/0012-task-agent-loop.md), [ADR-0019](../../decisions/0019-conversations-and-priority-delivery.md)

A **Task** is the environment an agent acts in, in the reinforcement-learning sense: it defines the tools (the
action space), produces the first observation, responds to every model turn, and scores the episode. It is a Python
class whose hooks are ordinary `async` methods with ordinary control flow. A task may use compute environments
([environments](../../environments/README.md)), but many need none.

## Interface

```python
class Task:
    # Declarations (class attributes)
    models: ClassVar[dict[str, ModelSlot]] = {"policy": ModelSlot()}   # the agent acts through "policy"
    imports: ClassVar[list[str]] = []                                  # external tool sets, bound per run
    max_turns: ClassVar[int | None] = None                             # loop truncates after this many turns
    context_hints: ClassVar[ContextHints] = ContextHints()             # advisory for the agent

    def __init__(self, parameters: Any) -> None: ...                   # once per run

    # Lifecycle hooks
    async def setup(self, run: RunContext) -> None: ...                # optional; once per run
    async def start(self, run: RunContext) -> Observation | WaitFor: ...           # REQUIRED
    async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor:
        return await self.run_tools(run, reply)                        # default: execute tool calls
    async def resume(self, run: RunContext, envelope: Envelope) -> Observation | WaitFor:
        return Observation(envelope.content)                           # default: the message becomes the observation
    async def steer(self, run: RunContext, envelopes: list[Envelope],
                    observation: Observation) -> Observation: ...      # default: append as USER content
    async def score(self, run: RunContext) -> float | None: ...        # optional; episode-level reward
    async def teardown(self, run: RunContext) -> None: ...             # optional; always runs; idempotent

    # Tools
    def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]: ...   # default: all declared tools
    async def run_tools(self, run: RunContext, reply: Message) -> Observation: ...
        # executes every tool call in `reply` concurrently and returns one TOOL message with a tool_result per
        # call; returns End() if `reply` contains no tool calls
```

| Hook | Required | What the framework does with it |
|---|---|---|
| `setup` | no | Timed and attributed separately; failures are infrastructure errors, not agent failures |
| `start` | **yes** | Opens the episode, or waits for the first message (`WaitFor`) |
| `respond` | no | The environment's step; its observation's reward is bound to the reply it answers |
| `resume` | no | Turns a message that satisfied a `WaitFor`, or interrupted a turn, into the next observation |
| `steer` | no | Merges messages delivered with mode `STEER` into the next observation |
| `score` | no | Episode-level reward, attached to the end of the trajectory |
| `teardown` | no | Always runs if `setup` began |

## Observation, Ending, WaitFor

```python
@dataclass(frozen=True)
class Observation:
    messages: list[Message]                                # shown to the model next
    reward: float | None = None                            # bound to the reply this observation answers
    end: Ending | None = None                              # None → the episode continues
    info: dict[str, Any] = field(default_factory=dict)     # logged; never shown to the model

class Ending(Enum):
    TERMINATED = "terminated"   # a real end state (value methods do not bootstrap)
    TRUNCATED = "truncated"     # stopped by a limit: turns, time, budget (value methods may bootstrap)

def End(reward: float | None = None, *, truncated: bool = False, info: dict[str, Any] | None = None,
        continue_as: ContinueAs | None = None) -> Observation: ...
    # continue_as: end this run and start the conversation's next run with explicit carried state

@dataclass(frozen=True)
class WaitFor:
    kind: str = "message"                                   # envelope kind to wait for
    timeout: timedelta | None = None
    on_timeout: Observation = field(default_factory=lambda: End(truncated=True))
```

`Observation("text")` builds one USER text message; `Observation(message)` and `Observation([messages])` take
canonical `Message`s.

| Field | Consumed by | Becomes |
|---|---|---|
| `messages` | the agent, then the model | a context span in the recorder's session tree (loss mask 0) |
| `reward` | the trajectory assembler | a reward at the last sampled token of the reply it answers |
| `end` | the loop and the trainer | the sample's `ending` |
| `info` | people, metrics | the run's events only |

**Validation** (violations fail the run with `INVALID_OBSERVATION`):

1. `messages` contains only USER and TOOL roles. The environment cannot write the policy's turns or the system
   prompt (the agent owns it).
2. If the reply contained tool calls, `respond` returns an `Observation` with a `tool_result` for every `call_id` —
   including when it also ends the episode. `WaitFor` is valid from `start`, from `resume`, and from `respond` after
   a reply without tool calls.
3. A non-terminal observation has at least one message.
4. Blocks larger than 64 KiB are moved to blob storage automatically.

Rewards for other model slots (an opponent, a simulated user) are assigned with `run.reward(value, slot=...)`.
A task can mark a run as unsuitable for training (e.g. an infrastructure fault that is not the policy's fault) with
`run.exclude_from_training(reason)`.

## Tools

```python
class Calculator(Task):
    @tool
    async def evaluate(self, expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return str(safe_evaluate(expression))
```

- `@tool` methods are collected by the base class. The `ToolSpecification` comes from the signature (type hints →
  JSON Schema) and the docstring; `@tool(name=..., retry_class=..., timeout=...)` overrides defaults.
- An exception raised in a tool body becomes a `tool_result` with `is_error = true` and the exception message; the
  model can react to it.
- **Imported tools** (`imports = ["github"]`) are external tool sets — MCP servers, HTTP services, other agents,
  humans — bound per run in `RunBinding.imports` and executed by a `ToolBinding` (in process, or by the platform's
  [tool router](../../platform/tool-router/README.md)). Anything that keeps state across runs — memory, knowledge
  bases, user profiles — is a tool.
- The tool set is resolved and pinned at run start (`tools.resolved`); later changes are explicit `tools.changed`
  events. `tools_for_turn` may expose any subset of the declared tools on a given turn.

## Environments (optional)

A task that needs a computer creates one through `run.environments`. The environment system is designed separately
([environments](../../environments/README.md)); the core only sees opaque handles whose methods are effects:

```python
class Environments(Protocol):                       # run.environments; absent when no environment layer is bound
    async def create(self, specification: EnvironmentSpecification) -> Environment: ...

class Environment(Protocol):                        # a handle; serializes as a reference
    environment_id: str
    async def execute(self, command: str | list[str], **options: Any) -> ExecutionResult: ...
    async def put(self, source: bytes | BlobReference | Asset, destination: str, **options: Any) -> None: ...
    async def get(self, path: str) -> bytes: ...
    async def destroy(self) -> None: ...
```

Environments created by a run are owned by it (P12); `teardown` should destroy them, and runners destroy any the run
still owns when it ends.

## RunContext

```python
class RunContext:
    run_id: str
    conversation: ConversationKey | None             # set when the run serves a conversation (conversations.md)
    turn: int                                         # completed model turns so far
    history: History                                  # read-only: observations and replies
    models: Mapping[str, Model]                       # the task's declared slots
    model: Model                                      # models["policy"]
    tools: Tools                                      # imported tools
    environments: Environments | None
    random: random.Random                             # seeded from run_id
    context_hints: ContextHints                       # the task's declared hints, for the agent

    def now(self) -> datetime: ...
    def reward(self, value: float, *, slot: str = "policy", key: str = "default") -> None: ...
    def exclude_from_training(self, reason: str) -> None: ...
    async def emit(self, kind: str, payload: Any, *, to: Address | None = None) -> None: ...   # durable output
    async def send(self, to: Address, envelope: Envelope, *, priority: Priority = Priority.NORMAL) -> None: ...
    async def spawn(self, specification: RunSpecification) -> ChildRun: ...     # await child.result()
    async def sleep(self, duration: timedelta) -> None: ...
    async def gather(self, *awaitables: Awaitable[T]) -> list[T]: ...
    def patched(self, change_id: str) -> bool: ...                    # see determinism.md#versioning

class Model:
    capabilities: CapabilityContract
    async def sample(self, messages: list[Message], *, tools: list[ToolSpecification] = (),
                     max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None) -> Message: ...
```

`Model.sample` takes the full message list; the core computes the `ContextDelta` against the slot's previous request
(longest common prefix by digest), so compaction and non-append contexts need no special handling by authors.

Under a durable runner, `now()`, `random` and every awaited operation are recorded, and the code must follow the
[determinism rules](determinism.md).

## Examples

**Single turn, no tools, no environment:**

```python
class ArithmeticReasoning(Task):
    def __init__(self, parameters: dict):
        self.question, self.answer = parameters["question"], parameters["answer"]

    async def start(self, run):
        return Observation(self.question)

    async def respond(self, run, reply):
        return End(reward=1.0 if extract_answer(reply.text) == self.answer else 0.0)
```

**Environment-driven** — `respond` is the environment's step:

```python
class Wordle(Task):
    max_turns = 6

    def __init__(self, parameters: dict):
        self.secret = parameters["word"]

    async def start(self, run):
        return Observation("Guess the 5-letter word.")

    async def respond(self, run, reply):
        guess = parse_guess(reply.text)
        if guess is None:
            return Observation("Invalid. Reply with one 5-letter word.", reward=-0.1)
        if guess == self.secret:
            return End(reward=1.0)
        return Observation(feedback(guess, self.secret))
```

**Tool-driven with a computer** — keep the default `respond`:

```python
class FixFailingTest(Task):
    async def setup(self, run):
        self.workspace = await run.environments.create(self.parameters["environment"])

    async def start(self, run):
        return Observation(self.parameters["issue"])

    async def score(self, run):
        result = await self.workspace.execute("pytest -q", cwd="/workspace")
        return 1.0 if result.exit_code == 0 else 0.0

    async def teardown(self, run):
        await self.workspace.destroy()

    @tool
    async def bash(self, command: str) -> str:
        """Run a shell command in /workspace."""
        return (await self.workspace.execute(command, cwd="/workspace")).output
```

**A conversation** — wait for the next message instead of ending:

```python
class SupportConversation(Task):
    imports = ["tickets", "memory"]                      # memory is a tool like any other

    async def start(self, run):
        return WaitFor("message")                         # the first message starts the episode

    async def respond(self, run, reply):
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        await run.emit("reply", reply.content, to=run.conversation.origin)   # delivered by a connector
        return WaitFor("message", timeout=timedelta(hours=24))
```

**A second model inside the environment** — a simulated user on a frozen slot:

```python
class SimulatedSupportConversation(Task):
    models = {"policy": ModelSlot(), "user": ModelSlot(trainable=False)}

    async def respond(self, run, reply):
        user_turn = await run.models["user"].sample(self.user_view(run.history, reply))
        if "[resolved]" in user_turn.text:
            return End(reward=1.0)
        return Observation(user_turn.text)
```

Orchestrating several agents (planner → workers → reviewer) is a composition of runs: a tool or `respond` spawns
child runs with `run.spawn(...)`, each an ordinary task loop.
