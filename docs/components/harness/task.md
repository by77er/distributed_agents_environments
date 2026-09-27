# Task

Status: **Proposed** · See [ADR-0012](../../decisions/0012-task-agent-loop.md)

A **Task** is the environment an agent acts in, in the reinforcement-learning sense: it owns compute
environments, defines the tools (the action space), produces the first observation, responds to every model turn,
and scores the episode. It is written as a Python class; its hooks are ordinary `async` methods with ordinary
control flow. Durability comes from replay ([durability.md](durability.md)), not from how the code is structured.

## Interface

```python
class Task:
    # Declarations (class attributes)
    models: ClassVar[dict[str, ModelSlot]] = {"policy": ModelSlot()}   # the agent acts through "policy"
    imports: ClassVar[list[str]] = []                                  # external tool sets, bound per run
    max_turns: ClassVar[int | None] = None                             # loop truncates after this many turns
    context_hints: ClassVar[ContextHints] = ContextHints()             # advisory for the agent

    def __init__(self, parameters: Any) -> None: ...                   # once per run; deterministic

    # Lifecycle hooks
    async def setup(self, run: RunContext) -> None: ...                # optional; once per run
    async def start(self, run: RunContext) -> Observation: ...         # REQUIRED; the first observation
    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return await self.run_tools(run, reply)                        # default: execute tool calls
    async def score(self, run: RunContext) -> float | None: ...        # optional; episode-level reward
    async def teardown(self, run: RunContext) -> None: ...             # optional; always runs; idempotent

    # Tools
    def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]: ...   # default: all declared tools
    async def run_tools(self, run: RunContext, reply: Message) -> Observation: ...
        # executes every tool call in `reply` (concurrently), returns one TOOL message with a tool_result per
        # call; returns End() if `reply` contains no tool calls
```

| Hook | Required | What the framework does with it |
|---|---|---|
| `setup` | no | Timed and attributed separately; failures are infrastructure errors, not agent failures |
| `start` | **yes** | Its observation opens the episode |
| `respond` | no | The environment's step; its observation's reward is bound to the reply it answers |
| `score` | no | Episode-level reward, attached to the end of the trajectory |
| `teardown` | no | Always runs if `setup` began |

## Observation

The environment's response to one model turn.

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

def End(reward: float | None = None, *, truncated: bool = False,
        info: dict[str, Any] | None = None) -> Observation: ...
```

`Observation("text")` builds one USER text message; `Observation(message)` and `Observation([messages])` take
canonical `Message`s.

| Field | Consumed by | Becomes |
|---|---|---|
| `messages` | the agent, then the model | a context span in the recorder's session tree (loss mask 0) |
| `reward` | the trajectory assembler | a reward at the last sampled token of the reply it answers |
| `end` | the loop and the trainer | the trajectory's `ending` |
| `info` | people, metrics | the run log only |

**Validation** (violations fail the run with `INVALID_OBSERVATION`):

1. `messages` contains only USER and TOOL roles. The environment cannot write the policy's turns or the system
   prompt (the agent owns it).
2. If the reply contained tool calls, the observation contains a `tool_result` for every `call_id` — including
   when it also ends the episode.
3. A non-terminal observation has at least one message.
4. Blocks larger than 64 KiB are moved to blob storage automatically.

Rewards for other model slots (an opponent, a simulated user) are assigned with `run.reward(value, slot=...)`.

## Tools

```python
class FixFailingTest(Task):
    @tool
    async def bash(self, command: str, timeout_seconds: int = 120) -> str:
        """Run a shell command in /workspace; returns combined output."""
        result = await self.workspace.execute(command, cwd="/workspace", timeout=timeout_seconds)
        return result.output
```

- `@tool` methods are collected by the base class. The `ToolSpecification` comes from the signature (type hints →
  JSON Schema) and the docstring; `@tool(name=..., retry_class=..., timeout=...)` overrides defaults.
- Tool bodies run **in the task host**, under the same replay rules as hooks. They reach the outside world only
  through handles (environments, imported tools).
- An exception raised in a tool body becomes a `tool_result` with `is_error = true` and the exception message; the
  model can react to it.
- **Imported tools** (`imports = ["github"]`) are external tool sets — MCP servers, HTTP services, other agents,
  humans — bound per run in `RunBinding.imports` and executed by the [tool router](../tool-router/README.md). They
  appear in the action space alongside `@tool` methods. Credentialed external access always goes through imports:
  task code has no network and no credentials.
- **Environment-provided tools** (tool servers shipped in an environment image) are added with
  `run.tools.add(await environment.provided_tools())`, typically in `setup`.
- The tool set is resolved and pinned at run start (`tools.resolved`); every later change (`run.tools.add`) is an
  explicit `tools.changed` event. `tools_for_turn` may expose any subset of the declared tools on a given turn.

## Environments

Compute environments are Python handles. Every method is an effect executed by the runtime; the task never talks
to a driver, so the same task runs on Firecracker, a pod, or a VPS.

```python
@dataclass(frozen=True)
class Template:                                     # a content-addressed build recipe
    base: str                                       # any OCI image, e.g. one baked by your own pipeline
    files: Mapping[str, bytes | BlobReference | Asset] = field(default_factory=dict)   # path → content (archives extracted)
    run: Sequence[str] = ()                         # build commands, executed in order
    environment_variables: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class EnvironmentSpecification:
    template: Template | None = None                # build recipe (a bare `Template(base=...)` is a prebuilt image), or
    snapshot: Snapshot | None = None                # restore a snapshot this task took earlier (long-lived workspaces)
    resources: Resources = Resources()
    security: str = "untrusted-offline"             # security class; concrete profile from RunBinding.security_classes
    persistence: Persistence = Persistence.EPHEMERAL
    time_to_live: timedelta = timedelta(hours=4)

class Environments:                                 # run.environments
    async def create(self, specification: EnvironmentSpecification) -> Environment: ...
    def scratch(self, specification: EnvironmentSpecification) -> AsyncContextManager[Environment]: ...  # destroyed on exit

class Environment:                                  # a handle; pickles as a reference to environment_id
    environment_id: str
    async def execute(self, command: str | list[str], *, cwd: str | None = None,
                      environment_variables: dict[str, str] | None = None, user: str | None = None,
                      stdin: bytes | None = None, timeout: float | None = None) -> ExecutionResult: ...
    async def put(self, source: bytes | BlobReference | Asset, destination: str, *,
                  extract: bool = False, mode: int | None = None) -> None: ...
    async def get(self, path: str) -> bytes: ...                        # up to the inline limit
    async def get_reference(self, path: str) -> BlobReference: ...     # any size, via blob storage
    async def read_text(self, path: str) -> str: ...
    async def write_text(self, path: str, text: str) -> None: ...
    async def list(self, path: str, *, recursive: bool = False) -> list[FileInformation]: ...
    async def remove(self, path: str, *, recursive: bool = False) -> None: ...
    async def move(self, source: str, destination: str) -> None: ...
    async def provided_tools(self) -> list[ToolSpecification]: ...
    async def snapshot(self, kind: SnapshotKind = SnapshotKind.FULL) -> Snapshot: ...
    async def hibernate(self) -> None: ...
    async def resume(self) -> None: ...
    async def destroy(self) -> None: ...
```

`Asset` is a file bundled in the task's code package (`asset("fixtures/repository.tar")`); `BlobReference`s come
from parameters, other environments, or earlier `get_reference` calls. Host-produced bytes above 64 KiB are moved
to blob storage by the runtime when the effect is committed.

**Templates are how setup cost is paid once.** The Environment Manager hashes a `Template` recipe, builds it the
first time it is requested (concurrent requests for the same hash wait for one build), and keeps the result as a
golden snapshot. Every environment created from the same recipe restores that snapshot, so runs of the same row
start from identical state and share memory pages on a host. Complicated environments can be baked into an image
by any external pipeline and used as `Template(base="registry/…")`. Setup that must be Python logic runs in
`setup` for every run.

| Handle call | Effect | Served by |
|---|---|---|
| `create`, `snapshot`, `hibernate`, `resume`, `destroy` | `environment.requested` (lifecycle) | [Environment Manager](../environments/README.md) |
| `execute`, `put`, `get`, `get_reference`, `read_text`, `write_text`, `list`, `remove`, `move`, `provided_tools`, provided-tool calls | `environment.requested` (call) | envlet → envd ([Environment API](../environments/env-api.md)) |

**Ownership** (P11): environments created by a run are owned by it. `teardown` should destroy them; the runtime
destroys any the run still owns when it reaches a terminal state. A run can only address environments it owns or is attached to — the runtime rejects anything else.

**Verification**: `score` should verify in a `run.environments.scratch(...)` environment, not in one the agent
controlled (see [trust-boundaries](../../architecture/trust-boundaries.md)).

## RunContext

```python
class RunContext:
    run_id: str
    turn: int                                         # completed model turns so far
    history: History                                  # read-only: start observation, then (reply, observation) pairs
    models: Mapping[str, Model]                       # the task's declared slots
    model: Model                                      # models["policy"]
    environments: Environments
    tools: Tools                                      # imported tools; add environment-provided tools
    random: random.Random                             # seeded from run_id; deterministic under replay
    context_hints: ContextHints                       # the task's declared hints, for the agent

    def now(self) -> datetime: ...                    # commit time of the latest input event; deterministic
    def reward(self, value: float, *, slot: str = "policy", key: str = "default") -> None: ...
    async def signal(self, kind: str, *, timeout: timedelta | None = None) -> Signal: ...
    async def sleep(self, duration: timedelta) -> None: ...
    async def spawn(self, specification: RunSpecification) -> ChildRun: ...     # await child.result()
    async def send(self, run_id: str, message: Message) -> None: ...
    async def gather(self, *awaitables: Awaitable[T]) -> list[T]: ...        # deterministic concurrency
    def patched(self, change_id: str) -> bool: ...                            # see durability.md#versioning

class Model:
    capabilities: CapabilityContract
    async def sample(self, messages: list[Message], *, tools: list[ToolSpecification] = (),
                     max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None) -> Message: ...
```

`Model.sample` takes the full message list; the SDK computes the `ContextDelta` against the slot's previous request
(longest common prefix by digest), so compaction and non-append contexts need no special handling by authors.

## Examples

**Tool-driven** — keep the default `respond`:

```python
class FixFailingTest(Task):
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.environment = EnvironmentSpecification(
            template=Template(
                base="ghcr.io/acme/py312-dev",
                files={"/workspace": parameters["repository"]},
                run=["pip install -e /workspace"],
            ),
            security="untrusted-allowlist",
        )

    async def setup(self, run):
        self.workspace = await run.environments.create(self.environment)   # built once per row, then restored

    async def start(self, run):
        return Observation(self.parameters["issue"])

    async def score(self, run):
        patch = await self.workspace.execute("git diff", cwd="/workspace")
        async with run.environments.scratch(self.environment) as clean:           # same template, clean state
            await clean.put(patch.stdout, "/tmp/patch.diff")
            result = await clean.execute("git apply /tmp/patch.diff && pytest -q", cwd="/workspace")
        return 1.0 if result.exit_code == 0 else 0.0

    async def teardown(self, run):
        await self.workspace.destroy()

    @tool
    async def bash(self, command: str) -> str:
        """Run a shell command in /workspace."""
        return (await self.workspace.execute(command, cwd="/workspace")).output
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

**Hybrid** — default behavior plus environment behavior:

```python
    async def respond(self, run, reply):
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        report = await self.run_tests()
        if report.passed or self.attempts == 3:
            return End()
        self.attempts += 1
        return Observation(f"Tests still fail:\n{report.tail}")
```

**A second model inside the environment** — a simulated user on a frozen slot:

```python
class SupportConversation(Task):
    models = {"policy": ModelSlot(), "user": ModelSlot(trainable=False)}

    async def respond(self, run, reply):
        user_turn = await run.models["user"].sample(self.user_view(run.history, reply))
        if "[resolved]" in user_turn.text:
            return End(reward=1.0)
        return Observation(user_turn.text)
```

Orchestrating several agents (planner → workers → reviewer) is a composition of runs: a tool or `respond` spawns
child runs with `run.spawn(...)`, each an ordinary task loop.
