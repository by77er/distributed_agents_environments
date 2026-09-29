# API reference

Generated from the source by `scripts/generate_reference.py`; do not edit by hand. Every public name,
grouped by module, alphabetically. Types and defaults appear as written in the source. The
[guide](README.md) explains how the pieces fit together.

## Contents

- **[`rollout.core.harness`](#rolloutcoreharness)** — Writing tasks and agents. [`Address`](#address), [`Agent`](#agent), [`agent_program`](#agent_program), [`AgentProgram`](#agentprogram), [`ContextHints`](#contexthints), [`ConversationKey`](#conversationkey), [`DeliveryMode`](#deliverymode), [`DeliveryPolicy`](#deliverypolicy), [`Deployment`](#deployment), [`DirectModel`](#directmodel), [`Effects`](#effects), [`End`](#end), [`Ending`](#ending), [`EndpointModel`](#endpointmodel), [`Envelope`](#envelope), [`History`](#history), [`HistoryShape`](#historyshape), [`instantiate`](#instantiate), [`Interrupted`](#interrupted), [`InvalidObservation`](#invalidobservation), [`Model`](#model), [`ModelBinding`](#modelbinding), [`ModelSlot`](#modelslot), [`Observation`](#observation), [`Priority`](#priority), [`Program`](#program), [`ProgramReference`](#programreference), [`RecordedModel`](#recordedmodel), [`register`](#register), [`resolve`](#resolve), [`rollout`](#rollout), [`RunBinding`](#runbinding), [`RunContext`](#runcontext), [`RunHandle`](#runhandle), [`Runner`](#runner), [`RunOutcome`](#runoutcome), [`RunSpecification`](#runspecification), [`RunStatus`](#runstatus), [`SamplingParameters`](#samplingparameters), [`Task`](#task), [`tool`](#tool), [`Turn`](#turn), [`WaitFor`](#waitfor)
- **[`rollout.core.contracts`](#rolloutcorecontracts)** — Types that cross layers: canonical content, identifiers, digests, effects, events. [`arguments_digest`](#arguments_digest), [`BlobReference`](#blobreference), [`Block`](#block), [`CallContext`](#callcontext), [`canonical_json`](#canonical_json), [`CapabilityContract`](#capabilitycontract), [`Conflict`](#conflict), [`context_digests`](#context_digests), [`ContextDelta`](#contextdelta), [`ContextOverflow`](#contextoverflow), [`ContractModel`](#contractmodel), [`ContractViolation`](#contractviolation), [`DeadlineExceeded`](#deadlineexceeded), [`digest`](#digest), [`effect_id`](#effect_id), [`EffectCompletion`](#effectcompletion), [`EffectIdentity`](#effectidentity), [`EffectKind`](#effectkind), [`EffectRequest`](#effectrequest), [`EffectStatus`](#effectstatus), [`EMPTY_DIGEST`](#empty_digest), [`FinishReason`](#finishreason), [`FrozenSequence`](#frozensequence), [`InternalError`](#internalerror), [`Media`](#media), [`Message`](#message), [`message_digest`](#message_digest), [`ModelEndpoint`](#modelendpoint), [`ModelEndpointError`](#modelendpointerror), [`NamedToolChoice`](#namedtoolchoice), [`NeedFullContext`](#needfullcontext), [`new_job_id`](#new_job_id), [`new_run_id`](#new_run_id), [`new_ulid`](#new_ulid), [`Overloaded`](#overloaded), [`Provenance`](#provenance), [`Reasoning`](#reasoning), [`ReasoningScope`](#reasoningscope), [`ReasoningSupport`](#reasoningsupport), [`ResultBlock`](#resultblock), [`RetryClass`](#retryclass), [`Role`](#role), [`RUN_EVENT_SCHEMA_VERSION`](#run_event_schema_version), [`RunEvent`](#runevent), [`RunEventType`](#runeventtype), [`RunFailureClass`](#runfailureclass), [`SampleRequest`](#samplerequest), [`SampleResult`](#sampleresult), [`session_id`](#session_id), [`SessionIdentity`](#sessionidentity), [`spec_hash`](#spec_hash), [`TERMINAL_EVENT_TYPES`](#terminal_event_types), [`Text`](#text), [`ToolAnnotations`](#toolannotations), [`ToolCall`](#toolcall), [`ToolChoice`](#toolchoice), [`ToolChoiceMode`](#toolchoicemode), [`ToolResult`](#toolresult), [`ToolResultBlock`](#toolresultblock), [`ToolSpecification`](#toolspecification), [`Usage`](#usage)
- **[`rollout.core.local`](#rolloutcorelocal)** — In-process implementations for the local profile. [`EndpointFactory`](#endpointfactory), [`LocalRunContext`](#localruncontext), [`LocalRunHandle`](#localrunhandle), [`LocalRunner`](#localrunner), [`RewardAssignment`](#rewardassignment), [`RunNotLive`](#runnotlive)
- **[`rollout.core.testing`](#rolloutcoretesting)** — Test doubles: a scripted model endpoint and helpers. [`events_of`](#events_of), [`local_run`](#local_run), [`payload`](#payload), [`ScriptedModelEndpoint`](#scriptedmodelendpoint), [`ScriptedReply`](#scriptedreply), [`tool_call_reply`](#tool_call_reply)

## `rollout.core.harness`

Writing tasks and agents.

### `Address`

*class* · `src/rollout/core/harness/conversations.py`

```python
class Address(ContractModel)
```

Where a message or output goes.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `Literal['conversation', 'run', 'external']` | required |  |
| `value` | `str` | required | `{deployment}/{key}`, a `run_id`, or a connector target. |

### `Agent`

*class* · `src/rollout/core/harness/agent.py`

```python
class Agent
```

The policy side of the loop: what the model sees each turn, and how its output becomes one action.

The default agent samples the policy slot once per turn with the whole history. Subclass it to change the
context (compaction, windows) or the way it acts (plan-then-act, self-critique).

| Field | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str \| None` | `None` | Sent first in every context when set. |

**Methods**

- `def __init__(self, configuration: Any = None) -> None` — Must be deterministic: under a durable runner the agent is re-created on replay.
- `def select_context(self, history: History, hints: ContextHints) -> list[Message]` — What the model sees this turn. Default: the system prompt and the history shaped by the task's hints.
- `async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message` — Produce one action. Default: one sample of the policy slot.

### `agent_program`

*function* · `src/rollout/core/harness/runner.py`

```python
def agent_program(task: type[Task], agent: type[Agent] = Agent, *, task_parameters: JsonValue = None, agent_configuration: JsonValue = None) -> ProgramReference
```

A reference to the task loop for `task` and `agent`.

### `AgentProgram`

*class* · `src/rollout/core/harness/program.py`

```python
class AgentProgram(Program)
```

The task loop: a task and an agent.

**Methods**

- `def __init__(self, task: Task, agent: Agent) -> None`
- `def model_slots(self) -> Mapping[str, ModelSlot]`
- `def context_hints(self) -> ContextHints`
- `async def main(self, run: RunContext) -> None`

### `ContextHints`

*class* · `src/rollout/core/harness/history.py`

```python
class ContextHints
```

Advisory for the agent: how much history the task needs the model to see.

| Field | Type | Default | Description |
|---|---|---|---|
| `history` | `HistoryShape` | `HistoryShape.FULL` |  |
| `window` | `int \| None` | `None` | For `WINDOW`: the number of recent turns. |

### `ConversationKey`

*class* · `src/rollout/core/harness/conversations.py`

```python
class ConversationKey(ContractModel)
```

Identifies a conversation: a deployment and a caller-chosen key. One live run per conversation.

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment` | `str` | required | e.g. `acme/support-bot`. |
| `key` | `str` | required | Caller-chosen, e.g. `slack:T1/C2/171.2` or `user:42`. |
| `origin` | `Address \| None` | `None` | Where replies go by default (e.g. a connector target). |

### `DeliveryMode`

*class* · `src/rollout/core/harness/conversations.py`

```python
class DeliveryMode(StrEnum)
```

How a message reaches a run that is busy.

| Member | Value | Description |
|---|---|---|
| `QUEUE` | `'queue'` | Deliver only when the run next waits (`WaitFor`). |
| `STEER` | `'steer'` | Add to the next observation at the next turn boundary; cancel nothing. |
| `INTERRUPT` | `'interrupt'` | Cancel the in-flight reply now; the message becomes the next observation. |

### `DeliveryPolicy`

*class* · `src/rollout/core/harness/conversations.py`

```python
class DeliveryPolicy(ContractModel)
```

Maps priorities to delivery modes. Default: LOW → QUEUE, NORMAL → STEER, HIGH → INTERRUPT.

| Field | Type | Default | Description |
|---|---|---|---|
| `modes` | `Mapping[Priority, DeliveryMode]` | `Field(default_factory=_default_modes)` |  |
| `max_priority_by_sender` | `Mapping[str, Priority]` | `Field(default_factory=dict[str, Priority])` | Caps per sender class, so an external system cannot interrupt when it should only queue. |

**Methods**

- `def mode(self, priority: Priority, sender: str | None = None) -> DeliveryMode` — The delivery mode for a message, after capping the priority by sender.

### `Deployment`

*class* · `src/rollout/core/harness/runner.py`

```python
class Deployment(ContractModel)
```

A named, addressable agent: conversations addressed to it start runs of its specification.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | `{namespace}/{name}`, e.g. `acme/support-bot`. |
| `specification` | `RunSpecification` | required |  |

### `DirectModel`

*class* · `src/rollout/core/harness/runner.py`

```python
class DirectModel(ContractModel)
```

A model served by a provider's API through a direct adapter; nothing is recorded.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | The key of an endpoint factory registered with the runner, e.g. `codex`. |
| `model` | `str` | required |  |
| `sampling` | `SamplingParameters` | `SamplingParameters()` |  |

### `Effects`

*class* · `src/rollout/core/harness/model.py`

```python
class Effects(Protocol)
```

How a run performs effects. The runner decides what performing means: a direct call, or a durable step.

**Methods**

- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue]) -> T` — Assign the next `effect_id`, digest `arguments`, and run `execute(effect_id, arguments_digest)`.
  
    `completion` renders the result for the run's events.

### `End`

*function* · `src/rollout/core/harness/observation.py`

```python
def End(reward: float | None = None, *, truncated: bool = False, info: Mapping[str, Any] | None = None) -> Observation
```

A terminal observation.

### `Ending`

*class* · `src/rollout/core/harness/observation.py`

```python
class Ending(StrEnum)
```

How an episode ended.

| Member | Value | Description |
|---|---|---|
| `TERMINATED` | `'terminated'` | A real end state (value methods do not bootstrap). |
| `TRUNCATED` | `'truncated'` | Stopped by a limit: turns, time, budget (value methods may bootstrap). |

### `EndpointModel`

*class* · `src/rollout/core/harness/model.py`

```python
class EndpointModel
```

A model slot bound to an endpoint. Sends the full context; context deltas come with the direct adapters.

**Methods**

- `def __init__(self, endpoint: ModelEndpoint, session_id: str, effects: Effects) -> None`
- `@property def capabilities(self) -> CapabilityContract`
- `@property def usage(self) -> Usage | None`
- `async def sample(self, messages: Sequence[Message], *, tools: Sequence[ToolSpecification] = (), max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None) -> Message`

### `Envelope`

*class* · `src/rollout/core/harness/conversations.py`

```python
class Envelope(ContractModel)
```

A message delivered to a run.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `'message'` | `message`, or an application-defined kind that `WaitFor` can select. |
| `content` | `FrozenSequence[Block]` | `()` |  |
| `data` | `JsonValue` | `None` | A structured payload. |
| `reply_to` | `Address \| None` | `None` |  |
| `message_id` | `str` | `''` | Set by the runner: the sender's `effect_id` or the caller's idempotency key. |
| `sender` | `str \| None` | `None` | Set by the runner; never trusted from the payload. |

### `History`

*class* · `src/rollout/core/harness/history.py`

```python
class History
```

Read-only for agents and tasks; the loop appends to it through the run context.

**Methods**

- `def __init__(self) -> None`
- `@property def turns(self) -> tuple[Turn, ...]` — Every turn, the start observation first.
- `def append(self, turn: Turn) -> None` — For run contexts only.
- `def messages(self, hints: ContextHints | None = None) -> list[Message]` — The episode's messages in order, shaped by `hints`.

### `HistoryShape`

*class* · `src/rollout/core/harness/history.py`

```python
class HistoryShape(StrEnum)
```

How much of the history the model should see.

| Member | Value | Description |
|---|---|---|
| `FULL` | `'full'` | Everything. |
| `LATEST_OBSERVATION` | `'latest_observation'` | Only the latest observation: observations are complete states. |
| `WINDOW` | `'window'` | The start observation and the most recent `window` turns. |

### `instantiate`

*function* · `src/rollout/core/harness/runner.py`

```python
def instantiate(reference: ProgramReference) -> Program
```

Create the program a reference names.

### `Interrupted`

*class* · `src/rollout/core/harness/context.py`

```python
class Interrupted(Exception)
```

The reply in progress was cancelled by a message delivered with mode `INTERRUPT`.

**Methods**

- `def __init__(self, envelope: Envelope, reply_effect_id: str | None) -> None`

### `InvalidObservation`

*class* · `src/rollout/core/harness/observation.py`

```python
class InvalidObservation(Exception)
```

A hook returned an observation that breaks the validation rules; the run fails with `INVALID_OBSERVATION`.

### `Model`

*class* · `src/rollout/core/harness/context.py`

```python
class Model(Protocol)
```

A model slot as code sees it. Nothing here identifies the policy, weights or engine.

**Methods**

- `@property def capabilities(self) -> CapabilityContract`
- `@property def usage(self) -> Usage | None` — Usage reported by the latest sample, if any: drives compaction decisions.
- `async def sample(self, messages: Sequence[Message], *, tools: Sequence[ToolSpecification] = (), max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None) -> Message`

### `ModelBinding`

*class* · `src/rollout/core/harness/runner.py`

```python
class ModelBinding(ContractModel)
```

Exactly one of `direct` or `recorded`.

| Field | Type | Default | Description |
|---|---|---|---|
| `direct` | `DirectModel \| None` | `None` |  |
| `recorded` | `RecordedModel \| None` | `None` |  |

### `ModelSlot`

*class* · `src/rollout/core/harness/task.py`

```python
class ModelSlot
```

A model the task declares. The agent acts through `policy`; other slots (a simulated user, an opponent) are
sampled by the task itself.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainable` | `bool` | `True` |  |

### `Observation`

*class* · `src/rollout/core/harness/observation.py`

```python
class Observation
```

What the model is shown next, with the reward for the reply it answers.

`Observation("text")` builds one USER text message; `Observation(message)` and `Observation([messages])` take
canonical messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | `tuple[Message, ...]` | see constructor | Shown to the model next: USER and TOOL messages only. |
| `reward` | `float \| None` | see constructor | Bound to the reply this observation answers. |
| `end` | `Ending \| None` | see constructor | `None` means the episode continues. |
| `info` | `Mapping[str, Any]` | see constructor | Logged; never shown to the model. |

**Methods**

- `def __init__(self, messages: ObservationContent = (), *, reward: float | None = None, end: Ending | None = None, info: Mapping[str, Any] | None = None) -> None`

### `Priority`

*class* · `src/rollout/core/harness/conversations.py`

```python
class Priority(StrEnum)
```

A sender's priority; the run's `DeliveryPolicy` maps it to a delivery mode.

| Member | Value | Description |
|---|---|---|
| `LOW` | `'low'` |  |
| `NORMAL` | `'normal'` |  |
| `HIGH` | `'high'` |  |

### `Program`

*class* · `src/rollout/core/harness/program.py`

```python
class Program
```

What a run executes. `AgentProgram` is the task loop; plain durable workflows are other programs.

**Methods**

- `def model_slots(self) -> Mapping[str, ModelSlot]` — The model slots the program samples; a runner binds an endpoint to each.
- `def context_hints(self) -> ContextHints`
- `async def main(self, run: RunContext) -> None`

### `ProgramReference`

*class* · `src/rollout/core/harness/runner.py`

```python
class ProgramReference(ContractModel)
```

What a run executes, by name, so a runner in another process can re-create it.

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `str` | required | `module:QualifiedName` of a `Program` class. |
| `parameters` | `JsonValue` | `None` |  |
| `code_reference` | `str \| None` | `None` | `{package}@{content_hash}`; pins durable runs to the code they started with. |

### `RecordedModel`

*class* · `src/rollout/core/harness/runner.py`

```python
class RecordedModel(ContractModel)
```

A channel served through the recorder (M1).

| Field | Type | Default | Description |
|---|---|---|---|
| `channel` | `str` | required |  |
| `sampling` | `SamplingParameters` | `SamplingParameters()` |  |

### `register`

*function* · `src/rollout/core/harness/runner.py`

```python
def register(cls: type) -> str
```

Make a class resolvable by name in this process, even if it cannot be imported (e.g. defined in a script).

### `resolve`

*function* · `src/rollout/core/harness/runner.py`

```python
def resolve(name: str) -> type
```

The class a `module:QualifiedName` names: registered in this process, or imported.

### `rollout`

*function* · `src/rollout/core/harness/loop.py`

```python
async def rollout(task: Task, agent: Agent, run: RunContext) -> None
```

Run one episode of `task` with `agent`. Raises `InvalidObservation` or whatever a hook raised.

### `RunBinding`

*class* · `src/rollout/core/harness/runner.py`

```python
class RunBinding(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `models` | `Mapping[str, ModelBinding]` | required | Model slot → how it is served. |
| `delivery` | `DeliveryPolicy` | `DeliveryPolicy()` |  |

### `RunContext`

*class* · `src/rollout/core/harness/context.py`

```python
class RunContext(Protocol)
```

Everything task and agent code can reach during a run. Passed to every hook as `run`.

**Methods**

- `@property def run_id(self) -> str`
- `@property def conversation(self) -> ConversationKey | None`
- `@property def turn(self) -> int` — Completed model turns so far.
- `@property def history(self) -> History`
- `@property def models(self) -> Mapping[str, Model]`
- `@property def model(self) -> Model` — `models["policy"]`.
- `@property def random(self) -> random.Random` — Seeded from `run_id`.
- `@property def context_hints(self) -> ContextHints`
- `def now(self) -> datetime` — The current time. Use it instead of the wall clock, which durable runs cannot replay.
- `def reward(self, value: float, *, slot: str = 'policy', key: str = 'default') -> None` — Assign a reward to a model slot outside an observation (e.g. to an opponent, or several keyed rewards).
- `def exclude_from_training(self, reason: str) -> None` — Mark the run as unsuitable for training, e.g. after an infrastructure fault that is not the policy's.
- `async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]` — Await concurrently, in order. Equivalent to `asyncio.gather`.
- `def patched(self, change_id: str) -> bool` — `True` unless replaying history recorded before the change (see docs/core/harness/determinism.md).
- `async def emit(self, kind: str, payload: JsonValue, *, to: Address | None = None) -> None` — Durable output, such as a reply to a person; a connector or client delivers it.
- `def record(self, observation: Observation | WaitFor, *, reply: Message | None = None) -> None` — Append a turn to the history. A `WaitFor` records only the reply it answers.
- `async def wait_for_message(self, wait: WaitFor) -> Envelope | None` — Suspend until a message of `wait.kind` arrives; `None` on timeout.
- `def take_steering_messages(self) -> list[Envelope]` — Messages delivered with mode `STEER` since the last turn boundary.
- `async def interruptible[T](self, reply: Awaitable[T]) -> T` — Await an agent's reply; raises `Interrupted` if a message with mode `INTERRUPT` arrives meanwhile.

### `RunHandle`

*class* · `src/rollout/core/harness/runner.py`

```python
class RunHandle(Protocol)
```

**Methods**

- `@property def run_id(self) -> str`
- `async def result(self) -> RunOutcome`
- `def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]` — Every event from `from_seq`, then new ones as they are recorded, until the run ends.

### `Runner`

*class* · `src/rollout/core/harness/runner.py`

```python
class Runner(Protocol)
```

**Methods**

- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None) -> RunHandle`
- `async def send(self, to: Address, envelope: Envelope, *, priority: Priority = Priority.NORMAL, idempotency_key: str | None = None) -> None` — Deliver a message; a message to a conversation starts its run when none is live.
- `async def cancel(self, run_id: str, *, reason: str) -> None`

### `RunOutcome`

*class* · `src/rollout/core/harness/runner.py`

```python
class RunOutcome(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `status` | `RunStatus` | required |  |
| `failure_class` | `RunFailureClass \| None` | `None` |  |
| `detail` | `str \| None` | `None` |  |

### `RunSpecification`

*class* · `src/rollout/core/harness/runner.py`

```python
class RunSpecification(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `ProgramReference` | required |  |
| `binding` | `RunBinding` | required |  |

### `RunStatus`

*class* · `src/rollout/core/harness/runner.py`

```python
class RunStatus(StrEnum)
```

| Member | Value | Description |
|---|---|---|
| `COMPLETED` | `'completed'` |  |
| `FAILED` | `'failed'` |  |
| `CANCELLED` | `'cancelled'` |  |

### `SamplingParameters`

*class* · `src/rollout/core/harness/runner.py`

```python
class SamplingParameters(ContractModel)
```

Configured on bindings, never by task or agent code.

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | `1.0` |  |
| `top_p` | `float` | `1.0` |  |
| `top_k` | `int \| None` | `None` |  |
| `max_output_tokens` | `int \| None` | `None` |  |
| `stop` | `FrozenSequence[str]` | `()` |  |
| `seed` | `int \| None` | `None` |  |
| `reasoning_effort` | `str \| None` | `None` | For providers with reasoning controls, e.g. `low`, `medium`, `high`. |

### `Task`

*class* · `src/rollout/core/harness/task.py`

```python
class Task
```

The environment an agent acts in: tools, the first observation, responses to replies, and scoring.

Subclass it and implement `start`; override the other hooks as needed. Declarations are class attributes.

| Field | Type | Default | Description |
|---|---|---|---|
| `models` | `ClassVar[dict[str, ModelSlot]]` | `{'policy': ModelSlot()}` | The model slots the task uses. The agent acts through `policy`. |
| `imports` | `ClassVar[list[str]]` | `[]` | External tool sets, bound per run. |
| `max_turns` | `ClassVar[int \| None]` | `None` | The loop truncates the episode after this many model turns. |
| `context_hints` | `ClassVar[ContextHints]` | `ContextHints()` | Advisory for the agent: how much history the model should see. |
| `declared_tools` | `ClassVar[dict[str, DeclaredTool]]` | `{}` | The `@tool` methods of this class, collected when the class is defined. |

**Methods**

- `def __init__(self, parameters: Any = None) -> None` — Once per run, with the row's parameters. Subclasses may define their own signature.
- `async def setup(self, run: RunContext) -> None` — Once per run, before `start`.
- `async def start(self, run: RunContext) -> Observation | WaitFor` — Open the episode, or wait for the first message.
- `async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor` — The environment's step. Default: execute the reply's tool calls; end the episode when there are none.
- `async def resume(self, run: RunContext, envelope: Envelope) -> Observation | WaitFor` — Turn a message that satisfied a `WaitFor`, or interrupted a turn, into the next observation.
- `async def steer(self, run: RunContext, envelopes: list[Envelope], observation: Observation) -> Observation` — Merge messages delivered with mode `STEER` into the next observation. Default: append as USER content.
- `async def score(self, run: RunContext) -> float | None` — Episode-level reward, attached to the end of the trajectory.
- `async def teardown(self, run: RunContext) -> None` — Always runs if `setup` began; must be idempotent.
- `def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]` — The tools offered this turn. Default: every declared tool.
- `async def run_tools(self, run: RunContext, reply: Message) -> Observation` — Execute every tool call in `reply` concurrently; one TOOL message answers them all.

### `tool`

*function* · `src/rollout/core/harness/tools.py`

```python
def tool[F: Callable[..., Any]](function: F | None = None, /, *, name: str | None = None, retry_class: RetryClass = RetryClass.PURE, timeout: timedelta | None = None) -> F | Callable[[F], F]
```

Declare a task method as a tool: `@tool` or `@tool(name=..., retry_class=..., timeout=...)`.

### `Turn`

*class* · `src/rollout/core/harness/history.py`

```python
class Turn
```

One step of the episode: a reply and the observation that answers it.

The start observation, and observations produced by `resume`, have no reply. A reply answered by a `WaitFor`
has no observation until the run resumes.

| Field | Type | Default | Description |
|---|---|---|---|
| `reply` | `Message \| None` | required |  |
| `observation` | `Observation \| None` | required |  |

### `WaitFor`

*class* · `src/rollout/core/harness/observation.py`

```python
class WaitFor
```

Suspend the run until a message of `kind` arrives or `timeout` passes.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `'message'` |  |
| `timeout` | `timedelta \| None` | `None` |  |
| `on_timeout` | `Observation` | `field(default_factory=lambda: End(truncated=True))` |  |

## `rollout.core.contracts`

Types that cross layers: canonical content, identifiers, digests, effects, events.

### `arguments_digest`

*function* · `src/rollout/core/contracts/digests.py`

```python
def arguments_digest(arguments: JsonValue | BaseModel) -> str
```

Sent with every `effect_id`; receivers reject a known `effect_id` whose arguments digest differs.

### `BlobReference`

*class* · `src/rollout/core/contracts/content.py`

```python
class BlobReference(ContractModel)
```

Content kept in object storage (any block larger than 64 KiB).

| Field | Type | Default | Description |
|---|---|---|---|
| `uri` | `str` | required |  |
| `sha256` | `str` | required |  |
| `size` | `int` | required |  |
| `media_type` | `str` | required |  |

### `Block`

*type alias* · `src/rollout/core/contracts/content.py`

```python
type Block = Annotated[Text | Media | ToolCall | ToolResultBlock | Reasoning, Field(discriminator='type')]
```

### `CallContext`

*class* · `src/rollout/core/contracts/effects.py`

```python
class CallContext(ContractModel)
```

Assembled by the runner, never by task code.

| Field | Type | Default | Description |
|---|---|---|---|
| `labels` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` |  |
| `tenant` | `str \| None` | `None` |  |

### `canonical_json`

*function* · `src/rollout/core/contracts/digests.py`

```python
def canonical_json(value: JsonValue | BaseModel) -> bytes
```

RFC 8785 canonical JSON; a model is dumped with `None` fields omitted.

### `CapabilityContract`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class CapabilityContract(ContractModel)
```

What a model slot guarantees. It must not weaken during a run.

| Field | Type | Default | Description |
|---|---|---|---|
| `contract_version` | `str` | `'1'` |  |
| `context_limit` | `int` | required | Minimum guaranteed. |
| `max_output_tokens` | `int` | required |  |
| `modalities_in` | `frozenset[str]` | `frozenset({'text'})` |  |
| `tool_calling` | `bool` | `True` |  |
| `parallel_tool_calls` | `bool` | `True` |  |
| `reasoning` | `ReasoningSupport` | `ReasoningSupport.NONE` |  |
| `accepts_context_delta` | `bool` | `False` |  |

### `Conflict`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class Conflict(ModelEndpointError)
```

A known `effect_id` arrived with a different arguments digest.

### `context_digests`

*function* · `src/rollout/core/contracts/digests.py`

```python
def context_digests(messages: Sequence[Message]) -> list[str]
```

The digest chain `d₀ … dₙ` of a context: `dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(itemᵢ)))` over raw digest bytes.

`dₖ` identifies the prefix of length `k`, so a retained prefix is recognizable by its own chain value.

### `ContextDelta`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ContextDelta(ContractModel)
```

The context of a request as an edit of the previous request's context in the same slot.

Digests are values of the chain computed by `rollout.core.contracts.digests.context_digests`.

| Field | Type | Default | Description |
|---|---|---|---|
| `parent_digest` | `str \| None` | `None` | Digest of the previous request's context; `None` means `append` is the full context. |
| `keep_prefix` | `int` | `0` | Number of parent items retained (the parent's length for a pure append). |
| `append` | `FrozenSequence[Message]` | `()` |  |
| `digest` | `str` | required |  |

### `ContextOverflow`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ContextOverflow(ModelEndpointError)
```

The context exceeds the contract's limit; agents compact and retry.

**Methods**

- `def __init__(self, context_limit: int) -> None`

### `ContractModel`

*class* · `src/rollout/core/contracts/base.py`

```python
class ContractModel(BaseModel)
```

Base for every contract type: immutable, and unknown fields are kept.

Keeping unknown fields lets a component read and re-write a record written by newer code without dropping
what it does not understand (contracts evolution rule 4).

### `ContractViolation`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ContractViolation(ModelEndpointError)
```

The request exceeds the capability contract.

### `DeadlineExceeded`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class DeadlineExceeded(ModelEndpointError)
```

The request's deadline passed.

### `digest`

*function* · `src/rollout/core/contracts/digests.py`

```python
def digest(value: JsonValue | BaseModel) -> str
```

Lowercase hexadecimal SHA-256 of the canonical JSON.

### `effect_id`

*function* · `src/rollout/core/contracts/identifiers.py`

```python
def effect_id(run_id: str, generation: int, ordinal: int) -> str
```

`{run_id}:{generation}:{ordinal}`: the same on every re-execution, so it is the universal idempotency key.

### `EffectCompletion`

*class* · `src/rollout/core/contracts/effects.py`

```python
class EffectCompletion(ContractModel)
```

The first completion recorded for an `effect_id` wins; later ones are dropped.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | required |  |
| `status` | `EffectStatus` | required |  |
| `payload` | `JsonValue` | `None` |  |
| `error_class` | `str \| None` | `None` |  |

### `EffectIdentity`

*class* · `src/rollout/core/contracts/identifiers.py`

```python
class EffectIdentity
```

The parts of an `effect_id`: `{run_id}:{generation}:{ordinal}`.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required |  |
| `generation` | `int` | required |  |
| `ordinal` | `int` | required |  |

**Methods**

- `@classmethod def parse(cls, effect_id: str) -> 'EffectIdentity'`

### `EffectKind`

*class* · `src/rollout/core/contracts/effects.py`

```python
class EffectKind(StrEnum)
```

The catalog of effects.

| Member | Value | Description |
|---|---|---|
| `MODEL_SAMPLE` | `'model.sample'` |  |
| `TOOL_CALL` | `'tool.call'` |  |
| `ENVIRONMENT_CALL` | `'environment.call'` |  |
| `ENVIRONMENT_LIFECYCLE` | `'environment.lifecycle'` |  |
| `MESSAGE_SEND` | `'message.send'` |  |
| `MESSAGE_WAIT` | `'message.wait'` |  |
| `RUN_SPAWN` | `'run.spawn'` |  |
| `TIMER_SLEEP` | `'timer.sleep'` |  |
| `OUTPUT_EMIT` | `'output.emit'` |  |

### `EffectRequest`

*class* · `src/rollout/core/contracts/effects.py`

```python
class EffectRequest(ContractModel)
```

What an executor receives. Transport is per implementation; these fields are mandatory everywhere.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | required | `{run_id}:{generation}:{ordinal}`; identical on every attempt. |
| `arguments_digest` | `str` | required |  |
| `kind` | `EffectKind` | required |  |
| `run_id` | `str` | required |  |
| `attempt` | `int` | `1` | 1-based; informational. |
| `deadline` | `datetime` | required | Absolute. |
| `payload` | `JsonValue` | required |  |
| `retry_class` | `RetryClass` | `RetryClass.UNKNOWN` |  |
| `context` | `CallContext` | `CallContext()` |  |

### `EffectStatus`

*class* · `src/rollout/core/contracts/effects.py`

```python
class EffectStatus(StrEnum)
```

How an effect completed.

| Member | Value | Description |
|---|---|---|
| `OK` | `'ok'` |  |
| `FAILED` | `'failed'` |  |
| `OUTCOME_UNKNOWN` | `'outcome_unknown'` | A possible duplicate of a side effect that could not be deduplicated; never silently retried. |

### `EMPTY_DIGEST`

*constant* · `src/rollout/core/contracts/digests.py`

```python
EMPTY_DIGEST = hashlib.sha256(b'').hexdigest()
```

`d₀` of every context digest chain.

### `FinishReason`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class FinishReason(StrEnum)
```

Why a sample stopped.

| Member | Value | Description |
|---|---|---|
| `STOP` | `'stop'` |  |
| `LENGTH` | `'length'` |  |
| `TOOL_USE` | `'tool_use'` |  |
| `CONTENT_FILTER` | `'content_filter'` |  |

### `FrozenSequence`

*type alias* · `src/rollout/core/contracts/base.py`

```python
type FrozenSequence[T] = Annotated[Sequence[T], AfterValidator(tuple)]
```

A sequence field that accepts any sequence and is stored as a tuple, so contract values stay immutable.

### `InternalError`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class InternalError(ModelEndpointError)
```

The endpoint failed.

### `Media`

*class* · `src/rollout/core/contracts/content.py`

```python
class Media(ContractModel)
```

An image, audio clip or document.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['media']` | `'media'` |  |
| `media_type` | `str` | required |  |
| `source` | `BlobReference` | required |  |

### `Message`

*class* · `src/rollout/core/contracts/content.py`

```python
class Message(ContractModel)
```

A message in canonical form: a role and a sequence of content blocks.

TOOL messages contain only `ToolResultBlock`s, and `ToolCall`s appear only in ASSISTANT messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `Role` | required |  |
| `content` | `FrozenSequence[Block]` | `()` |  |
| `name` | `str \| None` | `None` | Optional speaker name (multi-agent). |
| `meta` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Not model-visible; never rendered and not covered by the context digest. |

**Methods**

- `@property def text(self) -> str` — The concatenated text blocks.
- `@property def tool_calls(self) -> list[ToolCall]` — The tool calls, in order.
- `@classmethod def user(cls, text: str) -> 'Message'` — A USER message with one text block.
- `@classmethod def assistant(cls, text: str) -> 'Message'` — An ASSISTANT message with one text block.
- `@classmethod def system(cls, text: str) -> 'Message'` — A SYSTEM message with one text block.

### `message_digest`

*function* · `src/rollout/core/contracts/digests.py`

```python
def message_digest(message: Message) -> str
```

Covers what the model can see: `meta` is excluded.

### `ModelEndpoint`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ModelEndpoint(Protocol)
```

Serves model slots: implemented by the recorder and by direct adapters.

**Methods**

- `def describe(self, session_id: str) -> CapabilityContract` — The capability contract of the session's model slot.
- `async def sample(self, request: SampleRequest) -> SampleResult` — One reply. Where the endpoint deduplicates, a repeated `effect_id` returns the recorded result.
- `async def cancel(self, effect_id: str) -> None` — Best-effort.

### `ModelEndpointError`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ModelEndpointError(Exception)
```

Errors an endpoint raises; see the table in the contract for how the core handles each.

### `NamedToolChoice`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class NamedToolChoice(ContractModel)
```

The model must call this tool.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |

### `NeedFullContext`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class NeedFullContext(ModelEndpointError)
```

The delta's parent is unknown to the endpoint; the core resends the full context.

### `new_job_id`

*function* · `src/rollout/core/contracts/identifiers.py`

```python
def new_job_id() -> str
```

`j_{ulid}`.

### `new_run_id`

*function* · `src/rollout/core/contracts/identifiers.py`

```python
def new_run_id() -> str
```

`r_{ulid}`.

### `new_ulid`

*function* · `src/rollout/core/contracts/identifiers.py`

```python
def new_ulid() -> str
```

A ULID: 48 bits of Unix time in milliseconds, then 80 random bits, in Crockford base 32 (26 characters).

Minted by runners and services, never by task code (which has no ambient randomness under a durable runner).

### `Overloaded`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class Overloaded(ModelEndpointError)
```

Admission control: retry after `retry_after` seconds.

**Methods**

- `def __init__(self, retry_after: float | None = None) -> None`

### `Provenance`

*class* · `src/rollout/core/contracts/content.py`

```python
class Provenance(ContractModel)
```

Where a tool result came from.

| Field | Type | Default | Description |
|---|---|---|---|
| `untrusted` | `bool` | `False` | True for anything originating in a guest or a third-party server. |
| `binding_kind` | `str \| None` | `None` | `task`, `environment`, `mcp`, `http`, `agent` or `human`. |

### `Reasoning`

*class* · `src/rollout/core/contracts/content.py`

```python
class Reasoning(ContractModel)
```

The model's reasoning. `POLICY`-scoped reasoning is handled by the recorder; code never needs to.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['reasoning']` | `'reasoning'` |  |
| `scope` | `ReasoningScope` | required |  |
| `text` | `str \| None` | `None` | `PORTABLE` only. |
| `producer` | `str \| None` | `None` | `POLICY` only: the renderer or provider that can consume `opaque`. |
| `opaque` | `BlobReference \| None` | `None` | `POLICY` only. |

### `ReasoningScope`

*class* · `src/rollout/core/contracts/content.py`

```python
class ReasoningScope(StrEnum)
```

Who can consume a reasoning block.

| Member | Value | Description |
|---|---|---|
| `PORTABLE` | `'portable'` | Plain text that any renderer may render or drop. |
| `POLICY` | `'policy'` | Opaque, valid only for its producer (e.g. encrypted provider reasoning). |

### `ReasoningSupport`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ReasoningSupport(StrEnum)
```

Which kinds of reasoning blocks a model slot produces.

| Member | Value | Description |
|---|---|---|
| `NONE` | `'none'` |  |
| `PORTABLE` | `'portable'` |  |
| `POLICY_SCOPED` | `'policy_scoped'` |  |

### `ResultBlock`

*type alias* · `src/rollout/core/contracts/content.py`

```python
type ResultBlock = Annotated[Text | Media, Field(discriminator='type')]
```

### `RetryClass`

*class* · `src/rollout/core/contracts/content.py`

```python
class RetryClass(StrEnum)
```

What a durable runner may do with a tool call after a crash (docs/architecture/delivery-semantics.md).

| Member | Value | Description |
|---|---|---|
| `PURE` | `'pure'` |  |
| `IDEMPOTENT` | `'idempotent'` |  |
| `SIDE_EFFECTING` | `'side_effecting'` |  |
| `UNKNOWN` | `'unknown'` |  |

### `Role`

*class* · `src/rollout/core/contracts/content.py`

```python
class Role(StrEnum)
```

Who a message is from. Observations contain only USER and TOOL messages.

| Member | Value | Description |
|---|---|---|
| `SYSTEM` | `'system'` |  |
| `USER` | `'user'` |  |
| `ASSISTANT` | `'assistant'` |  |
| `TOOL` | `'tool'` |  |

### `RUN_EVENT_SCHEMA_VERSION`

*constant* · `src/rollout/core/contracts/events.py`

```python
RUN_EVENT_SCHEMA_VERSION = 1
```

### `RunEvent`

*class* · `src/rollout/core/contracts/events.py`

```python
class RunEvent(ContractModel)
```

One entry in a run's event stream.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required |  |
| `seq` | `int` | required | Position in this run's event stream, gapless from 0; `run.created` is 0. |
| `type` | `RunEventType` | required |  |
| `schema_version` | `int` | `RUN_EVENT_SCHEMA_VERSION` |  |
| `recorded_at` | `datetime` | required | Exposed to code as `run.now()` for inputs. |
| `payload` | `JsonValue` | `None` | Type-specific; see docs/contracts/run-events.md. |

### `RunEventType`

*class* · `src/rollout/core/contracts/events.py`

```python
class RunEventType(StrEnum)
```

The closed catalog of run events.

| Member | Value | Description |
|---|---|---|
| `RUN_CREATED` | `'run.created'` |  |
| `GENERATION_STARTED` | `'generation.started'` |  |
| `RUN_SUSPENDED` | `'run.suspended'` |  |
| `RUN_COMPLETED` | `'run.completed'` |  |
| `RUN_FAILED` | `'run.failed'` |  |
| `RUN_CANCEL_REQUESTED` | `'run.cancel_requested'` |  |
| `RUN_CANCELLED` | `'run.cancelled'` |  |
| `PATCH_MARKED` | `'patch.marked'` |  |
| `OBSERVATION_RECORDED` | `'observation.recorded'` |  |
| `REWARD_ASSIGNED` | `'reward.assigned'` |  |
| `TRAINING_EXCLUDED` | `'training.excluded'` |  |
| `OUTPUT_EMITTED` | `'output.emitted'` |  |
| `EFFECT_REQUESTED` | `'effect.requested'` |  |
| `EFFECT_COMPLETED` | `'effect.completed'` |  |
| `MESSAGE_RECEIVED` | `'message.received'` |  |
| `TURN_INTERRUPTED` | `'turn.interrupted'` |  |
| `TOOLS_RESOLVED` | `'tools.resolved'` |  |
| `TOOLS_CHANGED` | `'tools.changed'` |  |

### `RunFailureClass`

*class* · `src/rollout/core/contracts/events.py`

```python
class RunFailureClass(StrEnum)
```

Why a run failed (the `class` of a `run.failed` event).

| Member | Value | Description |
|---|---|---|
| `TASK_ERROR` | `'task_error'` |  |
| `INVALID_OBSERVATION` | `'invalid_observation'` |  |
| `NON_DETERMINISM` | `'non_determinism'` |  |
| `POISONED` | `'poisoned'` |  |
| `INFRASTRUCTURE` | `'infrastructure'` |  |

### `SampleRequest`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class SampleRequest(ContractModel)
```

A request for one reply. Sampling parameters are not here: they belong to the policy.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | required | Required: the idempotency key. |
| `arguments_digest` | `str` | required |  |
| `session_id` | `str` | required |  |
| `context` | `ContextDelta` | required |  |
| `tools` | `FrozenSequence[ToolSpecification]` | `()` | The subset exposed this turn; endpoints use only the model-visible fields. |
| `max_output_tokens` | `int \| None` | `None` | Must not exceed the contract's `max_output_tokens`. |
| `tool_choice` | `ToolChoice \| None` | `None` |  |
| `deadline` | `datetime \| None` | `None` |  |

### `SampleResult`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class SampleResult(ContractModel)
```

Nothing here identifies the policy, weights version or engine.

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | `Message` | required |  |
| `finish_reason` | `FinishReason` | required |  |
| `usage` | `Usage` | required |  |

### `session_id`

*function* · `src/rollout/core/contracts/identifiers.py`

```python
def session_id(run_id: str, model_slot: str) -> str
```

`{run_id}/{model_slot}`: one recorder session per model slot per run.

### `SessionIdentity`

*class* · `src/rollout/core/contracts/identifiers.py`

```python
class SessionIdentity
```

The parts of a `session_id`: `{run_id}/{model_slot}` for runs, `u_{ulid}/{model_slot}` when unmanaged.

| Field | Type | Default | Description |
|---|---|---|---|
| `owner` | `str` | required |  |
| `model_slot` | `str` | required |  |

**Methods**

- `@classmethod def parse(cls, session_id: str) -> 'SessionIdentity'`

### `spec_hash`

*function* · `src/rollout/core/contracts/digests.py`

```python
def spec_hash(specification: ToolSpecification) -> str
```

Covers only the model-visible fields of a tool specification.

### `TERMINAL_EVENT_TYPES`

*constant* · `src/rollout/core/contracts/events.py`

```python
TERMINAL_EVENT_TYPES = frozenset({RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED, RunEventType.RUN_CANCELLED})
```

### `Text`

*class* · `src/rollout/core/contracts/content.py`

```python
class Text(ContractModel)
```

Plain text.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['text']` | `'text'` |  |
| `text` | `str` | required |  |

### `ToolAnnotations`

*class* · `src/rollout/core/contracts/content.py`

```python
class ToolAnnotations(ContractModel)
```

MCP-compatible hints; not model-visible, and ignored from untrusted servers.

| Field | Type | Default | Description |
|---|---|---|---|
| `title` | `str \| None` | `None` |  |
| `read_only_hint` | `bool \| None` | `None` |  |
| `destructive_hint` | `bool \| None` | `None` |  |
| `idempotent_hint` | `bool \| None` | `None` |  |
| `open_world_hint` | `bool \| None` | `None` |  |

### `ToolCall`

*class* · `src/rollout/core/contracts/content.py`

```python
class ToolCall(ContractModel)
```

A request by the model to call a tool. Appears only in ASSISTANT messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['tool_call']` | `'tool_call'` |  |
| `call_id` | `str` | required | Unique within the context; the `ToolResultBlock` that answers the call repeats it. |
| `name` | `str` | required |  |
| `arguments` | `Mapping[str, JsonValue]` | required | A JSON object. |

### `ToolChoice`

*type alias* · `src/rollout/core/contracts/model_endpoint.py`

```python
type ToolChoice = ToolChoiceMode | NamedToolChoice
```

### `ToolChoiceMode`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class ToolChoiceMode(StrEnum)
```

Whether the model may, must not, or must call a tool.

| Member | Value | Description |
|---|---|---|
| `AUTO` | `'auto'` |  |
| `NONE` | `'none'` |  |
| `REQUIRED` | `'required'` |  |

### `ToolResult`

*class* · `src/rollout/core/contracts/content.py`

```python
class ToolResult(ContractModel)
```

What a tool produces. Platform failures are `tool.failed` events, not results.

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `FrozenSequence[ResultBlock]` | `()` |  |
| `structured` | `JsonValue` | `None` | Optional; must validate against the tool's `output_schema` when it has one. |
| `is_error` | `bool` | `False` | A tool-level error the model should see and reason about (non-zero exit, file not found). |
| `truncated` | `bool` | `False` | Content was cut to `max_result_bytes`; the full output is in `overflow`. |
| `overflow` | `BlobReference \| None` | `None` |  |
| `provenance` | `Provenance` | `Provenance()` |  |

### `ToolResultBlock`

*class* · `src/rollout/core/contracts/content.py`

```python
class ToolResultBlock(ContractModel)
```

A tool's result as it appears in a TOOL message, answering the `ToolCall` with the same `call_id`.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['tool_result']` | `'tool_result'` |  |
| `call_id` | `str` | required |  |
| `result` | `ToolResult` | required |  |

### `ToolSpecification`

*class* · `src/rollout/core/contracts/content.py`

```python
class ToolSpecification(ContractModel)
```

What the model sees about a tool, plus extensions that are never model-visible.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `description` | `str` | `''` |  |
| `input_schema` | `Mapping[str, JsonValue]` | `Field(default_factory=lambda: {'type': 'object', 'properties': {}})` | JSON Schema 2020-12 with `type: object`. |
| `output_schema` | `Mapping[str, JsonValue] \| None` | `None` |  |
| `annotations` | `ToolAnnotations \| None` | `None` |  |
| `retry_class` | `RetryClass` | `RetryClass.UNKNOWN` |  |
| `timeout_ms` | `int \| None` | `None` |  |
| `max_result_bytes` | `int \| None` | `None` |  |

**Methods**

- `def model_visible(self) -> dict[str, JsonValue]` — The fields a model sees and the spec hash covers; absent fields are omitted.

### `Usage`

*class* · `src/rollout/core/contracts/model_endpoint.py`

```python
class Usage(ContractModel)
```

Context use after a sample.

| Field | Type | Default | Description |
|---|---|---|---|
| `context_used` | `int` | required | Drives the agent's compaction decisions. |
| `context_limit` | `int` | required |  |
| `input_tokens` | `int \| None` | `None` |  |
| `output_tokens` | `int \| None` | `None` |  |

## `rollout.core.local`

In-process implementations for the local profile.

### `EndpointFactory`

*type alias* · `src/rollout/core/local/runner.py`

```python
type EndpointFactory = Callable[[DirectModel], ModelEndpoint]
```

Creates the endpoint for a direct model binding; registered with the runner by provider name.

### `LocalRunContext`

*class* · `src/rollout/core/local/context.py`

```python
class LocalRunContext
```

Implements `RunContext` and `Effects` in process.

**Methods**

- `def __init__(self, run_id: str, endpoints: Mapping[str, ModelEndpoint], *, context_hints: ContextHints | None = None, conversation: ConversationKey | None = None, generation: int = 0, on_event: Callable[[RunEvent], None] | None = None) -> None`
- `@property def run_id(self) -> str`
- `@property def conversation(self) -> ConversationKey | None`
- `@property def turn(self) -> int`
- `@property def history(self) -> History`
- `@property def models(self) -> Mapping[str, Model]`
- `@property def model(self) -> Model`
- `@property def random(self) -> random.Random`
- `@property def context_hints(self) -> ContextHints`
- `def now(self) -> datetime`
- `def reward(self, value: float, *, slot: str = 'policy', key: str = 'default') -> None`
- `def exclude_from_training(self, reason: str) -> None`
- `async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]`
- `def patched(self, change_id: str) -> bool`
- `async def emit(self, kind: str, payload: JsonValue, *, to: Address | None = None) -> None` — Durable output, e.g. a reply that a connector delivers. Recorded as an `output.emit` effect.
- `def record(self, observation: Observation | WaitFor, *, reply: Message | None = None) -> None`
- `async def wait_for_message(self, wait: WaitFor) -> Envelope | None`
- `def take_steering_messages(self) -> list[Envelope]`
- `async def interruptible[T](self, reply: Awaitable[T]) -> T`
- `def take_undelivered(self) -> list[Envelope]` — Messages the run never consumed; the runner hands them to the conversation's next run.
- `def deliver(self, envelope: Envelope, mode: DeliveryMode) -> None` — Deliver a message to this run (docs/core/harness/conversations.md#priority-and-delivery-mode).
- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue]) -> T`
- `def record_event(self, event_type: RunEventType, payload: JsonValue) -> RunEvent`

### `LocalRunHandle`

*class* · `src/rollout/core/local/runner.py`

```python
class LocalRunHandle
```

A run started by a `LocalRunner`. Its context is available for inspection in tests and tools.

**Methods**

- `def __init__(self, run_id: str, specification: RunSpecification, conversation: ConversationKey | None) -> None`
- `@property def run_id(self) -> str`
- `@property def done(self) -> bool`
- `@property def outcome(self) -> RunOutcome | None`
- `async def result(self) -> RunOutcome`
- `async def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]`
- `@property def task(self) -> asyncio.Task[None] | None`
- `def attach(self, task: asyncio.Task[None]) -> None`
- `def finish(self, outcome: RunOutcome) -> None`
- `def notify(self, event: RunEvent | None = None) -> None` — Wake event streams: a new event was recorded, or the run ended.

### `LocalRunner`

*class* · `src/rollout/core/local/runner.py`

```python
class LocalRunner
```

Implements `Runner` in process.

Direct model bindings are served by endpoint factories registered by provider name. Conversations addressed to a
deployment start a run of its specification when none is live; one run consumes a conversation's messages at a
time, and messages a run never consumed start the conversation's next run.

**Methods**

- `def __init__(self, *, providers: Mapping[str, EndpointFactory] | None = None) -> None`
- `def deploy(self, deployment: Deployment) -> None` — Register or replace a deployment; a conversation's next run uses the current version.
- `def run(self, run_id: str) -> LocalRunHandle`
- `def conversation_runs(self, deployment: str, key: str) -> list[LocalRunHandle]` — The conversation's runs, oldest first.
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None) -> LocalRunHandle`
- `async def send(self, to: Address, envelope: Envelope, *, priority: Priority = Priority.NORMAL, idempotency_key: str | None = None, sender: str | None = None) -> None`
- `async def cancel(self, run_id: str, *, reason: str) -> None`

### `RewardAssignment`

*class* · `src/rollout/core/local/context.py`

```python
class RewardAssignment
```

| Field | Type | Default | Description |
|---|---|---|---|
| `slot` | `str` | required |  |
| `value` | `float` | required |  |
| `key` | `str` | required |  |

### `RunNotLive`

*class* · `src/rollout/core/local/runner.py`

```python
class RunNotLive(Exception)
```

A message was addressed to a run that has ended.

## `rollout.core.testing`

Test doubles: a scripted model endpoint and helpers.

### `events_of`

*function* · `src/rollout/core/testing.py`

```python
def events_of(run: LocalRunContext, event_type: RunEventType) -> list[RunEvent]
```

The run's events of one type, in order.

### `local_run`

*function* · `src/rollout/core/testing.py`

```python
def local_run(task: Task, replies: Iterable[ScriptedReply] = ()) -> tuple[LocalRunContext, ScriptedModelEndpoint]
```

A local run context for `task` whose model slots all reply from one script.

### `payload`

*function* · `src/rollout/core/testing.py`

```python
def payload(event: RunEvent) -> dict[str, JsonValue]
```

An event's payload as a JSON object.

### `ScriptedModelEndpoint`

*class* · `src/rollout/core/testing.py`

```python
class ScriptedModelEndpoint
```

Replies with the scripted entries in order and records every request and cancellation.

**Methods**

- `def __init__(self, replies: Iterable[ScriptedReply], *, contract: CapabilityContract | None = None) -> None`
- `def describe(self, session_id: str) -> CapabilityContract`
- `async def sample(self, request: SampleRequest) -> SampleResult`
- `async def cancel(self, effect_id: str) -> None`

### `ScriptedReply`

*type alias* · `src/rollout/core/testing.py`

```python
type ScriptedReply = Message | str | Callable[[SampleRequest], Message | Awaitable[Message]]
```

A reply, its text, or a function of the request (which may await, e.g. to hold a sample open).

### `tool_call_reply`

*function* · `src/rollout/core/testing.py`

```python
def tool_call_reply(*calls: ToolCall, text: str = '') -> Message
```

An assistant reply that makes tool calls.
