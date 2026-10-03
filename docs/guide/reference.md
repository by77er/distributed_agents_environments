# API reference

Generated from the source by `scripts/generate_reference.py`; do not edit by hand. Every public name,
grouped by module, alphabetically. Types and defaults appear as written in the source. The
[guide](README.md) explains how the pieces fit together.

## Contents

- **[`rollout.harness`](#rolloutharness)** — Writing tasks, agents and programs; runners; memory; tool sets. [`Address`](#address), [`Agent`](#agent), [`agent_program`](#agent_program), [`AgentProgram`](#agentprogram), [`bind`](#bind), [`Blobs`](#blobs), [`CompactingAgent`](#compactingagent), [`ContextHints`](#contexthints), [`ConversationKey`](#conversationkey), [`DeduplicatingToolSet`](#deduplicatingtoolset), [`DeliveryMode`](#deliverymode), [`DeliveryPolicy`](#deliverypolicy), [`Deployment`](#deployment), [`DirectModel`](#directmodel), [`Effects`](#effects), [`End`](#end), [`Ending`](#ending), [`EndpointModel`](#endpointmodel), [`Envelope`](#envelope), [`Environment`](#environment), [`Environments`](#environments), [`EnvironmentService`](#environmentservice), [`EnvironmentSpecification`](#environmentspecification), [`ExecutionResult`](#executionresult), [`FileBlobStore`](#fileblobstore), [`History`](#history), [`HistoryShape`](#historyshape), [`instantiate`](#instantiate), [`Interrupted`](#interrupted), [`InvalidObservation`](#invalidobservation), [`Memory`](#memory), [`MessageRouter`](#messagerouter), [`Model`](#model), [`ModelBinding`](#modelbinding), [`ModelSample`](#modelsample), [`ModelSlot`](#modelslot), [`Observation`](#observation), [`Priority`](#priority), [`Program`](#program), [`ProgramReference`](#programreference), [`RecordedEndpoints`](#recordedendpoints), [`RecordedModel`](#recordedmodel), [`register`](#register), [`resolve`](#resolve), [`rollout`](#rollout), [`RunBinding`](#runbinding), [`RunContext`](#runcontext), [`RunHandle`](#runhandle), [`RunHooks`](#runhooks), [`Runner`](#runner), [`RunNotLive`](#runnotlive), [`RunOutcome`](#runoutcome), [`RunSpecification`](#runspecification), [`RunStatus`](#runstatus), [`SamplingParameters`](#samplingparameters), [`Task`](#task), [`tool`](#tool), [`ToolBinding`](#toolbinding), [`Tools`](#tools), [`ToolSet`](#toolset), [`Turn`](#turn), [`WaitFor`](#waitfor), [`with_row`](#with_row)
- **[`rollout.contracts`](#rolloutcontracts)** — Types that cross layers: canonical content, identifiers, digests, effects, events. [`address_of`](#address_of), [`AddressableEndpoint`](#addressableendpoint), [`arguments_digest`](#arguments_digest), [`BlobReference`](#blobreference), [`Block`](#block), [`canonical_json`](#canonical_json), [`CapabilityContract`](#capabilitycontract), [`Conflict`](#conflict), [`context_digests`](#context_digests), [`ContextDelta`](#contextdelta), [`ContextOverflow`](#contextoverflow), [`ContractModel`](#contractmodel), [`ContractViolation`](#contractviolation), [`digest`](#digest), [`effect_id`](#effect_id), [`EffectIdentity`](#effectidentity), [`EffectKind`](#effectkind), [`EffectStatus`](#effectstatus), [`EMPTY_DIGEST`](#empty_digest), [`FinishReason`](#finishreason), [`FrozenSequence`](#frozensequence), [`InternalError`](#internalerror), [`Media`](#media), [`Message`](#message), [`message_digest`](#message_digest), [`ModelAddress`](#modeladdress), [`ModelEndpoint`](#modelendpoint), [`ModelEndpointError`](#modelendpointerror), [`NamedToolChoice`](#namedtoolchoice), [`new_message_id`](#new_message_id), [`new_run_id`](#new_run_id), [`new_ulid`](#new_ulid), [`OutcomeUnknown`](#outcomeunknown), [`Overloaded`](#overloaded), [`Reasoning`](#reasoning), [`ReasoningScope`](#reasoningscope), [`ResultBlock`](#resultblock), [`RetryClass`](#retryclass), [`Role`](#role), [`RUN_EVENT_SCHEMA_VERSION`](#run_event_schema_version), [`RunEvent`](#runevent), [`RunEventType`](#runeventtype), [`RunFailureClass`](#runfailureclass), [`SampleRequest`](#samplerequest), [`SampleResult`](#sampleresult), [`session_id`](#session_id), [`SessionIdentity`](#sessionidentity), [`spec_hash`](#spec_hash), [`TERMINAL_EVENT_TYPES`](#terminal_event_types), [`Text`](#text), [`ToolCall`](#toolcall), [`ToolChoice`](#toolchoice), [`ToolChoiceMode`](#toolchoicemode), [`ToolResult`](#toolresult), [`ToolResultBlock`](#toolresultblock), [`ToolSpecification`](#toolspecification), [`Usage`](#usage)
- **[`rollout.catalog`](#rolloutcatalog)** — What an environment offers to be trained on. [`binding_for`](#binding_for), [`Catalog`](#catalog), [`Row`](#row)
- **[`rollout.local`](#rolloutlocal)** — The runner in this process. [`EndpointFactory`](#endpointfactory), [`LocalRunContext`](#localruncontext), [`LocalRunHandle`](#localrunhandle), [`LocalRunner`](#localrunner), [`RewardAssignment`](#rewardassignment)
- **[`rollout.testing`](#rollouttesting)** — Test doubles: a scripted model endpoint and helpers. [`events_of`](#events_of), [`LedgerEndpoint`](#ledgerendpoint), [`LedgerEnvironments`](#ledgerenvironments), [`local_run`](#local_run), [`payload`](#payload), [`read_ledger`](#read_ledger), [`ScriptedModelEndpoint`](#scriptedmodelendpoint), [`ScriptedReply`](#scriptedreply), [`tool_call_reply`](#tool_call_reply)
- **[`rollout_train.rollouts`](#rollout_trainrollouts)** — Rollout jobs: rows in, episodes out, weights published. [`Episode`](#episode), [`events_of`](#events_of), [`Job`](#job), [`JobHooks`](#jobhooks), [`Jobs`](#jobs), [`loaded`](#loaded), [`Outcome`](#outcome), [`Record`](#record), [`Recorded`](#recorded), [`Refused`](#refused), [`RolloutJob`](#rolloutjob), [`RolloutJobs`](#rolloutjobs), [`RolloutTicket`](#rolloutticket), [`Status`](#status), [`stored`](#stored), [`Ticket`](#ticket), [`Trajectory`](#trajectory)
- **[`rollout_train`](#rollout_train)** — The training loop, the group algorithm, the curriculum, and what they ask of a trainer. [`Algorithm`](#algorithm), [`Batch`](#batch), [`Budget`](#budget), [`Checkpoint`](#checkpoint), [`Colocated`](#colocated), [`complete_groups`](#complete_groups), [`Curriculum`](#curriculum), [`Fence`](#fence), [`Fenced`](#fenced), [`FileLedger`](#fileledger), [`group_advantages`](#group_advantages), [`Grpo`](#grpo), [`Ledger`](#ledger), [`Manifest`](#manifest), [`Policies`](#policies), [`Result`](#result), [`results`](#results), [`Step`](#step), [`StepFailed`](#stepfailed), [`train`](#train), [`Trained`](#trained), [`trained`](#trained), [`Trainer`](#trainer), [`Version`](#version), [`Weighted`](#weighted)
- **[`rollout_train.inference`](#rollout_traininference)** — Channels: policies being served, and what they ask of an engine. [`Channel`](#channel), [`Engine`](#engine), [`Generation`](#generation), [`Limits`](#limits)
- **[`rollout_train.recorder`](#rollout_trainrecorder)** — The model endpoint for trainable channels: token-exact recording. [`ChatTemplateRenderer`](#chattemplaterenderer), [`JsonToolCalls`](#jsontoolcalls), [`RecordedEndpoint`](#recordedendpoint), [`Recorder`](#recorder), [`Renderer`](#renderer), [`Segment`](#segment), [`Span`](#span), [`ThinkingFormat`](#thinkingformat), [`ToolCallFormat`](#toolcallformat), [`XmlFunctionCalls`](#xmlfunctioncalls)
- **[`rollout_train.profile`](#rollout_trainprofile)** — A deployment, described and opened. [`ChannelSpec`](#channelspec), [`NotEnoughMemory`](#notenoughmemory), [`Platform`](#platform), [`Profile`](#profile), [`TrainerSpec`](#trainerspec)
- **[`rollout_train.monitor`](#rollout_trainmonitor)** — A live web page over a job and its runs. [`FeedReader`](#feedreader), [`plain`](#plain), [`RunFeed`](#runfeed), [`System`](#system)
- **[`rollout_train.testing`](#rollout_traintesting)** — Test doubles: a scripted engine and a readable token format. [`Characters`](#characters), [`plain_channel`](#plain_channel), [`plain_renderer`](#plain_renderer), [`PlainRenderer`](#plainrenderer), [`sample_request`](#sample_request), [`scripted_engine`](#scripted_engine), [`ScriptedEngine`](#scriptedengine)
- **[`rollout_durable`](#rollout_durable)** — A runner whose runs survive their process, on DBOS. [`DurableRunContext`](#durableruncontext), [`DurableRunHandle`](#durablerunhandle), [`DurableRunner`](#durablerunner), [`RunCancelled`](#runcancelled), [`RunStore`](#runstore)
- **[`rollout_vllm`](#rollout_vllm)** — An engine on vLLM. [`VllmEngine`](#vllmengine)
- **[`rollout_lora`](#rollout_lora)** — A trainer for 4-bit checkpoints with LoRA. [`LoraSettings`](#lorasettings), [`LoraTrainer`](#loratrainer)
- **[`rollout_qwen`](#rollout_qwen)** — Renderers for the Qwen model families. [`qwen3`](#qwen3), [`qwen35`](#qwen35), [`tokenizer_of`](#tokenizer_of)
- **[`rollout_gemma`](#rollout_gemma)** — Renderers for the Gemma model families. [`arguments`](#arguments), [`gemma4`](#gemma4), [`GemmaFunctionCalls`](#gemmafunctioncalls), [`tokenizer_of`](#tokenizer_of)
- **[`rollout_computers`](#rollout_computers)** — Environment backends: services that give runs computers. [`ImageStore`](#imagestore), [`LocalEnvironments`](#localenvironments), [`NamespaceEnvironments`](#namespaceenvironments)
- **[`rollout_computers.tools`](#rollout_computerstools)** — Tools for agents that work on a computer: shell, files, edits and images. [`apply_edits`](#apply_edits), [`ComputerTools`](#computertools), [`page_text`](#page_text), [`prepare_image`](#prepare_image), [`Replacement`](#replacement)
- **[`rollout_openai`](#rollout_openai)** — A model endpoint for the OpenAI Responses API, on an API key or a Codex login. [`ApiKey`](#apikey), [`codex_provider`](#codex_provider), [`CodexLogin`](#codexlogin), [`Credentials`](#credentials), [`ResponsesContract`](#responsescontract), [`ResponsesEndpoint`](#responsesendpoint)
- **[`rollout_s3`](#rollout_s3)** — Blobs in S3 or any S3-compatible object store. [`S3BlobStore`](#s3blobstore)

## `rollout.harness`

Writing tasks, agents and programs; runners; memory; tool sets.

### `Address`

*class* · `libraries/rollout/src/rollout/harness/conversations.py`

```python
class Address(ContractModel)
```

Where a message or output goes.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `Literal['conversation', 'run', 'external']` | required |  |
| `value` | `str` | required | `{deployment}/{key}`, a `run_id`, or a connector target. |

### `Agent`

*class* · `libraries/rollout/src/rollout/harness/agent.py`

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

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def agent_program(task: type[Task], agent: type[Agent] = Agent, *, task_parameters: JsonValue = None, agent_configuration: JsonValue = None) -> ProgramReference
```

A reference to the task loop for `task` and `agent`.

### `AgentProgram`

*class* · `libraries/rollout/src/rollout/harness/program.py`

```python
class AgentProgram(Program)
```

The task loop: a task and an agent.

**Methods**

- `def __init__(self, task: Task, agent: Agent) -> None`
- `def model_slots(self) -> Mapping[str, ModelSlot]`
- `def context_hints(self) -> ContextHints`
- `def imports(self) -> list[str]`
- `def tool_specifications(self) -> list[ToolSpecification]`
- `async def main(self, run: RunContext) -> None`

### `bind`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def bind(reference: ProgramReference, channel: str, *, tools: Mapping[str, ToolBinding] | None = None) -> RunBinding
```

A binding that serves every model slot of a program from one recorded channel, and each of its imports from
the tool set registered under the import's own name (or as `tools` says).

### `Blobs`

*class* · `libraries/rollout/src/rollout/harness/blobs.py`

```python
class Blobs(Protocol)
```

**Methods**

- `async def put(self, data: bytes, media_type: str) -> BlobReference` — Store bytes, or find them already stored; either way return their reference.
- `async def read(self, reference: BlobReference) -> bytes`
- `async def delete(self, reference: BlobReference) -> None` — Remove a blob if it is there. Whoever stored the same bytes holds the same blob: delete only what nothing
  else names.

### `CompactingAgent`

*class* · `libraries/rollout/src/rollout/harness/memory.py`

```python
class CompactingAgent(Agent)
```

The default agent, with a memory that fits: one sample of the policy slot per turn, over the recent turns and
the agent's own summary of the older ones.

| Field | Type | Default | Description |
|---|---|---|---|
| `compact_prompt` | `str` | `PROMPT` |  |

**Methods**

- `def __init__(self, configuration: object = None) -> None`
- `async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message`

### `ContextHints`

*class* · `libraries/rollout/src/rollout/harness/history.py`

```python
class ContextHints
```

Advisory for the agent: how much history the task needs the model to see.

| Field | Type | Default | Description |
|---|---|---|---|
| `history` | `HistoryShape` | `HistoryShape.FULL` |  |
| `window` | `int \| None` | `None` | For `WINDOW`: the number of recent turns. |

### `ConversationKey`

*class* · `libraries/rollout/src/rollout/harness/conversations.py`

```python
class ConversationKey(ContractModel)
```

Identifies a conversation: a deployment and a caller-chosen key. One live run per conversation.

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment` | `str` | required | e.g. `acme/support-bot`. |
| `key` | `str` | required | Caller-chosen, e.g. `slack:T1/C2/171.2` or `user:42`. |
| `origin` | `Address \| None` | `None` | Where replies go by default (e.g. a connector target). |

**Methods**

- `@property def address(self) -> str` — `{deployment}/{key}`: the value of the conversation's `Address`.
- `@classmethod def parse(cls, address: str, *, origin: Address | None = None) -> 'ConversationKey'` — The conversation an address value names. A deployment is `{namespace}/{name}`; the rest is the key, which
  may itself contain `/`.

### `DeduplicatingToolSet`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class DeduplicatingToolSet(ToolSet, Protocol)
```

A tool set that says whether it performs each `effect_id` at most once (a `deduplicates = True` attribute, on
a class). The side-effecting tools of one that does are re-executed after a crash rather than guarded
(docs/libraries/rollout/contracts/effects.md).

**Methods**

- `@property def deduplicates(self) -> bool`

### `DeliveryMode`

*class* · `libraries/rollout/src/rollout/harness/conversations.py`

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

*class* · `libraries/rollout/src/rollout/harness/conversations.py`

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

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class Deployment(ContractModel)
```

A named, addressable agent: conversations addressed to it start runs of its specification.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | `{namespace}/{name}`, e.g. `acme/support-bot`. |
| `specification` | `RunSpecification` | required |  |

### `DirectModel`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

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

*class* · `libraries/rollout/src/rollout/harness/model.py`

```python
class Effects(Protocol)
```

How a run performs effects. The runner decides what performing means: a direct call, or a durable step.

**Methods**

- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue], guard: bool = False) -> T` — Assign the next `effect_id`, digest `arguments`, and run `execute(effect_id, arguments_digest)`.
  
    `completion` renders the result for the run's events. `guard` marks a side effect whose receiver cannot
  deduplicate: a durable runner performs it at most once, raising `OutcomeUnknown` after a crash interrupted it.

### `End`

*function* · `libraries/rollout/src/rollout/harness/observation.py`

```python
def End(reward: float | None = None, *, truncated: bool = False, info: Mapping[str, Any] | None = None) -> Observation
```

A terminal observation.

### `Ending`

*class* · `libraries/rollout/src/rollout/harness/observation.py`

```python
class Ending(StrEnum)
```

How an episode ended.

| Member | Value | Description |
|---|---|---|
| `TERMINATED` | `'terminated'` | A real end state (value methods do not bootstrap). |
| `TRUNCATED` | `'truncated'` | Stopped by a limit: turns, time, budget (value methods may bootstrap). |

### `EndpointModel`

*class* · `libraries/rollout/src/rollout/harness/model.py`

```python
class EndpointModel
```

A model slot bound to an endpoint. Every sample sends the full context.

**Methods**

- `def __init__(self, endpoint: ModelEndpoint, session_id: str, effects: Effects, *, retries: int = 3, backoff: float = 1.0) -> None`
- `@property def capabilities(self) -> CapabilityContract`
- `@property def usage(self) -> Usage | None`
- `def address(self) -> ModelAddress`
- `async def sample(self, messages: Sequence[Message], *, tools: Sequence[ToolSpecification] = (), max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None) -> Message`

### `Envelope`

*class* · `libraries/rollout/src/rollout/harness/conversations.py`

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
| `message_id` | `str` | `''` | Set by the runner: the caller's idempotency key (a sending run's `effect_id`, say), or a new `m_{ulid}`. |
| `sender` | `str \| None` | `None` | Set by the runner; never trusted from the payload. |

### `Environment`

*class* · `libraries/rollout/src/rollout/harness/environments.py`

```python
class Environment
```

A handle to one environment. Its methods are effects.

**Methods**

- `def __init__(self, environment_id: str, service: EnvironmentService, effects: Effects) -> None`
- `async def execute(self, command: str, *, timeout_seconds: float = 120.0, cwd: str | None = None) -> ExecutionResult` — Run a shell command. Raises `OutcomeUnknown` if a crash interrupted an earlier attempt of this effect.
- `async def put(self, path: str, data: bytes | str) -> None` — Write a file (relative paths are under the working directory).
- `async def get(self, path: str) -> bytes`
- `async def destroy(self) -> None`

### `Environments`

*class* · `libraries/rollout/src/rollout/harness/environments.py`

```python
class Environments
```

`run.environments`: creates environments the run owns.

**Methods**

- `def __init__(self, service: EnvironmentService, effects: Effects) -> None`
- `async def create(self, specification: EnvironmentSpecification | None = None) -> Environment`
- `def attach(self, environment_id: str) -> Environment` — A handle to an existing environment.
- `async def release_all(self) -> None` — Destroy every environment the run created, directly (not as effects); for runners when a run ends.

### `EnvironmentService`

*class* · `libraries/rollout/src/rollout/harness/environments.py`

```python
class EnvironmentService(Protocol)
```

An environment backend. Every method must be safe to repeat with the same arguments, except `execute`.

**Methods**

- `async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None` — Create the environment, or do nothing if it already exists.
- `async def execute(self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = '') -> ExecutionResult` — Run a command. `effect_id` identifies the attempt (it is the same on every re-execution).
- `async def put(self, environment_id: str, path: str, data: bytes) -> None`
- `async def get(self, environment_id: str, path: str) -> bytes`
- `async def destroy(self, environment_id: str) -> None` — Destroy the environment, or do nothing if it is already gone.

### `EnvironmentSpecification`

*class* · `libraries/rollout/src/rollout/harness/environments.py`

```python
class EnvironmentSpecification(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | `str` | `'alpine'` | A base image the backend knows, e.g. `alpine` (latest) or `alpine:3.24.2`. |
| `setup` | `FrozenSequence[str]` | `()` | Shell commands run once at creation: a recipe for identical start states. |

### `ExecutionResult`

*class* · `libraries/rollout/src/rollout/harness/environments.py`

```python
class ExecutionResult(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `exit_code` | `int \| None` | required | None when the command timed out. |
| `output` | `str` | required | Standard output and standard error, interleaved. |
| `truncated` | `bool` | `False` | `output` is only the end of the output (see `full_output_path`). |
| `timed_out` | `bool` | `False` |  |
| `full_output_path` | `str \| None` | `None` | When truncated: where, inside the environment, the full output was saved. |

### `FileBlobStore`

*class* · `libraries/rollout/src/rollout/harness/blobs.py`

```python
class FileBlobStore
```

Implements `Blobs` in a directory: one file per blob, named by its SHA-256.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def put(self, data: bytes, media_type: str) -> BlobReference`
- `async def read(self, reference: BlobReference) -> bytes`
- `async def delete(self, reference: BlobReference) -> None`

### `History`

*class* · `libraries/rollout/src/rollout/harness/history.py`

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

*class* · `libraries/rollout/src/rollout/harness/history.py`

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

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def instantiate(reference: ProgramReference) -> Program
```

Create the program a reference names.

### `Interrupted`

*class* · `libraries/rollout/src/rollout/harness/context.py`

```python
class Interrupted(Exception)
```

The reply in progress was cancelled by a message delivered with mode `INTERRUPT`.

**Methods**

- `def __init__(self, envelope: Envelope, reply_effect_id: str | None) -> None`

### `InvalidObservation`

*class* · `libraries/rollout/src/rollout/harness/observation.py`

```python
class InvalidObservation(Exception)
```

A hook returned an observation that breaks the validation rules; the run fails with `INVALID_OBSERVATION`.

### `Memory`

*class* · `libraries/rollout/src/rollout/harness/memory.py`

```python
class Memory
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | `PROMPT` | What the agent is asked when its oldest turns are compacted into a summary. |
| `remembered` | `str` | `REMEMBERED` | How the summary is shown to it afterwards. |
| `summary` | `str` | `''` |  |
| `turns` | `list[list[Message]]` | `field(default_factory=list[list[Message]])` | Its recent turns, oldest first: each a few messages (what it saw, what it replied, how that went). |
| `compactions` | `int` | `0` |  |

**Methods**

- `def context(self, system: Message | None = None, current: Sequence[Message] = ()) -> list[Message]` — The system prompt, the summary, the remembered turns, and what is in front of the agent now.
- `def remember(self, *messages: Message) -> None` — Add a turn.
- `def answer(self, result: str, *, others: str = 'Not done: only your first call of a turn counts.') -> None` — Close the latest turn with how its reply went: the result of its first tool call (further calls are
  answered with `others`), or, for a reply that called nothing, a note to the agent.
- `def crowded(self, model: Model) -> bool` — Whether one more turn might leave the model less than its full room to reply (by what its last prompt
  took, which the model reports, and by how much a turn has been seen to add).
- `async def compact(self, model: Model, system: Message | None = None, *, keep: int | None = None) -> None` — Replace the oldest turns with what the agent says it needs to remember of them. The newest `keep` stay as
  they are (by default the newest third).
- `async def sample(self, model: Model, *, system: Message | None = None, current: Sequence[Message] = (), tools: Sequence[ToolSpecification] = (), keep: int = 0) -> Message` — One reply to the context. If the model refuses the context as too long, memory is compacted and the reply
  asked for again (the newest `keep` turns are never compacted).

### `MessageRouter`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class MessageRouter(ABC)
```

`Runner.send`, and the hand-over between a conversation's runs, over the transport a runner supplies: where
its runs are, how a message reaches one, and where delivered messages are remembered.

A message is claimed (remembered as delivered, under its `message_id`) only once it is delivered. A send that
fails before that can be retried with the same idempotency key; a retry of a delivered message is dropped.

**Methods**

- `async def send(self, to: Address, envelope: Envelope, *, priority: Priority = Priority.NORMAL, idempotency_key: str | None = None, sender: str | None = None) -> str`

### `Model`

*class* · `libraries/rollout/src/rollout/harness/context.py`

```python
class Model(Protocol)
```

A model slot as code sees it. Nothing here identifies the policy, weights or engine.

**Methods**

- `@property def capabilities(self) -> CapabilityContract`
- `@property def usage(self) -> Usage | None` — Usage reported by the latest sample, if any: drives compaction decisions.
- `async def sample(self, messages: Sequence[Message], *, tools: Sequence[ToolSpecification] = (), max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None) -> Message`
- `def address(self) -> ModelAddress` — For a harness that brings its own loop (a coding agent running inside the environment, say): where it
  reaches this slot's model. Hand it the base URL and key; what it samples there is this slot's, recorded like
  any other sample. Raises if the deployment does not serve models over HTTP.

### `ModelBinding`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class ModelBinding(ContractModel)
```

Exactly one of `direct` or `recorded`.

| Field | Type | Default | Description |
|---|---|---|---|
| `direct` | `DirectModel \| None` | `None` |  |
| `recorded` | `RecordedModel \| None` | `None` |  |

### `ModelSample`

*class* · `libraries/rollout/src/rollout/harness/hooks.py`

```python
class ModelSample
```

One model sample: what a slot's model was sent and what it replied.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required |  |
| `slot` | `str` | required |  |
| `request` | `SampleRequest` | required | `request.context.append` holds the messages sent; `request.tools` the tools offered. |
| `result` | `SampleResult` | required |  |
| `seconds` | `float` | required | How long the endpoint took. |

### `ModelSlot`

*class* · `libraries/rollout/src/rollout/harness/task.py`

```python
class ModelSlot
```

A model the task declares. The agent acts through `policy`; other slots (a simulated user, an opponent) are
sampled by the task itself.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainable` | `bool` | `True` |  |

### `Observation`

*class* · `libraries/rollout/src/rollout/harness/observation.py`

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

*class* · `libraries/rollout/src/rollout/harness/conversations.py`

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

*class* · `libraries/rollout/src/rollout/harness/program.py`

```python
class Program
```

What a run executes. `AgentProgram` is the task loop; plain durable workflows are other programs.

**Methods**

- `def model_slots(self) -> Mapping[str, ModelSlot]` — The model slots the program samples; a runner binds an endpoint to each.
- `def context_hints(self) -> ContextHints`
- `def imports(self) -> list[str]` — The imported tool sets the program needs; the run's binding says how each is served.
- `def tool_specifications(self) -> list[ToolSpecification]` — Tools the program itself defines (`@tool` methods), for the run's `tools.resolved` event.
- `async def main(self, run: RunContext) -> None`

### `ProgramReference`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class ProgramReference(ContractModel)
```

What a run executes, by name, so a runner in another process can re-create it.

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `str` | required | `module:QualifiedName` of a `Program` class. |
| `parameters` | `JsonValue` | `None` |  |

### `RecordedEndpoints`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RecordedEndpoints(Protocol)
```

Serves recorded bindings: the recorder (`rollout_train.recorder.Recorder`), as runners see it.

**Methods**

- `def endpoint(self, binding: RecordedModel) -> ModelEndpoint`

### `RecordedModel`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RecordedModel(ContractModel)
```

A channel served through the recorder.

| Field | Type | Default | Description |
|---|---|---|---|
| `channel` | `str` | required |  |
| `sampling` | `SamplingParameters` | `SamplingParameters()` |  |

### `register`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def register(cls: type) -> str
```

Make a class resolvable by name in this process, even if it cannot be imported (e.g. defined in a script).

### `resolve`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def resolve(name: str) -> type
```

The class a `module:QualifiedName` names: registered in this process, or imported.

### `rollout`

*function* · `libraries/rollout/src/rollout/harness/loop.py`

```python
async def rollout(task: Task, agent: Agent, run: RunContext) -> None
```

Run one episode of `task` with `agent`. Raises `InvalidObservation` or whatever a hook raised.

### `RunBinding`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunBinding(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `models` | `Mapping[str, ModelBinding]` | required | Model slot → how it is served. |
| `imports` | `Mapping[str, ToolBinding]` | `Field(default_factory=dict[str, ToolBinding])` | Import name → how the tool set is served. |
| `delivery` | `DeliveryPolicy` | `DeliveryPolicy()` |  |

### `RunContext`

*class* · `libraries/rollout/src/rollout/harness/context.py`

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
- `@property def tools(self) -> Tools` — Imported tools; each call is a `tool.call` effect.
- `@property def environments(self) -> Environments | None` — Creates environments the run owns; None when the runner has no environment backend.
- `@property def blobs(self) -> Blobs | None` — Stores bytes such as images for `Media` blocks; None when the runner has no blob store.
- `@property def random(self) -> random.Random` — Seeded from `run_id`.
- `@property def context_hints(self) -> ContextHints`
- `def now(self) -> datetime` — The current time. Use it instead of the wall clock, which durable runs cannot replay.
- `def reward(self, value: float, *, slot: str = 'policy', key: str = 'default') -> None` — Assign a reward to a model slot outside an observation (e.g. to an opponent, or several keyed rewards).
- `def exclude_from_training(self, reason: str) -> None` — Mark the run as unsuitable for training, e.g. after an infrastructure fault that is not the policy's.
- `async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]` — Await concurrently, in order. Equivalent to `asyncio.gather`.
- `async def emit(self, kind: str, payload: JsonValue, *, to: Address | None = None) -> None` — Durable output, such as a reply to a person; a connector or client delivers it.
- `def record(self, observation: Observation | WaitFor, *, reply: Message | None = None) -> None` — Append a turn to the history. A `WaitFor` records only the reply it answers.
- `async def wait_for_message(self, wait: WaitFor) -> Envelope | None` — Suspend until a message of `wait.kind` arrives; `None` on timeout.
- `async def take_steering_messages(self) -> list[Envelope]` — Messages delivered with mode `STEER` since the last turn boundary.
- `async def interruptible[T](self, reply: Awaitable[T]) -> T` — Await an agent's reply; raises `Interrupted` if a message with mode `INTERRUPT` arrives meanwhile.

### `RunHandle`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunHandle(Protocol)
```

**Methods**

- `@property def run_id(self) -> str`
- `@property def done(self) -> bool`
- `@property def outcome(self) -> RunOutcome | None` — How the run ended; None while it is live.
- `async def result(self) -> RunOutcome` — Wait for the run to end.
- `def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]` — Every event from `from_seq`, then new ones as they are recorded, until the run ends.
- `def recorded_events(self) -> list[RunEvent]` — Every event recorded so far.

### `RunHooks`

*class* · `libraries/rollout/src/rollout/harness/hooks.py`

```python
class RunHooks
```

Subclass and override what you need; pass instances to a runner (`LocalRunner(hooks=[...])`).

**Methods**

- `def on_event(self, event: RunEvent) -> None` — A run event was recorded (of any run of the runner; `event.run_id` says which).
- `def on_sample(self, sample: ModelSample) -> None` — A model replied.

### `Runner`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class Runner(Protocol)
```

**Methods**

- `async def launch(self) -> None` — Make the runner ready: call it once, before anything else that starts or reaches a run.
- `async def close(self) -> None` — Release what the runner holds; it cannot be used afterwards.
- `def deploy(self, deployment: Deployment) -> None` — Register or replace a deployment; a conversation's next run uses the current version.
- `def run(self, run_id: str) -> RunHandle` — The handle of a run; `KeyError` if the runner does not know it.
- `def conversation_of(self, run_id: str) -> ConversationKey | None` — The conversation a run serves, if any.
- `def conversation_runs(self, deployment: str, key: str) -> Sequence[RunHandle]` — The conversation's runs, oldest first.
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None) -> RunHandle`
- `async def send(self, to: Address, envelope: Envelope, *, priority: Priority = Priority.NORMAL, idempotency_key: str | None = None, sender: str | None = None) -> str` — Deliver a message and return its `message_id`; a message to a conversation starts its run when none
  is live. A message sent again with the same `idempotency_key` is delivered once.
- `async def cancel(self, run_id: str, *, reason: str) -> None`

### `RunNotLive`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunNotLive(Exception)
```

A message was addressed to a run that has ended.

### `RunOutcome`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunOutcome(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `status` | `RunStatus` | required |  |
| `failure_class` | `RunFailureClass \| None` | `None` |  |
| `detail` | `str \| None` | `None` |  |

### `RunSpecification`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunSpecification(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `ProgramReference` | required |  |
| `binding` | `RunBinding` | required |  |

### `RunStatus`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunStatus(StrEnum)
```

| Member | Value | Description |
|---|---|---|
| `COMPLETED` | `'completed'` |  |
| `FAILED` | `'failed'` |  |
| `CANCELLED` | `'cancelled'` |  |

### `SamplingParameters`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class SamplingParameters(ContractModel)
```

Configured on bindings, never by task or agent code.

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | `1.0` |  |
| `top_p` | `float` | `1.0` |  |
| `reasoning_effort` | `str \| None` | `None` | For providers with reasoning controls, e.g. `low`, `medium`, `high`. |

### `Task`

*class* · `libraries/rollout/src/rollout/harness/task.py`

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
- `def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]` — The tools offered this turn. Default: every `@tool` method and every imported tool.
- `async def run_tools(self, run: RunContext, reply: Message) -> Observation` — Execute every tool call in `reply` concurrently; one TOOL message answers them all.

### `tool`

*function* · `libraries/rollout/src/rollout/harness/tools.py`

```python
def tool[F: Callable[..., Any]](function: F | None = None, /, *, name: str | None = None, retry_class: RetryClass = RetryClass.PURE, timeout: timedelta | None = None) -> F | Callable[[F], F]
```

Declare a task method as a tool: `@tool` or `@tool(name=..., retry_class=..., timeout=...)`.

### `ToolBinding`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class ToolBinding(ContractModel)
```

How an import is served. Exactly one kind is set.

| Field | Type | Default | Description |
|---|---|---|---|
| `local` | `str \| None` | `None` | The name of a tool set registered with the runner, in process. |
| `url` | `str \| None` | `None` | A tool set served over HTTP (`rollout.harness.remote`): an environment's own infrastructure, wherever it runs. |

### `Tools`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class Tools
```

The imported tools of a run (`run.tools`).

**Methods**

- `def __init__(self, tool_sets: Mapping[str, ToolSet], effects: Effects) -> None`
- `def specifications(self) -> list[ToolSpecification]`
- `async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> ToolResult` — Call an imported tool as a `tool.call` effect.

### `ToolSet`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class ToolSet(Protocol)
```

A provider of tools: in process, or a client of an MCP server, an HTTP service or another agent.

**Methods**

- `def specifications(self) -> Sequence[ToolSpecification]`
- `async def call(self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult` — Perform one call. Tool-level errors are results with `is_error`; exceptions are platform failures.

### `Turn`

*class* · `libraries/rollout/src/rollout/harness/history.py`

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

*class* · `libraries/rollout/src/rollout/harness/observation.py`

```python
class WaitFor
```

Suspend the run until a message of `kind` arrives or `timeout` passes.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `'message'` |  |
| `timeout` | `timedelta \| None` | `None` |  |
| `on_timeout` | `Observation` | `field(default_factory=lambda: End(truncated=True))` |  |

### `with_row`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def with_row(reference: ProgramReference, row: JsonValue) -> ProgramReference
```

The same program for another row of parameters (for the task loop: the task's parameters).

## `rollout.contracts`

Types that cross layers: canonical content, identifiers, digests, effects, events.

### `address_of`

*function* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
def address_of(endpoint: ModelEndpoint, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress
```

`AddressableEndpoint.address` of an endpoint; raises if the endpoint has no address.

### `AddressableEndpoint`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class AddressableEndpoint(ModelEndpoint, Protocol)
```

A model endpoint that also serves its slots over HTTP, to a harness that brings its own loop.

**Methods**

- `def address(self, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress` — Where such a harness reaches the session's slot. What it samples there goes `through` an endpoint
  wrapping this one, if one is given (a runner's, which reports samples to its hooks).

### `arguments_digest`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def arguments_digest(arguments: JsonValue | BaseModel) -> str
```

Sent with every `effect_id`; a tool set refuses a known `effect_id` whose arguments digest differs
(`Conflict`).

### `BlobReference`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class BlobReference(ContractModel)
```

Content kept in a blob store, named by where it is and what it hashes to.

| Field | Type | Default | Description |
|---|---|---|---|
| `uri` | `str` | required |  |
| `sha256` | `str` | required |  |
| `size` | `int` | required |  |
| `media_type` | `str` | required |  |

### `Block`

*type alias* · `libraries/rollout/src/rollout/contracts/content.py`

```python
type Block = Annotated[Text | Media | ToolCall | ToolResultBlock | Reasoning, Field(discriminator='type')]
```

### `canonical_json`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def canonical_json(value: JsonValue | BaseModel) -> bytes
```

RFC 8785 canonical JSON; a model is dumped with `None` fields omitted.

### `CapabilityContract`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class CapabilityContract(ContractModel)
```

What a model slot guarantees. It must not weaken during a run.

| Field | Type | Default | Description |
|---|---|---|---|
| `context_limit` | `int` | required | Minimum guaranteed. |
| `max_output_tokens` | `int` | required |  |

### `Conflict`

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

```python
class Conflict(Exception)
```

A receiver that deduplicates by `effect_id` was sent a known `effect_id` with a different arguments digest.

### `context_digests`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def context_digests(messages: Sequence[Message]) -> list[str]
```

The digest chain `d₀ … dₙ` of a context: `dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(itemᵢ)))` over raw digest bytes.

`dₖ` identifies the prefix of length `k`, so a retained prefix is recognizable by its own chain value.

### `ContextDelta`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ContextDelta(ContractModel)
```

The context of a request: its messages, and the digest that names them.

| Field | Type | Default | Description |
|---|---|---|---|
| `append` | `FrozenSequence[Message]` | `()` | The whole context, in order. |
| `digest` | `str` | required | The last value of the chain `rollout.contracts.digests.context_digests` computes over `append`. |

### `ContextOverflow`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ContextOverflow(ModelEndpointError)
```

The context exceeds the contract's limit; agents compact and retry.

**Methods**

- `def __init__(self, context_limit: int) -> None`

### `ContractModel`

*class* · `libraries/rollout/src/rollout/contracts/base.py`

```python
class ContractModel(BaseModel)
```

Base for every contract type: immutable, and unknown fields are kept.

Keeping unknown fields lets a component read and re-write a record written by newer code without dropping
what it does not understand.

### `ContractViolation`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ContractViolation(ModelEndpointError)
```

The request exceeds the capability contract.

### `digest`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def digest(value: JsonValue | BaseModel) -> str
```

Lowercase hexadecimal SHA-256 of the canonical JSON.

### `effect_id`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def effect_id(run_id: str, generation: int, ordinal: int) -> str
```

`{run_id}:{generation}:{ordinal}`: the same on every re-execution, so it is the universal idempotency key.

### `EffectIdentity`

*class* · `libraries/rollout/src/rollout/contracts/identifiers.py`

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

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

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
| `OUTPUT_EMIT` | `'output.emit'` |  |

### `EffectStatus`

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

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

*constant* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
EMPTY_DIGEST = hashlib.sha256(b'').hexdigest()
```

`d₀` of every context digest chain.

### `FinishReason`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class FinishReason(StrEnum)
```

Why a sample stopped.

| Member | Value | Description |
|---|---|---|
| `STOP` | `'stop'` |  |
| `LENGTH` | `'length'` |  |
| `TOOL_USE` | `'tool_use'` |  |

### `FrozenSequence`

*type alias* · `libraries/rollout/src/rollout/contracts/base.py`

```python
type FrozenSequence[T] = Annotated[Sequence[T], AfterValidator(tuple)]
```

A sequence field that accepts any sequence and is stored as a tuple, so contract values stay immutable.

### `InternalError`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class InternalError(ModelEndpointError)
```

The endpoint failed.

### `Media`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

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

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Message(ContractModel)
```

A message in canonical form: a role and a sequence of content blocks.

TOOL messages contain only `ToolResultBlock`s, and `ToolCall`s appear only in ASSISTANT messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `Role` | required |  |
| `content` | `FrozenSequence[Block]` | `()` |  |
| `meta` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Not model-visible; never rendered and not covered by the context digest. |

**Methods**

- `@property def text(self) -> str` — The concatenated text blocks.
- `@property def tool_calls(self) -> list[ToolCall]` — The tool calls, in order.
- `@classmethod def user(cls, text: str) -> 'Message'` — A USER message with one text block.
- `@classmethod def assistant(cls, text: str) -> 'Message'` — An ASSISTANT message with one text block.
- `@classmethod def system(cls, text: str) -> 'Message'` — A SYSTEM message with one text block.

### `message_digest`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def message_digest(message: Message) -> str
```

Covers what the model can see: `meta` is excluded.

### `ModelAddress`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ModelAddress(ContractModel)
```

Where a harness that brings its own loop reaches a model slot: an OpenAI-compatible endpoint. Whatever
answers there is the slot's model; the harness only sets its base URL and key.

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | required |  |
| `api_key` | `str` | required | Names the session: valid for this run's slot only. |
| `model` | `str` | required | What to send as the model's name (the endpoint ignores it). |

### `ModelEndpoint`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ModelEndpoint(Protocol)
```

Serves model slots: implemented by the recorder and by direct adapters. An endpoint that can also be reached
over HTTP is an `AddressableEndpoint`.

**Methods**

- `def describe(self, session_id: str) -> CapabilityContract` — The capability contract of the session's model slot.
- `async def sample(self, request: SampleRequest) -> SampleResult` — One reply. Where the endpoint deduplicates, a repeated `effect_id` returns the recorded result.
- `async def cancel(self, effect_id: str) -> None` — Best-effort.

### `ModelEndpointError`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ModelEndpointError(Exception)
```

Errors an endpoint raises; see the table in the contract for how the core handles each.

### `NamedToolChoice`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class NamedToolChoice(ContractModel)
```

The model must call this tool.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |

### `new_message_id`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def new_message_id() -> str
```

`m_{ulid}`: the id of a message sent without an idempotency key.

### `new_run_id`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def new_run_id() -> str
```

`r_{ulid}`.

### `new_ulid`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def new_ulid() -> str
```

A ULID: 48 bits of Unix time in milliseconds, then 80 random bits, in Crockford base 32 (26 characters).

Minted by runners and services, never by task code (which has no ambient randomness under a durable runner).

### `OutcomeUnknown`

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

```python
class OutcomeUnknown(Exception)
```

A guarded effect was interrupted by a crash in an earlier attempt: it may or may not have happened.

**Methods**

- `def __init__(self, effect_id: str) -> None`

### `Overloaded`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class Overloaded(ModelEndpointError)
```

Admission control: retry after `retry_after` seconds.

**Methods**

- `def __init__(self, retry_after: float | None = None) -> None`

### `Reasoning`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Reasoning(ContractModel)
```

The model's reasoning.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['reasoning']` | `'reasoning'` |  |
| `scope` | `ReasoningScope` | required |  |
| `text` | `str` | required |  |

### `ReasoningScope`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ReasoningScope(StrEnum)
```

Who can consume a reasoning block.

| Member | Value | Description |
|---|---|---|
| `PORTABLE` | `'portable'` | Plain text that any renderer may render or drop. |

### `ResultBlock`

*type alias* · `libraries/rollout/src/rollout/contracts/content.py`

```python
type ResultBlock = Annotated[Text | Media, Field(discriminator='type')]
```

### `RetryClass`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class RetryClass(StrEnum)
```

What a durable runner may do with a tool call after a crash (docs/architecture/overview.md).

| Member | Value | Description |
|---|---|---|
| `PURE` | `'pure'` |  |
| `IDEMPOTENT` | `'idempotent'` |  |
| `SIDE_EFFECTING` | `'side_effecting'` |  |
| `UNKNOWN` | `'unknown'` |  |

### `Role`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

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

*constant* · `libraries/rollout/src/rollout/contracts/events.py`

```python
RUN_EVENT_SCHEMA_VERSION = 1
```

### `RunEvent`

*class* · `libraries/rollout/src/rollout/contracts/events.py`

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
| `payload` | `JsonValue` | `None` | Type-specific; see docs/libraries/rollout/contracts/run-events.md. |

### `RunEventType`

*class* · `libraries/rollout/src/rollout/contracts/events.py`

```python
class RunEventType(StrEnum)
```

The closed catalog of run events.

| Member | Value | Description |
|---|---|---|
| `RUN_CREATED` | `'run.created'` |  |
| `RUN_SUSPENDED` | `'run.suspended'` |  |
| `RUN_COMPLETED` | `'run.completed'` |  |
| `RUN_FAILED` | `'run.failed'` |  |
| `RUN_CANCEL_REQUESTED` | `'run.cancel_requested'` |  |
| `RUN_CANCELLED` | `'run.cancelled'` |  |
| `OBSERVATION_RECORDED` | `'observation.recorded'` |  |
| `REWARD_ASSIGNED` | `'reward.assigned'` |  |
| `TRAINING_EXCLUDED` | `'training.excluded'` |  |
| `OUTPUT_EMITTED` | `'output.emitted'` |  |
| `EFFECT_REQUESTED` | `'effect.requested'` |  |
| `EFFECT_COMPLETED` | `'effect.completed'` |  |
| `MESSAGE_RECEIVED` | `'message.received'` |  |
| `TURN_INTERRUPTED` | `'turn.interrupted'` |  |
| `TOOLS_RESOLVED` | `'tools.resolved'` |  |

### `RunFailureClass`

*class* · `libraries/rollout/src/rollout/contracts/events.py`

```python
class RunFailureClass(StrEnum)
```

Why a run failed (the `class` of a `run.failed` event).

| Member | Value | Description |
|---|---|---|
| `TASK_ERROR` | `'task_error'` |  |
| `INVALID_OBSERVATION` | `'invalid_observation'` |  |

### `SampleRequest`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

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

### `SampleResult`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

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

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def session_id(run_id: str, model_slot: str) -> str
```

`{run_id}/{model_slot}`: one recorder session per model slot per run.

### `SessionIdentity`

*class* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
class SessionIdentity
```

The parts of a `session_id`: `{run_id}/{model_slot}`.

| Field | Type | Default | Description |
|---|---|---|---|
| `owner` | `str` | required |  |
| `model_slot` | `str` | required |  |

**Methods**

- `@classmethod def parse(cls, session_id: str) -> 'SessionIdentity'`

### `spec_hash`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def spec_hash(specification: ToolSpecification) -> str
```

Covers only the model-visible fields of a tool specification.

### `TERMINAL_EVENT_TYPES`

*constant* · `libraries/rollout/src/rollout/contracts/events.py`

```python
TERMINAL_EVENT_TYPES = frozenset({RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED, RunEventType.RUN_CANCELLED})
```

### `Text`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Text(ContractModel)
```

Plain text.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['text']` | `'text'` |  |
| `text` | `str` | required |  |

### `ToolCall`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

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

*type alias* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
type ToolChoice = ToolChoiceMode | NamedToolChoice
```

### `ToolChoiceMode`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

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

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ToolResult(ContractModel)
```

What a tool produces. Platform failures are exceptions, not results.

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `FrozenSequence[ResultBlock]` | `()` |  |
| `structured` | `JsonValue` | `None` | Optional: the result as JSON, for code that reads it. |
| `is_error` | `bool` | `False` | A tool-level error the model should see and reason about (non-zero exit, file not found). |
| `truncated` | `bool` | `False` | `content` is only part of what the tool produced. |

### `ToolResultBlock`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

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

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ToolSpecification(ContractModel)
```

What the model sees about a tool, plus an extension that is never model-visible.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `description` | `str` | `''` |  |
| `input_schema` | `Mapping[str, JsonValue]` | `Field(default_factory=lambda: {'type': 'object', 'properties': {}})` | JSON Schema 2020-12 with `type: object`. |
| `retry_class` | `RetryClass` | `RetryClass.UNKNOWN` |  |

**Methods**

- `def model_visible(self) -> dict[str, JsonValue]` — The fields a model sees and the spec hash covers.

### `Usage`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

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

## `rollout.catalog`

What an environment offers to be trained on.

### `binding_for`

*function* · `libraries/rollout/src/rollout/catalog.py`

```python
def binding_for(catalog: Catalog, channel: str, tools: Mapping[str, ToolBinding] | None = None) -> RunBinding
```

How a catalog's runs are served: every model slot of its program from `channel`, and each of its imports
from the tool set of its own name, or where `tools` says. (A program says which slots and imports it has once
it is given a row: the catalog's first.)

### `Catalog`

*class* · `libraries/rollout/src/rollout/catalog.py`

```python
class Catalog(Protocol)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `ProgramReference` | required | What a run executes; its parameters are a start. |

**Methods**

- `def rows(self) -> Sequence[Row]` — Every situation, easiest first.
- `def start(self, row: Row, rng: random.Random) -> JsonValue` — The parameters of one start of `row` (a seed drawn with `rng`, say): what every run of a group is given.

### `Row`

*class* · `libraries/rollout/src/rollout/catalog.py`

```python
class Row
```

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | Its name among the catalog's rows. |
| `title` | `str` | required | What it is, for people. |
| `parameters` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |
| `counts_for` | `tuple[str, ...]` | `()` | The keys of other rows that a group of this one is evidence about too: the same situation with more help, say. What it teaches about this row it teaches about them. |

## `rollout.local`

The runner in this process.

### `EndpointFactory`

*type alias* · `libraries/rollout/src/rollout/local/runner.py`

```python
type EndpointFactory = Callable[[DirectModel], ModelEndpoint]
```

Creates the endpoint for a direct model binding; registered with the runner by provider name.

### `LocalRunContext`

*class* · `libraries/rollout/src/rollout/local/context.py`

```python
class LocalRunContext
```

Implements `RunContext` and `Effects` in process.

**Methods**

- `def __init__(self, run_id: str, endpoints: Mapping[str, ModelEndpoint], *, context_hints: ContextHints | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environment_service: EnvironmentService | None = None, blobs: Blobs | None = None, conversation: ConversationKey | None = None, on_event: Callable[[RunEvent], None] | None = None, retain_events: bool = True) -> None`
- `@property def run_id(self) -> str`
- `@property def conversation(self) -> ConversationKey | None`
- `@property def turn(self) -> int`
- `@property def history(self) -> History`
- `@property def models(self) -> Mapping[str, Model]`
- `@property def model(self) -> Model`
- `@property def tools(self) -> Tools`
- `@property def environments(self) -> Environments | None`
- `@property def blobs(self) -> Blobs | None`
- `@property def random(self) -> random.Random`
- `@property def context_hints(self) -> ContextHints`
- `def now(self) -> datetime`
- `def reward(self, value: float, *, slot: str = 'policy', key: str = 'default') -> None`
- `def exclude_from_training(self, reason: str) -> None`
- `async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]`
- `async def emit(self, kind: str, payload: JsonValue, *, to: Address | None = None) -> None` — Durable output, e.g. a reply that a connector delivers. Recorded as an `output.emit` effect.
- `def record(self, observation: Observation | WaitFor, *, reply: Message | None = None) -> None`
- `async def wait_for_message(self, wait: WaitFor) -> Envelope | None`
- `async def take_steering_messages(self) -> list[Envelope]`
- `async def interruptible[T](self, reply: Awaitable[T]) -> T`
- `def take_undelivered(self) -> list[Envelope]` — Messages the run never consumed; the runner hands them to the conversation's next run.
- `def deliver(self, envelope: Envelope, mode: DeliveryMode) -> None` — Deliver a message to this run (docs/guide/conversations.md#priority-and-delivery-mode).
- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue], guard: bool = False) -> T`
- `def record_event(self, event_type: RunEventType, payload: JsonValue) -> RunEvent`

### `LocalRunHandle`

*class* · `libraries/rollout/src/rollout/local/runner.py`

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
- `def recorded_events(self) -> list[RunEvent]` — Every event recorded so far.
- `@property def task(self) -> asyncio.Task[None] | None`
- `def attach(self, task: asyncio.Task[None]) -> None`
- `def finish(self, outcome: RunOutcome) -> None`
- `def notify(self, event: RunEvent | None = None) -> None` — Wake event streams: a new event was recorded, or the run ended.

### `LocalRunner`

*class* · `libraries/rollout/src/rollout/local/runner.py`

```python
class LocalRunner(MessageRouter)
```

Implements `Runner` in process.

Direct model bindings are served by endpoint factories registered by provider name. Conversations addressed to a
deployment start a run of its specification when none is live; one run consumes a conversation's messages at a
time, and messages a run never consumed start the conversation's next run.

**Methods**

- `def __init__(self, *, providers: Mapping[str, EndpointFactory] | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environments: EnvironmentService | None = None, blobs: Blobs | None = None, recorder: RecordedEndpoints | None = None, hooks: Sequence[RunHooks] = ()) -> None` — `recorder` serves recorded model bindings (trainable channels); direct bindings use `providers`. `hooks`
  watch every run: each event recorded and each model sample.
- `async def launch(self) -> None` — Nothing to start: runs execute on the caller's event loop.
- `async def close(self) -> None` — Nothing to release: nothing outlives the process.
- `def deploy(self, deployment: Deployment) -> None` — Register or replace a deployment; a conversation's next run uses the current version.
- `def run(self, run_id: str) -> LocalRunHandle`
- `def conversation_of(self, run_id: str) -> ConversationKey | None` — The conversation a run serves, if any.
- `def conversation_runs(self, deployment: str, key: str) -> list[LocalRunHandle]` — The conversation's runs, oldest first.
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None) -> LocalRunHandle`
- `async def cancel(self, run_id: str, *, reason: str) -> None`

### `RewardAssignment`

*class* · `libraries/rollout/src/rollout/local/context.py`

```python
class RewardAssignment
```

| Field | Type | Default | Description |
|---|---|---|---|
| `slot` | `str` | required |  |
| `value` | `float` | required |  |
| `key` | `str` | required |  |

## `rollout.testing`

Test doubles: a scripted model endpoint and helpers.

### `events_of`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def events_of(run: LocalRunContext, event_type: RunEventType) -> list[RunEvent]
```

The run's events of one type, in order.

### `LedgerEndpoint`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class LedgerEndpoint
```

Wraps a model endpoint and appends every sample's `effect_id` to a file: to count calls across processes.

**Methods**

- `def __init__(self, inner: ModelEndpoint, ledger: Path) -> None`
- `def describe(self, session_id: str) -> CapabilityContract`
- `def address(self, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress`
- `async def sample(self, request: SampleRequest) -> SampleResult`
- `async def cancel(self, effect_id: str) -> None`

### `LedgerEnvironments`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class LedgerEnvironments
```

Wraps an environment service and appends every command it starts to a file, with its `effect_id`.

**Methods**

- `def __init__(self, inner: EnvironmentService, ledger: Path) -> None`
- `async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None`
- `async def execute(self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = '') -> ExecutionResult`
- `async def put(self, environment_id: str, path: str, data: bytes) -> None`
- `async def get(self, environment_id: str, path: str) -> bytes`
- `async def destroy(self, environment_id: str) -> None`

### `local_run`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def local_run(task: Task, replies: Iterable[ScriptedReply] = ()) -> tuple[LocalRunContext, ScriptedModelEndpoint]
```

A local run context for `task` whose model slots all reply from one script.

### `payload`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def payload(event: RunEvent) -> dict[str, JsonValue]
```

An event's payload as a JSON object.

### `read_ledger`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def read_ledger(ledger: Path) -> list[dict[str, str]]
```

### `ScriptedModelEndpoint`

*class* · `libraries/rollout/src/rollout/testing.py`

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

*type alias* · `libraries/rollout/src/rollout/testing.py`

```python
type ScriptedReply = Message | str | Callable[[SampleRequest], Message | Awaitable[Message]]
```

A reply, its text, or a function of the request (which may await, e.g. to hold a sample open).

### `tool_call_reply`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def tool_call_reply(*calls: ToolCall, text: str = '') -> Message
```

An assistant reply that makes tool calls.

## `rollout_train.rollouts`

Rollout jobs: rows in, episodes out, weights published.

### `Episode`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Episode
```

| Field | Type | Default | Description |
|---|---|---|---|
| `cursor` | `int` | required | Its place in the job's log: episodes are numbered from 1 in the order they ended. |
| `job` | `str` | required |  |
| `ticket` | `str` | required |  |
| `run_id` | `str` | required |  |
| `labels` | `Mapping[str, str]` | required |  |
| `outcome` | `Outcome` | required |  |
| `detail` | `str \| None` | `None` |  |
| `info` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the program reported as its result (`run.emit("result", {...})`). `solved`, `saturated` and `duration` read the three entries training knows about. |
| `excluded` | `str \| None` | `None` | Why the program asked for the run to be left out of training, if it did. |
| `trajectories` | `Mapping[str, Trajectory]` | `field(default_factory=dict[str, Trajectory])` |  |

**Methods**

- `@property def reward(self) -> float` — The mean of the slots' rewards (a team that is rewarded together has one reward).
- `@property def trainable(self) -> bool`
- `@property def solved(self) -> bool` — Whether the program said its task was solved (`info["solved"]`).
- `@property def saturated(self) -> bool` — Whether the program said nothing was left to earn (`info["saturated"]`).
- `@property def duration(self) -> float | None` — How long the program said it took, in the task's own units (`info["duration"]`), if it said.

### `events_of`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def events_of(record: Record, blobs: Blobs) -> list[RunEvent]
```

The events of the run a record names, as its runner recorded them.

### `Job`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class Job(Protocol)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |

**Methods**

- `async def run(self, parameters: JsonValue, *, labels: Mapping[str, str] | None = None, count: int = 1, key: str = '') -> Ticket` — Queue `count` runs of one row. They start as there is room, after the runs of tickets queued before. With a
  `key`, asking again is asking for the same ticket: a caller that died after asking gets it back, with whatever
  episodes it has.
- `def episodes(self, cursor: int = 0) -> AsyncIterator[Episode]` — Every episode after `cursor` that the job has, then new ones as runs end, until the job is closed.
- `async def acknowledge(self, cursor: int) -> None` — The caller has consumed everything through `cursor`: a job started again goes on from there.
- `async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int` — Serve new weights on a channel from now on; returns the version they are served as (`version`, where
  the caller's policy numbers its own).
- `async def note(self, kind: str, payload: Mapping[str, JsonValue]) -> None` — Put something of the caller's own (an update's statistics, say) where whoever watches the job sees it.
- `async def status(self) -> Status`

### `JobHooks`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class JobHooks
```

Watch a job at the level its caller thinks at: tickets, episodes, published weights, the caller's own notes.

**Methods**

- `def on_job(self, event: Mapping[str, JsonValue]) -> None` — `event["kind"]` is `ticket`, `admitted`, `episode`, `published`, or a kind the caller noted.

### `Jobs`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class Jobs(Protocol)
```

Where jobs are started: `RolloutJobs` in this process, or `rollout_train.rollouts.service.RolloutClient` for jobs
served elsewhere.

**Methods**

- `async def start(self, *, program: ProgramReference, binding: RunBinding, in_flight: int, name: str = '') -> 'Job'`

### `loaded`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def loaded(record: Record, blobs: Blobs) -> Episode
```

The episode a record names, with its trajectories read back from `blobs`.

### `Outcome`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Outcome(StrEnum)
```

| Member | Value | Description |
|---|---|---|
| `COMPLETED` | `'completed'` |  |
| `FAILED` | `'failed'` | The program raised: its `detail` says what. |
| `CANCELLED` | `'cancelled'` |  |

### `Record`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Record
```

An episode as it is logged and sent: everything but its trajectories, and where those and its events are kept.

| Field | Type | Default | Description |
|---|---|---|---|
| `episode` | `Episode` | required | With no segments in its trajectories: their rewards only. |
| `trajectories` | `BlobReference \| None` | `None` |  |
| `events` | `BlobReference \| None` | `None` |  |
| `sampled` | `Mapping[str, int]` | `field(default_factory=dict[str, int])` | Tokens the policy sampled, by model slot. |

**Methods**

- `def to_json(self) -> dict[str, Any]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Record'`

### `Recorded`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class Recorded(Protocol)
```

What a job needs of the recorder: each run's segments, and somewhere to publish weights.

**Methods**

- `def sessions(self, run_id: str) -> dict[str, list[Segment]]`
- `def forget(self, run_id: str) -> None`
- `async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int`

### `Refused`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class Refused(Exception)
```

A job would not run a ticket: its guard refused (a machine out of memory, say), or the job was closed.

### `RolloutJob`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class RolloutJob
```

A job over a runner. Created by `RolloutJobs.start`.

**Methods**

- `def __init__(self, job_id: str, specification: RunSpecification, runner: Runner, recorder: Recorded, *, in_flight: int, log: Path | None, blobs: Blobs | None, hooks: Sequence[JobHooks], guard: Callable[[], None] | None) -> None`
- `async def run(self, parameters: JsonValue, *, labels: Mapping[str, str] | None = None, count: int = 1, key: str = '') -> RolloutTicket`
- `async def episodes(self, cursor: int = 0) -> AsyncIterator[Episode]`
- `async def news(self, cursor: int, seconds: float) -> bool` — Wait up to `seconds` for an episode after `cursor`; False once the job is closed and none is left.
- `def ticket(self, ticket: str) -> RolloutTicket` — A ticket by its id (for a caller that holds only the id), until its episodes are acknowledged.
- `def after(self, cursor: int) -> list[Record]` — The records after `cursor` that are in the log now.
- `async def episode(self, record: Record) -> Episode` — The episode a record of this job's log names, with its trajectories.
- `async def acknowledge(self, cursor: int) -> None`
- `async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int`
- `async def note(self, kind: str, payload: Mapping[str, JsonValue]) -> None`
- `async def status(self) -> Status`
- `async def close(self) -> None` — Stop admitting, cancel what is running, and end every `episodes` stream. A ticket that is not over is
  refused to whoever waits on it here; a job with a log runs what it still owes when it is started again.

### `RolloutJobs`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class RolloutJobs
```

Starts jobs on a runner. `log` is where each job keeps its log (under `log/JOB`), so that every episode
outlives the process and a caller that stops can go on from its cursor; the episodes' trajectories and events go to
`blobs` (by default a store in files under `log/blobs`). `guard` is called before runs are admitted and raises
to refuse them (a machine out of memory, say).

**Methods**

- `def __init__(self, runner: Runner, recorder: Recorded, *, log: Path | None = None, blobs: Blobs | None = None, hooks: Sequence[JobHooks] = (), guard: Callable[[], None] | None = None) -> None`
- `async def start(self, *, program: ProgramReference, binding: RunBinding, in_flight: int, name: str = '') -> RolloutJob` — A job that runs `program` (with each ticket's row as its parameters) under `binding`, at most `in_flight`
  runs at a time. A `name` makes the job's log one a later caller finds again: a job of that name that is
  still open is closed first (its runs are cancelled), and the new one goes on over its log.
- `def job(self, job: str) -> RolloutJob` — A job by its id (for a caller that holds only the id).
- `async def close(self) -> None`

### `RolloutTicket`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class RolloutTicket
```

A ticket of a `RolloutJob`. The job keeps it until its episodes are acknowledged.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |
| `parameters` | `JsonValue` | required |  |
| `labels` | `Mapping[str, str]` | required |  |
| `count` | `int` | required |  |
| `ended` | `list[Record]` | `field(default_factory=list[Record])` | Its runs that have ended, as the log has them. (A run the job itself cut short by closing is in the log, and is not one of these: it is run again when the job next starts.) |
| `refused` | `str \| None` | `None` | Why the job would not run it, if it would not. |

**Methods**

- `async def episodes(self) -> list[Episode]`
- `async def ready(self, seconds: float) -> bool` — Whether the ticket is over (its runs have all ended, or it was refused), waiting up to `seconds`.
- `@property def owed(self) -> int` — Runs that have neither ended nor been started.

### `Status`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class Status
```

| Field | Type | Default | Description |
|---|---|---|---|
| `queued` | `int` | required | Runs waiting for room. |
| `running` | `int` | required |  |
| `finished` | `int` | required | Episodes in the log, of every outcome. |
| `acknowledged` | `int` | required | The cursor the caller has consumed through. |

### `stored`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def stored(episode: Episode, events: Sequence[RunEvent], blobs: Blobs) -> Record
```

Keep an episode's trajectories and its run's events in `blobs`; returns the record that names them.

### `Ticket`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/jobs.py`

```python
class Ticket(Protocol)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |

**Methods**

- `async def episodes(self) -> list[Episode]` — The ticket's episodes, once every one of its runs has ended (in the order they ended). Raises `Refused`
  if the job would not run it.

### `Trajectory`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Trajectory
```

What one model slot's rollout leaves to train on: its segments, and its rewards.

| Field | Type | Default | Description |
|---|---|---|---|
| `segments` | `list[Segment]` | required |  |
| `rewards` | `Mapping[str, float]` | required | By key; a program that assigns one reward uses the key `default`. |

**Methods**

- `@property def reward(self) -> float`

## `rollout_train`

The training loop, the group algorithm, the curriculum, and what they ask of a trainer.

### `Algorithm`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Algorithm(Protocol)
```

What the training loop asks of an algorithm.

**Methods**

- `@property def group_size(self) -> int` — How many episodes of one start it compares.
- `def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch` — What to train on from a group's episodes (of every outcome), within what the trainer can afford.

### `Batch`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Batch
```

What an algorithm makes of a group of episodes.

| Field | Type | Default | Description |
|---|---|---|---|
| `segments` | `Sequence[Weighted]` | `()` | What to train on. |
| `skipped` | `str \| None` | `None` | Why there is nothing to train on, if there is not. |
| `notes` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the algorithm wants logged with the group. |

### `Budget`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Budget
```

| Field | Type | Default | Description |
|---|---|---|---|
| `segment_tokens` | `int \| None` | `None` | The longest segment the trainer can train on (None: any). |
| `segments` | `int \| None` | `None` | How many segments a step can afford (None: any number). |

### `Checkpoint`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Checkpoint
```

A version's files on this machine: what a step starts from.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` | `Path` | required |  |
| `state` | `Path \| None` | `None` | What the trainer left for itself beside the weights (an optimizer's state, say), if it left any. |

### `Colocated`

*class* · `libraries/rollout-train/src/rollout_train/colocated.py`

```python
class Colocated
```

A trainer that shares an accelerator with the engines of some channels: requests to them are held back and
the engines sleep while it steps. `guard` is called once they are asleep and raises if the step should not
start (too little memory, say).

**Methods**

- `def __init__(self, trainer: Trainer, channels: Sequence[Pausable], *, guard: Callable[[], None] | None = None) -> None`
- `async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step`

### `complete_groups`

*function* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
async def complete_groups(episodes: AsyncIterator[Episode], *, by: str = 'group', size: int) -> AsyncIterator[list[Episode]]
```

The episodes of a stream, gathered by the label `by`: each group is yielded when `size` of its episodes have
ended (whatever their outcome).

### `Curriculum`

*class* · `libraries/rollout-train/src/rollout_train/curriculum.py`

```python
class Curriculum
```

| Field | Type | Default | Description |
|---|---|---|---|
| `rows` | `Sequence[Row]` | required |  |
| `rng` | `random.Random` | `field(default_factory=random.Random)` |  |
| `start` | `int` | `3` | This many rows are unlocked from the beginning. |
| `reach` | `int` | `4` | Rows unlocked past the hardest one solved. |
| `smoothing` | `float` | `0.5` | Weight of the newest group in the moving averages. |
| `floor` | `float` | `0.05` |  |
| `records` | `dict[str, Record]` | `field(default_factory=dict[str, Record])` |  |

**Methods**

- `def unlocked(self) -> list[Row]`
- `def sample(self, pending: Collection[str] = (), rng: random.Random | None = None) -> Row` — The next row. `pending` names rows whose latest group has not been recorded yet: choosing one again would
  be choosing on what was known before it, so the others come first.
- `def recorded(self, line: Result) -> None` — Take a group's result into account: the row of its title, or failing that of its key (a key that is a
  place in a catalog changes when rows are added). A curriculum is the fold of a run's results.
- `def weight(self, row: Row) -> float`
- `def update(self, row: Row, rewards: Sequence[float], solved: Sequence[bool]) -> None` — Record a group of episodes of `row`: each one's reward and whether it solved the row.
- `def failed(self, row: Row) -> None` — Record a group of `row` none of whose episodes completed. After `FAILED_GROUPS` of them in a row it is no
  longer untried, and it taught nothing.
- `def record(self, row: Row) -> Record`

### `Fence`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class Fence
```

The right to write within a scope, until someone takes it again.

| Field | Type | Default | Description |
|---|---|---|---|
| `scope` | `str` | required |  |
| `number` | `int` | required |  |

### `Fenced`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class Fenced(Exception)
```

A writer whose fence is no longer the newest tried to write: another has taken its place.

### `FileLedger`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class FileLedger
```

A `Ledger` in a directory: a table is `<table>.jsonl`, one `{"key", "fence", "record"}` per line. Processes
on one machine may share it: every operation holds a lock on the directory.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def take(self, scope: str) -> Fence`
- `async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool`
- `async def read(self, table: str) -> dict[str, JsonValue]`
- `async def tables(self) -> list[str]`
- `async def fences(self) -> dict[str, int]`

### `group_advantages`

*function* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
def group_advantages(scores: Sequence[float]) -> list[float] | None
```

Each score minus the group's mean; None when all are equal (no signal).

### `Grpo`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Grpo
```

| Field | Type | Default | Description |
|---|---|---|---|
| `group_size` | `int` | `4` |  |
| `tie_break` | `bool` | `True` | Whether the fastest of a group's saturated episodes scores a point more. |

**Methods**

- `def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch` — The segments of the group's episodes that are fit to train on (completed, and not excluded), each with
  its episode's advantage.

### `Ledger`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class Ledger(Protocol)
```

**Methods**

- `async def take(self, scope: str) -> Fence` — Take a scope's fence. Whoever held it can no longer write within the scope.
- `async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool` — Append a record under `key`, unless the table has that key: then nothing changes and False is returned.
  Raises `Fenced` if `fence` is not its scope's newest.
- `async def read(self, table: str) -> dict[str, JsonValue]` — A table's records by key, in the order they were appended.
- `async def tables(self) -> list[str]` — The tables that have records, by name.
- `async def fences(self) -> dict[str, int]` — The newest fence of every scope that has been taken.

### `Manifest`

*class* · `libraries/rollout-train/src/rollout_train/policies.py`

```python
class Manifest
```

The files of a checkpoint, by their paths within it, each kept as a blob.

| Field | Type | Default | Description |
|---|---|---|---|
| `files` | `Mapping[str, BlobReference]` | required |  |
| `layout` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | How the weights are divided among the files, where they are divided: whoever wrote them says, so that a reader with the same division reads its own files and no others. |

### `Policies`

*class* · `libraries/rollout-train/src/rollout_train/policies.py`

```python
class Policies
```

The versions of every policy, in a ledger, and their files in a blob store.

**Methods**

- `def __init__(self, ledger: Ledger, blobs: Blobs) -> None`
- `async def writer(self, policy: str) -> Fence` — Become the one that may add versions to a policy: whoever was is shut out.
- `async def versions(self, policy: str) -> list[Version]` — A policy's versions, oldest first.
- `async def head(self, policy: str) -> Version | None` — A policy's newest version, if it has one.
- `async def version(self, name: str) -> Version` — The version a name says.
- `async def add(self, fence: Fence, policy: str, number: int, *, weights: Path, state: Path | None = None, parent: str | None = None, batch: BlobReference | None = None, metrics: Mapping[str, float] | None = None) -> Version` — Keep a checkpoint's files and append the version that names them. The append is what makes the version
  exist: a writer that dies before it has made nothing, and one that repeats it (the same number) gets the
  version that is there.
- `async def thin(self, fence: Fence, policy: str, retention: 'Retention') -> list[str]` — Let go of the files (weights and trainer state) of the versions `retention` does not keep, and return their
  names. A release is appended to the ledger before its blobs are deleted, and a blob is deleted only if no
  version still names it, so this may be repeated after a crash at any point.
- `async def files(self, manifest: Manifest, directory: Path) -> Path` — A manifest's files under `directory`, read from the blob store if they are not there. The directory
  appears whole or not at all, so whatever looks for a file in it never finds half a checkpoint.

### `Result`

*class* · `libraries/rollout-train/src/rollout_train/record.py`

```python
class Result
```

| Field | Type | Default | Description |
|---|---|---|---|
| `group` | `int` | required | The group's number in the run, from 1. |
| `time` | `float` | required | When it was written, in seconds since the epoch. |
| `task` | `str` | required | The row's key. |
| `title` | `str` | `''` |  |
| `rollout_seconds` | `float` | `0.0` | From the group's decision to its last episode's end (when its result was written). |
| `rewards` | `list[float]` | `field(default_factory=list[float])` | Of the episodes fit to train on, as are `solved` and `durations`. |
| `solved` | `list[bool]` | `field(default_factory=list[bool])` |  |
| `durations` | `list[float \| None]` | `field(default_factory=list[float \| None])` |  |
| `failed` | `int` | `0` | Episodes that did not complete, or asked to be left out. |
| `failures` | `list[str]` | `field(default_factory=list[str])` |  |
| `notes` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the algorithm said of the group. |
| `segments_recorded` | `int` | `0` |  |
| `segments` | `int` | `0` | Segments the algorithm found to train on: none if it skipped the group. |
| `skipped` | `str \| None` | `None` | Why the algorithm found nothing to train on, if it did not. |
| `unlocked` | `int` | `0` | Rows of the catalog unlocked after this group. |

**Methods**

- `def to_json(self) -> dict[str, Any]` — The record as the `results` table keeps it: without what the group's own record and key say (`JOINED`).
- `@classmethod def from_json(cls, data: Mapping[str, Any], number: int, group: Mapping[str, Any]) -> 'Result'` — A result as it is kept, with what its group's record (`group`, under `number`) says.

### `results`

*function* · `libraries/rollout-train/src/rollout_train/record.py`

```python
async def results(ledger: Ledger, run: str = 'train') -> list[Result]
```

How a run's groups went, by their numbers.

### `Step`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Step
```

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `Mapping[str, float]` | required |  |

### `StepFailed`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class StepFailed(Exception)
```

A step did not produce weights: the policy is as it was, and a later step may succeed.

### `train`

*function* · `libraries/rollout-train/src/rollout_train/loop.py`

```python
async def train(jobs: Jobs, catalog: Catalog, trainer: Trainer, policies: Policies, *, policy: str, channel: str, directory: Path, run: str = 'train', algorithm: Algorithm | None = None, groups: int = 100, groups_per_step: int = 4, episodes_at_once: int = 6, seed: int = 0, binding: RunBinding | None = None, curriculum: Curriculum | None = None, retention: Retention | None = None) -> None
```

Train `policy` on `catalog` until `groups` more groups have been played (those a stopped loop left unplayed
among them) and every group played has been trained on, serving it on `channel`. A step is taken over the groups
queued once at least `groups_per_step` have something to train on (and, at the end, over what is left).
`directory` is where versions' files are kept on this machine while they are in use: the one being served and
the one before it (a turn in progress finishes under the weights it began with); every version's files are in
the blob store. `algorithm` is `Grpo()` unless given. `episodes_at_once` caps the episodes running at once,
whatever groups they are of. `binding` says how the program's model slots and imports are
served (by default: every slot from `channel`, each import from the tool set of its own name). `curriculum` is
one that has recorded nothing: the run's results are folded into it. `retention` says which versions keep their
files (weights and trainer state) once a newer one is served (`Retention()` unless given).

### `Trained`

*class* · `libraries/rollout-train/src/rollout_train/record.py`

```python
class Trained
```

What was done with a group: the step that covered it, and the version that step made or why it failed.

| Field | Type | Default | Description |
|---|---|---|---|
| `step` | `int` | required |  |
| `version` | `str \| None` | `None` | By name, once made. |
| `error` | `str \| None` | `None` |  |

### `trained`

*function* · `libraries/rollout-train/src/rollout_train/record.py`

```python
async def trained(ledger: Ledger, run: str = 'train') -> dict[int, Trained]
```

For each group a step covers: that step, and its outcome if it has one.

### `Trainer`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Trainer(Protocol)
```

A trainer keeps nothing between steps that it cannot be given again: a step says what it starts from and
where its files go, so any trainer can take any step of any policy.

| Field | Type | Default | Description |
|---|---|---|---|
| `budget` | `Budget` | required |  |

**Methods**

- `async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step` — Train on the batch, starting from `parent` (None: from the base model). The new weights are left in
  `into/weights`, and what a later step starts from in `into/state`. Raises `StepFailed` if the step
  produced no weights.

### `Version`

*class* · `libraries/rollout-train/src/rollout_train/policies.py`

```python
class Version
```

| Field | Type | Default | Description |
|---|---|---|---|
| `policy` | `str` | required |  |
| `number` | `int` | required | From 1, in the order the policy's versions were made. |
| `weights` | `Manifest \| None` | required | None once it was released (`Policies.thin`). |
| `parent` | `str \| None` | `None` | The version it was trained from, by name: of this policy, or of another (a fork). None: from the base. |
| `state` | `Manifest \| None` | `None` | What a trainer goes on from: the optimizer's state, say. |
| `batch` | `BlobReference \| None` | `None` | What it was trained on: the segments, each as its place in a job's log and its advantage. |
| `metrics` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` |  |
| `made` | `float` | `0.0` | When, in seconds since the epoch. |
| `released` | `float \| None` | `None` | When its files were let go (`Policies.thin`), if they were: its weights and its trainer state are then None. Its record stays: where it came from, what it was trained on, and its metrics. |

**Methods**

- `@property def name(self) -> str`

### `Weighted`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Weighted
```

A segment to train on, and its advantage: every token the policy sampled in it counts by that much.

| Field | Type | Default | Description |
|---|---|---|---|
| `segment` | `Segment` | required |  |
| `advantage` | `float` | required |  |
| `source` | `str` | `''` | Where the segment is from, for the record of what a step trained on: `cursor/slot/index` in the job's log. |

## `rollout_train.inference`

Channels: policies being served, and what they ask of an engine.

### `Channel`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Channel
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `engines` | `Sequence[Engine]` | required |  |
| `renderer` | `'Renderer'` | required | The model family's token format. |
| `limits` | `Limits` | `Limits()` |  |
| `adapter` | `str \| None` | `None` | The adapter sampling now (None: the base model). |
| `version` | `int` | `0` | How many times weights have been published; recorded with every sampled token. |

**Methods**

- `@property def context_limit(self) -> int` — The longest turn the channel takes, and what it tells programs.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '') -> Generation` — Sample from one of the engines: the same one for a session every time, where its prompts' shared
  beginnings are cached.
- `async def publish(self, adapter: str, path: str, version: int | None = None) -> int` — Serve `adapter` (a LoRA directory every engine can read at `path`) from now on; returns the version it
  is served as: `version` if one is given (a policy's own numbering, which means the same in every process),
  or one more than the last. The adapter before stays loaded, so that a turn in progress finishes under the
  weights it began with; the one before that is dropped. Publishing what is being served changes nothing.
- `async def pause(self) -> None` — Hold new requests back, and wait for those in flight to finish.
- `def resume(self) -> None`
- `async def sleep(self) -> None`
- `async def wake(self) -> None`
- `def take(self) -> dict[str, float]` — What passed through since the last call: requests, tokens in and out, and throughput.
  `tokens_per_second` is everything generated over the time the channel was generating.
- `def close(self) -> None`

### `Engine`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Engine(Protocol)
```

One replica serving a model: in this process, or a client of a server elsewhere.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_model_len` | `int` | required | The longest sequence (prompt and completion) it accepts. |

**Methods**

- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None) -> Generation`
- `async def load_adapter(self, name: str, path: str) -> None` — Register a LoRA adapter under `name`; requests name it to sample from it.
- `async def remove_adapter(self, name: str) -> None`
- `async def sleep(self) -> None` — Free the accelerator (for a trainer that shares it).
- `async def wake(self) -> None`
- `@property def processes(self) -> Sequence[int]` — The processes it started on this machine, for whoever must end them if this process is killed.
- `def close(self) -> None`

### `Generation`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Generation
```

| Field | Type | Default | Description |
|---|---|---|---|
| `tokens` | `list[int]` | required |  |
| `logprobs` | `list[float]` | required | Of each sampled token, under the distribution it was sampled from. |
| `finish_reason` | `str` | required | `stop` (a stop token, included in `tokens`) or `length`. |

### `Limits`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Limits
```

What a turn may take, in tokens: the deployment's hardware decides, and code above it receives the outcome
(a context limit in a model's capability contract, a refusal when a context is full), never these numbers.

| Field | Type | Default | Description |
|---|---|---|---|
| `thinking` | `int` | `1024` | Tokens of thinking per turn before it is closed by force. |
| `answer` | `int` | `400` | Room for the answer after the thinking. |
| `sequence` | `int \| None` | `None` | The longest turn (prompt and completion): the smaller of what the engines accept and what the trainer can train on. A long prompt leaves less room to think, so that every turn can be trained on. |

## `rollout_train.recorder`

The model endpoint for trainable channels: token-exact recording.

### `ChatTemplateRenderer`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class ChatTemplateRenderer
```

Renders with the tokenizer's chat template; parses with a family's tool-call and thinking formats.

`end` ends an assistant turn, and so do `stops` (a family whose model stops to wait for a tool's response, say).
`options` are passed to the chat template (`enable_thinking`, say). `opens` is appended to every generation
prompt: for a family whose template leaves the thinking block for the model to open, a renderer whose
`ThinkingFormat` says the prompt opens it opens it here.

**Methods**

- `def __init__(self, name: str, tokenizer: Tokenizer, tool_calls: ToolCallFormat, thinking: ThinkingFormat | None, end: str, *, stops: Sequence[str] = (), options: Mapping[str, Any] | None = None, opens: str = '') -> None`
- `def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]`
- `def encode(self, text: str) -> list[int]`
- `def decode(self, tokens: Sequence[int]) -> str`
- `def stop_token_ids(self) -> list[int]`
- `def thinking_end_token_ids(self) -> list[int]`
- `def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message`

### `JsonToolCalls`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class JsonToolCalls
```

`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3, Hermes).

| Field | Type | Default | Description |
|---|---|---|---|
| `CALL` |  | `re.compile('<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>', re.DOTALL)` |  |

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]`

### `RecordedEndpoint`

*class* · `libraries/rollout-train/src/rollout_train/recorder/recorder.py`

```python
class RecordedEndpoint
```

Implements `ModelEndpoint` for one channel.

**Methods**

- `def __init__(self, recorder: Recorder, channel: Channel, sampling: SamplingParameters) -> None`
- `def describe(self, session_id: str) -> CapabilityContract`
- `def address(self, session_id: str, through: ModelEndpoint | None = None) -> ModelAddress` — Where a harness outside the run's own loop reaches this session, and the key that names it. Its samples
  go `through` an endpoint wrapping this one, if one is given (a runner's, which reports them to its hooks).
- `async def cancel(self, effect_id: str) -> None` — Nothing to do: the generation stops when the task awaiting `sample` is cancelled.
- `async def sample(self, request: SampleRequest) -> SampleResult`

### `Recorder`

*class* · `libraries/rollout-train/src/rollout_train/recorder/recorder.py`

```python
class Recorder
```

| Field | Type | Default | Description |
|---|---|---|---|
| `channels` | `Mapping[str, Channel]` | required |  |
| `base_url` | `str \| None` | `None` | Where `rollout_train.recorder.compat` serves this recorder, as harnesses reach it (None: it is not served). |

**Methods**

- `def endpoint(self, binding: RecordedModel) -> 'RecordedEndpoint'`
- `def export(self, session_id: str) -> list[Segment]` — The session's segments, oldest first (see the module's description).
- `def sessions(self, run_id: str) -> dict[str, list[Segment]]` — What each model slot of a run exports, by slot.
- `def forget(self, run_id: str) -> None`
- `async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int` — Serve new weights on a channel; returns the version they are served as.
- `def served(self, key: str) -> tuple[str, ModelEndpoint] | None` — The session a harness's key names, and the endpoint that samples for it.

### `Renderer`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class Renderer(Protocol)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `thinking` | `ThinkingFormat \| None` | required |  |

**Methods**

- `def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]` — The prompt: every message, then the generation prompt for the assistant's next turn.
- `def encode(self, text: str) -> list[int]`
- `def decode(self, tokens: Sequence[int]) -> str` — The text of tokens, special tokens and all: what `encode` reads back.
- `def stop_token_ids(self) -> list[int]` — Tokens that end an assistant turn.
- `def thinking_end_token_ids(self) -> list[int]` — Tokens that end thinking (to stop a thinking phase on), or none if it is not a single token.
- `def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message` — A sampled turn as a canonical assistant message.

### `Segment`

*class* · `libraries/rollout-train/src/rollout_train/recorder/recorder.py`

```python
class Segment
```

A piece of a session's trajectory: tokens that only grew, as the policy saw and continued them.

| Field | Type | Default | Description |
|---|---|---|---|
| `tokens` | `list[int]` | required |  |
| `spans` | `list[Span]` | required |  |
| `logprobs` | `list[float]` | required | Behavior logprobs of the tokens inside the spans, in order. |
| `channel` | `str` | `''` | The channel that sampled them. Its policy's versions are what the spans' `version`s count. |

**Methods**

- `@property def sampled(self) -> int`

### `Span`

*class* · `libraries/rollout-train/src/rollout_train/recorder/recorder.py`

```python
class Span
```

Tokens `start` to `end` (exclusive) of a segment were sampled by the policy, at weights `version`.

| Field | Type | Default | Description |
|---|---|---|---|
| `start` | `int` | required |  |
| `end` | `int` | required |  |
| `version` | `int` | required |  |
| `effect_id` | `str` | `''` | The sample that produced them: the `effect_id` its run's events know it by. |

### `ThinkingFormat`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class ThinkingFormat
```

How a family delimits thinking. Its generation prompt may already open the block.

| Field | Type | Default | Description |
|---|---|---|---|
| `open` | `str` | required |  |
| `close` | `str` | required |  |
| `prompt_opens` | `bool` | required | Whether the generation prompt ends inside an opened thinking block (the model only closes it). |
| `forced_close` | `str` | required | Text appended to end thinking that ran out of budget (masked from training). |

### `ToolCallFormat`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class ToolCallFormat(Protocol)
```

How a family writes tool calls in its output.

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]` — Split text into what precedes the calls and the calls; arguments are converted to their schema types.

### `XmlFunctionCalls`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class XmlFunctionCalls
```

`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` (Qwen3.5, Qwen3-Coder).

| Field | Type | Default | Description |
|---|---|---|---|
| `CALL` |  | `re.compile('<tool_call>\\s*<function=([^>\\s]+)>(.*?)</function>\\s*</tool_call>', re.DOTALL)` |  |
| `PARAMETER` |  | `re.compile('<parameter=([^>\\s]+)>\\n?(.*?)\\n?</parameter>', re.DOTALL)` |  |

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]`

## `rollout_train.profile`

A deployment, described and opened.

### `ChannelSpec`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class ChannelSpec
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | The checkpoint every engine of the channel serves. |
| `renderer` | `str` | required | `module:name` of the model family's renderer, called with `model`. |
| `engine` | `str` | required | `module:name` of what makes an engine, called with `model` and one entry of `engines`. |
| `engines` | `tuple[Mapping[str, Any], ...]` | `({},)` | One entry per replica: what that engine is told (its share of a GPU, which device, where it listens). |
| `thinking_tokens` | `int \| None` | `None` | Tokens of thinking per turn, and of answer after it, where the channel should not use `Limits`' own. |
| `answer_tokens` | `int \| None` | `None` |  |

### `NotEnoughMemory`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class NotEnoughMemory(Exception)
```

Stopping is better than exhausting the machine (a host may shut down rather than kill one process).

### `Platform`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class Platform
```

An open profile: `jobs` to run episodes with, a `trainer` to step, and the `policies` it trains.

**Methods**

- `def __init__(self, profile: Profile) -> None`
- `@classmethod async def start(cls, profile: Profile, stack: contextlib.AsyncExitStack) -> 'Platform'` — Start everything, registering with `stack` how each thing is stopped (the engines last).

### `Profile`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class Profile
```

| Field | Type | Default | Description |
|---|---|---|---|
| `directory` | `Path` | required | The run's state: adapters, the job's log, metrics, the monitor's feed. |
| `channels` | `Mapping[str, ChannelSpec]` | required |  |
| `trainer` | `TrainerSpec \| None` | `None` |  |
| `runner` | `str` | `'local'` | `local` runs episodes in this process; `durable` records them so that they survive it. |
| `serve` | `str \| None` | `None` | `host:port` to serve the rollout jobs and the model endpoint for harnesses on. |
| `address` | `str \| None` | `None` | The URL others reach `serve` at (by default `http://` and `serve`). |
| `tools` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | Each tool set by name: a URL, or `module:name` of what makes it, called with `directory`. |
| `ledger` | `Path \| None` | `None` | Where the run's tables and the policies' versions are kept, in files (by default `directory/ledger`). Runs that share it see each other's policies. |
| `blobs` | `Mapping[str, Any]` | `field(default_factory=dict[str, Any])` | Where episodes (and what programs store) are kept: `kind` is `module:name` of what makes the store, called with the other entries. Without one, files under `directory/blobs`. |
| `runs_gib` | `float` | `0.0` | System memory that must be available to admit runs. |
| `training_gib` | `float` | `0.0` | And to start a step of a colocated trainer. |
| `episodes_at_once` | `int` | `6` | The most episodes a run plays at once (whatever groups they are of): what the machine's engines and its memory for the programs' worlds can take. |
| `feed_runs` | `int \| None` | `None` | Episodes kept in the monitor's feed, where it should not keep `RunFeed`'s own number (the oldest are deleted). |

**Methods**

- `@classmethod def load(cls, path: Path, *, directory: Path | None = None) -> 'Profile'` — The profile a TOML file describes; `directory` replaces the file's (one profile, many runs). A key the
  file has and a profile does not is an error: a misspelt guard would otherwise be no guard.
- `async def open(self) -> AsyncGenerator['Platform']` — Start what the profile describes, and stop it on the way out (also if starting fails half way).

### `TrainerSpec`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class TrainerSpec
```

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | required | `module:name` of what makes the trainer, called with the channel's model and `settings`. |
| `channel` | `str` | required | The channel that serves the policy it trains. |
| `policy` | `str \| None` | `None` | The policy it trains, by name (by default the run directory's name). A policy that has versions is gone on with. |
| `colocated` | `bool` | `False` | Whether it shares the channels' accelerator: their engines then sleep while it steps. |
| `settings` | `Mapping[str, Any]` | `field(default_factory=dict[str, Any])` |  |

## `rollout_train.monitor`

A live web page over a job and its runs.

### `FeedReader`

*class* · `libraries/rollout-train/src/rollout_train/monitor/feed.py`

```python
class FeedReader
```

Reads a feed directory incrementally: each call picks up what was appended since the last. Of a run it
keeps a summary; the run's lines are read from its file when they are asked for. Its methods may be called from
several threads at once.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `def refresh(self) -> None`
- `def runs(self) -> list[dict[str, Any]]` — Every run in the feed, newest first: its labels, state, rewards and how much it has done.
- `def job(self, after: int = 0) -> list[dict[str, Any]]` — What the rollout job did, from index `after` on: tickets, episodes, published weights, the loop's results
  and steps, the engines' throughput.
- `def lines(self, run_id: str, after: int = 0) -> list[dict[str, Any]]` — A run's lines from index `after` on.

### `plain`

*function* · `libraries/rollout-train/src/rollout_train/monitor/feed.py`

```python
def plain(message: Message) -> dict[str, JsonValue]
```

A message as the page shows it: its text, its reasoning, the tools it called and the results it carries.

### `RunFeed`

*class* · `libraries/rollout-train/src/rollout_train/monitor/feed.py`

```python
class RunFeed(RunHooks, JobHooks)
```

Writes every run's events and samples under `directory`, one file per run, as they happen; and what a
rollout job did, at the level its caller thinks at, in one file more.

`keep` bounds the directory: when more runs than that have files, the oldest are deleted. A directory has one
writer at a time: runs that an earlier writer left without an end (its process was stopped) are marked cancelled
when the next one starts, so that a monitor does not show them running for ever.

**Methods**

- `def __init__(self, directory: Path, *, keep: int = 200) -> None`
- `def on_event(self, event: RunEvent) -> None`
- `def on_job(self, event: Mapping[str, JsonValue]) -> None`
- `def on_sample(self, sample: ModelSample) -> None`
- `def close(self) -> None`

### `System`

*class* · `libraries/rollout-train/src/rollout_train/monitor/system.py`

```python
class System
```

**Methods**

- `def __init__(self, directory: Path, feed: FeedReader) -> None`
- `async def snapshot(self) -> dict[str, Any]` — Where everything stands now: the runs' groups that are not done with and the ones that are, the
  policies' versions, the jobs, what each channel serves and how fast, the machine, and what is kept.
- `async def group(self, run: str, number: int) -> dict[str, Any] | None` — One group: what was decided (the row and its start), its stage, its episodes with what each reported,
  its step and the version it made, and its outcome.
- `async def episode(self, run_id: str, after: int = 0) -> dict[str, Any]` — One episode: the run's lines from index `after` on (from the feed, or, once the feed has let it go, its
  replies and tool calls from the events the job kept), from which its rollouts (one per model slot) are
  drawn; what it reported when it ended; and where it sits: its job, its group and its labels.

## `rollout_train.testing`

Test doubles: a scripted engine and a readable token format.

### `Characters`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class Characters
```

A tokenizer of one token per character.

**Methods**

- `def encode(self, text: str, add_special_tokens: bool = False) -> list[int]`
- `def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str`

### `plain_channel`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def plain_channel(script: Sequence[tuple[str, str]] = (), *, name: str = 'policy', **options: Any) -> Channel
```

A channel over a scripted engine in the plain format; `always=` repeats a script for ever.

### `plain_renderer`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def plain_renderer(model: str) -> Renderer
```

### `PlainRenderer`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class PlainRenderer
```

A token format for tests: each message is `role: text` on a line, a tool call is `call NAME {json}`, and a
turn ends with the line. A reply renders back exactly as it was sampled, so a conversation that only grows is
one segment.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` |  | `'plain'` |  |
| `thinking` |  | `None` |  |

**Methods**

- `def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]`
- `def encode(self, text: str) -> list[int]`
- `def decode(self, tokens: Sequence[int]) -> str`
- `def stop_token_ids(self) -> list[int]`
- `def thinking_end_token_ids(self) -> list[int]`
- `def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message`

### `sample_request`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def sample_request(messages: list[Message], effect_id: str = 'r_1:0:0', *, session_id: str = 'r_1/ada', tools: Sequence[ToolSpecification] = ()) -> SampleRequest
```

### `scripted_engine`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def scripted_engine(model: str, **options: Any) -> ScriptedEngine
```

An engine whose policy says yes and no in turn; `fails=true` makes one that cannot start.

### `ScriptedEngine`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class ScriptedEngine
```

Answers each generate with the next scripted (text, finish reason), or with `always` once the script is
spent; logprobs are -0.5 per token. Keeps what it was asked and told.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_model_len` |  | `32768` |  |
| `processes` | `Sequence[int]` | `()` |  |

**Methods**

- `def __init__(self, tokenizer: Tokenizer, script: Sequence[tuple[str, str]] = (), *, always: Sequence[tuple[str, str]] = ()) -> None`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None) -> Generation`
- `async def load_adapter(self, name: str, path: str) -> None`
- `async def remove_adapter(self, name: str) -> None`
- `async def sleep(self) -> None`
- `async def wake(self) -> None`
- `def close(self) -> None`

## `rollout_durable`

A runner whose runs survive their process, on DBOS.

### `DurableRunContext`

*class* · `implementations/rollout-durable/src/rollout_durable/context.py`

```python
class DurableRunContext(LocalRunContext)
```

**Methods**

- `def __init__(self, run_id: str, endpoints: Mapping[str, ModelEndpoint], *, started_at: datetime, context_hints: ContextHints | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environment_service: EnvironmentService | None = None, blobs: Blobs | None = None, conversation: ConversationKey | None = None, on_event: Callable[[RunEvent], None] | None = None, mark_attempt: Callable[[str], bool] = lambda effect_id: True) -> None`
- `def now(self) -> datetime`
- `async def wait_for_message(self, wait: WaitFor) -> Envelope | None`
- `async def take_steering_messages(self) -> list[Envelope]`
- `async def interruptible[T](self, reply: Awaitable[T]) -> T`
- `async def undelivered(self) -> list[Envelope]` — Messages the run never consumed, including any still in the inbox; handed to the next run.

### `DurableRunHandle`

*class* · `implementations/rollout-durable/src/rollout_durable/runner.py`

```python
class DurableRunHandle
```

A durable run, read from the store: valid across processes and restarts.

**Methods**

- `def __init__(self, runner: 'DurableRunner', run_id: str) -> None`
- `@property def run_id(self) -> str`
- `@property def outcome(self) -> RunOutcome | None`
- `@property def done(self) -> bool`
- `async def result(self) -> RunOutcome`
- `async def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]`
- `def recorded_events(self) -> list[RunEvent]` — Every event recorded so far.

### `DurableRunner`

*class* · `implementations/rollout-durable/src/rollout_durable/runner.py`

```python
class DurableRunner(MessageRouter)
```

Implements `Runner` on DBOS. Call `await launch()` before use and `await close()` after.

**Methods**

- `def __init__(self, directory: Path, *, providers: Mapping[str, EndpointFactory] | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environments: EnvironmentService | None = None, blobs: Blobs | None = None, recorder: RecordedEndpoints | None = None, hooks: Sequence[RunHooks] = (), application: str = 'rollout', evict_after: timedelta | None = timedelta(minutes=5), eviction_interval: float = 5.0, database: str | Database | None = None, runner_id: str | None = None, heartbeat_interval: float = 2.0, takeover_after: timedelta = timedelta(seconds=15)) -> None` — `evict_after`: unload runs that have waited this long for a message (None keeps every run resident);
  `eviction_interval`: how often, in seconds, to look for runs to evict or wake
  (docs/implementations/rollout-durable/eviction.md).
  
    `database`: a Postgres URL (or `Database`) shared with other runners; without it, state is SQLite in
  `directory` and this runner is the only one. `runner_id` names this runner among them: a runner restarted
  with its id puts its unfinished runs back on the queue at once, without waiting for a takeover.
  `takeover_after`: how long a runner's heartbeat may stop before another runner recovers its runs.
  `directory` holds local files either way.
- `async def launch(self) -> None` — Start DBOS, which recovers the runs a crash left unfinished, and follow them.
- `async def close(self) -> None`
- `def deploy(self, deployment: Deployment) -> None`
- `def run(self, run_id: str) -> DurableRunHandle`
- `def conversation_of(self, run_id: str) -> ConversationKey | None` — The conversation a run serves, if any.
- `def conversation_runs(self, deployment: str, key: str) -> list[DurableRunHandle]`
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None) -> DurableRunHandle`
- `async def cancel(self, run_id: str, *, reason: str) -> None` — Ask the run to stop at its next effect, wait or turn boundary; `teardown` runs.
- `async def execute(self, run_id: str, specification_json: dict[str, Any], conversation_json: dict[str, Any] | None, labels: dict[str, str], started_at: str) -> dict[str, Any]`
- `def after_run(self, run_id: str, result: dict[str, Any]) -> None` — Called by the workflow when a run ends, where it ran: start the follow-up outside the workflow.

### `RunCancelled`

*class* · `implementations/rollout-durable/src/rollout_durable/context.py`

```python
class RunCancelled(Exception)
```

A cancellation request reached the run; the program unwinds and `teardown` runs.

**Methods**

- `def __init__(self, reason: str) -> None`

### `RunStore`

*class* · `implementations/rollout-durable/src/rollout_durable/store.py`

```python
class RunStore
```

**Methods**

- `def __init__(self, database: Database | Path) -> None` — A `Database`, or the path of a SQLite file.
- `def create_run(self, run_id: str, specification: JsonValue, conversation: str | None, conversation_key: JsonValue = None) -> None`
- `def finish_run(self, run_id: str, status: str, outcome: JsonValue) -> None`
- `def run(self, run_id: str) -> RunRecord | None`
- `def read_all_run_ids(self) -> list[str]`
- `def evict(self, run_id: str, wake_at: str | None) -> None`
- `def wake(self, run_id: str, at: str) -> None`
- `def touch(self, run_id: str, at: str) -> None` — Record that the run was messaged: it must not be evicted until it suspends again.
- `def last_activity(self, run_id: str) -> str | None` — When the run was last messaged or woken.
- `def evictions(self) -> int` — How many times runs were evicted, in total.
- `def is_evicted(self, run_id: str) -> bool`
- `def due_for_waking(self, now: str) -> list[str]` — Evicted runs whose wait times out by `now`.
- `def last_events(self, run_ids: list[str]) -> list[RunEvent]` — The latest event of each of these runs that is running and resident.
- `def live_run(self, address: str) -> str | None`
- `def conversation_key(self, address: str) -> JsonValue`
- `def conversation_runs(self, address: str) -> list[str]`
- `def is_claimed(self, message_id: str) -> bool`
- `def claim_message(self, message_id: str, address: str) -> bool` — Record a message as delivered; False if it already was (a retry).
- `def mark_attempt(self, effect_id: str) -> bool` — Record that a guarded effect is starting; False if an earlier attempt already started it.
- `def heartbeat(self, runner_id: str, at: float) -> None`
- `def stale_runners(self, before: float) -> list[str]` — Runners whose last heartbeat is older than `before`.
- `def forget_runner(self, runner_id: str) -> None`
- `def append(self, event: RunEvent) -> None`
- `def events(self, run_id: str, from_seq: int = 0) -> list[RunEvent]`
- `async def changed(self, run_id: str, wait_seconds: float) -> None` — Wait until the run records something, or `wait_seconds` pass (other processes write without notifying).
- `def close(self) -> None` — Close the database if this store opened it (a shared `Database` is closed by its owner).

## `rollout_vllm`

An engine on vLLM.

### `VllmEngine`

*class* · `implementations/rollout-vllm/src/rollout_vllm/engine.py`

```python
class VllmEngine
```

**Methods**

- `def __init__(self, model: str, *, gpu_memory_utilization: float = 0.72, max_model_len: int = 8192, max_num_seqs: int = 32, max_num_batched_tokens: int = 4096, max_lora_rank: int = 32, max_loras: int = 2, language_model_only: bool = True, speculative: Mapping[str, Any] | None = None, seed: int = 0) -> None`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None) -> Generation`
- `async def load_adapter(self, name: str, path: str) -> None` — Register a LoRA adapter (a PEFT directory) under `name`; samples name it to use it.
- `async def remove_adapter(self, name: str) -> None`
- `async def sleep(self) -> None` — Free the GPU: the cache is discarded and the weights dropped (they are read again on waking).
- `async def wake(self) -> None`
- `@property def processes(self) -> list[int]`
- `def close(self) -> None`

## `rollout_lora`

A trainer for 4-bit checkpoints with LoRA.

### `LoraSettings`

*class* · `implementations/rollout-lora/src/rollout_lora/settings.py`

```python
class LoraSettings
```

| Field | Type | Default | Description |
|---|---|---|---|
| `rank` | `int` | `32` | Of the adapter. Its scaling is twice the rank (`alpha`). |
| `learning_rate` | `float` | `5e-05` |  |
| `clip_low` | `float` | `0.2` |  |
| `clip_high` | `float` | `0.28` | A token's ratio to its logprob at the step's start is clipped to 1 - `clip_low` .. 1 + `clip_high` (DAPO's clip-higher). |
| `segment_clip_low` | `float` | `0.0003` |  |
| `segment_clip_high` | `float` | `0.0004` | With `ratio = "segment"`, the segment's ratio is clipped to 1 - `segment_clip_low` .. 1 + `segment_clip_high` (GSPO's). |
| `truncate` | `float \| None` | `2.0` | The most a token's importance weight (its logprob at the step's start against the one it was sampled at) may be (None: not truncated). |
| `tokens_per_step` | `int` | `4096` | Sampled tokens per optimizer step (gradients accumulate over segments until then). Adam moves a weight by at most the learning rate a step, so how far an update goes is set by how many steps its tokens make. |
| `max_kl` | `float \| None` | `0.02` | Stop the pass when a minibatch, before its step, finds the policy this far from where the step began (in nats per token, estimated on the sampled tokens). |
| `max_gradient_norm` | `float` | `1.0` |  |
| `segment_tokens` | `int \| None` | `None` | The longest segment a step can hold on its accelerator (None: any). Longer ones are left out and counted (`segments_too_long`): one too long would end or stall the whole step. Leaving segments out biases training, so whoever serves the policy takes this as the longest turn to sample; the count says whether that held. |
| `layer_inputs_on_host` | `bool` | `False` | Keep each layer's input in pinned system memory between the forward and backward passes, instead of on the GPU (`rollout_lora.activations`): a quarter of a megabyte a token, for Qwen3.5-9B. |
| `mlp_rows` | `int \| None` | `None` | Run each layer's MLP over this many tokens at a time when it is computed again for the backward pass, and in passes without a gradient (None: the whole segment at once). The same numbers, at a lower peak. |
| `segments_per_step` | `int \| None` | `None` | How many segments a step can afford (None: any number). |
| `objective` | `str` | `'policy_gradient'` | `policy_gradient`: the clipped policy gradient over the sampled tokens, each weighted by its segment's advantage, with an importance weight for where they were sampled. `likelihood`: raise the log-likelihood of the sampled tokens, each weighted by its segment's advantage (imitation: what was sampled is what to do), with no ratio, weight or stop at `max_kl` (`rollout_lora.objectives`). |
| `ratio` | `str` | `'token'` | `token`: a ratio for each token (PPO). `segment`: one for each segment, the geometric mean of its tokens' (GSPO). |

**Methods**

- `@property def loss(self) -> Objective` — The objective a step takes, by these settings.
- `@property def alpha(self) -> float`

### `LoraTrainer`

*class* · `implementations/rollout-lora/src/rollout_lora/trainer.py`

```python
class LoraTrainer
```

Trains a LoRA adapter over `model`'s checkpoint, one step at a time, each in a fresh process on the GPU
(`rollout_lora.worker`). It keeps nothing between steps: a step starts from the adapter and the optimizer's
state it is given and leaves the new ones where it is told. `settings` are `LoraSettings`' fields.

**Methods**

- `def __init__(self, model: str, **settings: Any) -> None`
- `async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step`

## `rollout_qwen`

Renderers for the Qwen model families.

### `qwen3`

*function* · `implementations/rollout-qwen/src/rollout_qwen/__init__.py`

```python
def qwen3(model: str | Tokenizer) -> Renderer
```

Qwen3: JSON tool calls, and thinking the model opens. `model` is a checkpoint's name, or its tokenizer.

### `qwen35`

*function* · `implementations/rollout-qwen/src/rollout_qwen/__init__.py`

```python
def qwen35(model: str | Tokenizer) -> Renderer
```

Qwen3.5: XML function calls, and thinking the prompt opens. `model` is a checkpoint's name, or its tokenizer.

### `tokenizer_of`

*function* · `implementations/rollout-qwen/src/rollout_qwen/__init__.py`

```python
def tokenizer_of(model: str) -> Tokenizer
```

The tokenizer of a checkpoint, by its name or path.

## `rollout_gemma`

Renderers for the Gemma model families.

### `arguments`

*function* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
def arguments(text: str) -> dict[str, JsonValue]
```

A call's arguments (what is between its braces) as values: strings quoted with `<|"|>`, numbers, `true`,
`false`, `null`, objects in braces and lists in brackets; keys are bare (or quoted).

### `gemma4`

*function* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
def gemma4(model: str | Tokenizer) -> Renderer
```

Gemma 4, thinking: the template is asked for thinking, and the generation prompt opens the thought channel
(as Gemma's template itself does after a tool's response), so that a thinking budget can close it. `model` is a
checkpoint's name, or its tokenizer.

### `GemmaFunctionCalls`

*class* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
class GemmaFunctionCalls
```

`<|tool_call>call:name{key:<|"|>text<|"|>,count:3,flag:true,nested:{...},items:[...]}<tool_call|>`.

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]`

### `tokenizer_of`

*function* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
def tokenizer_of(model: str) -> Tokenizer
```

The tokenizer of a checkpoint, by its name or path.

## `rollout_computers`

Environment backends: services that give runs computers.

### `ImageStore`

*class* · `implementations/rollout-computers/src/rollout_computers/images.py`

```python
class ImageStore
```

Resolves image names (`alpine`, `alpine:3.24.2`) to verified, cached root filesystem tarballs.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def tarball(self, image: str) -> Path`

### `LocalEnvironments`

*class* · `implementations/rollout-computers/src/rollout_computers/local.py`

```python
class LocalEnvironments
```

Implements `EnvironmentService`.

**Methods**

- `def __init__(self, directory: Path, *, variables: Mapping[str, str] | None = None) -> None` — `variables` replaces the inherited environment variables when given.
- `def workspace(self, environment_id: str) -> Path`
- `async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None`
- `async def execute(self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = '') -> ExecutionResult`
- `async def put(self, environment_id: str, path: str, data: bytes) -> None`
- `async def get(self, environment_id: str, path: str) -> bytes`
- `async def destroy(self, environment_id: str) -> None`

### `NamespaceEnvironments`

*class* · `implementations/rollout-computers/src/rollout_computers/namespaces.py`

```python
class NamespaceEnvironments
```

Implements `EnvironmentService`.

**Methods**

- `def __init__(self, directory: Path, images: ImageStore | None = None) -> None`
- `def root(self, environment_id: str) -> Path`
- `async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None`
- `async def execute(self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = '') -> ExecutionResult`
- `async def put(self, environment_id: str, path: str, data: bytes) -> None`
- `async def get(self, environment_id: str, path: str) -> bytes`
- `async def destroy(self, environment_id: str) -> None`

## `rollout_computers.tools`

Tools for agents that work on a computer: shell, files, edits and images.

### `apply_edits`

*function* · `implementations/rollout-computers/src/rollout_computers/tools.py`

```python
def apply_edits(original: str, edits: list[tuple[str, str]], path: str = 'the file') -> str
```

Apply exact replacements, each matched once against `original`. Line endings and a byte-order mark are kept:
matching ignores the difference between CRLF and LF.

### `ComputerTools`

*class* · `implementations/rollout-computers/src/rollout_computers/tools.py`

```python
class ComputerTools
```

`@tool` methods over the environment `environment_id`. Mix into a `Task`.

| Field | Type | Default | Description |
|---|---|---|---|
| `environment_id` | `str \| None` | `None` |  |

**Methods**

- `async def shell(self, run: RunContext, command: str, timeout_seconds: float = 120) -> ToolResult` — Run a shell command in your working directory. Each call starts a fresh shell, and nothing it started
  keeps running afterwards. Returns the exit code and the combined standard output and error. Long output keeps
  its last 2000 lines or 50 KB, and the full output is saved to a file you can read.
- `async def read_file(self, run: RunContext, path: str, offset: int = 1, limit: int | None = None) -> str` — Read a text file (relative paths are under your working directory). Returns at most 2000 lines or 50 KB
  from line `offset` (1-based), and says how to continue when there is more. `limit` caps the number of lines.
  Use read_image for images.
- `async def write_file(self, run: RunContext, path: str, content: str) -> str` — Write a text file (relative paths are under your working directory), replacing it if it exists and
  creating parent directories.
- `async def edit_file(self, run: RunContext, path: str, edits: list[Replacement]) -> str` — Edit a text file by exact replacement. Each old_text must occur exactly once in the original file, and
  edits must not overlap: every edit is matched against the original, not against the result of the others.
  For several changes in one file, make one call with several edits. Keep each old_text short but unique.
- `async def read_image(self, run: RunContext, path: str) -> ToolResult` — Look at an image file (PNG, JPEG, GIF, WebP, BMP and other common formats; relative paths are under your
  working directory). Large images are scaled down.
- `def computer(self, run: RunContext) -> Environment`

### `page_text`

*function* · `implementations/rollout-computers/src/rollout_computers/tools.py`

```python
def page_text(text: str, path: str, offset: int, limit: int | None) -> str
```

Up to `MAX_READ_LINES` lines or `MAX_READ_BYTES` bytes of `text` from line `offset`, with a hint to continue.

### `prepare_image`

*function* · `implementations/rollout-computers/src/rollout_computers/tools.py`

```python
def prepare_image(data: bytes) -> tuple[bytes, str, str]
```

An image the model can read: common formats within the limits as they are, others converted to PNG and
scaled down to fit. Returns the bytes, their media type and a description.

### `Replacement`

*class* · `implementations/rollout-computers/src/rollout_computers/tools.py`

```python
class Replacement(BaseModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `old_text` | `str` | `Field(description='Exact text to replace. It must occur exactly once in the original file.')` |  |
| `new_text` | `str` | `Field(description='The text to put in its place.')` |  |

## `rollout_openai`

A model endpoint for the OpenAI Responses API, on an API key or a Codex login.

### `ApiKey`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class ApiKey
```

An OpenAI API key against the public Responses API.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required |  |
| `base_url` | `str` | `'https://api.openai.com/v1'` |  |
| `accepts_max_output_tokens` | `bool` | `True` |  |

**Methods**

- `async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]`
- `@property def url(self) -> str`

### `codex_provider`

*function* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
def codex_provider(contract: ResponsesContract | None = None, *, blobs: Blobs | None = None) -> Callable[[DirectModel], ResponsesEndpoint]
```

An endpoint factory for `LocalRunner(providers={"codex": codex_provider()})`, using the local Codex login.

### `CodexLogin`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class CodexLogin
```

ChatGPT account tokens from a local Codex login.

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | `Path.home() / '.codex' / 'auth.json'` |  |
| `url` | `str` | `'https://chatgpt.com/backend-api/codex/responses'` |  |
| `refresh_margin_seconds` | `int` | `300` |  |
| `accepts_max_output_tokens` | `bool` | `False` |  |

**Methods**

- `async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]`

### `Credentials`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class Credentials(Protocol)
```

Where requests go and how they authenticate.

**Methods**

- `async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]`
- `@property def url(self) -> str`
- `@property def accepts_max_output_tokens(self) -> bool` — Whether the backend accepts `max_output_tokens`.

### `ResponsesContract`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class ResponsesContract
```

The capability contract the endpoint advertises; the provider does not report it.

| Field | Type | Default | Description |
|---|---|---|---|
| `context_limit` | `int` | `200000` |  |
| `max_output_tokens` | `int` | `32000` |  |

### `ResponsesEndpoint`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class ResponsesEndpoint
```

Serves one model through the Responses API. Direct adapters do not deduplicate: a retried effect re-samples.

**Methods**

- `def __init__(self, credentials: Credentials, model: str, *, sampling: SamplingParameters | None = None, contract: ResponsesContract | None = None, client: httpx.AsyncClient | None = None, timeout: float = 600.0, blobs: Blobs | None = None) -> None` — `blobs` reads the bytes of `Media` blocks (images); without it, a context with media cannot be sent.
- `def describe(self, session_id: str) -> CapabilityContract`
- `async def cancel(self, effect_id: str) -> None` — Nothing to do: the request stops when the task awaiting `sample` is cancelled.
- `async def sample(self, request: SampleRequest) -> SampleResult`
- `def request_body(self, request: SampleRequest, media: Mapping[str, bytes] | None = None) -> dict[str, JsonValue]` — The Responses API request for a sample request (public for tests and debugging). `media` holds the bytes
  of the context's `Media` blocks by SHA-256.

## `rollout_s3`

Blobs in S3 or any S3-compatible object store.

### `S3BlobStore`

*class* · `implementations/rollout-s3/src/rollout_s3/store.py`

```python
class S3BlobStore
```

Implements `Blobs` in an S3 bucket.

**Methods**

- `def __init__(self, bucket: str, *, prefix: str = 'blobs/', endpoint_url: str | None = None, region: str | None = None, client: 'S3Client | None' = None) -> None` — `client` replaces the boto3 client this store would create (e.g. with custom credentials).
- `@classmethod def from_url(cls, url: str, **options: Any) -> 'S3BlobStore'` — A store for `s3://bucket/prefix`.
- `async def put(self, data: bytes, media_type: str) -> BlobReference`
- `async def read(self, reference: BlobReference) -> bytes`
- `async def delete(self, reference: BlobReference) -> None`
