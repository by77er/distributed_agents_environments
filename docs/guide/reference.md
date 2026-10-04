# API reference

Generated from the source by `scripts/generate_reference.py`; do not edit by hand. Every public name,
grouped by module, alphabetically. Types and defaults appear as written in the source. The
[guide](README.md) explains how the pieces fit together.

## Contents

- **[`rollout.harness`](#rolloutharness)** — Writing tasks, agents and programs; runners; memory; tool sets. [`Address`](#address), [`Agent`](#agent), [`agent_program`](#agent_program), [`AgentProgram`](#agentprogram), [`bind`](#bind), [`Blobs`](#blobs), [`Capacity`](#capacity), [`CompactingAgent`](#compactingagent), [`ContextHints`](#contexthints), [`ConversationKey`](#conversationkey), [`DeduplicatingToolSet`](#deduplicatingtoolset), [`DeliveryMode`](#deliverymode), [`DeliveryPolicy`](#deliverypolicy), [`Deployment`](#deployment), [`DirectModel`](#directmodel), [`Effects`](#effects), [`End`](#end), [`Ending`](#ending), [`EndpointModel`](#endpointmodel), [`Envelope`](#envelope), [`Environment`](#rolloutharnessenvironment), [`Environments`](#environments), [`EnvironmentService`](#environmentservice), [`EnvironmentSpecification`](#environmentspecification), [`ExecutionResult`](#executionresult), [`FileBlobStore`](#fileblobstore), [`History`](#history), [`HistoryShape`](#historyshape), [`instantiate`](#instantiate), [`Interrupted`](#interrupted), [`InvalidObservation`](#invalidobservation), [`Lease`](#lease), [`LeaseRefused`](#leaserefused), [`Leases`](#leases), [`Memory`](#memory), [`MemoryLeases`](#memoryleases), [`MessageRouter`](#messagerouter), [`Model`](#model), [`ModelBinding`](#modelbinding), [`ModelSample`](#modelsample), [`ModelSlot`](#modelslot), [`Mount`](#mount), [`Network`](#network), [`NoCapacity`](#nocapacity), [`Observation`](#observation), [`Pool`](#pool), [`PoolBinding`](#poolbinding), [`Priority`](#priority), [`Process`](#process), [`Program`](#program), [`ProgramReference`](#programreference), [`Provider`](#provider), [`Reach`](#reach), [`RecordedEndpoints`](#recordedendpoints), [`RecordedModel`](#recordedmodel), [`register`](#register), [`resolve`](#resolve), [`rollout`](#rollout), [`RunBinding`](#runbinding), [`RunContext`](#runcontext), [`RunHandle`](#runhandle), [`RunHooks`](#runhooks), [`Runner`](#runner), [`RunNotLive`](#runnotlive), [`RunOutcome`](#runoutcome), [`RunSpecification`](#runspecification), [`RunStatus`](#runstatus), [`SamplingParameters`](#samplingparameters), [`Sandbox`](#sandbox), [`SandboxLimits`](#sandboxlimits), [`SandboxLost`](#sandboxlost), [`SandboxPool`](#sandboxpool), [`SandboxSpec`](#sandboxspec), [`Scratch`](#scratch), [`Task`](#task), [`tool`](#tool), [`ToolBinding`](#toolbinding), [`Tools`](#tools), [`ToolSet`](#toolset), [`Turn`](#turn), [`WaitFor`](#waitfor), [`with_row`](#with_row)
- **[`rollout.contracts`](#rolloutcontracts)** — Types that cross layers: canonical content, identifiers, digests, effects, events. [`address_of`](#address_of), [`AddressableEndpoint`](#addressableendpoint), [`arguments_digest`](#arguments_digest), [`BlobReference`](#blobreference), [`Block`](#block), [`canonical_json`](#canonical_json), [`CapabilityContract`](#capabilitycontract), [`Conflict`](#conflict), [`context_digests`](#context_digests), [`ContextDelta`](#contextdelta), [`ContextOverflow`](#contextoverflow), [`ContractModel`](#contractmodel), [`ContractViolation`](#contractviolation), [`digest`](#digest), [`effect_id`](#effect_id), [`EffectIdentity`](#effectidentity), [`EffectKind`](#effectkind), [`EffectStatus`](#effectstatus), [`EMPTY_DIGEST`](#empty_digest), [`FinishReason`](#finishreason), [`FrozenSequence`](#frozensequence), [`InternalError`](#internalerror), [`Media`](#media), [`Message`](#message), [`message_digest`](#message_digest), [`ModelAddress`](#modeladdress), [`ModelEndpoint`](#modelendpoint), [`ModelEndpointError`](#modelendpointerror), [`NamedToolChoice`](#namedtoolchoice), [`new_message_id`](#new_message_id), [`new_run_id`](#new_run_id), [`new_ulid`](#new_ulid), [`OutcomeUnknown`](#outcomeunknown), [`Overloaded`](#overloaded), [`Reasoning`](#reasoning), [`ReasoningScope`](#reasoningscope), [`ResultBlock`](#resultblock), [`RetryClass`](#retryclass), [`Role`](#role), [`RUN_EVENT_SCHEMA_VERSION`](#run_event_schema_version), [`RunEvent`](#runevent), [`RunEventType`](#runeventtype), [`RunFailureClass`](#runfailureclass), [`SampleRequest`](#samplerequest), [`SampleResult`](#sampleresult), [`session_id`](#session_id), [`SessionIdentity`](#sessionidentity), [`spec_hash`](#spec_hash), [`TERMINAL_EVENT_TYPES`](#terminal_event_types), [`Text`](#text), [`ToolCall`](#toolcall), [`ToolChoice`](#toolchoice), [`ToolChoiceMode`](#toolchoicemode), [`ToolResult`](#toolresult), [`ToolResultBlock`](#toolresultblock), [`ToolSpecification`](#toolspecification), [`Usage`](#usage)
- **[`rollout.environment`](#rolloutenvironment)** — What a run trains on and an eval measures: rows, starts, eval data, what results say. [`binding_for`](#binding_for), [`Description`](#description), [`drawn`](#drawn), [`Environment`](#rolloutenvironmentenvironment), [`held_out`](#held_out), [`Row`](#row), [`Start`](#start), [`start_key`](#start_key), [`train_start`](#train_start)
- **[`rollout.curriculum`](#rolloutcurriculum)** — Which row to train on next, and gates on evals. [`Curriculum`](#curriculum), [`curriculum_of`](#curriculum_of), [`GroupResult`](#groupresult), [`solved_share`](#solved_share)
- **[`rollout.local`](#rolloutlocal)** — The runner in this process. [`EndpointFactory`](#endpointfactory), [`LocalRunContext`](#localruncontext), [`LocalRunHandle`](#localrunhandle), [`LocalRunner`](#localrunner), [`RewardAssignment`](#rewardassignment)
- **[`rollout.testing`](#rollouttesting)** — Test doubles: a scripted model endpoint and helpers. [`events_of`](#rollouttestingevents_of), [`FakeSandbox`](#fakesandbox), [`FakeSandboxes`](#fakesandboxes), [`LedgerEndpoint`](#ledgerendpoint), [`LedgerEnvironments`](#ledgerenvironments), [`local_run`](#local_run), [`payload`](#payload), [`read_ledger`](#read_ledger), [`ScriptedModelEndpoint`](#scriptedmodelendpoint), [`ScriptedReply`](#scriptedreply), [`tool_call_reply`](#tool_call_reply)
- **[`rollout_train.rollouts`](#rollout_trainrollouts)** — Episodes a run asks for in the ledger, claimed and played by runners, and read back. [`Episode`](#episode), [`EpisodeRunner`](#episoderunner), [`episodes_of`](#episodes_of), [`events_of`](#rollout_trainrolloutsevents_of), [`Hooks`](#hooks), [`loaded`](#loaded), [`Outcome`](#outcome), [`Plan`](#plan), [`plan`](#plan), [`playing`](#playing), [`Record`](#record), [`Recorded`](#recorded), [`stored`](#stored), [`Trajectory`](#trajectory)
- **[`rollout_train.sandboxes`](#rollout_trainsandboxes)** — Sandboxes' leases beside the ledger, each ending with its episode's claim. [`admits`](#admits), [`ended`](#ended), [`FileLeases`](#fileleases), [`keep`](#keep), [`leases_of`](#leases_of), [`sweep`](#sweep)
- **[`rollout_train`](#rollout_train)** — The training loop, the group algorithm, evals, and what they ask of a trainer. [`Algorithm`](#algorithm), [`Batch`](#batch), [`Budget`](#budget), [`Changeable`](#changeable), [`Checkpoint`](#checkpoint), [`Checkpoints`](#checkpoints), [`Colocated`](#colocated), [`Dataset`](#dataset), [`dataset_of`](#dataset_of), [`evaluate`](#evaluate), [`Fence`](#fence), [`Fenced`](#fenced), [`FileLedger`](#fileledger), [`Files`](#files), [`Follower`](#follower), [`group_advantages`](#group_advantages), [`Grpo`](#grpo), [`Ledger`](#ledger), [`make_dataset`](#make_dataset), [`make_suite`](#make_suite), [`Manifest`](#manifest), [`record_serving`](#record_serving), [`Result`](#result), [`results`](#results), [`Retention`](#retention), [`Schedule`](#schedule), [`Serving`](#serving), [`Step`](#step), [`StepFailed`](#stepfailed), [`Suite`](#suite), [`suite_for`](#suite_for), [`suite_of`](#suite_of), [`train`](#train), [`Trained`](#trained), [`trained`](#trained), [`Trainer`](#trainer), [`wanted`](#wanted), [`Weighted`](#weighted)
- **[`rollout_train.inference`](#rollout_traininference)** — Channels: trainable models being served, and what they ask of an engine. [`Channel`](#channel), [`Connection`](#connection), [`Engine`](#engine), [`Generation`](#generation), [`Limits`](#limits), [`RemoteChannel`](#remotechannel), [`RemoteEngine`](#remoteengine), [`Route`](#route), [`Routes`](#routes), [`Sampler`](#sampler), [`Unserved`](#unserved)
- **[`rollout_train.recorder`](#rollout_trainrecorder)** — The model endpoint for trainable channels: token-exact recording. [`ChatTemplateRenderer`](#chattemplaterenderer), [`JsonToolCalls`](#jsontoolcalls), [`RecordedEndpoint`](#recordedendpoint), [`Recorder`](#recorder), [`Renderer`](#renderer), [`Segment`](#segment), [`Span`](#span), [`ThinkingFormat`](#thinkingformat), [`ToolCallFormat`](#toolcallformat), [`XmlFunctionCalls`](#xmlfunctioncalls)
- **[`rollout_train.profile`](#rollout_trainprofile)** — A deployment, described and opened. [`ChannelSpec`](#channelspec), [`EvalsSpec`](#evalsspec), [`NotEnoughMemory`](#notenoughmemory), [`Platform`](#platform), [`Profile`](#profile), [`TrainerSpec`](#trainerspec)
- **[`rollout_train.monitor`](#rollout_trainmonitor)** — A live web page over every run of a ledger. [`FeedReader`](#feedreader), [`plain`](#plain), [`RunFeed`](#runfeed), [`System`](#system)
- **[`rollout_train.testing`](#rollout_traintesting)** — Test doubles: a scripted engine and a readable token format. [`Characters`](#characters), [`plain_channel`](#plain_channel), [`plain_renderer`](#plain_renderer), [`PlainRenderer`](#plainrenderer), [`sample_request`](#sample_request), [`scripted_engine`](#scripted_engine), [`ScriptedEngine`](#scriptedengine)
- **[`rollout_durable`](#rollout_durable)** — A runner whose runs survive their process, on DBOS. [`DurableRunContext`](#durableruncontext), [`DurableRunHandle`](#durablerunhandle), [`DurableRunner`](#durablerunner), [`RunCancelled`](#runcancelled), [`RunStore`](#runstore)
- **[`rollout_vllm`](#rollout_vllm)** — An engine on vLLM. [`VllmEngine`](#vllmengine)
- **[`rollout_lora`](#rollout_lora)** — A trainer for 4-bit checkpoints with LoRA. [`FullTrainer`](#fulltrainer), [`LoraSettings`](#lorasettings), [`LoraTrainer`](#loratrainer)
- **[`rollout_qwen`](#rollout_qwen)** — Renderers for the Qwen model families. [`qwen3`](#qwen3), [`qwen35`](#qwen35), [`tokenizer_of`](#rollout_qwentokenizer_of)
- **[`rollout_gemma`](#rollout_gemma)** — Renderers for the Gemma model families. [`arguments`](#arguments), [`gemma4`](#gemma4), [`GemmaFunctionCalls`](#gemmafunctioncalls), [`tokenizer_of`](#rollout_gemmatokenizer_of)
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
- `def sandboxes(self) -> Mapping[str, SandboxSpec]`
- `def tool_specifications(self) -> list[ToolSpecification]`
- `async def main(self, run: RunContext) -> None`

### `bind`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def bind(reference: ProgramReference, channel: str, *, tools: Mapping[str, ToolBinding] | None = None, pools: Mapping[str, PoolBinding] | None = None) -> RunBinding
```

A binding that serves every model slot of a program from one recorded channel, each of its imports from the
tool set registered under the import's own name (or as `tools` says), and each kind of sandbox it declares from
the pool registered under the kind's name (or as `pools` says).

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

### `Capacity`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Capacity(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `size` | `int` | required | How many sandboxes the pool can hold at once. |
| `leased` | `int` | required | How many it holds, or is making, now. |

**Methods**

- `@property def free(self) -> int`
- `def to_json(self) -> dict[str, JsonValue]`

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

### `Environment` {#rolloutharnessenvironment}

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
- `async def put_file(self, path: Path, media_type: str) -> BlobReference` — Store the file at `path`, or find it already stored, without copying its bytes where the store is on the
  same filesystem: the blob is then a hard link to the file, and both are made read-only, since they are one file.
  Elsewhere the file is copied. The file is read in pieces, never whole.
- `async def link(self, reference: BlobReference, target: Path) -> bool` — Put the blob at `target`: a hard link to it where `target` is on the store's filesystem, else a copy.
  Returns False, putting nothing, if the store does not have it.

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

### `Lease`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Lease(ContractModel)
```

A sandbox held under a key: what a pool hands out, and what its `Leases` table keeps.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | What it was acquired under: a run's lease and the sandbox's name (`RUN/GROUP/EPISODE/ATTEMPT/world` for an episode, whose claim it ends with). |
| `kind` | `str` | required |  |
| `pool` | `str` | required | The pool that holds it. |
| `handle` | `str` | required | The pool's name for the sandbox. |
| `addresses` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` |  |
| `environment` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` |  |
| `at` | `float` | `0.0` | When it was made, in seconds since the epoch. |
| `ends` | `float \| None` | `None` | When its wall time is over (`SandboxLimits.seconds`), in seconds since the epoch. |
| `lost` | `bool` | `False` | Its sandbox is gone (it ended with the pool's process, say): the key cannot have it back. |

### `LeaseRefused`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class LeaseRefused(Exception)
```

The key may hold no lease now: the claim it was acquired under no longer holds.

### `Leases`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Leases(Protocol)
```

Where a pool keeps its leases, by key: ordinary state, changed in place.

**Methods**

- `async def get(self, key: str) -> Lease | None`
- `async def put(self, lease: Lease) -> None`
- `async def delete(self, key: str) -> None`
- `async def all(self) -> list[Lease]`

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

### `MemoryLeases`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class MemoryLeases
```

`Leases` in this process: they end with it.

**Methods**

- `def __init__(self) -> None`
- `async def get(self, key: str) -> Lease | None`
- `async def put(self, lease: Lease) -> None`
- `async def delete(self, key: str) -> None`
- `async def all(self) -> list[Lease]`

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

### `Mount`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Mount(ContractModel)
```

Files the sandbox sees, read-only: an environment version's files, its virtual environment.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | required | Where they are, as the pool finds them: a path on its machine, or a name it resolves. |
| `target` | `str` | required | Where they appear inside the sandbox. |

### `Network`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Network(ContractModel)
```

What the sandbox may reach besides its connection to the runner (its lease's addresses): nothing, unless hosts
are allowed.

| Field | Type | Default | Description |
|---|---|---|---|
| `allow` | `FrozenSequence[str]` | `()` | Hosts it may reach (`pypi.org`, say). |

### `NoCapacity`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class NoCapacity(Exception)
```

The pool holds as many sandboxes as it can; an acquire may succeed once one is released.

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

### `Pool`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Pool(Protocol)
```

Hands out sandboxes of one kind under leases: `SandboxPool`, or one served over HTTP (`RemotePool`). Runners
acquire and release; programs perform operations through `run.sandbox(name)`.

**Methods**

- `@property def deduplicates(self) -> bool` — Whether it performs each operation's `effect_id` at most once.
- `def operations(self) -> Sequence[ToolSpecification]`
- `async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease` — The lease of `key`: the one there is, or a new sandbox. Raises `NoCapacity` when the pool is full,
  `LeaseRefused` for a key that may hold no lease now (its claim lapsed), and `SandboxLost` for a key whose
  sandbox is gone.
- `async def release(self, key: str) -> None` — End the lease of `key` and delete its sandbox; nothing if there is no such lease.
- `async def capacity(self) -> Capacity`
- `async def call(self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult` — Perform an operation on the sandbox leased under `key`.

### `PoolBinding`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class PoolBinding(ContractModel)
```

How a kind of sandbox is served. Exactly one kind is set.

| Field | Type | Default | Description |
|---|---|---|---|
| `local` | `str \| None` | `None` | The name of a pool registered with the runner, in process. |
| `url` | `str \| None` | `None` | A pool served over HTTP (`rollout.harness.remote.serve_pool`), wherever its sandboxes live. |

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

### `Process`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Process(ContractModel)
```

A process the sandbox runs from its start (an environment's worker, a coding agent): the pool launches it.

| Field | Type | Default | Description |
|---|---|---|---|
| `command` | `FrozenSequence[str]` | required |  |
| `environment` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Its own variables; the lease's (its slots' model addresses) are added to them. |
| `directory` | `str \| None` | `None` | Its working directory, inside the sandbox. |

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
- `def sandboxes(self) -> Mapping[str, SandboxSpec]` — The sandboxes the program runs against, by name: the runner acquires each from the pool the binding names
  for its kind before `main`, and releases it after; the program reaches it as `run.sandbox(name)`.
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

### `Provider`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Provider(Protocol)
```

Makes, deletes and operates sandboxes of one kind: Paper servers, containers, a provider's API. A
`SandboxPool` leases them out.

**Methods**

- `@property def kind(self) -> str`
- `@property def size(self) -> int` — How many sandboxes it can hold at once.
- `def operations(self) -> Sequence[ToolSpecification]` — What can be done to one of its sandboxes (none: a harness inside reaches it by its addresses).
- `async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach` — Make the sandbox `handle`, giving what runs inside it `environment`, or say how to reach it if it is
  there already.
- `async def delete(self, handle: str) -> None` — Delete the sandbox, or do nothing if it is gone.
- `async def held(self) -> Sequence[str]` — The handles of the sandboxes it has now.
- `async def call(self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult` — Perform an operation on a sandbox. Errors the operation reports are results with `is_error`; exceptions
  are platform failures.

### `Reach`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Reach(ContractModel)
```

How a sandbox is reached, as its provider says once it has made it.

| Field | Type | Default | Description |
|---|---|---|---|
| `addresses` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Where its services listen, by name (`{"game": "127.0.0.1:25565"}`, say). |
| `environment` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Environment variables for what runs inside it: those it was given, and any of the provider's own. |

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
| `pools` | `Mapping[str, PoolBinding]` | `Field(default_factory=dict[str, PoolBinding])` | Sandbox kind → the pool its sandboxes are acquired from. |
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
- `def sandbox(self, name: str) -> Sandbox` — A sandbox the program declared (`Program.sandboxes()`), acquired for this run: its addresses, its
  environment, its operations. `KeyError` for a name the program did not declare.
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
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None, lease: str | None = None) -> RunHandle` — Start a run. Its sandboxes are acquired under `lease` and each one's name (by default the `run_id`): an
  episode's claim, say, so that they end with it.
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

### `Sandbox`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Sandbox
```

A sandbox a run holds, as `run.sandbox(name)` gives it: how to reach it, and its operations, each a
`tool.call` effect.

**Methods**

- `def __init__(self, name: str, lease: Lease, pool: Pool, effects: Effects) -> None`
- `@property def addresses(self) -> Mapping[str, str]`
- `@property def environment(self) -> Mapping[str, str]` — For what runs inside it: those the runner gave it (its slots' model addresses) and the pool's own.
- `def specifications(self) -> list[ToolSpecification]` — Its operations.
- `async def call(self, operation: str, arguments: Mapping[str, JsonValue] | None = None) -> ToolResult` — Perform an operation as a `tool.call` effect. A side-effecting one is guarded unless the pool
  deduplicates: after a crash it completes as `OUTCOME_UNKNOWN` rather than happen twice.

### `SandboxLimits`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxLimits(ContractModel)
```

What the sandbox may use. Unset: as much as the pool gives.

| Field | Type | Default | Description |
|---|---|---|---|
| `cpus` | `float \| None` | `None` |  |
| `memory_mib` | `int \| None` | `None` |  |
| `processes` | `int \| None` | `None` |  |
| `seconds` | `float \| None` | `None` | Wall time from its start: past it, its lease ends and the pool deletes it. |

### `SandboxLost`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxLost(Exception)
```

The key's sandbox is gone, and a new one would not be the one its run was using.

### `SandboxPool`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxPool
```

A `Pool` over a `Provider`: at most `provider.size` leases at once, kept in `leases` under the pool's `name`
(by default the provider's kind; several pools sharing a table need names of their own).

**Methods**

- `def __init__(self, provider: Provider, *, name: str | None = None, leases: Leases | None = None, admits: Callable[[str], Awaitable[bool]] | None = None) -> None` — `admits` says whether a key may hold a lease now (beside a ledger: whether its claim holds,
  `rollout_train.sandboxes.admits`); without it, every key may.
- `@property def deduplicates(self) -> bool`
- `def operations(self) -> Sequence[ToolSpecification]`
- `async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease` — The lease of `key`, or a new sandbox. Raises `LeaseRefused` for a key `admits` refuses (releasing a lease
  it has), `SandboxLost` for a key whose sandbox is gone, and `NoCapacity` when the pool is full.
- `async def release(self, key: str) -> None`
- `async def capacity(self) -> Capacity`
- `async def call(self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult`
- `async def held(self) -> list[Lease]` — This pool's leases, those whose sandboxes are lost included.
- `async def sweep(self, ended: Callable[[Lease], bool] = lambda lease: False) -> list[str]` — Release the leases `ended` says have ended, and those past their wall time; mark lost those whose sandbox
  is gone (the pool's process was started again, say), which their keys cannot have back; and delete the
  sandboxes no lease names. Returns the keys released or marked lost.
- `async def close(self, *, release: bool = True) -> None` — Release every lease the pool holds (deleting its sandboxes), and close the provider. With `release` False,
  the leases stay, for runs a durable runner resumes to acquire again: a sandbox that outlived the provider is
  theirs again, and one that did not is lost.

### `SandboxSpec`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxSpec(ContractModel)
```

A sandbox a program needs: its kind, what it is made from, and what it may do. A world for an episode needs
only a kind and parameters; a worker for an environment names a process, mounts, scratch, network and limits.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | required | The kind of sandbox (`minecraft`, say): the binding names the pool that serves each kind. |
| `parameters` | `Mapping[str, JsonValue]` | `Field(default_factory=dict[str, JsonValue])` | What the pool makes it from: a task and its seeds, an image. |
| `slots` | `FrozenSequence[str]` | `()` | Model slots a harness inside the sandbox samples. Each one's address is put in the sandbox's environment: `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL`, suffixed with the slot's name in capitals (`_AGENT_1`), and unsuffixed too when there is one slot. A key names the run's session of its slot, and stops working once the recorder forgets the run: an episode runner has it forget the run as the episode ends. |
| `process` | `Process \| None` | `None` |  |
| `mounts` | `FrozenSequence[Mount]` | `()` |  |
| `scratch` | `Scratch \| None` | `None` | Without it, the sandbox writes nowhere. |
| `network` | `Network` | `Network()` |  |
| `limits` | `SandboxLimits` | `SandboxLimits()` |  |

### `Scratch`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Scratch(ContractModel)
```

A directory the sandbox may write, empty when it starts.

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `'/scratch'` |  |
| `mib` | `int` | `1024` | The most it may hold, in MiB. |

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
| `sandboxes` | `ClassVar[dict[str, SandboxSpec]]` | `{}` | Sandboxes the task runs against, by name: acquired before `setup`, reached as `run.sandbox(name)`. |
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

Where a harness that brings its own loop reaches a model slot: an endpoint that speaks OpenAI's and
Anthropic's APIs. Whatever answers there is the slot's model; the harness only sets its base URL and key.

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
| `SANDBOXES_ACQUIRED` | `'sandboxes.acquired'` |  |

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

## `rollout.environment`

What a run trains on and an eval measures: rows, starts, eval data, what results say.

### `binding_for`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def binding_for(environment: Environment, channel: str, tools: Mapping[str, ToolBinding] | None = None, pools: Mapping[str, PoolBinding] | None = None) -> RunBinding
```

How an environment's runs are served: every model slot of its program from `channel`, each of its imports from
the tool set of its own name, or where `tools` says, and each kind of sandbox from the pool of its own name, or
where `pools` says. (A program says which slots, imports and sandboxes it has once it is given a row: the
environment's first.)

### `Description`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Description
```

What an environment's results say, for whatever shows or compares them.

| Field | Type | Default | Description |
|---|---|---|---|
| `rewards` | `tuple[float \| None, float \| None]` | `(0.0, 1.0)` | The range an episode's reward falls in (None: no bound on that side). |
| `solved` | `bool` | `True` | Whether its results say `solved`. |
| `saturated` | `bool` | `False` | Whether its results say `saturated` (nothing was left to earn). |
| `duration` | `str \| None` | `None` | What a result's `duration` counts (`turns`, `minutes of game time`); None: its results say no duration. |
| `observations` | `str \| None` | `None` | How its observations are shown (`minecraft`, say); None: as text. |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`

### `drawn`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def drawn(environment: Environment, *, seeds: Sequence[int], rows: Sequence[str] | None = None) -> list[Start]
```

A start of each row (of `rows`, by key; else every row) for each seed, drawn with `random.Random(seed)`: eval
data derived from rows and seeds. Raises `ValueError` for a row the environment lacks, or no seeds.

### `Environment` {#rolloutenvironmentenvironment}

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Environment(Protocol)
```

**Methods**

- `@property def program(self) -> ProgramReference` — What a run executes; its parameters are a start.
- `@property def version(self) -> str` — Changed whenever its rows, starts, eval data or scoring change: runs and suites record it.
- `@property def description(self) -> Description`
- `def rows(self) -> Sequence[Row]` — Every situation, easiest first.
- `def start(self, row: Row, rng: random.Random) -> JsonValue` — The parameters of one start of `row` (a seed drawn with `rng`, say): what every run of a group is given.
- `def evals(self) -> Mapping[str, Sequence[Start]]` — Its eval data: named lists of starts, never drawn for training. Each is frozen as a suite of its name the
  first time it is played (`rollout_train.evals`). `drawn` derives one from rows and seeds.

### `held_out`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def held_out(environment: Environment) -> frozenset[str]
```

The keys of every one of the environment's eval starts (`start_key`): what training never draws.

### `Row`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Row
```

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | Its name among the environment's rows. |
| `title` | `str` | required | What it is, for people. |
| `parameters` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |
| `counts_for` | `tuple[str, ...]` | `()` | The keys of other rows that a group of this one is evidence about too: the same situation with more help, say. What it teaches about this row it teaches about them. |

### `Start`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Start
```

One start of a row, drawn with a seed of its own: what an eval plays.

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | required | The row's key. |
| `title` | `str` | required |  |
| `seed` | `int` | required |  |
| `parameters` | `JsonValue` | required | What every episode of it is given: the row's start, drawn with `seed`. |

### `start_key`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def start_key(parameters: JsonValue) -> str
```

A start's parameters as canonical JSON: two starts are the same start when their keys are equal.

### `train_start`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def train_start(environment: Environment, row: Row, rng: random.Random, held: Collection[str]) -> JsonValue
```

A start of `row` for training, drawn with `rng`, and drawn again while it is an eval start (its key in `held`,
`held_out`). Raises `ValueError` when `DRAWS` draws in a row are.

## `rollout.curriculum`

Which row to train on next, and gates on evals.

### `Curriculum`

*class* · `libraries/rollout/src/rollout/curriculum.py`

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
| `evaluations` | `dict[str, tuple[str \| None, list[GroupResult]]]` | `field(default_factory=dict[str, tuple[str \| None, list[GroupResult]]])` | The newest eval of each suite, by the suite's name: the checkpoint that played it (None: the base model), and how it did at each start. |

**Methods**

- `def unlocked(self) -> list[Row]`
- `def sample(self, pending: Collection[str] = (), rng: random.Random | None = None) -> Row` — The next row. `pending` names rows whose latest group has not been recorded yet: choosing one again would
  be choosing on what was known before it, so the others come first.
- `def recorded(self, line: GroupResult) -> None` — Take a group's result into account: the row of its title, or failing that of its key (a key that is a
  place in an environment changes when rows are added). A curriculum is the fold of a run's results.
- `def weight(self, row: Row) -> float`
- `def update(self, row: Row, rewards: Sequence[float], solved: Sequence[bool]) -> None` — Record a group of episodes of `row`: each one's reward and whether it solved the row.
- `def failed(self, row: Row) -> None` — Record a group of `row` none of whose episodes completed. After `FAILED_GROUPS` of them in a row it is no
  longer untried, and it taught nothing.
- `def evaluated(self, suite: str, checkpoint: str | None, results: Sequence[GroupResult]) -> None` — Take an eval into account: `suite` played by `checkpoint` (None: the base model), one result per start of
  the suite. The generic curriculum keeps the newest of each suite and decides nothing from it; one that gates on
  evals overrides this.
- `def record(self, row: Row) -> Record`

### `curriculum_of`

*function* · `libraries/rollout/src/rollout/curriculum.py`

```python
def curriculum_of(environment: Environment) -> Curriculum
```

A curriculum that has recorded nothing: the environment's own (`environment.curriculum()`, if it has one), else
the generic one over its rows.

### `GroupResult`

*class* · `libraries/rollout/src/rollout/curriculum.py`

```python
class GroupResult(Protocol)
```

A group's result, as a curriculum reads it (`rollout_train.record.Result` is one).

**Methods**

- `@property def task(self) -> str` — The row's key.
- `@property def title(self) -> str`
- `@property def rewards(self) -> Sequence[float]` — Of the episodes that completed, as is `solved`.
- `@property def solved(self) -> Sequence[bool]`

### `solved_share`

*function* · `libraries/rollout/src/rollout/curriculum.py`

```python
def solved_share(results: Sequence[GroupResult]) -> float
```

The share of the episodes of `results` that solved their row (0 when there are none).

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
- `def sandbox(self, name: str) -> Sandbox`
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
- `async def acquire_sandboxes(self, specs: Mapping[str, SandboxSpec], pools: Mapping[str, Pool], lease: str) -> None` — Acquire each declared sandbox from the pool of its kind, under `lease` and its name, giving a harness
  inside it its slots' model addresses; then record them. Idempotent: a replay gets the same sandboxes.
- `async def release_sandboxes(self) -> None` — Release every sandbox the run acquired, or began to: when the program has ended.
- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue], guard: bool = False) -> T`
- `def record_event(self, event_type: RunEventType, payload: JsonValue) -> RunEvent`

### `LocalRunHandle`

*class* · `libraries/rollout/src/rollout/local/runner.py`

```python
class LocalRunHandle
```

A run started by a `LocalRunner`. Its context is available for inspection in tests and tools.

**Methods**

- `def __init__(self, run_id: str, specification: RunSpecification, conversation: ConversationKey | None, lease: str | None = None) -> None`
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

- `def __init__(self, *, providers: Mapping[str, EndpointFactory] | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environments: EnvironmentService | None = None, blobs: Blobs | None = None, recorder: RecordedEndpoints | None = None, hooks: Sequence[RunHooks] = (), pools: Mapping[str, Pool] | None = None) -> None` — `recorder` serves recorded model bindings (trainable channels); direct bindings use `providers`. `pools`
  are the sandbox pools a binding names as `local`. `hooks` watch every run: each event recorded and each model
  sample.
- `async def launch(self) -> None` — Nothing to start: runs execute on the caller's event loop.
- `async def close(self) -> None` — Nothing to release: nothing outlives the process.
- `def deploy(self, deployment: Deployment) -> None` — Register or replace a deployment; a conversation's next run uses the current version.
- `def run(self, run_id: str) -> LocalRunHandle`
- `def conversation_of(self, run_id: str) -> ConversationKey | None` — The conversation a run serves, if any.
- `def conversation_runs(self, deployment: str, key: str) -> list[LocalRunHandle]` — The conversation's runs, oldest first.
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None, lease: str | None = None) -> LocalRunHandle`
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

### `events_of` {#rollouttestingevents_of}

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def events_of(run: LocalRunContext, event_type: RunEventType) -> list[RunEvent]
```

The run's events of one type, in order.

### `FakeSandbox`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class FakeSandbox
```

One of `FakeSandboxes`: what it was made from and given, the process it runs, and every operation asked of
it.

| Field | Type | Default | Description |
|---|---|---|---|
| `handle` | `str` | required |  |
| `spec` | `SandboxSpec` | required |  |
| `environment` | `Mapping[str, str]` | required |  |
| `process` | `Process \| None` | `None` | What its spec says to run, with the lease's environment added to the process's own. |
| `written` | `int` | `0` | Bytes written to its scratch directory. |
| `calls` | `list[tuple[str, Mapping[str, JsonValue]]]` | `field(default_factory=list[tuple[str, Mapping[str, JsonValue]]])` |  |

### `FakeSandboxes`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class FakeSandboxes
```

A sandbox `Provider` whose sandboxes are only records, honouring what their specs allow: at most `size` of
them, of `kind`. A spec's process is "launched" with the lease's environment added to its own, and the runner
reaches it at the address `process`. Its operations are `describe` (the sandbox's handle, parameters, environment
and process), `write` (`path`, `bytes`: only within its scratch directory and its size, never on a mount), `fetch`
(`host`: only a host its network allows), and any in `operations`; it keeps every sandbox it made and deleted.
Lease them out with `SandboxPool(FakeSandboxes())`.

**Methods**

- `def __init__(self, kind: str = 'fake', size: int = 4, operations: Mapping[str, Operation] | None = None) -> None`
- `def operations(self) -> Sequence[ToolSpecification]`
- `async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach`
- `async def delete(self, handle: str) -> None`
- `async def held(self) -> Sequence[str]`
- `async def call(self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult`

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

Episodes a run asks for in the ledger, claimed and played by runners, and read back.

### `Episode`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Episode
```

| Field | Type | Default | Description |
|---|---|---|---|
| `run` | `str` | required | The training run (or other caller) that asked for it. |
| `group` | `int` | required |  |
| `number` | `int` | required | Its number in its group, from 1. |
| `run_id` | `str` | required | The program's run that played it. |
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

### `EpisodeRunner`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class EpisodeRunner
```

Claims the episodes runs ask for in `ledger` and plays them on `runner`, at most `places` at once: those of the
runs it can serve (whose models its recorder's channels serve, whose imports are among `imports` and whose local
pools are among `pools`), and of `runs` only, if given. An episode is claimed only while the pools of its
sandboxes have room for them, and its run's sandboxes are leased under its claim. `guard` is called before
claiming and raises to wait (a machine short of memory, say). With `presence`, it beats every `beating` seconds,
with what `about` says of its machine besides its places, how many it plays and how full its pools are, and a
claim holds only while its runner beats. Over a runner whose runs survive it (`resumes`), closing leaves its runs
to be resumed, and starting again adopts them (`prepare`).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `ledger` | `Ledger` | required |  |
| `runner` | `Runner` | required |  |
| `recorder` | `Recorded` | required |  |
| `blobs` | `Blobs` | required |  |
| `places` | `int` | required |  |
| `imports` | `Collection[str]` | `()` |  |
| `runs` | `Collection[str] \| None` | `None` |  |
| `hooks` | `Sequence[Hooks]` | `()` |  |
| `guard` | `Callable[[], None] \| None` | `None` |  |
| `presence` | `Presence \| None` | `None` |  |
| `about` | `Callable[[], Mapping[str, JsonValue]] \| None` | `None` | What the runner says of its machine in each beat (called in a thread: it may measure). |
| `pools` | `Mapping[str, Pool]` | `field(default_factory=dict[str, Pool])` | The sandbox pools its runner has, by the name a binding gives as `local`. |
| `every` | `float` | `0.5` | Seconds between looks for work while nothing ends. |
| `beating` | `float` | `15.0` | Seconds between beats. |

**Methods**

- `@property def resumes(self) -> bool` — Whether its runner's runs survive it (`Runner.resumes`, which a durable runner has).
- `async def prepare(self) -> None` — Take its fence, beat, and adopt what it finds of its runs: over a runner whose runs survive it, call this
  before launching the runner, so that the runs it recovers find their claims holding. `serve` calls it if it
  has not been.
- `async def serve(self) -> None` — Claim and play episodes until cancelled; what is playing then is cut short and noted (over a runner whose
  runs survive it, left to be resumed).
- `async def beat(self) -> None` — Beat now, beside the beats every `beating` seconds: after what it says of itself changed (a channel serves
  a new checkpoint, say), so that whoever reads the beats does not wait for the next.
- `async def open(self) -> list[Open]` — The episodes nobody plays now, of the runs this runner serves, oldest group first.

### `episodes_of`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
async def episodes_of(ledger: Ledger, blobs: Blobs, run: str, group: int, count: int, *, every: float = 0.5) -> list[Episode]
```

A group's episodes once all `count` have ended, waiting for them.

### `events_of` {#rollout_trainrolloutsevents_of}

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def events_of(record: Record, blobs: Blobs) -> list[RunEvent]
```

The events of the run a record names, as its runner recorded them.

### `Hooks`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class Hooks(Protocol)
```

Watch a runner (episodes as they start and end) or a run (its results and steps).

**Methods**

- `def on_note(self, event: Mapping[str, JsonValue]) -> None` — `event["kind"]` is `started` or `ended` (an episode, by a runner: an adopted one is started again), or the
  run's `result`, `step` or `published`.

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

### `Plan`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class Plan
```

How a run's episodes are played: its program (each group's start is its row) and its binding.

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `ProgramReference` | required |  |
| `binding` | `RunBinding` | required |  |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Plan'`

### `plan`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
async def plan(ledger: Ledger, run: str, played: Plan, fence: Fence) -> None
```

Say how a run's episodes are played, from now on (each start of a run may say it anew).

### `playing`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
async def playing(runner: EpisodeRunner) -> AsyncGenerator[None]
```

`async with playing(runner):` — the runner serves while the block runs.

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

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class Recorded(Protocol)
```

What a runner needs of the recorder: each run's segments, and the channels it serves. A recorder that routes
channels to engines elsewhere (`rollout_train.recorder.Recorder` with `routes`) also says whether it reaches a
run's (`reaches`), and names them within the run in its binding (`for_run`).

| Field | Type | Default | Description |
|---|---|---|---|
| `channels` | `Mapping[str, Any]` | required |  |

**Methods**

- `def sessions(self, run_id: str) -> dict[str, list[Segment]]`
- `def forget(self, run_id: str) -> None`

### `stored`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def stored(episode: Episode, events: Sequence[RunEvent], blobs: Blobs) -> Record
```

Keep an episode's trajectories and its run's events in `blobs`; returns the record that names them.

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

## `rollout_train.sandboxes`

Sandboxes' leases beside the ledger, each ending with its episode's claim.

### `admits`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
def admits(ledger: Ledger, presence: Presence | None) -> Callable[[str], Awaitable[bool]]
```

For a pool beside a ledger (`SandboxPool(admits=...)`): whether a key may hold a lease now. A key whose run the
ledger knows may while its claim holds; any other key may.

### `ended`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def ended(leases: list[Lease], ledger: Ledger, presence: Presence | None) -> Callable[[Lease], bool]
```

Which of `leases` have ended, as the ledger says now: those whose run it knows and whose claim does not hold.

### `FileLeases`

*class* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
class FileLeases
```

`Leases` in `sandboxes.json` in a ledger's directory, under the lock the ledger's files are written under.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def get(self, key: str) -> Lease | None`
- `async def put(self, lease: Lease) -> None`
- `async def delete(self, key: str) -> None`
- `async def all(self) -> list[Lease]`

### `keep`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def keep(pool: SandboxPool, ledger: Ledger, presence: Presence | None, *, beat_as: str | None = None, every: float = 15.0) -> None
```

Sweep the pool every `every` seconds, until cancelled, releasing a lease once its claim was found lapsed at two
looks running; with `beat_as`, beat under that name too.

### `leases_of`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
def leases_of(ledger: Ledger) -> Leases | None
```

The leases beside a ledger: a file beside a ledger of files, a table in a database ledger's database.

### `sweep`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def sweep(pool: SandboxPool, ledger: Ledger, presence: Presence | None) -> list[str]
```

Release the pool's leases whose claims have ended (and delete what no lease names); the keys released.

## `rollout_train`

The training loop, the group algorithm, evals, and what they ask of a trainer.

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

### `Changeable`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Changeable(Protocol)
```

A trainer that takes some of its settings between steps (its learning rate, say): those that change neither
what its weights are nor what it can take (`Budget`).

**Methods**

- `@property def changeable(self) -> Mapping[str, JsonValue]` — The settings it takes between steps, by its name for each, with their values now.
- `def change(self, settings: Mapping[str, JsonValue]) -> None` — Take these settings (some of `changeable`) from its next step on. Raises `ValueError` for one it does not
  take, or a value it cannot.

### `Checkpoint`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Checkpoint
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |
| `weights` | `Manifest \| None` | required | None once it was released (`Checkpoints.thin`). |
| `parents` | `tuple[str, ...]` | `()` | What it was made from, by id: first the checkpoint it was trained from, then any others it learned from (the teachers of a distillation, say). None: from the base model. |
| `depth` | `int` | `1` | Steps from the base model along its first parents: its first parent's depth and one. |
| `base` | `str \| None` | `None` | What its weights build on: the model its line began from, by name (`Qwen/Qwen3.5-9B`, say); or, for an adapter trained over a full checkpoint, that checkpoint, by id. |
| `kind` | `str` | `'lora'` | What its weights are: `lora` (an adapter over its base) or `full` (all of a model's weights). |
| `run` | `str \| None` | `None` | The run that made it, by id. |
| `step` | `int \| None` | `None` | The run's step that made it (none for a checkpoint made outside a run's steps, such as by imitation). |
| `state` | `Manifest \| None` | `None` | What a trainer goes on from: the optimizer's state, say. |
| `batch` | `BlobReference \| None` | `None` | What it was trained on: the segments, each as its source (`RUN/GROUP/EPISODE/SLOT/INDEX`) and its advantage. |
| `metrics` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` |  |
| `dataset` | `str \| None` | `None` | The dataset it was trained on, by id (`rollout_train.datasets`), if a supervised step on one made it: its parents after the first are then the checkpoints that sampled the dataset's examples. |
| `made` | `float` | `0.0` | When, in seconds since the epoch. |
| `released` | `float \| None` | `None` | When its files were deleted (`Checkpoints.thin`), if they were: its weights and its trainer state are then None. Its record stays: where it came from, what it was trained on, and its metrics. |

**Methods**

- `@property def parent(self) -> str | None` — The checkpoint it was trained from, if any.

### `Checkpoints`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Checkpoints
```

Every checkpoint, in a ledger, and their files in a blob store.

**Methods**

- `def __init__(self, ledger: Ledger, blobs: Blobs) -> None`
- `async def all(self) -> list[Checkpoint]` — Every checkpoint, oldest first.
- `async def checkpoint(self, id: str) -> Checkpoint` — The checkpoint an id says.
- `async def under(self, checkpoint: Checkpoint) -> Checkpoint | None` — The full checkpoint whose weights `checkpoint` is served over: itself, if it is full; the full checkpoint it
  builds on, if it is an adapter over one; None for an adapter over a model. Raises `ValueError` if that
  checkpoint was released.
- `async def head(self, run: str) -> Checkpoint | None` — The newest checkpoint a run made, if it made one.
- `async def add(self, fence: Fence, id: str, *, weights: Path, run: str | None, base: str | None = None, kind: str = 'lora', step: int | None = None, state: Path | None = None, parents: Sequence[str] = (), batch: BlobReference | None = None, metrics: Mapping[str, float] | None = None, dataset: str | None = None) -> Checkpoint` — Keep a checkpoint's files and append the checkpoint that names them, under `fence` (the run's that makes it).
  Its base is what its weights build on (`_base`): for an adapter over a full checkpoint, that checkpoint (by
  id); for a merge (full weights from an adapter), the `base` it names; else its first parent's base, or `base`
  for a checkpoint made from the base model. The append is what
  makes the checkpoint exist: a writer that dies before it has made nothing, and one that repeats it (the same id,
  decided before) gets the checkpoint that is there.
- `async def thin(self, fence: Fence, run: str, retention: 'Retention', keep: Collection[str] = ()) -> list[str]` — Delete the files (weights and trainer state) of the checkpoints `run` made that `retention` does not keep,
  nor `keep` (what is served, what is bookmarked, what another run starts from), and return their ids. A
  release is appended to the ledger before its blobs are deleted, and a blob is deleted only if no checkpoint
  still names it, so this may be repeated after a crash at any point.
- `async def files(self, manifest: Manifest, directory: Path) -> Path` — A manifest's files under `directory`, read from the blob store if they are not there. The directory
  appears whole or not at all, so whatever looks for a file in it never finds half a checkpoint. A file this
  store lacks is read from the store of any run that has it (a checkpoint made by a run that kept its blobs
  elsewhere, or a merge of one), as each run's start says where its store is.

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
- `@property def changeable(self) -> Mapping[str, JsonValue]` — The settings the trainer it wraps takes between steps (`rollout_train.trainer.Changeable`), if any.
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step`

### `Dataset`

*class* · `libraries/rollout-train/src/rollout_train/datasets.py`

```python
class Dataset
```

A dataset's record (`DATASETS`): how its examples were chosen, what came of it, and where its manifest is.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Sixteen random letters, like a checkpoint's. |
| `rule` | `str` | required | The episode rule, by name (`RULES`). |
| `runs` | `list[str]` | required | The runs its episodes are from, by id. |
| `turns` | `list[str]` | required | The turn filters, by name. |
| `cut` | `list[str]` | required | The kinds of guidance cut from its examples' prompts when they are made. |
| `manifest` | `BlobReference` | required | One JSON line per example, compressed (`manifest_of` reads it). |
| `blobs` | `Mapping[str, JsonValue]` | required | Where the manifest is kept, as any process opens it (`rollout_train.stores`). |
| `per_task` | `int \| None` | `None` | For `capped-per-task`: the most episodes of each task. |
| `counts` | `Mapping[str, int]` | `field(default_factory=dict[str, int])` | `episodes` the rule picked and the `groups` they are of; `turns_seen`, every turn of those episodes; and of its examples, `tasks`, `turns`, `sampled_tokens` and `context_tokens`. |
| `left_out` | `Mapping[str, int]` | `field(default_factory=dict[str, int])` | Turns of its episodes that are no examples, by why. |
| `checkpoints` | `list[str]` | `field(default_factory=list[str])` | The checkpoints that sampled its examples, by id, by depth (examples sampled by the base model name none). |
| `made` | `float` | `0.0` | When, in seconds since the epoch. |
| `by` | `str` | `''` | Who made it: `user@host`. |

### `dataset_of`

*function* · `libraries/rollout-train/src/rollout_train/datasets.py`

```python
async def dataset_of(ledger: Ledger, id: str) -> Dataset
```

The dataset an id says.

### `evaluate`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def evaluate(environment: Environment, checkpoints: Checkpoints, *, run: str, suite: Suite, subject: str | None, base: str | None, channel: str, directory: Path, publish: Publisher | None, episodes: int = 1, binding: RunBinding | None = None, started: Mapping[str, JsonValue] | None = None, asked_by: str = 'by hand', reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None, hooks: Sequence[Hooks] = (), served_by: str | None = None) -> dict[str, Any]
```

Play `suite` with `subject` (a checkpoint's id; None: the base model, named `base`) served on `channel`,
`episodes` episodes of each start, as the run `run`; returns how it went (`played`, `solved`, `reward`, and each
start's `results`, a `Result` each). `publish` serves a checkpoint on the channel (a full one in place of the
engines' weights; for an adapter over a full checkpoint, the engines must already hold that checkpoint's weights, as
`rollout eval` sees to); None: the channel serves `subject` already (a training run's newest checkpoint). `reshard`
gives its files in the engines' layout (`rollout_train.resharding`); `directory` holds its files on this machine.
What the eval's channel serves is written down (`rollout_train.serving`), so that runners anywhere play it on
replicas that serve `subject` and no other checkpoint; `served_by` names the channel whose replicas serve it
(`RUN/NAME`: the training run's, for an eval its schedule asks for), where it is not the eval's own.

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

### `Files`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Files
```

A checkpoint's files on this machine: what a step starts from.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` | `Path` | required |  |
| `state` | `Path \| None` | `None` | What the trainer left for itself beside the weights (an optimizer's state, say), if it left any. |

### `Follower`

*class* · `libraries/rollout-train/src/rollout_train/following.py`

```python
class Follower
```

Keeps `channels` serving what `run` says each should (by the channel's name within the run), checking every
`every` seconds, with each checkpoint's files under `directory` while they are served. With `presence`, it beats as
`name` every `beating` seconds, and at once when a channel serves something new: what `about` says of the machine,
and what each channel serves.

**Methods**

- `def __init__(self, name: str, checkpoints: Checkpoints, run: str, channels: Mapping[str, Channel], directory: Path, *, presence: Presence | None = None, about: Callable[[], Mapping[str, JsonValue]] | None = None, every: float = 2.0, beating: float = 15.0) -> None`
- `async def serve(self) -> None` — Follow until cancelled.
- `async def follow(self) -> bool` — Give every channel what the run says it should serve, if it serves something older; whether any changed.
- `async def beat(self) -> None`
- `def served(self) -> list[JsonValue]` — What each channel serves, how fast since the last call, and each of its engines: its address (where it is a
  server elsewhere) and what it serves.

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

### `make_dataset`

*function* · `libraries/rollout-train/src/rollout_train/datasets.py`

```python
async def make_dataset(ledger: Ledger, rule: str, runs: Sequence[str], *, into: Blobs, at: Mapping[str, JsonValue], turns: Sequence[str] = (ALL,), cut: Sequence[str] = ('way',), per_task: int | None = None, by: str | None = None) -> Dataset
```

Make a dataset of the episodes `runs` (by id) completed: those `rule` picks, and of them the turns every filter
of `turns` keeps. Its manifest is kept in `into`, which is at `at` (as `rollout_train.stores.opened` reads it).
Episodes are read one at a time, from where each run's blobs are. Raises `ValueError` for a rule or filter that
does not exist, or a dataset of no examples.

### `make_suite`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def make_suite(ledger: Ledger, name: str, environment_name: str, environment: Environment, *, rows: Sequence[str] | None = None, seeds: Sequence[int] = (), starts: Sequence[Start] | None = None) -> Suite
```

Make a suite of `environment`: `starts`, the environment's eval data of that name (held out of training); or,
by hand, a start of each row (of `rows`, by key; else every row) for each seed (`drawn`). Raises `ValueError` for a
name that is no name or is taken (a suite is never changed), a row the environment lacks, or no starts.

### `Manifest`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Manifest
```

The files of a checkpoint, by their paths within it, each kept as a blob.

| Field | Type | Default | Description |
|---|---|---|---|
| `files` | `Mapping[str, BlobReference]` | required |  |
| `layout` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | How the weights are divided among the files, where they are divided: whoever wrote them says, so that a reader with the same division reads its own files and no others. |

### `record_serving`

*function* · `libraries/rollout-train/src/rollout_train/serving.py`

```python
async def record_serving(ledger: Ledger, run: str, serving: Serving, fence: Fence) -> bool
```

Write down, under the run's fence, that its channel serves `serving` from now on; False if it was written
before (a loop started again serves what it served).

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
| `unlocked` | `int` | `0` | Rows of the environment unlocked after this group. |

**Methods**

- `def to_json(self) -> dict[str, Any]` — The record as the `results` table keeps it: without what the group's own record and key say (`JOINED`).
- `@classmethod def from_json(cls, data: Mapping[str, Any], number: int, group: Mapping[str, Any]) -> 'Result'` — A result as it is kept, with what its group's record (`group`, under `number`) says.

### `results`

*function* · `libraries/rollout-train/src/rollout_train/record.py`

```python
async def results(ledger: Ledger, run: str = 'train') -> list[Result]
```

How a run's groups went, by their numbers.

### `Retention`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Retention
```

Which of a run's checkpoints keep their files, their weights and their trainer state (what can be served, and
what a step can go on from): the newest `recent`, and every `every`-th by depth, so that saves thin out with
age.

| Field | Type | Default | Description |
|---|---|---|---|
| `recent` | `int` | `2` |  |
| `every` | `int` | `20` |  |

**Methods**

- `def kept(self, depths: list[int]) -> set[int]`

### `Schedule`

*class* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
class Schedule
```

Evals a training run makes of its own checkpoints: `suite` played by the checkpoint of every `every`th step,
`episodes` episodes of each start, between that step and the next. `environment` is the suite's environment, and
`binding` how its episodes are played (by default every slot from the trained channel). `run` gives the eval's run
for a step: the same each time it is asked for that step, and one the run's episode runners play.

| Field | Type | Default | Description |
|---|---|---|---|
| `suite` | `Suite` | required |  |
| `environment` | `Environment` | required |  |
| `run` | `Callable[[int], Awaitable[str]]` | required |  |
| `every` | `int` | `1` |  |
| `episodes` | `int` | `1` |  |
| `binding` | `RunBinding \| None` | `None` |  |

**Methods**

- `def due(self, checkpoint: Checkpoint, run: str) -> bool` — Whether `checkpoint` is evaluated: a checkpoint `run` made at a step the schedule names.

### `Serving`

*class* · `libraries/rollout-train/src/rollout_train/serving.py`

```python
class Serving
```

That a run's channel serves a checkpoint from now on (or, with no checkpoint, the base model).

| Field | Type | Default | Description |
|---|---|---|---|
| `channel` | `str` | required | The channel's name within the run. |
| `checkpoint` | `str \| None` | `None` | By id; None: the model the channel's engines are started with. |
| `depth` | `int` | `0` | The checkpoint's depth: the version its samples are stamped with. |
| `kind` | `str` | `'lora'` | `lora`, an adapter; `full`, weights loaded in place of the engines' own. |
| `files` | `Manifest \| None` | `None` | What the engines load: the checkpoint's weights, or the files they were resharded into. |
| `layout` | `str \| None` | `None` | The layout `files` are in (`rollout_train.resharding`), if they were resharded. |
| `over` | `str \| None` | `None` | For an adapter over a full checkpoint, that checkpoint, by id: the engines hold its weights first. |
| `model` | `str \| None` | `None` | The model the channel's line began from, by name. |
| `sequence` | `int \| None` | `None` | The longest turn the trainer can train on (`Limits.sequence`), for every runner that samples the channel. |
| `served_by` | `str \| None` | `None` | The channel whose engines serve this (`RUN/NAME`), where it is another run's: an eval played on the channel of the training run whose checkpoint it plays. None: the channel's own. |
| `max_lag` | `int \| None` | `None` | How many checkpoints behind this a sample may be, where the run says (0 for an eval, which plays one checkpoint); None: as the runner's channel says. |
| `at` | `float` | `field(default_factory=lambda: round(time.time(), 1))` |  |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Serving'`

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

### `Suite`

*class* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
class Suite
```

A named list of starts of an environment's rows, frozen: what every subject plays, start for start.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `environment` | `str` | required | The environment, as `module:name`. |
| `starts` | `list[Start]` | required | Its starts, in order: each row it names, once with each seed. |
| `made` | `float` | `0.0` |  |
| `rows` | `list[str] \| None` | `None` | The rows it names, by key. |
| `seeds` | `list[int] \| None` | `None` |  |
| `version` | `str \| None` | `None` | The environment's version when the suite was made. |
| `held_out` | `bool` | `False` | Whether it is the environment's eval data, whose starts training never draws. |

### `suite_for`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def suite_for(ledger: Ledger, name: str, environment_name: str, environment: Environment) -> Suite
```

The suite `name`: the one in the ledger, or else the environment's eval data of that name, frozen now (on first
use). Raises `KeyError` when neither has it, `ValueError` when the ledger's is another environment's.

### `suite_of`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def suite_of(ledger: Ledger, name: str) -> Suite | None
```

A suite, if there is one by that name.

### `train`

*function* · `libraries/rollout-train/src/rollout_train/loop.py`

```python
async def train(environment: Environment, trainer: Trainer, checkpoints: Checkpoints, *, start: str | None = None, base: str | None = None, channel: str, directory: Path, publish: Publisher, run: str = 'train', algorithm: Algorithm | None = None, groups: int = 100, groups_per_step: int = 4, episodes_at_once: int = 6, seed: int = 0, binding: RunBinding | None = None, curriculum: Curriculum | None = None, retention: Retention | None = None, started: Mapping[str, JsonValue] | None = None, hooks: Sequence[Hooks] = (), kept: Callable[[], Awaitable[Collection[str]]] | None = None, made: Callable[[Checkpoint], Awaitable[object]] | None = None, reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None, evals: Schedule | None = None, desired: Callable[[], Awaitable[Mapping[str, JsonValue]]] | None = None, scheduled: Callable[[str, int, int], Awaitable[Schedule | None]] | None = None) -> None
```

Train from `start` (a checkpoint's id; else the base model, named `base`) on `environment` until `groups` more
groups have been played (those a stopped loop left unplayed among them) and every group played has been trained on,
serving each checkpoint made on `channel`; a run started again goes on from the newest checkpoint it made. A step is
taken over the groups queued once at least `groups_per_step` have something to train on (and, at the end, over what
is left). `directory` is where checkpoints' files are kept on this machine while they are in use: the one being
served and the one before it (a turn in progress finishes under the weights it began with); every checkpoint's files
are in the blob store; `publish` serves a checkpoint on `channel`. `algorithm` is `Grpo()` unless given.
`episodes_at_once` is how many episodes the run keeps work waiting for, whatever groups they are of (runners play
them, as many at once as each has places). `binding` says how the program's model slots and imports are served (by
default: every slot from `channel`, each import from the tool set of its own name). `curriculum` is one that has
recorded nothing (by default the environment's own, else the generic one: `curriculum_of`): the run's results are
folded into it. Each group's start is drawn with `train_start`, never one of the environment's eval starts.
`retention` says which of the checkpoints the run made keep their files (weights and trainer state) once a newer one
is served (`Retention()` unless given); besides those, what is served, what any run starts from, and whatever `kept`
says (the bookmarked checkpoints, say) keep theirs. `started` is what the run's `starts` record says beside what the
loop knows (where it starts from, this host, the time): where the run's directory is, where the monitor on its
machine serves (`address`), and what profile started it, say. `hooks` are told of each result and step; `made` is
called with each checkpoint made, once it is served (to move a bookmark, say). `reshard` gives the files the engines
load for a checkpoint (in their layout: `rollout_train.resharding`), told the run's fence to note it under; without
it, they load the trainer's. `evals` says which checkpoints the run evaluates as it makes them, between their step
and the next. `desired` reads what is wanted of the run's changeable settings (`rollout_train.settings`:
`groups_per_step`, `evals.…`, `trainer.…`), each time a step is about to be decided; `scheduled` makes the schedule
of evals they name (a suite by name, every, episodes; None for a suite the run cannot play), without which only
`evals`' suite can be played.

### `Trained`

*class* · `libraries/rollout-train/src/rollout_train/record.py`

```python
class Trained
```

What was done with a group: the step that covered it, and the checkpoint that step made or why it failed.

| Field | Type | Default | Description |
|---|---|---|---|
| `step` | `int` | required |  |
| `checkpoint` | `str \| None` | `None` | By id, once made. |
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
| `weights` | `str` | required | What its steps make: `lora` (an adapter over the weights the engines hold) or `full` (all the weights). |

**Methods**

- `async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step` — Train on the batch, starting from `parent` (None: from the base model). The new weights are left in
  `into/weights`, and what a later step starts from in `into/state`. Raises `StepFailed` if the step
  produced no weights.

### `wanted`

*function* · `libraries/rollout-train/src/rollout_train/serving.py`

```python
async def wanted(ledger: Ledger, run: str, channel: str) -> Serving | None
```

What a run's channel should serve now: its record of the greatest depth (the newest among equals); None if the
run has said nothing of it.

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
| `source` | `str` | `''` | Where the segment is from, for the record of what a step trained on: `RUN/GROUP/EPISODE/SLOT/INDEX`. |

## `rollout_train.inference`

Channels: trainable models being served, and what they ask of an engine.

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
| `adapter` | `str \| None` | `None` | The adapter sampling now (None: the weights the engines hold, the model's own or a full checkpoint's). |
| `serving` | `str \| None` | `None` | What is served, by name: the adapter, or the full checkpoint the engines hold (None: the model's own). |
| `version` | `int` | `0` | How many times weights have been published; recorded with every sampled token. |
| `held` | `str \| None` | `None` | The full checkpoint the engines hold, by name (None: the model's own). |

**Methods**

- `@property def context_limit(self) -> int` — The longest turn the channel takes, and what it tells programs.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', version: int | None = None, request: str | None = None) -> Generation` — Sample from one of the engines: the same one for a session every time, where its prompts' shared
  beginnings are cached. `version` and `request` (the version the caller stamps the tokens with, and a name for
  the request) are for samplers elsewhere: this process's own callers read what it publishes.
- `async def weights(self, session: str) -> tuple[str | None, int]` — The adapter a session's next turn samples from (None: the weights the engines hold), and the version its
  tokens are stamped with: the same for every session, as this process publishes to every engine at once.
- `@property def loaded(self) -> list[str]` — The adapters loaded on the engines, oldest first: the one served, and the one before.
- `async def publish(self, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int` — Serve `adapter` from now on: a LoRA directory every engine can read at `path`, or with `full`, a full
  checkpoint's weights there, which the engines load in place of what they hold. Returns the version it is
  served as: `version` if one is given (the checkpoint's depth, which means the same in every process), or
  one more than the last. An adapter before stays loaded, so that a turn in progress finishes under the
  weights it began with; the one before that is dropped. Full weights replace the engines' at once, and the
  adapters trained on the weights before go with them. Publishing what is being served changes nothing.
- `async def pause(self) -> None` — Hold new requests back, and wait for those in flight to finish.
- `def resume(self) -> None`
- `async def sleep(self) -> None`
- `async def wake(self) -> None`
- `def take(self) -> dict[str, float]` — What passed through since the last call: requests, tokens in and out, and throughput.
  `tokens_per_second` is everything generated over the time the channel was generating.
- `def close(self) -> None`

### `Connection`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class Connection
```

How servers are reached: a bearer token read from an environment variable (`token_env`) or a file
(`token_file`), never written down; TLS verified against a CA bundle (`ca`), and a client certificate and its key
(`certificate`, `key`) for servers that ask for one. Nothing: plain HTTP, no token.

| Field | Type | Default | Description |
|---|---|---|---|
| `token_env` | `str \| None` | `None` |  |
| `token_file` | `str \| None` | `None` |  |
| `ca` | `str \| None` | `None` |  |
| `certificate` | `str \| None` | `None` |  |
| `key` | `str \| None` | `None` |  |

**Methods**

- `def token(self) -> str | None`
- `def client(self, timeout: float = 600.0) -> httpx.AsyncClient` — A client that reaches servers so.

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
- `async def load_weights(self, path: str) -> None` — Serve the full weights in `path` (a checkpoint's files) in place of the model's own, from now on.
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
| `model` | `str \| None` | `None` | The model that sampled it, where a server elsewhere says (`rollout_train.inference.remote`): the checkpoint, by the name it is served as. The recorder checks that it is the checkpoint it stamps the tokens with. |

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

### `RemoteChannel`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class RemoteChannel
```

One run's channel, sampled on servers elsewhere: what the recorder samples from (`Sampler`).

Each turn asks for the checkpoint the run says the channel should serve (`wanted`), by its id as the model's name;
or, where its server does not have it yet, the newest one before it that the server has, no more than `max_lag`
checkpoints behind (`Serving.max_lag`, where the run says, as an eval does: 0). A server that answers that it does
not have the model is asked for the one before, and for the newest again at the next look. A turn waits while no
server has a checkpoint close enough, for `patience` seconds at most (then `NoReplica`). Every token is stamped with
the depth of the checkpoint its answer names; an answer that names another model is refused, and the turn sampled
again.

The channel's servers are one URL (a router, a proxy, or a single server), or a list: a session's turns then go to
one of those that answer and have a checkpoint close enough, worked out from the session's id alone, so that nothing
is kept per session. What the run says and what each server has are asked again every `every` seconds.

**Methods**

- `def __init__(self, name: str, renderer: 'Renderer', limits: Limits, *, model: str, servers: Sequence[str], wanted: Callable[[], Awaitable[Sequence[Serving]]], max_lag: int = MAX_LAG, connection: Connection | None = None, every: float | None = None, patience: float = 300.0) -> None`
- `@property def limits(self) -> Limits` — The profile's limits; the longest turn the trainer can train on, as the run says, unless they say one.
- `@property def context_limit(self) -> int`
- `@property def bound(self) -> int` — How many checkpoints behind what the channel should serve a sample may be.
- `def name_of(self, said: Serving) -> str` — The model a server serves a checkpoint as: its id; the base model's name for none.
- `def choices(self) -> list[Serving]` — The checkpoints a turn may sample from now, newest first: what the channel should serve, and those before it
  no more than `bound` behind (a full checkpoint, which a server cannot serve under its own name, is none).
- `def offered(self, address: str) -> Serving | None` — What a server would sample a turn from now: the newest of the `choices` it has.
- `def server_of(self, session: str) -> str | None` — The server a session's turns go to now (none while none has a checkpoint close enough): of those that do,
  the one a hash of the session and its address ranks first.
- `async def refresh(self, *, now: bool = False) -> None` — Ask again what the channel should serve and what each server has (unless that was asked within `every`
  seconds and `now` is not said). A server that does not answer is given no turn.
- `async def reaches(self) -> bool` — Whether a server would take a turn now.
- `async def weights(self, session: str) -> tuple[str | None, int]` — The checkpoint a session's next turn samples from (None: the base model) and the version its tokens are
  stamped with: its depth.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', version: int | None = None, request: str | None = None) -> Generation` — Sample on the session's server, from the checkpoint `adapter` names. `Unserved` if the server does not have
  it any more (`NotLoaded`), answers for another, or does not answer (`Unreachable`).
- `def servers(self) -> list[dict[str, JsonValue]]` — The servers it samples on, as last asked: each one's address, the checkpoint it would sample from now and its
  depth, and how far that is behind what the channel should serve.
- `def take(self) -> dict[str, float]` — What passed through since the last call, as `Channel.take` counts it.
- `def close(self) -> None`

### `RemoteEngine`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class RemoteEngine
```

An engine served elsewhere: a vLLM OpenAI-compatible server at `address` (or a router or a proxy in front of
several), serving `model` under its own name. `Engine` over its API: a request names the adapter it samples from as
its model (the base model's name for none), and an answer that names another is refused (`Unserved`), as is a model
the server does not have (`NotLoaded`). Adapters are loaded and removed by name; the server must allow it
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) and read the path given on its own machine. Full weights cannot be served
under a name of their own by the server: `load_weights` refuses.

| Field | Type | Default | Description |
|---|---|---|---|
| `processes` | `Sequence[int]` | `()` |  |

**Methods**

- `def __init__(self, model: str = '', *, address: str, connection: Connection | None = None, client: httpx.AsyncClient | None = None, max_model_len: int | None = None) -> None`
- `async def models(self, within: float = 2.0) -> dict[str, Any]` — The models the server has (`/v1/models`), by name; `Unreachable` if it does not answer `within` seconds.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', request: str | None = None) -> Generation` — Complete the prompt's tokens with the model `adapter` names (the base model for none): the tokens sampled,
  the logprob of each, how it ended, and the model the server says sampled it.
- `async def load_adapter(self, name: str, path: str) -> None`
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None`
- `async def sleep(self) -> None` — Free the server's accelerator (it must allow it: `VLLM_SERVER_DEV_MODE=1`).
- `async def wake(self) -> None`
- `def close(self) -> None`

### `Route`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class Route
```

How a channel whose engines serve elsewhere is sampled: its model family's renderer, its limits, its base
model's name, its servers (a router, a proxy, a server, or a list), how far behind a sample may be (`max_lag`), and
how the servers are reached (`connection`).

| Field | Type | Default | Description |
|---|---|---|---|
| `renderer` | `'Renderer'` | required |  |
| `model` | `str` | required |  |
| `servers` | `tuple[str, ...]` | required |  |
| `limits` | `Limits` | `field(default_factory=Limits)` |  |
| `max_lag` | `int` | `MAX_LAG` |  |
| `connection` | `Connection` | `field(default_factory=Connection)` |  |

### `Routes`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class Routes
```

The routed channels of every run a runner plays (`RemoteChannel`), each made when first asked for, choosing from
what that run says its channel serves (in `ledger`): implements the recorder's `Routes`.

**Methods**

- `def __init__(self, routes: Mapping[str, Route], ledger: Ledger, *, every: float | None = None, patience: float = 300.0) -> None`
- `def routed(self, channel: str) -> bool`
- `def channel(self, run: str, channel: str) -> RemoteChannel`
- `async def reaches(self, run: str, channel: str) -> bool`
- `def channels(self) -> dict[str, RemoteChannel]` — Every routed channel made so far, by its name within its run (`RUN/NAME`).
- `def close(self) -> None`

### `Sampler`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Sampler(Protocol)
```

What the recorder samples from: a `Channel`, whose engines this process publishes to, or a channel sampled on
servers elsewhere (`rollout_train.inference.remote.RemoteChannel`).

**Methods**

- `@property def name(self) -> str`
- `@property def renderer(self) -> 'Renderer'`
- `@property def limits(self) -> Limits`
- `@property def context_limit(self) -> int`
- `async def weights(self, session: str) -> tuple[str | None, int]` — The adapter (the checkpoint) a session's next turn samples from (None: the weights the engines hold), and
  the version its tokens are stamped with.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', version: int | None = None, request: str | None = None) -> Generation` — Sample from the checkpoint `adapter` names, stamped `version`; `Unserved` where it is not served (or the
  server is gone). `request` names the request, for whatever logs it.

### `Unserved`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Unserved(Exception)
```

The checkpoint a turn began with is not served where it is asked for (not loaded yet, dropped, or another
answered), or the server cannot be reached: the turn is sampled again, from what is served then.

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

- `def __init__(self, recorder: Recorder, channel: Sampler, sampling: SamplingParameters) -> None`
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
| `channels` | `Mapping[str, Channel]` | required | The channels whose engines this process publishes to, by name. |
| `base_url` | `str \| None` | `None` | Where `rollout_train.recorder.compat` serves this recorder, as harnesses reach it (None: it is not served). |
| `routes` | `Routes \| None` | `None` | The channels whose engines serve elsewhere: a binding names one within its run (`RUN/NAME`, `for_run`). |

**Methods**

- `def endpoint(self, binding: RecordedModel) -> 'RecordedEndpoint'`
- `def for_run(self, run: str, binding: RunBinding) -> RunBinding` — A run's binding, with each recorded model on a routed channel named within the run (`RUN/NAME`), so that
  its samples are of the checkpoints that run says its channel serves; the same binding if it names none.
- `async def reaches(self, run: str, binding: RunBinding) -> bool` — Whether every recorded model of a run's binding can be sampled here: from a channel whose engines this
  process publishes to, or from a routed channel whose servers have a checkpoint close enough to what the run
  says it should serve.
- `def export(self, session_id: str) -> list[Segment]` — The session's segments, oldest first (see the module's description).
- `def sessions(self, run_id: str) -> dict[str, list[Segment]]` — What each model slot of a run exports, by slot.
- `def forget(self, run_id: str) -> None`
- `async def publish(self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int` — Serve new weights on a channel (an adapter, or with `full` a full checkpoint's weights); returns the
  version they are served as (a checkpoint's depth).
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
| `channel` | `str` | `''` | The channel that sampled them. The spans' `version`s are the depths of the checkpoints it served. |

**Methods**

- `@property def sampled(self) -> int`

### `Span`

*class* · `libraries/rollout-train/src/rollout_train/recorder/recorder.py`

```python
class Span
```

Tokens `start` to `end` (exclusive) of a segment were sampled by the policy, at weights `version` (the depth
of the checkpoint served then).

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
| `reshard` | `str \| None` | `None` | `module:name` of the layout the engines load a checkpoint's files in (`rollout_train.resharding`); none: the trainer's files as they are, with no reshard. |
| `max_lag` | `int` | `MAX_LAG` | For a channel whose engines serve elsewhere (`engine` is `RemoteEngine`, each entry of `engines` a server's `address`): how many checkpoints behind what the channel should serve a sample may be, where its server does not have the newest yet. |
| `via` | `str \| None` | `None` | For a channel whose engines serve elsewhere: the URL its runners send every request to (a router or a proxy in front of its servers); none: its servers' addresses. Its engine hosts load checkpoints at the addresses. |
| `connection` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | How servers elsewhere are reached (`Connection`): `token_env` or `token_file`, `ca`, `certificate`, `key`. |

**Methods**

- `@property def routed(self) -> bool` — Whether its engines serve elsewhere (said by name: an engine's module is not imported to load a profile).
- `def route(self, renderer: Any, sequence: int | None = None) -> Route` — How a runner samples it on its servers elsewhere; `sequence`, the trainer's longest turn, where this process
  trains it.

### `EvalsSpec`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class EvalsSpec
```

Evals a training run makes of its checkpoints as it makes them (`rollout_train.evals.Schedule`).

| Field | Type | Default | Description |
|---|---|---|---|
| `suite` | `str` | required | The suite each plays, by name: the environment's eval data of that name (frozen on first use), or a suite made by hand (`rollout suite make`). |
| `every` | `int` | `1` | The checkpoint of every `every`th step is evaluated. |
| `episodes` | `int` | `1` | Episodes of each of the suite's starts. |

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

An open profile: its `run`, the checkpoint it trains from (`origin`), the `checkpoints`, a `trainer` to step,
`publish` to serve a checkpoint, and a `runner` that plays the episodes its run asks for
(`rollout_train.rollouts.scheduler.EpisodeRunner`).

**Methods**

- `def __init__(self, profile: Profile) -> None`
- `@classmethod async def start(cls, profile: Profile, stack: contextlib.AsyncExitStack, *, training: bool = True, plays: Collection[str] | None = None) -> 'Platform'` — Start everything (the trainer only with `training`), registering with `stack` how each thing is stopped
  (the engines last). With `plays`, a runner and nothing else (`rollout runner`): no run is registered in the
  directory and no trainer is made; the runner plays those runs (by id), or with none named every run whose
  channels it reaches, and the channels whose engines serve in this process follow what the one run named says
  they should serve (`rollout_train.following`).
- `@property def layout(self) -> str | None` — The layout the trained channel's engines load checkpoints in, if they are resharded.
- `async def reshard(self, checkpoint: Checkpoint, fence: Fence) -> Manifest` — A checkpoint's files in the trained channel's layout: resharded as a Ray task when the profile names a Ray
  cluster, else here.
- `async def eval_run(self, step: int) -> str` — The run of the eval of the checkpoint this run made at `step` (`rollout_train.evals.Schedule`), by id:
  registered the first time as `NAME-eval-STEP` and kept in `directory/evals`; the runner plays its episodes.
- `async def bookmarked(self) -> set[str]` — The checkpoints bookmarks name (which keep their files).
- `async def made(self, checkpoint: Checkpoint) -> None` — Carry the profile's bookmark, if it names one, to a checkpoint the run made.
- `async def publish(self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int` — Serve new weights on a channel from now on (with `full`, a full checkpoint's); returns the number its
  samples are stamped with (a checkpoint's depth). The runner beats at once, saying what the channel serves. A
  channel whose engines serve elsewhere is served there: they follow what the training loop wrote down that it
  serves (`rollout_train.serving`), and this returns the version given.

### `Profile`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class Profile
```

| Field | Type | Default | Description |
|---|---|---|---|
| `directory` | `Path` | required | The run's state: checkpoints' files while in use, the monitor's feed, and (unless the profile names other places) its ledger and blobs. |
| `channels` | `Mapping[str, ChannelSpec]` | required |  |
| `trainer` | `TrainerSpec \| None` | `None` |  |
| `runner` | `str` | `'local'` | `local` runs episodes in this process; `durable` records them so that they survive it. |
| `serve` | `str \| None` | `None` | `host:port` to serve the model endpoint for harnesses on. |
| `address` | `str \| None` | `None` | The URL others reach `serve` at (by default `http://` and `serve`). |
| `tools` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | Each tool set by name: a URL, or `module:name` of what makes it, called with `directory`. |
| `pools` | `Mapping[str, str \| Mapping[str, Any]]` | `field(default_factory=dict[str, str \| Mapping[str, Any]])` | Each sandbox pool by the kind of sandbox it serves: a URL, or `module:name` of the provider that makes them, called with `directory`; or a table whose `kind` is that and whose other entries are passed to it too. |
| `ledger` | `Mapping[str, Any]` | `field(default_factory=dict[str, Any])` | Where the run's tables and the checkpoints are kept (`rollout_train.ledger.opened`): `{"directory": …}`, in files; `{"kind": "module:name", …}`, what that makes from the other entries, such as a database (`rollout_train.database:DatabaseLedger` with a `url`). By default files under `directory/ledger`. Runs that share a ledger see each other's checkpoints. |
| `blobs` | `Mapping[str, Any]` | `field(default_factory=dict[str, Any])` | Where episodes (and what programs store) are kept: `kind` is `module:name` of what makes the store, called with the other entries. Without one, files under `directory/blobs`. |
| `runs_gib` | `float` | `0.0` | System memory that must be available to admit runs. |
| `training_gib` | `float` | `0.0` | And to start a step of a colocated trainer. |
| `episodes_at_once` | `int` | `6` | The most episodes a run plays at once (whatever groups they are of): what the machine's engines and its memory for the programs' worlds can take. |
| `feed_runs` | `int \| None` | `None` | Episodes kept in the monitor's feed, where it should not keep `RunFeed`'s own number (the oldest are deleted). |
| `ray` | `str \| None` | `None` | The Ray cluster to connect to (`auto`, or `ray://host:port`): reshards then run as Ray tasks on it. |
| `name` | `str \| None` | `None` | What a run first started in `directory` is called (by default the directory's name). It is named again with `rollout rename`; its id, in the directory's `run.json`, never changes. |
| `evals` | `EvalsSpec \| None` | `None` | The evals a training run makes of its checkpoints as it makes them. |

**Methods**

- `@classmethod def load(cls, path: Path, *, directory: Path | None = None, settings: Mapping[str, Any] | None = None) -> 'Profile'` — The profile a TOML file describes; `directory` replaces the file's (one profile, many runs), and
  `settings` replace or add its keys, by dotted name (`trainer.learning_rate`, `episodes_at_once`). A key the
  file has and a profile does not is an error: a misspelt guard would otherwise be no guard.
- `async def open(self, *, training: bool = True, plays: Collection[str] | None = None) -> AsyncGenerator['Platform']` — Start what the profile describes, and stop it on the way out (also if starting fails half way). Without
  `training` (an eval), no trainer is made: the trained channel's engines still load what the trainer's `start`
  is served over. With `plays`, a runner and nothing else (`rollout runner`): see `Platform.start`.
- `async def engines(self) -> AsyncGenerator[dict[str, Channel]]` — The channels whose engines are servers elsewhere, as clients of the servers at their addresses, and nothing
  else: what an engine host (`rollout engines`) loads checkpoints into. Closed on the way out.

### `TrainerSpec`

*class* · `libraries/rollout-train/src/rollout_train/profile.py`

```python
class TrainerSpec
```

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | required | `module:name` of what makes the trainer, called with the channel's model and `settings`. |
| `channel` | `str` | required | The channel that serves the policy it trains. |
| `start` | `str \| None` | `None` | The checkpoint a new run trains from (`rollout_train.registry.resolved`: a bookmark, `RUN:STEP`, `RUN`, or a checkpoint's id or the start of one); by default the base model. A run started again goes on from the newest checkpoint it made. |
| `bookmark` | `str \| None` | `None` | A bookmark the run carries: moved to each checkpoint it makes. |
| `colocated` | `bool` | `False` | Whether it shares the channels' accelerator: their engines then sleep while it steps. |
| `settings` | `Mapping[str, Any]` | `field(default_factory=dict[str, Any])` |  |

## `rollout_train.monitor`

A live web page over every run of a ledger.

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
- `def notes(self, after: int = 0) -> list[dict[str, Any]]` — The notes from index `after` on: episodes as runners start and end them, published weights, the loop's
  results and steps, the engines' throughput.
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
class RunFeed(RunHooks, Hooks)
```

Writes every run's events and samples under `directory`, one file per run, as they happen; and what a
runner and the loop note, at the level they think at, in one file more.

`keep` bounds the directory: when more runs than that have files, the oldest are deleted. A directory has one
writer at a time: runs that an earlier writer left without an end (its process was stopped) are marked cancelled
when the next one starts, so that a monitor does not show them running for ever.

**Methods**

- `def __init__(self, directory: Path, *, keep: int = 200) -> None`
- `def on_event(self, event: RunEvent) -> None`
- `def on_note(self, event: Mapping[str, JsonValue]) -> None`
- `def on_sample(self, sample: ModelSample) -> None`
- `def close(self) -> None`

### `System`

*class* · `libraries/rollout-train/src/rollout_train/monitor/system.py`

```python
class System
```

**Methods**

- `def __init__(self, directory: Path | None = None, feed: FeedReader | None = None, *, ledger: Ledger | None = None, client: httpx.Client | None = None) -> None` — Over a run's `directory` (its ledger, as `rollout_train.ledger.of_run` finds it: every run that shares
  it), or over a `ledger` alone. `feed` reads the directory's feed (by default its `feed`). `client` asks the
  monitors on other machines for their runs' episodes.
- `@property def ledger(self) -> str` — Where the ledger is: its directory, or its database's URL.
- `async def snapshot(self, relayed: bool = False) -> dict[str, Any]` — Where everything stands now: every run (where it is and whether it is running; its groups that are not
  done with and the ones that are), the checkpoints (each with where it came from and the bookmarks that name it),
  the runners and what they play, what each channel serves and how fast, the machine, and what is kept.
- `async def rename(self, who: str, name: str) -> Entry` — Call the run that `who` is (its id or its name) `name` from now on, in the registry beside the ledger. A
  run from before the registry is registered under its key first. Raises `Taken` for a name it cannot have,
  `KeyError` when there is no such run (or no registry).
- `async def bookmark(self, name: str, checkpoint: str) -> Bookmark` — Make a bookmark name the checkpoint `checkpoint` says (its id, the start of one, `RUN:STEP`, `RUN` or another
  bookmark), or move it there. Raises `Taken` for a name that cannot be one, `KeyError` for a reference that
  says no checkpoint (or no registry).
- `async def unbookmark(self, name: str) -> None` — Take a bookmark away (the checkpoint stays). Raises `KeyError` when there is no such bookmark.
- `async def launches(self) -> dict[str, Any]` — The runs asked for, newest first, and the launchers alive with what each offers (its profiles, with the
  settings a launch may change, its environments, and whether it has room). A launch whose launcher stopped
  beating while it was claimed, running or stopping is shown as `lost`: what became of its run is not known.
- `async def launch(self, body: Mapping[str, Any]) -> Launch` — Ask for a run or an eval (`rollout_train.launches.Asked`'s fields): a launcher alive that offers its profile
  and its environment starts it. An eval names a suite (whose environment it plays; a suite not made yet, the
  environment whose eval data it is) and the checkpoint that plays it. Raises `Taken` for what cannot be asked for
  (a name taken or no name, a setting the profile does not have), `KeyError` for what no launcher offers or a
  checkpoint no reference says.
- `async def stop(self, id: str) -> Launch` — Ask a launch to stop: one not started yet is stopped at once; a run going is stopped by its launcher, at a
  group boundary. Raises `KeyError` when there is no such launch going.
- `async def settings(self, run: str) -> dict[str, Any] | None` — A training run's settings (`rollout_train.settings`): its fixed ones and its changeable ones as its newest
  start says, what is wanted of them now, those its newest step used, and each step that used other settings than
  the one before, with what changed. None where there is no such run.
- `async def want(self, run: str, settings: Mapping[str, Any]) -> Desired` — Want these of a run's changeable settings from its next step on. Raises `Taken` for a setting it does not
  have or cannot change, or a value it cannot take (a suite of another environment than the run's, say);
  `KeyError` where there is no such run, or nowhere to keep what is wanted. A suite the ledger does not have is
  taken: the run resolves it from its environment's eval data (`rollout_train.evals.suite_for`), or evaluates
  nothing.
- `async def checkpoint_evals(self, checkpoint: str) -> dict[str, Any] | None` — Every eval a checkpoint (by its id or the start of it) has had, by hand or by a schedule, newest first
  (`rollout_train.monitor.scores.evals_of`); None where there is no such checkpoint.
- `async def path(self, checkpoint: str) -> dict[str, Any] | None` — A checkpoint's line from the base model, with each point's scores at each suite
  (`rollout_train.monitor.scores.path_of`); None where there is no such checkpoint.
- `async def evals(self) -> dict[str, Any]` — Every suite (its environment and starts, and each subject that played it, with how it did at each start) and
  every eval (its suite, its checkpoint, how far it has got), newest first (`rollout_train.evals`).
- `async def lineage(self, sample: bool = False) -> dict[str, Any]` — The policies as a graph, with what trains, serves and evaluates them (`rollout_train.monitor.lineage`).
  With `sample`, the fixture of the tables proposed for distillation, trainers, workers and evaluations is read
  beside the ledger.
- `async def statistics(self) -> dict[str, Any]` — Every run of the ledger in figures (`rollout_train.monitor.statistics`), with each run's engines'
  throughput from its runners' heartbeats, what the runs are called, and the runners' machines.
- `async def machines(self) -> dict[str, Any]` — Every runner's machine, as its heartbeats say: now, and over its recent beats.
- `async def group(self, run: str, number: int, relayed: bool = False) -> dict[str, Any] | None` — One group: what was decided (the row and its start), its stage, its episodes with what each reported,
  its step and the checkpoint it made, and its outcome.
- `def feeds(self, relayed: bool = False) -> list[dict[str, Any]]` — Every episode in the feeds of the runs' directories on this machine (and, unless `relayed`, those the
  monitors elsewhere serve), summarised, newest first.
- `async def episode(self, run_id: str, after: int = 0, relayed: bool = False) -> dict[str, Any]` — One episode: the run's lines from index `after` on (from the feed, or, once the feed has let it go, its
  replies and tool calls from the events its runner kept), from which its rollouts (one per model slot) are
  drawn; what it reported when it ended; and where it sits: its run, its group and its labels. An episode of a
  run on another machine is asked of the monitor there.

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
- `async def load_weights(self, path: str) -> None`
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

- `def __init__(self, run_id: str, endpoints: Mapping[str, ModelEndpoint], *, started_at: datetime, context_hints: ContextHints | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environment_service: EnvironmentService | None = None, blobs: Blobs | None = None, conversation: ConversationKey | None = None, on_event: Callable[[RunEvent], None] | None = None, mark_attempt: Callable[[str], bool] = lambda effect_id: True, last_seq: Callable[[], int | None] = lambda: None) -> None`
- `def record_event(self, event_type: RunEventType, payload: JsonValue) -> RunEvent` — Events are stored by `seq`, and a replay's are the ones stored. A replay that took another way (a sandbox
  refused, say) would give its terminal event a `seq` already taken; it comes after every stored event.
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

| Field | Type | Default | Description |
|---|---|---|---|
| `resumes` |  | `True` | Its runs survive it: started again over its state, it resumes the runs it had (an episode runner adopts them). |

**Methods**

- `def __init__(self, directory: Path, *, providers: Mapping[str, EndpointFactory] | None = None, tool_sets: Mapping[str, ToolSet] | None = None, environments: EnvironmentService | None = None, blobs: Blobs | None = None, recorder: RecordedEndpoints | None = None, hooks: Sequence[RunHooks] = (), application: str = 'rollout', evict_after: timedelta | None = timedelta(minutes=5), eviction_interval: float = 5.0, database: str | Database | None = None, runner_id: str | None = None, heartbeat_interval: float = 2.0, takeover_after: timedelta = timedelta(seconds=15), pools: Mapping[str, Pool] | None = None) -> None` — `evict_after`: unload runs that have waited this long for a message (None keeps every run resident);
  `eviction_interval`: how often, in seconds, to look for runs to evict or wake
  (docs/implementations/rollout-durable/eviction.md).
  
    `database`: a Postgres URL (or `Database`) shared with other runners; without it, state is SQLite in
  `directory` and this runner is the only one. `runner_id` names this runner among them: a runner restarted
  with its id puts its unfinished runs back on the queue at once, without waiting for a takeover.
  `takeover_after`: how long a runner's heartbeat may stop before another runner recovers its runs.
  `directory` holds local files either way.
  
    `pools`: the sandbox pools a binding names as `local`. A run acquires its sandboxes each time it is executed
  (recovered, or woken) under the same lease, and so gets the same ones back while their leases hold.
- `async def launch(self) -> None` — Start DBOS, which recovers the runs a crash left unfinished, and follow them.
- `async def close(self) -> None`
- `def deploy(self, deployment: Deployment) -> None`
- `def run(self, run_id: str) -> DurableRunHandle`
- `def conversation_of(self, run_id: str) -> ConversationKey | None` — The conversation a run serves, if any.
- `def conversation_runs(self, deployment: str, key: str) -> list[DurableRunHandle]`
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, conversation: ConversationKey | None = None, labels: Mapping[str, str] | None = None, lease: str | None = None) -> DurableRunHandle`
- `async def cancel(self, run_id: str, *, reason: str) -> None` — Ask the run to stop at its next effect, wait or turn boundary; `teardown` runs.
- `async def execute(self, run_id: str, specification_json: dict[str, Any], conversation_json: dict[str, Any] | None, labels: dict[str, str], started_at: str, lease: str) -> dict[str, Any]`
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
- `def last_seq(self, run_id: str) -> int | None` — The `seq` of a run's latest stored event; None when it has none.
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

- `def __init__(self, model: str, *, gpu_memory_utilization: float = 0.72, max_model_len: int = 8192, max_num_seqs: int = 32, max_num_batched_tokens: int = 4096, max_lora_rank: int = 32, max_loras: int = 2, language_model_only: bool = True, speculative: Mapping[str, Any] | None = None, quantization: str | None = None, seed: int = 0) -> None`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None) -> Generation`
- `async def load_adapter(self, name: str, path: str) -> None` — Register a LoRA adapter (a PEFT directory) under `name`; samples name it to use it.
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None` — Serve the full weights in `path` (a checkpoint's files, in the model's own layout) in place of the ones
  held, from now on: they are read into the model as it is, and read again from there on waking. (vLLM warns
  that `ParallelLMHead` failed to load where the output layer is tied to the embeddings: it shares them, and
  serves the new ones.)
- `async def sleep(self) -> None` — Free the GPU: the cache is discarded and the weights dropped (they are read again on waking).
- `async def wake(self) -> None`
- `@property def processes(self) -> list[int]`
- `def close(self) -> None`

## `rollout_lora`

A trainer for 4-bit checkpoints with LoRA.

### `FullTrainer`

*class* · `implementations/rollout-lora/src/rollout_lora/trainer.py`

```python
class FullTrainer(LoraTrainer)
```

Trains every weight of a text model (`rollout_lora.full`), one step at a time in a fresh process: a step
starts from its parent's full weights (the model's own for the first) and the optimizer's state, and leaves the
new ones where it is told. `settings` are `LoraSettings`' fields; the adapter's (`rank`) are not used.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` |  | `'full'` |  |

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
| `passes` | `int` | `1` | Passes a step takes over its segments, each shuffled anew and cut into minibatches of its own: a small batch makes more optimizer updates (a supervised step on a small dataset, say). |
| `warmup_updates` | `int` | `0` | When a step's optimizer starts afresh (no state to go on from), its rate rises linearly over its first this many updates, from `learning_rate / warmup_updates` to `learning_rate`: a fresh Adam's first update moves every weight by about the full rate. A step that goes on from an optimizer's state is not warmed up. |
| `objective` | `str` | `'policy_gradient'` | `policy_gradient`: the clipped policy gradient over the sampled tokens, each weighted by its segment's advantage, with an importance weight for where they were sampled. `likelihood`: raise the log-likelihood of the sampled tokens, each weighted by its segment's advantage (imitation: what was sampled is what to do), with no ratio, weight or stop at `max_kl` (`rollout_lora.objectives`). |
| `ratio` | `str` | `'token'` | `token`: a ratio for each token (PPO). `segment`: one for each segment, the geometric mean of its tokens' (GSPO). |

**Methods**

- `@property def loss(self) -> Objective` — The objective a step takes, by these settings.
- `def rate(self, update: int, *, fresh: bool) -> float` — The learning rate of a step's `update`-th optimizer update (from 0): warmed up if its optimizer is
  `fresh`.
- `@property def alpha(self) -> float`

### `LoraTrainer`

*class* · `implementations/rollout-lora/src/rollout_lora/trainer.py`

```python
class LoraTrainer
```

Trains a LoRA adapter over `model`'s checkpoint, one step at a time, each in a fresh process on the GPU
(`rollout_lora.worker`). It keeps nothing between steps: a step starts from the adapter and the optimizer's
state it is given and leaves the new ones where it is told. `settings` are `LoraSettings`' fields; those in
`CHANGEABLE` it takes between steps (`rollout_train.trainer.Changeable`).

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` |  | `'lora'` |  |

**Methods**

- `def __init__(self, model: str, **settings: Any) -> None`
- `@property def changeable(self) -> Mapping[str, JsonValue]`
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step`

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

### `tokenizer_of` {#rollout_qwentokenizer_of}

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

### `tokenizer_of` {#rollout_gemmatokenizer_of}

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
