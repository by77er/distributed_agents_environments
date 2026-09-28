# Our interface vs agent frameworks and durable-agent frameworks

Status: **Draft** · 2026-09-27 · Question owner: research charter ("compare the interface we are creating with
existing tools")

**The question.** How does our interface compare with the agent frameworks, durable-agent frameworks and sandbox
SDKs in use in 2026, and what should we change or borrow? Our interface covers:

- `Program`, `Task`, `Agent`, `Observation`, `@tool`, environment handles and `Template` recipes, `RunContext`,
  imported tools and model slots;
- the proposed additions: `run.emit`, connectors, run keys, approvals, `run.memory`, `WaitFor` suspension and
  durable agent identities.

**How to read this.**
- Sections 2 and 3 are **facts**, each with a source. Anything we could not confirm from a primary source is
  marked **[unverified]**.
- Sections 4 to 10 are **analysis and recommendations**.
- Versions were checked on 2026-09-27.

---

## 1. Summary of recommendations

1. **Keep the core shape.** Keep the Task/Agent split, the framework-owned loop, replay durability, and a
   ban on I/O except through effects. Every serious durable-agent integration in 2026 converged on the same
   architecture:
   - the loop runs as deterministic workflow code;
   - model calls and tool I/O are journaled steps.

   This holds for the OpenAI Agents SDK on Temporal, DBOS and Restate, Pydantic AI `*Durability`, the Vercel
   `WorkflowAgent`, and the ADK plugins. Ours is stricter (receiver-side deduplication, typed effects) and is the
   only one designed around RL.
2. **Replace the generator-style driver with a deterministic asyncio event loop**, as Temporal does. Third-party
   agent libraries use `asyncio.gather`, `create_task`, locks and queues. Without this change, no foreign framework
   can run in-process under replay.
3. **Define durable agent identities as a resource, not as a long-lived run.** An identity is:
   - an address, a deployment, versioned state, scoped memory, a single-consumer mailbox, grants, owned
     resources, a policy binding and a lifecycle;
   - activated by **bounded runs**, one per conversation episode, so every trajectory is finite and replayable.

   Run keys and triggers should route to identity mailboxes, not to runs.
4. **Upgrade approvals** from `requires_approval=True` to the pattern the industry converged on:
   - a policy that is a boolean or a predicate over the arguments;
   - decisions `Approve(arguments=…)`, `Reject(message)` and `Respond(content)`;
   - sticky "always" decisions;
   - approval rules settable in the `RunBinding`, not only in code.
5. **Add `WaitFor`, a `resume` hook, continue-as-new, and delayed rewards** on completed replies. These make
   conversational identities first-class RL episodes.
6. **Add a messaging API.** It needs addresses, envelopes, `send`, `request` (correlated reply), topics, and
   bounded fan-out (`run.map`). Handoffs belong in Agent composition, not in the platform.
7. **Streaming in two tiers:**
   - the durable log as a resumable stream, with `run.emit` and a cursor;
   - ephemeral token and stdout deltas keyed by `effect_id`.
8. **Memory:** scoped, versioned key–value with compare-and-set, plus pinned context blocks. Semantic search is
   an imported binding, not a core feature.
9. **Environments.** Borrow from the sandbox SDKs:
   - named, idempotent `get_or_create`;
   - background processes;
   - exposed ports;
   - layered templates with a start command and readiness check.

   Ship our Environment API as a *sandbox provider* for the OpenAI Agents SDK, Deep Agents, Mastra and the AI SDK.
10. **Interoperability in three tiers:**
    - unmanaged proxy (exists);
    - foreign harness inside an environment;
    - in-process framework adapters (OpenAI Agents `Model` / Pydantic AI durability backend).

    Export identities over A2A and long-running imported tools over MCP Tasks.

---

## 2. Survey

Each entry covers:
- the core abstractions;
- tools;
- the loop;
- durability and human-in-the-loop (HITL);
- identity and memory;
- multi-agent;
- streaming;
- sandboxes;
- RL access;
- extension points.

### 2.1 OpenAI Agents SDK (Python, `openai-agents` 0.22.3, 2026-09-17)

- **Abstractions.** `Agent`, `Runner` (`run`, `run_sync`, `run_streamed`), `RunResult`, `RunContextWrapper[T]`,
  handoffs, guardrails, tracing, sessions ([docs](https://openai.github.io/openai-agents-python/),
  [running agents](https://openai.github.io/openai-agents-python/running_agents/)). `RunConfig` also carries
  `call_model_input_filter`, `error_handlers` (keys `max_turns`, `model_refusal`, `invalid_final_output`) and
  `sandbox`.
- **Tools.** `@function_tool` has these options: `name_override`, `timeout`, `failure_error_function`,
  `needs_approval`, `defer_loading`, and others.

  | Kind | Examples |
  |---|---|
  | Hosted | `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `HostedMCPTool` |
  | Local runtime | `ComputerTool`, `ShellTool`, `ApplyPatchTool` |
  | MCP | `MCPServerStdio`, `MCPServerStreamableHttp` |

  Sources: [tools](https://openai.github.io/openai-agents-python/tools/),
  [MCP](https://openai.github.io/openai-agents-python/mcp/).
- **Loop.** The SDK calls the model, then:
  - a typed final output ends the run;
  - a handoff switches the agent;
  - tool calls are executed and the loop repeats;
  - `MaxTurnsExceeded` ends the run at the turn limit.
- **HITL.**
  - `needs_approval` is a bool or `async (ctx, params, call_id) -> bool`.
  - Paused runs expose `result.interruptions` (`ToolApprovalItem`).
  - Resume flow: `state = result.to_state()` → persist → `RunState.from_json(agent, s)` →
    `state.approve(item, always_approve=...)` or `state.reject(item, rejection_message=...)` →
    `Runner.run(agent, state)`.
  - Sticky always-approve/always-reject decisions survive serialization. Approvals from nested agents-as-tools
    surface on the outer run.

  Source: [HITL doc](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/human_in_the_loop.md).
  The core SDK has no crash recovery; that is left to integrations.
- **Durable integrations.**
  - **Temporal** (`temporalio.contrib.openai_agents`):
    - `OpenAIAgentsPlugin` makes every model call an Activity, and `activity_as_tool` turns Activities into tools.
    - A plain `@function_tool` runs inside the workflow and must be deterministic.
    - Streaming is experimental; sandbox clients are pre-release.

    Sources: [README](https://github.com/temporalio/sdk-python/blob/main/temporalio/contrib/openai_agents/README.md),
    [Temporal docs](https://docs.temporal.io/develop/python/integrations/openai-agents).
  - **DBOS:** `DBOSRunner.run` inside `@DBOS.workflow()`
    ([docs](https://docs.dbos.dev/integrations/openai-agents)).
  - **Restate:** `DurableRunner`, `@durable_function_tool`
    ([blog](https://restate.dev/blog/durable-orchestration-for-ai-agents-with-restate-and-openai-sdk);
    API details **[unverified]**).
  - **Dapr:** a `DaprSession` backend is listed.
- **Memory.**
  - The `Session` protocol (`get_items`, `add_items`, `pop_item`, `clear_session`) has SQLite, Redis,
    SQLAlchemy, Dapr, OpenAI Conversations and encrypted implementations.
  - Server-side state is the alternative: `conversation_id` / `previous_response_id`.
  - Source: [sessions](https://openai.github.io/openai-agents-python/sessions/).
- **Multi-agent.**
  - Handoffs are shown to the model as `transfer_to_<agent>` tools.
  - `agent.as_tool(...)` wraps an agent as a tool.
  - Source: [handoffs](https://openai.github.io/openai-agents-python/handoffs/).
- **Streaming.** `RawResponsesStreamEvent`, `RunItemStreamEvent` and `AgentUpdatedStreamEvent`.
  `result.cancel(mode="after_turn")` stops a stream
  ([streaming](https://openai.github.io/openai-agents-python/streaming/)).
- **Sandboxes (2026).**
  - `SandboxAgent` has `default_manifest=Manifest(entries=…)` (`File`, `Dir`, `LocalDir`, `GitRepo`, mounts),
    capabilities (Filesystem, Shell, Memory, Skills, Compaction) and `run_as`.
  - It attaches through `RunConfig(sandbox=SandboxRunConfig(client=…))`.
  - Clients: Unix-local, Docker, E2B, Modal, Daytona, Cloudflare, Vercel, Runloop and Blaxel. Session state and
    snapshots allow rehydration.
  - Source: [sandbox clients](https://openai.github.io/openai-agents-python/sandbox/clients/). The introduction
    date is **[unverified]**.
- **RL.**
  - `ModelSettings.top_logprobs` returns logprobs for output text only. Token IDs are not exposed.
  - Full token data would need a custom `Model`.
  - Source: [ModelSettings](https://openai.github.io/openai-agents-python/ref/model_settings/).
- **Extension points.**
  - `Model.get_response(system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, …)`
    and `stream_response`; `ModelProvider.get_model(name)`
    ([interface](https://openai.github.io/openai-agents-python/ref/models/interface/)).
  - Also trace processors, lifecycle hooks and guardrails.

### 2.2 Claude Agent SDK (Python 0.2.160, TypeScript 0.3.283, 2026-09-25)

- **Runtime model.** "When your code calls `query()`, the SDK spawns a separate `claude` CLI process." That
  subprocess owns the shell, the working directory and the JSONL transcripts. One session is one subprocess
  ([hosting](https://code.claude.com/docs/en/agent-sdk/hosting)).
- **API.** Two entry points ([Python reference](https://code.claude.com/docs/en/agent-sdk/python)):
  - `query()` starts one session per call.
  - `ClaudeSDKClient` keeps a session open and adds `interrupt`, `set_permission_mode`, `set_model` and
    `rewind_files`.
- **Tools.**
  - `@tool(name, description, schema)` plus `create_sdk_mcp_server(...)` create in-process MCP tools, named
    `mcp__{server}__{tool}`.
  - Built-in tools include Bash, Read, Edit, Write, Grep and the Agent tool.
  - Tool search defers schema loading.
  - Source: [custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools).
- **Loop.** It runs inside the CLI and yields `SystemMessage(init)`, then `AssistantMessage`s, then a
  `ResultMessage` (subtype, usage, `total_cost_usd`, `session_id`). Limits are `max_turns` and `max_budget_usd`
  ([agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)).
- **HITL.**
  - `can_use_tool(name, input, context)` returns allow (optionally with `updated_input`) or deny (optionally
    `interrupt`).
  - Permission modes include `default`, `acceptEdits`, `plan`, `dontAsk`, `bypassPermissions` and `auto`.
  - For long waits, a `PreToolUse` hook returns `permissionDecision: "defer"`. The query then ends with
    `tool_deferred`, and the caller resumes by `session_id` later
    ([user input](https://code.claude.com/docs/en/agent-sdk/user-input)).
- **Durability.**
  - Sessions are resumable (`resume`, `fork_session`, `resume_session_at`).
  - A `SessionStore` adapter (`append`, `load`) mirrors transcripts so another host can resume
    ([session storage](https://code.claude.com/docs/en/agent-sdk/session-storage)).
  - There is no workflow-engine integration in the docs we read (our inference).
- **Multi-agent.** Subagents are defined in `agents={...}` and invoked through the Agent tool. They have isolated
  contexts and only their final message returns to the parent
  ([subagents](https://code.claude.com/docs/en/agent-sdk/subagents)).
- **Streaming.** `include_partial_messages=True` yields raw API stream events.
- **Sandbox.** `SandboxSettings` is an OS-level sandbox for Bash (bubblewrap on Linux). By default the SDK falls
  back to running unsandboxed if the sandbox is unavailable.
- **RL.** No logprobs and no token IDs; only usage counts.
- **Extension points.** In-process MCP, hooks (`PreToolUse`, `PostToolUse`, `Stop`, `PreCompact`, …),
  `can_use_tool`, plugins, skills, and a custom `Transport`.

### 2.3 Google Agent Development Kit (`google-adk` 2.10.0, 2026-09-25)

- **Abstractions.**
  - `Agent`/`LlmAgent`, `Runner`, `App`, `Session`/`SessionService`, `State`, `MemoryService`,
    `ArtifactService` and `Event`.
  - ADK 2.0 (GA 2026-05-19) added a graph **Workflow Runtime**: agents, tools and functions are nodes, and routing
    uses `Event(route=[…])`.
  - Source: [ADK docs repo](https://github.com/google/adk-docs/tree/main/docs), 2.0 and graphs pages.
- **Loop.** The agent yields `Event`s. The Runner commits `event.actions` (`state_delta`, `artifact_delta`) through
  `session_service.append_event`, and only then does the agent resume. The event log is the session
  ([event loop](https://github.com/google/adk-docs/tree/main/docs/runtime)).
- **Tools.**
  - `FunctionTool(func, require_confirmation=bool|callable)`.
  - `LongRunningFunctionTool` pauses the run until the client sends a `FunctionResponse`.
  - Also `AgentTool`, MCP toolsets, OpenAPI tools, and an experimental `EnvironmentToolset`.
- **HITL.**
  - Tool Confirmation is experimental. The docs say it is not supported with `DatabaseSessionService` or
    `VertexAiSessionService`.
  - Graph HITL uses `yield RequestInput(message, payload, response_schema)`.
- **Durability.**
  - Resumability (`ResumabilityConfig(is_resumable=True)`) rebuilds progress from logged Events, but tools run
    **at least once**.
  - Temporal (`temporalio.contrib.google_adk_agents`, experimental), DBOS (`DBOSPlugin`), Restate and Dapr
    plugins exist. The Restate and Dapr APIs are **[unverified]**.
  - "Ambient agents" are triggered by events, queues or schedules.
- **Identity and memory.**
  - Sessions are keyed by `(app_name, user_id, session_id)`.
  - State key prefixes set scope: none (session), `user:` (per user, across sessions), `app:` (global) and
    `temp:` (one invocation).
  - `MemoryService` (`add_session_to_memory`, `search_memory`) includes the Vertex Memory Bank.
- **Multi-agent.**
  - `sub_agents` with LLM-driven `transfer_to_agent`.
  - ADK 2.0 collaboration modes: `chat`, `task` and `single_turn`.
  - **A2A:** `to_a2a(agent)` exposes an agent, and `RemoteA2aAgent(agent_card=…)` consumes one.
- **Streaming.** `StreamingMode.NONE|SSE|BIDI`, and `run_live` for Gemini Live (audio and video).
- **RL.** `LlmResponse.logprobs_result` and `avg_logprobs` are filled when the backend returns them. No token IDs.
- **Extension points.** `BaseLlm`, custom session, memory and artifact services, callbacks
  (`before_/after_model|tool|agent`) and plugins. Sources: `src/google/adk/{runners.py, models/llm_response.py,
  models/base_llm.py}` in [adk-python](https://github.com/google/adk-python).

### 2.4 LangGraph and LangChain v1 (`langgraph` 1.2.12, `langchain` 1.4.2)

- **Abstractions.**
  - `StateGraph(State)` with per-key reducers, `Command(update, goto)`, and `Send` for map-reduce fan-out.
  - Functional API: `@entrypoint` / `@task`. On resume, task results are replayed from the checkpoint, so "the
    workflow replays forward until it reaches the pause again"
    ([functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)).
- **Loop.**
  - `create_agent(model, tools, middleware=[…])` builds a model→tools graph.
  - `AgentMiddleware` hooks: `before_model`, `after_model`, `wrap_model_call` and `wrap_tool_call`, with jumps to
    `end`, `tools` or `model` ([middleware](https://docs.langchain.com/oss/python/langchain/middleware)).
- **Durability.**
  - Checkpointers (`PostgresSaver`, among others) save one checkpoint per super-step per `thread_id`.
  - Durability modes: `sync`, `async` and `exit`.
  - Time travel: `get_state_history`, and `update_state` to fork.
  - Sources: [checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers),
    [Durability](https://reference.langchain.com/python/langgraph/types/Durability).
- **HITL.**
  - `interrupt(payload)` pauses and `Command(resume=value)` resumes. On resume "the runtime restarts the entire
    node from the beginning", so code before `interrupt` must be idempotent.
  - `HumanInTheLoopMiddleware` decisions: `approve`, `edit`, `reject` and `respond`.
  - Sources: [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts),
    [HITL](https://docs.langchain.com/oss/python/langchain/human-in-the-loop).
- **Memory.** `BaseStore`: `put(namespace_tuple, key, value, index)`, `get` and
  `search(namespace_prefix, query, filter)`, with optional embedding indexes
  ([stores](https://docs.langchain.com/oss/python/langgraph/stores)).
- **Multi-agent.**
  - The docs recommend five patterns: subagents-as-tools, handoffs, skills, router and custom workflows
    ([multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)).
  - `langgraph-supervisor` is archived; `langgraph-swarm` (`create_handoff_tool`, `active_agent`) is still
    active.
- **Streaming.** Modes `values`, `updates`, `messages` (token plus metadata), `custom` (`get_stream_writer()`),
  `checkpoints`, `tasks` and `debug` ([streaming](https://docs.langchain.com/oss/python/langgraph/streaming)).
- **Platform (LangSmith Deployment / Agent Server).**
  - Resources: assistants, threads, runs, crons, and a cross-thread store.
  - Storage: Postgres plus Redis.
  - "Double-texting" strategies: `enqueue`, `reject`, `interrupt` and `rollback`
    ([agent server](https://docs.langchain.com/langsmith/agent-server),
    [double texting](https://docs.langchain.com/langsmith/double-texting)).
  - The **Agent Protocol** is an OpenAPI spec for runs, threads and store
    ([repo](https://github.com/langchain-ai/agent-protocol)).
- **Sandboxes.** Deep Agents takes a `backend` implementing `SandboxBackendProtocol.execute()`. Providers include
  Daytona, E2B, Modal, Runloop and Vercel ([sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes)).
- **RL.** Provider logprobs are available through `response_metadata`. No token IDs.

### 2.5 Pydantic AI (V2, `pydantic-ai` 2.51.0; V2 stable 2026-06-23)

- **Abstractions.**
  - `Agent(model, deps_type, output_type, instructions, tools, toolsets, capabilities)`.
  - `RunContext[Deps]`.
  - `agent.iter()` exposes the loop's graph nodes (`UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End`).
  - `UsageLimits` includes `cost_limit`.
  - V2 is built around **capabilities**: bundles of tools, hooks, instructions and settings
    ([capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)).
- **Tools.**
  - `@agent.tool` and `@agent.tool_plain`. `ModelRetry` sends feedback to the model, and `prepare` changes tool
    definitions per step.
  - Toolsets can be combined: `Filtered`, `Prefixed`, `Renamed`, `Prepared`, `ApprovalRequired`, `External`,
    `MCPToolset` ([toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)).
- **HITL (deferred tools).**
  - Triggers: `requires_approval=True`, `ApprovalRequired`, or `CallDeferred` (the tool is executed externally).
  - The run **ends** with `DeferredToolRequests` and is resumed with `DeferredToolResults(approvals={id: True |
    ToolApproved(override_args=…) | ToolDenied(message)}, calls={id: value})`.
  - Source: [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/).
- **Durability.**
  - Capabilities: `TemporalDurability`, `DBOSDurability` and `PrefectDurability`. The `TemporalAgent`/`DBOSAgent`
    wrappers are deprecated.
  - Model requests, toolset calls and MCP traffic become activities or steps; the loop stays in workflow code.
  - Stable agent `name` and toolset `id` are required.
  - There is a "durable execution backend builder" for third-party runtimes.
  - Sources: [overview](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/),
    [Temporal](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/).
- **Memory.** The application manages message history; the harness adds a Memory capability.
- **Multi-agent.** Delegation through tools (with shared usage), programmatic hand-off, and `pydantic_graph`.
- **Sandboxes.** Harness `FileSystem`/`Shell`, a Modal sandbox, and "Code Mode" in the Monty sandbox
  ([harness](https://pydantic.dev/docs/ai/harness/)).
- **RL.** OpenAI logprobs in `provider_details['logprobs']`. No token IDs.

### 2.6 CrewAI (1.15.22)

- **Abstractions.** `Agent(role, goal, backstory)`, `Task`, `Crew(process=sequential|hierarchical)`, and **Flows**
  (`@start`, `@listen`, `@router`, `@persist`) ([flows](https://docs.crewai.com/en/concepts/flows)).
- **HITL.** `@human_feedback`, plus a provider that raises `HumanFeedbackPending`. The flow state is persisted and
  resumed with `F.from_pending(id).resume(feedback)`
  ([HITL](https://docs.crewai.com/en/learn/human-feedback-in-flows)).
- **Durability.** `checkpoint=True` or `CheckpointConfig(on_events=[…])` on `Crew`, `Flow` and `Agent`, with fork
  and restore ([checkpointing](https://docs.crewai.com/en/concepts/checkpointing)).
- **Memory.** One `Memory` class (LanceDB) with hierarchical scopes such as `/agent/researcher`. Recall combines
  semantic similarity, recency and importance ([memory](https://docs.crewai.com/en/concepts/memory)).
- **Sandboxes.** External (E2B, Modal); built-in code execution is deprecated.
- **RL.** Nothing native found **[unverified]**.

### 2.7 Mastra (`@mastra/core` 1.71.0)

- **Agent.** `new Agent({id, instructions, model, tools, memory, agents, workspace})`; tools come from
  `createTool({inputSchema, outputSchema, execute})`.
- **HITL.**
  - `requireApproval: true`, then `approveToolCall({runId, toolCallId})`.
  - A tool can call `context.agent.suspend(...)` and later `resumeStream(...)`.
  - `listSuspendedRuns` finds pending runs after a restart
    ([approval](https://mastra.ai/docs/agents/agent-approval)).
- **Workflows.**
  - `createWorkflow().then().parallel().branch()…`.
  - Steps declare `suspendSchema`/`resumeSchema`; `.sleep` and `.sleepUntil` are available.
  - Snapshots live in configured storage.
  - Inngest runner; Temporal runner (experimental).
  - Source: [suspend/resume](https://mastra.ai/docs/workflows/suspend-and-resume).
- **Memory.**
  - Calls are scoped by `{resource, thread}`.
  - Working memory (resource- or thread-scoped), semantic recall, and "observational memory" (background
    compression) ([memory](https://mastra.ai/docs/memory/overview)).
- **Multi-agent.** Supervisor agents with delegation hooks; `agent.network()` is deprecated.
- **Sandboxes.** `Workspace({sandbox, filesystem})` with providers: E2B, Daytona, Modal, Vercel, Docker,
  Cloudflare, and others ([workspace](https://mastra.ai/docs/workspace/overview)).

### 2.8 Vercel AI SDK (v7, `ai` 7.0.118; AI SDK 7 announced 2026-06-25)

- **Loop.**
  - `generateText`/`streamText` with `tools`, `stopWhen` and `prepareStep` (per-step model, `activeTools`,
    `toolChoice`).
  - `ToolLoopAgent`.
  - Sources: [tools](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling),
    [agents](https://ai-sdk.dev/docs/agents/overview).
- **HITL.**
  - v7 moved approval to `toolApproval: {tool: 'user-approval' | fn}`.
  - Messages carry `tool-approval-request` and `tool-approval-response` parts, and HMAC-bound approvals are
    available ([AI SDK 7](https://vercel.com/changelog/ai-sdk-7)).
  - The `WorkflowAgent` page still shows `needsApproval` **[inconsistent in the docs]**.
- **Durability.**
  - `WorkflowAgent` (`@ai-sdk/workflow`) runs inside a `'use workflow'` function. That function is replayed from
    an event log with `Math.random` and `Date` fixed.
  - Tools marked `'use step'` are persisted, and hooks (`createHook`, `resumeHook`) handle waits.
  - Runs are pinned to their deployment.
  - Sources: [workflow agent](https://ai-sdk.dev/docs/agents/workflow-agent),
    [workflows and steps](https://workflow-sdk.dev/docs/foundations/workflows-and-steps).
- **Sandboxes.** `experimental_sandbox`; `HarnessAgent` wraps Claude Code, Codex and others inside a Vercel Sandbox.
- **RL.** OpenAI logprobs through `providerMetadata`.

### 2.9 Letta

- **Re-platformed in 2026.**
  - The V1 server is archived. Current source is `letta-code`, which contains the harness, App Server and runtime.
  - "Letta Cloud stores agents' memory, identity, and conversations."
  - Sources: [letta repo](https://github.com/letta-ai/letta), [letta-code](https://github.com/letta-ai/letta-code),
    [terminology](https://docs.letta.com/reference/terminology/index.md).
- **Identity.**
  - An agent (`agent-xxx`) has a name, a system prompt, memory, model/tool configuration, and many conversations
    (`conv-xxx`).
  - "Memory is shared across an agent's conversations."
  - Messages sent during an active turn are queued, and pending approvals survive disconnection.
  - Sources: [stateful agents](https://docs.letta.com/concepts/stateful-agents/index.md),
    [sessions](https://docs.letta.com/agent-sdk/sessions/index.md).
- **Memory.**
  - **MemFS** is a git-backed memory filesystem. Files under `system/` are pinned into the prompt, and every edit
    is a commit.
  - Org-shared memory repos.
  - "Dreaming" (sleep-time) subagents consolidate memory in the background.
  - Sources: [MemFS](https://docs.letta.com/concepts/memfs/index.md),
    [shared memory](https://docs.letta.com/concepts/shared-memory/index.md).
  - The V1 API had labelled memory blocks with `limit` and `read_only`, shared blocks, and archival passages
    ([memory blocks](https://docs.letta.com/guides/agents/memory-blocks)).
- **Messaging.**
  - `SendAgentMessage` is asynchronous, attaches a return address (the caller's agent and conversation IDs), and
    expects an explicit reply ([PR #4392](https://github.com/letta-ai/letta-code/pull/4392)).
  - V1 had `send_message_to_agents_matching_tags` **[snippet only]**.
- **Portability.** Agent File `.af` serializes an agent's full state with secrets nulled
  ([agent-file](https://github.com/letta-ai/agent-file)).
- **RL.**
  - `messages.create` accepts `return_logprobs` ("Useful for RL training") and `return_token_ids` ("for ALL LLM
    generations in the agent step")
    ([API](https://docs.letta.com/api/typescript/resources/agents/subresources/messages/methods/create)).
  - Research items: "Memory Models … memory-native RL" and "Trajectory" (a format for agent experience data)
    ([research](https://www.letta.com/research/)).
- **Gaps.** The V1 docs warned that concurrent requests to one agent are "undefined behavior". No first-class
  per-agent credential policy was found.

### 2.10 Microsoft Agent Framework (MAF 1.0 GA April 2026) and AutoGen

- **Abstractions.**
  - `Agent` (Python) / `AIAgent` (.NET). `AgentThread` was renamed `AgentSession` (`session_id`,
    `service_session_id`, `state`, serializable) ([session](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/session)).
  - Tools: `@tool(approval_mode="always_require")`, and `FunctionTool(func=None)` for declaration-only tools
    ([function tools](https://learn.microsoft.com/en-us/agent-framework/agents/tools/function-tools)).
- **Workflows.**
  - Executors and edges, with fan-out/fan-in edge groups.
  - HITL: `ctx.request_info()`.
  - Superstep checkpoints use a restricted unpickler. Stable agent IDs are required to rehydrate
    ([checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints)).
  - Orchestrations: sequential, concurrent, handoff (a mesh with approval requests), group chat and magentic.
- **Durable agents.**
  - "Each agent session is a durable entity" on Durable Task. The HTTP API returns
    `x-ms-thread-id: @dafx-<agent>@<guid>`.
  - Session TTL defaults to 14 days; entity state is limited to 1 MB; streaming goes through callbacks such as
    Redis Streams.
  - Orchestrations race `wait_for_external_event` against timers for HITL.
  - Sources: [durable agents](https://learn.microsoft.com/en-us/azure/durable-task/sdks/durable-agents-microsoft-agent-framework),
    [Functions hosting](https://learn.microsoft.com/en-us/agent-framework/hosting/azure-functions).
- **AutoGen.**
  - Now in maintenance mode.
  - Actor core: `AgentId(type, key)`, lazily instantiated. `send_message` for RPC and `publish_message` to
    `TopicId(type, source)` with subscriptions. The gRPC distributed runtime is experimental. "Paging in/out… not
    yet implemented."
  - Sources: [identity](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/agent-identity-and-lifecycle.html),
    [messaging](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html).

### 2.11 Cloudflare Agents SDK

- **Addressing and state.**
  - `Agent` extends a Durable Object and is addressed at `/agents/{class}/{name}` or with
    `getAgentByName(env.X, name)`.
  - `this.setState` persists to embedded SQLite and syncs to clients; `this.sql` runs queries.
  - `schedule()` and `scheduleEvery()` (alarm-driven), `queue()`, `destroy()`, and hibernation that keeps
    WebSockets open.
  - Sources: [agent class](https://developers.cloudflare.com/agents/runtime/lifecycle/agent-class/index.md),
    [state](https://developers.cloudflare.com/agents/runtime/lifecycle/state/index.md).
- **Durable execution.**
  - "Fibers": `runFiber` and `startFiber(name, fn, {idempotencyKey})`, `ctx.stash(snapshot)` and
    `onFiberRecovered`. This is explicit snapshotting, not replay.
  - The `Think` harness runs each turn inside a recovery fiber.
  - Sources: [durable execution](https://developers.cloudflare.com/agents/runtime/execution/durable-execution/index.md),
    [Project Think](https://blog.cloudflare.com/project-think/).
- **Sub-agents.** `this.subAgent(Class, name)` returns a typed RPC stub for a child that has its own SQLite, under
  nested URLs ([sub-agents](https://developers.cloudflare.com/agents/runtime/execution/sub-agents/index.md)).
- **HITL.** `AgentWorkflow` with `waitForApproval(step, {timeout: "7 days"})`, and chat tools with `needsApproval`
  ([HITL](https://developers.cloudflare.com/agents/concepts/agentic-patterns/human-in-the-loop/index.md)).
- **Streaming.** Resumable chat streaming: chunks are buffered in SQLite.
- **Channels.** Email (`onEmail`, `routeAgentEmail`). `McpAgent` is deprecated in favour of the stateless
  `createMcpHandler`.
- **Sandbox.** `getSandbox(env.Sandbox, this.name)` gives one persistent container per agent instance
  ([sandbox](https://developers.cloudflare.com/agents/tools/sandbox/index.md)).

### 2.12 Restate, DBOS and Temporal agent guidance

- **Restate.**
  - Agent loop: each LLM and tool call is wrapped in `ctx.run` and journaled.
  - Sessions are **Virtual Objects** keyed by session ID, with a single writer per key; concurrent calls queue.
  - HITL uses awakeables with `.orTimeout`; the agent suspends and uses no compute while it waits.
  - Remote agents are called with `call` (request/response) or `send` (one-way); `RestatePromise.all` fans out.
  - Streaming goes through `@restatedev/pubsub`.
  - Sources: [durable agents](https://docs.restate.dev/ai/patterns/durable-agents),
    [sessions](https://docs.restate.dev/ai/patterns/sessions),
    [HITL](https://docs.restate.dev/ai/patterns/human-in-the-loop),
    [multi-agent](https://docs.restate.dev/ai/patterns/multi-agent).
- **DBOS.**
  - The loop is a `@DBOS.workflow()` and every call is a `@DBOS.step()`; steps run at least once.
  - `DBOS.send(workflow_id, message, topic)` / `DBOS.recv(topic, timeout)`: the workflow ID acts as the agent
    address.
  - `write_stream` / `read_stream`.
  - Sources: [HITL](https://docs.dbos.dev/ai/hitl),
    [communication](https://docs.dbos.dev/python/tutorials/workflow-communication).
- **Temporal.**
  - Entity workflows: one workflow per entity ID, fed by Signals and Updates, with `continue_as_new` to bound
    history.
  - **Update-with-Start** creates the entity lazily and returns a synchronous reply in one call
    ([entity workflow](https://docs.temporal.io/design-patterns/entity-workflow),
    [message passing](https://docs.temporal.io/develop/python/message-passing) **[snippet]**).

### 2.13 Sandbox SDKs

| SDK | Handle and identity | Templates | Persistence | Networking and exposure |
|---|---|---|---|---|
| **E2B** (`e2b` 2.51.0) | `Sandbox.create(template, timeout, metadata, envs, network, lifecycle, …)`; `connect(id)` (also resumes); `commands.run(background=True)` returns a handle; reattach with `commands.connect(pid)`; `files.watch_dir` ([source](https://raw.githubusercontent.com/e2b-dev/E2B/main/packages/python-sdk/e2b/sandbox_sync/main.py)) | Code-defined builder `Template().from_ubuntu_image().copy().run_cmd().set_start_cmd(cmd, wait_for…)`; **per-instruction layer cache**; copied files are content-hashed; the start command's process is captured in the snapshot ([caching](https://docs.e2b.dev/template/caching.md)) | `pause()` keeps memory and disk, retained indefinitely; snapshots; `fork(count)` ([persistence](https://docs.e2b.dev/sandbox/persistence.md)) | Domain allowlists via SNI/Host; `update_network`; `get_host(port)`; MCP gateway |
| **Modal** (`modal` 1.5.5) | `Sandbox.create(app, name, tags, image, timeout, block_network, outbound_*_allowlist, …)`; `from_id` / `from_name`; `exec` returns a process with streaming stdout ([reference](https://modal.com/docs/reference/modal.Sandbox)) | Chained `Image` builder with per-layer cache | Filesystem snapshots as Images (TTL); memory snapshots alpha; no pause | Tunnels (`encrypted_ports`), connect tokens |
| **Daytona** (`daytona` 0.218.0) | `create(params)`; `get(id_or_name)`; `process.exec`, sessions, `fs`, `git`, `code_interpreter`, `computer_use` ([SDK](https://www.daytona.io/docs/en/python-sdk/sync/sandbox.md)) | Declarative `Image` builder; "snapshots" are templates | States started/stopped/paused/archived; auto-stop, auto-archive and auto-delete intervals; `ttl_minutes` | Preview links, SSH, allowlists |
| **agent-sandbox** (v1.0.4, CRDs `v1beta1`) | `Sandbox` (singleton stateful pod, `operatingMode: Running|Suspended`), `SandboxTemplate`, `SandboxClaim`, `SandboxWarmPool`; Python `SandboxClient` ([repo](https://github.com/kubernetes-sigs/agent-sandbox)) | Pod template | Suspend; snapshots **[unverified]** | NetworkPolicy from the template |
| **Cloudflare Sandbox** | `getSandbox(ns, id)` gets or creates by ID; `exec`, `startProcess` with `waitForPort`, `exposePort` ([API](https://developers.cloudflare.com/sandbox/api/)) | Container image | Backups to R2 | Preview URLs |
| **Vercel Sandbox** (Firecracker) | `Sandbox.create({name, source, persistent, networkPolicy, tags})`; `get({name})`; `runCommand({detached})` ([SDK](https://vercel.com/docs/sandbox/sdk-reference)) | Snapshots | Auto-snapshot on stop | `domain(port)`; header-injection credential transforms |

None of these SDKs documents idempotency keys on create. Name-based identity is the practical substitute
([sandbox notes](#11-sources)).

### 2.14 Protocols and cross-framework training

- **MCP.**
  - The 2025-11-25 spec added experimental **Tasks** (call-now, fetch-later; states `working`,
    `input_required`, `completed`, `failed`, `cancelled`) ([spec](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks)).
  - The 2026-07-28 spec makes MCP stateless (no `initialize` handshake, no session header) and moves Tasks into an
    official extension: `tools/call` may return a task handle that the client drives with `tasks/get`,
    `tasks/update` and `tasks/cancel`.
  - In the same spec, elicitation returns `InputRequiredResult` rather than holding a stream open
    ([release notes](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/)).
- **A2A.**
  - Linux Foundation project; v1.0 in January 2026 with signed Agent Cards.
  - A task lifecycle protocol: discovery, delegation, status updates and artifacts
    ([repo](https://github.com/a2aproject/A2A), [LF](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year)).
  - The v1.0 date comes from a secondary source **[unverified]**.
- **Agent Lightning** (Microsoft, v1.0 August 2026). It trains agents from any framework through an API gateway
  that "proxies model requests and captures training data" with "ZERO changes". It notes that returning token IDs
  through the OpenAI-compatible API matters for avoiding retokenization drift ([repo](https://github.com/microsoft/agent-lightning)).
- **Polar** (May 2026). It "proxies LLM API calls, records token-level model interactions, and reconstructs
  token-faithful trajectories" for arbitrary harnesses ([arXiv 2605.24220](https://arxiv.org/abs/2605.24220)).
- **OpenTelemetry GenAI semantic conventions** (`invoke_agent`, `chat`, `execute_tool`) are widely adopted but
  still in "Development" status ([Greptime](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions)).

---

## 3. Comparison matrix

The cells are condensed from section 2. "Ours" is the current design plus the proposals in
[research/README](README.md).

| | Loop owner / expression | Tool definition | Durability mechanism | HITL / long waits | Identity and memory | Multi-agent | Streaming | Sandboxes | RL data |
|---|---|---|---|---|---|---|---|---|---|
| **Ours** | Framework loop; Task (environment side) / Agent (policy side); `Program` for free-form | `@tool` methods run in the sandboxed host; imports through the router (MCP, HTTP, agent, human); environment-provided MCP | Log + deterministic replay; effects deduplicated three ways at receivers; disposable snapshots | `run.signal` / timers; proposed `requires_approval`, `WaitFor` | None yet (proposed `run.memory`, identities) | `spawn` / `send` / `signal` by `run_id`; `agent` binding | Proposed: ephemeral tokens + `run.emit` | Handles over microVMs; content-addressed templates | Recorder: exact tokens, behavior logprobs, versions; agent sees none |
| OpenAI Agents | Agent-owned loop (`Runner`) | `@function_tool`, hosted, MCP | None in core; Temporal / DBOS / Restate / Dapr wrappers | Approvals end the run; `RunState` serialized | Sessions (history) | Handoffs, agents-as-tools | Rich event stream | `SandboxAgent` + 9 clients | Output-text logprobs |
| Claude Agent SDK | CLI subprocess loop | In-process MCP `@tool` | Session JSONL + `SessionStore` | `can_use_tool` (edit input), `defer` | CLAUDE.md, skills | Subagents via Agent tool | Raw stream events | OS sandbox for Bash | None |
| Google ADK | Event loop + 2.0 graph | `FunctionTool`, long-running tools | Event log; resume (tools at least once); Temporal / DBOS plugins | Confirmation (limited backends), `RequestInput` | `user:` / `app:` state; MemoryService | sub_agents, transfer, A2A | SSE / BIDI | Code executors, EnvironmentToolset | Logprobs if backend returns them |
| LangGraph | Graph / functional; middleware | LangChain tools | Checkpoint per super-step; node re-runs on resume | `interrupt()` / `Command(resume)`; approve / edit / reject / respond | Store (namespaces, semantic search) | Send, handoffs, swarm | 7 stream modes | Deep Agents backends | Provider logprobs |
| Pydantic AI | Agent-owned; capabilities | Tools + composable toolsets | Temporal / DBOS / Prefect / Restate capabilities; backend builder | Deferred tools end the run | App-managed | Delegation, graph | Event stream (not across workflow boundary) | Harness capabilities | Provider logprobs |
| CrewAI | Crew process / Flows | `BaseTool`, `@tool`, MCP | Checkpoints on events | `HumanFeedbackPending` + resume | Unified scoped Memory | Hierarchical manager | Stream flag | External | None |
| Mastra | Agent + workflow DSL | `createTool` | Storage snapshots; Inngest | `requireApproval`, suspend / resume | resource / thread, working memory | Supervisor | text / full streams | Workspace providers | None found |
| Vercel AI SDK | `ToolLoopAgent`; `WorkflowAgent` | `tool()` | Event-log replay (`'use workflow'`) | `toolApproval`, hooks | App-managed | Subagents as tools | UI message streams | `experimental_sandbox`, HarnessAgent | Provider logprobs |
| Letta | Server-side agent step | Server / client tools, MCP | DB-persisted agent state | `requires_approval` stop reason | **Identity-first**: MemFS, shared memory, conversations | Async messages with return address | Token / step SSE | E2B for tools; client-side | `return_token_ids`, logprobs |
| MAF | Agent + workflows | `@tool(approval_mode)` | Superstep checkpoints; durable entities | `request_info`, external events | AgentSession; entity per session | 5 orchestrations | Callbacks | Hosted CodeAct / shell | None |
| Cloudflare | Durable Object per agent | AI-SDK tools | SQLite state + fibers (stash) | `waitForApproval`, `needsApproval` | **Name-addressed** DO + SQLite | Sub-agents (RPC) | Resumable chat stream | One sandbox per agent | None |
| Restate / DBOS / Temporal | Loop in workflow / handler | Steps / activities | Journal replay | awakeable / `recv` / signal | Virtual object / workflow ID / entity workflow | call / send; queues | pubsub / `write_stream` | Activities | None |

---

## 4. Analysis: where our interface differs and why it matters

### 4.1 Where ours is stronger

1. **Durability that reaches the receivers.**
   - Every durable integration surveyed makes the model call and the tool call a journaled step. Steps run at
     least once: DBOS and ADK say so explicitly; LangGraph re-runs the whole node on resume.
   - Idempotency then depends on the tool author.
   - Ours puts `effect_id` into every receiver: envd, the recorder and the router bindings (P6). Retried model
     calls return the *same sample*, and retried `execute` calls join the in-flight process.
   - None of the sandbox SDKs even offers an idempotency key on `create`.
2. **RL-exact data without trusting the harness.**
   - The frameworks at best pass through provider logprobs for output text.
   - Letta is the one exception: it exposes `return_token_ids`.
   - Cross-framework trainers (Agent Lightning, Polar) have converged on *proxy capture*, which is what our
     recorder already is.
   - P4 goes further: the agent never sees tokens, so there is no path for a harness to corrupt training data.
3. **One loop for RL and applications.** The Task/Agent split is the Gym environment/policy split. No application
   framework separates "what executes tools and scores" from "what chooses the context and acts". That separation
   is what lets us hold a task fixed and swap agents (evaluation), or hold an agent fixed and swap tasks (training).
4. **Environments as owned, typed resources** (P11) with security classes enforced outside the guest. Framework
   sandboxes are either OS-level and fail open (the Claude SDK runs unsandboxed with a warning if the sandbox is
   unavailable) or are vendor clients with no ownership model.
5. **Credential isolation.** Task code has no network; credentialed access goes only through imports and brokers
   (P8). The frameworks inject secrets into tool processes: Letta V1 passes `LETTA_API_KEY` and environment
   variables into E2B, and Vercel v7 per-tool `toolsContext` is the best of them.

### 4.2 Where ours is weaker, or missing something

1. **Foreign async code cannot run in our host.**
   - The replay driver "drives coroutines like generators; no asyncio event loop" and "rejects foreign
     awaitables".
   - The OpenAI Agents `Runner`, Pydantic AI, LangGraph and MCP client libraries all use `asyncio` primitives
     internally.
   - Temporal can host the OpenAI Agents SDK and ADK in-workflow only because its Python SDK provides a
     *deterministic asyncio event loop*.
   - This is the biggest interoperability gap.
2. **No identity, memory or mailbox concepts.**
   - Letta, Cloudflare, Restate Virtual Objects, MAF durable entities, AutoGen `AgentId` and ADK `user:` state all
     have a stable address with state behind it.
   - We only have `run_id`s, which are ephemeral by design.
3. **HITL is under-specified.**
   - The industry vocabulary is now stable: approve, reject with a message, edit the arguments, respond instead of
     executing, sticky decisions, and predicate-based policies.
   - Our `requires_approval=True` covers only the first of these.
4. **No streaming design.** Every framework has a typed event stream, and LangGraph and Cloudflare make it
   resumable. We have only a proposal.
5. **Tools lack a composable bundle and a per-call context.**
   - Pydantic AI V2 capabilities, LangChain middleware, ADK plugins and OpenAI tool guardrails all let a reusable
     bundle add tools *and* intercept calls.
   - Our only reuse mechanism is subclassing, and `@tool` bodies cannot see their `call_id` (which approvals and
     progress need).
6. **Environment handles lack what agents use daily:** background processes, exposed ports and named reconnection.
   The sandbox SDKs all have these.
7. **Template caching is coarse.**
   - We hash the whole recipe, so changing one `run` line rebuilds everything.
   - E2B, Modal and Daytona cache per instruction, and E2B content-hashes copied files and captures a running start
     command in the snapshot.

### 4.3 Where ours is merely different

- **Code-first, not graphs.** LangGraph, ADK 2.0, MAF workflows and Mastra are graph DSLs. Temporal, DBOS,
  Restate, the Vercel Workflow SDK and LangGraph's functional API are code-first with replay, like ours. Graphs
  offer visualisation and superstep checkpoints, but we don't need them because replay gives exact resumption.
  No change.
- **Suspension by replay vs "stop the world".**
  - The OpenAI SDK, Pydantic AI and the Claude SDK (`defer`) implement HITL by *ending* the run and serializing
    state the caller must bring back.
  - Ours, like Temporal, DBOS, Restate and Cloudflare, suspends in place without holding compute (R1).
  - Ours is the better model. But the foreign-framework adapters must translate "run ended with pending approvals"
    into our suspension (section 8).
- **Handoffs.** OpenAI, ADK and langgraph-swarm model handoffs as the model calling `transfer_to_X`, which changes
  which prompt and tools are active *within one conversation*. In our split this is Agent-level policy
  (`select_context` + `act` + `tools_for_turn`), not a platform primitive. We should keep it that way and supply a
  library class for it.
- **The Task owns the tools.** In every framework the tools belong to the agent. Ours puts them on the Task
  (environment side), so the same agent can be evaluated on different action spaces. Application authors will find
  this unusual, which is what the `Program`-level convenience in R11 is for.

---

## 5. Durable agent identities: definition and proposed interface

### 5.1 What the prior art gets right and wrong (analysis)

- **Address.** The address should be stable and human-readable, and separate from the internal ID. Examples:
  Cloudflare `/agents/{class}/{name}`, AutoGen `AgentId(type, key)`, Letta `agent-…` IDs, MAF
  `@dafx-{agent}@{session}`.
- **Single consumer.** Serial processing per key is essential: Restate Virtual Objects, Durable Objects and
  Durable Entities all do it. Letta V1 shows what happens without it ("undefined behavior").
- **Bounded history.** Temporal needs `continue_as_new`, and MAF caps entity state at 1 MB. An identity must
  **not** be one infinitely long replayed run.
- **Memory shared across conversations but not across identities by default,** with explicit shared blocks or
  repositories (Letta).
- **Gap: credentials and policy.** Nobody binds credentials and policy to the identity. Letta, Cloudflare and MAF
  leave authorization to the application, and MAF explicitly warns that its session ID is not an authorization
  boundary. This is where we can do better.

### 5.2 Definition (proposed, normative wording)

> A **durable agent identity** is a named, long-lived, owned resource that has these properties:
>
> 1. **Address**: an immutable `agent_id` (`ag_{cell_id}_{ulid}`, which embeds the home cell) and a mutable,
>    unique name `{tenant}/{namespace}/{name}`.
> 2. **Deployment**: a reference to a named deployment, which maps to code references (Program, or Task + Agent)
>    and a `RunBinding` template. The identity has a **version policy**: `PINNED` (fixed code references) or
>    `FOLLOW` (take the deployment's current version at the next activation).
> 3. **State**: small, typed, versioned records owned by the identity's code. They are read and written only
>    through effects and carried across activations. They are never a pickled coroutine.
> 4. **Memory**: namespaced key–value records and pinned context blocks, scoped to the identity. They can be
>    shared through explicitly granted shared scopes.
> 5. **Mailbox**: a durable, ordered inbox of `Envelope`s with deduplication by `message_id`. It has a **single
>    consumer**: at most one activation processes the mailbox of a given *lane* (default: one lane per identity;
>    optionally one per conversation).
> 6. **Grants**: the identity is a *principal*. Imported tools called by its runs are authorized, and brokered
>    credentials resolved, as that principal, possibly on behalf of a user through a delegated grant. An access
>    control list (ACL) says who may send to its mailbox and who may read its streams.
> 7. **Owned resources** (P11): environments (named workspaces), snapshots, schedules, and child identities.
>    Deleting the identity destroys what it owns.
> 8. **Policy binding**: model slots are bound to recorder channels (for example `acme/support@stable`), so
>    training moves the identity's policy without changing its code (R5). Every activation is recorded as sessions
>    of those channels.
> 9. **Lifecycle**: `ACTIVE` (idle or running) ⇄ `PAUSED` (mailbox accepts, nothing is consumed) → `ARCHIVED`
>    (read-only, sends rejected) → `DELETED` (state crypto-shredded with the tenant or identity key).
>
> **Relation to runs.** An identity does no work by itself. Work happens in **activation runs**, which are ordinary
> runs with `run.identity` set:
> - an activation is started by the platform when the mailbox lane has messages and no activation is live;
> - it consumes messages through its Task, suspends with `WaitFor` between them, and ends when its episode ends
>   (the conversation closes, a budget runs out, or an idle timeout passes);
> - the next message starts a new activation.
>
> Runs therefore stay bounded, replayable and one-trajectory-per-episode. State and memory, not the run log, carry
> continuity between episodes.

This matches proposal 5 in [research/README](README.md): an entity row plus activations with
`partition_concurrency = 1`, and suspension at turn boundaries. It also makes the choice of substrate a detail of
how activations are scheduled.

### 5.3 Interface sketch

```python
# --- Control plane (Control API / SDK client) ---------------------------------
@dataclass(frozen=True)
class AgentIdentitySpecification:
    name: str                                         # "{namespace}/{name}" within the tenant
    deployment: str                                   # deployment name → code references + RunBinding template
    version_policy: VersionPolicy = VersionPolicy.FOLLOW
    binding_overrides: RunBinding | None = None       # e.g. a per-identity channel or security classes
    grants: Sequence[GrantReference] = ()             # broker entries this principal may use
    mailbox: MailboxPolicy = MailboxPolicy()
    accept_from: AccessControlList = AccessControlList.tenant()
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class MailboxPolicy:
    lanes: LaneMode = LaneMode.PER_IDENTITY           # PER_IDENTITY | PER_CONVERSATION
    on_busy: BusyPolicy = BusyPolicy.ENQUEUE          # ENQUEUE | INTERRUPT | REJECT (LangGraph "double texting")
    capacity: int = 10_000                            # beyond this, send() fails with MAILBOX_FULL
    idle_activation_timeout: timedelta = timedelta(minutes=30)   # end the activation after this long with no messages

client.identities.create(AgentIdentitySpecification(name="support/alice-bot", deployment="support-agent"),
                         request_id="…")            # idempotent on request_id and on name
client.identities.send(Address.identity("support/alice-bot"),
                       Envelope(kind="message", content=[Text("Hi")], conversation="slack:C1/171.2"),
                       idempotency_key="slack-event-Ev123")      # signal-with-start: activates if idle

# --- Inside an activation run (task code) --------------------------------------
class IdentityContext:                                 # run.identity; None for plain runs
    agent_id: str
    address: Address
    conversation: str | None                           # the lane / conversation this activation serves
    state: IdentityState                               # versioned records
    memory: Memory                                     # section 7.3
    resources: IdentityResources                       # named environments, schedules

class IdentityState:
    async def get(self, key: str) -> VersionedValue | None: ...
    async def put(self, key: str, value: Any, *, expected_version: int | None = None) -> int: ...
        # compare-and-set; raises VersionConflict. Committed as an effect, so it is replay-safe.

class IdentityResources:
    async def workspace(self, name: str, specification: EnvironmentSpecification) -> Environment: ...
        # get-or-create by (agent_id, name); owned by the identity, attached to this run
    async def schedule(self, name: str, cron: str, envelope: Envelope) -> None: ...
        # an identity-owned trigger that sends to its own mailbox
```

A conversational identity is an ordinary Task whose `respond` waits for the next message instead of ending
(section 7.1):

```python
class SupportConversation(Task):
    imports = ["zendesk"]

    async def setup(self, run):
        self.workspace = await run.identity.resources.workspace("notes", NOTES_ENVIRONMENT)

    async def start(self, run):
        envelope = await run.mailbox.next()                        # the message that caused activation
        profile = await run.identity.memory.get(("users", envelope.sender.subject))
        return Observation(render_first_turn(envelope, profile))

    async def respond(self, run, reply):
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        await run.emit("reply", reply.content, to=run.origin)      # delivered by the Slack connector
        return WaitFor("message", timeout=timedelta(hours=24),
                       on_timeout=End(truncated=True))

    async def resume(self, run, event: Signal) -> Observation:     # called when WaitFor is satisfied
        return Observation(event.envelope.content)
```

### 5.4 RL on identities (analysis)

- **Episodes.** Each activation is an episode, and its trajectory is the activation's recorder sessions.
- **Rewards.**
  - Task-side rewards work as today.
  - Late human feedback ("thumbs up" two days later) needs **delayed rewards on completed runs**:
    `client.runs.reward(run_id, reply_effect_id, value, key="user_feedback")`. This appends a `reward.assigned`
    event to a terminal run's log (a new, runtime-only exception to "terminal means closed") and re-emits the
    affected samples with a new `sample_id` revision.
  - Whether trainers accept revised samples is an open question (section 10).
- **Policy per identity.**
  - By default, many identities share one channel.
  - Per-identity adapters (for example LoRA) would be a channel per identity and are an inference-plane concern.
  - Nothing in this interface prevents it.

---

## 6. Swarms: messaging and coordination interface

### 6.1 Observed patterns (facts from section 2)

1. **Agent-as-tool / call-return:**
   - OpenAI `as_tool`, ADK `AgentTool`, Claude subagents, Mastra supervisors, Cloudflare `subAgent` stubs, and
     Pydantic delegation.
   - Ours: the `agent` import binding and `run.spawn`.
2. **Handoff** (control transfer within a conversation): OpenAI handoffs, ADK `transfer_to_agent`,
   langgraph-swarm, and the MAF handoff mesh.
3. **Peer messaging with addresses:**
   - AutoGen `send_message(AgentId)` and `publish_message(TopicId)`.
   - Letta `SendAgentMessage` (asynchronous, return address).
   - Restate `call` and `send`.
   - DBOS `send` / `recv(topic)`.
4. **Structured fan-out / fan-in:** LangGraph `Send`, MAF fan-out/fan-in edges, `RestatePromise.all`, DBOS queues
   with `worker_concurrency`.
5. **Group chat / magentic** orchestrators (MAF): a manager picks the next speaker. This is an application
   pattern built on 1–4.

### 6.2 Proposed interface

We add addressing and envelopes, and keep spawn and gather:

```python
@dataclass(frozen=True)
class Address:
    kind: Literal["run", "identity", "topic", "external"]
    value: str                                       # run_id | agent_id or name | "{tenant}/{topic}" | connector target
    @staticmethod
    def identity(name_or_id: str, conversation: str | None = None) -> "Address": ...
    @staticmethod
    def topic(name: str) -> "Address": ...

@dataclass(frozen=True)
class Envelope:
    kind: str                                        # "message", "task", "result", application-defined
    content: list[Block]                             # canonical content
    data: JsonValue | None = None                    # structured payload (validated if the kind declares a schema)
    conversation: str | None = None
    correlation_id: str | None = None                # set on replies
    reply_to: Address | None = None                  # defaults to the sender
    message_id: str = ""                             # set by the runtime: the sender's effect_id → deduplication key
    sender: Principal | None = None                  # set by the runtime; never trusted from the payload

class RunContext:                                    # additions
    mailbox: Mailbox
    async def send(self, to: Address, envelope: Envelope) -> None: ...              # fire-and-forget; durable
    async def request(self, to: Address, envelope: Envelope, *,
                      timeout: timedelta | None = None) -> Envelope: ...          # send + await correlated reply
    async def reply(self, to: Envelope, envelope: Envelope) -> None: ...
    async def publish(self, topic: str, envelope: Envelope) -> None: ...
    async def map(self, specifications: Iterable[RunSpecification], *,
                  concurrency: int = 16, quorum: int | None = None,
                  on_error: ErrorPolicy = ErrorPolicy.COLLECT) -> list[ChildResult]: ...

class Mailbox:
    async def next(self, *, kinds: Collection[str] | None = None,
                   timeout: timedelta | None = None) -> Envelope: ...              # an effect; suspends
    def pending(self) -> int: ...                                                   # as of the latest input event
```

**Semantics.**
- **Delivery.**
  - Sends are effects (`message.sent`), committed with the sender's log. Delivery into the recipient's inbox is
    at-least-once, and the inbox deduplicates by `message_id`.
  - Order is first-in, first-out per (sender, recipient lane).
  - Cross-cell delivery uses the asynchronous relay (P13).
- **Fan-out.**
  - `run.map` is the bounded version of `gather(spawn…)`. It admits at most `concurrency` children at a time
    (backpressure, and the per-tenant quota applies).
  - `quorum` returns early and cancels the rest.
  - Child results are `child.completed` events; nothing new is needed in the log.
- **Topics.**
  - Subscriptions are identity-owned records (`identity.subscribe(topic, kinds)`); publishing expands to one
    inbox write per subscriber.
  - At swarm scale (10⁴ subscribers) this is a fan-out amplifier. It must be quota-bounded or backed by a shared
    log that subscribers read by cursor. Which one is an open question.
- **Handoffs** stay in the Agent layer:

```python
class HandoffAgent(Agent):
    def __init__(self, configuration):
        self.members = {name: build_agent(c) for name, c in configuration["members"].items()}
        self.active = configuration["start"]
    async def act(self, run, history, tools):
        handoff_tools = [transfer_tool(name) for name in self.members if name != self.active]
        reply = await self.members[self.active].act(run, history, tools + handoff_tools)
        if (target := find_transfer(reply)) is not None:
            self.active = target      # the agent's state is pickled in snapshots; replay-safe
        return reply
```

  The `transfer_to_*` calls must still be answered with tool results. The loop requires a result for every call.
  So `HandoffAgent` pairs with a Task mixin `HandoffTools` that answers them.
- **Multi-agent RL attribution.**
  - Each spawned run and each identity activation records its own sessions, joined by labels (`swarm_id`,
    `job_id`, parent `run_id`).
  - Credit assignment across agents (for example a team reward) is `run.reward(value, slot=…)` on the parent plus
    a reward-propagation rule in the trajectory assembler. That rule is not an interface change.

---

## 7. Human-in-the-loop, streaming, memory

### 7.1 Suspension: `WaitFor`, `resume`, continue-as-new

- **`WaitFor` return value.** `WaitFor(kind, timeout, on_timeout)` is a return value of `Task.respond` (and of
  `start`). The loop commits `run.suspended{waiting_for: SIGNAL}` and lets the host evict the run. When a matching
  signal or mailbox message arrives, the loop calls `Task.resume(run, signal) -> Observation`.
- **Why a return value instead of `await run.signal()` inside `respond`?** The run is then at an *implicit
  resumable point* (after a hook), so the snapshot is exact and the host can drop the coroutine entirely. This
  answers the "idle workflows stay resident" problem in [research/README](README.md) item 4.
- **Short waits** (an approval in the middle of a tool) still use in-hook `await`.
- **Trajectory.** The wait is invisible: the next observation is what the user said.
- **Continue-as-new.**
  - Returning `End(truncated=True, continue_as=ContinueAs(carry=…))` ends the episode and starts a fresh
    activation in the same lane, carrying explicit state.
  - This bounds log length for chats that never end, as Temporal's `continue_as_new` does. It also gives RL a
    natural episode boundary.

### 7.2 Approvals

What converged across frameworks (facts):
- predicate-based `needs_approval(context, arguments, call_id)` (OpenAI);
- edit or override the arguments (`ToolApproved(override_args)` in Pydantic, `updated_input` in the Claude SDK,
  `edit` in LangChain);
- reject with a message the model sees (`ToolDenied`, `rejection_message`);
- respond in place of execution (LangChain `respond`);
- sticky decisions (`always_approve` in OpenAI);
- approvals on imported MCP tools too (OpenAI `require_approval`).

Proposal:

```python
ApprovalPredicate = Callable[["Task", RunContext, Mapping[str, Any]], bool]   # deterministic; no effects

@tool(approval=True)                                    # or approval=ApprovalPredicate, or a named policy
async def deploy(self, service: str, version: str) -> str: ...

@dataclass(frozen=True)
class ApprovalRequest:                                  # delivered by connectors; addressable by approval_id
    approval_id: str                                    # = effect_id of the approval.requested event
    tool_name: str
    arguments: Mapping[str, Any]
    reason: str | None
    approvers: AccessControlList                        # from the RunBinding approval policy
    expires_at: datetime | None

Decision = Approve | Reject | Respond
@dataclass(frozen=True)
class Approve:  arguments: Mapping[str, Any] | None = None; remember: Remember = Remember.ONCE   # ONCE | RUN | IDENTITY
@dataclass(frozen=True)
class Reject:   message: str = "The user rejected this call."; remember: Remember = Remember.ONCE
@dataclass(frozen=True)
class Respond:  content: list[Block]                    # becomes the tool_result instead of executing
```

**Semantics.**
- **Where the policy lives.**
  - `RunBinding.approvals: map<tool_name_pattern, ApprovalPolicy>` can *add* approval requirements to any tool,
    including imports, without code changes. Policy belongs to the deployment, like security classes.
  - Code can require approval; the binding can only strengthen that requirement.
- **Effects.** Approval is an effect (`approval.requested` → `approval.decided`), implemented on the signal
  mechanism.
- **Timeouts.** The run suspends with no compute. A timeout resolves to the policy's `default_on_timeout`
  (`Reject` unless configured).
- **Model visibility.** `Reject` and `Respond` produce `tool_result`s (`is_error = true` for `Reject`), so the
  model sees the outcome, and training does too.
- **Recording.** Approval decisions are recorded in `info` and never hidden from the trajectory. Rejected calls
  are valuable negative signal.

### 7.3 Memory

What the frameworks do (facts):

| Framework | Memory shape |
|---|---|
| LangGraph | Namespaced tuple keys + optional semantic index |
| ADK | Scope prefixes `user:` / `app:` / `temp:` + a search service |
| Letta | Pinned labelled blocks with size limits, shared blocks, git-versioned files |
| CrewAI | Hierarchical scopes, recall scoring |
| Mastra | Resource vs thread scoping |

Proposal (analysis):

```python
class Memory:                                    # run.memory (scope: run's identity or tenant) — all calls are effects
    async def get(self, key: MemoryKey) -> VersionedValue | None: ...
    async def put(self, key: MemoryKey, value: JsonValue, *, expected_version: int | None = None,
                  time_to_live: timedelta | None = None) -> int: ...
    async def list(self, prefix: MemoryKey, *, limit: int = 100) -> list[MemoryRecord]: ...
    async def delete(self, key: MemoryKey, *, expected_version: int | None = None) -> None: ...
    def scope(self, name: str) -> "Memory": ...  # "identity" (default), "user:{subject}", "shared:{grant}", "tenant"
    async def blocks(self) -> list[MemoryBlock]: ...   # pinned blocks: label, text, limit, read_only

MemoryKey = tuple[str, ...]
```

- **Reads are effects.** Their results are committed to the log, so replay sees the same values even if memory
  has changed since. This is the one place where an unbounded external store meets deterministic replay.
- **Pinned blocks** reach the model only through the Agent. `ContextHints` gains `pinned: list[MemoryBlock]`, and
  `select_context` decides placement (P4: the agent owns context). For training they are ordinary context tokens.
- **Semantic recall and "dreaming"** are *not* core.
  - Recall is an imported tool binding (a vector store over MCP or HTTP).
  - Consolidation is a scheduled activation of the same identity on a frozen `summarizer` slot.
  - Large memory (Letta-style git-backed files) belongs in an identity-owned workspace environment with
    `VOLUME` persistence. The platform needs nothing new for it.

### 7.4 Streaming

Facts: typed event streams everywhere. LangGraph (`messages` and `custom` modes) and Cloudflare (chunks buffered
in SQLite) make them resumable, and Letta explicitly does *not* replay missed events. Temporal and Pydantic AI say
streaming cannot cross the workflow boundary, so events are published from inside the step.

Proposal: two tiers, one subscription API.

| Tier | Content | Durability | Resume |
|---|---|---|---|
| **Durable** | Run events projected for clients: turn started/completed, tool called/completed, `approval.requested`, `run.emit` outputs, status | the log | by `seq` cursor, exactly |
| **Ephemeral** | Token deltas (from the recorder or adapter streaming variant) and `execute` stdout/stderr, keyed by `effect_id` | none; a bounded buffer per effect | best effort; the final `model.completed` / `environment.completed` is authoritative |

```python
# client side
async for event in client.runs.stream(run_id, cursor=last_seen_seq, include=("durable", "tokens")):
    ...   # StreamEvent{seq?, effect_id?, kind, payload}; durable events carry seq, deltas carry effect_id + offset

# task side — durable, connector-delivered output (LangGraph "custom" mode, but persisted)
await run.emit("progress", {"step": "tests", "passed": 41, "failed": 2})
await run.emit("reply", reply.content, to=run.origin)   # to: an Address (external connector target)
```

- **`run.emit`** is a log event (`output.emitted`) delivered at least once by connectors, which deduplicate on
  `effect_id`.
- **Identity streams** are the union of their activations' streams, by `(activation run_id, seq)`.
- **Agents never see deltas.** Streaming is purely an observation channel for clients (P4).

---

## 8. Interoperability adapters

### 8.1 Running third-party agents on our platform

There are three tiers. They are ordered from least to most integration, and from least to most durability.

| Tier | How | Durable | Trainable | Status |
|---|---|---|---|---|
| **0: Unmanaged proxy** | The foreign harness points its base URL at recorder compatibility endpoints; tools come from the MCP facade | no | yes | designed (B18). Agent Lightning v1.0 and Polar independently chose this architecture (facts, section 2.14) |
| **1: Harness in an environment** | A Task starts the foreign harness process (Claude Agent SDK / Codex / OpenHands) inside an owned environment and waits on it. The harness's model traffic goes to a *managed* recorder session (`{run_id}/policy`), with the credential injected at egress (P8) | the outer run is durable; the inner harness survives while its environment does (`FULL_SNAPSHOT`); on environment loss it restarts from the harness's own session resume (e.g. Claude `SessionStore`) | yes, the same sessions as tier 0 | new |
| **2: In-process framework adapter** | The foreign loop runs *inside the task host* under replay. Its model interface calls `run.model.sample` and its tools become effects | fully, like the Temporal / DBOS integrations | yes, exact | new; needs R1 (asyncio driver) |

Tier 1 sketch:

```python
class ForeignHarnessTask(Task):
    """Runs any CLI harness against a repository; scores with tests."""
    async def setup(self, run):
        self.workspace = await run.environments.create(self.environment)
        self.endpoint = await run.models["policy"].compatibility_endpoint(protocol="anthropic-messages")
        # base URL for a managed session; the credential is injected at egress, never placed in the guest

    async def start(self, run):
        self.process = await self.workspace.start_process(
            ["claude", "-p", self.parameters["issue"], "--output-format", "stream-json"],
            environment_variables={"ANTHROPIC_BASE_URL": self.endpoint.url},
            cwd="/workspace")
        result = await self.process.wait()            # durable wait; joins the process if we crash and retry
        return End(info={"exit_code": result.exit_code})

    async def score(self, run):
        ...                                            # verify in a scratch environment, as in task.md
```

This only needs a zero-turn episode (an `End` from `start`) and turns reconstructed from the recorder sessions.
The loop allows that today.

Tier 2 sketch, for the OpenAI Agents SDK. `Model.get_response` is the documented extension point:

```python
class PlatformModel(agents.Model):
    def __init__(self, run: RunContext, slot: str = "policy"):
        self.run, self.slot = run, slot
    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None):
        messages = openai_items_to_messages(system_instructions, input)      # → canonical Message
        specifications = [function_tool_to_specification(t) for t in tools] + handoff_specifications(handoffs)
        reply = await self.run.models[self.slot].sample(messages, tools=specifications)
        return message_to_model_response(reply)                                # usage from the effect result

class OpenAIAgentsProgram(Program):
    """Hosts an OpenAI Agents SDK agent durably; tools become effects, model calls go to the recorder."""
    def __init__(self, configuration): self.factory = import_object(configuration["agent_factory"])
    async def main(self, run):
        agent = self.factory(tool_adapter=PlatformToolAdapter(run))   # function tools → @tool-equivalents / imports
        result = await agents.Runner.run(agent, run.input_text(),
                                         run_config=agents.RunConfig(model=PlatformModel(run),
                                                                     tracing_disabled=True))
        while result.interruptions:                                  # translate "stop the world" to suspension
            decisions = await run.approvals.request_many(result.interruptions)
            state = result.to_state(); apply_decisions(state, decisions)
            result = await agents.Runner.run(agent, state, run_config=...)
        await run.emit("result", result.final_output)
```

**Rules for tier-2 adapters.** These are the same constraints Temporal and Pydantic AI impose.
- Stable names for agents and toolsets.
- Tracing is disabled or redirected to `run.emit`.
- No network access from framework code.
- Tool bodies either run as deterministic host code or are declared as imports. For example, a Pydantic AI
  `PlatformDurability` capability is built with its "durable execution backend builder".
- For LangGraph, a `BaseChatModel` adapter plus a no-op checkpointer. Our log replaces the checkpointer. Whether
  LangGraph's internal executor is compatible with a deterministic loop is **[unverified]** and needs a spike.

**Episodes and rewards for tier 2.** The foreign loop owns tool execution, so it is a `Program`, not a Task/Agent
pair. Trajectories come from recorder sessions, and rewards are episode-level (`run.reward`). This is the main
reason the `Program` layer is worth having.

### 8.2 Using our tasks and agents from other frameworks

1. **Task as MCP server (session export).** The Control API opens a *task session*:
   - it runs `setup`;
   - the MCP facade serves `tools_for_turn` plus task `@tool` methods (a new facade feature: calls go into the
     task host);
   - `score` and `teardown` run on close.

   Any framework (Claude SDK, OpenAI `MCPServerStreamableHttp`, LangChain) can then act in our environments, and
   training still works through a tier-0 recorder session.
2. **Environments as a sandbox provider.** Every major framework now has a sandbox-provider plug-in point:
   - OpenAI `SandboxClient`;
   - Deep Agents `SandboxBackendProtocol`;
   - Mastra `Workspace` sandbox;
   - AI SDK `experimental_sandbox`;
   - Pydantic AI harness.

   Thin packages over our Environment API (through the Control API's attachment tokens) are a cheap distribution
   channel. They also make our environments the default target for tier-0 users.
3. **Identities over A2A.**
   - Each identity publishes an Agent Card.
   - A2A tasks map to identity conversations, A2A `input-required` maps to `WaitFor` / approvals, and A2A artifacts
     map to `run.emit`.
   - Conversely, an `a2a` binding kind in the tool router consumes remote A2A agents as imported tools, with
     `LongRunning = true` and completion through the inbox.
4. **Long-running imported tools over MCP Tasks.**
   - The `mcp` binding should support the 2026-07-28 Tasks extension: store the task handle as the effect's
     external reference, poll or subscribe, and complete through the inbox.
   - The facade should return task handles for long calls (for example `agent` bindings).
   - `InputRequiredResult` elicitation maps onto approvals and `WaitFor`.
5. **Tracing.** Project the log into OpenTelemetry GenAI spans (`invoke_agent` per run, `chat` per model effect,
   `execute_tool` per tool effect). Mark this as tracking a Development-status convention.
6. **Agent Protocol compatibility** (LangChain's runs / threads / store OpenAPI) is optional. Identities map to
   assistants and threads, and memory maps to the store. Low priority.

---

## 9. Recommended interface changes (concrete)

These are listed in priority order. "Where" names the document the change would land in.

**R1. Deterministic asyncio event loop driver** ([durability.md](../components/harness/durability.md)).
- Replace generator-style driving with a custom `asyncio.AbstractEventLoop` that:
  - runs ready callbacks in a deterministic order;
  - forbids real I/O (`sock_*`, `create_connection`, `run_in_executor`, subprocess) by raising
    `NonDeterminismError`;
  - maps `loop.time()` to `run.now()` and timers to `timer.requested`;
  - resolves `Effect` futures from `Step` inputs.
- `asyncio.gather`, `create_task`, `Lock`, `Queue` and `wait_for` then work, and third-party code runs unchanged.
  Temporal's Python SDK is the reference design.
- Keep "no threads".

**R2. Approvals** (section 7.2):
- `@tool(approval=…)`;
- `RunBinding.approvals`;
- the `Approve` / `Reject` / `Respond` decisions with `remember`;
- events `approval.requested` and `approval.decided`;
- a Control API `DecideApproval(approval_id, decision)`.

**R3. Suspension and episodes** (section 7.1):
- `WaitFor` as a return value of `start` / `respond`;
- a `Task.resume` hook;
- `End(continue_as=…)`;
- delayed rewards on terminal runs through the Control API.

```python
@dataclass(frozen=True)
class WaitFor:
    kind: str                                   # signal kind or mailbox envelope kind
    timeout: timedelta | None = None
    on_timeout: Observation | None = None       # default: End(truncated=True)
    info: Mapping[str, Any] = field(default_factory=dict)
```

**R4. Durable agent identities** (section 5):
- the `AgentIdentitySpecification` resource;
- the `run.identity` context (state, memory, resources);
- mailbox lanes and busy policies;
- identity-owned workspaces and schedules;
- `ag_…` identifiers;
- deployments as the unit identities bind to.

Also:
- **Run keys** become `(identity, conversation)` routing keys: a Slack thread maps to the conversation lane of an
  identity, with signal-with-start semantics via `identities.send`.
- **Triggers** (cron, webhooks) become identity-owned schedules or connector routes that send envelopes.

**R5. Messaging** (section 6): `Address`, `Envelope`, `run.send` / `request` / `reply` / `publish`,
`run.mailbox.next`, `run.map(concurrency, quorum)`. Retire the `run.send(run_id, Message)` signature: it becomes
`run.send(Address.run(run_id), Envelope(...))`.

**R6. Streaming** (section 7.4):
- `client.runs.stream(cursor)` over the log;
- ephemeral deltas keyed by `effect_id` from the recorder streaming variant and envd `Execute` streams;
- `run.emit(kind, payload, to=Address | None)` as event `output.emitted`.

**R7. Memory** (section 7.3): `run.memory` with scopes, versions, compare-and-set, time-to-live and pinned blocks,
plus `ContextHints.pinned`.

**R8. Tools.**
- **Toolsets.** Reusable tool bundles are attachable without subclassing:
  `tools: ClassVar[list[Toolset]] = [ShellTools("workspace")]`, where a `Toolset` contributes specifications, a
  `call` method and optional `before_call` / `after_call` interception (the capability and middleware pattern).
- **`ToolCallContext`.** An optional parameter carrying `call_id`, `turn` and `report_progress(...)`, which emits
  ephemeral progress.
- **New binding kinds** in the router:
  - `external`: a tool executed by the caller or UI, completed through the inbox. This generalizes `human` and
    covers Pydantic `ExternalToolset`, ADK long-running tools and AI SDK client tools.
  - `a2a`.
- **MCP Tasks** support in the `mcp` binding.

**R9. Environments.**

```python
class Environments:
    async def get_or_create(self, name: str, specification: EnvironmentSpecification) -> Environment: ...
        # idempotent by (owner, name); the owner is the run, or the identity when called via run.identity.resources

class Environment:
    async def start_process(self, command: str | list[str], *, cwd: str | None = None,
                            environment_variables: dict[str, str] | None = None) -> Process: ...
    async def expose(self, port: int, *, audience: AccessControlList) -> ExposedEndpoint: ...  # envlet proxy + scoped token

class Process:                                   # pickles as (environment_id, effect_id)
    async def wait(self, *, timeout: float | None = None) -> ExecutionResult: ...
    async def output(self, *, since: int = 0) -> ProcessOutput: ...   # bounded; full logs by reference
    async def signal(self, signal: int) -> None: ...
    async def wait_for_port(self, port: int, *, timeout: float = 60) -> None: ...

@dataclass(frozen=True)
class Template:
    base: str | "Template"                       # a parent template → layered builds, prefix reuse
    files: Mapping[str, bytes | BlobReference | Asset] = field(default_factory=dict)
    run: Sequence[str] = ()
    environment_variables: Mapping[str, str] = field(default_factory=dict)
    start: str | None = None                     # started at build time; running in the golden snapshot
    ready: ReadinessCheck | None = None          # port open / command succeeds / log line
```

- **Hashing.** The template hash is a hash chain over (parent hash, step), so shared prefixes build once. This
  keeps ADR-0014's "one golden snapshot per recipe" and adds E2B-style prefix reuse.
- **Naming.** Consider aligning our names with agent-sandbox's `Template` / `Claim` / `WarmPool`. Our `Claim` and
  `Pool` already match.

**R10. Interoperability packages** (section 8):
- tier-1 `ForeignHarnessTask` helpers and a `compatibility_endpoint()` on model slots;
- tier-2 adapters: OpenAI Agents `PlatformModel` + Program, and a Pydantic AI durability backend;
- a LangGraph spike;
- a task-session MCP export;
- sandbox-provider packages;
- an A2A server and binding;
- an OpenTelemetry projection.

**R11. Application convenience** (Program layer):

```python
support = ChatProgram(
    instructions="You are Acme's support agent.",
    tools=[ShellTools("workspace"), Import("zendesk")],
    memory=True, approvals={"zendesk.refund": True},
    wait_for="message", idle_timeout=timedelta(hours=24),
)
```

This builds a default Task and Agent pair, so application authors write one object, while RL authors keep the
full split.

**R12. Budgets** (proposal 2, made concrete): `Budget(turns, input_tokens, output_tokens, cost, wall_time,
tool_calls)` on the RunBinding and per identity. Exhaustion ends the episode with `TRUNCATED`, which matches RL
semantics. It follows OpenAI `max_turns`, Claude `max_budget_usd` and Pydantic `UsageLimits.cost_limit`.

---

## 10. Open questions

1. **Mailbox lanes.**
   - Per identity (simple; one activation at a time) or per conversation (parallel conversations sharing memory,
     which needs compare-and-set discipline)?
   - Letta allows parallel conversations with shared memory; Restate and Durable Objects serialize per key.
2. **Busy policy default.** `ENQUEUE` (LangGraph default, Letta) or `INTERRUPT` (the user sends a correction
   mid-turn)? `INTERRUPT` cancels the in-flight model effect and needs a defined observation ("the user
   interrupted with …").
3. **Delayed rewards on terminal runs.** Does this break the sample log's immutability contract
   ([ADR-0015](../decisions/0015-rollout-interface.md))? The alternative is to model feedback as a separate
   "feedback episode" linked by `reply_effect_id`.
4. **Topic scaling.** Inbox fan-out vs a shared topic log with subscriber cursors, for swarms of 10⁴–10⁵ agents.
5. **Q10 and tier-2 adapters.** Running LangGraph or the OpenAI SDK in the task host means third-party libraries
   run inside our determinism sandbox. That is acceptable if the host sandbox is the security boundary (it is,
   per durability.md). It still enlarges the attack surface if tenants supply code.
6. **Should approval predicates be allowed to perform effects** (for example reading memory to check a spending
   limit)? The proposal says no, to keep them pure. That may be too restrictive.
7. **Identity versioning.** When `FOLLOW` picks up a new deployment, is identity state migrated by code
   (`on_upgrade(state_version)`) or must state schemas be forward-compatible?
8. **Identity placement.** The home cell is fixed at creation (as for runs). Moving an identity between cells, for
   example to follow a user's region, would need export and import, similar to Letta `.af`.
9. **LangGraph feasibility** under a deterministic loop (thread pools for sync nodes, background executors):
   spike before committing to a tier-2 adapter.

---

## 11. Sources

**Framework documentation (verified 2026-09-27 unless marked)**

- OpenAI Agents SDK:
  - [docs](https://openai.github.io/openai-agents-python/), [running agents](https://openai.github.io/openai-agents-python/running_agents/),
    [tools](https://openai.github.io/openai-agents-python/tools/), [HITL](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/human_in_the_loop.md),
    [sessions](https://openai.github.io/openai-agents-python/sessions/), [handoffs](https://openai.github.io/openai-agents-python/handoffs/),
    [streaming](https://openai.github.io/openai-agents-python/streaming/), [sandbox clients](https://openai.github.io/openai-agents-python/sandbox/clients/),
    [Model interface](https://openai.github.io/openai-agents-python/ref/models/interface/), [ModelSettings](https://openai.github.io/openai-agents-python/ref/model_settings/),
    [PyPI](https://pypi.org/project/openai-agents/)
  - Temporal integration: [README](https://github.com/temporalio/sdk-python/blob/main/temporalio/contrib/openai_agents/README.md),
    [docs](https://docs.temporal.io/develop/python/integrations/openai-agents), [announcement](https://temporal.io/blog/announcing-openai-agents-sdk-integration)
  - [DBOS integration](https://docs.dbos.dev/integrations/openai-agents), [Restate integration](https://restate.dev/blog/durable-orchestration-for-ai-agents-with-restate-and-openai-sdk),
    [Dapr](https://docs.dapr.io/developing-ai/agent-integrations/openai-agents/)
- Claude Agent SDK: [python](https://code.claude.com/docs/en/agent-sdk/python), [typescript](https://code.claude.com/docs/en/agent-sdk/typescript),
  [hosting](https://code.claude.com/docs/en/agent-sdk/hosting), [agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop),
  [user input](https://code.claude.com/docs/en/agent-sdk/user-input), [session storage](https://code.claude.com/docs/en/agent-sdk/session-storage),
  [subagents](https://code.claude.com/docs/en/agent-sdk/subagents), [custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools),
  [PyPI](https://pypi.org/project/claude-agent-sdk/)
- Google ADK: [adk-docs](https://github.com/google/adk-docs/tree/main/docs) (2.0, graphs, runtime/resume, runtime/event-loop,
  tools-custom/confirmation, sessions/state, sessions/memory, a2a, integrations/temporal, integrations/dbos),
  [adk-python](https://github.com/google/adk-python), [PyPI](https://pypi.org/pypi/google-adk/json)
- LangGraph / LangChain: [graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), [functional API](https://docs.langchain.com/oss/python/langgraph/functional-api),
  [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers),
  [Durability](https://reference.langchain.com/python/langgraph/types/Durability), [stores](https://docs.langchain.com/oss/python/langgraph/stores),
  [streaming](https://docs.langchain.com/oss/python/langgraph/streaming), [middleware](https://docs.langchain.com/oss/python/langchain/middleware),
  [HITL](https://docs.langchain.com/oss/python/langchain/human-in-the-loop), [multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent),
  [agent server](https://docs.langchain.com/langsmith/agent-server), [double texting](https://docs.langchain.com/langsmith/double-texting),
  [Agent Protocol](https://github.com/langchain-ai/agent-protocol), [langgraph-swarm](https://github.com/langchain-ai/langgraph-swarm-py),
  [langgraph-supervisor (archived)](https://github.com/langchain-ai/langgraph-supervisor-py), [Deep Agents sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes)
- Pydantic AI: [v2.0.0 release](https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0), [agent](https://pydantic.dev/docs/ai/core-concepts/agent/),
  [capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/), [toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/),
  [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), [durable overview](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/),
  [Temporal](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/), [DBOS](https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/),
  [Prefect](https://pydantic.dev/docs/ai/integrations/durable_execution/prefect/), [harness](https://pydantic.dev/docs/ai/harness/),
  [OpenAI logprobs](https://pydantic.dev/docs/ai/models/openai/)
- CrewAI: [agents](https://docs.crewai.com/en/concepts/agents), [flows](https://docs.crewai.com/en/concepts/flows),
  [memory](https://docs.crewai.com/en/concepts/memory), [checkpointing](https://docs.crewai.com/en/concepts/checkpointing),
  [human feedback](https://docs.crewai.com/en/learn/human-feedback-in-flows)
- Mastra: [agents](https://mastra.ai/docs/agents/overview), [approval](https://mastra.ai/docs/agents/agent-approval),
  [supervisor](https://mastra.ai/docs/agents/supervisor-agents), [suspend/resume](https://mastra.ai/docs/workflows/suspend-and-resume),
  [Inngest](https://mastra.ai/docs/workflows/inngest-workflow), [memory](https://mastra.ai/docs/memory/overview),
  [workspace](https://mastra.ai/docs/workspace/overview)
- Vercel AI SDK: [AI SDK 7](https://vercel.com/changelog/ai-sdk-7), [agents](https://ai-sdk.dev/docs/agents/overview),
  [WorkflowAgent](https://ai-sdk.dev/docs/agents/workflow-agent), [tools](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling),
  [tool UI](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage), [Workflow SDK](https://workflow-sdk.dev/docs/foundations/workflows-and-steps),
  [hooks](https://workflow-sdk.dev/docs/foundations/hooks)
- Letta: [repo](https://github.com/letta-ai/letta), [letta-code](https://github.com/letta-ai/letta-code), [PR #4392](https://github.com/letta-ai/letta-code/pull/4392),
  [terminology](https://docs.letta.com/reference/terminology/index.md), [stateful agents](https://docs.letta.com/concepts/stateful-agents/index.md),
  [sessions](https://docs.letta.com/agent-sdk/sessions/index.md), [MemFS](https://docs.letta.com/concepts/memfs/index.md),
  [shared memory](https://docs.letta.com/concepts/shared-memory/index.md), [memory blocks](https://docs.letta.com/guides/agents/memory-blocks),
  [messages.create](https://docs.letta.com/api/typescript/resources/agents/subresources/messages/methods/create),
  [agent file](https://github.com/letta-ai/agent-file), [research](https://www.letta.com/research/)
- Microsoft: [MAF overview](https://learn.microsoft.com/en-us/agent-framework/overview/), [session](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/session),
  [function tools](https://learn.microsoft.com/en-us/agent-framework/agents/tools/function-tools), [workflows](https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/),
  [checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints), [handoff](https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/handoff),
  [durable agents](https://learn.microsoft.com/en-us/azure/durable-task/sdks/durable-agents-microsoft-agent-framework),
  [Functions hosting](https://learn.microsoft.com/en-us/agent-framework/hosting/azure-functions),
  [BUILD 2026](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-at-build-2026-announce/),
  [AutoGen](https://github.com/microsoft/autogen), [AutoGen identity](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/agent-identity-and-lifecycle.html),
  [AutoGen messaging](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html)
- Cloudflare: [agent class](https://developers.cloudflare.com/agents/runtime/lifecycle/agent-class/index.md), [state](https://developers.cloudflare.com/agents/runtime/lifecycle/state/index.md),
  [routing](https://developers.cloudflare.com/agents/runtime/communication/routing/index.md), [schedule](https://developers.cloudflare.com/agents/runtime/execution/schedule-tasks/index.md),
  [durable execution](https://developers.cloudflare.com/agents/runtime/execution/durable-execution/index.md), [sub-agents](https://developers.cloudflare.com/agents/runtime/execution/sub-agents/index.md),
  [workflows](https://developers.cloudflare.com/agents/runtime/execution/run-workflows/index.md), [HITL](https://developers.cloudflare.com/agents/concepts/agentic-patterns/human-in-the-loop/index.md),
  [chat agents](https://developers.cloudflare.com/agents/communication-channels/chat/chat-agents/index.md), [MCP handler](https://developers.cloudflare.com/agents/model-context-protocol/apis/handler-api/index.md),
  [sandbox tool](https://developers.cloudflare.com/agents/tools/sandbox/index.md), [Think](https://developers.cloudflare.com/agents/harnesses/think/index.md),
  [Project Think](https://blog.cloudflare.com/project-think/)
- Restate: [AI](https://docs.restate.dev/ai), [durable agents](https://docs.restate.dev/ai/patterns/durable-agents), [sessions](https://docs.restate.dev/ai/patterns/sessions),
  [HITL](https://docs.restate.dev/ai/patterns/human-in-the-loop), [multi-agent](https://docs.restate.dev/ai/patterns/multi-agent),
  [streaming](https://docs.restate.dev/ai/patterns/streaming-responses), [Vercel middleware](https://restate.dev/blog/building-durable-agents-with-vercel-and-restate)
- DBOS: [AI quickstart](https://docs.dbos.dev/ai/ai-quickstart), [HITL](https://docs.dbos.dev/ai/hitl), [communication](https://docs.dbos.dev/python/tutorials/workflow-communication),
  [Pydantic AI](https://docs.dbos.dev/integrations/pydantic-ai), [distributing agents](https://docs.dbos.dev/ai/distributing-agents)
- Temporal: [AI](https://docs.temporal.io/ai), [entity workflow](https://docs.temporal.io/design-patterns/entity-workflow),
  [message passing](https://docs.temporal.io/develop/python/message-passing)

**Sandbox SDKs**

- E2B: [SDK source](https://raw.githubusercontent.com/e2b-dev/E2B/main/packages/python-sdk/e2b/sandbox_sync/main.py), [persistence](https://docs.e2b.dev/sandbox/persistence.md),
  [snapshots](https://docs.e2b.dev/sandbox/snapshots.md), [fork](https://docs.e2b.dev/sandbox/fork.md), [templates](https://docs.e2b.dev/template/defining-template.md),
  [caching](https://docs.e2b.dev/template/caching.md), [internet access](https://docs.e2b.dev/network/internet-access.md), [MCP gateway](https://docs.e2b.dev/mcp-gateway.md),
  [billing](https://docs.e2b.dev/billing.md)
- Modal: [Sandbox reference](https://modal.com/docs/reference/modal.Sandbox), [guide](https://modal.com/docs/guide/sandbox), [files](https://modal.com/docs/guide/sandbox-files),
  [snapshots](https://modal.com/docs/guide/sandbox-snapshots), [networking](https://modal.com/docs/guide/sandbox-networking), [images](https://modal.com/docs/guide/images)
- Daytona: [SDK](https://www.daytona.io/docs/en/python-sdk/sync/daytona.md), [sandbox](https://www.daytona.io/docs/en/python-sdk/sync/sandbox.md),
  [lifecycle](https://www.daytona.io/docs/en/sandboxes.md), [snapshots](https://www.daytona.io/docs/en/snapshots.md), [declarative builder](https://www.daytona.io/docs/en/declarative-builder.md)
- kubernetes-sigs/agent-sandbox: [repo](https://github.com/kubernetes-sigs/agent-sandbox), [releases](https://github.com/kubernetes-sigs/agent-sandbox/releases),
  [Sandbox types](https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/api/v1beta1/sandbox_types.go)
- Cloudflare Sandbox: [API](https://developers.cloudflare.com/sandbox/api/)
- Vercel Sandbox: [overview](https://vercel.com/docs/vercel-sandbox), [SDK reference](https://vercel.com/docs/sandbox/sdk-reference)

**Protocols and training**

- MCP: [Tasks 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks), [2026-07-28 release](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/)
- A2A: [repo](https://github.com/a2aproject/A2A), [Linux Foundation](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year)
- Agent Lightning: [repo](https://github.com/microsoft/agent-lightning), [project](https://www.microsoft.com/en-us/research/project/agent-lightning/)
- Polar: [arXiv 2605.24220](https://arxiv.org/abs/2605.24220)
- OpenTelemetry GenAI conventions: [overview (secondary)](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions)
