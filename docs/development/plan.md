# Development plan

Status: **Proposed** · See [ADR-0023](../decisions/0023-development-baseline.md)

Work is ordered so that each milestone produces something that runs on one machine and validates the interfaces
the next milestone builds on. Later milestones start only when a consumer needs them (P14).

**Order of work: M0 → P1 → M2 → P2 → M1 → M3** ([ADR-0024](../decisions/0024-product-before-rl.md),
[ADR-0025](../decisions/0025-agent-sessions.md)). Products on a third-party model validate the loop, conversations,
tools, environments and the durable runner before RL work resumes.

## Milestones

### M0 — Core loop, no GPU

**Scope**
- Contract types as frozen pydantic models: canonical content, `ToolSpecification`, `ToolResult`, run events,
  effects, identifiers and digests ([contracts](../contracts/README.md)).
- The harness: `Program`, `AgentProgram`, `Task`, `Agent`, `Observation`, `End`, `WaitFor`, `RunContext`,
  `@tool`, the normative loop ([harness](../core/harness/README.md)).
- Conversations in memory: `Address`, `Envelope`, priorities, delivery modes (queue / steer / interrupt).
- `LocalRunner`; `run.spawn`, `run.send`, `run.emit`, `run.sleep`.
- Model endpoint: the protocol, `ContextDelta` computation, and direct adapters for an OpenAI-compatible server and
  the Anthropic Messages API.
- Imported tools through an in-process `ToolBinding` (MCP client, HTTP).
- Examples: single-turn arithmetic, Wordle, a tool-using task, a conversation with steering and interruption.

**Exit criteria**
- Examples run end to end against an API model and against a scripted fake model endpoint.
- Loop semantics covered by tests: validation rules, `WaitFor` / `resume` / timeout, `STEER` merge, `INTERRUPT`
  during a sample and during tools, `teardown` on every path, reward binding.

### P1 — Project assistant

A long-lived conversational agent about a local code repository, on a third-party model
([ADR-0024](../decisions/0024-product-before-rl.md)).

Status: **done on the `LocalRunner`**: the adapter, the assistant, the HTTP API and the evaluation harness, with a
baseline of 21/21 scenario runs passing ([project assistant](../products/project-assistant.md#evaluations)). A
follow-up firing is verified in seconds; surviving a pause of days needs the durable runner and is verified in M2.

**Scope**
- The Responses API adapter on the local Codex login, with token refresh.
- The `ProjectAssistant` task: read-only repository tools (search, read, list, git history), notes as an imported
  tool, follow-ups through `WaitFor` timeouts.
- A small HTTP API: send a message to a conversation, stream its events, read its transcript, cancel.
- The evaluation harness: scenarios with scripted users; task-success checks; judged quality with a rubric; cost and
  latency from run events.

**Exit criteria**
- The evaluation suite runs end to end against the model through the HTTP API, and baseline numbers are recorded.
- A conversation survives a pause of days through `WaitFor`, and a scheduled follow-up fires.

**Continues in M2**: the product moves onto `DurableRunner` unchanged, and the durability-under-faults evaluation
kills processes at random points (no lost messages, no duplicated model calls or note writes).

### P2 — Agent sessions

Independent agent sessions with isolated environments that create and message each other, share a board, and are
managed from a CLI ([ADR-0025](../decisions/0025-agent-sessions.md)).

Status: **working**: environments, attempt markers, coordination tools and the product are built; a live fan-out
over the board succeeded, and the fault evaluation passes across 13 SIGKILLs
([agent sessions](../products/agent-sessions.md#durability-under-faults)).

**Scope**
- The environment protocol in the core and `run.environments`; a namespace backend over an Alpine image.
- Attempt markers in the `DurableRunner` for side effects that cannot be deduplicated.
- Coordination tool sets: sessions (list, create, message) and a board (channels, notes, tasks, claims,
  subscriptions).
- The agent sessions product: the session task and agent, an HTTP API and an `agents` CLI.

**Exit criteria**
- A coordinator session fans three tasks out over the board; workers in their own environments claim, complete and
  report them; the whole scenario passes, also with the server killed at random points.

### M1 — Recorder and local RL on one GPU

Status: **working**: the recorder, channels over vLLM, rollout jobs, the training loop and a LoRA trainer are built,
and train the [Minecraft swarm](../products/minecraft-swarm.md) on one 16 GB GPU.

**Scope**
- The recorder with `renderers` for Qwen3.5: token-exact turns, epochs, deduplication by effect id, a thinking
  budget ([recorder](../core/recorder/README.md)).
- Channels and the vLLM engine: tokens in, logprobs of the sampled distribution out, sleep and wake, adapters
  loaded by name ([inference](../inference/README.md)).
- Rollout jobs and episodes, in process and over HTTP ([rollouts](../core/rollouts/README.md)).
- The training loop, group-relative advantages, the curriculum, and a LoRA trainer for 4-bit checkpoints
  ([training](../core/training.md)).

### M2 — Durable runner

Status: `DurableRunner` in trusted mode is built ([status](../durability/README.md#implementation-status)). The
project assistant runs on it unchanged, and the durability-under-faults evaluation passes: across 8 SIGKILLs, no
lost messages, no duplicated replies or notes, one repeated model call per kill
([results](../products/project-assistant.md#durability-under-faults)). Next: the sandboxed task host, suspension
without compute, generations, Postgres.

**Scope**
- `DurableRunner` on DBOS, SQLite first then Postgres: the pump, effects as steps with identities and digests,
  attempt markers, generations, conversation activations on a partitioned queue, cooperative cancellation, the
  reaper, the recovery controller ([durability](../durability/README.md)).
- The task host: separate process, deterministic asyncio event loop, `HarnessHost`
  ([task host](../durability/task-host.md)).

**Exit criteria**
- The M0 examples and the P1 evaluation suite pass unchanged under `DurableRunner`, including with processes killed
  at random (the durability-under-faults evaluation).
- Phase-1 durability spikes pass ([open questions](../architecture/open-questions.md#validation-spikes)).

### M3 — RL at scale

**Scope**
- Routing replay (a small Qwen3-MoE configuration locally), `score_tokens`, gang admission, run cancellation,
  several trainable channels, staged publish, swarm rewards ([ADR-0022](../decisions/0022-rl-interface-extensions.md)).
- The SGLang engine adapter; the rollout service form; the first production trainer adapter (chosen here).

**Exit criteria**: the phase-3 RL spikes pass on a multi-GPU node.

### Later, when needed

The environment system; the platform layer (cells, trust tiers, Control API, connectors); interoperability adapters.

## First tasks (M0)

1. Repository scaffold: `pyproject.toml`, `src/rollout/`, tooling, a smoke test. **Done with this plan.**
2. Contract types and digests, with round-trip and canonical-JSON tests. **Done** (`rollout.core.contracts`).
3. `Task`, `Agent`, `Observation`, `WaitFor` and the loop, driven by a scripted fake model endpoint. **Done**
   (`rollout.core.harness`, `rollout.core.local.LocalRunContext`, `rollout.core.testing`).
4. `@tool` → `ToolSpecification` (pydantic JSON Schema), tool execution and `ToolResult` normalization. **Done**.
5. `LocalRunner` with conversations and delivery modes. **Done** (`rollout.core.local.LocalRunner`).
6. The Responses API adapter on the Codex login (P1 needs it first). **Done** (`rollout.adapters.responses`). The
   Anthropic adapter and `ContextDelta` computation follow when a consumer needs them.
7. Examples.
