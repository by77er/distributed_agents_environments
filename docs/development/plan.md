# Development plan

Status: **Proposed** · See [ADR-0023](../decisions/0023-development-baseline.md)

Work is ordered so that each milestone produces something that runs on one machine and validates the interfaces
the next milestone builds on. Later milestones start only when a consumer needs them (P14).

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

### M1 — Recorder and local RL on one GPU

**Scope**
- The recorder in active mode with `renderers` for Qwen3 / Qwen3.5: session trees, incremental rendering,
  deduplication by `(effect_id, arguments_digest)`, turn spans.
- The vLLM engine adapter: tokens in, processed logprobs, abort with partials, sleep / wake, in-process weight
  updates ([engine adapter](../core/recorder/engine-adapter.md)).
- The in-process weight update controller and policy registry.
- `LocalRolloutJobs`, the trajectory assembler, the local sample log.
- The minimal reference GRPO trainer.

**Exit criteria**
- GRPO on Qwen3-1.7B improves arithmetic and Wordle reward on the development GPU.
- Recorded behavior logprobs match the trainer's recomputation at the same weights: mean absolute difference
  within tolerance (bf16 decode and prefill disagree per token by up to ~0.15; see
  [ADR-0023](../decisions/0023-development-baseline.md#local-engine-spike)).
- A generation interrupted by a weight update is recorded as two spans with correct versions.

### M2 — Durable runner

**Scope**
- `DurableRunner` on DBOS, SQLite first then Postgres: the pump, effects as steps with identities and digests,
  attempt markers, generations, conversation activations on a partitioned queue, cooperative cancellation, the
  reaper, the recovery controller ([durability](../durability/README.md)).
- The task host: separate process, deterministic asyncio event loop, `HarnessHost`
  ([task host](../durability/task-host.md)).

**Exit criteria**
- The M0 and M1 examples pass unchanged under `DurableRunner`, including with processes killed at random.
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
5. `LocalRunner` with conversations and delivery modes.
6. Direct adapters and `ContextDelta`.
7. Examples.
