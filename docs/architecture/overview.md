# Architecture overview

Status: **Proposed**

## The system in one paragraph

A **run** is one episode of an **agent** acting in a **task**. The task is the environment in the reinforcement
learning sense — it owns compute environments, defines the tools, and responds to every model turn; the agent
decides what the model sees and how it acts. Both are Python `async` code executed by a **task host** under
deterministic replay. Runs are hosted by stateless **runtime** workers that own partitions of runs via leases,
persist every step to a per-cell **run store**, and perform the effects task code requests: model samples go to a
**model endpoint** (optionally through the **recorder**, which captures exact tokens, logprobs and policy
versions), environment operations go to the **Environment Manager** and to the in-guest daemon **envd** via the
host's **envlet**, and imported external tools go to the **tool router**. For reinforcement learning, a **rollout
controller** runs the task rows a trainer submits, under a bounded buffer; a **trajectory assembler** joins
recorder data with the rewards and endings in the run log into `Sample`s in one durable log per job, which the
**trainer** reads with a cursor and groups however its algorithm needs. New weights flow back through the
**policy registry** without any task or agent noticing.

## Component map

```
                        ┌─────────────────────────── Global ────────────────────────────┐
  clients ─────────────▶│ Control API (edge) · Global Router · Policy Registry          │
                        │ Rollout Controller · Trajectory Assembler · sample logs       │
                        └──────┬───────────────────────────────────────────────▲────────┘
                               │ route by cell (cell id embedded in run_id)    │ weights
┌──────────────────────────────▼──── Cell ─────────────────────────────────────┼─────────┐
│                                                                              │         │
│  Control API (cell) ──▶ Run Store (Postgres) ◀──▶ Runtime workers ◀──▶ Task hosts       │
│                                                    │ effects         (task + agent code)│
│             model.request ┌────────────────────────┼─────────────────┐ tool.request      │
│                           ▼                        │                 ▼                   │
│                  Recorder (optional)    environment.lifecycle   Tool Router              │
│                           │             environment.call        ┌────┼─────┬──────┐      │
│                           │                        │           mcp  http  agent  human   │
│                           │                        ▼                                     │
│                           │        Environment Manager ──▶ envlet (host)                 │
│                           │                                  │ vsock                     │
│                           │                                  ▼                           │
│                           │                              envd (guest)                    │
└───────────────────────────┼─────────────────────────────────────────────────────────────┘
                            ▼
         Inference (region): engines · router · Weight Update Controller
```

## Components

| Component | Role | Scope | Durable state | Build / buy | Doc |
|---|---|---|---|---|---|
| Control API | External API for runs, signals, environments, jobs | edge + per cell | none (delegates) | build | [control-api](../components/control-api/README.md) |
| Run Store | Event log, leases, inbox, timers | per cell | **yes** (Postgres) | build on Postgres | [run-store](../components/run-store/README.md) |
| Runtime | Durable actor host: leases, mailboxes, dispatch | per cell | none (cache) | build | [runtime](../components/runtime/README.md) |
| Task host (harness) | Runs the loop, task and agent code under replay | per cell (co-located with workers) | none (derived from log) | build SDK; tasks and agents pluggable | [harness](../components/harness/README.md) |
| Tool Router | Executes imported external tools | per cell | none | build | [tool-router](../components/tool-router/README.md) |
| Environment Manager | Environment lifecycle, pools, placement | per cell | **yes** (environment registry) | build | [environments](../components/environments/README.md) |
| envlet | Node agent: runs microVMs, enforces security | per host | snapshots (local cache) | **fork E2B orchestrator** | [placement](../components/environments/placement.md) |
| envd | In-guest daemon serving the Environment API | per environment | idempotency cache | fork E2B envd | [env-api](../components/environments/env-api.md) |
| Recorder | Active tokenization, session trees, versions | per cell | **yes** (object store) | build on `renderers` | [recorder](../components/recorder/README.md) |
| Inference | Engines, cache-aware router | region | weights | **buy** (SGLang/vLLM, sgl-model-gateway/llm-d) | [inference](../components/inference/README.md) |
| Policy Registry + WUC | Versions, channels, weight rollout | global + region | **yes** | build (thin) | [inference](../components/inference/README.md) |
| Rollout Controller | Jobs: rows in, runs created under a bounded buffer, weights published | global | **yes** (job state) | build | [rollouts](../components/rollouts/README.md) |
| Trajectory Assembler | Joins session paths + run rewards → `Sample`s in a per-job log | global | **yes** (object store) | build | [trajectories](../components/trajectories/README.md) |

## Boundary map

Every arrow in the system is one of these. Each has exactly one defining contract.

| # | Consumer → Provider | Contract | Proposed transport | Guarantee |
|---|---|---|---|---|
| B1 | Client → Control API | [control-api](../components/control-api/README.md) | HTTPS/gRPC | idempotent creates |
| B2 | Control API → Run Store | [run-store](../components/run-store/README.md) | SQL | atomic, per-cell |
| B3 | Runtime ↔ Run Store | [run-store](../components/run-store/README.md) | SQL | fenced, linearizable per run |
| B4 | Runtime ↔ Task host | [harness](../components/harness/README.md#interface-harnesshost-b4) | local gRPC (Go ↔ Python) | effects only; deterministic replay |
| B5 | Runtime → Model endpoint (Recorder or direct adapter) | [model-endpoint](../contracts/model-endpoint.md) | HTTP/gRPC | idempotent on `effect_id` |
| B6 | Recorder → Inference engines | [engine-adapter](../components/recorder/engine-adapter.md) | engine-native HTTP/gRPC | single weights version per engine response |
| B7 | Runtime → Tool Router (imported tools) | [tool-router](../components/tool-router/README.md) | gRPC | idempotency key forwarded |
| B8 | Tool Router → Bindings | [tool-router](../components/tool-router/README.md) (Binding SPI) | per binding | per `retry_class` |
| B9 | Runtime (environment client) → envlet → envd | [env-api](../components/environments/env-api.md) | gRPC (mTLS) → vsock | three-state idempotency |
| B10 | Runtime → Environment Manager | [environments](../components/environments/README.md) | gRPC | idempotent lifecycle operations |
| B11 | Environment Manager → Driver / envlet | [driver](../components/environments/driver.md) | gRPC | capabilities declared, enforced or rejected |
| B12 | Rollout Controller → Control API | [control-api](../components/control-api/README.md) | gRPC | idempotent bulk create |
| B13 | Assembler ← run events (rewards, observations, endings) | [trajectories](../components/trajectories/README.md) | Control API event stream | complete once the run is terminal |
| B14 | Assembler ← Recorder session export | [session-tree](../components/recorder/session-tree.md) | object storage + manifest | complete-or-flagged sessions |
| B15 | Sample log → caller (`RolloutJobs.Samples`) | [trajectories](../components/trajectories/README.md), [rollouts](../components/rollouts/README.md) | object storage, read by cursor | at-least-once, deduplicated by `sample_id` |
| B16 | Caller → Rollout Controller → Policy Registry + WUC (`Publish`) | [rollouts](../components/rollouts/README.md), [inference](../components/inference/README.md) | gRPC | versions monotonic per lineage |
| B17 | Policy Registry → WUC → engines; WUC → Recorder | [inference](../components/inference/README.md), [engine-adapter](../components/recorder/engine-adapter.md) | gRPC | abort-before-update |
| B18 | Unmanaged harness → Recorder session API + Tool Router MCP facade | [session-api](../components/recorder/session-api.md), [tool-router](../components/tool-router/README.md) | OpenAI/Anthropic-compatible HTTP, MCP | trainable, not durable |

## Visibility matrix

What each component is allowed to see. A ✗ is a contract violation, not an optimization opportunity.

| | Canonical content | Tokens / logprobs | Policy version | Environment identity | Credentials | Guest data |
|---|---|---|---|---|---|---|
| Agent code | ✓ (history) | ✗ | ✗ | ✗ | ✗ | only through observations |
| Task code | ✓ | ✗ | ✗ | ✓ (environments it owns) | ✗ | ✓ (untrusted) |
| Runtime | ✓ | ✗ | ✗ | ✓ | attachment tokens only | as effect results |
| Tool Router | arguments / results | ✗ | ✗ | ✗ | via broker references only | ✗ |
| Recorder | ✓ | ✓ | ✓ | ✗ | ✗ | as context |
| Inference | via recorder tokens | ✓ | ✓ | ✗ | ✗ | as context |
| envlet / egress proxy | ✗ | ✗ | ✗ | ✓ | ✓ (injects) | ✓ |
| envd (guest) | ✗ | ✗ | ✗ | own only | ✗ | ✓ |
| Trainer / Assembler | ✓ | ✓ | ✓ | metadata only | ✗ | as context |

## Planes

- **Control plane**: Control API, Global Router, Environment Manager, Policy Registry, Rollout Controller.
- **Execution plane**: Runtime, Task hosts, Tool Router, envlet, envd, Recorder (request path).
- **Data plane (reinforcement learning)**: Recorder (tree storage), Trajectory Assembler, sample logs.

The execution plane MUST keep running when the reinforcement-learning data plane is absent or degraded (P10).
