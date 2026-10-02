# Architecture overview

Status: **Proposed** · See [layers and profiles](layers-and-profiles.md)

## The system in one paragraph

A **run** is one episode of a **program** — usually an agent acting in a **task**. The task is the environment in
the reinforcement-learning sense: it defines tools and responds to every model turn; the agent decides what the model
sees and how it acts. Both are Python `async` code in a **core library** that runs in one process. Model samples go
to a **model endpoint** — usually the **recorder**, which captures exact tokens, logprobs, policy versions and routed
experts for training. **Rollout jobs** run task rows in bulk and deliver `Episode`s to a trainer, whose new
weights are published to a **channel** without any task or agent noticing. Optional
layers deploy the same protocols differently: the **durability layer** (a DBOS-backed runner) makes runs survive
crashes and suspend without holding compute; the **platform layer** adds a multi-tenant service at fleet scale;
the **environment system** provides computers for tasks that need them.

## Layers

```
┌──────────────────────────────────────────────── core (a Python library) ─────────────────────────────────────────┐
│ Program · Task · Agent · Observation · tools · conversations · Runner · model endpoint · recorder · rollout jobs   │
│ · Episode · training loop                                              LocalRunner · RolloutJobs · Recorder      │
└───────────────┬───────────────────────┬──────────────────────────────┬─────────────────────────────┬──────────────┘
                │ engine adapter        │ Environments protocol        │ Runner protocol             │ network forms
┌───────────────▼─────────┐ ┌───────────▼──────────────┐ ┌─────────────▼──────────────┐ ┌────────────▼─────────────┐
│ inference               │ │ environments             │ │ durability (optional)      │ │ platform (optional)      │
│ channels, engines,      │ │ separate system,         │ │ DurableRunner on DBOS:     │ │ Control API, cells,      │
│ publishing weights      │ │ designed later           │ │ pump + sandboxed task host │ │ trust tiers, tool router,│
└─────────────────────────┘ └──────────────────────────┘ └────────────────────────────┘ │ connectors               │
                                                                                         └──────────────────────────┘
```

## Components

| Component | Layer | Role | Doc |
|---|---|---|---|
| Harness (loop, Task, Agent, Program) | core | The authoring model and the loop | [core/harness](../core/harness/README.md) |
| `LocalRunner` | core | Runs programs in process | [core/harness](../core/harness/README.md#runner) |
| Recorder | core | Token-exact recording; model endpoint for recorded channels | [core/recorder](../core/recorder/README.md) |
| Rollout jobs | core | Rows in, episodes out, weights published | [core/rollouts](../core/rollouts/README.md) |
| Episode assembly, the job's log | core | Builds `Episode`s | [core/trajectories](../core/trajectories/README.md) |
| Training loop, group algorithm, curriculum, trainer | core | Turns episodes into weights | [core/training](../core/training.md) |
| Channels and engines | inference | Serve and version policies | [inference](../inference/README.md) |
| Environment system | environments | Computers for tasks that need them | [environments](../environments/README.md) (preliminary) |
| `DurableRunner`: pump, task host, recovery controller, reaper | durability | Crash-surviving runs on DBOS | [durability](../durability/README.md) |
| Control API, cells, trust tiers, tool router, connectors | platform | Multi-tenant service at fleet scale | [platform](../platform/README.md) |

## Interfaces between layers

Each is defined once.

| Interface | Between | Defined in |
|---|---|---|
| Model endpoint | runners → recorder / direct adapters | [contracts/model-endpoint](../contracts/model-endpoint.md) |
| `Engine` | channels → engines | [core/recorder/engine-adapter](../core/recorder/engine-adapter.md) |
| `Runner` | callers → local or durable runner | [core/harness](../core/harness/README.md#runner) |
| Conversations | callers and runs → runners | [core/harness/conversations](../core/harness/conversations.md) |
| `Jobs` | trainers → rollout implementation | [core/rollouts](../core/rollouts/README.md) |
| `Episode` | rollout jobs → trainers | [core/trajectories](../core/trajectories/README.md) |
| `ToolBinding` | runners → imported tools (in process or over HTTP) | [guide/tools](../guide/tools.md) |
| `Environments` / `Environment` | task code → environment system | [core/harness/task](../core/harness/task.md#environments-optional) |
| `HarnessHost` | durable pump → sandboxed task host | [durability/task-host](../durability/task-host.md) |
| Effects | runners → executors | [contracts/effects](../contracts/effects.md) |
| Run events | runners → assembler, clients | [contracts/run-events](../contracts/run-events.md) |

## Visibility

What each component may see. A ✗ is a contract violation, not an optimization opportunity.

| | Canonical content | Tokens / logprobs | Policy version | Environment identity | Credentials |
|---|---|---|---|---|---|
| Agent code | ✓ (history) | ✗ | ✗ | ✗ | ✗ |
| Task code | ✓ | ✗ | ✗ | ✓ (environments it creates) | ✗ |
| Runner (local or durable pump) | ✓ | ✗ | ✗ | ✓ | references only |
| Recorder | ✓ | ✓ | ✓ | ✗ | ✗ |
| Inference | via recorder tokens | ✓ | ✓ | ✗ | ✗ |
| Trainer / assembler | ✓ | ✓ | ✓ | metadata only | ✗ |
| Tool bindings | arguments / results | ✗ | ✗ | ✗ | ✓ (they inject them) |
