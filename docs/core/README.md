# Core

Status: **Proposed** · See [layers and profiles](../architecture/layers-and-profiles.md)

The core is a Python library. It defines everything task authors, agent authors and trainers use, and it runs in
one process. Optional layers ([durability](../durability/README.md), [platform](../platform/README.md)) provide
other implementations of the same protocols.

| Document | Defines |
|---|---|
| [harness/README.md](harness/README.md) | The loop, `Program`, `RunSpecification`, deployments, the `Runner` protocol |
| [harness/task.md](harness/task.md) | `Task`, `Observation`, `WaitFor`, tools, `RunContext` |
| [harness/agent.md](harness/agent.md) | `Agent`: context selection and acting |
| [harness/conversations.md](harness/conversations.md) | Conversations, messages, priorities and delivery modes |
| [harness/determinism.md](harness/determinism.md) | The rules code must follow to run under a durable runner |
| [recorder/](recorder/README.md) | Token-exact recording for training: renderers, session trees, engine adapter |
| [rollouts/](rollouts/README.md) | The rollout API: rows in, samples out, weights published |
| [trajectories/](trajectories/README.md) | `Sample` and its assembly |

Shared types used across these documents are defined once in [contracts/](../contracts/README.md).

## Protocols and their implementations

| Protocol | In-process implementation (local profile) | Other implementations |
|---|---|---|
| `Runner` | `LocalRunner` | `DurableRunner` ([durability](../durability/README.md)) |
| `ModelEndpoint` | `Recorder` (in-process), `DirectAdapter` | recorder service |
| `EngineAdapter` | `LocalEngine` (SGLang / vLLM in-process) | engine fleet behind a router ([inference](../inference/README.md)) |
| `RolloutJobs` | `LocalRolloutJobs` | rollout service |
| `ToolBinding` (imported tools) | in-process MCP / HTTP clients | tool router service ([platform](../platform/tool-router/README.md)) |
| `Environments` | none, or a local driver | environment service ([environments](../environments/README.md)) |
