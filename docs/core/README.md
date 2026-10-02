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
| [harness/hooks.md](harness/hooks.md) | `RunHooks`: watch every run event and model sample as it happens |
| [harness/memory.md](harness/memory.md) | `Memory`: a context that fits any model, for long episodes |
| [monitor.md](monitor.md) | A live web page over a job and its runs: each model's context, reasoning, actions and results |
| [recorder/](recorder/README.md) | Token-exact recording for training: renderers, epochs, engines, the endpoint for harnesses |
| [rollouts/](rollouts/README.md) | Rollout jobs: rows in, episodes out, weights published |
| [trajectories/](trajectories/README.md) | `Episode` and its assembly |
| [training.md](training.md) | The training loop, the group algorithm, the curriculum, the trainer |

Shared types used across these documents are defined once in [contracts/](../contracts/README.md).

## Protocols and their implementations

| Protocol | In-process implementation (local profile) | Other implementations |
|---|---|---|
| `Runner` | `LocalRunner` | `DurableRunner` ([durability](../durability/README.md)) |
| `ModelEndpoint` | `Recorder` (in process), the Responses API adapter | the recorder over HTTP, for harnesses ([session API](recorder/session-api.md)) |
| `Engine` | `VllmEngine` | any replica behind a `Channel` ([inference](../inference/README.md)) |
| `Jobs` | `RolloutJobs` | `RolloutClient` over the rollout service |
| `Trainer` | `LoraTrainer`, `Colocated` | |
| `ToolSet` (imported tools) | a tool set in process | `RemoteToolSet` over HTTP (`rollout tools`) |
| `Environments` | none, or a local driver | environment service ([environments](../environments/README.md)) |
