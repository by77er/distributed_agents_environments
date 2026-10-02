# Layers and deployment profiles

Status: **Accepted** · See [ADR-0016](../decisions/0016-layers-and-profiles.md)

The system is a **core library** plus **optional layers**. Everything a task author, agent author or trainer
touches is defined in the core as Python protocols and runs in one process on one machine. Durability, fleet-scale
execution, multi-tenancy and environment infrastructure are ways to *deploy* those protocols, not part of them.

## Layers

| Layer | Contents | Required? | Docs |
|---|---|---|---|
| **Core** | `Program`, `Task`, `Agent`, `Observation`, tools, canonical content, the model endpoint contract, conversations and messages, the `Runner` protocol, the recorder (renderers, token capture, epochs), `Episode`, rollout jobs, the training loop | **yes** | [core/](../core/README.md), [contracts/](../contracts/README.md) |
| **Inference** | Channels: engines, limits, publishing weights | yes, in some form (a local engine, a fleet, or an API provider) | [inference/](../inference/README.md) |
| **Environments** | Compute environments that tools act on | **no**: many tasks need none (e.g. one-turn reasoning) | [environments/](../environments/README.md) — a separate system, designed later |
| **Durability** | Runs that survive crashes: a durable `Runner` on DBOS, the pump and task host, recovery | no | [durability/](../durability/README.md) |
| **Platform** | Control API, cells, tenancy and trust tiers, tool router service, connectors | no | [platform/](../platform/README.md) |

Rules:

1. **Core interfaces are Python protocols first.** A network form (gRPC, HTTP) is one implementation of a protocol,
   never the definition. Every core protocol has an in-process implementation.
2. **Upper layers depend on the core, never the reverse.** Core documents may mention an optional layer only as
   "one implementation of this protocol".
3. **Task and agent code is layer-agnostic.** The same `Task` runs under the local runner and the durable runner.
   Code meant to run durably must follow the [determinism rules](../core/harness/determinism.md); the local runner
   does not require them.
4. **A task that uses no environment is a complete task.**

## Deployment profiles

| | **Local** | **Cluster** | **Fleet** |
|---|---|---|---|
| Typical use | development, tests, research on one machine (one GPU is enough) | one team's training or product deployment | the design point: hundreds of thousands of runs (N1) |
| Runner | `LocalRunner` (plain asyncio, no persistence), or `DurableRunner` on SQLite | `DurableRunner` on one Postgres | `DurableRunner` on one Postgres per cell |
| Recorder | in-process library | service | service per cell |
| Inference | one engine colocated with the trainer; it sleeps during train steps and loads each new adapter from disk | engine fleet + router + weight update controller | regional fleets |
| Weight transfer | LoRA adapters on disk | distributed transfer (NCCL / RDMA) or checkpoints | distributed transfer |
| Environments | none, local processes or containers, or local microVMs | an environment service | environment fleet |
| Rollout API | in-process implementation | service | service |
| Platform layer | none | optional | yes: cells, tenancy, trust tiers |

### The local profile, concretely

```python
channel = Channel("policy", [VllmEngine(model, gpu_memory_utilization=0.78)], renderer, Limits(sequence=8000))
recorder = Recorder({"policy": channel})
jobs = RolloutJobs(LocalRunner(recorder=recorder), recorder)
trainer = Colocated(LoraTrainer(model, directory), [channel])     # same GPU: the engine sleeps while it steps

job = await jobs.start(program=catalog.program, binding=bind(catalog.program, "policy"), in_flight=5)
ticket = await job.run(catalog.start(row, rng), labels={"group": "0001"}, count=4)
step = await trainer.step(Grpo().batch(await ticket.episodes(), trainer.budget, rng), seed=1)
await job.publish("policy", step.adapter, step.path)              # a LoRA adapter, loaded by name
```

A [profile](../guide/deploying.md) describes the same thing in a file, and `rollout.training.train` is the loop.

With one GPU, training and generation cannot overlap. The weight update protocol still applies, reduced to
sleep → update → wake, and asynchronous RL becomes time-slicing.
