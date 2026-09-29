# Layers and deployment profiles

Status: **Accepted** · See [ADR-0016](../decisions/0016-layers-and-profiles.md)

The system is a **core library** plus **optional layers**. Everything a task author, agent author or trainer
touches is defined in the core as Python protocols and runs in one process on one machine. Durability, fleet-scale
execution, multi-tenancy and environment infrastructure are ways to *deploy* those protocols, not part of them.

## Layers

| Layer | Contents | Required? | Docs |
|---|---|---|---|
| **Core** | `Program`, `Task`, `Agent`, `Observation`, tools, canonical content, the model endpoint contract, conversations and messages, the `Runner` protocol, the recorder (renderers, token capture, session trees), `Sample`, the rollout API, `WeightsSource` | **yes** | [core/](../core/README.md), [contracts/](../contracts/README.md) |
| **Inference** | Engine adapters, the policy registry, weight transfer and the weight update controller | yes, in some form (a local engine, a fleet, or an API provider) | [inference/](../inference/README.md) |
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
| Inference | one engine colocated with the trainer; it sleeps during train steps and receives weights by CUDA IPC | engine fleet + router + weight update controller | regional fleets |
| Weight transfer | in-process tensors (`update_weights_from_tensor`) | distributed transfer (NCCL / RDMA) or checkpoints | distributed transfer |
| Environments | none, local processes or containers, or local microVMs | an environment service | environment fleet |
| Rollout API | in-process implementation | service | service |
| Platform layer | none | optional | yes: cells, tenancy, trust tiers |

### The local profile, concretely

```python
engine = LocalEngine("Qwen/Qwen3-1.7B", gpu_memory_fraction=0.45)            # vLLM in process, sleep mode enabled
recorder = Recorder(engines={"exp/latest": engine}, mode=RecorderMode.ACTIVE)
jobs = LocalRolloutJobs(runner=LocalRunner(recorder=recorder))

job = jobs.start(program=AgentProgramReference(Wordle, DefaultAgent),
                 binding=RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="exp/latest"))}),
                 trainable_channels={"policy": "exp/latest"}, buffer_samples=256)
for row in rows:
    job.run(row, labels={"group": row["id"]}, count=8, admit_together=True)
async for group in complete_groups(job.samples(), by="group", size=8):
    trainer.step(group)                                   # same process, same GPU
    await job.publish("exp/latest", WeightsSource(tensors=trainer.merged_weights()))   # LoRA merged into the base
```

With one GPU, training and generation cannot overlap. The weight update protocol still applies, reduced to
sleep → update → wake, and asynchronous RL becomes time-slicing.
