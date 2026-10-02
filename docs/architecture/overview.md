# Architecture overview

Status: **Working** (2026-10-02)

## The system in one paragraph

A **run** is one episode of a **program**: usually an agent acting in a **task**. The task is the environment in the
reinforcement-learning sense: it defines tools and responds to every model turn; the agent decides what the model
sees and how it acts. Both are Python `async` code. Model samples go to a **model endpoint**: an adapter to a
third-party model, or the **recorder**, which samples from a trainable **channel** and keeps the exact tokens,
logprobs and weights versions. **Rollout jobs** run a catalog's rows in bulk and deliver **episodes** to a training
loop, whose trainer's new weights are published to the channel without any task or agent noticing. A **profile**
says which engines, trainer, runner and tool sets stand behind all of it.

## Layers

| Layer | Package | What it holds |
|---|---|---|
| Harness | `rollout.core` | Programs, tasks, agents, tools, conversations, the loop, the `Runner` protocol and `LocalRunner`, contract types, hooks, memory |
| Durability | `rollout_durable` | `DurableRunner`: runs that survive their process, on DBOS |
| Environments | `rollout_computers` | Computers for tasks: backends and the tools that act on them |
| Inference | `rollout_train.inference`, `rollout_train.recorder` | Channels and engines; token-exact recording; the endpoint for harnesses |
| Training | `rollout_train.rollouts`, `rollout_train` | Rollout jobs and episodes; the loop, the group algorithm, the curriculum, the trainer |
| Deployment | `rollout_train.profile`, `rollout_train.cli`, `rollout_train.monitor` | A deployment described and opened; the `rollout` command; the live page |

Code above a protocol never learns which implementation it holds. Task and agent code is the same under either
runner; a training loop is the same with everything in one process or with runs, engines and trainer elsewhere.

## Protocols and their implementations

| Protocol | Between | Implementations | Defined in |
|---|---|---|---|
| `Runner` | callers → runs | `LocalRunner`, `DurableRunner` | [harness](../core/harness/README.md#runner) |
| `ModelEndpoint` | runners → models | `Recorder` endpoints, the Responses API adapter, `ScriptedModelEndpoint` | [model endpoint](../contracts/model-endpoint.md) |
| `Engine` | channels → replicas | `VllmEngine` | [engines](../core/recorder/engine-adapter.md) |
| `ToolSet` | runs → imported tools | a tool set in process, `RemoteToolSet` over HTTP | [tools](../guide/tools.md) |
| `EnvironmentService` | runs → computers | `NamespaceEnvironments`, `LocalEnvironments` | [environments](../environments/README.md) |
| `Jobs`, `Job`, `Ticket` | training → runs | `RolloutJobs`, `RolloutClient` over HTTP | [rollouts](../core/rollouts/README.md) |
| `Catalog` | training → an environment's rows | one per environment | [three ways in](../guide/perspectives.md#building-an-environment) |
| `Trainer` | training → weights | `LoraTrainer`, `Colocated` | [training](../core/training.md#the-trainer) |
| `RunHooks`, `JobHooks` | runners and jobs → observers | `RunFeed` | [hooks](../core/harness/hooks.md), [monitor](../core/monitor.md) |

Types that cross these boundaries are defined once, in [contracts](../contracts/README.md).

## Visibility

What each part sees. A ✗ is a boundary the code keeps, not an optimization left undone.

| | Canonical content | Tokens and logprobs | Weights version | Engines, trainer, machines |
|---|---|---|---|---|
| Task and agent code | ✓ | ✗ | ✗ | ✗ |
| Runner | ✓ | ✗ | ✗ | ✗ |
| Recorder and channels | ✓ | ✓ | ✓ | engines only |
| Rollout jobs | labels and results | ✓ (in episodes) | ✓ | ✗ |
| Training loop and trainer | labels and results | ✓ | ✓ | ✗ |
| Profile | ✗ | ✗ | ✗ | ✓ |

## A turn

The agent samples a reply; the task responds with an observation.

```
agent.act ──▶ Model.sample ──▶ model endpoint ──▶ (recorder ──▶ channel ──▶ engine: tokens in; tokens, logprobs out)
          ◀── canonical reply + usage
task.respond(reply) ──▶ run_tools ──▶ @tool method ──▶ (an imported tool set, an environment, a blob store)
          ◀── Observation(tool results) ──▶ steering messages merged ──▶ next turn
```

Everything that leaves task or agent code on the way is an **effect** with an identity:
`{run_id}:{generation}:{ordinal}` and a digest of its arguments ([effects](../contracts/effects.md)).

| | `LocalRunner` | `DurableRunner` |
|---|---|---|
| Where code runs | the caller's process | the runner's process, inside a DBOS workflow |
| An effect | a call, recorded as run events | a durable step: its first recorded result is final |
| After a crash | runs in flight are lost; a rollout job reports them as failed episodes | the program is run again and recorded effects return their results; what was in flight is performed again under the same identity |
| A side effect that cannot be deduplicated | performed once | guarded by an attempt marker: a possible duplicate is reported as `OUTCOME_UNKNOWN` to the model, never silently retried |
| Messages | delivered in process | delivered once by `message_id`, to whichever runner holds the conversation |

A tool's `retry_class` says what re-execution may do: `PURE` and `IDEMPOTENT` calls are performed again;
`SIDE_EFFECTING` and `UNKNOWN` calls are performed again only against a tool set that deduplicates by effect
identity, and are guarded otherwise. Cancellation is cooperative: it is delivered at the next turn boundary, and
`teardown` runs. The durable runner is described in [durability](../durability/README.md).
