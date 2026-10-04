# Architecture overview

## The system in one paragraph

A **run** is one episode of a **program**: usually an agent acting in a **task**. The task is the environment in the
reinforcement-learning sense: it defines tools and responds to every model turn; the agent decides what the model
sees and how it acts. Both are Python `async` code. Model samples go to a **model endpoint**: an adapter to a
third-party model, or the **recorder**, which samples from a trainable **channel** and keeps the exact tokens,
logprobs and weights versions. A training loop asks for a row's start to be played as a **group** of **episodes**,
in the **ledger**; **episode runners**, on any machine that reaches the ledger, claim the episodes, play them and
record them there. The loop takes a **step** over several groups at a time; each step's new weights are published to
the channel without any task or agent noticing. A **profile** says which
engines, trainer, runner and tool sets stand behind all of it.

## Layers

The repository is a workspace of packages in four layers. Each package's directory, import name and page are in the
[documentation index](../README.md#packages).

| Layer | Packages | What it holds |
|---|---|---|
| Libraries | `rollout` | What environments are written against: programs, tasks, agents, tools, conversations, the loop, the `Runner` protocol and `LocalRunner`, contract types, hooks, memory, the catalog |
| | `rollout-train` | Reinforcement learning on `rollout`: episode runners and episodes; the loop, the group algorithm, the curriculum and the `Trainer` protocol; channels and the `Engine` protocol; the recorder and the `Renderer` protocol; the graph of versions, the ledger and the registry; resharding; heartbeats, launches and the launcher; the profile, the `rollout` command and the monitor |
| Implementations | `rollout-durable`, `rollout-vllm`, `rollout-lora`, `rollout-qwen`, `rollout-gemma`, `rollout-computers`, `rollout-openai`, `rollout-s3` | One implementation each of an interface a library defines |
| Products | `project-assistant`, `agent-sessions` | Applications built on the libraries and implementations |
| Environments | `minecraft-team` | An environment to train on |

What may depend on what is checked by `tests/test_layers.py`:

- a package imports only the workspace packages its project file declares;
- the libraries require no implementation, and `rollout` requires no other workspace package;
- an environment imports `rollout` and nothing above it.

Code above a protocol never learns which implementation it holds. Task and agent code is the same under either
runner; a training loop is the same with everything in one process or with runs, engines and trainer elsewhere. A
profile names engines, renderers, trainers and tool sets as `module:name`, so `rollout-train` imports none of them.

## Protocols and their implementations

| Protocol | Defined in | Between | Implementations |
|---|---|---|---|
| [`Runner`](../guide/reference.md#runner) | `rollout.harness` | callers → runs | `LocalRunner` (`rollout.local`), `DurableRunner` ([`rollout_durable`](../implementations/rollout-durable/README.md)) |
| [`ModelEndpoint`](../libraries/rollout/contracts/model-endpoint.md) | `rollout.contracts` | runners → models | the [recorder](../libraries/rollout-train/recorder.md)'s endpoints, `ResponsesEndpoint` ([`rollout_openai`](../guide/models.md)), `ScriptedModelEndpoint` ([`rollout.testing`](../guide/testing.md)) |
| [`Engine`](../guide/reference.md#engine) | `rollout_train.inference` | channels → replicas | `VllmEngine` ([`rollout_vllm`](../implementations/rollout-vllm.md)), `ScriptedEngine` (`rollout_train.testing`) |
| [`Renderer`](../guide/reference.md#renderer) | `rollout_train.recorder` | the recorder → a model family's tokens | `qwen35`, `qwen3` ([`rollout_qwen`](../implementations/rollout-qwen.md)), `gemma4` ([`rollout_gemma`](../implementations/rollout-gemma.md)), `PlainRenderer` (`rollout_train.testing`) |
| [`Trainer`](../guide/reference.md#trainer) | `rollout_train` | training → weights | `LoraTrainer` ([`rollout_lora`](../implementations/rollout-lora.md)); `Colocated` wraps one that shares the engines' accelerator |
| [`Algorithm`](../guide/reference.md#algorithm) | `rollout_train` | the training loop → what to train on | `Grpo` ([training](../libraries/rollout-train/training.md#the-algorithm-grpo)) |
| [`Ledger`](../guide/reference.md#ledger) | `rollout_train` | training and versions → append-only tables | `FileLedger`, `DatabaseLedger` ([the ledger](../libraries/rollout-train/versions.md#the-ledger)) |
| [`ToolSet`](../guide/reference.md#toolset) | `rollout.harness` | runs → imported tools | a tool set in process, `RemoteToolSet` over HTTP ([tools](../guide/tools.md#imported-tools)) |
| [`EnvironmentService`](../guide/reference.md#environmentservice) | `rollout.harness` | runs → computers | `NamespaceEnvironments`, `LocalEnvironments` ([`rollout_computers`](../implementations/rollout-computers.md)) |
| [`Blobs`](../guide/reference.md#blobs) | `rollout.harness` | runs → stored bytes | `FileBlobStore` (`rollout.harness`), `S3BlobStore` ([`rollout_s3`](../guide/content.md#media-and-blobs)) |
| [the ledger's `plans`, `groups`, `claims` and `episodes`](../libraries/rollout-train/rollouts.md) | `rollout_train.rollouts` | training → runs | `EpisodeRunner`, on any machine that reaches the ledger and the blob store |
| [`Catalog`](../guide/reference.md#catalog) | `rollout.catalog` | training → an environment's rows | one per environment ([three ways in](../guide/perspectives.md#building-an-environment)) |
| [`RunHooks`](../libraries/rollout/hooks.md), `Hooks` | `rollout.harness`, `rollout_train.rollouts` | runners and runs → observers | `RunFeed` ([monitor](../libraries/rollout-train/monitor.md)) |
| `Presence`, `Launches` | `rollout_train.presence`, `rollout_train.launches` | runners and launchers → whoever watches or asks for runs | `FilePresence`, `DatabasePresence`; `FileLaunches`, `DatabaseLaunches` ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats), [launchers](../guide/deploying.md#launchers)) |
| A layout (`module:name`) | `rollout_train.resharding` | versions → the files their engines load | `verbatim`; run in the run's process or as a Ray task (`on_ray`) ([resharding](../libraries/rollout-train/versions.md#resharding)) |

Types that cross these boundaries are defined once, in [contracts](../libraries/rollout/contracts/README.md).

## Visibility

What each part sees. A ✗ is a boundary the code keeps, not an optimization left undone.

| | Canonical content | Tokens and logprobs | Weights version | Engines, trainer, machines |
|---|---|---|---|---|
| Task and agent code | ✓ | ✗ | ✗ | ✗ |
| Runner | ✓ | ✗ | ✗ | ✗ |
| Recorder and channels | ✓ | ✓ | ✓ | engines only |
| Episode runners | labels and results | ✓ (in episodes) | ✓ | ✗ |
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
`{run_id}:{generation}:{ordinal}` and a digest of its arguments ([effects](../libraries/rollout/contracts/effects.md)).

| | `LocalRunner` | `DurableRunner` |
|---|---|---|
| Where code runs | the caller's process | the runner's process, inside a DBOS workflow |
| An effect | a call, recorded as run events | a durable step: its first recorded result is final |
| After a crash | runs in flight are lost; their episodes are open again in the ledger, and an episode runner plays them again | the program is run again and recorded effects return their results; what was in flight is performed again under the same identity |
| A side effect that cannot be deduplicated | performed once | guarded by an attempt marker: a possible duplicate is reported as `OUTCOME_UNKNOWN` to the model, never silently retried |
| Messages | delivered once by `message_id`, in process | delivered once by `message_id`, through any runner sharing the database |

A tool's `retry_class` says what re-execution may do: `PURE` and `IDEMPOTENT` calls are performed again;
`SIDE_EFFECTING` and `UNKNOWN` calls are performed again only against a tool set that deduplicates by effect
identity, and are guarded otherwise. Cancellation is cooperative: it is delivered at the next turn boundary, and
`teardown` runs. The durable runner is described in [durable runner](../implementations/rollout-durable/README.md).
