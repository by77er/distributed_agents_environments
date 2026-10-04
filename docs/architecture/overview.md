# Architecture overview

## The system in one paragraph

A **run** is one episode of a **program**: usually an agent acting in a **task**. The task is the environment in the
reinforcement-learning sense: it defines tools and responds to every model turn; the agent decides what the model
sees and how it acts. Both are Python `async` code. Model samples go to a **model endpoint**: an adapter to a
third-party model, or the **gateway**, which samples from a trainable **channel** and records the exact tokens,
logprobs and the version of the weights (the depth of the checkpoint served). It keeps no session of its own:
programs and harnesses that hold a signed key for a run's model slot speak a standard model API to it, and it chooses
the checkpoint, samples, and records every turn in the ledger and the blob store before it replies; a run's episode is
assembled from those turns. A training loop asks for a row's start to be played as a **group** of **episodes**,
in the **ledger**; **episode runners**, on any machine that reaches the ledger, claim the episodes, play them and
record them there. A program runs against the **sandboxes** it declares (a Minecraft world, a container), which the
runner leases from **pools** before the program starts, under the episode's claim, and releases when it ends. The loop
takes a **step** over several groups at a time; each step's new weights are published to the channel without any task
or agent noticing. What a channel should serve is written down in the ledger: engines in the trainer's process are
published to directly, **engine hosts** load it into vLLM servers on other machines, and runners ask those for each
checkpoint by name. A **profile** says which engines, trainer, runner, tool sets and pools stand behind all of it.

## Layers

The repository is a workspace of packages in four layers. Each package's directory, import name and page are in the
[documentation index](../README.md#packages).

| Layer | Packages | What it holds |
|---|---|---|
| Libraries | `rollout` | What environments are written against: programs, tasks, agents, tools, sandboxes and their pools, conversations, the loop, the `Runner` protocol and `LocalRunner`, contract types, hooks, memory, the environment and the curriculum |
| | `rollout-train` | Reinforcement learning on `rollout`: episode runners and episodes; sandboxes' leases beside the ledger, ending with their claims; the loop, the group algorithm and the `Trainer` protocol; checking an environment (`rollout env check`); channels and the `Engine` protocol; recording (the thinking budget, segments) and the `Renderer` protocol; the gateway, its signed keys and its turn store; the graph of checkpoints, the ledger and the registry; resharding; evaluation suites and evals; heartbeats, launches and the launcher; the profile, the `rollout` command and the monitor |
| Implementations | `rollout-durable`, `rollout-vllm`, `rollout-lora`, `rollout-qwen`, `rollout-gemma`, `rollout-computers`, `rollout-openai`, `rollout-s3` | One implementation each of an interface a library defines |
| Products | `project-assistant`, `agent-sessions` | Applications built on the libraries and implementations |
| Environments | `minecraft-team` | An environment to train on |

What may depend on what is checked by `tests/test_layers.py`:

- a package imports only the workspace packages its project file declares;
- the libraries require no implementation, and `rollout` requires no other workspace package;
- an environment imports `rollout` and nothing above it.

Code above a protocol never learns which implementation it holds. Task and agent code is the same under either
runner; a training loop is the same with everything in one process or with runs, engines, trainer and pools elsewhere.
A profile names engines, renderers, trainers, tool sets and sandbox providers as `module:name`, so `rollout-train`
imports none of them.

## Protocols and their implementations

| Protocol | Defined in | Between | Implementations |
|---|---|---|---|
| [`Runner`](../guide/reference.md#runner) | `rollout.harness` | callers → runs | `LocalRunner` (`rollout.local`), `DurableRunner` ([`rollout_durable`](../implementations/rollout-durable/README.md)) |
| [`ModelEndpoint`](../libraries/rollout/contracts/model-endpoint.md) | `rollout.contracts` | runners → models | the [gateway](../libraries/rollout-train/gateway.md)'s endpoints (`GatewayEndpoints`), `ResponsesEndpoint` ([`rollout_openai`](../guide/models.md)), `ScriptedModelEndpoint` ([`rollout.testing`](../guide/testing.md)) |
| [`Engine`](../guide/reference.md#engine) | `rollout_train.inference` | channels → replicas | `VllmEngine` ([`rollout_vllm`](../implementations/rollout-vllm.md)), `RemoteEngine` (a vLLM server elsewhere, over its OpenAI-compatible API: [engines elsewhere](../libraries/rollout-train/channels.md#engines-elsewhere)), `ScriptedEngine` (`rollout_train.testing`) |
| [`Renderer`](../guide/reference.md#renderer) | `rollout_train.recorder` | the gateway → a model family's tokens | `qwen35`, `qwen3` ([`rollout_qwen`](../implementations/rollout-qwen.md)), `gemma4` ([`rollout_gemma`](../implementations/rollout-gemma.md)), `PlainRenderer` (`rollout_train.testing`) |
| [`Trainer`](../guide/reference.md#trainer) | `rollout_train` | training → weights | `LoraTrainer` ([`rollout_lora`](../implementations/rollout-lora.md)); `Colocated` wraps one that shares the engines' accelerator |
| [`Algorithm`](../guide/reference.md#algorithm) | `rollout_train` | the training loop → what to train on | `Grpo` ([training](../libraries/rollout-train/training.md#the-algorithm-grpo)) |
| [`Ledger`](../guide/reference.md#ledger) | `rollout_train` | training and checkpoints → append-only tables | `FileLedger`, `DatabaseLedger` ([the ledger](../libraries/rollout-train/checkpoints.md#the-ledger)) |
| [`ToolSet`](../guide/reference.md#toolset) | `rollout.harness` | runs → imported tools | a tool set in process, `RemoteToolSet` over HTTP ([tools](../guide/tools.md#imported-tools)) |
| [`Pool`](../guide/reference.md#pool), [`Provider`](../guide/reference.md#provider) | `rollout.harness` | runners → sandboxes, leased per run; pools → the sandboxes of one kind | `SandboxPool` over a provider, `RemotePool` over HTTP; `MinecraftWorlds`, `FakeSandboxes` ([sandboxes](../libraries/rollout/sandboxes.md)) |
| [`Leases`](../guide/reference.md#leases) | `rollout.harness` | pools → where their leases are kept | `MemoryLeases`; `FileLeases`, `DatabaseLeases` beside the ledger, where a pool's keeper ends a lease with its claim ([sandboxes](../libraries/rollout/sandboxes.md#in-training-a-lease-ends-with-its-claim)) |
| [`EnvironmentService`](../guide/reference.md#environmentservice) | `rollout.harness` | runs → computers | `NamespaceEnvironments`, `LocalEnvironments` ([`rollout_computers`](../implementations/rollout-computers.md)) |
| [`Blobs`](../guide/reference.md#blobs) | `rollout.harness` | runs → stored bytes | `FileBlobStore` (`rollout.harness`), `S3BlobStore` ([`rollout_s3`](../guide/content.md#media-and-blobs)) |
| [the ledger's `plans`, `groups`, `claims` and `episodes`](../libraries/rollout-train/rollouts.md) | `rollout_train.rollouts` | training → runs | `EpisodeRunner`, on any machine that reaches the ledger and the blob store |
| [the ledger's `serving`](../libraries/rollout-train/channels.md#what-a-channel-should-serve) | `rollout_train.serving` | training → whatever serves and samples its channels | `Follower` (`rollout engines`, or a runner's own engines), `RemoteChannel` in a gateway |
| [`Environment`](../guide/reference.md#rolloutenvironmentenvironment) | `rollout.environment` | training and evals → an environment's rows, starts, eval data, description and curriculum | one per environment ([three ways in](../guide/perspectives.md#building-an-environment)) |
| [`RunHooks`](../libraries/rollout/hooks.md), `Hooks` | `rollout.harness`, `rollout_train.rollouts` | runners and runs → observers | `RunFeed` ([monitor](../libraries/rollout-train/monitor.md)) |
| `Presence`, `Launches` | `rollout_train.presence`, `rollout_train.launches` | runners and launchers → whoever watches or asks for runs | `FilePresence`, `DatabasePresence`; `FileLaunches`, `DatabaseLaunches` ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats), [launchers](../guide/deploying.md#launchers)) |
| A layout (`module:name`) | `rollout_train.resharding` | checkpoints → the files their engines load | `verbatim`; run in the run's process or as a Ray task (`on_ray`) ([resharding](../libraries/rollout-train/checkpoints.md#resharding)) |

Types that cross these boundaries are defined once, in [contracts](../libraries/rollout/contracts/README.md).

## Visibility

What each part sees. A ✗ is a boundary the code keeps, not an optimization left undone.

| | Canonical content | Tokens and logprobs | Weights version (checkpoint depth) | Engines, trainer, machines |
|---|---|---|---|---|
| Task and agent code | ✓ | ✗ | ✗ | ✗ |
| Runner | ✓ | ✗ | ✗ | ✗ |
| Gateway and channels | ✓ | ✓ | ✓ | the engines and endpoints it samples |
| Engine hosts | ✗ | ✗ | ✓ | their servers only |
| Episode runners | labels and results | ✓ (in episodes) | ✓ | ✗ |
| Training loop and trainer | labels and results | ✓ | ✓ | ✗ |
| Profile | ✗ | ✗ | ✗ | ✓ |

## A turn

The agent samples a reply; the task responds with an observation.

```
agent.act ──▶ Model.sample ──▶ model endpoint ──▶ (gateway ──▶ channel ──▶ engine: tokens in; tokens, logprobs out
          ◀── canonical reply + usage                                ──▶ turn kept in the ledger and the blob store)
harness ──▶ a model API + signed key ──▶ gateway ──▶ endpoint (the checkpoint, by name) ──▶ turn kept ──▶ reply
task.respond(reply) ──▶ run_tools ──▶ @tool method ──▶ (an imported tool set, a sandbox, an environment, a blob store)
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
