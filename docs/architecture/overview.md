# Architecture overview

The roles the platform is made of, what each exchanges with the others, the packages that hold them, and the protocols
between them. This page is for whoever changes the platform or adds a backend.

**Read first:** [Three ways in](../guide/perspectives.md). **Next:** [Extend the platform](../extend/README.md). Terms
are defined in the [glossary](glossary.md).

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
or agent noticing. What a channel should serve is written down in the ledger: **engine hosts** and followers load it
into their engines, and the gateway asks those for each checkpoint by name. A run is asked for with its **settings**
(the environment, the trainer, each channel's provider and model, the numbers each takes) against a **cluster
config** (the inference providers, trainers, pools and environments a cluster offers); asking records a **launch** and
submits the run's job, whose **driver** builds the run from the two and asks for what it needs: its trainer, its engine
hosts, its gateway and its runner. The **monitor** asks for runs from its page as the command line does.

## Roles

| Role | Does | Exchanges |
|---|---|---|
| Monitor, command line | Ask for runs: check their settings against the cluster config, record a launch, submit its job; pause, resume and stop runs; import environments from git; keep presets and suites; show everything | the launches; the cluster config's offers; presets and environment versions beside the ledger; the ledger, heartbeats and the blob store, read |
| A run's driver | Builds the run from the cluster config and its settings, claims its trainer and engine hosts, runs the loop of its kind | the launch (its state); the ledger (decisions, groups, steps, serving records); the blob store; heartbeats (what it waits for) |
| Trainer | Takes steps: a batch and a parent's files in, a new checkpoint's files out | the driver; the blob store and heartbeats, for a trainer on a machine of its own |
| Engine hosts, followers | Serve what a run's serving records say, loading each checkpoint from the blob store | the ledger (serving records, read); the blob store; heartbeats |
| Gateway | Samples channels for programs and harnesses holding a signed key, and records every turn | the ledger (runs' starts and serving records, read; turns) and the blob store (turns); engines, by checkpoint name; hosted APIs; heartbeats |
| Episode runners | Claim a run's episodes, play them, record them | the ledger (claims, episodes); the blob store; the gateway; sandbox pools; heartbeats |
| Sandbox pools | Lease sandboxes under an episode's claim | the ledger (leases beside it); heartbeats, for a pool served on its own |
| Ledger, blob store | Hold everything durable: the record of every run, checkpoints, turns, episodes | every role |
| Ledger service | Serves the ledger and the stores beside it over HTTP to whoever holds a token: the platform's roles may do everything, a pod reads its run's serving records and checkpoints and writes its own beat | the ledger; every role that reaches the ledger by URL |

### Placement is configuration

The roles coordinate through the ledger, the blob store, the gateway and the services the cluster config names, so
where each runs is configuration: the cluster config says where runs' jobs go, the inference providers and trainers
it offers, and the resources each role asks for ([the cluster config](../guide/cluster.md)), and a run's settings
choose among what it offers. The same roles run on one machine or each on machines of their own;
[what runs where](../deploy/roles.md) describes the deployments.

## Layers

The repository is a workspace of packages in three layers. Each package's directory, import name and page are in the
[documentation index](../README.md#packages).

| Layer | Packages | What it holds |
|---|---|---|
| Libraries | `rollout` | What environments are written against: programs, tasks, agents, tools, sandboxes and their pools, the loop, the `Runner` protocol and `LocalRunner`, contract types, hooks, memory, the environment and the curriculum |
| | `rollout-train` | Reinforcement learning on `rollout`: episode runners and episodes; sandboxes' leases beside the ledger, ending with their claims; the loop, the group algorithm, objectives as declared, distillation's teachers and the `Trainer` protocol; datasets and supervised steps; checking an environment (`rollout env check`) and importing one from git; channels and the `Engine` protocol; recording (the thinking budget, segments) and the `Renderer` protocol; the gateway, its signed keys and its turn store; the graph of checkpoints, the ledger, the ledger service and the registry; bridges; evaluation suites and evals; heartbeats and launches; the cluster config, providers, run settings, presets and the check of a run; runs built from settings, their jobs and submitting them; pods leased for a run and the services that run on them; the `rollout` command and the monitor |
| Implementations | `rollout-vllm`, `rollout-lora`, `rollout-objectives`, `rollout-tinker`, `rollout-qwen`, `rollout-gemma`, `rollout-openai`, `rollout-anthropic`, `rollout-s3`, `rollout-runpod`, `rollout-verifiers` | Each implements, for one backend, something a library defines: an interface, or (`rollout-objectives`) the losses `rollout_train.objectives` declares. `rollout-runpod` is a client of RunPod's pods API and of step-ca |
| Environments | `minecraft-team`, `minecraft-horizons`, `gridworld`, `judging` | Environments to train on |

`rollout-verifiers` is a project of its own, locked apart from the workspace.

What may depend on what is checked by `tests/test_layers.py`, from the source alone:

- a package imports exactly the workspace packages its project file declares, among its dependencies and extras;
- `rollout` imports no other workspace package, and `rollout-train` imports `rollout` and no implementation or
  environment;
- an implementation imports the libraries and no environment;
- an environment imports `rollout` and nothing of `rollout-train` or an implementation;
- one implementation imports another, or one environment another, only where the test lists the pair with the reason
  for it: `rollout-lora` and `rollout-tinker` take their objectives from `rollout-objectives`, and `minecraft-horizons`
  plays its objectives in `minecraft-team`'s worlds.

An import counts wherever it is in a package's modules: at the top, inside a function, or under `TYPE_CHECKING`.

Code above a protocol never learns which implementation it holds. Task and agent code is the same whatever serves
its models, tools and sandboxes; a training loop is the same with everything in one process or with runs, engines,
trainer and pools elsewhere. A cluster config and a run's settings name engines, renderers, trainers, sandbox
providers, blob stores and hosted-API endpoints by kind or as `module:name`, so `rollout-train` imports none of them
until a run's driver does.

## Protocols and their implementations

| Protocol | Defined in | Between | Implementations |
|---|---|---|---|
| [`Runner`](../guide/reference.md#runner) | `rollout.harness` | callers → runs | `LocalRunner` (`rollout.local`) |
| [`ModelEndpoint`](../libraries/rollout/contracts/model-endpoint.md) | `rollout.contracts` | runners → models | the [gateway](../libraries/rollout-train/gateway.md)'s endpoints (`GatewayEndpoint`, one per recorded binding, from `GatewayEndpoints`), `ResponsesEndpoint` ([`rollout_openai`](../guide/models.md)), `MessagesEndpoint` ([`rollout_anthropic`](../implementations/rollout-anthropic.md)), `ScriptedModelEndpoint` ([`rollout.testing`](../guide/testing.md)) |
| `Sampler` | `rollout_train.inference` | the gateway → a channel's tokens | `Channel` (engines this process publishes to), `RemoteChannel` (servers elsewhere, each request naming its checkpoint); a channel on a hosted API is an `ApiChannel`, sampled by message |
| [`Engine`](../guide/reference.md#engine) | `rollout_train.inference` | channels → replicas | `VllmEngine` ([`rollout_vllm`](../implementations/rollout-vllm.md)), `RemoteEngine` (a vLLM server elsewhere, over its OpenAI-compatible API: [engines elsewhere](../libraries/rollout-train/channels.md#engines-elsewhere)), `TinkerEngine` ([`rollout_tinker`](../implementations/rollout-tinker.md)), `ScriptedEngine` (`rollout_train.testing`) |
| [`Renderer`](../guide/reference.md#renderer) | `rollout_train.recorder` | the gateway → a model family's tokens | `qwen35`, `qwen3` ([`rollout_qwen`](../implementations/rollout-qwen.md)), `gemma4` ([`rollout_gemma`](../implementations/rollout-gemma.md)), `PlainRenderer` (`rollout_train.testing`) |
| [`Trainer`](../guide/reference.md#trainer) | `rollout_train` | training → weights | `LoraTrainer`, `FullTrainer` ([`rollout_lora`](../implementations/rollout-lora.md)), `TinkerTrainer` ([`rollout_tinker`](../implementations/rollout-tinker.md)), `RemoteTrainer` and `LeasedTrainer` (steps on a training pod: `rollout_train.pods`), `ScriptedTrainer` (`rollout_train.testing`); `Colocated` wraps one that shares the engines' accelerator |
| [`Algorithm`](../guide/reference.md#algorithm) | `rollout_train` | the training loop → what to train on | `Grpo`, `Preferences`, `Distillations` ([training](../libraries/rollout-train/training.md#the-algorithm)) |
| [`Ledger`](../guide/reference.md#ledger) | `rollout_train` | training and checkpoints → append-only tables | `FileLedger`, `DatabaseLedger` ([the ledger](../libraries/rollout-train/checkpoints.md#the-ledger)), `HttpLedger` (the ledger service: `rollout_train.ledger_service`) |
| [`ToolSet`](../guide/reference.md#toolset) | `rollout.harness` | runs → imported tools | a tool set in process, `RemoteToolSet` over HTTP ([tools](../guide/tools.md#imported-tools)) |
| [`Pool`](../guide/reference.md#pool), [`Provider`](../guide/reference.md#provider) | `rollout.harness` | runners → sandboxes, leased per run; pools → the sandboxes of one kind | `SandboxPool` over a provider, `RemotePool` over HTTP, `PodPools` over a run's pods; `MinecraftWorlds`, `HorizonWorlds`, `FakeSandboxes` ([sandboxes](../libraries/rollout/sandboxes.md)) |
| [`Leases`](../guide/reference.md#leases) | `rollout.harness` | pools → where their leases are kept | `MemoryLeases`, `JsonLeases`; `FileLeases`, `DatabaseLeases` and `HttpLeases` beside the ledger, where a pool's keeper ends a lease with its claim ([sandboxes](../libraries/rollout/sandboxes.md#in-training-a-lease-ends-with-its-claim)) |
| [`Blobs`](../guide/reference.md#blobs) | `rollout.harness` | runs → stored bytes | `FileBlobStore` (`rollout.harness`), `S3BlobStore` ([`rollout_s3`](../guide/content.md#media-and-blobs)), `CachedBlobs` (a copy on a pod's disk in front of another store) |
| [the ledger's `plans`, `groups`, `claims` and `episodes`](../libraries/rollout-train/rollouts.md) | `rollout_train.rollouts` | training → runs | `EpisodeRunner`, on any machine that reaches the ledger and the blob store |
| [the ledger's `serving`](../libraries/rollout-train/channels.md#what-a-channel-should-serve) | `rollout_train.serving` | training → whatever serves and samples its channels | `Follower`: in an `EngineHost` (a Ray actor: [engine hosts](../libraries/rollout-train/channels.md#engine-hosts)), or `InferencePod` beside a vLLM server (`python -m rollout_train.pods.inference`); `RemoteChannel` in a gateway |
| [`Environment`](../guide/reference.md#environment) | `rollout.environment` | training and evals → an environment's rows, starts, eval data, description and curriculum | one per environment ([three ways in](../guide/perspectives.md#building-an-environment)) |
| [`RunHooks`](../libraries/rollout/hooks.md), `Hooks` | `rollout.harness`, `rollout_train.rollouts` | runners and runs → observers | `RunFeed` ([monitor](../libraries/rollout-train/monitor.md)) |
| `Presence`, `Launches` | `rollout_train.presence`, `rollout_train.launches` | runners, drivers and whoever asks for runs → whoever watches | `FilePresence`, `DatabasePresence`, `HttpPresence`; `FileLaunches`, `DatabaseLaunches`, `HttpLaunches` ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats), [launching runs](../libraries/rollout-train/launching.md#the-launches-table)) |
| `Backend` | `rollout_train.submitting` | whoever asks for a run → where its job runs | `RayJobs` (Ray's job API), `RayJobResources` (a RayJob through the Kubernetes API server) ([launching runs](../libraries/rollout-train/launching.md)) |
| A bridge's task (`module:name`) | `rollout_train.bridges` | checkpoints → the files their engines load | `verbatim` (the bridges `verbatim` and `full-reload`), `rollout_tinker.bridges:peft` (`peft-from-tinker`), `rollout_lora.bridges:merge_quantize` (`merge-quantize`); run in the calling process or as Ray tasks (`on_ray`) ([bridges](../libraries/rollout-train/checkpoints.md#bridges)) |

Types that cross these boundaries are defined once, in [contracts](../libraries/rollout/contracts/README.md).

## Visibility

What each part sees. A ✗ is a boundary the code keeps, not an optimization left undone.

| | Canonical content | Tokens and logprobs | Weights version (checkpoint depth) | Engines, trainer, machines |
|---|---|---|---|---|
| Task and agent code | ✓ | ✗ | ✗ | ✗ |
| Runner | ✓ | ✗ | ✗ | ✗ |
| Gateway and channels | ✓ | ✓ | ✓ | the engines and endpoints it samples |
| Engine hosts | ✗ | ✓ (those they sample) | ✓ | their servers only |
| Episode runners | labels and results | ✓ (in episodes) | ✓ | ✗ |
| Training loop and trainer | labels and results | ✓ | ✓ | ✗ |
| Cluster config | ✗ | ✗ | ✗ | ✓ |

## A turn

The agent samples a reply; the task responds with an observation.

```text
agent.act ──▶ Model.sample ──▶ model endpoint ──▶ (gateway ──▶ channel ──▶ engine: tokens in; tokens, logprobs out
          ◀── canonical reply + usage                                ──▶ turn kept in the ledger and the blob store)
harness ──▶ a model API + signed key ──▶ gateway ──▶ endpoint (the checkpoint, by name) ──▶ turn kept ──▶ reply
task.respond(reply) ──▶ run_tools ──▶ @tool method ──▶ (an imported tool set, a sandbox, a blob store)
          ◀── Observation(tool results) ──▶ next turn
```

Everything that leaves task or agent code on the way is an **effect** with an identity:
`{run_id}:{generation}:{ordinal}` and a digest of its arguments ([effects](../libraries/rollout/contracts/effects.md)).

The `LocalRunner` runs code in the caller's process, and performs each effect once, as a call recorded as run
events. After a crash the runs in flight are lost: their episodes are open again in the ledger, and an episode runner
plays them again.

A tool's `retry_class` says whether a call is safe to make again: `PURE` and `IDEMPOTENT` calls are; `SIDE_EFFECTING`
and `UNKNOWN` calls are only against a tool set that deduplicates by effect identity
([effects](../libraries/rollout/contracts/effects.md#receivers-that-deduplicate)). Cancelling a run cancels its task,
and `teardown` runs.
