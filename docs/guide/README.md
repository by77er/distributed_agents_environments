# Write an environment

This section is for people who write environments: the worlds agents are trained and evaluated in. It shows how to
write a task, give the model tools, choose what the model sees, test it all without a GPU, and publish it so that a
cluster can import it.

**Read first:** [Start here](../start/README.md). **Next:** [Run a first episode](getting-started.md).

## Read in this order

1. [Run a first episode](getting-started.md): install, write a first task, and play one episode against a scripted
   model.
2. [Write a task](tasks.md): hooks, observations, rewards, endings, and more model slots such as a judge.
3. [Give the model tools](tools.md): `@tool` methods, schemas from type hints, errors as observations, and tool sets
   imported by name.
4. [Choose what the model sees](agents.md): context selection, context hints, and how an agent turns the model's
   output into one reply.
5. [Messages, files and digests](content.md): canonical content, tool calls and results, blobs, digests.
6. [What a run records](runs-and-events.md): effects, identifiers, run events, and the local runner.
7. [Use a hosted model](models.md): bind a model slot to a model behind the OpenAI Responses API.
8. [Test with a scripted model](testing.md): scripted endpoints, helpers, and `rollout env check`.
9. [Import an environment from git](publishing.md): what a repository needs so that a cluster can import it from
   the monitor, and what the import checks.

Then, as you need them:

- **The harness in depth.** [The harness](../libraries/rollout/README.md) (the loop, programs and runners),
  [sandboxes](../libraries/rollout/sandboxes.md), [determinism](../libraries/rollout/determinism.md),
  [hooks](../libraries/rollout/hooks.md), and [memory for long episodes](../libraries/rollout/memory.md).
- **Example environments.** The [Minecraft team](../products/minecraft-team.md), the
  [gridworld](../products/gridworld.md), and [judging](../products/judging.md), where a judge scores open-ended
  answers.

## The ideas this section uses

The [glossary](../architecture/glossary.md) defines every term. These are the ones the pages here use most:

- **Environment.** What a run trains on and an eval measures: rows (situations, easiest first), how to draw a start
  of one, eval data that training never draws, what its results say, a version, and perhaps a curriculum of its own.
- **Task.** The world an agent acts in: the first observation, a response to each reply, tools, and scoring.
- **Agent.** The policy side: what the model sees each turn, and how its output becomes one reply.
- **Run.** One execution of a program, identified by `run_id` (`r_{ulid}`). For an agent, one episode.
- **Observation.** What the model is shown next, with the reward for the reply it answers and whether the episode
  ended.
- **Reply** and **turn.** The reply is the assistant `Message` the agent returns. A turn is one reply and the
  observation that answers it; `run.turn` counts replies.
- **Model slot.** A named model a task uses. The agent acts through `policy`; a task may declare others, such as a
  judge.
- **Model endpoint.** What serves a model slot: the gateway, an API adapter, or a scripted endpoint in tests.
- **Effect.** An operation that leaves task or agent code, such as a model sample. It has a stable `effect_id`.
- **Run event.** A typed record of something that happened in a run, in a gapless sequence.

## Where things live

Each row is a module of one package. The [documentation home](../README.md#packages) lists the packages.

| Import from | Package | For |
|---|---|---|
| `rollout.harness` | `rollout` | `Task`, `Agent`, `tool`, `Observation`, `End`, `RunContext`, `rollout`, `Program`, `Memory` |
| `rollout.contracts` | `rollout` | `Message`, content blocks, `ToolSpecification`, `ToolResult`, identifiers, digests, events |
| `rollout.local` | `rollout` | `LocalRunner`, `LocalRunContext`: runs in this process |
| `rollout.environment` | `rollout` | `Environment`, `Row`, `Start`, `Description`, `drawn`, `train_start`, `binding_for` |
| `rollout.curriculum` | `rollout` | `Curriculum`, `curriculum_of`, `solved_share` |
| `rollout.testing` | `rollout` | `ScriptedModelEndpoint`, `local_run`, `events_of`, `payload`, `tool_call_reply` |
| `rollout_train.rollouts` | `rollout-train` | `EpisodeRunner`, `Plan`, `plan`, `episodes_of`, `playing`, `Hooks`, `Episode`, `Record` |
| `rollout_train` | `rollout-train` | `train`, `Grpo`, `Trainer`, `Colocated`, `Checkpoints`, `FileLedger` |
| `rollout_train.inference`, `rollout_train.recorder`, `rollout_train.gateway` | `rollout-train` | `Channel`, `Engine`, `Limits`; `Segment`, `Renderer`; `Gateway`, `GatewayEndpoints` |
| `rollout_train.profile`, `rollout_train.monitor` | `rollout-train` | `Profile`, `Platform`; `RunFeed` |
| `rollout_train.testing` | `rollout-train` | `ScriptedEngine`, `PlainRenderer`, `plain_channel` |
| `rollout_openai` | `rollout-openai` | `ResponsesEndpoint`, `codex_provider`, `ApiKey` |
| `rollout_s3` | `rollout-s3` | `S3BlobStore` |
| `rollout_vllm` | `rollout-vllm` | `VllmEngine` |
| `rollout_lora` | `rollout-lora` | `LoraTrainer`, `LoraSettings` |
| `rollout_qwen` | `rollout-qwen` | `qwen35`, `qwen3` |
| `rollout_gemma` | `rollout-gemma` | `gemma4` |

## Conventions

- Every block tagged exactly `python` in these pages runs as part of the test suite (`tests/test_docs.py`). A block
  tagged `py` shows a shape and is not run. The [API reference](reference.md) is generated from the source, and a
  test fails when it is out of date.
- Hooks and tools are `async` methods. Everything a task or agent does outside its own code goes through `run`.
- Contract types are immutable pydantic models: construct new values rather than modifying existing ones.
- Type names are never abbreviated: `Observation`, not `Obs`.
