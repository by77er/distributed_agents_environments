# Glossary

| Term | Meaning |
|---|---|
| **Program** | `async` code with a `main(run)` method. The agent loop is one program (`AgentProgram`). |
| **Task** | The environment an agent acts in, in the reinforcement-learning sense: tools, lifecycle hooks, responses to model turns, scoring. See [tasks](../guide/tasks.md). |
| **Agent** | The policy side of the loop: selects what the model sees and produces one reply per turn. See [agents](../guide/agents.md). |
| **Harness** | The loop that drives a task with an agent. See [harness](../libraries/rollout/README.md). |
| **Run** | One execution of a program under a `RunBinding`, identified by `run_id`: one episode. A training run is the other sense of the word ([below](#training-terms)). |
| **RunSpecification / RunBinding** | What to run (a program reference and parameters) / how it is served here (model endpoints, imported tool sets, delivery policy). |
| **Deployment** | A named, addressable `RunSpecification`. |
| **Runner** | Executes runs: `LocalRunner` in process, or `DurableRunner`, whose runs survive it ([durable runner](../implementations/rollout-durable/README.md)). |
| **Observation** | The task's response to a model turn: messages shown to the model next, an optional reward, an optional ending, logged info. |
| **Ending** | How an episode ended: `TERMINATED` (a real end state) or `TRUNCATED` (stopped by a limit). |
| **WaitFor** | A return value of `start`, `respond` or `resume` that suspends the run until a message arrives or a timeout passes. |
| **Conversation** | A sequence of runs of one deployment under a conversation key; at most one run consumes its messages at a time. See [conversations](../guide/conversations.md). |
| **Envelope** | A message: kind, canonical content, optional data, sender, reply address, `message_id`. |
| **Priority / delivery mode** | `LOW` / `NORMAL` / `HIGH`, mapped by the run's `DeliveryPolicy` to `QUEUE` (wait for the next `WaitFor`), `STEER` (merge into the next observation) or `INTERRUPT` (cancel the sample in flight). |
| **Model slot** | A named model a program uses (`policy`, `user`, …); each is bound to an endpoint and recorded as its own session. |
| **Effect** | An operation that reaches outside code: a model sample, a tool call, an environment operation, an output. Identified by `effect_id = {run_id}:{generation}:{ordinal}` and a digest of its arguments. |
| **Replay** | Running a program again while its recorded effects return their results; how a durable runner resumes. |
| **Tool** | A `ToolSpecification` (what the model sees) and its implementation: a `@tool` method, or an imported tool. |
| **Tool set / ToolBinding** | Tools a program imports by name / where a run finds them: in the runner's process or at a URL. |
| **Canonical content** | Model-agnostic messages and content blocks; the only content form that code, run events and tool sets use. |
| **Model endpoint** | Anything that implements the [model endpoint contract](../libraries/rollout/contracts/model-endpoint.md): the recorder or an adapter. |
| **Profile** | A deployment, described: channels and engines, the trainer, the runner, where tool sets live. See [deploying](../guide/deploying.md). |
| **Library / implementation / product / environment** | The four kinds of package in the repository: what code is written against; one implementation of an interface a library defines; an application; something to train on. See [overview](overview.md#layers). |
| **Environment** | A computer a task creates through `run.environments` ([computers](../implementations/rollout-computers.md)). Also, in the reinforcement-learning sense, what a task is; the packages under `environments/` are environments in that sense. |
| **Harness inside an environment** | A program's own agent, given an OpenAI-compatible address for a model slot: recorded like any other sample. |

## Training terms

A training run has steps; a step covers groups; a group is one start of one row played as several episodes; an
episode is one run of a program and has one rollout per agent (model slot); each rollout becomes a trajectory, made
of segments.

| Term | Meaning |
|---|---|
| **Catalog / row** | What an environment offers to train on, easiest first / one situation of it, of which a start is drawn for each group. See [three ways in](../guide/perspectives.md#building-an-environment). |
| **Training run** | One training loop (`rollout_train.train`, `rollout train`) over a catalog, kept in one directory and in the ledger: its groups, their results and its steps. See [training](../libraries/rollout-train/training.md). |
| **Step** | One call of the trainer, over the groups queued with something to train on (at least `groups_per_step` of them, except at the end of the run); it makes one version of the policy. |
| **Group** | One start of one row, played as several episodes (one ticket of `group_size` runs) that are compared with each other. Its episodes carry the labels `group` and `iteration` (its number). |
| **Episode** | One run of a program, as training sees it once it has ended: labels, outcome, result, and a trajectory per model slot. See [episodes](../libraries/rollout-train/episodes.md). |
| **Rollout** | One model slot's part of an episode as it plays: every turn of one agent. Each rollout becomes a trajectory. |
| **Session** | The recorder's record of one rollout: every sample of one model slot of one run. |
| **Trajectory** | What a rollout leaves to train on: its segments and its rewards (`Trajectory`). |
| **Segment** | A piece of a trajectory: tokens that only grew by appending, with the spans the policy sampled, their behavior logprobs and weights versions (`Segment`). An edited context (a compaction, thinking dropped) starts the next. |
| **Behavior logprob** | The log-probability of a sampled token under the distribution it was sampled from. |
| **Rollout job** | Runs rows submitted by a caller and delivers their episodes through one ordered log. It knows no algorithm. See [rollouts](../libraries/rollout-train/rollouts.md). |
| **Policy / version** | One line of training, by name / one of its versions, `policy@number`, which a step makes. See [policies](../libraries/rollout-train/policies.md). |
| **Ledger** | Append-only tables that hold a training run's decisions and results and the policies' versions, with fences so that one writer holds each. |
| **Channel** | A trainable policy being served, by name: its engines, its limits, and the version it samples from. |
| **Engine** | One replica serving a model: tokens in; tokens, logprobs and a finish reason out. `VllmEngine` is one ([vLLM engine](../implementations/rollout-vllm.md)). |
| **Recorder** | The model endpoint for channels: renders contexts to tokens, samples, and keeps what was sampled. |
| **Renderer** | The chat template, tokenizer and parser of one model family ([Qwen](../implementations/rollout-qwen.md), [Gemma](../implementations/rollout-gemma.md)). |
| **Weights version** | The number of the policy version a channel serves; every sampled span carries the one it was sampled at. |
| **Trainer** | Turns weighted segments into new weights, within a budget it states. `LoraTrainer` is one ([LoRA trainer](../implementations/rollout-lora.md)). |
