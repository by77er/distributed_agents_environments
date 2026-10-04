# Glossary

| Term | Meaning |
|---|---|
| **Program** | `async` code with a `main(run)` method. The agent loop is one program (`AgentProgram`). |
| **Task** | The environment an agent acts in, in the reinforcement-learning sense: tools, lifecycle hooks, responses to model turns, scoring. See [tasks](../guide/tasks.md). |
| **Agent** | The policy side of the loop: selects what the model sees and produces one reply per turn. See [agents](../guide/agents.md). |
| **Harness** | The loop that drives a task with an agent. See [harness](../libraries/rollout/README.md). |
| **Run** | One execution of a program under a `RunBinding`, identified by `run_id`: one episode. A training run is the other sense of the word ([below](#training-terms)). |
| **RunSpecification / RunBinding** | What to run (a program reference and parameters) / how it is served here (model endpoints, imported tool sets, sandbox pools, delivery policy). |
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
| **Sandbox** | Something a program runs against for one run, outside its own code (a Minecraft world, a container, an environment's worker): declared by the program (`SandboxSpec`), acquired by the runner before the program starts and released when it ends, reached as `run.sandbox(name)`. See [sandboxes](../libraries/rollout/sandboxes.md). |
| **Pool / provider / PoolBinding** | Hands out sandboxes of one kind under leases, saying how many it has room for / makes, deletes and operates them / which pool serves a kind for a run: in the runner's process or at a URL. |
| **Lease** | A sandbox held under a key (the run's lease and the sandbox's name): the same key gets the same sandbox. An episode's are held under its claim, and end with it. |
| **Canonical content** | Model-agnostic messages and content blocks; the only content form that code, run events and tool sets use. |
| **Model endpoint** | Anything that implements the [model endpoint contract](../libraries/rollout/contracts/model-endpoint.md): the gateway or an adapter. |
| **Profile** | A deployment, described: channels and engines, the trainer, the runner, where tool sets and sandbox pools live. See [deploying](../guide/deploying.md). |
| **Library / implementation / product / environment** | The four kinds of package in the repository: what code is written against; one implementation of an interface a library defines; an application; something to train on. See [overview](overview.md#layers). |
| **Environment** | What a run trains on and an eval measures (`rollout.environment.Environment`, [below](#training-terms)); the packages under `environments/` are environments in that sense. Also, a computer a task creates through `run.environments` ([computers](../implementations/rollout-computers.md)). |
| **Harness inside an environment** | A program's own agent, given an address for a model slot that speaks OpenAI's and Anthropic's APIs: recorded like any other sample. |

## Training terms

A training run has steps; a step covers groups; a group is one start of one row played as several episodes; an
episode is one run of a program and has one rollout per agent (model slot); each rollout becomes a trajectory, made
of segments.

| Term | Meaning |
|---|---|
| **Environment / row** | What a run trains on and an eval measures: its program, its rows (easiest first) and how a start of one is drawn, its eval data, its description, its version, and perhaps a curriculum of its own / one situation of it, of which a start is drawn for each group. See [environments](../libraries/rollout-train/rollouts.md#environment). |
| **Eval data** | An environment's named lists of starts (`evals()`), which training never draws (`train_start`); each is frozen as a suite of its name the first time it is played. See [train and eval](../libraries/rollout-train/rollouts.md#train-and-eval). |
| **Description / version** | What an environment's results say: the range of its rewards, whether they say `solved` and `saturated`, what `duration` counts, how its observations are shown / a name for what an environment is now, changed whenever its rows, starts, eval data or scoring change. Each run's start records both; each version of a suite records the version. |
| **Curriculum** | Which row a run trains on next, given its results (and its evals' results, for a gate): the environment's own, or the generic one (`rollout.curriculum`). See [the curriculum](../libraries/rollout-train/training.md#the-curriculum). |
| **Training run** | One training loop (`rollout_train.train`, `rollout train`) over an environment, kept in one directory and in the ledger: its groups, their results and its steps. See [training](../libraries/rollout-train/training.md). |
| **Step** | One call of the trainer, over the groups queued with something to train on (at least `groups_per_step` of them, except at the end of the run); it makes one checkpoint, from the one the step before made. |
| **Group** | One start of one row, played as several episodes (its record's `episodes`, the algorithm's `group_size`) that are compared with each other. Its episodes are numbered from 1 within it and carry the labels `run`, `group` and `episode`; the rest of it is in its record. |
| **Episode** | One run of a program, as training sees it once it has ended: labels, outcome, result, and a trajectory per model slot. See [episodes](../libraries/rollout-train/episodes.md). |
| **Rollout** | One model slot's part of an episode as it plays: every turn of one agent. Each rollout becomes a trajectory. |
| **Session** | The gateway's record of one rollout: every sample of one model slot of one run. |
| **Trajectory** | What a rollout leaves to train on: its segments and its rewards (`Trajectory`). |
| **Segment** | A piece of a trajectory: tokens that only grew by appending, with the spans the policy sampled, their behavior logprobs and weights checkpoints (`Segment`). An edited context (a compaction, thinking dropped) starts the next. |
| **Behavior logprob** | The log-probability of a sampled token under the distribution it was sampled from. |
| **Plan** | How a training run's episodes are played: its program and its binding, in the run's `plans` table. |
| **Episode runner** | Claims the episodes runs ask for in the ledger, plays them on a runner and records them, at most as many at once as it has places. Several, on one machine or many, share the work. It knows no algorithm. See [rollouts](../libraries/rollout-train/rollouts.md). |
| **Claim** | An episode runner's append under `GROUP/EPISODE/ATTEMPT` in a run's `claims` table: the first append wins, and it holds while the runner keeps the fence it made it under (or adopted it under, started again over a durable runner) and beats. It names the run that plays it. The episode's sandboxes are leased under it, and their leases end with it. |
| **Heartbeat** | What a runner, an engine host, a launcher or a pool served on its own writes every 15 seconds beside the ledger: its host, its machine's measurements, its engines and channels, how full its pools are, or what it offers. One silent for 90 seconds is taken to be gone. See [heartbeats](../libraries/rollout-train/rollouts.md#heartbeats). |
| **Launch / launcher** | A training run or an eval asked for (its profile, environment, name, checkpoint and settings; an eval's suite and episodes a start) / the process on a training machine that claims launches for the profiles it offers and starts `rollout train` or `rollout eval` for them, as a process of its own or as a Ray job. See [launchers](../guide/deploying.md#launchers). |
| **Suite / eval** | An eval configuration, by name: an environment, its starts (its eval data of a name, each of some rows' start drawn with each of some seeds, or starts given), episodes per start and the eval channel's sampling limits; kept in versions (`NAME@N`), each never changed, an edit making the next, and the name pointing to the newest / one version of a suite played by one checkpoint or the base model, training nothing: a run of its own whose start says `kind: eval` and the version, recording each episode's outcome under `evaluations/SUITE/EVAL/results`. See [evals](../libraries/rollout-train/evals.md). |
| **Dataset** | Examples to imitate, chosen from runs' episodes by an episode rule and turn filters, made once: a record in the ledger's `datasets` table and a manifest blob of one line per example, and a name if it is given one. A supervised step on one makes a checkpoint whose parents after the first sampled its examples. See [datasets](../libraries/rollout-train/datasets.md). |
| **Bridge** | How a checkpoint's files in its trainer's format (`peft`, `full`, `tinker`) become files a provider loads: a task named `module:name`, run once per checkpoint and bridge, noted in `checkpoints/resharding` and `checkpoints/resharded` under `CHECKPOINT@BRIDGE`; in the calling process or as a Ray task. See [bridges](../libraries/rollout-train/checkpoints.md#bridges). |
| **Checkpoint** | Weights a step (or imitation) made: a node of a graph, with an id of its own (shown by its shortest unique start), its parents (what it was trained from, and any others it learned from), its base model, its depth, and the run and step that made it. See [checkpoints](../libraries/rollout-train/checkpoints.md). |
| **Base model** | The model a checkpoint adapts (`Qwen/Qwen3.5-9B`): the root every line of checkpoints grows from. |
| **Depth** | A checkpoint's steps from its base model along its first parents: the number stamped on the tokens it samples. |
| **Fork** | A run started from a checkpoint of another run (or an earlier one of its own): it trains on from there, sharing its parent's files. |
| **Bookmark** | A name for a checkpoint, kept in the registry, moved by hand or carried by a run to each checkpoint it makes. A checkpoint needs none. See [bookmarks](../libraries/rollout-train/checkpoints.md#bookmarks). |
| **Id / name** | What a run is kept under, which never changes / what it is called, which can be chosen and changed, in the registry beside the ledger. See [runs](../libraries/rollout-train/checkpoints.md#runs). |
| **Ledger** | Append-only tables that hold a training run's decisions and results and the checkpoints, with fences so that one writer holds each. |
| **Channel** | A trainable model being served, by name: its engines, its limits, and the checkpoint it samples from. |
| **Engine** | One replica serving a model: tokens in; tokens, logprobs and a finish reason out. `VllmEngine` is one ([vLLM engine](../implementations/rollout-vllm.md)); `RemoteEngine` is a vLLM server elsewhere, a request naming the checkpoint it samples from as its model. |
| **What a channel should serve** | A run's record, in its `serving` table, that its channel (`RUN/NAME`) serves a checkpoint from then on: its id, depth and kind, and the files its engines load. The training loop writes it each time it serves a checkpoint; whatever serves or samples the channel elsewhere reads it. See [what a channel should serve](../libraries/rollout-train/channels.md#what-a-channel-should-serve). |
| **Follower / engine host** | Keeps a process's channels serving what a run says, loading each checkpoint from the blob store, named by its id / a process that does that for the vLLM servers on its machine, and nothing else (`rollout engines`). See [deploying](../guide/deploying.md#engines-on-other-machines). |
| **Max lag** | How many checkpoints behind what its channel should serve a sample may be, where its server does not have the newest yet (1 unless a profile says otherwise; 0 for an eval). |
| **Gateway** | The model endpoint for channels, as a service that keeps no session: renders contexts to tokens, samples, and records every turn in the ledger and the blob store; in a runner's own process, or replicas of its own. |
| **Renderer** | The chat template, tokenizer and parser of one model family ([Qwen](../implementations/rollout-qwen.md), [Gemma](../implementations/rollout-gemma.md)). |
| **Weights checkpoint** | The depth of the checkpoint a channel serves; every sampled span carries the one it was sampled at. |
| **Trainer** | Turns weighted segments into new weights, within a budget it states. `LoraTrainer` is one ([LoRA trainer](../implementations/rollout-lora.md)). |
