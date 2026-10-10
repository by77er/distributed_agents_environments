# Glossary

Every term the documentation uses, defined once, each under a heading of its own so that a page can link to it. This
page is for looking a word up.

**Read first:** [Start here](../start/README.md). **Next:** [Architecture overview](overview.md).

## Harness terms

### Program

`async` code with a `main(run)` method. The agent loop is one program (`AgentProgram`).

### Task

The environment an agent acts in, in the reinforcement-learning sense: tools, lifecycle hooks, responses to model turns, scoring. See [tasks](../guide/tasks.md).

### Agent

The policy side of the loop: selects what the model sees and produces one reply per turn. See [agents](../guide/agents.md).

### Harness

The loop that drives a task with an agent. See [harness](../libraries/rollout/README.md).

### Run

One execution of a program under a `RunBinding`, identified by `run_id`: one episode. A training run is the other sense of the word ([training terms](#training-terms)).

### RunSpecification / RunBinding

What to run: a program reference (`module:QualifiedName` and its parameters) and its binding / how the run is served here: an endpoint for each model slot (direct, or a channel through the gateway), imported tool sets, sandbox pools.

### Runner

Executes runs: `LocalRunner`, in process ([runs and events](../guide/runs-and-events.md)).

### Observation

The task's response to a model turn: messages shown to the model next, an optional reward, an optional ending, logged info.

### Ending

How an episode ended: `TERMINATED` (a real end state) or `TRUNCATED` (stopped by a limit).

### Model slot

A named model a program uses (`policy`, `user`, …), declared as a `ModelSlot`: whether a run may train on its turns (`trained`) and whether it judges the others (`judge`). A run's binding serves each, directly or through the gateway, which records each slot as its own session.

### Effect

An operation that reaches outside code: a model sample, a tool call, an output. Identified by `effect_id = {run_id}:{generation}:{ordinal}` and a digest of its arguments.

### Tool

A `ToolSpecification` (what the model sees) and its implementation: a `@tool` method, or an imported tool.

### Tool set / ToolBinding

Tools a program imports by name / where a run finds them: in the runner's process or at a URL.

### Sandbox

Something a program runs against for one run, outside its own code (a Minecraft world, a container, an environment's worker): declared by the program (`SandboxSpec`), acquired by the runner before the program starts and released when it ends, reached as `run.sandbox(name)`. See [sandboxes](../libraries/rollout/sandboxes.md).

### Pool / provider / PoolBinding

Hands out sandboxes of one kind under leases, saying how many it has room for / makes, deletes and operates them / which pool serves a kind for a run: in the runner's process or at a URL.

### Lease

A sandbox held under a key (the run's lease and the sandbox's name): the same key gets the same sandbox. An episode's are held under its claim, and end with it.

### Canonical content

Model-agnostic messages and content blocks; the only content form that code, run events and tool sets use.

### Model endpoint

Anything that implements the [model endpoint contract](../libraries/rollout/contracts/model-endpoint.md): the gateway or an adapter.

### Cluster config

A cluster, described once: where the ledger and the blob store are, where runs' jobs go, the inference providers, trainers, sandbox pools and environments it offers. See [the cluster config](../guide/cluster.md).

### Run settings / preset

What a run is: its environment, trainer, each channel's provider, model and renderer, and the numbers each takes, by dotted key / named, versioned run settings kept beside the ledger. See [run settings](../guide/cluster.md#run-settings).

### Library / implementation / environment

The three kinds of package in the repository: what code is written against; something a library defines (an interface, or the losses it declares), implemented for one backend; something to train on. See [overview](overview.md#layers).

### Environment

What a run trains on and an eval measures (`rollout.environment.Environment`, [training terms](#training-terms)); the packages under `environments/` are environments in that sense.

### Harness inside an environment

A program's own agent, given an address for a model slot that speaks OpenAI's and Anthropic's APIs: recorded like any other sample.

## Training terms

A training run has steps; a step covers groups; a group is one start of one row played as several episodes; an
episode is one run of a program and has one rollout per agent (model slot); each rollout becomes a trajectory, made
of segments.

### Environment / row

What a run trains on and an eval measures: its program, its rows (easiest first) and how a start of one is drawn, its eval data, its description, its version, and perhaps a curriculum of its own / one situation of it, of which a start is drawn for each group. See [environments](../libraries/rollout-train/rollouts.md#environment).

### Eval data

An environment's named lists of starts (`evals()`), which training never draws (`train_start`); a suite's entry can play one (`rollout suite make`), and a suite of its name is made of it the first time it is played. See [train and eval](../libraries/rollout-train/rollouts.md#train-and-eval).

### Description / version

What an environment's results say: the range of its rewards, whether they say `solved` and `saturated`, what `duration` counts, how its observations are shown, and how much an episode samples, for estimating a run's spend / a name for what an environment is now, changed whenever its rows, starts, eval data or scoring change. Each run's start records both; each version of a suite records the version.

### Published environment

An environment imported from git: its project's source at a commit, kept in the blob store as a version whose id is the source's hash (`NAME@VERSION`), and run in a Ray runtime environment of its own. See [writing an environment others can import](../guide/publishing.md).

### Curriculum

Which row a run trains on next, given its results (and its evals' results, for a gate): the environment's own, or the generic one (`rollout.curriculum`). See [the curriculum](../libraries/rollout-train/training.md#the-curriculum).

### Training run

One training loop (`rollout_train.train`, `rollout train`) over an environment, kept in the ledger: its groups, their results and its steps. See [training](../libraries/rollout-train/training.md).

### Step

One call of the trainer, over the groups queued with something to train on (at least `groups_per_step` of them, except at the end of the run); it makes one checkpoint, from the one the step before made.

### Group

One start of one row, played as several episodes (its record's `episodes`, the algorithm's `group_size`) that are compared with each other. Its episodes are numbered from 1 within it and carry the labels `run`, `group` and `episode`; the rest of it is in its record.

### Episode

One run of a program, as training sees it once it has ended: labels, outcome, result, and a trajectory per model slot. See [episodes](../libraries/rollout-train/episodes.md).

### Rollout

One model slot's part of an episode as it plays: every turn of one agent. Each rollout becomes a trajectory.

### Session

The gateway's record of one rollout: every sample of one model slot of one run.

### Trajectory

What a rollout leaves to train on: its segments and its rewards (`Trajectory`).

### Segment

A piece of a trajectory: tokens that only grew by appending, with the spans the policy sampled, their behaviour logprobs and weights versions (`Segment`, `Span`), what its turns were sampled with, and a teacher's scores where one scored it. An edited context (a compaction, thinking dropped) starts the next.

### Behaviour logprob

The log-probability of a sampled token under the distribution it was sampled from.

### Plan

How a run's episodes are played: its program and its binding, in the run's `plans` table.

### Episode runner

Claims the episodes runs ask for in the ledger, plays them on a runner and records them, at most as many at once as it has places. Several, on one machine or many, share the work. It knows no algorithm. See [rollouts](../libraries/rollout-train/rollouts.md).

### Claim

An episode runner's append under `GROUP/EPISODE/ATTEMPT` in a run's `claims` table: the first append wins, and it holds while the runner keeps the fence it made it under (or adopted it under) and beats. It names the run that plays it. The episode's sandboxes are leased under it, and their leases end with it.

### Heartbeat

What a runner, an engine host, a gateway replica, a pool served on its own, a pod or a run's driver waiting for its resources writes every 15 seconds beside the ledger: its host, its machine's measurements, its engines and channels (a gateway replica's: where it listens and each channel it samples), how full its pools are, or what it waits for. One silent for 90 seconds is taken to be gone. See [heartbeats](../libraries/rollout-train/rollouts.md#heartbeats).

### Launch / job / driver

A run asked for (its kind, name, settings and preset), the run it is and the job it became, with its state / where a run runs: a Ray job, or a RayJob on Kubernetes, running `python -m rollout_train.jobs LAUNCH` / that job's process, which builds the run from the cluster config and its settings, claims its trainer and engine hosts, and runs the loop. See [launching runs](../libraries/rollout-train/launching.md).

### Suite / eval

An eval configuration, by name: a list of entries, one for each environment it plays, each with the environment and its version, its starts (its eval data of a name, each of some rows' start drawn with each of some seeds, or starts given), episodes per start and its sampling limits, each entry scored apart; kept in versions (`NAME@N`), each never changed, an edit making the next, and the name pointing to the newest / one version of a suite played by one checkpoint or the base model, training nothing: a run of its own whose start says `kind: eval` and the version (with a run for each entry, its parts, where it has several), recording each episode's outcome under `evaluations/SUITE/EVAL/results`. A training run can evaluate its own checkpoints on a schedule. See [evals](../libraries/rollout-train/evals.md).

### Dataset

Examples chosen from runs' episodes, made once: examples to imitate, by an episode rule and turn filters, or pairs and labelled examples for a preference loss, by a preference rule; a record in the ledger's `datasets` table and a manifest blob of one line per example, and a name if it is given one. Its supervision says what a step can do with it: `importance`, `supervised`, or `teacher` (examples a teacher scored, for distillation). A step on one makes a checkpoint whose parents after the first sampled its examples. See [datasets](../libraries/rollout-train/datasets.md).

### Bridge

How a checkpoint's files in its trainer's format (`peft`, `full`, `tinker`) become files a provider loads: a named bridge between two formats (`verbatim`, `full-reload`, `peft-from-tinker`, `merge-quantize`, `none`), or a chain of them, each a task named `module:name`, run once per checkpoint and bridge, noted in `checkpoints/resharding` and `checkpoints/resharded` under `CHECKPOINT@BRIDGE`; in the calling process or as a Ray task. See [bridges](../libraries/rollout-train/checkpoints.md#bridges).

### Checkpoint

Weights a step (or a supervised step, or a merge) made: a node of a graph, with an id of its own (shown by its shortest unique start), its parents (what it was trained from, and any others it learned from), its base model, its depth, and the run and step that made it. See [checkpoints](../libraries/rollout-train/checkpoints.md).

### Base model

The model a checkpoint adapts (`Qwen/Qwen3.5-9B`): the root every line of checkpoints grows from.

### Depth

A checkpoint's steps from its base model along its first parents: the number stamped on the tokens it samples.

### Fork

A run started from a checkpoint of another run (or an earlier one of its own): it trains on from there, sharing its parent's files.

### Bookmark

A name for a checkpoint, kept in the registry, moved by hand or carried by a run to each checkpoint it makes. A checkpoint needs none. See [bookmarks](../libraries/rollout-train/checkpoints.md#bookmarks).

### Id / name

What a run is kept under, which never changes / what it is called, which can be chosen and changed, in the registry beside the ledger. See [runs](../libraries/rollout-train/checkpoints.md#runs).

### Ledger

Append-only tables that hold every run's decisions and results and the checkpoints, with fences so that one writer holds each: files, a database (SQLite or Postgres), or a database through the ledger service.

### Ledger service

The ledger and the stores beside it over HTTP (`rollout_train.ledger_service`), for every role that reaches the ledger by URL (`HttpLedger`): a request carries a token, the platform's (which may do everything) or a pod's (which reads its run's serving records and checkpoints and writes its own beat).

### Channel

A model being served under a name in a run (`RUN/NAME`): its providers, its limits, and the checkpoint it samples from. The trained channel serves what the run trains; a `follows` channel serves another channel's checkpoint, `lag` checkpoints back; a `fixed` channel serves one checkpoint or the base model. A channel on a hosted API samples by message, and its turns are never trained on.

### Engine

One replica serving a model: tokens in; tokens, logprobs and a finish reason out; it also scores given tokens. `VllmEngine` is one ([vLLM engine](../implementations/rollout-vllm.md)); `RemoteEngine` is a vLLM server elsewhere, a request naming the checkpoint it samples from as its model.

### What a channel should serve

A run's record, in its `serving` table, that its channel (`RUN/NAME`) serves a checkpoint from then on: its id, depth and kind, and the files its engines load. The training loop writes it each time it serves a checkpoint; whatever serves or samples the channel elsewhere reads it. See [what a channel should serve](../libraries/rollout-train/channels.md#what-a-channel-should-serve).

### Follower / engine host

Keeps a process's channels serving what a run says, loading each checkpoint from the blob store, named by its id / what does that for a replica's engines, and nothing else: an `EngineHost`, a Ray actor that serves every run bound to it, or a follower beside a vLLM server (`InferencePod`). See [deploying](../guide/deploying.md#engines-on-other-machines) and [engine hosts](../libraries/rollout-train/channels.md#engine-hosts).

### Pod

A GPU machine rented by the hour from RunPod for a run: it serves a channel beside a vLLM server, takes the run's training steps, or hosts its sandboxes, and is reached over mutual TLS at the address its lease says. A pod's lease, beside the ledger, says which run holds it; a released pod stays warm for the next run for a while, and the reaper deletes the pods no run holds. See [pods on RunPod](../guide/deploying.md#pods-on-runpod).

### Max lag

How many checkpoints behind what its channel should serve a sample may be, where its server does not have the newest yet (1 unless the run says otherwise; 0 for an eval).

### Gateway

The model endpoint for channels, as a service that keeps no session: verifies a request's signed key, renders contexts to tokens, samples, and records every turn in the ledger and the blob store before it replies; it samples channels on hosted APIs by message. It runs in a run's driver, or as replicas of the cluster's.

### Renderer

The chat template, tokenizer and parser of one model family ([Qwen](../implementations/rollout-qwen.md), [Gemma](../implementations/rollout-gemma.md)).

### Weights version

The depth of the checkpoint a channel serves; every sampled span carries the one it was sampled at (`Span.version`).

### Trainer

Turns a batch (weighted segments, pairs, labelled or distilled segments, as its objective's family takes) into new weights, within a budget it states. `LoraTrainer` is one ([LoRA trainer](../implementations/rollout-lora.md)); `RemoteTrainer` takes its steps on a training pod.

### Objective

What a trainer's loss is: a family (`policy_gradient`, `preference`, `likelihood`, `distillation`), which fixes what a batch item is, and components that compose the loss (`objective.clip.low`, say), as `rollout_train.objectives` declares them; a preset (`objective.preset`) is the literature's objective as a family and component values. A run's start records the objective it resolved to; `rollout_objectives` computes it. See [objectives](../libraries/rollout-train/training.md#objectives).

### Teacher

A channel whose scores of a segment's sampled tokens (their logprobs, and its most likely tokens at each) a distillation trains toward. An objective's routes give each episode one teacher, by its environment or row. See [distillation](../libraries/rollout-train/training.md#distillation).
