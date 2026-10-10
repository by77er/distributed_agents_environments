# Train a model

This section is for people who design and run training: how episodes are played and recorded, how the training loop
turns them into checkpoints, which objectives it can optimise, and how checkpoints are kept, served and named.

**Read first:** [Start here](../start/README.md) and [Three ways in](../guide/perspectives.md#designing-training).
**Next:** [Episode runners](../libraries/rollout-train/rollouts.md).

## Read in this order

1. [Episode runners](../libraries/rollout-train/rollouts.md): how a run asks for episodes in the ledger, and how
   runners claim, play and record them; environments, and their train and eval data.
2. [Episodes](../libraries/rollout-train/episodes.md): what a finished episode carries for training: labels, outcome,
   result, and one trajectory per model slot.
3. [The training loop and objectives](../libraries/rollout-train/training.md): the loop, its steps over groups, the
   objectives and their presets, the curriculum, the trainer protocol, changing settings while a run goes, pausing
   and resuming.
4. [Launching runs](../libraries/rollout-train/launching.md): how a run is asked for with its settings, the job it
   becomes (a Ray job or a RayJob), what its driver starts and claims, resuming by run id, and the `rollout` commands
   that ask for runs.
5. [Checkpoints, runs and the ledger](../libraries/rollout-train/checkpoints.md): the checkpoint graph, forks, bridges
   between checkpoint formats, the ledger and its fences, bookmarks.
6. [Datasets and supervised steps](../libraries/rollout-train/datasets.md): examples chosen from runs' episodes, and
   supervised fine-tuning (SFT) on them.
7. [Channels and engines](../libraries/rollout-train/channels.md): how a channel serves what a run trains, engine
   hosts, engines on other machines, and sharing a GPU with the trainer.
8. [Record turns for training](../libraries/rollout-train/recorder.md): what a recorded sample keeps, the thinking
   budget, segments, renderers.
9. [The gateway](../libraries/rollout-train/gateway.md): the service that samples channels and records every turn,
   its keys, and running Claude Code or Codex against it.
10. [Train any harness over HTTP](../libraries/rollout-train/harness-endpoint.md): the gateway's OpenAI and Anthropic
    APIs for a harness that brings its own loop.

## Choose a trainer and an objective

- The trainers are the [LoRA and full-weight trainers](../implementations/rollout-lora.md) on your own GPU or on a
  pod rented from RunPod ([GPU pods on RunPod](../deploy/providers.md#gpu-pods-on-runpod)), and
  [Tinker](../implementations/rollout-tinker.md) at Thinking Machines. Their capabilities are compared in
  [the cluster config](../guide/cluster.md#trainers).
- A run's objective is a preset (`default`, `grpo`, `dapo`, `sft`, `dpo`, `distillation` and others) with components you
  can change. The presets are listed in [run settings](../guide/cluster.md#run-settings); what each component does is in
  [objectives](../libraries/rollout-train/training.md#objectives), and how the local trainers compute it is in
  [objectives in torch](../implementations/rollout-objectives.md).

## Start a run

On a deployed platform, a run is started from the monitor's New run form or from the command line:
[Start runs and evals](../deploy/runs.md) says where each is described. To try the loop on a laptop without a GPU,
see [trying it without a GPU](../libraries/rollout-train/training.md#trying-it-without-a-gpu).
