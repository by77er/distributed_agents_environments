# Reference

This section is for looking things up once you know what you need: every public name, the command line, the run
settings and cluster config, the objective presets, and the glossary.

**Read first:** [Start here](../start/README.md), for the ideas these pages assume.

## The pages

- [API reference](../guide/reference.md): every public name, by module, with signatures, fields and docstrings,
  generated from the source.
- [Cluster config and run settings](../guide/cluster.md): every section of the cluster config, the kinds of inference
  provider and trainer and what each can do, the bridges between checkpoint formats, every run setting, presets, and
  the rules that check a run.
- [Glossary](../architecture/glossary.md): every term, defined once.

## The command line

The `rollout` command is installed with the workspace (`uv sync`). `uv run rollout --help` lists its commands, and
`uv run rollout COMMAND --help` lists each command's options. These pages describe them:

<!-- Follow-up: once the command line has a page of its own, link it here and keep this list to that link. -->

| Commands | Described in |
|---|---|
| `rollout env check` | [checking an environment](../libraries/rollout-train/rollouts.md#checking-an-environment) |
| `rollout suite`, `rollout eval` | [suites and evals](../libraries/rollout-train/evals.md) |
| `rollout dataset`, `rollout imitate` | [datasets and supervised steps](../libraries/rollout-train/datasets.md) |
| `rollout checkpoints`, `rollout bookmark`, `rollout rename`, `rollout merge` | [checkpoints, runs and the ledger](../libraries/rollout-train/checkpoints.md#the-command-line) |
| `rollout pause`, `rollout resume` | [pausing and resuming](../libraries/rollout-train/training.md#pausing-and-resuming) |
| `rollout monitor` | [the monitor](../libraries/rollout-train/monitor.md) |
| `rollout gateway` | [the gateway](../libraries/rollout-train/gateway.md#running-it) |
| `rollout pool`, `rollout tools` | [sandboxes over HTTP](../libraries/rollout/sandboxes.md#over-http), [serving a tool set over HTTP](../guide/tools.md#serving-a-tool-set-over-http) |
| `rollout cluster check`, `rollout preset` | [the cluster config](../guide/cluster.md) |
| `rollout ledger copy` | [Postgres and S3](../deploy/stores.md#move-an-existing-ledger-and-blob-store) |
| `rollout train` and the commands that start runs | [Start runs and evals](../deploy/runs.md) |

## Objective presets

A run's objective is chosen by `objective.preset`: `default`, `reinforce`, `rloo`, `ppo_clip`, `grpo`, `dr_grpo`,
`dapo`, `gspo`, `cispo`, `sft`, `dpo`, `ipo`, `simpo`, `kto` or `orpo`. Each sets the components of one family
(policy gradient, likelihood or preference), and each component can be changed with an `objective.COMPONENT` setting.

- [Objectives](../libraries/rollout-train/training.md#objectives): the families, their components and the presets.
- [Run settings](../guide/cluster.md#run-settings): the `objective.*` keys, and which are changeable while a run goes.
- [Objectives in torch](../implementations/rollout-objectives.md): how the local trainers compute each loss.
- [Objectives design](../research/objectives-design.md): the design note, with what is built and what is proposed.

## Configuration files in the repository

- `deploy/clusters/example.toml`: a cluster config for one machine with one 16 GB GPU, described in
  [the cluster config](../guide/cluster.md#the-cluster-config).
- `deploy/chart/rollout/values.yaml`: the Helm chart's values, described in
  [Install the Helm chart](../deploy/helm.md#the-charts-values).
- `deploy/local/compose.yaml`: Postgres and an S3-compatible store for development, described in
  [local Postgres and S3](../development/local-services.md).
