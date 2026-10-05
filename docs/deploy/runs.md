# Start runs and evals

What to do once the platform is up: check it, import an environment, and start a training run or an eval. This page
is for whoever uses a freshly deployed platform; it points to the pages that describe each step.

**Read first:** [Install the Helm chart](helm.md) or [Choose a setup](setups.md). **Next:**
[Operate the platform](../operate/README.md).

## Check the platform

1. Every secret and project the cluster config names resolves:

    ```bash
    kubectl -n rollout exec deploy/gateway -- rollout cluster check
    ```

2. The monitor's page opens at its host, and its Machines page lists the
   [gateway](../libraries/rollout-train/gateway.md)'s replicas ([the
   machines](../libraries/rollout-train/monitor.md#the-machines)).

## Offer an environment

A run plays an environment the cluster offers:

- an environment built into the image, declared in the cluster config's `[environments]`;
- or one imported from a git repository from the monitor's Environments page, with nothing redeployed
  ([import an environment from git](../guide/publishing.md)).

## Start a run or an eval

A run is asked for with its [run settings](../guide/cluster.md#run-settings), usually starting from a
[preset](../guide/cluster.md#presets): the chart saves the presets in `deploy/chart/rollout/files/presets` beside the
ledger at every install and upgrade (`rollout preset load`), and `rollout preset list --cluster` lists them. Asking
checks the settings against the cluster config, records a launch and submits the run's job: a RayJob made from the
chart's `files/rayjob.yaml` on Kubernetes, a Ray job on one machine
([launching runs](../libraries/rollout-train/launching.md)).

- **From the monitor:** the **New run** form (a preset, the environment, the name, the settings to change), and an
  eval form on each suite's, checkpoint's and base model's page. Both post to `POST /api/launches`; a refusal is shown
  beside the setting it is about ([launching a run](../libraries/rollout-train/monitor.md#launching-a-run)). The
  monitor makes each RayJob with its own service account, which `templates/rbac.yaml` allows to create, read and delete
  RayJobs.
- **From the command line**, with the cluster config:

    ```bash
    rollout train minecraft_team.environment:environment --preset minecraft-one-gpu --name team-8 --cluster
    rollout eval math --checkpoint team-8:12 --cluster
    rollout train gridworld.environment:environment --preset gridworld-qwen3-0.6b --check --cluster   # check only
    ```

    Each command builds the settings from `--preset`, `--settings` and `--set`, says each refusal with its setting,
    and follows the run until it ends unless `--detach` ([the command line](../guide/deploying.md#asking-for-a-run)).

- **A training run:** [the training loop](../libraries/rollout-train/training.md) describes what a run does, and
  [run settings](../guide/cluster.md#run-settings) every setting it takes: the environment, the trainer, the provider
  of each [channel](../libraries/rollout-train/channels.md), the objective, and the numbers each takes.
- **An eval:** [suites and evals](../libraries/rollout-train/evals.md) describes making a suite and playing it with a
  [checkpoint](../libraries/rollout-train/checkpoints.md) or a base model.
- **Watching it:** [the monitor](../libraries/rollout-train/monitor.md) shows each run's steps,
  [episodes](../libraries/rollout-train/episodes.md), checkpoints and evals as they happen.

A run that waits for resources says why in the monitor: see
[a run waits for resources](troubleshooting.md#a-run-waits-for-resources).
