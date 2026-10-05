# Evaluate a model

This section is for people who measure models: how to define a suite of environments to evaluate on, how to play it
with any checkpoint or a base model, and how to run evals on a schedule while a model trains.

**Read first:** [Start here](../start/README.md). **Next:** [Suites and evals](../libraries/rollout-train/evals.md).

## What to read

1. [Suites and evals](../libraries/rollout-train/evals.md): suites and their versions, an eval of one checkpoint,
   suites made and edited from the monitor, and evals during training.
2. [Environments' train and eval data](../libraries/rollout-train/rollouts.md#train-and-eval): the eval data an
   environment declares, which training never draws.
3. [Scores along a line](../libraries/rollout-train/monitor.md#scores-along-a-line) and
   [a subject's history](../libraries/rollout-train/monitor.md#a-subjects-history): how the monitor shows a
   checkpoint's evals and the scores along a line of checkpoints.

## Related pages

- [Judging](../products/judging.md): open-ended answers scored by a judge, a model slot that is not trained,
  against a versioned rubric.
- [verifiers environments](../implementations/rollout-verifiers.md): Prime Intellect's environments, such as GSM8K,
  played and evaluated here.
- [Curricula and evaluation suites](../research/curricula.md): a design note on building frozen, stratified suites
  from a run's data.
- [Start runs and evals](../deploy/runs.md): where starting an eval on a deployed platform is described.
