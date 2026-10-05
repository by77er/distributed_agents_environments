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

## Hosted models

A model behind OpenAI's or Anthropic's API is evaluated as any base model is: the cluster config declares the hosted
API as an inference provider of the kind `api` ([hosted APIs](../guide/cluster.md#hosted-apis)), with each model's
prices, and an eval plays a suite on a channel of it (`channels.policy.provider = "anthropic"`, a model, no renderer).
The eval forms offer its models among the base models, say the provider is metered with the eval's estimated spend,
and take a limit (`limits.spend`), which ends the eval once it spends that ([an eval of a hosted
model](../libraries/rollout-train/evals.md#an-eval)). Its turns are recorded with what each cost, and never trained on.
The same providers serve the slots of a training run that are not trained, such as a [judge](../products/judging.md).

## Related pages

- [Judging](../products/judging.md): open-ended answers scored by a judge, a model slot that is not trained,
  against a versioned rubric.
- [verifiers environments](../implementations/rollout-verifiers.md): Prime Intellect's environments, such as GSM8K,
  played and evaluated here.
- [Curricula and evaluation suites](../research/curricula.md): a design note on building frozen, stratified suites
  from a run's data.
- [Start runs and evals](../deploy/runs.md): where starting an eval on a deployed platform is described.
