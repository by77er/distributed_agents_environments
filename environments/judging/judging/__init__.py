"""Open-ended requests with no exact answer, scored by a judge: an environment whose reward comes from a model slot that
is not trained.

- `environment`: the rows, the training and eval requests, and how a start is drawn
  (`rollout env check judging.environment:environment`).
- `episode`: the episode program: the policy answers, the judge scores the answer, the reward is the score.
- `rubric`: the versioned rubrics, the judge's instructions, and the strict reading of its JSON verdict.
- `data`: the bundled concepts and passages, split into what training draws and what only the eval asks.
"""
