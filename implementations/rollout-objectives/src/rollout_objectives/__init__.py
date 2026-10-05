"""Objectives in torch: what `rollout_train.objectives` declares, computed, for any trainer.

- `settings`: `StepSettings`, a step's settings, which the LoRA, full-weight and Tinker trainers take alike (importing
  it does not load torch).
- `terms`: an objective's loss, composed from its components: a policy gradient's, a likelihood's, a preference loss
  of pairs or labelled examples; and what a minibatch's terms add up to (`SUMS`, `tally`).
- `distillation`: a distillation's loss of a segment a teacher scored, in its policy-gradient or top-k form, and a
  policy gradient's distillation term.
- `step`: a step over a batch on a local policy (`PolicyStep`), the plan of minibatches any trainer shares (`Plan`),
  and the step's statistics (`metrics`, `line`).
"""
