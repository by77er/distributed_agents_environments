"""Objectives in torch: what `rollout_train.objectives` declares, computed, for any trainer.

- `settings`: `StepSettings`, a step's settings, which the LoRA, full-weight and Tinker trainers take alike (importing
  it does not load torch).
- `terms`: an objective's loss, composed from its components: a policy gradient's, a likelihood's, a preference loss
  of pairs or labelled examples; and what a minibatch's terms add up to (`SUMS`, `tally`).
- `distillation`: a distillation's loss of a segment a teacher scored, in its policy-gradient or top-k form, and a
  policy gradient's distillation term.
- `packing`: segments laid out in one row of a model's input (`Pack`), a prefix several share once, and segments cut
  into packs (`packs`).
- `step`: a step over a batch on a local policy (`PolicyStep`), the plan of minibatches any trainer shares (`Plan`),
  the step's statistics (`metrics`, `line`), and how far a step has got as it goes (`StepProgress`).
- `ranks`: the processes a step is shared among, one per GPU (`Ranks`), and a pass's packs shared out among them
  (`shares`).
"""
