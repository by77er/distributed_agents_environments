"""Training and sampling at Thinking Machines (Tinker): implementations of `rollout_train.Trainer` and of `Engine`.

- `TinkerTrainer`: LoRA steps on Tinker, the objective (`rollout_objectives`) sent as Tinker's losses; a version's
  files point at its Tinker checkpoints. `TinkerSettings`: its settings.
- `TinkerEngine`: token-in sampling at Tinker, of the base model or of a version's sampler checkpoint.
- `service`: what they ask of Tinker's SDK, and the session they open by default (the SDK finds the key).
  `testing`: a fake service for tests. `weights`: pointers, and adapters in PEFT's layout. `bridges`: a checkpoint's
  adapter downloaded and written in PEFT's layout, as a bridge's task. `data`: segments as Tinker's `Datum`s.
"""

from rollout_tinker.engine import TinkerEngine
from rollout_tinker.settings import TinkerSettings
from rollout_tinker.trainer import TinkerTrainer

__all__ = ["TinkerEngine", "TinkerSettings", "TinkerTrainer"]
