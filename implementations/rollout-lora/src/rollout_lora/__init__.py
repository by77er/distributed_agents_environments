"""A trainer for 4-bit checkpoints with LoRA: one implementation of `rollout_train.Trainer`.

- `LoraTrainer`, `LoraSettings`: the trainer and its settings (importing them does not load torch).
- `worker`: each step in a process of its own. `step`: a policy step. `objectives`: the losses it takes, by name.
  `policy`: the checkpoint as a trainable policy. `layers`: LoRA. `quantized`: 4-bit linear layers. (These import
  torch.)
"""

from rollout_lora.settings import LoraSettings
from rollout_lora.trainer import LoraTrainer

__all__ = ["LoraSettings", "LoraTrainer"]
