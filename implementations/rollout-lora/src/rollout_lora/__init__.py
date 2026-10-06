"""Trainers on PyTorch: implementations of `rollout_train.Trainer`.

- `LoraTrainer`: a LoRA adapter over a model (a 4-bit image-text checkpoint, or a text model in bfloat16).
  `FullTrainer`: every weight of a text model. `LoraSettings`: their settings, a policy step's (`StepSettings`, which
  the Tinker trainer takes too: `rollout_objectives.settings`), the adapter's scaling and the full-weight trainer's
  reference (importing these does not load torch).
- `merge`: an adapter folded into the weights it was trained over, as a model of its own.
- `worker`: on one GPU, each step in a process of its own, a step of `rollout_objectives.step` (which computes the
  objective). `resident`: on several GPUs, a process per GPU kept between steps (`workers`, under torchrun), the
  policy and optimizer sharded over them with FSDP2 (`sharded`). `policy`: the checkpoint as a trainable policy, with
  its reference (the adapter switched off). `full`: every weight as one. `layers`: LoRA. `quantized`: 4-bit linear
  layers. (These import torch, but for `resident`, which imports it only to count GPUs.)
"""

from rollout_lora.settings import LoraSettings
from rollout_lora.trainer import FullTrainer, LoraTrainer

__all__ = ["FullTrainer", "LoraSettings", "LoraTrainer"]
