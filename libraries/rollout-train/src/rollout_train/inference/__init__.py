"""Inference: the engines that serve a policy, and the channel that names it (docs/inference).

- `Channel`: a policy as the rest of the system knows it. A name, the engines serving it, its model family's
  renderer, the adapter in use and its version. Requests go through it; new weights are published to it.
- `Engine`: tokens in; tokens, logprobs and a finish reason out. `VllmEngine` runs vLLM in this process's care;
  an engine on another machine is another implementation of the same protocol.
"""

from rollout_train.inference.channel import Channel, Engine, Generation, Limits

__all__ = ["Channel", "Engine", "Generation", "Limits"]
