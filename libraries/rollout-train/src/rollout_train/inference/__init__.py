"""Inference: the engines that serve a policy, and the channel that names it (docs/libraries/rollout-train/channels.md).

- `Channel`: a policy as the rest of the system knows it. A name, the engines serving it, its model family's
  renderer, the adapter in use and the version it is served as (its checkpoint's depth). Requests go through it; new
  weights are published to it.
- `Engine`: tokens in; tokens, logprobs and a finish reason out. `VllmEngine` runs vLLM in this process's care;
  `RemoteEngine` is a vLLM server elsewhere, over its OpenAI-compatible API (`remote`).
- `remote`: `RemoteChannel`, one run's channel sampled on servers elsewhere as a runner samples it, each request naming
  the checkpoint it samples from.
"""

from rollout_train.inference.channel import Channel, Engine, Generation, Limits, Sampler, Unserved
from rollout_train.inference.remote import Connection, RemoteChannel, RemoteEngine, Route, Routes

__all__ = [
    "Channel",
    "Connection",
    "Engine",
    "Generation",
    "Limits",
    "RemoteChannel",
    "RemoteEngine",
    "Route",
    "Routes",
    "Sampler",
    "Unserved",
]
