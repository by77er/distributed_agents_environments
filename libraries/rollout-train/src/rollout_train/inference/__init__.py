"""Inference: the engines that serve a policy, and the channel that names it (docs/libraries/rollout-train/channels.md).

- `Channel`: a policy as the rest of the system knows it. A name, the engines serving it, its model family's
  renderer, the adapter in use and the version it is served as (its checkpoint's depth). Requests go through it; new
  weights are published to it.
- `Engine`: tokens in; tokens, logprobs and a finish reason out, with the most likely tokens at each position when
  asked. It also scores given tokens (`Scores`: each one's logprob and the most likely tokens there), sampling
  nothing. `VllmEngine` runs vLLM in this process's care; `RemoteEngine` is a vLLM server elsewhere, over its
  OpenAI-compatible API (`remote`).
- `remote`: `RemoteChannel`, one run's channel sampled on servers elsewhere as the gateway samples it, each request
  naming the checkpoint it samples from; the servers are `CheckpointServer`s.
- `hosts`: `EngineHost`, one replica's engines as a Ray actor that follows what the runs bound to it serve, and
  `HostServer`, a `CheckpointServer` over its handle.
"""

from rollout_train.inference.channel import Channel, Engine, Generation, Limits, NotLoaded, Sampler, Scores, Unserved
from rollout_train.inference.remote import CheckpointServer, Connection, RemoteChannel, RemoteEngine, Route, Routes

__all__ = [
    "Channel",
    "CheckpointServer",
    "Connection",
    "Engine",
    "Generation",
    "Limits",
    "NotLoaded",
    "RemoteChannel",
    "RemoteEngine",
    "Route",
    "Routes",
    "Sampler",
    "Scores",
    "Unserved",
]
