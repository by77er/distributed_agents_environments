"""Inference: the engines that serve a policy, and the channel that names it (docs/libraries/rollout-train/channels.md).

- `Channel`: a policy as the rest of the system knows it. A name, the engines serving it, its model family's
  renderer, the adapter in use and the version it is served as (its checkpoint's depth). Requests go through it; new
  weights are published to it.
- `Engine`: tokens in; tokens, logprobs and a finish reason out. `VllmEngine` runs vLLM in this process's care;
  `RemoteEngine` is a replica served on another machine (`remote`).
- `remote`: a process's engines served over HTTP (`serve_engines`), and `RemoteChannel`, one run's channel routed to
  the replicas that serve it elsewhere, as a runner samples it.
"""

from rollout_train.inference.channel import Channel, Engine, Generation, Limits, Sampler, Unserved
from rollout_train.inference.remote import (
    Connection,
    RemoteChannel,
    RemoteEngine,
    Replica,
    Route,
    Routes,
    replica_id,
    serve_engines,
)

__all__ = [
    "Channel",
    "Connection",
    "Engine",
    "Generation",
    "Limits",
    "RemoteChannel",
    "RemoteEngine",
    "Replica",
    "Route",
    "Routes",
    "Sampler",
    "Unserved",
    "replica_id",
    "serve_engines",
]
