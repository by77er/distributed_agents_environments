"""Pods: GPU machines elsewhere, rented by the hour, that serve a run's channel or take its training steps.

A pod is reached at a public address over mutual TLS. Its certificate names it (`spiffe://rollout/pod/NAME`), and it
takes requests only from a client whose certificate names the gateway (`spiffe://rollout/gateway`). Both come from the
cluster's certificate authority (step-ca). A proxy in front of the pod's own servers (Envoy) checks the client and lets
through only the requests the pod's role needs; the servers behind it listen on the pod's loopback interface.

- `identity`: the names certificates carry, and the pods whose heartbeats say they are alive and where (`live`).
- `leases`: each pod's lease (which run holds it), and the time each run held a pod, charged at its price.
- `leasing`: a run's pods, claimed (started, or taken warm), renewed and released (`Pods`); the reaper (`reap`).
- `inference`: what runs beside a stock vLLM server on an inference pod: a follower that loads what the channel of the
  run that holds the pod should serve, beats, and says when the pod is ready (`InferencePod`).
- `training`: the training service on a training pod: one step at a time, each idempotent by the checkpoint it makes
  (`TrainerService`).
- `trainer`: `RemoteTrainer`, a `Trainer` over that service.

The images that run these, and how the certificates are issued and renewed, are in `deploy/images`
(docs/research/runpod-providers.md).
"""

from rollout_train.pods.identity import GATEWAY_IDENTITY, PodAddress, live, pod_identity
from rollout_train.pods.leases import HELD, IDLE, STARTING, PodLease, PodLeases, PodTime, pod_leases_of
from rollout_train.pods.leasing import LeaseLost, PodNeed, Pods, PodsDidNotStart, needs_of, reap
from rollout_train.pods.trainer import RemoteTrainer, TrainerBusy, TrainerRefused, TrainerUnreachable

__all__ = [
    "GATEWAY_IDENTITY",
    "HELD",
    "IDLE",
    "STARTING",
    "LeaseLost",
    "PodAddress",
    "PodLease",
    "PodLeases",
    "PodNeed",
    "PodTime",
    "Pods",
    "PodsDidNotStart",
    "RemoteTrainer",
    "TrainerBusy",
    "TrainerRefused",
    "TrainerUnreachable",
    "live",
    "needs_of",
    "pod_identity",
    "pod_leases_of",
    "reap",
]
