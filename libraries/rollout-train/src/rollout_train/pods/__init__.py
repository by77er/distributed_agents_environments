"""Pods: GPU machines elsewhere, rented by the hour, that serve a run's channel or take its training steps.

A pod is reached over mutual TLS at the public address RunPod's API gives for it, which its lease keeps. Its certificate
names it (`spiffe://rollout/pod/NAME`), and it takes requests only from a client whose certificate names the gateway
(`spiffe://rollout/gateway`). Both come from the cluster's certificate authority (step-ca). A proxy in front of the
pod's own servers (Envoy) checks the client and lets through only the requests the pod's role needs; the servers behind
it listen on the pod's loopback interface.

- `identity`: the names certificates carry, and the pods whose heartbeats say they are alive (`live`).
- `certificates`: the cluster's step-ca (`StepCa`): one-time tokens for pods' first certificates, revoking a pod's
  certificate, and certificates for the platform's own clients; `decrypted_key`, a provisioner's key from step-ca's
  configuration.
- `client`: what leasing asks of a provider's pods API: a pod as it is asked for (`PodSpec`) and as the provider says
  it is (`Pod`), and the client (`PodClient`) a provider's table names as `module:name` (`rollout_runpod:RunPod`).
- `leases`: each pod's lease (which run holds it, and where the pod is reached), and the time each run held a pod,
  charged at its price.
- `leasing`: a run's pods, claimed (started, or taken warm), renewed and released (`Pods`); the reaper (`reap`).
- `inference`: what runs beside a stock vLLM server on an inference pod: a follower that loads what the channel of the
  run that holds the pod should serve, beats, and says when the pod is ready (`InferencePod`).
- `training`: the training service on a training pod: one step at a time, each idempotent by the checkpoint it makes
  (`TrainerService`).
- `trainer`: `RemoteTrainer`, a `Trainer` over that service.
- `sources`: what a pod needs to serve a kind of sandbox, as data its lease gives it (`SandboxSource`).
- `sandboxes`: a host pod's sandbox host: each kind its lease asks for, in a Python environment made on the pod and a
  process of its own, following the lease (`SandboxHost`).
- `pools`: the pools of a kind a run's pods serve, as one pool in the run's driver (`PodPools`), found and reached by
  `routing` (`LeasedPools`), as the run's engines are (`LeasedServers`).

The images that run these, and how the certificates are issued and renewed, are in `deploy/images`
(docs/research/runpod-providers.md).
"""

from rollout_train.pods.identity import GATEWAY_IDENTITY, LivePod, live, pod_identity
from rollout_train.pods.leases import HELD, IDLE, STARTING, PodLease, PodLeases, PodTime, pod_leases_of
from rollout_train.pods.leasing import LeaseLost, PodNeed, Pods, PodsDidNotStart, needs_of, reap
from rollout_train.pods.trainer import LeasedTrainer, RemoteTrainer, TrainerBusy, TrainerRefused, TrainerUnreachable

__all__ = [
    "GATEWAY_IDENTITY",
    "HELD",
    "IDLE",
    "STARTING",
    "LeaseLost",
    "LeasedTrainer",
    "LivePod",
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
