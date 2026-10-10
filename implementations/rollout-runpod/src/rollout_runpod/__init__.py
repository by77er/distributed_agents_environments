"""GPU pods on RunPod: what starting, stopping and deleting pods that serve a channel, take a run's training
steps or serve its sandboxes takes (docs/research/runpod-providers.md).

- `api`: `RunPod`, a client of RunPod's pods API (create, start, stop, terminate, list), its key read from
  `RUNPOD_API_KEY` and never written down: the client a RunPod provider's table names by default
  (`client = "rollout_runpod:RunPod"`). It asks for a pod as a `PodSpec` and says one as a `Pod`, the types of
  `rollout_train.pods.client`; `body` and `pod_of` translate them to and from RunPod's JSON.

What runs on the pods (the follower beside vLLM, the training service, the sandbox host) and the certificates they
get from the cluster's step-ca are `rollout_train.pods`; the images are `deploy/images`.
"""

from rollout_runpod.api import RunPod, RunPodError, body, pod_of

__all__ = ["RunPod", "RunPodError", "body", "pod_of"]
