"""GPU pods on RunPod, and certificates for them: what starting, stopping and trusting pods that serve a channel or
take a run's training steps takes (docs/research/runpod-providers.md).

- `api`: `RunPod`, a client of RunPod's pods API (create, start, stop, terminate, list), its key read from
  `RUNPOD_API_KEY` and never written down. `PodSpec`, a pod as it is asked for; `Pod`, as RunPod says it is.
- `certificates`: `StepCa`, the cluster's certificate authority (step-ca): a one-time token that
  lets one pod get its first certificate, and the revocation of a pod's certificate, after which it is not renewed.

What runs on the pods (the follower beside vLLM, the training service) is `rollout_train.pods`; the images are
`deploy/images`.
"""

from rollout_runpod.api import Pod, PodSpec, RunPod, RunPodError
from rollout_runpod.certificates import StepCa, fingerprint

__all__ = ["Pod", "PodSpec", "RunPod", "RunPodError", "StepCa", "fingerprint"]
