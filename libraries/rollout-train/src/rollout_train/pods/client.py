"""What leasing asks of a pod provider's API: a pod as it is asked for (`PodSpec`), a pod as the provider says it is
(`Pod`), and the client that asks for, lists, reads and deletes pods (`PodClient`).

A provider's table names its client as `module:name` (`client`, by default `rollout_runpod:RunPod`:
`rollout_train.providers.PodTable.client`), which is called
with `key_env`, the environment variable its API key is read from (`client_of`). The client reads the key when it makes
a request, and never keeps it, writes it down or puts it in an error.

A pod's environment is given in three parts: values (`PodSpec.env`); references to secrets the provider keeps
(`PodSpec.secrets`, which the provider fills in on the pod); and values that are sent and never logged
(`PodSpec.sensitive`: a one-time token minted for this pod alone), never in the pod's `repr`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rollout.names import named

__all__ = ["Pod", "PodClient", "PodSpec", "client_of"]


@dataclass(frozen=True)
class PodSpec:
    """A pod, as it is asked for."""

    name: str
    """The pod's name, which its certificate's identity is made from (`spiffe://rollout/pod/NAME`)."""
    image: str
    gpu_types: Sequence[str]
    """The provider's GPU type ids, in order of preference (`NVIDIA GeForce RTX 4090`, `NVIDIA H100 80GB HBM3`)."""
    gpu_count: int = 1
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    secrets: Mapping[str, str] = field(default_factory=dict[str, str])
    """Variables whose values are secrets the provider keeps (RunPod's console), by the secret's name."""
    sensitive: Mapping[str, str] = field(default_factory=dict[str, str], repr=False)
    """Variables whose values are sent and never logged."""
    ports: Sequence[str] = ("8443/tcp",)
    """`PORT/tcp` (a public port mapped to it; an HTTPS proxy of the provider's would end TLS, so none is `/http`)."""
    volume_gb: int = 50
    volume_mount: str = "/workspace"
    container_disk_gb: int = 50
    cloud: str = "SECURE"
    """`SECURE` or `COMMUNITY`."""
    data_centers: Sequence[str] = ()
    cuda_versions: Sequence[str] = ()
    """The CUDA versions the pod's machine may support (none: any); its driver must run the image's CUDA."""
    interruptible: bool = False
    min_vcpus_per_gpu: int | None = None
    """The fewest vCPUs the pod may be given for each GPU (none: the provider's default)."""
    min_memory_gb_per_gpu: int | None = None
    """The least memory, in GB, the pod may be given for each GPU (none: the provider's default)."""


@dataclass(frozen=True)
class Pod:
    """A pod, as the provider says it is."""

    id: str
    name: str
    status: str
    """`RUNNING`, `EXITED` (stopped) or `TERMINATED`."""
    image: str = ""
    public_ip: str | None = None
    ports: Mapping[int, int] = field(default_factory=dict[int, int])
    """Each exposed port of the pod, and the public port it is reached at."""
    cost_per_hour: float | None = None
    gpu: str | None = None
    """The GPU type the provider gave it, by its id (`NVIDIA H100 80GB HBM3`), where it says."""
    vcpus: int | None = None
    """The vCPUs the provider gave it, where it says."""
    memory_gb: float | None = None
    """The memory the provider gave it, in GB, where it says."""

    def address(self, port: int = 8443) -> str | None:
        """Where `port` is reached from outside, `https://IP:PORT`; None until the provider has said."""
        public = self.ports.get(port)
        return f"https://{self.public_ip}:{public}" if self.public_ip and public else None


class PodClient(Protocol):
    """A provider's pods API, as leasing uses it. How many pods a cluster may have, and when an idle pod is deleted,
    are leasing's rules: the client does what it is asked."""

    async def create(self, spec: PodSpec) -> Pod:
        """Ask for a pod; it starts as soon as the provider has a machine for it."""
        ...

    async def pods(self, *, name: str | None = None) -> list[Pod]:
        """Every pod of the account (those named `name`, if given)."""
        ...

    async def pod(self, id: str) -> Pod: ...

    async def terminate(self, id: str) -> None:
        """Delete a pod and its volume."""
        ...

    async def aclose(self) -> None: ...


def client_of(client: str, key_env: str) -> PodClient:
    """The client `client` (`module:name`) names, with its API key in the environment variable `key_env`."""
    return named(client)(key_env=key_env)
