"""RunPod's pods API (`https://rest.runpod.io/v1`): create, start, stop and terminate a pod, list pods, and read a
pod's public address.

The API key is read from an environment variable (`RUNPOD_API_KEY`) when a request is made, sent as a bearer token,
and never kept on the client, written down, logged or put in an error. Every request says who sends it
(`User-Agent: rollout/VERSION`): RunPod's front refuses a request without a User-Agent it accepts (403). A pod's
environment is given in two parts:
values (`PodSpec.env`), and references to secrets kept in RunPod's console (`PodSpec.secrets`: a variable's value is
then `{{ RUNPOD_SECRET_name }}`, which RunPod fills in on the pod). A value that must not be logged but is not a
console secret (a one-time token minted for this pod alone, which RunPod's API cannot store as a secret) goes in
`PodSpec.sensitive`: sent, never logged, never in the pod's `repr`.

How many pods a cluster may have, and when an idle pod is stopped, are the caller's rules: this client does what it is
asked.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

API = "https://rest.runpod.io/v1"
KEY = "RUNPOD_API_KEY"
"""The environment variable the API key is read from, unless the client is told another."""


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("rollout-runpod")
    except PackageNotFoundError:
        return "0"


USER_AGENT = f"rollout/{_version()}"
"""What every request says sent it."""

log = logging.getLogger(__name__)


class RunPodError(Exception):
    """RunPod refused a request, or could not be reached. Says the request, the status and RunPod's message: never
    the key."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PodSpec:
    """A pod, as it is asked for."""

    name: str
    """The pod's name, which its certificate's identity is made from (`spiffe://rollout/pod/NAME`)."""
    image: str
    gpu_types: Sequence[str]
    """RunPod's GPU type ids, in order of preference (`NVIDIA GeForce RTX 4090`, `NVIDIA H100 80GB HBM3`)."""
    gpu_count: int = 1
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    secrets: Mapping[str, str] = field(default_factory=dict[str, str])
    """Variables whose values are secrets kept in RunPod's console, by the secret's name."""
    sensitive: Mapping[str, str] = field(default_factory=dict[str, str], repr=False)
    """Variables whose values are sent and never logged."""
    ports: Sequence[str] = ("8443/tcp",)
    """`PORT/tcp` (a public port mapped to it; RunPod's HTTPS proxy would end TLS, so none is `/http`)."""
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
    """The fewest vCPUs the pod may be given for each GPU (RunPod's `minVCPUPerGPU`; none: RunPod's default, 2)."""
    min_memory_gb_per_gpu: int | None = None
    """The least memory, in GB, the pod may be given for each GPU (RunPod's `minRAMPerGPU`; none: RunPod's default,
    8)."""

    def body(self) -> dict[str, Any]:
        """The pod as RunPod's API takes it."""
        env = {**self.env, **{name: f"{{{{ RUNPOD_SECRET_{secret} }}}}" for name, secret in self.secrets.items()}}
        body: dict[str, Any] = {
            "name": self.name, "imageName": self.image, "computeType": "GPU", "gpuTypeIds": list(self.gpu_types),
            "gpuCount": self.gpu_count, "env": {**env, **self.sensitive}, "ports": list(self.ports),
            "volumeInGb": self.volume_gb, "volumeMountPath": self.volume_mount,
            "containerDiskInGb": self.container_disk_gb, "cloudType": self.cloud, "interruptible": self.interruptible,
            "supportPublicIp": True,
        }  # fmt: skip
        if self.data_centers:
            body["dataCenterIds"] = list(self.data_centers)
        if self.cuda_versions:
            body["allowedCudaVersions"] = list(self.cuda_versions)
        if self.min_vcpus_per_gpu is not None:
            body["minVCPUPerGPU"] = self.min_vcpus_per_gpu
        if self.min_memory_gb_per_gpu is not None:
            body["minRAMPerGPU"] = self.min_memory_gb_per_gpu
        return body


@dataclass(frozen=True)
class Pod:
    """A pod, as RunPod says it is."""

    id: str
    name: str
    status: str
    """RunPod's `desiredStatus`: `RUNNING`, `EXITED` (stopped) or `TERMINATED`."""
    image: str = ""
    public_ip: str | None = None
    ports: Mapping[int, int] = field(default_factory=dict[int, int])
    """Each exposed port of the pod, and the public port it is reached at."""
    cost_per_hour: float | None = None
    gpu: str | None = None
    """The GPU type RunPod gave it, by its id (`NVIDIA H100 80GB HBM3`), where it says."""
    vcpus: int | None = None
    """The vCPUs RunPod gave it (`vcpuCount`), where it says."""
    memory_gb: float | None = None
    """The memory RunPod gave it, in GB (`memoryInGb`), where it says."""

    def address(self, port: int = 8443) -> str | None:
        """Where `port` is reached from outside, `https://IP:PORT`; None until RunPod has said."""
        public = self.ports.get(port)
        return f"https://{self.public_ip}:{public}" if self.public_ip and public else None

    @classmethod
    def of(cls, said: Mapping[str, Any]) -> "Pod":
        mappings: Any = said.get("portMappings") or {}
        ports = {int(inside): int(outside) for inside, outside in cast(dict[str, Any], mappings).items()}
        cost = said.get("adjustedCostPerHr", said.get("costPerHr"))
        vcpus, memory = _counted(said.get("vcpuCount")), _counted(said.get("memoryInGb"))
        return cls(
            id=str(said["id"]), name=str(said.get("name") or ""), status=str(said.get("desiredStatus") or ""),
            image=str(said.get("image") or said.get("imageName") or ""), public_ip=said.get("publicIp") or None,
            ports=ports, cost_per_hour=float(cost) if cost is not None else None, gpu=_gpu_of(said),
            vcpus=int(vcpus) if vcpus is not None else None, memory_gb=memory,
        )  # fmt: skip


class RunPod:
    """RunPod's pods API at `url`, with the key in the environment variable `key_env`."""

    def __init__(self, *, key_env: str = KEY, url: str = API, client: httpx.AsyncClient | None = None) -> None:
        self.key_env = key_env
        self.url = url.rstrip("/")
        self._http = client or httpx.AsyncClient(timeout=60.0)
        self._owned = client is None

    def __repr__(self) -> str:
        return f"RunPod(url={self.url!r}, key_env={self.key_env!r})"

    async def create(self, spec: PodSpec) -> Pod:
        """Ask for a pod; it starts as soon as RunPod has a machine for it."""
        pod = Pod.of(await self._call("POST", "/pods", spec.body()))
        log.info("created pod %s (%s) from %s", pod.id, pod.name, spec.image)
        return pod

    async def pods(self, *, name: str | None = None) -> list[Pod]:
        """Every pod of the account (those named `name`, if given)."""
        said: Any = await self._call("GET", "/pods", params={"name": name} if name else None)
        listed = cast(list[dict[str, Any]], said if isinstance(said, list) else said.get("pods", []))
        return [Pod.of(each) for each in listed]

    async def pod(self, id: str) -> Pod:
        return Pod.of(await self._call("GET", f"/pods/{id}"))

    async def start(self, id: str) -> None:
        """Start a stopped pod (its volume as it was left; its container disk afresh)."""
        await self._call("POST", f"/pods/{id}/start")
        log.info("started pod %s", id)

    async def stop(self, id: str) -> None:
        """Stop a pod: its GPU is released and no longer billed; its volume is kept (and billed) until it is deleted."""
        await self._call("POST", f"/pods/{id}/stop")
        log.info("stopped pod %s", id)

    async def terminate(self, id: str) -> None:
        """Delete a pod and its volume."""
        await self._call("DELETE", f"/pods/{id}")
        log.info("terminated pod %s", id)

    async def aclose(self) -> None:
        if self._owned:
            await self._http.aclose()

    async def _call(self, method: str, path: str, body: Any = None, *, params: Mapping[str, str] | None = None) -> Any:
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise RunPodError(f"{self.key_env} is not set: RunPod's API needs a key")
        try:
            response = await self._http.request(
                method, self.url + path, json=body, params=params,
                headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT},
            )  # fmt: skip
        except httpx.TransportError as error:
            raise RunPodError(f"{method} {path}: RunPod did not answer: {type(error).__name__}") from None
        if response.status_code >= 400:
            message = _message(response).replace(key, "…")
            raise RunPodError(f"{method} {path}: {response.status_code} {message}", response.status_code)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            raise RunPodError(f"{method} {path}: RunPod's answer is not JSON", response.status_code) from None


def _counted(value: Any) -> float | None:
    """A count RunPod said: a positive number (it says 0 or nothing while it has not placed the pod), else none."""
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _gpu_of(said: Mapping[str, Any]) -> str | None:
    """The GPU type a pod's description names: its `gpu`'s id, or its machine's `gpuTypeId`."""
    gpu: Any = said.get("gpu")
    if isinstance(gpu, dict) and (found := cast(dict[str, Any], gpu).get("id")):
        return str(found)
    machine: Any = said.get("machine")
    if isinstance(machine, dict) and (found := cast(dict[str, Any], machine).get("gpuTypeId")):
        return str(found)
    return None


def _message(response: httpx.Response) -> str:
    try:
        said: Any = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(said, dict):
        found = cast(dict[str, Any], said)
        return str(found.get("error") or found.get("message") or found)[:300]
    return str(said)[:300]
