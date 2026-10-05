"""The gateway reaching a run's pods: a run leases a pod (the fake RunPod API starts a process standing in for the
inference image: a fake vLLM behind a TLS server that takes only the gateway's certificate, and the real follower,
reading the ledger through the ledger service with the pod's own token); the pod beats ready; the run's channel finds
it by its lease and beat and samples it over mutual TLS, checking its identity; once released, it is no server."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoints
from rollout_train.database import DatabaseLedger
from rollout_train.inference import Limits, RemoteChannel
from rollout_train.ledger_service.service import held_by_run
from rollout_train.pods import GATEWAY_IDENTITY, pod_identity
from rollout_train.pods.inference import InferencePod
from rollout_train.pods.leasing import PodNeed, Pods
from rollout_train.pods.routing import LeasedServers
from rollout_train.presence import presence_of
from rollout_train.providers import BEATS, Auth, Tls
from rollout_train.serving import serving_of
from rollout_train.testing import plain_renderer, served_ledger
from tests.rollout_runpod.fake import KEY, FakeRunPod
from tests.rollout_train.machines import MODEL, Saying, fake_vllm
from tests.rollout_train.pods.authority import Authority, served_tls, server_context
from tests.rollout_train.pods.test_leasing import ENVIRON, cluster_of


class Pod:
    """What runs on a pod the fake starts: vLLM (fake) on loopback, Envoy (a TLS server taking only the gateway's
    certificate), the follower (reading its lease through the ledger service, with the token RunPod was given)."""

    def __init__(self, ledger: DatabaseLedger, authority: Authority, directory: Path) -> None:
        self.ledger, self.authority, self.directory = ledger, authority, directory
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.followers: list[InferencePod] = []

    def created(self, pod: dict[str, Any], body: dict[str, Any]) -> None:
        self.tasks[pod["id"]] = asyncio.get_running_loop().create_task(self._run(body["name"], body["env"]))

    def deleted(self, id: str) -> None:
        if (task := self.tasks.pop(id, None)) is not None:
            task.cancel()

    async def _run(self, name: str, env: dict[str, str]) -> None:
        issued = self.authority.issue(name, pod_identity(name))
        app = fake_vllm(Saying())
        envoy = server_context(self.authority, issued, client=GATEWAY_IDENTITY)
        ledger = served_ledger(self.ledger, token=env["ROLLOUT_LEDGER_TOKEN"], honoured=held_by_run)
        assert json.loads(env["ROLLOUT_LEDGER"])["token_env"] == "ROLLOUT_LEDGER_TOKEN"
        async with served_tls(app) as loopback, served_tls(app, envoy) as public:
            follower = InferencePod(
                name, Checkpoints(ledger, FileBlobStore(self.directory / "blobs")), None, None, MODEL,
                self.directory / name, vllm=loopback, address=public, presence=presence_of(ledger), every=0.05,
                beating=0.05,
            )  # fmt: skip
            self.followers.append(follower)
            await follower.serve()


@pytest.fixture
async def gateway_and_pods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[DatabaseLedger, FakeRunPod, Pod, Authority]]:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    authority = Authority(tmp_path / "ca")
    pods = Pod(ledger, authority, tmp_path)
    fake = FakeRunPod(key=KEY, created=pods.created, deleted=pods.deleted)
    yield ledger, fake, pods, authority
    for task in pods.tasks.values():
        task.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.gather(*pods.tasks.values(), return_exceptions=True)


async def test_a_runs_channel_samples_its_leased_pod_over_mutual_tls_by_the_identity_its_beat_names(
    tmp_path: Path, gateway_and_pods: tuple[DatabaseLedger, FakeRunPod, Pod, Authority]
) -> None:
    ledger, fake, stand_ins, authority = gateway_and_pods
    cluster = cluster_of(tmp_path)
    fence = await ledger.take("runs/run_1")
    from rollout_train.record import STARTS, table

    await ledger.append(table("run_1", STARTS), "1", {"run_settings": {"fixed": {}, "changeable": {}}}, fence)
    pods = Pods("run_1", cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None, environ=ENVIRON,
                look=0.05)  # fmt: skip
    (lease,) = await pods.claim([PodNeed("pods", "inference", 1, MODEL, "policy")])
    issued = authority.issue("gateway", GATEWAY_IDENTITY)
    tls = Tls(ca=str(authority.root), certificate=str(issued.certificate), key=str(issued.key))
    servers = LeasedServers(ledger, "run_1", "policy", ["pods"], MODEL, Auth("mtls", identity=BEATS), tls)

    async def wanted() -> Any:
        return await serving_of(ledger, "run_1", "policy")

    channel = RemoteChannel("policy", plain_renderer(MODEL), Limits(), model=MODEL, servers=[], wanted=wanted,
                            discover=servers, every=0.05, patience=5)  # fmt: skip
    (found,) = await servers()
    assert found.address.startswith("https://127.0.0.1:")  # (where the pod's beat says it is reached)
    adapter, _ = await channel.weights("session-1")
    reply = await channel.generate([65], adapter=adapter, max_tokens=8, temperature=1.0, top_p=1.0, stop_token_ids=[10],
                                   session="session-1")  # fmt: skip
    assert "".join(map(chr, reply.tokens)) == "base\n" and adapter is None
    (follower,) = stand_ins.followers
    assert follower.held == ("run_1", "policy") and follower.ready
    await pods.release()
    for _ in range(100):  # (the follower sees its lease idle, and stops saying it is ready)
        if not follower.ready:
            break
        await asyncio.sleep(0.05)
    assert follower.held is None and not follower.ready and await servers() == []
    assert lease.id in fake.pods  # (warm)
    channel.close()
    await asyncio.sleep(0)  # (the clients close)
