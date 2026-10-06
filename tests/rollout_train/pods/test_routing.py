"""The gateway reaching a run's pods: a run leases a pod (the fake RunPod API starts a process standing in for the
inference image: a fake vLLM behind a TLS server that takes only the gateway's certificate, and the real follower,
reading the ledger through the ledger service with the pod's own token); RunPod says where the pod is reached, which
its lease keeps; the pod beats ready; the run's channel finds it by its lease and beat and samples it at its lease's
address over mutual TLS, checking its identity; once released, it is no server. Where a pod's beat says it is reached
is never read, and a pod is never reached over plain HTTP."""

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoints
from rollout_train.database import DatabaseLedger
from rollout_train.inference import Limits, RemoteChannel
from rollout_train.inference.remote import Connection, RemoteEngine
from rollout_train.ledger_service.service import held_by_run
from rollout_train.pods import GATEWAY_IDENTITY, pod_identity
from rollout_train.pods.inference import InferencePod
from rollout_train.pods.leases import pod_leases_of
from rollout_train.pods.leasing import PodNeed, Pods
from rollout_train.pods.routing import LeasedServers
from rollout_train.presence import presence_of
from rollout_train.providers import LEASED, Auth, Tls
from rollout_train.serving import serving_of
from rollout_train.testing import plain_renderer, served_ledger
from tests.rollout_runpod.fake import KEY, FakeRunPod
from tests.rollout_train.machines import MODEL, Saying, fake_vllm
from tests.rollout_train.pods.authority import Authority, served_tls, server_context
from tests.rollout_train.pods.test_leasing import ENVIRON, NEED, StandIns, cluster_of


class Pod:
    """What runs on a pod the fake starts: vLLM (fake) on loopback, Envoy (a TLS server taking only the gateway's
    certificate), the follower (reading its lease through the ledger service, with the token RunPod was given)."""

    def __init__(self, ledger: DatabaseLedger, authority: Authority, directory: Path) -> None:
        self.ledger, self.authority, self.directory = ledger, authority, directory
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.followers: list[InferencePod] = []
        self.runpod: FakeRunPod | None = None
        """The fake told where each pod is reached once it serves (as RunPod says a pod's address once it runs)."""

    def created(self, pod: dict[str, Any], body: dict[str, Any]) -> None:
        self.tasks[pod["id"]] = asyncio.get_running_loop().create_task(self._run(pod["id"], body["name"], body["env"]))

    def deleted(self, id: str) -> None:
        if (task := self.tasks.pop(id, None)) is not None:
            task.cancel()

    async def _run(self, id: str, name: str, env: dict[str, str]) -> None:
        issued = self.authority.issue(name, pod_identity(name))
        app = fake_vllm(Saying())
        envoy = server_context(self.authority, issued, client=GATEWAY_IDENTITY)
        ledger = served_ledger(self.ledger, token=env["ROLLOUT_LEDGER_TOKEN"], honoured=held_by_run)
        assert json.loads(env["ROLLOUT_LEDGER"])["token_env"] == "ROLLOUT_LEDGER_TOKEN"
        async with served_tls(app) as loopback, served_tls(app, envoy) as public:
            if self.runpod is not None:
                self.runpod.address(id, public)
            follower = InferencePod(
                name, Checkpoints(ledger, FileBlobStore(self.directory / "blobs")), None, None, MODEL,
                self.directory / name, vllm=loopback, presence=presence_of(ledger), every=0.05, beating=0.05,
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
    fake = FakeRunPod(key=KEY, created=pods.created, deleted=pods.deleted, addressed=False)
    pods.runpod = fake
    yield ledger, fake, pods, authority
    for task in pods.tasks.values():
        task.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.gather(*pods.tasks.values(), return_exceptions=True)


async def test_a_runs_channel_samples_its_leased_pod_at_its_leases_address_over_mutual_tls_by_its_identity(
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
    servers = LeasedServers(ledger, "run_1", "policy", ["pods"], MODEL, Auth("mtls", identity=LEASED), tls)

    async def wanted() -> Any:
        return await serving_of(ledger, "run_1", "policy")

    channel = RemoteChannel("policy", plain_renderer(MODEL), Limits(), model=MODEL, servers=[], wanted=wanted,
                            discover=servers, every=0.05, patience=5)  # fmt: skip
    (found,) = await servers()
    assert lease.address is not None and lease.address.startswith("https://127.0.0.1:")  # (where RunPod says it is)
    assert found.address == lease.address
    adapter, _ = await channel.weights("session-1")
    reply = await channel.generate([65], adapter=adapter, max_tokens=8, temperature=1.0, top_p=1.0, stop_token_ids=[10],
                                   session="session-1")  # fmt: skip
    assert "".join(map(chr, reply.tokens)) == "base\n" and adapter is None
    (follower,) = stand_ins.followers
    assert follower.held == ("run_1", "policy") and follower.ready
    await pods.release()
    for _ in range(100):  # (the follower sees its lease idle: it holds nothing, and says it is not ready)
        if follower.held is None and not follower.ready:
            break
        await asyncio.sleep(0.05)
    assert follower.held is None and not follower.ready and await servers() == []
    assert lease.id in fake.pods  # (warm)
    channel.close()
    await asyncio.sleep(0)  # (the clients close)


def gateway_tls(authority: Authority) -> Tls:
    issued = authority.issue("gateway", GATEWAY_IDENTITY)
    return Tls(ca=str(authority.root), certificate=str(issued.certificate), key=str(issued.key))


async def test_a_pod_is_reached_where_runpod_says_never_where_its_beat_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    stand_ins = StandIns(ledger, says={"address": "http://203.0.113.66:80"})  # (what anyone with its token may beat)
    fake = FakeRunPod(key=KEY, created=stand_ins.created, deleted=stand_ins.deleted, addressed=False)
    told: list[str] = []

    async def say(waits: Any) -> None:
        told.extend(waits)

    pods = Pods("run_1", cluster_of(tmp_path), ledger, api=lambda provider: fake.client(), ca=lambda table: None,
                environ=ENVIRON, told=say, look=0.05)  # fmt: skip
    claiming = asyncio.ensure_future(pods.claim([NEED]))
    try:
        await asyncio.sleep(0.5)
        assert not claiming.done()  # (it beats ready, and RunPod has not said where it is: not ready yet)
        assert any("has not said its public address" in each for each in told)
        (id,) = fake.pods
        fake.address(id, "https://198.51.100.4:41000")
        (lease,) = await asyncio.wait_for(claiming, 5)
    finally:
        claiming.cancel()
    assert lease.address == "https://198.51.100.4:41000"
    servers = LeasedServers(ledger, "run_1", "policy", ["pods"], "m", Auth("mtls", identity=LEASED),
                            gateway_tls(Authority(tmp_path / "ca")))  # fmt: skip
    (found,) = await servers()
    assert found.address == "https://198.51.100.4:41000"  # (the lease's: never the beat's)
    store = pod_leases_of(ledger)
    assert store is not None
    there = await store.get(lease.pod)
    assert there is not None
    await store.put(dataclasses.replace(there, address="http://198.51.100.4:41000"), expect=there.version)
    assert await servers() == []  # (a lease that says plain HTTP is no server)
    servers.close()
    for task in stand_ins.tasks.values():
        task.cancel()


async def test_a_connection_that_checks_an_identity_never_sends_over_plain_http() -> None:
    identity = pod_identity("inference-1")
    with pytest.raises(ValueError, match="https only"):
        RemoteEngine(MODEL, address="http://127.0.0.1:9", connection=Connection(identity=identity))
    client = Connection(identity=identity).client()
    async with client:
        with pytest.raises(httpx.UnsupportedProtocol, match="https only"):
            await client.get("http://127.0.0.1:9/v1/models")
    plain = Connection().client()  # (no identity: a server on this machine, reached as it is said)
    async with plain:
        with pytest.raises(httpx.ConnectError):
            await plain.get("http://127.0.0.1:9/v1/models")
