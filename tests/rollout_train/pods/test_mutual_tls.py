"""Mutual TLS between the gateway and a pod: the gateway reaches a pod by the identity its certificate carries, not by
its address; a pod takes requests only from a client whose certificate carries the gateway's identity. The pod here is
a fake vLLM server behind a TLS server that checks clients as the pod's Envoy does."""

import ssl
from pathlib import Path

import pytest

from rollout_train.inference import Connection, RemoteEngine
from rollout_train.inference.remote import Unreachable, requiring
from rollout_train.pods import GATEWAY_IDENTITY, pod_identity
from tests.rollout_train.machines import MODEL, Saying, fake_vllm
from tests.rollout_train.pods.authority import Authority, Issued, served_tls, server_context

POD = pod_identity("inference-1")


class Counting:
    """An app that counts the requests that reach it, in front of another."""

    def __init__(self, app: object) -> None:
        self.app = app
        self.reached = 0

    async def __call__(self, scope: dict[str, object], receive: object, send: object) -> None:
        if scope["type"] == "http":
            self.reached += 1
        await self.app(scope, receive, send)  # pyright: ignore[reportCallIssue, reportUnknownMemberType]


def reaching(authority: Authority, client: Issued | None, identity: str | None = POD) -> Connection:
    """How the gateway reaches the pod: the cluster's root, a client certificate, the identity the pod must carry."""
    certificate = {"certificate": str(client.certificate), "key": str(client.key)} if client else {}
    return Connection(ca=str(authority.root), identity=identity, **certificate)


async def test_the_gateway_reaches_a_pod_by_the_identity_its_certificate_carries(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    pod, gateway = authority.issue("pod", POD), authority.issue("gateway", GATEWAY_IDENTITY)
    async with served_tls(fake_vllm(Saying()), server_context(authority, pod, client=GATEWAY_IDENTITY)) as address:
        engine = RemoteEngine(MODEL, connection=reaching(authority, gateway), address=address)
        assert set(await engine.models()) == {MODEL}  # (reached by its IP, which its certificate does not name)
        reply = await engine.generate([65], adapter=None, max_tokens=8, temperature=1.0, top_p=1.0, stop_token_ids=[10])
        assert "".join(map(chr, reply.tokens)) == "base\n"
        engine.close()


@pytest.mark.parametrize("client", ["another pod", "no identity", "another root", "none"])
async def test_a_pod_refuses_a_client_that_is_not_the_gateway(tmp_path: Path, client: str) -> None:
    authority, elsewhere = Authority(tmp_path / "ca"), Authority(tmp_path / "elsewhere", "another root")
    pod = authority.issue("pod", POD)
    presented = {
        "another pod": authority.issue("other", pod_identity("inference-2")),
        "no identity": authority.issue("plain", None),
        "another root": elsewhere.issue("gateway", GATEWAY_IDENTITY),
        "none": None,
    }[client]
    app = Counting(fake_vllm(Saying()))
    async with served_tls(app, server_context(authority, pod, client=GATEWAY_IDENTITY)) as address:
        engine = RemoteEngine(MODEL, connection=reaching(authority, presented), address=address)
        with pytest.raises(Unreachable):
            await engine.models()
        engine.close()
    assert app.reached == 0


@pytest.mark.parametrize("server", ["another pod", "another root", "no identity"])
async def test_the_gateway_refuses_a_server_that_is_not_the_pod_it_asked_for(tmp_path: Path, server: str) -> None:
    authority, elsewhere = Authority(tmp_path / "ca"), Authority(tmp_path / "elsewhere", "another root")
    gateway = authority.issue("gateway", GATEWAY_IDENTITY)
    serving = {
        "another pod": authority.issue("other", pod_identity("inference-2")),
        "another root": elsewhere.issue("impostor", POD),
        "no identity": authority.issue("plain", None, ips=["127.0.0.1"]),  # (its address is right: not enough)
    }[server]
    issuer = elsewhere if server == "another root" else authority
    app = Counting(fake_vllm(Saying()))
    async with served_tls(app, server_context(issuer, serving, client=None)) as address:
        engine = RemoteEngine(MODEL, connection=reaching(authority, gateway), address=address)
        with pytest.raises(Unreachable, match=r"CERTIFICATE_VERIFY_FAILED|not spiffe://rollout/pod/inference-1"):
            await engine.models()
        engine.close()
    assert app.reached == 0  # (refused in the handshake: nothing was sent)


async def test_without_an_identity_a_server_is_known_by_its_host_name_as_before(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    pod, gateway = authority.issue("pod", POD), authority.issue("gateway", GATEWAY_IDENTITY)
    async with served_tls(fake_vllm(Saying()), server_context(authority, pod, client=None)) as address:
        engine = RemoteEngine(MODEL, connection=reaching(authority, gateway, identity=None), address=address)
        with pytest.raises(Unreachable, match="CERTIFICATE_VERIFY_FAILED"):  # (127.0.0.1 is not in its certificate)
            await engine.models()
        engine.close()


def test_an_identity_is_checked_only_on_a_certificate_that_is_verified() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)  # (asks for no client certificate)
    with pytest.raises(ValueError, match="verified"):
        requiring(context, GATEWAY_IDENTITY)
