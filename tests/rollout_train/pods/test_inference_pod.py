"""An inference pod's follower: it keeps the pod's vLLM server serving what the run says its channel should, says when
the pod is ready, and beats with the pod's name, identity and address. The server is a fake of vLLM's; the ledger a
scratch SQLite one; the blob store files."""

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.database import DatabaseLedger
from rollout_train.inference.remote import ENGINES
from rollout_train.ledger import Fence
from rollout_train.pods import live, pod_identity
from rollout_train.pods.inference import InferencePod, health
from rollout_train.presence import presence_of
from rollout_train.record import scope
from rollout_train.serving import Serving, record_serving
from tests.rollout_train.machines import MODEL, Saying, fake_vllm
from tests.rollout_train.pods.authority import served_tls

ADDRESS = "https://203.0.113.7:40123"


def stores(tmp_path: Path) -> Checkpoints:
    return Checkpoints(DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}"), FileBlobStore(tmp_path / "blobs"))


async def made(checkpoints: Checkpoints, fence: Fence, tmp_path: Path, parent: Checkpoint | None = None) -> Checkpoint:
    """A checkpoint of run `r`, its weights one small file."""
    id = new_id()
    weights = tmp_path / "trained" / id
    weights.mkdir(parents=True)
    (weights / "adapter_model.safetensors").write_text(id)
    parents = [parent.id] if parent else []
    return await checkpoints.add(fence, id, weights=weights, run="r", base=MODEL, parents=parents)


async def serve(checkpoints: Checkpoints, fence: Fence, checkpoint: Checkpoint) -> None:
    said = Serving("policy", checkpoint.id, checkpoint.depth, checkpoint.kind, checkpoint.weights, model=MODEL)
    await record_serving(checkpoints.ledger, "r", said, fence)


@contextlib.asynccontextmanager
async def pod(checkpoints: Checkpoints, tmp_path: Path, vllm: str | None = None) -> AsyncGenerator[InferencePod]:
    serial = tmp_path / "serial"
    serial.write_text("98765\n")
    engine = Saying()
    async with served_tls(fake_vllm(engine)) as address:
        yield InferencePod(
            "inference-1", checkpoints, "r", "policy", MODEL, tmp_path / "checkpoints", vllm=vllm or address,
            address=ADDRESS, presence=presence_of(checkpoints.ledger), serial_file=serial,
        )  # fmt: skip


async def test_a_pod_is_ready_once_its_server_has_what_the_channel_should_serve(tmp_path: Path) -> None:
    checkpoints = stores(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    async with pod(checkpoints, tmp_path) as following:
        assert not following.ready and not await following.follow()  # (the run says nothing: the base model)
        channel = following.channel
        assert following.ready and channel is not None and channel.serving is None
        first = await made(checkpoints, fence, tmp_path)
        await serve(checkpoints, fence, first)
        assert await following.follow() and following.ready and channel.serving == first.id
        loaded = tmp_path / "checkpoints" / first.id / "weights" / "adapter_model.safetensors"
        assert loaded.read_text() == first.id  # (fetched from the blob store into the pod's volume)
        second = await made(checkpoints, fence, tmp_path, first)
        await serve(checkpoints, fence, second)
        await following.look()  # (before it loads it: not ready, and says why)
        assert not following.ready and following.why == f"{second.id} is not served yet"
        assert await following.follow() and following.ready
        assert set(await following.engine.models()) == {MODEL, first.id, second.id}  # (the one before stays)


async def test_a_pod_whose_server_does_not_answer_is_alive_and_not_ready(tmp_path: Path) -> None:
    checkpoints = stores(tmp_path)
    async with pod(checkpoints, tmp_path, vllm="http://127.0.0.1:9") as following:
        await following.follow()
        assert not following.ready and "Unreachable" in following.why
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=health(following)), base_url="http://pod"
        ) as http:
            assert (await http.get("/healthz")).status_code == 200
            answer = await http.get("/readyz")
            assert answer.status_code == 503 and answer.json()["ready"] is False
            following.looked = following.started = -1e9  # (as if it had stopped looking long ago)
            assert (await http.get("/healthz")).status_code == 503


async def test_a_pod_beats_with_its_identity_address_readiness_and_serial(tmp_path: Path) -> None:
    checkpoints = stores(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    async with pod(checkpoints, tmp_path) as following:
        first = await made(checkpoints, fence, tmp_path)
        await serve(checkpoints, fence, first)
        await following.follow()
        await following.beat()
        presence = presence_of(checkpoints.ledger)
        assert presence is not None
        (beat,) = await presence.beats()
        assert beat.about["kind"] == ENGINES and beat.about["follows"] == "r"  # (what every follower says)
        (reached,) = live(await presence.beats(), "inference")
        assert (reached.name, reached.identity, reached.address) == (
            "inference-1",
            pod_identity("inference-1"),
            ADDRESS,
        )
        assert reached.ready and reached.serial == "98765"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=health(following)), base_url="http://pod"
        ) as http:
            said = (await http.get("/readyz")).json()
            assert said == {"ready": True, "serving": first.id, "why": ""}
