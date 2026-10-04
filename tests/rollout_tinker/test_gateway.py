"""One gateway, started from the workspace like any other (`rollout gateway PROFILE`), hosts a channel sampled at
Tinker (over the fake service) beside a channel on an engine of this machine (a scripted one): it lists both, samples
each turn on the channel its key names, and records both turns in the ledger and blob store it shares."""

import asyncio
from pathlib import Path

import httpx
import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import TurnStore
from rollout_train.ledger import FileLedger
from tests.rollout_train.gateway.support import bearer, grant, keyring
from tests.rollout_train.gateway.test_hosted import a_profile, gateway
from tests.rollout_train.support import free_port

pytest.importorskip("uvicorn")

PROFILE = """
directory = "{directory}/run"
ledger = "{directory}/ledger"

[blobs]
kind = "rollout.harness.blobs:FileBlobStore"
directory = "{directory}/blobs"

[channels.tinker]
model = "Qwen/Qwen3.5-9B"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_tinker:TinkerEngine"
engines = [{{ service = "rollout_tinker.testing:fake_service", max_model_len = 4096 }}]
thinking_tokens = 16
answer_tokens = 16

[channels.scripted]
model = "echo"
renderer = "rollout_train.testing:plain_renderer"
engine = "tests.rollout_train.gateway.support:echo_engine"
thinking_tokens = 48
answer_tokens = 16

[gateway]
url = "http://127.0.0.1:{port}"
listen = "127.0.0.1:{port}"
keys = "{directory}/keys"
"""


def test_one_gateway_hosts_a_tinker_channel_beside_a_channel_of_this_machine(tmp_path: Path) -> None:
    port = free_port()
    profile = a_profile(tmp_path, port, PROFILE)
    ledger = FileLedger(tmp_path / "ledger")

    async def keys() -> dict[str, str]:
        fence = await ledger.take("runs/train/episodes/1/1")
        return {
            channel: keyring().mint(await grant(ledger, slot=channel, channel=channel, fence=fence))
            for channel in ("tinker", "scripted")
        }

    minted = asyncio.run(keys())
    with gateway(profile, port):
        listed = httpx.get(f"http://127.0.0.1:{port}/v1/models").json()["data"]
        assert sorted(each["id"] for each in listed) == ["scripted", "tinker"]
        sampled: dict[str, str] = {}
        for channel, key in minted.items():
            body = {"model": channel, "messages": [{"role": "user", "content": "Say a."}]}
            answer = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body,
                                headers=bearer(key, f"e-{channel}"), timeout=30)  # fmt: skip
            assert answer.status_code == 200, answer.text
            assert answer.json()["usage"]["completion_tokens"] > 0
            sampled[channel] = answer.headers["x-rollout-checkpoint"]
            if channel == "scripted":
                assert answer.json()["choices"][0]["message"]["content"].startswith("I saw ")
    assert sampled == {"tinker": "Qwen/Qwen3.5-9B", "scripted": "echo"}  # (each channel's own model's weights)

    async def recorded() -> dict[str, list[str]]:
        sessions = await TurnStore(ledger, FileBlobStore(tmp_path / "blobs")).sessions("train", "r_1")
        return {slot: [span.effect_id for segment in segments for span in segment.spans] for slot, segments in
                sessions.items()}  # fmt: skip

    assert asyncio.run(recorded()) == {"tinker": ["e-tinker"], "scripted": ["e-scripted"]}
