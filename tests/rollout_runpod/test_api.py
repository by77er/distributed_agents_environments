"""RunPod's pods API, against a fake of it served here: pods are created, listed, started, stopped and deleted; a pod's
public address is read from its port mappings; the key is sent and never logged, kept or put in an error."""

import logging
from pathlib import Path
from typing import Any

import pytest
from starlette.applications import Starlette

from rollout_runpod import Pod, PodSpec, RunPod, RunPodError
from rollout_runpod.api import USER_AGENT
from tests.rollout_runpod.fake import FakeRunPod
from tests.rollout_train.pods.authority import served_tls

KEY = "rpa_TESTKEY0123456789abcdefABCDEF"
TOKEN = "eyJhbGciOiJFUzI1NiJ9.one-time.signature"


def fake_runpod() -> tuple[Starlette, list[tuple[str, str, Any]]]:
    """A fake of RunPod's pods API (`tests.rollout_runpod.fake`): pods kept in memory; a request without the key is
    refused (401), one without a User-Agent RunPod's front accepts too (403)."""
    fake = FakeRunPod(key=KEY, cost=0.69)
    return fake.app, fake.asked


SPEC = PodSpec(
    name="inference-r1-0",
    image="ghcr.io/by77er/rollout-inference:0.1.0",
    gpu_types=["NVIDIA GeForce RTX 4090"],
    env={"ROLLOUT_POD_NAME": "inference-r1-0", "ROLLOUT_RUN": "r1"},
    secrets={"AWS_SECRET_ACCESS_KEY": "r2_secret"},
    sensitive={"STEP_TOKEN": TOKEN},
)


async def test_a_pod_is_created_listed_stopped_started_and_deleted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    caplog.set_level(logging.DEBUG)  # (everything httpx, httpcore and the client log)
    app, asked = fake_runpod()
    async with served_tls(app) as address:
        runpod = RunPod(url=f"{address}/v1")
        pod = await runpod.create(SPEC)
        assert (pod.name, pod.status, pod.address()) == ("inference-r1-0", "RUNNING", "https://203.0.113.7:40123")
        assert pod.cost_per_hour == 0.69 and pod.address(22) is None
        _, _, body = asked[0]
        assert body["env"] == {
            "ROLLOUT_POD_NAME": "inference-r1-0", "ROLLOUT_RUN": "r1",
            "AWS_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_r2_secret }}", "STEP_TOKEN": TOKEN,
        }  # fmt: skip
        assert body["ports"] == ["8443/tcp"] and body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
        assert (body["volumeMountPath"], body["cloudType"], body["supportPublicIp"]) == ("/workspace", "SECURE", True)
        assert [each.id for each in await runpod.pods(name="inference-r1-0")] == [pod.id]
        assert await runpod.pods(name="another") == []
        await runpod.stop(pod.id)
        stopped = await runpod.pod(pod.id)
        assert stopped.status == "EXITED" and stopped.address() is None
        await runpod.start(pod.id)
        assert (await runpod.pod(pod.id)).status == "RUNNING"
        await runpod.terminate(pod.id)
        assert await runpod.pods() == []
        with pytest.raises(RunPodError, match="400 pod not found") as refused:
            await runpod.pod(pod.id)
        assert refused.value.status == 400
        await runpod.aclose()
    logged = "\n".join(f"{record.getMessage()} {record.args}" for record in caplog.records)
    assert "created pod" in logged and KEY not in logged and TOKEN not in logged
    assert KEY not in repr(runpod) and TOKEN not in repr(SPEC) and TOKEN not in repr(pod)


async def test_a_refused_or_missing_key_is_said_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = fake_runpod()
    async with served_tls(app) as address:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        runpod = RunPod(url=f"{address}/v1")
        with pytest.raises(RunPodError, match="RUNPOD_API_KEY is not set"):
            await runpod.pods()
        monkeypatch.setenv("RUNPOD_API_KEY", "rpa_WRONG_KEY_value")
        with pytest.raises(RunPodError, match="401 Unauthorized") as refused:
            await runpod.pods()
        assert "rpa_WRONG_KEY_value" not in str(refused.value)
        await runpod.aclose()
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    unreachable = RunPod(url="http://127.0.0.1:9/v1")
    with pytest.raises(RunPodError, match="did not answer") as gone:
        await unreachable.pods()
    assert KEY not in str(gone.value) and gone.value.__cause__ is None
    await unreachable.aclose()


def test_a_pod_is_read_from_what_runpod_says() -> None:
    said = {"id": "x1", "name": "trainer-0", "desiredStatus": "RUNNING", "imageName": "img", "publicIp": "",
            "portMappings": {"22": 10341}, "adjustedCostPerHr": 0.5, "costPerHr": 0.7}  # fmt: skip
    pod = Pod.of(said)
    assert (pod.image, pod.public_ip, pod.ports, pod.cost_per_hour, pod.address()) == (
        "img",
        None,
        {22: 10341},
        0.5,
        None,
    )


async def test_every_request_says_who_sends_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    fake = FakeRunPod(key=KEY)
    runpod = fake.client()
    pod = await runpod.create(SPEC)
    await runpod.pods()
    await runpod.terminate(pod.id)
    assert USER_AGENT.startswith("rollout/") and fake.agents == [USER_AGENT] * 3
    assert pod.gpu == "NVIDIA GeForce RTX 4090" and Pod.of({"id": "p", "machine": {"gpuTypeId": "H100"}}).gpu == "H100"
    bare = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app))  # (what a client without one sends)
    refused = await bare.get("http://runpod.test/v1/pods", headers={"Authorization": f"Bearer {KEY}"})
    assert refused.status_code == 403
