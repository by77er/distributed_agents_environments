"""RunPod's pods API, against a fake of it served here: pods are created, listed, started, stopped and deleted; a pod's
public address is read from its port mappings; the key is sent and never logged, kept or put in an error."""

import logging
from pathlib import Path
from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rollout_runpod import Pod, PodSpec, RunPod, RunPodError
from tests.rollout_train.pods.authority import served_tls

KEY = "rpa_TESTKEY0123456789abcdefABCDEF"
TOKEN = "eyJhbGciOiJFUzI1NiJ9.one-time.signature"


def fake_runpod() -> tuple[Starlette, list[tuple[str, str, Any]]]:
    """A fake of RunPod's pods API: pods kept in memory; a request without the key is refused (401)."""
    pods: dict[str, dict[str, Any]] = {}
    asked: list[tuple[str, str, Any]] = []

    def refused(request: Request) -> Response | None:
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return None

    async def collection(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        if request.method == "GET":
            asked.append(("GET", "/pods", dict(request.query_params)))
            name = request.query_params.get("name")
            return JSONResponse([pod for pod in pods.values() if name is None or pod["name"] == name])
        body = await request.json()
        asked.append(("POST", "/pods", body))
        id = f"pod{len(pods) + 1:011d}"
        pods[id] = {
            "id": id, "name": body["name"], "image": body["imageName"], "desiredStatus": "RUNNING",
            "publicIp": "203.0.113.7", "portMappings": {"8443": 40123}, "costPerHr": 0.69, "env": body["env"],
        }  # fmt: skip
        return JSONResponse(pods[id], status_code=201)

    async def one(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        id = request.path_params["id"]
        asked.append((request.method, f"/pods/{id}", None))
        if id not in pods:
            return JSONResponse({"error": "pod not found"}, status_code=400)
        if request.method == "DELETE":
            del pods[id]
            return Response(status_code=204)
        return JSONResponse(pods[id])

    async def action(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        id, verb = request.path_params["id"], request.path_params["verb"]
        asked.append(("POST", f"/pods/{id}/{verb}", None))
        pods[id]["desiredStatus"] = {"start": "RUNNING", "stop": "EXITED"}[verb]
        if verb == "stop":
            pods[id]["publicIp"], pods[id]["portMappings"] = "", {}
        return JSONResponse(pods[id])

    app = Starlette(
        routes=[
            Route("/v1/pods", collection, methods=["GET", "POST"]),
            Route("/v1/pods/{id}", one, methods=["GET", "DELETE"]),
            Route("/v1/pods/{id}/{verb}", action, methods=["POST"]),
        ]
    )
    return app, asked


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
