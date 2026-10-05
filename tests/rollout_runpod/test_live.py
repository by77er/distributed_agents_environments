"""RunPod itself: a real pod leased, waited for, and deleted; an orphan with the cluster's tag reaped; and, with a
deployment to reach, a pod of a cluster config's provider leased, ready, sampled over mutual TLS and released.

Run only when asked (`ROLLOUT_RUNPOD=1`), with a key (`RUNPOD_API_KEY`) and a bucket the pods could read
(`ROLLOUT_RUNPOD_BUCKET=s3://BUCKET/PREFIX`, its endpoint and key in `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, or the `R2_WRITER_*` variables). It never leaves a pod behind: each test deletes what it made,
and the reaper is run at its end.

    ROLLOUT_RUNPOD=1 uv run pytest -s tests/rollout_runpod/test_live.py -p no:cacheprovider \\
        --basetemp ~/.cache/pytest-tmp/runpod-live

`test_a_pod_is_leased_waited_for_and_deleted` starts two pods of the cheapest GPU it is given (`ROLLOUT_RUNPOD_GPU`,
default `NVIDIA RTX A4000`, community cloud, about $0.17 an hour) running a small image that never says it is ready,
for at most `ROLLOUT_RUNPOD_TIMEOUT` seconds (default 240) each: it costs a few cents.

`test_a_providers_pod_serves_a_run` runs only with a deployment to reach (`ROLLOUT_RUNPOD_CLUSTER`, a cluster config
whose RunPod provider `ROLLOUT_RUNPOD_PROVIDER` names the pushed inference or host image, whose `[ledger] url` is the
ledger service's public address with the platform's token, and whose step-ca RunPod reaches): it leases one pod for up
to 20 minutes, samples it once and releases it, deleted at once. On an RTX A4000 that is about ten cents; on an H100 SXM
about a dollar.
"""

import asyncio
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.skipif(os.environ.get("ROLLOUT_RUNPOD") != "1", reason="rents RunPod's pods (ROLLOUT_RUNPOD=1)"),
    pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="no RunPod key: set RUNPOD_API_KEY"),
    pytest.mark.skipif(not os.environ.get("ROLLOUT_RUNPOD_BUCKET"), reason="no bucket: set ROLLOUT_RUNPOD_BUCKET"),
]

GPU = os.environ.get("ROLLOUT_RUNPOD_GPU", "NVIDIA RTX A4000")
TIMEOUT = float(os.environ.get("ROLLOUT_RUNPOD_TIMEOUT", "240"))
IMAGE = "busybox:1.37"
"""What the first test's pods run: a small image whose pods never beat, so the startup timeout deletes them."""


def bucket() -> dict[str, Any]:
    """The bucket's `[stores.live]` table, its key named."""
    from urllib.parse import urlparse

    said = urlparse(os.environ["ROLLOUT_RUNPOD_BUCKET"])
    table: dict[str, Any] = {"kind": "rollout_s3:S3BlobStore", "bucket": said.netloc, "prefix": said.path.lstrip("/")}
    if os.environ.get("R2_WRITER_ACCESS_KEY_ID"):
        table |= {
            "access_key_id_env": "R2_WRITER_ACCESS_KEY_ID",
            "secret_access_key_env": "R2_WRITER_SECRET_ACCESS_KEY",
        }
    if endpoint := os.environ.get("AWS_ENDPOINT_URL"):
        table |= {"endpoint_url": endpoint, "region": os.environ.get("AWS_DEFAULT_REGION", "auto")}
    return table


async def test_a_pod_is_leased_waited_for_and_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rollout_runpod import PodSpec, RunPod
    from rollout_train.cluster import parsed
    from rollout_train.database import DatabaseLedger
    from rollout_train.pods.leasing import PodNeed, Pods, PodsDidNotStart, reap, tag_of
    from rollout_train.stores import Stores

    monkeypatch.setenv("ROLLOUT_LEDGER_TOKEN", "a-live-test-of-runpod")
    stores = json.dumps(bucket())
    described = tomllib.loads(f"""
name = "live"
[ledger]
url = "sqlite:///{tmp_path / "ledger.db"}"
token_env = "ROLLOUT_LEDGER_TOKEN"
public = "https://ledger.invalid"
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[inference.cheap]
kind = "runpod-inference"
image = "{IMAGE}"
gpu_types = ["{GPU}"]
cloud = "community"
max_pods = 1
idle_stop = 0
start_timeout = {TIMEOUT}
volume_gb = 10
container_disk_gb = 10
store = "live"
[inference.cheap.models."m"]
context = 1024
""")
    described["stores"] = {"live": json.loads(stores)}
    cluster = parsed(described)
    blobs = Stores.open(cluster, store="live").blobs  # (the bucket answers with the key it is given)
    reference = await blobs.put(b"a live test of the bucket", "text/plain")
    assert await blobs.read(reference) == b"a live test of the bucket"
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    seen: list[str] = []

    async def told(waits: Any) -> None:
        seen.extend(waits)

    pods = Pods("run_live", cluster, ledger, told=told, look=5.0, ca=lambda table: None)
    try:
        with pytest.raises(PodsDidNotStart, match="it was deleted"):
            await pods.claim([PodNeed("cheap", "inference", 1, "m", "policy")])
    finally:
        await pods.release()
    print("\n".join(sorted(set(seen))[-5:]), flush=True)
    runpod = RunPod()
    try:
        left = [each for each in await runpod.pods() if each.name.startswith(tag_of(cluster))]
        assert left == [], f"pods left behind: {[each.id for each in left]}"
        orphan = await runpod.create(PodSpec(f"{tag_of(cluster)}cheap-9-orphan", IMAGE, [GPU], volume_gb=10,
                                             container_disk_gb=10, cloud="COMMUNITY"))  # fmt: skip
        said = await reap(cluster, ledger, ca=lambda table: None)
        assert any(orphan.id in each for each in said), said
        await asyncio.sleep(5)
        assert orphan.id not in {each.id for each in await runpod.pods()}
    finally:
        for each in await runpod.pods():  # (whatever went wrong, nothing of this test is left)
            if each.name.startswith(tag_of(cluster)):
                await runpod.terminate(each.id)
        await runpod.aclose()


@pytest.mark.skipif(
    not os.environ.get("ROLLOUT_RUNPOD_CLUSTER"), reason="no deployment to reach (ROLLOUT_RUNPOD_CLUSTER)"
)
async def test_a_providers_pod_serves_a_run() -> None:
    from rollout_train.cluster import load
    from rollout_train.inference import Limits, RemoteChannel
    from rollout_train.pods.leasing import Pods, reap
    from rollout_train.pods.routing import LeasedServers
    from rollout_train.run_settings import RunSettings
    from rollout_train.serving import serving_of
    from rollout_train.stores import ledger_of
    from rollout_train.testing import plain_renderer

    cluster = await asyncio.to_thread(lambda: load(Path(os.environ["ROLLOUT_RUNPOD_CLUSTER"]).expanduser()))
    name = os.environ.get("ROLLOUT_RUNPOD_PROVIDER", "runpod")
    provider = cluster.inference[name]
    model = next(iter(provider.models))
    ledger = ledger_of(cluster)
    run = "run_live_serving"
    from rollout_train.record import STARTS, start_header, table

    fence = await ledger.take(f"runs/{run}")
    await ledger.append(table(run, STARTS), str(fence.number), {**start_header(kind="eval"), "run_settings": {
        "fixed": {"kind": "eval", "channels.policy.provider": name, "channels.policy.model": model}, "changeable": {},
    }}, fence)  # fmt: skip
    from rollout_train.pods.leasing import needs_of

    settings = RunSettings({"kind": "eval", "channels.policy.provider": name, "channels.policy.model": model,
                            "channels.policy.renderer": "rollout_train.testing:plain_renderer"})  # fmt: skip
    pods = Pods(run, cluster, ledger, look=10.0)
    try:
        await pods.claim(needs_of(settings, cluster))
        servers = LeasedServers(ledger, run, "policy", [name], model, provider.auth, cluster.tls)

        async def wanted() -> Any:
            return await serving_of(ledger, run, "policy")

        channel = RemoteChannel("policy", plain_renderer(model), Limits(), model=model, servers=[], wanted=wanted,
                                discover=servers, patience=120)  # fmt: skip
        adapter, _ = await channel.weights("live")
        reply = await channel.generate([1, 2, 3], adapter=adapter, max_tokens=4, temperature=1.0, top_p=1.0,
                                       stop_token_ids=[], session="live")  # fmt: skip
        assert reply.tokens and len(reply.logprobs) == len(reply.tokens)
        channel.close()
    finally:
        for lease in list(pods.leases.values()):  # (deleted at once, not kept warm)
            await pods.deleted(lease, "the live test is done")
        await pods.release()
        print("\n".join(await reap(cluster, ledger)), flush=True)
