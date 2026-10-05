"""What the page is offered of hosted APIs: each `api` provider, metered, with its models and their prices, which
render messages themselves and serve no trainer's checkpoints; and an eval on one checked with the whole eval's
spend."""

from pathlib import Path

import httpx
import pytest

from rollout.names import named
from rollout_train.evals import make_suite, suite_entry
from rollout_train.stores import Stores
from rollout_train.submitting import RayJobs
from tests.rollout_train.clusters import WORDS, a_cluster
from tests.rollout_train.test_hosted_evals import HOSTED
from tests.rollout_train.test_submitting import Jobs

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app


async def test_the_page_is_offered_hosted_apis_metered_with_their_prices_and_an_evals_spend(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, more=HOSTED.format(key="ROLLOUT_TEST_KEY", url="http://127.0.0.1:9/v1"))
    stores = Stores.open(cluster)
    await make_suite(stores.ledger, "held-out", [suite_entry(WORDS, named(WORDS), eval_data="words-held-out")])
    app = create_app(
        f"sqlite:///{tmp_path}/ledger.db", beat=0.0, cluster=cluster, backends={"ray": RayJobs("x", Jobs())}
    )
    asked = {
        "kind": "eval", "name": "held out, hosted", "settings": {
            "eval.suite": "held-out", "channels.policy.provider": "hosted", "channels.policy.model": "model-a",
            "limits.spend": 1,
        },
    }  # fmt: skip
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://monitor") as client:
        offers = (await client.get("/api/offers")).json()
        checked = (await client.post("/api/launches/check", json=asked)).json()
    (hosted,) = [each for each in offers["inference"] if each["name"] == "hosted"]
    assert (hosted["kind"], hosted["allocation"], hosted["concurrency"]) == ("api", "metered", 2)
    assert not hosted["capabilities"]["token_exact"] and not hosted["capabilities"]["sampled_logprobs"]
    (model,) = hosted["models"]
    assert model["model"] == "model-a" and model["cost"] == {"input": 2.0, "output": 10.0} and model["renderers"] == []
    assert hosted["weights"] == []  # (it serves no checkpoint)
    (pair,) = [each for each in offers["pairs"] if each["inference"] == "hosted"]
    assert pair["bridge"] is None and pair["refused"]
    assert checked["refusals"] == [] and checked["spend"]["per"] == "eval"
    assert checked["settings"]["environment"] == WORDS and "channels.policy.renderer" not in checked["settings"]
