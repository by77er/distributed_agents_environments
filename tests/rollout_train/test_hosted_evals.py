"""An eval of a hosted model, built from its settings on the session's Ray: the suite is played on a channel of an
`api` provider (a fake Responses API on this machine), each turn recorded with no tokens and what it cost, the cost
counted; with `limits.spend`, the eval ends failed once it spends that, and samples no more."""

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.names import named
from rollout_train.cluster import Cluster
from rollout_train.evals import make_suite, suite_entry
from rollout_train.gateway.turns import TURNS
from rollout_train.jobs import Run, SpendReached, ran
from rollout_train.record import ENDS, RESULTS, newest_record, table
from rollout_train.run_settings import RunSettings
from rollout_train.stores import Stores
from tests.hosted_apis import FakeApi, Said, openai_api
from tests.local_ray import LocalRay
from tests.rollout_train.clusters import WORDS, a_cluster

KEY = "ROLLOUT_TEST_EVAL_KEY"
HOSTED = """
[inference.hosted]
kind = "api"
endpoint = "rollout_openai:hosted"
api_key_env = "{key}"
base_url = "{url}"
concurrency = 2
[inference.hosted.models."model-a"]
context = 8192
cost = {{ input = 2.0, output = 10.0 }}
"""
EVAL: dict[str, JsonValue] = {
    "kind": "eval", "environment": WORDS, "eval.suite": "held-out", "channels.policy.provider": "hosted",
    "channels.policy.model": "model-a", "channels.policy.thinking_tokens": 64, "channels.policy.answer_tokens": 16,
    "episodes_at_once": 1,
}  # fmt: skip
TURN = Said(text="yes", input_tokens=100_000, output_tokens=10_000)
"""Each turn: 100,000 tokens in at $2 a million and 10,000 out at $10 a million, $0.30."""


async def an_eval(cluster: Cluster, settings: dict[str, JsonValue], name: str) -> Run:
    stores = Stores.open(cluster)
    entry = suite_entry(WORDS, named(WORDS), eval_data="words-held-out", episodes=1)
    await make_suite(stores.ledger, "held-out", [entry])
    made = await stores.registry.create(name)
    return Run(cluster, stores, RunSettings({**settings, "name": name}), made)


async def spent(run: Run) -> list[float]:
    ledger = run.ledger
    names = [each for each in await ledger.tables() if each.startswith(f"runs/{run.run.id}/{TURNS}/")]
    entries = [cast(dict[str, Any], entry) for name in names for entry in (await ledger.read(name)).values()]
    return [float(entry["spend"]) for entry in entries]


async def test_an_eval_plays_a_suite_on_a_hosted_api_and_counts_what_it_spent(
    tmp_path: Path, local_ray: LocalRay, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY, "test-key")
    with openai_api(FakeApi(default=TURN)) as api:
        cluster = a_cluster(tmp_path, more=HOSTED.format(key=KEY, url=api.url))
        evaluation = await an_eval(cluster, EVAL, "held out on a hosted model")
        await ran(evaluation)
    ledger, run = evaluation.ledger, evaluation.run.id
    results = await ledger.read(table(run, RESULTS))
    assert len(results) == 6 and len(api.requests) == 6  # (three rows, two seeds, one episode each, one turn each)
    solved = [bool(said) for each in results.values() for said in cast(dict[str, list[Any]], each)["solved"]]
    assert sum(solved) == 2  # (it says "yes")
    assert newest_record(await ledger.read(table(run, ENDS)))["how"] == "finished"
    assert await spent(evaluation) == pytest.approx([0.3] * 6)
    assert api.requests[0]["max_output_tokens"] == 80 and api.headers[0]["authorization"] == "Bearer test-key"
    assert not evaluation.hosts and evaluation.on_apis["policy"].provider.name == "hosted"  # (no engines of its own)


async def test_an_eval_ends_once_it_spends_its_limit(
    tmp_path: Path, local_ray: LocalRay, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY, "test-key")
    with openai_api(FakeApi(default=TURN)) as api:
        cluster = a_cluster(tmp_path, more=HOSTED.format(key=KEY, url=api.url))
        evaluation = await an_eval(cluster, {**EVAL, "limits.spend": 0.5}, "held out within a limit")
        with pytest.raises(SpendReached, match=r"spent \$0.60, which reaches limits.spend \$0.5"):
            await ran(evaluation)
    assert len(api.requests) == 2  # (one at a time: the second reaches the limit, and no third is sampled)
    ended = newest_record(await evaluation.ledger.read(table(evaluation.run.id, ENDS)))
    assert ended["how"] == "failed" and "limits.spend" in ended["detail"]
    assert await spent(evaluation) == pytest.approx([0.3, 0.3])
