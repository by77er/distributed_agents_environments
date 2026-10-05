"""Channels on hosted APIs, as the gateway samples them: a turn is recorded with no tokens and never trained on, with
its reply and what it cost at the model's catalog prices; a rate limit is asked again after a wait, credentials
refused fail at once; the provider's concurrency holds across its channels; a run's spend is counted, over what its
earlier starts spent, and a cap stops it; a run's start that names a hosted API's channel has it built by the
directory. Against fake APIs on this machine: nothing is sent elsewhere, nothing is paid."""

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.contracts import Message, ModelEndpointError, ToolCall, ToolSpecification
from rollout_train.gateway import ChannelDirectory, Gateway, Refused, TurnStore, create_app
from rollout_train.gateway.spending import Spending
from rollout_train.gateway.turns import turns_table
from rollout_train.inference import Limits
from rollout_train.inference.api import ApiChannel, Hosted, priced
from rollout_train.ledger import FileLedger
from rollout_train.providers import ModelOffer, Secret
from rollout_train.testing import keyring, sample_request
from tests.hosted_apis import FakeApi, Refusal, Said, anthropic_api, openai_api
from tests.rollout_train.gateway.support import bearer, client, grant, stores
from tests.rollout_train.gateway.test_directory import started
from tests.rollout_train.test_validation import the_cluster

KEY = "ROLLOUT_TEST_HOSTED_KEY"
OFFER = ModelOffer("model-a", context=8192, cost={"input": 2.0, "cached_input": 0.5, "output": 10.0})
"""Dollars per million tokens: input 2, cached input 0.5, output (and thinking) 10."""
WEATHER = ToolSpecification(name="weather", description="Look up the weather.")


def provider(url: str, endpoint: str = "rollout_openai:hosted", concurrency: int | None = None) -> Hosted:
    return Hosted("hosted", endpoint, Secret(env=KEY), {"model-a": OFFER}, base_url=url, concurrency=concurrency)


@pytest.fixture(autouse=True)
def a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "test-key")


def test_a_reply_costs_its_usage_at_the_catalog_prices() -> None:
    from rollout.contracts import Usage

    usage = Usage(context_used=0, context_limit=1, input_tokens=1000, cached_input_tokens=400, output_tokens=300,
                  thinking_tokens=100)  # fmt: skip
    assert priced(usage, OFFER.cost) == pytest.approx((600 * 2.0 + 400 * 0.5 + 300 * 10.0) / 1e6)
    assert priced(usage, {**OFFER.cost, "thinking": 20.0}) == pytest.approx((1200 + 200 + 2000 + 100 * 20.0) / 1e6)
    assert priced(usage, {}) is None and priced(Usage(context_used=0, context_limit=1), OFFER.cost) is None


@pytest.mark.parametrize(("serve", "endpoint"), [(openai_api, "rollout_openai:hosted"),
                                                 (anthropic_api, "rollout_anthropic:hosted")])  # fmt: skip
async def test_a_hosted_turn_is_recorded_with_no_tokens_never_trained_on_with_its_reply_and_spend(
    tmp_path: Path, serve: Callable[[FakeApi], AbstractContextManager[FakeApi]], endpoint: str
) -> None:
    ledger, blobs = stores(tmp_path)
    reply = Said(text="Sunny.", calls=(("call_1", "weather", {"city": "Porto"}),), input_tokens=1000,
                 cached_input_tokens=400, output_tokens=300, thinking_tokens=100)  # fmt: skip
    with serve(FakeApi(replies=[reply])) as api:
        judge = ApiChannel("judge", provider(api.url, endpoint), "model-a", Limits(thinking=64, answer=32))
        gateway = Gateway(TurnStore(ledger, blobs), keyring(), hosted={"judge": judge})
        granted = await grant(ledger, slot="judge", channel="judge", temperature=0.5, trained=True)
        request = sample_request([Message.user("Weather?")], "r_1:judge:0", session_id="r_1/judge", tools=[WEATHER])
        first = await gateway.sample(granted, request)
        again = await gateway.sample(granted, request)  # (the same request id: the recorded reply, not sampled again)
    assert len(api.requests) == 1 and again.replayed and not first.replayed
    assert first.result.message.text == "Sunny."
    assert first.result.message.tool_calls == [ToolCall(call_id="call_1", name="weather", arguments={"city": "Porto"})]
    (turn,) = await gateway.store.turns("train", "r_1")
    assert turn.sampled_with == () and not turn.trained  # (whatever the grant says: no exact tokens to train on)
    assert list(turn.prompt) == [] and turn.completion == [] and turn.checkpoint == "model-a"
    assert turn.result.usage.cached_input_tokens == 400 and turn.result.usage.thinking_tokens == 100
    assert turn.spend == pytest.approx((600 * 2.0 + 400 * 0.5 + 300 * 10.0) / 1e6)
    (entry,) = (await ledger.read(turns_table("train", "r_1"))).values()
    assert isinstance(entry, dict) and entry["spend"] == pytest.approx(turn.spend)
    assert await gateway.store.sessions("train", "r_1") == {}  # (no segment: nothing to train on)
    assert gateway.spending is not None and await gateway.spending.spent("train") == pytest.approx(turn.spend)
    sent = api.requests[0]
    assert ("max_output_tokens" in sent and sent["max_output_tokens"] == 96) or sent.get("max_tokens") == 96


async def test_a_rate_limit_is_asked_again_after_a_wait_and_refused_credentials_fail_at_once(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    limited = Refusal(429, '{"error": {"code": "rate_limit_exceeded"}}', {"retry-after": "0.05"})
    with openai_api(FakeApi(replies=[limited, Refusal(503), Said(text="Done.")])) as api:
        judge = ApiChannel("judge", provider(api.url), "model-a", backoff=0.01)
        gateway = Gateway(TurnStore(ledger, blobs), keyring(), hosted={"judge": judge})
        granted = await grant(ledger, slot="judge", channel="judge")
        reply = await gateway.sample(granted, sample_request([Message.user("Go.")], "r_1:judge:0",
                                                             session_id="r_1/judge"))  # fmt: skip
        assert reply.result.message.text == "Done." and len(api.requests) == 3
        assert judge.take()["retries"] == 2
        api.replies.extend([Refusal(401, '{"error": {"code": "invalid_api_key"}}')] * 2)
        with pytest.raises(ModelEndpointError, match="refused the credentials"):
            await gateway.sample(granted, sample_request([Message.user("Go.")], "r_1:judge:1", session_id="r_1/judge"))
        assert len(api.requests) == 5 and judge.take()["retries"] == 0  # (the key tried once more, not backed off)
        api.replies.extend([Refusal(429)] * 3)
        tired = ApiChannel("tired", provider(api.url), "model-a", attempts=3, backoff=0.01)
        with pytest.raises(ModelEndpointError):  # (an Overloaded, once its attempts run out)
            await tired.sample(sample_request([Message.user("Go.")], "r_1:judge:2", session_id="r_1/judge"))
        assert len(api.requests) == 8


async def test_a_provider_without_its_key_fails_with_the_variable_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY)
    channel = ApiChannel("judge", provider("http://127.0.0.1:9"), "model-a")
    with pytest.raises(ModelEndpointError, match=f"\\${KEY}"):
        await channel.sample(sample_request([Message.user("Go.")]))


async def test_the_providers_concurrency_holds_across_its_channels() -> None:
    with anthropic_api(FakeApi(delay=0.05)) as api:
        shared = provider(api.url, "rollout_anthropic:hosted", concurrency=2)
        channels = [ApiChannel(name, shared, "model-a") for name in ("judge", "opponent")]
        await asyncio.gather(*(
            channels[number % 2].sample(sample_request([Message.user(f"Go {number}.")], f"r_1:x:{number}"))
            for number in range(8)
        ))  # fmt: skip
    assert len(api.requests) == 8 and api.most_in_flight == 2


async def test_a_runs_spend_counts_its_earlier_starts_and_a_cap_stops_it(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    expensive = Said(input_tokens=100_000, output_tokens=10_000)  # (0.2 + 0.1 dollars)
    with openai_api(FakeApi(default=expensive)) as api:
        judge = ApiChannel("judge", provider(api.url), "model-a")
        earlier = Gateway(TurnStore(ledger, blobs), keyring(), hosted={"judge": judge})
        granted = await grant(ledger, slot="judge", channel="judge")
        await earlier.sample(granted, sample_request([Message.user("1")], "r_1:judge:0", session_id="r_1/judge"))
        later = Gateway(TurnStore(ledger, blobs), keyring(), hosted={"judge": judge}, spending=Spending(ledger))
        assert later.spending is not None
        cap = later.spending.cap({"train"}, 0.5)
        await later.sample(granted, sample_request([Message.user("2")], "r_1:judge:1", session_id="r_1/judge"))
        assert await later.spending.spent("train") == pytest.approx(0.6) and cap.reached.is_set()
        with pytest.raises(ModelEndpointError, match=r"spent \$0.60, which reaches its limits.spend \$0.5"):
            await later.sample(granted, sample_request([Message.user("3")], "r_1:judge:2", session_id="r_1/judge"))
    assert len(api.requests) == 2


async def test_a_hosted_channel_counts_and_scores_no_tokens_and_says_its_contract(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    judge = ApiChannel("judge", provider("http://127.0.0.1:9"), "model-a", Limits(thinking=100, answer=50))
    gateway = Gateway(TurnStore(ledger, blobs), keyring(), hosted={"judge": judge})
    granted = await grant(ledger, slot="judge", channel="judge")
    assert gateway.describe(granted).max_output_tokens == 150 and gateway.describe(granted).context_limit == 8192
    with pytest.raises(Refused, match="renders messages itself"):
        await gateway.count(granted, prompt=None)  # type: ignore[arg-type]
    key = gateway.keyring.mint(granted)
    async with client(create_app(gateway)) as http:
        listed = (await http.get("/v1/models")).json()["data"]
        counted = await http.post("/v1/messages/count_tokens", json={"model": "judge", "messages": []},
                                  headers=bearer(key))  # fmt: skip
    assert [each["id"] for each in listed] == ["judge"] and listed[0]["contract"]["max_output_tokens"] == 150
    assert counted.status_code == 400


async def test_a_runs_start_that_names_a_hosted_api_has_its_channel_built_by_the_directory(tmp_path: Path) -> None:
    ledger, _ = stores(tmp_path)
    cluster = the_cluster()
    settings: dict[str, JsonValue] = {
        "channels.judge.provider": "anthropic", "channels.judge.model": "claude-haiku-4-5-20251001",
        "channels.judge.thinking_tokens": 2048, "channels.judge.answer_tokens": 256, "slots.judge": "judge",
    }  # fmt: skip
    await started(ledger, "r", settings)
    directory = ChannelDirectory.of(cluster, FileLedger(tmp_path / "ledger"))
    assert set(directory.hosted) == {"openai", "anthropic"} and "openai" not in directory.providers
    built = await directory.load("r")
    judge = built["judge"]
    assert isinstance(judge, ApiChannel) and judge.provider.endpoint == "rollout_anthropic:hosted"
    assert judge.limits == Limits(2048, 256) and judge.context_limit == 200_000 and judge.max_output_tokens == 64_000
    assert judge.provider.key == Secret(env="ANTHROPIC_API_KEY") and judge.provider.concurrency == 16
    assert directory.hosted["anthropic"].admission is judge.provider.admission  # (one cap for every run's channels)
