"""What the gateway's tests share: an engine whose answer depends only on its prompt (so that replicas in other
processes, and an in-process recorder, sample the same), a keyring, and a gateway over a channel of this process."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import httpx

from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import Gateway, Grant, Keyring, TurnStore
from rollout_train.inference import Channel, Generation, Limits, Routes
from rollout_train.ledger import Fence, FileLedger, Ledger
from rollout_train.recorder import Recorder, Renderer
from rollout_train.recorder.compat import create_app as recorder_app
from rollout_train.testing import PlainRenderer

SECRETS = [("k2", "a-newer-secret-of-thirty-two-bytes!!"), ("k1", "an-older-secret-of-thirty-two-bytes!")]
"""The signing secret first."""


def keyring() -> Keyring:
    return Keyring.parse(SECRETS)


class EchoEngine:
    """Answers a prompt that offers tools (in the plain format) with `call move {"steps": N}` when it is a multiple of
    three tokens long, and any other with a line saying how long it was; logprobs are exact in 32 bits. It waits
    `delay` seconds first."""

    max_model_len = 32_768
    processes: Sequence[int] = ()

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.prompts: list[list[int]] = []

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
    ) -> Generation:
        self.prompts.append(list(prompt))
        if self.delay:
            await asyncio.sleep(self.delay)
        size = len(prompt)
        offered = [ord(character) for character in "tools: "] == list(prompt[:7])
        text = f'call move {{"steps": {size % 9}}}\n' if offered and size % 3 == 0 else f"I saw {size} tokens.\n"
        tokens = [ord(character) for character in text][:max_tokens]
        return Generation(tokens, [-((token % 7) + 1) / 8 for token in tokens], "stop")

    async def load_adapter(self, name: str, path: str) -> None: ...
    async def remove_adapter(self, name: str) -> None: ...
    async def load_weights(self, path: str) -> None: ...
    async def sleep(self) -> None: ...
    async def wake(self) -> None: ...
    def close(self) -> None: ...


def echo_engine(model: str, **options: Any) -> EchoEngine:
    """What a profile names (`tests.rollout_train.gateway.support:echo_engine`)."""
    return EchoEngine(float(options.get("delay", 0.0)))


def echo_channel(name: str = "policy", delay: float = 0.0, **limits: Any) -> Channel:
    return Channel(name, [EchoEngine(delay)], cast(Renderer, PlainRenderer()), Limits(**limits))


def stores(directory: Path) -> tuple[FileLedger, FileBlobStore]:
    return FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs")


def gateway_over(
    channel: Channel | None, ledger: Ledger, blobs: FileBlobStore, routes: Routes | None = None
) -> Gateway:
    """A gateway sampling `channel` (its model's own weights named `base`), and the channels `routes` route."""
    channels = {channel.name: channel} if channel is not None else {}
    return Gateway(TurnStore(ledger, blobs), keyring(), channels, routes, dict.fromkeys(channels, "base"))


async def grant(
    ledger: Ledger, *, run: str = "train", run_id: str = "r_1", slot: str = "policy", attempt: int = 1, **fields: Any
) -> Grant:
    """A grant for an attempt of episode 1/1, under that episode's fence (taken here)."""
    fence: Fence = fields.pop("fence", None) or await ledger.take(f"runs/{run}/episodes/1/1")
    expires = fields.pop("expires", time.time() + 3600)
    return Grant(
        run=run,
        run_id=run_id,
        slot=slot,
        channel=fields.pop("channel", "policy"),
        fence=fence,
        expires=expires,
        episode="1/1",
        attempt=attempt,
        **fields,
    )


TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "move", "description": "Move.", "parameters": {"type": "object", "properties": {}}},
}


def client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


def bearer(key: str, effect: str | None = None, **headers: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", **({"Idempotency-Key": effect} if effect else {}), **headers}


async def posted(http: httpx.AsyncClient, path: str, body: Any, headers: Mapping[str, str]) -> httpx.Response:
    """Post, and post again under the same request id while the gateway (or a proxy in front of it) fails."""
    failure = ""
    for _ in range(8):
        try:
            answer = await http.post(path, json=body, headers=dict(headers))
        except httpx.TransportError as error:
            failure = repr(error)
        else:
            if answer.status_code < 500:
                return answer
            failure = f"{answer.status_code} {answer.text}"
        await asyncio.sleep(0.25)
    raise AssertionError(f"the gateway did not answer: {failure}")


async def converse(
    http: httpx.AsyncClient,
    key: str,
    turns: int = 5,
    prefix: str = "",
    during: Callable[[int], Awaitable[None]] | None = None,
) -> list[dict[str, Any]]:
    """A harness's exchange over Chat Completions: it appends each reply and a result, edits its context once (as a
    compaction would), and sends one request exactly again under a new id. `during` is awaited beside each turn's
    request, told its number."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Walk."}, {"role": "user", "content": "Go."}]
    replies: list[dict[str, Any]] = []
    path = f"{prefix}/v1/chat/completions"
    for number in range(turns):
        if number == 3:
            messages = [messages[0], {"role": "user", "content": "You walked a while. Go on."}]
        body = {"model": "anything", "messages": messages, "tools": [TOOL]}
        asked = asyncio.ensure_future(posted(http, path, body, bearer(key, f"e{number}")))
        if during is not None:
            await during(number)
        answer = await asked
        assert answer.status_code == 200, answer.text
        replies.append(answer.json())
        said: dict[str, Any] = answer.json()["choices"][0]["message"]
        messages.append({name: value for name, value in said.items() if value is not None})
        calls: list[dict[str, Any]] = said.get("tool_calls") or []
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Moved."})
        if not said.get("tool_calls"):
            messages.append({"role": "user", "content": f"Turn {number}."})
    again = {"model": "anything", "messages": messages, "tools": [TOOL]}
    for effect in ("e-again", "e-again-later"):  # (the same prompt twice: the later replaces the earlier)
        assert (await posted(http, path, again, bearer(key, effect))).is_success
    return replies


async def recorded_in_process(turns: int = 5) -> list[Any]:
    """What the in-process recorder exports for the same exchange."""
    recorder = Recorder({"policy": echo_channel()}, base_url="http://recorder/v1")
    address = recorder.endpoint(RecordedModel(channel="policy")).address("r_1/policy")
    async with client(recorder_app(recorder)) as http:
        await converse(http, address.api_key, turns)
    return list(recorder.export("r_1/policy"))
