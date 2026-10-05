"""Test doubles for what stands above a run: an engine that answers from a script, a trainer that trains nothing, a
token format simple enough to read, and a gateway in this process that records what they sample. With them an
environment, an algorithm or a whole run built from its settings can be tried without a model or a GPU."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rollout.contracts import (
    ContextDelta,
    Message,
    Role,
    SampleRequest,
    Text,
    ToolCall,
    ToolResultBlock,
    ToolSpecification,
    context_digests,
)
from rollout.harness.blobs import Blobs
from rollout.harness.hooks import RunHooks
from rollout_train.gateway import Attempt, Gateway, GatewayEndpoints, Keyring, TurnStore
from rollout_train.inference import Channel, Generation, Limits, Routes, Scores
from rollout_train.inference.channel import scored_range
from rollout_train.ledger import Ledger
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import Tokenizer

__all__ = [
    "SECRETS",
    "Characters",
    "PlainRenderer",
    "Policy",
    "ScriptedEngine",
    "ScriptedTrainer",
    "admitted",
    "gateway_endpoints",
    "keyring",
    "plain_channel",
    "plain_renderer",
    "sample_request",
    "scripted_engine",
    "scripted_top",
]

SECRETS = [("k2", "a-newer-secret-of-thirty-two-bytes!!"), ("k1", "an-older-secret-of-thirty-two-bytes!")]
"""What a test's gateway (`gateway_endpoints`) signs keys with (the first) and takes keys signed with (each)."""


class ScriptedEngine:
    """Answers each generate with the next scripted (text, finish reason), or with `always` once the script is
    spent; logprobs are -0.5 per token. Scores each token at -0.25. Keeps what it was asked and told. With `gate` (a
    directory), its `N`th load of full weights (from 1) writes `loading-N` there and waits until a file `N` is there
    too."""

    max_model_len = 32_768
    processes: Sequence[int] = ()

    def __init__(
        self,
        tokenizer: Tokenizer,
        script: Sequence[tuple[str, str]] = (),
        *,
        always: Sequence[tuple[str, str]] = (),
        gate: Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.gate = gate
        self.loads = 0
        self.script = list(script)
        self.always = list(always)
        self.prompts: list[list[int]] = []
        self.budgets: list[int] = []
        self.adapters: list[str | None] = []
        """The adapter each request named."""
        self.told: list[str] = []
        """`load X`, `remove X`, `sleep`, `wake`, `close`, in order."""
        self.scored: list[tuple[list[int], int, int]] = []
        """What each scoring request asked for: the tokens up to its end, its start, and its `top`."""

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        top: int = 0,
    ) -> Generation:
        self.prompts.append(list(prompt))
        self.budgets.append(max_tokens)
        self.adapters.append(adapter)
        if not self.script:
            self.script = list(self.always)
        text, finish = self.script.pop(0)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        top_tokens, top_logprobs = scripted_top(tokens, top, -0.5)
        return Generation(tokens, [-0.5] * len(tokens), finish, top_tokens=top_tokens, top_logprobs=top_logprobs)

    async def score(
        self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None
    ) -> Scores:
        """Each token scored at -0.25, its `top` most likely being itself and the tokens after it (`scripted_top`)."""
        end = scored_range(len(tokens), start, end)
        self.scored.append((list(tokens[:end]), start, top))
        self.adapters.append(adapter)
        top_tokens, top_logprobs = scripted_top(tokens[start:end], top)
        return Scores(start, [-0.25] * (end - start), top_tokens=top_tokens, top_logprobs=top_logprobs)

    async def load_adapter(self, name: str, path: str) -> None:
        self.told.append(f"load {name}")

    async def remove_adapter(self, name: str) -> None:
        self.told.append(f"remove {name}")

    async def load_weights(self, path: str) -> None:
        self.loads += 1
        if self.gate is not None:
            await asyncio.to_thread((self.gate / f"loading-{self.loads}").touch)
            while not await asyncio.to_thread((self.gate / str(self.loads)).exists):  # noqa: ASYNC110 (a file lets it go)
                await asyncio.sleep(0.01)
        self.told.append(f"weights {path}")

    async def sleep(self) -> None:
        self.told.append("sleep")

    async def wake(self) -> None:
        self.told.append("wake")

    def close(self) -> None:
        self.told.append("close")


def scripted_top(tokens: Sequence[int], top: int, logprob: float = -0.25) -> tuple[list[list[int]], list[list[float]]]:
    """The `top` most likely tokens a scripted engine gives at each position of `tokens`: the token there (at
    `logprob`) and those after it, each a nat less likely than the one before (none when `top` is 0)."""
    if not top:
        return [], []
    ranks = range(top)
    return [[token + rank for rank in ranks] for token in tokens], [[logprob - rank for rank in ranks] for _ in tokens]


class Characters:
    """A tokenizer of one token per character."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)


class PlainRenderer:
    """A token format for tests: each message is `role: text` on a line, a tool call is `call NAME {json}`, and a
    turn ends with the line. A reply renders back exactly as it was sampled, so a conversation that only grows is
    one segment."""

    name = "plain"
    thinking = None

    def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]:
        lines = [f"tools: {', '.join(tool.name for tool in tools)}"] if tools else []
        for message in messages:
            if message.role is Role.TOOL:
                results = [block for block in message.content if isinstance(block, ToolResultBlock)]
                lines += ["tool: " + "".join(p.text for p in r.result.content if isinstance(p, Text)) for r in results]
            else:
                lines.append(f"{message.role.value}: {self._said(message)}")
        return self.encode("".join(f"{line}\n" for line in lines) + "assistant: ")

    def encode(self, text: str) -> list[int]:
        return Characters().encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return Characters().decode(tokens)

    def stop_token_ids(self) -> list[int]:
        return [ord("\n")]

    def thinking_end_token_ids(self) -> list[int]:
        return []

    def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message:
        text = Characters().decode(completion).removesuffix("\n")
        if text.startswith("call "):
            _, name, arguments = text.split(" ", 2)
            call = ToolCall(call_id=f"c{len(completion)}_{name}", name=name, arguments=json.loads(arguments))
            return Message(role=Role.ASSISTANT, content=[call])
        return Message.assistant(text)

    @staticmethod
    def _said(message: Message) -> str:
        calls = [f"call {call.name} {json.dumps(dict(call.arguments))}" for call in message.tool_calls]
        return message.text + "".join(calls)


def plain_channel(script: Sequence[tuple[str, str]] = (), *, name: str = "policy", **options: Any) -> Channel:
    """A channel over a scripted engine in the plain format; `always=` repeats a script for ever."""
    engine = ScriptedEngine(cast(Tokenizer, Characters()), script, always=options.pop("always", ()))
    return Channel(name, [engine], cast(Renderer, PlainRenderer()), Limits(**options))


def keyring() -> Keyring:
    """The keys of `SECRETS`."""
    return Keyring.parse(SECRETS)


def gateway_endpoints(
    *channels: Channel,
    ledger: Ledger,
    blobs: Blobs,
    url: str | None = None,
    routes: Routes | None = None,
    hooks: Sequence[RunHooks] = (),
    models: Mapping[str, str] | None = None,
) -> GatewayEndpoints:
    """Endpoints over a gateway in this process that samples `channels` (`models` names each one's base model, by
    channel) and those `routes` route, recording in `ledger` and `blobs`, with the keys of `SECRETS`; `url` is where it
    is served to harnesses, if it is, and `hooks` are told of the samples harnesses ask for there."""
    by_name = {channel.name: channel for channel in channels}
    gateway = Gateway(TurnStore(ledger, blobs), keyring(), by_name, routes, dict(models or {}), hooks=hooks)
    return GatewayEndpoints.of(gateway, url)


class Policy:
    """Channels whose engines are in this process, as a test's training loop and its runners see them: `publish` serves
    new weights on one (what a loop is given to publish with), and `gateway` is what a runner samples through."""

    def __init__(self, *channels: Channel) -> None:
        self.channels = {channel.name: channel for channel in channels}
        self._gateways: dict[str, GatewayEndpoints] = {}

    async def publish(
        self, channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False
    ) -> int:
        return await self.channels[channel].publish(adapter, path, version, full=full)

    def gateway(self, ledger: Ledger, blobs: Blobs) -> GatewayEndpoints:
        """A gateway in this process over the channels, recording in `ledger` and `blobs`: one for each place the
        ledger is, so that a runner made first and an episode runner made after it over the same place share it (the
        runs it admits are the runs the runner plays)."""
        where = str(getattr(ledger, "directory", id(ledger)))
        if where not in self._gateways:
            self._gateways[where] = gateway_endpoints(*self.channels.values(), ledger=ledger, blobs=blobs)
        return self._gateways[where]


async def admitted(endpoints: GatewayEndpoints, run_id: str, run: str = "train") -> Attempt:
    """Admit a run that no episode runner plays (one a test starts on a runner itself), under a fence of its own."""
    attempt = Attempt(run, await endpoints.store.ledger.take(f"tests/{run_id}"))
    endpoints.admit(run_id, attempt)
    return attempt


def sample_request(
    messages: list[Message],
    effect_id: str = "r_1:0:0",
    *,
    session_id: str = "r_1/ada",
    tools: Sequence[ToolSpecification] = (),
) -> SampleRequest:
    return SampleRequest(
        effect_id=effect_id,
        arguments_digest="d",
        session_id=session_id,
        context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
        tools=tools,
    )


# What a cluster config and run settings can name (`[inference.NAME] engine = "rollout_train.testing:scripted_engine"`,
# `channels.NAME.renderer = "rollout_train.testing:plain_renderer"`): an engine and a renderer
# that need no GPU.

STARTED: list[ScriptedEngine] = []
"""Every engine `scripted_engine` has made, for a test to look at."""


def scripted_engine(model: str, **options: Any) -> ScriptedEngine:
    """An engine whose policy says yes and no in turn; `fails=true` makes one that cannot start, and `gate` (a
    directory) holds its loads of full weights back until a test lets each go (`ScriptedEngine`)."""
    if options.get("fails"):
        raise RuntimeError("no such device")
    always = [("yes\n", "stop"), ("no\n", "stop")]
    gate = Path(str(options["gate"])) if options.get("gate") else None
    engine = ScriptedEngine(cast(Tokenizer, Characters()), always=always, gate=gate)
    engine.told.append(f"started {model} {sorted(options.items())}")
    STARTED.append(engine)
    return engine


class ScriptedTrainer:
    """A trainer that trains nothing: each step writes an adapter's files that say how many segments it was given (as
    PEFT's are named, so a bridge takes them for an adapter), and the trainer's state beside them. What a cluster
    config's trainer names as its `implementation` in tests."""

    weights = "lora"

    def __init__(self, model: str, *, segment_tokens: int | None = None, segments_per_step: int | None = None,
                 **settings: Any) -> None:  # fmt: skip
        from rollout_train.trainer import Budget

        self.model = model
        self.settings = settings
        self.budget = Budget(segment_tokens, segments_per_step)

    async def step(self, batch: Sequence[Any], *, seed: int, parent: Any, into: Path) -> Any:
        from rollout_train.trainer import STATE, WEIGHTS, Step

        def written() -> None:
            (into / WEIGHTS).mkdir(parents=True)
            (into / WEIGHTS / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": self.model}))
            (into / WEIGHTS / "adapter_model.safetensors").write_text(f"trained on {len(batch)} segments")
            (into / STATE).mkdir()
            (into / STATE / "optimizer.bin").write_text(f"after a step of {len(batch)} segments")

        await asyncio.to_thread(written)
        return Step({"segments": float(len(batch))})


def plain_renderer(model: str) -> Renderer:
    return cast(Renderer, PlainRenderer())
