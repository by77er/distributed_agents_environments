"""Test doubles for what stands above a run: an engine that answers from a script, and a token format simple enough
to read. With them a catalog, an algorithm or a whole profile can be tried without a model or a GPU."""

import json
from collections.abc import Sequence
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
from rollout_train.inference import Channel, Generation, Limits
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import Tokenizer

__all__ = [
    "Characters",
    "PlainRenderer",
    "ScriptedEngine",
    "plain_channel",
    "plain_renderer",
    "sample_request",
    "scripted_engine",
]


class ScriptedEngine:
    """Answers each generate with the next scripted (text, finish reason), or with `always` once the script is
    spent; logprobs are -0.5 per token. Keeps what it was asked and told."""

    max_model_len = 32_768
    processes: Sequence[int] = ()

    def __init__(
        self, tokenizer: Tokenizer, script: Sequence[tuple[str, str]] = (), *, always: Sequence[tuple[str, str]] = ()
    ) -> None:
        self.tokenizer = tokenizer
        self.script = list(script)
        self.always = list(always)
        self.prompts: list[list[int]] = []
        self.budgets: list[int] = []
        self.adapters: list[str | None] = []
        """The adapter each request named."""
        self.told: list[str] = []
        """`load X`, `remove X`, `sleep`, `wake`, `close`, in order."""

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
        self.budgets.append(max_tokens)
        self.adapters.append(adapter)
        if not self.script:
            self.script = list(self.always)
        text, finish = self.script.pop(0)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return Generation(tokens=tokens, logprobs=[-0.5] * len(tokens), finish_reason=finish)

    async def load_adapter(self, name: str, path: str) -> None:
        self.told.append(f"load {name}")

    async def remove_adapter(self, name: str) -> None:
        self.told.append(f"remove {name}")

    async def sleep(self) -> None:
        self.told.append("sleep")

    async def wake(self) -> None:
        self.told.append("wake")

    def close(self) -> None:
        self.told.append("close")


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


# What a profile can name (`rollout_train.testing:scripted_engine`, `:plain_renderer`): an engine and a renderer
# that need no GPU.

STARTED: list[ScriptedEngine] = []
"""Every engine `scripted_engine` has made, for a test to look at."""


def scripted_engine(model: str, **options: Any) -> ScriptedEngine:
    """An engine whose policy says yes and no in turn; `fails=true` makes one that cannot start."""
    if options.get("fails"):
        raise RuntimeError("no such device")
    engine = ScriptedEngine(cast(Tokenizer, Characters()), always=[("yes\n", "stop"), ("no\n", "stop")])
    engine.told.append(f"started {model} {sorted(options.items())}")
    STARTED.append(engine)
    return engine


def plain_renderer(model: str) -> Renderer:
    return cast(Renderer, PlainRenderer())
