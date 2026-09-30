"""The scripted model of the agent sessions tests, importable by server processes (`agents serve --providers`)."""

import asyncio

from rollout.core.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    Role,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from rollout.core.harness import DirectModel
from rollout.core.testing import tool_call_reply


def call(tool: str, /, **arguments: object) -> Message:
    return tool_call_reply(ToolCall(call_id=f"c-{tool}", name=tool, arguments=arguments))  # type: ignore[arg-type]


def script(request: SampleRequest) -> Message:
    """Each session's model: decided by who is asking and what it last saw."""
    messages = request.context.append
    me = messages[0].text.split('"')[1]
    last = messages[-1]
    results = [b.result for b in last.content if isinstance(b, ToolResultBlock)]
    last_text = "".join(p.text for r in results for p in r.content if isinstance(p, Text)) or last.text
    if me == "lead":
        if last.role is Role.USER and last.text.startswith("[message from worker]"):
            return Message.assistant("The worker reports 42.")
        if last.role is Role.USER:
            return call("create_session", name="worker", instructions="Compute 6*7 in your shell and send it to lead.")
        return Message.assistant("I started a worker.")
    if last.role is Role.USER:
        return call("shell", command="echo $((6*7)) && (cat /etc/alpine-release 2>/dev/null || echo $WORKSPACE)")
    if last_text.startswith("exit 0"):
        return call("send_message", to="lead", text=last_text.splitlines()[1])
    return Message.assistant("Reported to lead.")


class ScriptModel:
    """Serves `script`, a little slowly, so a server killed mid-conversation has model calls in flight."""

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=100_000, max_output_tokens=1_000)

    async def sample(self, request: SampleRequest) -> SampleResult:
        await asyncio.sleep(0.5)
        return SampleResult(
            message=script(request), finish_reason=FinishReason.STOP, usage=Usage(context_used=1, context_limit=100_000)
        )

    async def cancel(self, effect_id: str) -> None:
        pass


def _script_model(model: DirectModel) -> ScriptModel:
    return ScriptModel()


PROVIDERS = {"scripted": _script_model}
