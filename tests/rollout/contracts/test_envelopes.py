from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rollout.contracts import (
    CapabilityContract,
    ContextDelta,
    EffectCompletion,
    EffectStatus,
    FinishReason,
    Message,
    NamedToolChoice,
    RunEvent,
    RunEventType,
    SampleRequest,
    SampleResult,
    ToolChoiceMode,
    ToolSpecification,
    Usage,
    context_digests,
)

NOW = datetime(2026, 9, 28, tzinfo=UTC)


def test_effect_completions_round_trip() -> None:
    completion = EffectCompletion(effect_id="r_x:0:3", status=EffectStatus.OUTCOME_UNKNOWN)
    assert EffectCompletion.model_validate_json(completion.model_dump_json()) == completion


def test_run_event_round_trip() -> None:
    event = RunEvent(run_id="r_x", seq=0, type=RunEventType.RUN_CREATED, recorded_at=NOW, payload={"labels": {}})
    assert event.schema_version == 1
    assert RunEvent.model_validate_json(event.model_dump_json()) == event


def test_sample_request_and_result_round_trip() -> None:
    messages = [Message.user("hi")]
    request = SampleRequest(
        effect_id="r_x:0:0",
        arguments_digest="d",
        session_id="r_x/policy",
        context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
        tools=[ToolSpecification(name="a")],
        tool_choice=NamedToolChoice(name="a"),
    )
    assert SampleRequest.model_validate_json(request.model_dump_json()) == request
    assert request.model_copy(update={"tool_choice": ToolChoiceMode.AUTO}).tool_choice is ToolChoiceMode.AUTO
    result = SampleResult(
        message=Message.assistant("hello"),
        finish_reason=FinishReason.STOP,
        usage=Usage(context_used=10, context_limit=4096),
    )
    assert SampleResult.model_validate_json(result.model_dump_json()) == result
    contract = CapabilityContract(context_limit=4096, max_output_tokens=512)
    assert CapabilityContract.model_validate_json(contract.model_dump_json()) == contract


def test_sample_request_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValidationError):
        SampleRequest(
            effect_id="r_x:0:0",
            arguments_digest="d",
            session_id="r_x/policy",
            context=ContextDelta(digest="d"),
            tools=[ToolSpecification(name="a"), ToolSpecification(name="a")],
        )


def test_sample_result_is_an_assistant_message() -> None:
    with pytest.raises(ValidationError):
        SampleResult(
            message=Message.user("hi"),
            finish_reason=FinishReason.STOP,
            usage=Usage(context_used=1, context_limit=2),
        )
