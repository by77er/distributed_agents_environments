from rollout.core.contracts import Message, Role, ToolCall, ToolResult, ToolResultBlock
from rollout.core.harness import ContextHints, History, HistoryShape, Observation, Turn
from rollout.core.testing import tool_call_reply


def history() -> History:
    result = History()
    result.append(Turn(reply=None, observation=Observation("start")))
    for index in range(3):
        result.append(Turn(reply=Message.assistant(f"reply {index}"), observation=Observation(f"observation {index}")))
    return result


def texts(messages: list[Message]) -> list[str]:
    return [message.text for message in messages]


def test_full_history() -> None:
    assert len(history().messages()) == 7
    assert history().messages(ContextHints()) == history().messages()


def test_window_keeps_the_start_and_recent_turns() -> None:
    hints = ContextHints(history=HistoryShape.WINDOW, window=1)
    assert texts(history().messages(hints)) == ["start", "reply 2", "observation 2"]


def test_latest_observation() -> None:
    hints = ContextHints(history=HistoryShape.LATEST_OBSERVATION)
    assert texts(history().messages(hints)) == ["observation 2"]


def test_latest_observation_keeps_the_calls_its_tool_results_answer() -> None:
    reply = tool_call_reply(ToolCall(call_id="c1", name="t", arguments={}))
    answer = Message(role=Role.TOOL, content=[ToolResultBlock(call_id="c1", result=ToolResult())])
    result = history()
    result.append(Turn(reply=reply, observation=Observation(answer)))
    assert history_roles(result.messages(ContextHints(history=HistoryShape.LATEST_OBSERVATION))) == [
        Role.ASSISTANT,
        Role.TOOL,
    ]


def history_roles(messages: list[Message]) -> list[Role]:
    return [message.role for message in messages]
