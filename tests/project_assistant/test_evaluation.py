from project_assistant.evaluation.harness import EvaluationSettings, evaluate, summarize
from rollout.contracts import Role, ToolCall
from rollout.harness import DirectModel
from rollout.testing import ScriptedModelEndpoint, tool_call_reply


async def test_a_scenario_is_checked_judged_and_measured() -> None:
    search = ToolCall(call_id="c1", name="search", arguments={"pattern": "def invoice_total"})
    endpoint = ScriptedModelEndpoint(
        [
            tool_call_reply(search),
            "invoice_total is in src/tidepool/billing.py:4.",
            '{"correctness": 5, "grounding": 4, "concision": 5, "rationale": "Right file and function."}',
        ]
    )

    def factory(model: DirectModel) -> ScriptedModelEndpoint:
        return endpoint

    settings = EvaluationSettings(scenarios=["locate a function"], concurrency=1)
    (run,) = await evaluate(settings, providers={"scripted": factory})

    assert run.error is None
    assert run.success, run.checks
    assert run.model_calls == 2
    assert run.tool_calls == 1
    assert [(j.correctness, j.grounding, j.concision) for j in run.judgments] == [(5, 4, 5)]
    table = summarize([run])
    assert "| locate a function | 1/1 | 5.0 | 4.0 | 5.0 |" in table
    assert any(message.role is Role.TOOL for message in endpoint.requests[1].context.append)  # the search result
