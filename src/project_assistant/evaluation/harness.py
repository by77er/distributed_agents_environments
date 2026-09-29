"""Runs evaluation scenarios against the assistant's HTTP API and measures success, quality, cost and latency."""

import asyncio
import json
import re
import statistics
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from project_assistant.evaluation.fixture import create_fixture_repository, ground_truth
from project_assistant.evaluation.scenarios import SCENARIOS, AwaitUnprompted, Observed, Say, Scenario
from project_assistant.http import create_app
from project_assistant.service import AssistantService, Settings
from rollout.core.contracts import ContextDelta, Message, ModelEndpoint, RunEventType, SampleRequest, context_digests
from rollout.core.harness import DirectModel, SamplingParameters
from rollout.core.local import EndpointFactory

JUDGE_PROMPT = """You grade replies from an assistant that answers questions about a code repository.
You are given the question, the key facts a correct answer must contain, and the repository itself: every file with
numbered lines, and the git history. Details that agree with the repository are correct, including line numbers,
commit hashes and values the key facts do not mention. Penalize only claims that contradict the repository or the key
facts, or that the repository cannot support.

Score each from 1 (poor) to 5 (excellent):
- correctness: contains the key facts and nothing false;
- grounding: points to specific files, lines or commits, accurately;
- concision: answers the question directly, without padding or unrequested detail.
Reply with JSON only: {"correctness": n, "grounding": n, "concision": n, "rationale": "one sentence"}."""


@dataclass
class Judgment:
    turn: int
    correctness: int
    grounding: int
    concision: int
    rationale: str


@dataclass
class ScenarioRun:
    scenario: str
    repeat: int
    success: bool = False
    checks: dict[str, bool] = field(default_factory=dict[str, bool])
    replies: list[str | None] = field(default_factory=list[str | None])
    latencies: list[float] = field(default_factory=list[float])
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    judgments: list[Judgment] = field(default_factory=list[Judgment])
    error: str | None = None


@dataclass(frozen=True)
class EvaluationSettings:
    model: str = "gpt-6-astra"
    reasoning_effort: str | None = "low"
    judge_model: str = "gpt-6-astra"
    repeats: int = 1
    concurrency: int = 3
    judge: bool = True
    scenarios: Sequence[str] = ()
    """Names to run; empty runs all."""
    durable: bool = False
    """Run on the DurableRunner. DBOS allows one runner per process, so scenario runs then go one at a time."""


async def evaluate(settings: EvaluationSettings, providers: Mapping[str, EndpointFactory]) -> list[ScenarioRun]:
    selected = [s for s in SCENARIOS if not settings.scenarios or s.name in settings.scenarios]
    provider_name = next(iter(providers))
    judge = providers[provider_name](
        DirectModel(
            provider=provider_name, model=settings.judge_model, sampling=SamplingParameters(reasoning_effort="low")
        )
    )
    limit = asyncio.Semaphore(1 if settings.durable else settings.concurrency)
    with tempfile.TemporaryDirectory(prefix="tidepool-") as directory:
        repository = create_fixture_repository(Path(directory) / "tidepool")
        truth = ground_truth(repository)

        async def one(scenario: Scenario, repeat: int) -> ScenarioRun:
            async with limit:
                name = f"{scenario.name.replace(' ', '-')}-{repeat}"
                service = AssistantService(
                    Settings(
                        repository=repository,
                        model=settings.model,
                        reasoning_effort=settings.reasoning_effort,
                        notes_path=Path(directory) / f"notes-{name}.sqlite",
                        state=Path(directory) / f"state-{name}" if settings.durable else None,
                    ),
                    providers,
                )
                await service.start()
                try:
                    result = await run_scenario(service, scenario, repeat)
                    # Judge before closing: closing DBOS shuts down the event loop's default executor.
                    if settings.judge and result.error is None:
                        result.judgments = await judge_replies(judge, scenario, result, truth)
                finally:
                    await service.close()
                return result

        return list(await asyncio.gather(*(one(s, r) for s in selected for r in range(settings.repeats))))


async def run_scenario(service: AssistantService, scenario: Scenario, repeat: int) -> ScenarioRun:
    result = ScenarioRun(scenario.name, repeat)
    observed = Observed()
    conversations: set[str] = set()
    transport = httpx.ASGITransport(app=create_app(service))
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://assistant", timeout=600) as client:
            for turn in scenario.turns:
                conversations.add(turn.conversation)
                started = time.monotonic()
                if isinstance(turn, Say):
                    response = await client.post(
                        f"/conversations/{turn.conversation}/messages",
                        json={"text": turn.text, "priority": turn.priority.value, "wait": turn.wait},
                    )
                    body = response.json()
                    observed.replies.append(body.get("reply") if turn.wait else None)
                    if turn.wait and response.status_code != 200:
                        raise RuntimeError(f"turn failed ({response.status_code}): {body}")
                else:
                    observed.replies.append(await _unprompted_reply(service, turn))
                result.latencies.append(round(time.monotonic() - started, 2))
        _collect(service, conversations, observed, result)
        result.checks = {name: check(observed) for name, check in scenario.checks}
        result.success = all(result.checks.values())
    except Exception as error:  # a failed scenario is a result, not a crash of the evaluation
        result.error = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        for conversation in conversations:
            await service.cancel(conversation)
    result.replies = observed.replies
    return result


async def _unprompted_reply(service: AssistantService, turn: AwaitUnprompted) -> str | None:
    before = sum(1 for entry in service.transcript(turn.conversation) if entry.role == "assistant")
    deadline = time.monotonic() + turn.seconds
    while time.monotonic() < deadline:
        replies = [entry for entry in service.transcript(turn.conversation) if entry.role == "assistant"]
        if len(replies) > before:
            return replies[before].text
        await asyncio.sleep(0.5)
    return None


def _collect(service: AssistantService, conversations: set[str], observed: Observed, result: ScenarioRun) -> None:
    """Tool calls, model calls and tokens, from the runs' events (the same for both runners)."""
    for conversation in conversations:
        for run in service.runs(conversation):
            for event in run.recorded_events():
                data = event.payload if isinstance(event.payload, dict) else {}
                completion = data.get("payload")
                if event.type is not RunEventType.EFFECT_COMPLETED or not isinstance(completion, dict):
                    continue
                usage, message = completion.get("usage"), completion.get("message")
                if not isinstance(usage, dict) or not isinstance(message, dict):
                    continue
                result.model_calls += 1
                result.input_tokens += int(usage.get("input_tokens") or 0)  # type: ignore[arg-type]
                result.output_tokens += int(usage.get("output_tokens") or 0)  # type: ignore[arg-type]
                content = message.get("content")
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_call":
                        observed.tool_calls.append(str(block.get("name")))
    observed.notes = [f"{title}\n{body}" for _, title, body in service.notes.notes()]
    result.tool_calls = len(observed.tool_calls)


async def judge_replies(judge: ModelEndpoint, scenario: Scenario, result: ScenarioRun, truth: str) -> list[Judgment]:
    judgments: list[Judgment] = []
    for index, turn in enumerate(scenario.turns):
        reply = result.replies[index] if index < len(result.replies) else None
        if not isinstance(turn, Say) or turn.reference is None or not reply:
            continue
        question = (
            f"Question: {turn.text}\n\nKey facts: {turn.reference}\n\nRepository:\n{truth}\n\nReply to grade:\n{reply}"
        )
        messages = [Message.system(JUDGE_PROMPT), Message.user(question)]
        request = SampleRequest(
            effect_id=f"judge:{scenario.name}:{result.repeat}:{index}",
            arguments_digest="",
            session_id="judge/grader",
            context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
        )
        try:
            text = (await judge.sample(request)).message.text
            scores = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))  # type: ignore[union-attr]
            judgments.append(
                Judgment(
                    index,
                    int(scores["correctness"]),
                    int(scores["grounding"]),
                    int(scores["concision"]),
                    str(scores.get("rationale", "")),
                )
            )
        except Exception as error:  # an unparseable grade is skipped and reported
            print(f"judge failed for {scenario.name} turn {index + 1}: {error}")
    return judgments


def summarize(runs: list[ScenarioRun]) -> str:
    """A Markdown table: one row per scenario, then the totals."""
    lines = [
        "| Scenario | Success | Correctness | Grounding | Concision | Seconds per turn | Model calls "
        "| Tokens in / out |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = list(dict.fromkeys(run.scenario for run in runs))
    for name in [*names, "all"]:
        group = [run for run in runs if name in ("all", run.scenario)]
        judged = [judgment for run in group for judgment in run.judgments]

        def mean(values: list[float]) -> str:
            return f"{statistics.mean(values):.1f}" if values else "n/a"

        successes = sum(run.success for run in group)
        lines.append(
            f"| {'**all**' if name == 'all' else name} | {successes}/{len(group)} "
            f"| {mean([j.correctness for j in judged])} | {mean([j.grounding for j in judged])} "
            f"| {mean([j.concision for j in judged])} | {mean([s for run in group for s in run.latencies])} "
            f"| {mean([run.model_calls for run in group])} "
            f"| {mean([run.input_tokens for run in group])} / {mean([run.output_tokens for run in group])} |"
        )
    failures = [
        f"- {run.scenario} #{run.repeat + 1}: " + (run.error or ", ".join(n for n, ok in run.checks.items() if not ok))
        for run in runs
        if not run.success
    ]
    return "\n".join(lines + (["", "Failed checks:", *failures] if failures else []))


def save(runs: list[ScenarioRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(run) for run in runs], indent=2))
