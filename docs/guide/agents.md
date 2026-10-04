# Agents

An agent is the policy side of the loop. Each turn it decides what the model sees and turns the model's output into
one reply. Any agent runs on any task, so training and evaluation can vary one while holding the other fixed.

## The default agent

`Agent()` samples the `policy` model slot once per turn, with an optional system prompt followed by the history,
shaped by the task's context hints, and offers the tools the task exposes that turn.

```py
class Agent:
    system_prompt: str | None = None

    def __init__(self, configuration: Any = None) -> None:
        self.configuration = configuration

    def select_context(self, history: History, hints: ContextHints) -> list[Message]:
        system = [Message.system(self.system_prompt)] if self.system_prompt else []
        return system + history.messages(hints)

    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        return await run.model.sample(self.select_context(history, run.context_hints), tools=tools)
```

Override `select_context` to change what the model sees, `act` to change how a reply is produced, or set
`system_prompt`. A runner creates the agent from its class and an optional configuration
(`agent_program(MyTask, MyAgent, agent_configuration=...)`), again each time the durable runner resumes the run, so
`__init__` must be deterministic.

## What the model sees

`History.messages(hints)` returns the episode as messages: each turn's reply, then its observation. Tasks state
how much history is useful with `context_hints`; agents should honor them, and the default agent does.

| `HistoryShape` | The model sees |
|---|---|
| `FULL` (default) | every message |
| `LATEST_OBSERVATION` | only the latest observation, for tasks whose observations are complete states. If it holds tool results, the reply whose calls they answer is included too. |
| `WINDOW` | the start observation and the most recent `window` turns |

```python
import asyncio

from rollout.contracts import Message
from rollout.harness import Agent, ContextHints, HistoryShape, Observation, RunContext, Task, rollout
from rollout.testing import local_run


class Board(Task):
    """Each observation is the complete board, so earlier ones are noise."""

    max_turns = 3
    context_hints = ContextHints(history=HistoryShape.LATEST_OBSERVATION)

    async def start(self, run: RunContext) -> Observation:
        return Observation("Board: . . .")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return Observation(f"Board after move {run.turn + 1}")


class Player(Agent):
    system_prompt = "You are playing a board game. Reply with one move."


async def main() -> None:
    task = Board()
    run, endpoint = local_run(task, replies=["a1", "b2", "c3"])
    await rollout(task, Player(), run)

    third_request = endpoint.requests[2]
    assert [message.text for message in third_request.context.append] == [
        "You are playing a board game. Reply with one move.",
        "Board after move 2",
    ]


asyncio.run(main())
```

`select_context` returns the complete message list for the turn, and every sample sends it whole. A context that
is not an extension of the previous turn's (a window moving, a summary replacing old turns) needs no special
handling. For episodes that outgrow the model's context, [memory](../libraries/rollout/memory.md) keeps one that fits.

## How the agent acts

`act` returns exactly one assistant `Message`: the action the task responds to. It may sample more than once,
for example to plan and then act. Only the returned message becomes the turn's reply.

```python
class PlanThenAct(Agent):
    async def act(self, run, history, tools):
        context = self.select_context(history, run.context_hints)
        plan = await run.model.sample([*context, Message.user("Think step by step. Do not answer yet.")])
        return await run.model.sample([*context, plan, Message.user("Now answer.")], tools=tools)


async def plan_then_act() -> None:
    task = Board()
    run, endpoint = local_run(task, replies=["Center is strongest.", "b2"] * 3)
    await rollout(task, PlanThenAct(), run)
    assert [turn.reply.text for turn in run.history.turns[1:] if turn.reply] == ["b2", "b2", "b2"]
    assert len(endpoint.requests) == 6  # two samples per turn


asyncio.run(plan_then_act())
```

Samples that should not be trained on as the policy (summaries, critiques by a different model) belong on a
separate model slot that the task declares.

## The `Model` interface

`run.model` is the `policy` slot; `run.models[name]` is any slot the task declares.

| Member | Meaning |
|---|---|
| `await sample(messages, *, tools=(), max_output_tokens=None, tool_choice=None, links=())` | one assistant `Message`; `links` say how the request follows from earlier ones of the slot ([model endpoint](../libraries/rollout/contracts/model-endpoint.md#what-an-endpoint-is-asked)) |
| `capabilities` | the slot's `CapabilityContract`: the context limit and the maximum output tokens |
| `usage` | the latest sample's `Usage` (`context_used`, `context_limit`, and `input_tokens` and `output_tokens` where the endpoint counts them), or `None` before the first; for compaction decisions |
| `address()` | a `ModelAddress` (`base_url`, `api_key`, `model`) at which a harness that brings its own loop reaches this slot's model; raises `RuntimeError` when the endpoint is not served over HTTP |

`sample` raises `ContractViolation` when `max_output_tokens` exceeds the contract. It retries an endpoint that
reports `Overloaded` or `InternalError` three times, with exponential backoff. Other errors propagate unchanged;
`ContextOverflow` means the context does not fit, and an agent can compact and retry.

Each reply carries the `effect_id` of the sample that produced it in `reply.meta["effect_id"]`. `meta` is never
shown to the model.

## What an agent never sees

Agents see canonical messages, the tools offered this turn, the slot's capability contract and usage, and the
task's context hints. They never see tokens, logprobs, which policy or weights version answered, engines,
environments or credentials. The same agent code therefore runs against an API model, a local engine being
trained, or a script.

## Interruption

A message delivered with mode `INTERRUPT` while the agent is acting cancels `act`, including any sample in flight,
and the loop continues with `Task.resume`. Agents need no code for this. See [conversations](conversations.md).
