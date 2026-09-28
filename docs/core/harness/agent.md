# Agent

Status: **Proposed** · Layer: core · See [ADR-0012](../../decisions/0012-task-agent-loop.md), [ADR-0006](../../decisions/0006-harness-unaware-of-policy.md)

An **Agent** is the policy side of the loop: it decides what the model sees each turn and turns the model's
output into one action (an assistant `Message`). The same task runs under any agent, and the same agent runs on
any task — reinforcement learning and evaluation vary one while holding the other fixed.

## Interface

```python
class Agent:
    system_prompt: str | None = None

    def __init__(self, configuration: Any) -> None: ...          # deterministic

    def select_context(self, history: History, hints: ContextHints) -> list[Message]:
        """What the model sees this turn. Default: system prompt + every message in history."""

    async def act(self, run: RunContext, history: History,
                  tools: list[ToolSpecification]) -> Message:
        """Produce one action. Default: one sample of the policy slot."""
        return await run.model.sample(self.select_context(history, run.context_hints), tools=tools)
```

## What an agent sees

| Sees | Never sees |
|---|---|
| History as canonical messages (observations and its own replies) | Environments or environment identities |
| The tool specifications offered this turn | Tokens, logprobs, policy versions, engines |
| The model slot's `CapabilityContract` and usage (`context_used`, `context_limit`) | Credentials |
| The task's `ContextHints` | Task state |

## Context management

- `select_context` returns the complete message list for the turn. Compaction, windowing, and summaries are just
  shorter or rewritten lists; the SDK derives the `ContextDelta` (retained prefix + appended items) automatically.
- A context that is not an extension of the previous one starts a new renderer epoch in the recorder
  (`CONTEXT_EDIT`). That is correct for training — each turn becomes its own segment — but loses prefix-cache
  reuse; it shows up in metrics.
- Summaries produced by extra model calls are samples like any other: make them on a separate slot (for example a
  frozen `summarizer` channel) so they don't mix with the policy's trajectory.

## Context hints

```python
@dataclass(frozen=True)
class ContextHints:
    history: HistoryShape = HistoryShape.FULL     # FULL | LATEST_OBSERVATION | WINDOW
    window: int | None = None                     # for WINDOW: number of recent turns
```

Hints let a task state its intent (e.g. observations are complete states, so only the latest matters). Agents
SHOULD honor them; the default agent does.

## Acting

- `act` returns exactly one assistant `Message`: the action the task responds to.
- An agent MAY sample more than once per action (plan-then-act, self-critique). Only samples on the policy slot
  that lead to the returned message lie on the trajectory path; others become sibling branches in the session
  tree. Auxiliary reasoning that should not be trained belongs on a separate slot.
- Agent attributes are part of the run's state and, under a durable runner, follow the same
  [determinism rules](determinism.md) as task code.
- **Interruption.** A message delivered with mode `INTERRUPT` cancels the in-flight model sample inside `act`
  ([conversations](conversations.md)). `act` raises `Interrupted`; the partial reply is recorded as an aborted
  branch and never trained on. Agents that sample several times per action need no special handling.
