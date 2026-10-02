# Getting started

Write a task, run one episode against a scripted model, and read what happened.

## Install

`rollout` needs Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync                 # the core library and development tools; no GPU needed
uv run pytest           # tests, including every example in this guide
```

## A first task

A task defines the environment the model acts in. This one asks a question and ends the episode after one reply,
with a reward of 1 for the right answer.

```python
import asyncio

from rollout.core.contracts import Message
from rollout.core.harness import Agent, End, Observation, RunContext, Task, rollout
from rollout.core.testing import local_run


class Arithmetic(Task):
    def __init__(self, parameters: dict[str, str]) -> None:
        self.question = parameters["question"]
        self.answer = parameters["answer"]

    async def start(self, run: RunContext) -> Observation:
        return Observation(self.question)  # one USER message

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        correct = reply.text.strip() == self.answer
        return End(reward=1.0 if correct else 0.0)  # the reward belongs to this reply
```

Two hooks are enough:

- `start` returns the first observation.
- `respond` receives each reply the model makes and returns the next observation. `End(...)` is an observation
  that ends the episode.

## Run one episode

`rollout()` runs the loop: `start`, then the agent acts, then `respond`, until an observation ends the episode.
It needs a run context. `local_run` builds one whose model replies from a script, so the episode needs no model.

```python
async def main() -> None:
    task = Arithmetic({"question": "What is 17 * 23? Reply with the number only.", "answer": "391"})
    run, endpoint = local_run(task, replies=["391"])

    await rollout(task, Agent(), run)

    start, answer = run.history.turns
    print(start.observation.messages[0].text)  # What is 17 * 23? Reply with the number only.
    print(answer.reply.text)                    # 391
    print(answer.observation.reward)            # 1.0

    # The model saw exactly one USER message.
    (request,) = endpoint.requests
    assert [message.text for message in request.context.append] == [start.observation.messages[0].text]
    assert answer.observation.reward == 1.0


asyncio.run(main())
```

What happened:

1. `start` returned the question. The run recorded it as the first turn of the history.
2. The default `Agent` sampled the `policy` model slot with the history so far. The scripted endpoint replied
   `"391"`.
3. `respond` checked the reply and returned `End(reward=1.0)`. The loop recorded the reply with its observation
   and stopped.

Along the way the run emitted events you can inspect: see [runs and events](runs-and-events.md).

## Next

- Multi-step environments, more reward types and other endings: [tasks](tasks.md).
- Letting the model call functions: [tools](tools.md).
- Changing what the model sees: [agents](agents.md).
- Running against a real model: [models](models.md).
