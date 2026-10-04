# Tasks

A task is the environment an agent acts in, in the reinforcement-learning sense. It produces the first
observation, answers every reply, defines the tools, and scores the episode. One task instance serves one run, so
keep episode state on `self`.

## Anatomy

```py
class MyTask(Task):
    # Declarations: class attributes
    models = {"policy": ModelSlot()}      # model slots; the agent acts through "policy"
    imports = ["notes"]                    # imported tool sets, bound per run (tools.md)
    sandboxes = {"box": SandboxSpec(kind="code")}   # sandboxes acquired for the run (sandboxes.md)
    max_turns = 10                         # truncate after this many replies (None: no limit)
    context_hints = ContextHints()         # advice to the agent about how much history to show

    def __init__(self, parameters=None): ...   # once per run, with the row's parameters

    async def setup(self, run): ...        # optional: acquire resources
    async def start(self, run): ...        # required: first Observation, or WaitFor a message
    async def respond(self, run, reply): ...   # each reply → next Observation (default: run tools)
    async def resume(self, run, envelope): ... # a message → next Observation (conversations.md)
    async def steer(self, run, envelopes, observation): ...  # merge mid-turn messages (conversations.md)
    async def score(self, run): ...        # optional: episode-level reward
    async def teardown(self, run): ...     # optional: always runs once setup began

    def tools_for_turn(self, run): ...     # the tools offered this turn (tools.md)
    async def run_tools(self, run, reply): ...   # execute the reply's tool calls (tools.md)
```

| Hook | Called | Returns | Default |
|---|---|---|---|
| `setup` | once, first | nothing | does nothing |
| `start` | once, after `setup` | `Observation` or `WaitFor` | **must be implemented** |
| `respond` | after every reply | `Observation` or `WaitFor` | executes the reply's tool calls; ends the episode when there are none |
| `resume` | when a message satisfies a `WaitFor`, or interrupts a reply | `Observation` or `WaitFor` | the message becomes a USER observation |
| `steer` | when messages arrived during the turn with mode `STEER` | `Observation` | appends them as USER messages |
| `score` | once, after the episode ends normally | `float` or `None` | `None` |
| `teardown` | always, if `setup` began | nothing | does nothing |

The loop, in order: `setup`, `start`, then repeatedly the agent acts and `respond` answers, until an observation
has `end` set or `max_turns` replies were made. Then `score`, then `teardown`. If a hook raises, `rollout()`
re-raises after `teardown`, and `score` does not run. The loop itself is in
[harness](../libraries/rollout/README.md#the-loop).

## Observations

An `Observation` is what the model sees next. Its first argument takes text, a `Message`, or a list of messages:

```python
from rollout.contracts import Message, Role, Text
from rollout.harness import End, Ending, Observation, WaitFor

Observation("Guess the word.")                                   # one USER text message
Observation(Message.user("Guess the word."))                     # the same, explicitly
Observation([Message.user("Board:"), Message(role=Role.USER, content=[Text(text="_ _ _ _ _")])])

feedback = Observation("Too high.", reward=-0.1, info={"guess": 90})
assert feedback.reward == -0.1 and feedback.end is None

finished = End(reward=1.0)                                       # terminal, no messages
assert finished.end is Ending.TERMINATED
assert End(truncated=True).end is Ending.TRUNCATED
```

| Field | Meaning |
|---|---|
| `messages` | USER and TOOL messages shown to the model next. |
| `reward` | The reward for the reply this observation answers. |
| `end` | `None` while the episode continues. `Ending.TERMINATED` for a real end state; `Ending.TRUNCATED` when a limit stopped it (value-based methods may bootstrap from a truncated state). |
| `info` | Logged in the run's events; never shown to the model. |

`WaitFor(kind="message", timeout=None)` suspends the run until a message arrives; see
[conversations](conversations.md).

### Validation

Every observation a hook returns is checked before it is recorded. A violation raises `InvalidObservation`, and a
runner fails the run with class `invalid_observation`:

1. Observations contain only USER and TOOL messages. The system prompt belongs to the agent, and assistant turns
   belong to the model.
2. If the reply made tool calls, the observation contains a tool result for every call, even when it ends the
   episode. Tool results must answer calls of that reply. A reply with tool calls cannot be answered by `WaitFor`.
3. An observation that does not end the episode has at least one message.

## Rewards

| Way | Use it for | Where it ends up |
|---|---|---|
| `Observation(..., reward=r)` / `End(reward=r)` | a reward for one specific reply | the turn's observation and the `observation.recorded` event |
| `score()` returning a number | one reward for the whole episode | `run.reward(...)`, a `reward.assigned` event |
| `run.reward(value, slot=..., key=...)` | rewards for other model slots, or several named rewards | a `reward.assigned` event |
| `run.exclude_from_training(reason)` | a run the policy should not learn from (e.g. an infrastructure fault) | a `training.excluded` event |

## A multi-step environment

`respond` is the environment's step function. This Wordle variant gives feedback on each guess, penalizes invalid
guesses, and truncates after three replies.

```python
import asyncio

from rollout.contracts import Message
from rollout.harness import Agent, End, Ending, Observation, RunContext, Task, rollout
from rollout.testing import local_run


class Wordle(Task):
    max_turns = 3

    def __init__(self, parameters: dict[str, str]) -> None:
        self.secret = parameters["word"]

    async def start(self, run: RunContext) -> Observation:
        return Observation("Guess the 5-letter word. Reply with the word only.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        guess = reply.text.strip().lower()
        if len(guess) != 5 or not guess.isalpha():
            return Observation("Invalid. Reply with one 5-letter word.", reward=-0.1)
        if guess == self.secret:
            return End(reward=1.0)
        marks = "".join("G" if a == b else ("Y" if a in self.secret else ".") for a, b in zip(guess, self.secret))
        return Observation(f"{guess.upper()}: {marks}")

    async def score(self, run: RunContext) -> float | None:
        return None  # rewards are per reply here


async def main() -> None:
    task = Wordle({"word": "crane"})
    run, _ = local_run(task, replies=["hello there", "trace", "brine"])
    await rollout(task, Agent(), run)

    for turn in run.history.turns[1:]:
        print(turn.reply.text if turn.reply else "-", "->", [m.text for m in turn.observation.messages],
              turn.observation.reward, turn.observation.end)
    # hello there -> ['Invalid. Reply with one 5-letter word.'] -0.1 None
    # trace -> ['TRACE: .GGYG'] None None
    # brine -> ['BRINE: .G.GG'] None None
    # - -> [] None truncated        ← max_turns reached

    assert run.history.turns[2].observation.messages[0].text == "TRACE: .GGYG"
    assert run.turn == 3
    assert run.history.turns[-1].observation.end is Ending.TRUNCATED


asyncio.run(main())
```

## Other model slots

A task can sample models itself, for example a simulated user or an opponent. Declare the slot in `models` and
sample it through `run.models[...]`. Only the `policy` slot's replies form the agent's turns.

```python
from rollout.harness import ModelSlot


class SimulatedSupport(Task):
    models = {"policy": ModelSlot(), "user": ModelSlot(trainable=False)}

    async def start(self, run: RunContext) -> Observation:
        return Observation("Hi, my printer will not print.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        user_turn = await run.models["user"].sample([Message.user(f"The support agent said: {reply.text}")])
        if "[resolved]" in user_turn.text:
            return End(reward=1.0)
        return Observation(user_turn.text)


async def support_episode() -> None:
    task = SimulatedSupport()
    run, endpoint = local_run(
        task, replies=["Is it plugged in?", "Yes, it is.", "Try turning it off and on.", "That worked [resolved]"]
    )
    await rollout(task, Agent(), run)
    slots = [request.session_id.split("/")[1] for request in endpoint.requests]
    assert slots == ["policy", "user", "policy", "user"]
    assert run.turn == 2  # two policy replies


asyncio.run(support_episode())
```

`local_run` binds every slot to one scripted endpoint, so the script interleaves both slots' replies in call order.

## The run context

Every hook receives the run context as `run`. It is everything task and agent code can reach.

| Member | Meaning |
|---|---|
| `run_id` | the run's identifier, `r_{ulid}` |
| `conversation` | the `ConversationKey` when the run serves a conversation, otherwise `None` ([conversations](conversations.md)) |
| `turn` | the number of replies recorded so far |
| `history` | the episode, read-only: `turns`, and `messages(hints)` ([agents](agents.md#what-the-model-sees)) |
| `models`, `model` | the declared model slots by name; `model` is `models["policy"]` ([agents](agents.md#the-model-interface)) |
| `tools` | the imported tools: `specifications()`, `await call(name, arguments)`, `name in run.tools` ([tools](tools.md#imported-tools)) |
| `sandbox(name)` | a sandbox the task or program declared, acquired for the run: its addresses, its environment, and its operations, each an effect ([sandboxes](../libraries/rollout/sandboxes.md)) |
| `environments` | creates computers the run owns; `None` when the runner has no environment backend ([environments](../implementations/rollout-computers.md)) |
| `blobs` | stores bytes for `Media` blocks; `None` when the runner has no blob store ([content](content.md#media-and-blobs)) |
| `context_hints` | the task's `context_hints`, for the agent |
| `now()` | the current time, as a `datetime` |
| `random` | a `random.Random` seeded from `run_id` |
| `reward(value, *, slot="policy", key="default")` | assigns a reward ([rewards](#rewards)); an unknown slot raises `ValueError` |
| `exclude_from_training(reason)` | marks the run as unsuitable for training ([rewards](#rewards)) |
| `await emit(kind, payload, *, to=None)` | durable output, such as a reply to a person ([conversations](conversations.md#sending-and-replying)) |
| `await gather(*awaitables)` | awaits concurrently and returns the results in order, as `asyncio.gather` does |

## Training on a task

A run trains on, and an eval measures, an **environment** (`rollout.environment.Environment`): a program that plays
the task, its situations as rows and how one start of a row is drawn (the task's parameters), eval data that training
never draws, what its results say, a version, and perhaps a curriculum of its own
([rollouts](../libraries/rollout-train/rollouts.md#environment), [three ways in](perspectives.md#building-an-environment)).
`rollout env check module:name` checks one before anything trains on it.

## Environments

The computers below are another sense of the word. A task whose agent needs a computer creates one in `setup` with `await run.environments.create(specification)` and
keeps the handle on `self`. Every operation on the handle is an effect. The runner destroys any environment the run
still owns when the run ends. Handles, backends and the ready-made `ComputerTools` are described in
[environments](../implementations/rollout-computers.md).

## State and determinism

- Keep episode state on `self`. `__init__` receives the row's parameters once per run.
- Use `run.now()` and `run.random` rather than `time.time()` or the `random` module. The durable runner resumes a
  run by running its code again, and these give the same values each time. The rules are in
  [determinism](../libraries/rollout/determinism.md).
- `teardown` must be idempotent: it runs on every path once `setup` began, including failures and cancellation.
