# Testing

Tasks and agents are tested against a scripted model: no GPU, no network, and deterministic. The helpers live in
`rollout.testing`.

## The scripted endpoint

`ScriptedModelEndpoint(replies)` implements the `ModelEndpoint` protocol. Each sample request takes the next
entry of the script, in call order across all model slots bound to it:

| Entry | Reply |
|---|---|
| `"text"` | an assistant message with that text |
| a `Message` | that message, e.g. one with tool calls from `tool_call_reply(...)` |
| a function of the `SampleRequest` | its return value; it may be `async`, e.g. to hold a reply open |

It records what it was asked:

| Attribute | Holds |
|---|---|
| `requests` | every `SampleRequest`: `effect_id`, `session_id`, `context.append` (the messages), `tools` |
| `cancelled` | the `effect_id`s it was asked to cancel |

The endpoint raises `AssertionError` when the script runs out, so an episode that samples more than expected fails
loudly. Pass `contract=CapabilityContract(...)` to test behavior near limits.

## Helpers

| Helper | Does |
|---|---|
| `local_run(task, replies=())` | a `LocalRunContext` with one scripted endpoint bound to every declared slot; returns `(run, endpoint)` |
| `tool_call_reply(*calls, text="")` | an assistant message that makes tool calls |
| `events_of(run, RunEventType.X)` | the run's events of one type |
| `payload(event)` | an event's payload as a JSON object |
| `until(condition, seconds=15)` | waits until `condition()` (a function, or one returning an awaitable) is true, asking every hundredth of a second; raises `AssertionError` after `seconds` |

`local_run` gives the run no imported tools, sandboxes, environments or blob store. A task that uses them is tested
through a `LocalRunner` built with `tool_sets=`, `pools=`, `environments=` or `blobs=` and a scripted endpoint as its
provider, as in [tools](tools.md#imported-tools). `FakeSandboxes` is a sandbox provider whose sandboxes are records
that honour their specs: `pools={"fake": SandboxPool(FakeSandboxes())}` ([sandboxes](../libraries/rollout/sandboxes.md)).

### Counting calls across processes

Tests that kill and restart a durable runner count what ran in a file that outlives the process:

| Helper | Does |
|---|---|
| `LedgerEndpoint(inner, ledger)` | wraps a model endpoint; appends the `effect_id` and `session_id` of every sample to the file `ledger` |
| `LedgerEnvironments(inner, ledger)` | wraps an environment service; appends the `effect_id`, `environment_id` and `command` of every command it starts |
| `read_ledger(ledger)` | the entries, oldest first, each with its time `at`; an empty list when the file does not exist |

## A test

With `pytest-asyncio` in auto mode (configured in `pyproject.toml`), test functions can be `async`:

```python
import asyncio

from rollout.contracts import Message, RunEventType, SampleRequest, ToolCall
from rollout.harness import Agent, End, Observation, RunContext, Task, rollout, tool
from rollout.testing import events_of, local_run, payload, tool_call_reply


class Thermostat(Task):
    max_turns = 4

    def __init__(self, target: float) -> None:
        self.target = target
        self.temperature = 16.0

    async def start(self, run: RunContext) -> Observation:
        return Observation(f"It is {self.temperature} °C. Bring the room to {self.target} °C, then say done.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        return End(reward=1.0 if self.temperature == self.target else 0.0)

    @tool
    async def set_heating(self, degrees: float) -> str:
        """Set the heating to a temperature in °C."""
        self.temperature = degrees
        return f"Heating set to {degrees} °C."


async def test_the_agent_is_rewarded_for_reaching_the_target() -> None:
    task = Thermostat(target=21.0)
    heat = ToolCall(call_id="c1", name="set_heating", arguments={"degrees": 21})
    run, endpoint = local_run(task, [tool_call_reply(heat), "done"])

    await rollout(task, Agent(), run)

    assert task.temperature == 21.0
    assert run.history.turns[-1].observation.reward == 1.0
    assert [request.tools[0].name for request in endpoint.requests] == ["set_heating", "set_heating"]
    completed = events_of(run, RunEventType.EFFECT_COMPLETED)
    assert [payload(event)["status"] for event in completed] == ["ok", "ok"]


asyncio.run(test_the_agent_is_rewarded_for_reaching_the_target())  # pytest runs it for you; this runs the page
```

## Replies that depend on the request

A function entry can look at the request, for example to answer differently depending on the context:

```python
def echo_last_message(request: SampleRequest) -> Message:
    return Message.assistant(f"You said: {request.context.append[-1].text}")


async def test_a_reply_can_depend_on_the_request() -> None:
    task = Thermostat(target=16.0)
    run, _ = local_run(task, [echo_last_message])
    await rollout(task, Agent(), run)
    assert run.history.turns[1].reply.text.startswith("You said: It is 16.0 °C.")


asyncio.run(test_a_reply_can_depend_on_the_request())
```

## Holding a reply open

To test what happens while the model is replying (steering, interruption), use an `async` entry that waits for an
event. The [conversations](conversations.md#interrupting) page has a complete example. The pattern:

```py
started, release = asyncio.Event(), asyncio.Event()

async def held(request: SampleRequest) -> Message:
    started.set()
    await release.wait()             # stays in flight until released or cancelled
    return Message.assistant("finished")

episode = asyncio.create_task(rollout(task, Agent(), run))
await started.wait()                 # the reply is now in flight
run.deliver(envelope, DeliveryMode.STEER)
release.set()
await episode
```

## Checking an environment

`rollout env check module:name` plays one episode of an environment on a `LocalRunner` with a scripted endpoint
(`--reply` is what it says each turn), after checking its rows, its starts and its eval data, and holds the episode's
reward and result to the environment's description. With `--profile P --groups N` it plays N groups with the profile's
model and flags the groups whose episodes all scored the same, which teach nothing
([checking an environment](../libraries/rollout-train/rollouts.md#checking-an-environment)).

## What to assert on

| To check | Look at |
|---|---|
| what the model saw | `endpoint.requests[i].context.append` |
| which tools were offered | `endpoint.requests[i].tools` |
| replies and observations | `run.history.turns` |
| rewards per reply | `turn.observation.reward`, or `observation.recorded` events |
| episode rewards | `run.rewards` |
| effects, suspensions, interruptions | `events_of(run, RunEventType.…)` |

## Tests on Ray

A test that needs Ray asks for the `local_ray` fixture (`tests/conftest.py`): a Ray of the test session's own, started
once and shared by every test that asks for it, and shut down when the session ends.

| It is | So that |
|---|---|
| a new local instance (`address="local"`, `RAY_ADDRESS` unset) | a test never joins a cluster the machine runs already |
| on disk, in `~/.cache/rollout/ray-tests/session_DATE_PID` | `/tmp` may be memory; the session's directory is removed at its end |
| 4 CPUs, no GPU, a 100 MiB object store | tasks and actors ask for fractions of these, as on a cluster |
| the dashboard on, and its job agent, on free ports | jobs can be submitted (`local_ray.dashboard`); a cluster already running holds Ray's default ports |
| `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, `RAY_AUTH_MODE=disabled` | workers run in the test's environment, and nothing asks for a token |

Ray reads its environment once, when it is imported, so `tests/conftest.py` sets it before any test module loads.
`tests.local_ray.submitted` submits a job, asking again while the node's job agent is still starting.

## Live tests

Tests that need what a test cannot start alone are marked `live`, and the workspace's pytest settings leave them out
(`-m "not live"` in `pyproject.toml`). They are:

| Where | Needs |
|---|---|
| `environments/minecraft/tests/test_paper.py`, `test_harness.py`, `test_episode.py` (the tests that start a world) | Java (and Node for the harness), and the network once, to fetch Paper |
| `tests/rollout_lora/test_on_gpu.py`, `test_small_on_gpu.py`, `tests/rollout_train/gateway/test_on_gpu.py` | the GPU, with nothing else on it |

Ask for them with `-m live`, by path:

```bash
uv run pytest environments/minecraft/tests -m live
uv run pytest tests/rollout_lora -m live --basetemp ~/.cache/rollout/pytest-gpu   # (its steps write gigabytes)
```
