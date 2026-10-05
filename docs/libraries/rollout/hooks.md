# Hooks

For anyone who watches runs as they happen, for logging, metrics or a live view.

**Read first:** [What a run records](../../guide/runs-and-events.md). **Next:** [Memory for long episodes](memory.md).

Code: `rollout.harness.hooks` · See [`RunHooks`](../../guide/reference.md#runhooks),
[`ModelSample`](../../guide/reference.md#modelsample)

Hooks watch runs as they happen. A runner calls its hooks for every run event it records and for every model sample
it performs, so that logging, metrics and live views need nothing from the program being run.

```python
from rollout.harness import ModelSample, RunHooks
from rollout.local import LocalRunner

class Watch(RunHooks):
    def on_event(self, event):          # every event of every run, as it is recorded
        ...
    def on_sample(self, sample: ModelSample):  # every model reply, with what the model was sent
        ...

runner = LocalRunner(hooks=[Watch()])
```

| Hook | Called | Carries |
|---|---|---|
| `on_event(event)` | when a run event is recorded | the `RunEvent`: lifecycle, effects (tool calls with arguments and results), rewards, outputs. A model sample appears here by its context's digest only. |
| `on_sample(sample)` | when a model endpoint replies | the run and slot, the `SampleRequest` (every message sent, and the tools offered), the `SampleResult`, and how long the endpoint took |

Rules:

- Hooks run on the runner's event loop, in the middle of the run. They must be quick and must not block; hand slow work
  to a queue or a file.
- A hook that raises is logged and ignored. A hook never fails a run.
- Hooks see what their own runner's process records and performs.
- A harness that samples at a slot's address ([harnesses over HTTP](../rollout-train/harness-endpoint.md)) samples
  through the gateway, not the runner's endpoints ([which gateway tells the hooks](../rollout-train/gateway.md#a-runner-served-by-the-gateway)).
- Hooks are observers. They cannot change a request or a reply; code that must is a `ModelEndpoint`.

The [monitor](../rollout-train/monitor.md) is built on hooks. [Episode](../rollout-train/episodes.md) runners and the
training loop have hooks of their own ([`Hooks`](../rollout-train/rollouts.md#watching)).
