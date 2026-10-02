# Hooks

Status: **Working** (2026-10-01) · Code: `rollout.harness.hooks`

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

runner = LocalRunner(hooks=[Watch()])   # DurableRunner takes the same argument
```

| Hook | Called | Carries |
|---|---|---|
| `on_event(event)` | when a run event is recorded | the `RunEvent`: lifecycle, effects (tool calls with arguments and results), rewards, outputs. A model sample appears here by its context's digest only. |
| `on_sample(sample)` | when a model endpoint replies | a `ModelSample`: `run_id`, `slot`, the `SampleRequest` (`request.context.append` is every message sent; `request.tools` the tools offered), the `SampleResult`, and how long the endpoint took |

Rules:

- Hooks run on the runner's event loop, in the middle of the run. They must be quick and must not block; hand slow work
  to a queue or a file.
- A hook that raises is logged and ignored. A hook never fails a run.
- Hooks see what their own runner's process records and performs. Under a durable runner, a run that moves to another
  process after a crash is seen from there on by that process's hooks; a sample retried after a crash is seen again.
- Hooks are observers. They cannot change a request or a reply; code that must is a `ModelEndpoint`.

The [monitor](../monitor.md) is built on hooks.
