# Engines

Status: **Working** (2026-10-02) · Code: `rollout_train.inference` · See [inference](../../inference/README.md)

An engine is one replica serving a model: tokens in; tokens, logprobs and a finish reason out.

```python
class Engine(Protocol):
    max_model_len: int
    async def generate(self, prompt, *, max_tokens, temperature, top_p, stop_token_ids, adapter) -> Generation: ...
    async def load_adapter(self, name: str, path: str) -> None: ...
    async def remove_adapter(self, name: str) -> None: ...
    async def sleep(self) -> None: ...        # free the accelerator, for a trainer that shares it
    async def wake(self) -> None: ...
    def close(self) -> None: ...
```

- **Logprobs are of the distribution sampled from** (after temperature), and every sampled token has one: an engine
  that returns a token without its logprob is refused.
- **A stop token is part of what was sampled.**
- **Adapters are named**, and a request names the one it samples from.

`VllmEngine` drives vLLM, whose engine core runs in a process of its own. Asleep, it drops its weights (they are read
again from the checkpoint on waking, about 3 s from the file cache) rather than parking them in system memory. A
process that is killed cannot shut its engine down: `note_engines` writes the engine processes down and
`end_orphaned_engines` ends them from the next process, if their owner is gone.

## Weight transition protocol

`Channel.publish(adapter, path)` loads the adapter on every engine of the channel and samples from it from then on.
The adapter before stays loaded, so that a turn in progress (a thought, then its answer) finishes under the weights
it began with; the one before that is dropped. Every sampled span records the version it was sampled at.

A trainer that shares the engines' accelerator is wrapped in `Colocated`: requests are held back, those in flight
finish, the engines sleep, the trainer steps, the engines wake, and requests go on.
