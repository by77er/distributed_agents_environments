# Inference

Status: **Working** (2026-10-02) · Code: `rollout_train.inference` · See [engines](../core/recorder/engine-adapter.md), [recorder](../core/recorder/README.md)

A **channel** is a policy being served, by name. Task code never sees one: a run's binding names a channel for a
model slot, and the recorder samples from it. Whoever trains publishes new weights to it; whoever deploys decides
which engines stand behind it.

```python
channel = Channel("policy", engines=[engine_a, engine_b], renderer=renderer, limits=Limits(1024, 400, 8000))
generation = await channel.generate(prompt, max_tokens=64, ..., adapter=channel.adapter, session="r_1/ada")
version = await channel.publish("step-3", "/adapters/step-3")
```

| | |
|---|---|
| Engines | A channel has one or more replicas. A session's requests go to the same one every time, where its prompts' shared beginnings are cached; sessions spread over the replicas. |
| `Limits` | Tokens of thinking per turn, room for the answer, and the longest turn. The longest turn is the smaller of what the engines accept and what the trainer can train on. Code above the channel receives the outcome (a context limit in a model's capability contract, a refusal when a context is full), never these numbers. |
| `publish` | New weights, served from then on; see the [weight transition protocol](../core/recorder/engine-adapter.md#weight-transition-protocol). |
| `pause`, `resume`, `sleep`, `wake` | For a trainer that shares the engines' accelerator. |
| `take()` | What passed through since the last call: requests, tokens in and out, tokens per second in all and per stream, mean concurrency. |

Channels are described in a deployment's [profile](../guide/perspectives.md#deploying): the model, its renderer, its
limits, and one entry per engine.
