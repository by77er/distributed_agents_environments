# Model endpoint

For whoever implements a model endpoint: what an endpoint is asked, what it answers, its errors and its guarantees.

**Read first:** [Canonical content](canonical-content.md). **Next:** [The gateway](../../rollout-train/gateway.md).

Code: `rollout.contracts.model_endpoint`, `rollout.harness.model` · See
[`ModelEndpoint`](../../../guide/reference.md#modelendpoint),
[guide: agents](../../../guide/agents.md#the-model-interface)

A model endpoint serves model slots. It is the only thing task, agent and program code knows about models, and that
code reaches it only through [`Model`](../../../guide/reference.md#model) (`run.model`, `run.models[slot]`).

| Implementation | Serves | Code |
|---|---|---|
| The gateway's endpoint | a trainable [channel](../../rollout-train/channels.md), with everything training needs recorded | [the gateway](../../rollout-train/gateway.md#a-runner-served-by-the-gateway) |
| A direct adapter | a provider's API; nothing is recorded | [guide: models](../../../guide/models.md) |
| `ScriptedModelEndpoint` | a script, in tests | [guide: testing](../../../guide/testing.md) |

Which one serves a slot is decided by the run's binding ([run specifications](../README.md#run-specifications)).
Code cannot tell.

## What an endpoint is asked

- **`describe(session_id)`** returns the slot's
  [`CapabilityContract`](../../../guide/reference.md#capabilitycontract): the context limit, which is a minimum the
  endpoint guarantees, and the most tokens a reply may have. A run reads it once, when it starts.
- **`sample(request)`** returns one reply. The [`SampleRequest`](../../../guide/reference.md#samplerequest) carries
  the effect's identity, the session, the context, the tools offered this turn, and optionally a cap on the output
  and a tool choice.
- **`cancel(effect_id)`** is best-effort. The run calls it when the task awaiting a sample is cancelled.

What a request does not carry: sampling parameters, a model's name, a version. They belong to the binding, so that
code cannot depend on them and a trainer can reproduce the distribution a channel sampled from.

- **The context is whole.** Every request holds every message of the context, in
  [`ContextDelta`](../../../guide/reference.md#contextdelta)'s `append`, and the
  [context digest](canonical-content.md#digests) that names them. An endpoint keeps no context between requests.
- **Tools** are the subset offered this turn. An endpoint uses only their model-visible fields. Their names are
  unique within a request.
- **`max_output_tokens`** never exceeds the contract's: `Model.sample` raises `ContractViolation` before it
  requests anything.
- **`tool_choice`** says whether the model may, must not or must call a tool, or names the tool it must call. The
  Responses adapter sends it to the provider. The gateway ignores it.
- **`links`** say how the request follows from earlier ones of its session (`SampleLink`: a type and the earlier
  request's `effect_id`). The gateway keeps them with the turn; a direct adapter ignores them.

## What an endpoint answers

A [`SampleResult`](../../../guide/reference.md#sampleresult) is an ASSISTANT message in canonical content, a finish
reason and usage.

- **Nothing in it identifies the policy, the weights version or the engine.** Tokens and logprobs stay in the
  gateway's turn store.
- **The finish reason** is `tool_use` when the reply calls tools, `length` when the reply was cut off, and `stop`
  otherwise.
- **Usage** reports how much of the context is used and its limit. `input_tokens` and `output_tokens` are given by
  endpoints that count tokens; [`Memory`](../memory.md) decides from `input_tokens` when to compact.

## Errors

An endpoint raises a [`ModelEndpointError`](../../../guide/reference.md#modelendpointerror):

| Error | Meaning | What `Model.sample` does |
|---|---|---|
| `Overloaded(retry_after)` | the endpoint is refusing work for the moment | retries with exponential backoff, waiting at least `retry_after`, then raises it |
| `InternalError` | the endpoint failed | retries the same way, then raises it |
| `ContextOverflow(context_limit)` | the context does not fit | raises it. An agent compacts and samples again; `Memory.sample` does. |
| `ContractViolation` | the request exceeds the capability contract | raises it |

The retries happen inside one effect, under one `effect_id`. Any other exception propagates unchanged. An exception
that leaves the program fails the run ([failures](../README.md#failures)).

## Guarantees

- **Idempotency.** The gateway returns the recorded result for an `effect_id` it has recorded: the turn is in the
  [ledger](../../rollout-train/checkpoints.md#the-ledger), so a runner started again gets it back too. A direct adapter
  samples again.
- **A stable contract.** A slot's capability contract does not weaken during a run.
- **One session per slot per run.** The `session_id` is `{run_id}/{model_slot}` ([identifiers](identifiers.md)).

## Endpoints with an address

An [`AddressableEndpoint`](../../../guide/reference.md#addressableendpoint) also serves its slots over HTTP, to a
harness that brings its own loop. `Model.address()` returns a
[`ModelAddress`](../../../guide/reference.md#modeladdress): a base URL, a key that names the session, and a model
name to send.

- `address_of(endpoint, session_id)` returns the address, and raises `RuntimeError` for an endpoint that has none.
- The gateway's endpoint is the addressable one ([harnesses over HTTP](../../rollout-train/harness-endpoint.md)). A
  harness's samples go to the gateway itself ([hooks](../../rollout-train/gateway.md#a-runner-served-by-the-gateway)).
