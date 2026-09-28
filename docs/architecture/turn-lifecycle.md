# Lifecycle of a turn

Status: **Proposed** · Walks one turn of the [loop](../core/harness/README.md#the-loop-normative) under both runners.
Normative definitions live in the linked documents; this is the integration view.

## The turn

A turn is: the agent samples a reply, the task responds with an observation. With a tool call and an environment:

```
agent.act ──▶ Model.sample ──▶ recorder ──▶ engine          (tokens in; tokens, logprobs, version, routed experts out)
                                 │  records the sampled span; parses tokens into a canonical reply
          ◀────────────── canonical reply + usage
task.respond(reply) ──▶ run_tools ──▶ @tool bash ──▶ environment.execute ──▶ environment system
          ◀────────────── Observation(tool results) ──▶ steering messages merged ──▶ next turn
```

## Under the `LocalRunner` (local profile)

Every arrow is a Python call in one process. The recorder calls a local engine; `environment.execute` calls a local
environment driver, if the task uses one. Nothing is persisted except the recorder's session trees and the sample
log. A crash loses in-flight runs; a rollout job resamples them.

## Under the `DurableRunner`

The task host (sandboxed) runs the code; the pump (a DBOS workflow) performs each effect as a step.

1. The task host requests the effect `model.sample` with `effect_id = {run_id}:{generation}:{ordinal}` and an
   argument digest.
2. The pump authorizes it and runs a DBOS step that calls the recorder with those identifiers. The recorder
   deduplicates on them.
3. The step's result is recorded; the pump records the completion order and feeds the reply to the task host.
4. The task host runs `task.respond`; the tool body requests `environment.execute`; the pump runs it as another
   step with its own `effect_id`; the environment system deduplicates on it.
5. The observation is projected as run events (`observation.recorded`, rewards); the next turn begins.

**In-flight policy change** (step 2): if the weight update controller commits a new version while the sample is
generating, the engine aborts the request; the recorder records the partial span at the old version and resubmits
prompt + partial at the new version. The task and agent receive one ordinary reply.

**Messages during the turn**: a `STEER` message is merged into the observation after step 4; an `INTERRUPT` message
cancels the model sample at step 2 and the loop calls `Task.resume`
([conversations](../core/harness/conversations.md)).

## Failure branches (durable runner)

| Failure | What happens | Lost |
|---|---|---|
| Executor dies before the step's result is recorded | The recovery controller re-enqueues the run; the pump replays; the step re-runs with the same `effect_id`; the recorder or environment returns the recorded result, or joins the execution in progress | nothing |
| Executor dies with a non-deduplicating tool call in flight | The attempt marker exists → the completion is `OUTCOME_UNKNOWN`; the model sees it | the call's result |
| An executor declared dead keeps running | Its next record fails DBOS's ownership check; at most the steps it had started run twice and are absorbed by receivers | nothing |
| Task host crashes | The pump restarts it and re-feeds recorded completions | nothing |
| Replay requests a different effect, or produces a different observation or reward | `run.failed{NON_DETERMINISM}`; quarantined | the run |
| A run crash-loops its task host during recovery | `run.failed{POISONED}` | the run |
| Recorder instance dies mid-sample (service form) | The step retries; unflushed tree state is lost → the session is marked incomplete | training data for that session |
| Database failover | Recording stalls for the failover window; zero-data-loss replication means no acknowledged record is lost | latency |
