# 0019 — Conversations as keyed runs, with priority-based delivery

Status: **Accepted** · Date: 2026-09-27

## Context

Agents that talk to people or to each other receive messages over time. Research proposed a heavyweight "durable
agent identity" resource (state, memory, grants, owned resources). That over-reached: memory and other cross-run
state are tools, and an addressable agent is already expressible as a deployment.

## Decision

- An addressable agent is a **deployment**. A **conversation** is a run keyed by `(deployment, conversation key)`.
  Sending to a conversation starts its run if none is live (signal-with-start).
- **Lanes are per conversation**: at most one run consumes a conversation's messages at a time; conversations run in
  parallel.
- `WaitFor` (from `start`, `respond`, `resume`) suspends a run until a message arrives; `End(continue_as=…)` starts
  the conversation's next run with explicit state. Each episode is a bounded run.
- Messages carry a **priority** (`LOW`, `NORMAL`, `HIGH`) that the deployment maps to a **delivery mode**: `QUEUE`
  (wait for the next `WaitFor`), `STEER` (merge into the next observation), `INTERRUPT` (cancel the in-flight model
  sample). Defaults: LOW → QUEUE, NORMAL → STEER, HIGH → INTERRUPT. Senders' priorities are capped.
- No identity resource, no platform memory: state shared across conversations is a tool.

See [conversations](../core/harness/conversations.md).

## Consequences

- `Task` gains `resume` and `steer`; the loop handles suspension and interruption.
- Interrupted replies are recorded as aborted branches and never trained on.
- The durable runner schedules conversation activations on a queue partitioned by conversation key.

## Alternatives considered

- **Durable agent identities as resources** (research): more machinery than the use cases need.
- **One lane per agent**: simpler serialization, but conversations with different users would block each other.
