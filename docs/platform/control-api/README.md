# Control API

Status: **Proposed** · Layer: platform (optional)

## Purpose

The external surface of the service form: manage deployments, start and observe runs, send messages to
conversations, run rollout jobs, and issue sessions for unmanaged harnesses. It exposes the core
[`Runner`](../../core/harness/README.md#runner) and [`RolloutJobs`](../../core/rollouts/README.md) protocols over the
network, plus authentication, tenancy and quotas. A global edge routes requests to the owning cell.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Authentication, tenancy, quotas, [trust tier](../trust-tiers.md) assignment | Run execution (durable runner) |
| Request validation and idempotency (`request_id`) | Rollout admission (rollout service) |
| Cell selection for new runs and conversations | Recording (recorder) |
| Deployments registry | Environments ([environment system](../../environments/README.md)) |
| Durable event streams and the ephemeral token-delta channel | |

## Interface

```proto
service ControlAPI {
  // Deployments: addressable agents
  rpc PutDeployment(Deployment) returns (Deployment);              // {name, specification: RunSpecification}
  rpc GetDeployment(DeploymentReference) returns (Deployment);

  // Runs
  rpc StartRun(StartRunRequest) returns (Run);                     // {request_id, specification | deployment, labels}
  rpc GetRun(RunReference) returns (Run);                          // status, labels, parent, result
  rpc ListRuns(ListRunsRequest) returns (stream Run);              // filter by labels, status, time
  rpc StreamEvents(StreamRequest) returns (stream StreamEvent);    // {run_id, from_seq, include_token_deltas}
  rpc Cancel(CancelRequest) returns (Empty);                       // {run_id, reason, request_id}

  // Conversations and messages
  rpc Send(SendRequest) returns (Empty);
  //   {to: Address, envelope: Envelope, priority: LOW | NORMAL | HIGH, idempotency_key}
  //   starts the conversation's run if none is live (signal-with-start)

  // Rollout jobs (the RolloutJobs protocol)
  rpc StartJob(StartJobRequest) returns (Job);
  rpc RunRows(RunRowsRequest) returns (Tickets);
  rpc CancelRuns(CancelRunsRequest) returns (Empty);
  rpc Samples(SamplesRequest) returns (stream Sample);
  rpc Acknowledge(AcknowledgeRequest) returns (Empty);
  rpc Publish(PublishRequest) returns (Operation);

  // Unmanaged harness sessions
  rpc OpenSession(OpenUnmanagedRequest) returns (UnmanagedSession);
  //   {channel, sampling, imports, labels} → {session_id, model_base_url, model_credential, mcp_url, mcp_credential}
}
```

## Semantics and guarantees

- **Idempotent creates and sends**: `request_id` / `idempotency_key` are scoped per tenant; retries return the
  original result.
- **Sending** to a conversation is delivered with the deployment's [delivery policy](../../core/harness/conversations.md#priority-and-delivery-mode);
  a sender's priority is capped by its access rights.
- **Validation** at the edge: code references exist; every model slot the program declares is bound to an endpoint
  whose contract satisfies it; every import is bound.
- **Event streams** are the run's [run events](../../contracts/run-events.md), resumable by `seq`. Token deltas (by
  `effect_id`) are best-effort and never replayed; the final `effect.completed` is authoritative.

## Failure modes

The edge is stateless. A cell outage affects only that cell's runs; starts can be retried against another cell
(new `run_id`).

## Open questions

- Tenancy model and quota dimensions (concurrent runs, tokens, environments).
- Listing runs across cells (fan-out) or requiring a cell or label filter.
