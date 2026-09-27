# Control API

Status: **Proposed** · Boundaries B1, B2, B12, B18

## Purpose

The external surface of the system: create and observe runs, send them signals, manage environments and
templates, launch rollout jobs, and issue sessions for unmanaged harnesses. A global edge routes requests to the
owning cell; per-cell instances do the work.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Authentication, tenancy, quotas | Run execution (Runtime) |
| Request validation and idempotency (`request_id`) | Storage (Run Store, Environment Manager) |
| Cell selection for new runs (via Global Router) | Job orchestration (Rollout Controller) |
| Event streaming to clients | |

## Boundaries

| Direction | Counterpart | What |
|---|---|---|
| provides → | Clients (B1), Rollout Controller (B12) | this API |
| consumes | Run Store (B2) | create runs, deliver inbox items, read/watch events |
| consumes | Global Router | cell selection; `run_id` → cell is parsed from the id |
| consumes | Environment Manager | env/template/pool operations |
| consumes | Recorder | open sessions for unmanaged harnesses (B18) |
| consumes | Tool Router | MCP facade URLs for unmanaged sessions |

## Interface

```proto
service ControlAPI {
  // Runs
  rpc CreateRun(CreateRunRequest)     returns (Run);             // {request_id, specification: RunSpecification, labels, cell_hint?}
  rpc CreateRuns(CreateRunsRequest)   returns (CreateRunsResult);// bulk, per-item request_id (rollouts)
  rpc GetRun(RunReference)                  returns (Run);             // status, head_seq, labels, parent, result
  rpc ListRuns(ListRunsRequest)       returns (stream Run);      // filter by labels, status, time
  rpc StreamEvents(StreamRequest)     returns (stream RunEvent); // {run_id, from_seq, include_runtime_events}
  rpc Signal(SignalRequest)           returns (Empty);           // {run_id, kind: USER_MESSAGE | CUSTOM, payload, request_id}
  rpc Cancel(CancelRequest)           returns (Empty);           // {run_id, reason, request_id}

  // Unmanaged harness sessions (B18)
  rpc OpenSession(OpenUnmanagedRequest) returns (UnmanagedSession);
  //   {channel, sampling, imports (bindings), environments: [EnvironmentSpecification], labels}
  //   → {session_id, model_base_url, model_credential, mcp_url, mcp_credential}

  // Environments (manual / operator use; runs create environments from task code)
  rpc CreateEnvironment(CreateEnvironmentRequest) returns (Environment);
  rpc DestroyEnvironment(DestroyRequest)  returns (Empty);
  rpc PutTemplate(Template) returns (Template);
  rpc PutPool(Pool) returns (Pool);

  // Jobs (proxied to the Rollout Controller)
  rpc CreateJob(RolloutJob) returns (Job);
  rpc GetJob(JobReference) returns (Job);
}
```

## Semantics & guarantees

- **Idempotent creates**: `request_id` is scoped per tenant; retries return the original resource.
- **Signals** become inbox items keyed by `request_id`, then `signal.received` events. Delivery to a suspended
  run wakes it via a nudge to the partition owner.
- **Validation** at the edge: `RunSpecification` consistency — the task and agent code references exist; every
  model slot the task declares is bound to an endpoint whose contract satisfies the slot; every import is bound;
  every security class the task uses maps to a profile (or has an operator default).
- **Event streams** read from Run Store replicas; `include_runtime_events` requires operator privileges (runtime
  events contain environment identities).
- **Unmanaged sessions** create a recorder session, the requested environments, and an MCP facade over imports
  plus built-in environment tools; the environments are owned by the session and destroyed on close / time to live.

## Failure modes

Edge is stateless. A cell's Control API outage affects only that cell's runs; creates can be retried against
another cell (new `run_id`).

## Build vs buy

Build. gRPC with an HTTP/JSON gateway.

## Open questions

- Tenancy model and quota dimensions (concurrent runs, environments, tokens?).
- Do we expose `ListRuns` across cells (fan-out) or require a cell/label filter?
