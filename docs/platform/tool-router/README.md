# Tool Router

Status: **Proposed** · Layer: platform (optional) · See [ADR-0005](../../decisions/0005-tools-are-spec-plus-binding.md), [ADR-0012](../../decisions/0012-task-agent-loop.md)

## Purpose

Executes **imported tools**: external tool sets a task declares by name (`Task.imports`) and a `RunBinding` binds
per run — MCP servers, HTTP services, other agents, humans. It resolves them into pinned `ToolSpecification`s and
routes each call to its binding. It is the only place that holds external tool configuration and credential
references.

Tools defined *by the task* (`@tool` methods) and tools provided *by an environment image* do not go through the
router: they are task code ([task](../../core/harness/task.md#tools)). In the local profile, imported tools use the core's
in-process `ToolBinding` implementations; this service is the same protocol for the cluster and fleet profiles.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Resolving imports into `ToolSpecification`s | Which imports a task declares (Task) or how they are bound (RunBinding) |
| The Binding SPI and bindings: `mcp`, `http`, `agent`, `human` | `@tool` methods and environment-provided tools (task host) |
| MCP facade for unmanaged harnesses | Environments ([environment system](../../environments/README.md)) |
| Result normalization: size caps, overflow to blobs, provenance | Credentials (resolved by brokers; router holds references only) |
| | Retry decisions after a crash (runner, by `retry_class`) |

## Boundaries

| Direction | Counterpart | Contract |
|---|---|---|
| provides | runners | `ResolveImports`, `CallTool`, `Cancel` |
| provides | unmanaged harnesses | MCP facade per session |
| consumes | bindings | Binding SPI (below) |
| consumes | Control API | `agent` binding creates child runs |

## Interface

```proto
service ToolRouter {
  rpc ResolveImports(ResolveImportsRequest) returns (ResolveImportsResult);
  //   {run_id, imports: map<name, ToolBinding>, context: CallContext}
  //   → {specifications: [ToolSpecification] with retry_class / timeout / spec_hash filled}
  //   Deterministic given the same inputs and the same imported server responses.

  rpc CallTool(Effect) returns (Completion);      // payload: tool.requested; returns ToolResult or error class
  rpc Cancel(CancelRequest) returns (Empty);      // {effect_id}; best-effort
}

message ToolBinding {
  oneof kind {
    McpTool   mcp   = 1;  // {server_reference (cluster service), include[], rename{}}
    HttpTool  http  = 2;  // {endpoint, method, input_schema, grant_reference?}
    AgentTool agent = 3;  // {task and agent references, binding template} — runs a child run; completes via inbox
    HumanTool human = 4;  // {queue} — completes via inbox
  }
  RetryClass retry_class_override = 10;
  uint32     timeout_ms           = 11;
  map<string, string> description_overrides = 12;   // replaces untrusted third-party descriptions
}
```

### Binding SPI

```go
type Binding interface {
    Kind() string
    List(ctx context.Context, configuration BindingConfiguration, callContext CallContext) ([]ToolSpecification, error)
    Call(ctx context.Context, specification ToolSpecification, arguments json.RawMessage,
         callContext CallContext, effectID string) (ToolResult, error)
    Capabilities() BindingCapabilities // {Deduplicates, Cancel, Progress, LongRunning bool}
}
```

- `Deduplicates`: the binding guarantees three-state idempotency on `effectID` (agent, human: yes; http, mcp: only
  if the remote honors the forwarded key).
- `LongRunning`: the call returns an acknowledgement and the completion arrives via the run's inbox (agent, human).

## Semantics & guarantees

**Resolution and pinning**
1. `List` each bound import, apply `include` / `rename` / description overrides, compute `spec_hash`.
2. Resolve `retry_class` by precedence: trusted MCP annotations → binding default → operator configuration →
   RunBinding override. Untrusted servers' annotations are ignored.
3. The runner merges the result with the task's `@tool` specifications and commits `tools.resolved`. Resumes use
   the logged specifications; the router never re-lists for an existing run. If an imported server's spec hash has
   changed at call time, the call fails with `BINDING_ERROR{SPEC_DRIFT}` (or is allowed, per RunBinding policy).
4. MCP `list_changed` notifications are ignored; tool set changes are explicit `tools.changed` events.

**Calls**
- Validate arguments against `input_schema` (defense in depth; the runner validated first).
- Forward `effect_id` to the binding as an idempotency key (HTTP `Idempotency-Key`, MCP `_meta.idempotency_key`).
- Normalize: cap to `max_result_bytes` (overflow → `BlobReference`, `truncated = true`), set `provenance.untrusted`.
- Distinguish tool-level errors (`ToolResult.is_error`, shown to the model) from platform failures (`tool.failed`
  classes: `INVALID_ARGUMENTS`, `DENIED`, `DEADLINE`, `BINDING_ERROR`).

**MCP facade**
- For unmanaged harnesses, the router exposes a session's tool set as an MCP server (Streamable HTTP) at a
  per-session URL: the session's imports. Calls get synthetic `effect_id`s
  (`{session_id}:{MCP request id}`). Durability is the caller's problem.

## State & durability

Stateless. Idempotency state lives in the receivers (remote services that honor keys; the runner's mailbox for agent and
human completions). The router MAY keep a short-lived in-memory join table to collapse concurrent duplicate calls.

## Failure modes

| Failure | Result |
|---|---|
| Router instance dies mid-call | The runner re-dispatches the same `effect_id` |
| MCP server down | `tool.failed{BINDING_ERROR}`; retried per `retry_class` |
| Third-party tool with no deduplication dies mid-call | `tool.outcome_unknown` on takeover |

## Scale envelope

Only external tool traffic flows through the router (environment operations do not), so it
is far smaller than total tool traffic. Stateless horizontal scaling per cell.

## Build vs buy

Build (small). Buy MCP client/server SDKs for the `mcp` binding and the facade.

## Open questions

- Streaming tool progress to task code: skip initially (results only), or define `tool.progress` events?
