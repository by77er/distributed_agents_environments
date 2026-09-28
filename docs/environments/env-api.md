# Environment API (envd)

Status: **Preliminary** · Layer: environments (separate system, designed later)

> **Preliminary.** The environment system is a separate system with its own trade-offs and will be designed later.
> The core depends only on the minimal `Environments` / `Environment` handle protocols in
> [task.md](../core/harness/task.md#environments-optional); many tasks use no environment at all. The notes below
> record earlier thinking and research; none of it is decided.


The uniform contract every environment exposes, regardless of driver. Served by **envd** inside the guest;
reached only through the **envlet** on the host, over vsock (microVMs) or an equivalent private channel (pods,
VPS). Callers: the runner's environment client (every environment call made by task code), the Tool Router's
MCP facade (unmanaged sessions), operators via Control API.

## Path

```
Runner ──────mTLS gRPC──▶ envlet (host) ──vsock──▶ envd (guest)
               attachment token checked here     untrusted from here on
```

envlet is a **policy enforcement point**, not a pass-through: it authenticates the attachment token, enforces
size/rate/time limits on requests and responses, and records audit metadata.

## Interface

```proto
service Environment {
  // Processes
  rpc Execute(ExecutionRequest)            returns (stream ExecutionEvent);
  rpc ExecutionStatus(EffectReference)        returns (ExecutionState);          // three-state idempotency lookup
  rpc Signal(SignalRequest)        returns (Empty);              // {effect_id, signal}

  // Files
  rpc ReadFile(ReadFileRequest)    returns (stream Chunk);       // {path, offset?, length?}
  rpc WriteFile(stream WriteChunk) returns (WriteResult);        // first chunk: {effect_id, path, mode, atomic}
  rpc Stat(PathReference)                returns (FileInformation);
  rpc List(ListRequest)            returns (stream FileInformation);    // {path, recursive, limit}
  rpc Remove(RemoveRequest)        returns (Empty);              // {effect_id, path, recursive}
  rpc Move(MoveRequest)            returns (Empty);              // {effect_id, from, to}

  // Interactive (not used by default tools)
  rpc PseudoTerminal(stream PseudoTerminalInput)            returns (stream PseudoTerminalOutput);

  // Environment-provided tools
  rpc Manifest(Empty)              returns (EnvironmentManifest);
  rpc CallProvidedTool(ProvidedToolCall) returns (ToolResult);   // {effect_id, server, tool, arguments}

  // Lifecycle hooks (called by envlet, not by tool routers)
  rpc Quiesce(Empty)               returns (Empty);              // flush fs, pause for snapshot
  rpc Thaw(Empty)                  returns (Empty);
  rpc Health(Empty)                returns (HealthStatus);
}

message ExecutionRequest {
  string effect_id = 1;             // REQUIRED for tool calls
  oneof cmd { ArgumentVector argv = 2; string shell = 3; }
  string cwd = 4; map<string,string> env = 5; string user = 6;
  bytes  stdin = 7; uint32 timeout_ms = 8;
  uint64 max_output_bytes = 9;      // envlet also enforces its own cap
}

message ExecutionEvent {
  oneof e {
    bytes stdout = 1; bytes stderr = 2;
    ExitStatus exit = 3;            // {code, signal, timed_out, output_truncated}
    Joined joined = 4;              // this call joined an in-progress execution of the same effect_id
  }
}

message ExecutionState { oneof s { Empty absent = 1; Empty running = 2; ExecutionResult done = 3; } }

message EnvironmentManifest {
  string envd_version = 1;
  repeated string capabilities = 2;         // "exec", "fs", "pty", "provided_tools", …
  repeated ProvidedToolServer tool_servers = 3;  // task-specific MCP servers shipped in the image
}

message ProvidedToolServer { string name = 1; repeated string command = 2; string transport = 3; /* "stdio" */ }
```

## Semantics & guarantees

- **Three-state idempotency (P6)** on every mutating call carrying `effect_id`: absent → execute; running →
  join (stream the remaining output, emit `Joined`); done → replay the cached result.
- **Idempotency cache** lives on the guest's disk (not tmpfs), so it survives envd restarts and is captured in
  both disk and full snapshots. Retention ≥ 24 h or until the env is destroyed. It is inside the untrusted guest:
  a compromised guest can only lie about its own results (see [trust-boundaries](../architecture/trust-boundaries.md)).
- **Provided tools**: task images MAY ship MCP servers (stdio). envd launches them lazily and proxies calls.
  This is how an environment brings its own tools without any task or agent change. Their annotations are untrusted.
- **Limits** enforced by envlet regardless of what envd claims: max output bytes per call, max concurrent calls
  per attachment, max request size, wall-clock deadline.
- **No network assumptions**: envd MUST work with `network: none`. It never calls out.
- **Quiesce/Thaw** bracket snapshots: `Quiesce` syncs filesystems and stops accepting new mutating calls
  (in-flight ones finish or are paused with the VM).

## Compatibility

Proposed starting implementation: fork of E2B's envd (Apache-2.0). Its protocol is the *starting point*; this
document is the contract. Divergences are expected (effect_id idempotency, provided-tool proxy, quiesce).

## Open questions

- Should `edit_file`-style structured edits be envd primitives, or remain tool-router adapters over
  Read/WriteFile? (Leaning: adapters — keep envd minimal.)
- Port forwarding / preview URLs for remote coding (a human viewing a dev server): envd capability or
  envlet-level proxy?
