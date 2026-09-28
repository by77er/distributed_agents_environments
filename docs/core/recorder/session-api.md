# Recorder session API

Status: **Proposed** · Layer: core (service form)

The network form of the recorder: the same operations the in-process `Recorder` offers, plus
OpenAI/Anthropic-compatible endpoints for unmanaged harnesses.

## Session management

```proto
service RecorderSessions {
  rpc Open(OpenSessionRequest)   returns (Session);   // idempotent on session_id
  rpc Close(CloseSessionRequest) returns (Empty);     // {session_id, final: bool}
  rpc Get(SessionReference)            returns (Session);
}

message OpenSessionRequest {
  string session_id = 1;            // {run_id}/{slot} (managed) or minted by Control API (unmanaged)
  string run_id     = 2;            // empty for unmanaged
  string channel    = 3;            // resolved per request via Policy Registry
  Mode   mode       = 4;            // ACTIVE | PASSIVE
  SamplingParameters sampling = 5;      // from the RunBinding; validated against the channel contract
  ClientSamplingPolicy client_sampling = 6;  // REJECT_MISMATCH | OVERRIDE (compat endpoints only)
  FlushMode flush   = 7;            // TURN | BATCH
  map<string,string> labels = 8;    // from the run (job_id and caller labels such as "group"); copied into exports
}

message Session {
  string session_id = 1; SessionStatus status = 2;   // OPEN | CLOSED | ABANDONED
  CapabilityContract contract = 3;
  string base_url = 4;              // compat endpoint root for unmanaged harnesses
  string credential = 5;            // scoped to this session
}
```

- Runs: the runner opens a session per recorded model slot at run start and closes it when the run ends
  events with `final = true`.
- Unmanaged harnesses: the caller (in the platform layer, the Control API) opens the session and hands `base_url` + `credential` to whoever launches
  the foreign harness. Configuration is only "set your base URL"; no code changes (R6).

## Native endpoint (managed runs)

Implements the [model endpoint contract](../../contracts/model-endpoint.md) exactly, at
`/sessions/{session_id}` — `Describe`, `Sample` (accepts `ContextDelta`), `Cancel`.

## Compatibility endpoints (unmanaged harnesses)

| Path | Protocol |
|---|---|
| `{base_url}/v1/chat/completions` | OpenAI Chat Completions (streaming and non-streaming) |
| `{base_url}/v1/messages` | Anthropic Messages (streaming and non-streaming) |

Mapping rules:
- Requests are converted to canonical content; tool definitions to `ToolSpecification`; responses back to the protocol's
  format. The full message list is matched against the session tree (compat clients send full context).
- **Idempotency**: an `Idempotency-Key` header is honored as the effect id. Without one, every request is a new
  sample; client retries become sibling branches in the tree (harmless; excluded from the final path).
- **Sampling parameters** in the request (temperature, top_p, …): `OVERRIDE` (default for unmanaged) ignores them
  and uses the session's, recording what the client asked for; `REJECT_MISMATCH` returns 400 when they differ.
- Model name in the request is ignored; the session's channel decides.
- Responses never include logprobs or token ids, even if requested, unless the session was opened with a
  debug flag by an operator.

## Errors

As in the model endpoint contract, mapped to protocol-native status codes: `OVERLOADED` → 429 with
`retry-after`, `CONTEXT_OVERFLOW` → 400 with the protocol's context-length error shape, `CONTRACT_VIOLATION` → 400.
