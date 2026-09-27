# Canonical content

Status: **Proposed**

The model-agnostic representation of everything a model reads or writes. Task and agent code, the run log, the tool router and
the model endpoint contract use **only** this form (P4). Rendering to tokens happens exclusively in the recorder (or
inside a provider for direct adapters).

The shapes deliberately mirror MCP's `Tool` / `CallToolResult` and common chat APIs so that importing and
exporting is cheap — but this document, not MCP, is normative.

## Message

```proto
message Message {
  Role            role     = 1;  // SYSTEM | USER | ASSISTANT | TOOL
  repeated Block  content  = 2;
  string          name     = 3;  // optional speaker name (multi-agent)
  map<string,string> meta  = 4;  // not model-visible; never rendered
}
```

Rules:
- `TOOL` messages contain only `tool_result` blocks.
- `tool_call` blocks appear only in `ASSISTANT` messages.
- Every `tool_result.call_id` MUST match a `tool_call.call_id` earlier in the same context.

## Blocks

```proto
message Block {
  oneof kind {
    Text        text        = 1;
    Media       media       = 2;  // image, audio, document
    ToolCall    tool_call   = 3;
    ToolResult  tool_result = 4;
    Reasoning   reasoning   = 5;
  }
}

message Text      { string text = 1; }
message Media     { string media_type = 1; BlobReference source = 2; }
message ToolCall  { string call_id = 1; string name = 2; Json arguments = 3; }   // arguments: JSON object
message Reasoning {
  Scope   scope   = 1;  // PORTABLE: plain text any renderer may render (or drop)
                        // POLICY:   opaque, valid only for `producer`
  string  text    = 2;  // PORTABLE only
  string  producer = 3; // POLICY only: renderer_id or provider id that can consume `opaque`
  BlobReference opaque  = 4;  // POLICY only
}

message BlobReference {
  string uri        = 1;  // object storage URI
  bytes  sha256     = 2;
  uint64 size       = 3;
  string media_type = 4;
}
```

`POLICY`-scoped reasoning (e.g. encrypted provider reasoning) is dropped or summarized at renderer-epoch
boundaries by the recorder, according to the channel's configuration. The harness never needs to know.

## ToolSpecification

What the model sees about a tool, plus platform extensions that are never model-visible.

```proto
message ToolSpecification {
  // model-visible — covered by spec_hash
  string name          = 1;  // ^[a-zA-Z0-9_-]{1,64}$, unique within a run
  string description   = 2;
  Json   input_schema  = 3;  // JSON Schema 2020-12, type: object
  Json   output_schema = 4;  // optional

  // advisory (MCP-compatible), not model-visible
  ToolAnnotations annotations = 10;

  // platform extensions, not model-visible
  RetryClass retry_class      = 20;  // PURE | IDEMPOTENT | SIDE_EFFECTING | UNKNOWN
  uint32     timeout_ms       = 21;
  uint64     max_result_bytes = 22;
  bytes      spec_hash        = 23;  // sha256(JCS({name, description, input_schema, output_schema}))
}

message ToolAnnotations {  // hints only; ignored from untrusted servers
  string title = 1; bool read_only_hint = 2; bool destructive_hint = 3;
  bool idempotent_hint = 4; bool open_world_hint = 5;
}
```

## ToolResult

```proto
message ToolResult {
  repeated Block content    = 1;  // text | media only
  Json           structured = 2;  // optional; MUST validate against output_schema if present
  bool           is_error   = 3;  // tool-level error the model should see (not a platform failure)
  bool           truncated  = 4;  // content was cut to max_result_bytes; full output in `overflow`
  BlobReference        overflow   = 5;
  Provenance     provenance = 6;
}

message Provenance {
  bool   untrusted    = 1;  // true for anything originating in a guest or third-party server
  string binding_kind = 2;  // task | environment | mcp | http | agent | human
}
```

`is_error` is for failures the model should reason about (non-zero exit, file not found). Platform failures
(environment lost, deadline) are `tool.failed` events, not results — see [run-events](run-events.md).
