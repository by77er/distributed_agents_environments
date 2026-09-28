# Session trees

Status: **Proposed** · Layer: core

A session is not a list of turns. Harnesses retry, branch sub-agents off shared prefixes, sample best-of-n,
compact and edit history. The recorder therefore stores each session as a **prefix tree of token spans**. A
trajectory is a path.

## Data model

```proto
message SessionRecord {
  string session_id = 1; string run_id = 2; string channel = 3; string policy_id = 4;
  Mode mode = 5; map<string,string> labels = 6;
  SessionStatus status = 7;                 // OPEN | CLOSED | ABANDONED
  Completeness completeness = 8;            // COMPLETE | INCOMPLETE{reason}
  repeated Epoch epochs = 9;
}

message Epoch {                             // maximal stretch where every request extends the previous tokens
  uint32 epoch = 1;
  string renderer_id = 2;
  EpochReason reason = 3;                   // START | RENDERER_CHANGE | CONTEXT_EDIT (compaction) | PREFIX_MISMATCH | STATE_LOST
  string root_node_id = 4;
}

message Node {
  string node_id = 1; string parent_id = 2;
  uint32 epoch = 3;
  Span span = 4;
}

message Span {
  SpanSource source = 1;                    // CONTEXT | SAMPLED
  repeated int32 tokens = 2;
  // SAMPLED only
  repeated float behavior_logprobs = 3;     // processed distribution actually sampled from (P10)
  uint64 weights_version = 4;               // exactly one per span
  BlobReference top_logprobs = 5;                 // optional, opt-in
  // attribution
  CanonicalReference canonical = 6;               // which canonical items / effect produced these tokens
  repeated KVCacheEpoch kv_epochs = 7;           // best-effort, diagnostic only (see ADR-0009)
}

message KVCacheEpoch { uint32 pos_start = 1; uint32 pos_end = 2; oneof v { uint64 version = 3; bool unknown = 4; } }

message Turn {
  string effect_id = 1;                     // dedupe key
  string context_leaf = 2;                  // node where the request's context ends
  repeated string response_nodes = 3;       // ≥1 SAMPLED spans (more than one if split by a weight transition)
  SamplingParameters sampling = 4;
  SamplingParameters client_requested = 5;      // compat OVERRIDE mode: what the client asked for
  FinishReason finish = 6;
  repeated Interruption interruptions = 7;  // {at_token, from_version, to_version, cause: WEIGHT_UPDATE}
  string engine = 8; string replica = 9;
  Timing timing = 10;
}
```

## Matching a request

1. Resolve the request's context to a node: by `parent_digest` + `keep_prefix` (native) or by longest canonical
   prefix match (compat).
2. If the resolved node's epoch renderer is the channel's current renderer **and** the new items are a pure
   append: render only the new items with `bridge_to_next_turn` and append a `CONTEXT` span.
3. Otherwise: start a new epoch — render the full context as a `CONTEXT` span under a new epoch root, with the
   reason recorded.
4. Sample; append one or more `SAMPLED` spans; record the `Turn`.

## Path selection

- **Managed runs**: the trajectory path ends at the response node of the run's last `model.completed` effect.
- **Unmanaged sessions**: the most recent turn by completion time.
- Off-path branches (retries, best-of-n siblings, crash leftovers) are retained and exported as `siblings`.

## Export

- Segments are Parquet files under `sessions/{cell}/{date}/{session_id}/` (`nodes`, `turns`, `session`).
- On `Close(final = true)` the recorder writes a **manifest** listing all segments and the completeness verdict,
  then inserts a row in the export index (`session_id`, manifest URI, labels) that the Assembler polls.
- A session is `COMPLETE` iff it was closed with `final = true` by the same instance lineage with no lost
  unflushed state. Anything else is `INCOMPLETE{reason}`; the Assembler decides whether to use it.

## Retention

Hot trees: until close + flush. Exported segments: per-job TTL (default 30 days); trajectories are the durable
training artifact.
