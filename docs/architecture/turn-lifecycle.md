# Lifecycle of a turn

Status: **Proposed** · Walks one turn of the [loop](../components/harness/README.md#the-loop-normative) across
every execution-plane boundary, including failure branches. Normative definitions live in the linked contracts;
this document is the integration view.

## Preconditions

- Run `R` (durable) lives in partition `p`, owned by worker `W` at lease epoch `e`.
- `W`'s task host has `R` loaded: the loop is suspended inside `task.respond`, awaiting environment effect `E1`
  (an `EXECUTE` issued by the task's `bash` tool). The last committed event is at `seq = n-1`.
- `R` owns environment `environment_1` (attached at creation; attachment token held by the runtime).
- Model slot `policy` is bound to recorder session `R/policy` on trainable channel `exp42/latest`.

## Happy path

```
 W (runtime)            Task host          Run Store       Recorder        Engine       envlet → envd
 ───────────            ─────────          ─────────       ────────        ──────       ─────────────
 ① completion E1 ─────▶ Step: resume bash tool → respond returns Observation
                        → loop: agent.act → model.sample awaits
                     ② [observation.recorded, model.requested]
 ③ Append([environment.completed E1 @n, observation.recorded @n+1, model.requested @n+2], fence=e) ─▶ ok
 ④ Sample(effect_id=R:n+2) ───────────────────────────▶ ⑤ deduplicate, match tree,
                                                          render delta, validate ──▶ ⑥ generate
                                                       ◀─ tokens, logprobs, version
                                                       ⑦ append span, parse
 ⑧ ◀──────────────────────────────────── canonical assistant message + usage
 ⑨ Step(model.completed) ─▶ loop: task.respond(reply) → run_tools → bash, read_file (gather)
                        ─▶ [environment.requested E3 (EXECUTE), environment.requested E4 (GET)]
 ⑩ Append(...) ─▶ ok; dispatch E3, E4 concurrently ─────────────────────────────────────▶ ⑪ Execute / Get
 ⑫ ◀──────────────────────────────────────────────────────────────── results E3, E4
 ⑬ Step(E3) → [] (gather waits); Append · Step(E4) → Observation(tool results) → next model.requested
```

1. **Completion arrives.** `E1`'s result reaches `W` in-process (it awaited the call) or via `R`'s inbox plus a
   nudge. `W` enqueues it on `R`'s mailbox; mailboxes are processed serially per run.
2. **Step.** `W` checks the completion is not already in the log, then calls `Step(R, environment.completed{E1})`.
   The task host resumes the awaiting coroutine: the `bash` tool returns, `run_tools` assembles the TOOL message,
   `respond` returns an `Observation`, the loop records it and calls `agent.act`, which awaits `model.sample`.
   `Step` returns the resulting events. Host state has advanced speculatively.
3. **Persist.** One fenced transaction appends the input and the step's events
   ([run-store](../components/run-store/README.md)). Nothing has been dispatched yet (P5). On `FENCED` or
   `SEQ_CONFLICT` the worker calls `Evict` on the host and drops the run (or partition).
4. **Dispatch.** `W` derives effect `E2 = R:n+2` from `model.requested` and calls the model endpoint.
5. **Recorder request path.** Deduplicate on `effect_id`; match the context against the session tree; extend
   tokens with the rendered delta (or re-render and start a new renderer epoch); reject sampling parameters outside
   the channel contract.
6. **Generate.** Tokens-in request to an engine replica chosen by session affinity. Every engine response is
   produced by exactly one weights version ([engine-adapter](../components/recorder/engine-adapter.md)).
7. **Record.** Append the sampled span (tokens, behavior logprobs, weights version); parse into a canonical message.
8. **Return.** Canonical message + usage (`context_used`, `context_limit`). No tokens or versions cross B5.
9. **Next step.** `model.completed` → `Step` → the loop calls `task.respond(reply)`; the default `respond` runs the
   reply's tool calls concurrently; each tool body awaits environment handles → two `environment.requested` events.
10. **Persist, then dispatch** both effects. The runtime checks `R` owns `environment_1` and attaches the token
    from `CallContext`.
11. **Environment call.** envlet verifies the attachment token, applies limits, forwards `Execute(effect_id=E3)`
    and `Get(E4)` to envd over vsock.
12. **Results** return as `ExecutionResult` and file bytes (oversized output moved to blob references).
13. **Partial completion.** Each completion is its own step; `gather` resumes when both are done, the tools
    return, and the next `Observation` + `model.requested` start the next turn.

**Environment-driven tasks** follow the same path with a different step 9: `respond` computes the observation in
Python (or with its own environment calls) instead of running tools.

**In-flight policy change** (step 6): if the Weight Update Controller transitions the replica, the engine aborts
the request and returns the partial output; the recorder records it as a span at the old version and resubmits
prompt + partial as tokens-in at the new version. Task and agent receive one ordinary message.

## Failure branches

| Failure | What happens | Lost |
|---|---|---|
| `W` dies before ③ commits | New owner loads `R` (replay to `n-1`); `E1` is pending → re-dispatched; envd returns the cached result | nothing |
| `W` dies after ③, before ④ | New owner replays; `E2` requested without completion → re-dispatched with the same `effect_id` | nothing |
| `W` dies during ⑤–⑦ | Recorder finishes and caches under `E2`; re-dispatch returns the cached completion; no duplicate sample | nothing |
| Task host crashes | Worker evicts the host's runs and reloads them on another host (replay) | nothing |
| `W` loses its lease but keeps running | Its next append fails fencing → evict, stop. It never dispatches uncommitted effects | nothing |
| A deposed worker dispatched just before losing its lease | New owner re-dispatches the same `effect_id`; the receiver's state is *in progress* → join | nothing |
| Replay requests a different effect than the log | `run.failed{NON_DETERMINISM}`; run quarantined | the run |
| Recorder instance dies mid-sample | Runtime retries `E2`; unflushed tree state lost → session flagged **incomplete** | training data for that session |
| Engine replica dies | Recorder retries on another replica (fresh prefill) | latency |
| Environment host dies | `environment.failed{LOST}` raised in the tool body → becomes an `is_error` tool result, or the task handles it | environment state since its last snapshot |
| Run Store primary fails over | Appends stall for the failover window (cell-local); leases survive | latency |

## Latency budget (per half-turn, excluding model, tool and task compute)

| Stage | Target p99 |
|---|---|
| Mailbox + `Step` + validation | ≤ 5 ms |
| Fenced append (one transaction) | ≤ 20 ms |
| Dispatch to endpoint / envlet / router | ≤ 5 ms |
| Recorder overhead (match + render delta + record) | ≤ 20 ms |
| **Total control overhead** | **≤ 50 ms** (N5) |
