# Durability of task and agent code

Status: **Proposed** · See [ADR-0013](../../decisions/0013-replay-durability.md)

Task and agent code is ordinary `async` Python. It survives worker and host crashes and suspends for hours at no
cost, without authors writing state machines.

## Why replay

CPython cannot serialize a suspended coroutine frame. The only way to resume in the middle of a method is to
re-run it and feed back the results of the effects it already performed. So:

- **Durability comes from the log.** Every effect is committed as `*.requested` before dispatch and `*.completed`
  after (P5). Recovery re-runs code and resolves its effects from the log.
- **Snapshots are an optimization.** Pickled task and agent state at resumable points bounds how much is replayed.
  Snapshots never move between runs. A snapshot that fails to load is discarded and the run replays from its
  start.

Replay does no I/O, so it is fast: re-running a 500-turn episode costs milliseconds of CPU.

## The driver

Coroutines are driven like generators; no asyncio event loop is involved.

```python
class Effect:
    def __await__(self):
        result = yield self            # suspends the whole await chain up to the driver
        return result

def advance(coroutine, value):
    try:
        effect = coroutine.send(value)  # run task/agent code until it awaits an effect
    except StopIteration as finished:
        return Finished(finished.value)
    return Suspended(effect)
```

- **Live path**: suspended coroutines stay in memory. `Step(input)` sends the completion into the coroutine that
  awaits it and runs until every coroutine is blocked; newly requested effects become events.
- **Replay path** (`Load`): the code re-runs from the latest snapshot (or the start). The *k*-th effect it requests
  is matched to the *k*-th committed `*.requested` event and resolved with its committed completion; effects with
  no completion stay pending.
- **Concurrency**: `run.gather(...)` runs sub-coroutines under a deterministic scheduler. On replay, completions
  are delivered in the order the log recorded them.
- **Effect identity**: `effect_id = {run_id}:{seq}` of the `*.requested` event. Because replay reproduces the
  same requests in the same order, retried effects carry the same identity and receivers deduplicate them.

### Divergence detection

On replay, each requested effect is compared with the committed one (kind + hash of arguments). A mismatch raises
`NonDeterminismError`; the run fails with `NON_DETERMINISM` and is quarantined rather than continuing on a
history it did not produce.

## Checkpoints and snapshots

**Implicit resumable points**: after each hook and after each turn of the loop. There the loop's own state
(turn, last observation, history) is derivable from the log, so the driver can resume exactly by restoring pickled
task and agent state.

**`@checkpoint`**: memoizes a unit of work inside a hook.

```python
    @checkpoint
    async def run_tests(self) -> TestReport:
        result = await self.workspace.execute("pytest -q", cwd="/workspace")
        return TestReport.parse(result.output)
```

- First execution: runs normally; on return the host emits `checkpoint.completed{name, ordinal, return_value,
  task_state}`.
- Replay: returns the recorded value immediately and restores task state; nothing inside runs again.
- Identity: method name + call ordinal (the third call to `run_tests` is a different checkpoint from the first).
- Rule: a checkpoint communicates only through its return value and `self`. Mutations to anything else are lost
  on replay.

**Snapshot format** (`HarnessHost.Snapshot`):

| Content | Notes |
|---|---|
| pickled task and agent objects | protocol 5; restricted unpickler (`find_class` allowlist: the run's code package + safe standard types) |
| driver position | hook name, turn, checkpoint ordinals |
| code references | snapshot is only valid for the same code |

- Handles (environments, models, tools) pickle as references (`__reduce__` → identifiers) and are rehydrated by
  the host.
- History is not pickled; it is derived from the log.
- Snapshots are capped (default 16 MiB); large values should be `BlobReference`s.
- Cadence: every hook boundary; every N turns (default 10) or when replay since the last snapshot exceeds a budget.
- Only blobs written by task hosts are ever unpickled; nothing originating in a guest is.

## Determinism rules

| Rule | Why | Enforcement |
|---|---|---|
| No I/O except through `run` and handles | replay | the task host runs in a sandbox with **no network** and no writable filesystem except scratch; violations fail loudly |
| No wall clock or ambient randomness | replay | `run.now()` (time of the latest input event) and `run.random` (seeded from `run_id`) |
| No iteration over sets of strings, no `id()`/`hash()`-dependent ordering | hash randomization differs between processes | `PYTHONHASHSEED` fixed in hosts; lint rule |
| No threads, no asyncio primitives outside `run` | the driver owns scheduling | the SDK rejects foreign awaitables |
| Replay requests the same effects | correctness | divergence detection |

The sandbox that enforces determinism is also the security boundary for task code (see
[trust-boundaries](../../architecture/trust-boundaries.md)).

## Versioning

- A run is pinned to the `code_reference`s it was created with. Task hosts load one code version per process;
  the runtime routes runs to hosts serving their version.
- Short runs (most reinforcement-learning episodes) simply finish on their version.
- Long runs survive deploys with `run.patched(change_id)`: live execution records a `patch.marked` event and
  returns `True`; replay of history recorded before the change finds no marker and returns `False`, taking the old
  path. The runtime records the switch as `code.upgraded` at a resumable point.
