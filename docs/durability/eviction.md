# Evicting idle runs

Status: **Implemented** (options D and A, 2026-09-29) · Layer: durability · See [durability](README.md)

A conversation that waits for its next message is a DBOS workflow blocked in a receive. Agent sessions spend most of
their life like that. This document measures what idle runs cost, compares ways to unload them, and proposes one.

## What an idle run costs today

A waiting run holds its coroutine chain (program, task, agent, history, and every event it recorded, kept in memory
by the run context) and a DBOS receive. The receive does not hold a thread: it is an asyncio future. But with SQLite,
DBOS has no push notifications, so a background thread polls the database **once per second for every waiting
receive**, and each receive also re-checks the database on its own.

Measured with the `DurableRunner` on SQLite (RTX-5080 workstation, WSL2; conversations with 10 turns of about 2 KB
messages, then idle for 20 seconds):

| Idle conversations | Memory | Idle CPU |
|---|---|---|
| 1 | +4 MiB | 0.6% of a core |
| 100 | +92 MiB (about 0.9 MiB each) | 13% |
| 300 | +156 MiB (about 0.5 MiB each) | 36% |

CPU grows linearly with idle runs: about a thousand idle sessions would keep a core busy doing nothing. Memory grows
with history length. A restart also replays every idle run's full history before it can take messages.

## Options

| Option | Frees memory | Stops polling | Bounds replay | Cost |
|---|---|---|---|---|
| **A. Evict by cancel, wake by resume** (DBOS `cancel_workflow` / `resume_workflow`) | yes | yes | no | small; no change to task code |
| **B. One activation per message** (the design's end state): `WaitFor` exports task and agent state and ends the workflow; the next message starts a new one from that state | yes | yes | yes | task and agent state must be serializable, and the loop must be re-entrant at a wait |
| C. Postgres (LISTEN/NOTIFY instead of polling) | no | yes | no | needed later anyway; memory unchanged |
| D. Keep runs resident, drop the in-memory event list | partly | no | no | trivial; worth doing regardless |

### Option A, tested

A spike cancelled idle workflows blocked in a receive, sent them a message, and resumed them:

- the cancelled workflow's coroutine left memory (its asyncio task ended);
- a message sent while it was cancelled was received when it resumed;
- resuming replayed the recorded steps without executing any again (10 and 200 recorded steps: 0 re-executed);
- waking took about one second, bounded by the polling interval;
- an awaiter of the cancelled workflow's result gets `DBOSAwaitedWorkflowCancelledError`, so the runner's watcher must
  treat eviction as expected.

DBOS receive timeouts are durable (measured earlier: a timeout fires relative to when the wait began, across a
restart), so a run evicted while waiting with a timeout, such as a scheduled follow-up, only needs to be resumed at
its deadline; the replayed receive then returns immediately.

## Implementation

**D** and **A** are implemented in the `DurableRunner`; **B** stays the end state once generations and state export
exist. `DurableRunner(evict_after=timedelta(minutes=5), eviction_interval=5.0)`; `evict_after=None` keeps every run
resident. The agent sessions server takes `--evict-after SECONDS` and shows evicted sessions as `sleeping`.

Results (100 conversations of 10 turns, idle; the model endpoint keeps nothing, so only the runtime is measured):

| | Python heap | Idle CPU | Asyncio tasks |
|---|---|---|---|
| Resident | +13 MiB (128 KiB each) | 8.9% of a core | 201 |
| Evicted | +4 MiB (37 KiB each) | 0.4% of a core | 2 |

At 300 idle conversations, idle CPU fell from 36% of a core to 0.4%.

Two things the implementation had to add to the plan below:

- **Unloading the coroutine.** `cancel_workflow` stops DBOS tracking a workflow, but a coroutine blocked in a receive
  only notices at its next database re-check, so the runner also cancels the workflow's asyncio task. That
  cancellation carries the message `rollout: unload`, and the loop does not call `teardown` for it: an evicted run has
  not ended, and tearing it down would, for example, destroy its environment.
- **Not re-evicting a run that was just woken.** A woken run's latest event is still its old `run.suspended` until it
  processes the message or timeout, so the runner remembers when it last delivered to or woke each run, and only
  evicts runs that suspended after that. A run woken by a message it then ignores (another kind) and that suspends
  again without an effect stays resident; that errs on the safe side.

Verified by `tests/durable/test_eviction.py` (evict and wake by message, with no model call on replay and gapless
events; wake at a wait's deadline; cancel an evicted run, which tears it down) and by the agent sessions fault
evaluation with a 5-second eviction threshold, which kills the server while every session is evicted and again
while they wake (see [agent sessions](../products/agent-sessions.md#durability-under-faults)).

The design as proposed:

1. **Store.** A run's liveness in `runs.sqlite` gains `evicted`, and a `wake_at` time for runs waiting with a timeout.
   The store, not DBOS's status, is the source of truth: DBOS shows evicted runs as `CANCELLED`.
2. **Evict.** A background task finds runs whose last event is `run.suspended` and that have been idle longer than a
   threshold (default 5 minutes); for each it records `evicted` and `wake_at`, then calls `cancel_workflow`.
3. **Wake on a message.** `send` to an evicted run delivers to its inbox as usual (DBOS keeps messages for cancelled
   workflows), clears `evicted`, and calls `resume_workflow`. A per-run lock stops a wake racing an eviction.
4. **Wake on time.** The same background task resumes evicted runs whose `wake_at` has passed.
5. **Cancel.** Stopping an evicted run resumes it with a cancel request in its inbox, so `teardown` still runs and
   its environment is destroyed.
6. **Restarts.** Evicted runs are `CANCELLED` in DBOS, so a restart does not replay them: only active runs are
   recovered. This also bounds restart time.

Costs and risks: waking replays the run's recorded history (cheap per step, but linear in length, which generations
will bound); a wake takes about a second; DBOS's own tooling shows evicted runs as cancelled; and eviction must never
cancel a run that is between steps rather than waiting, which the `run.suspended` check and the per-run lock guard.

How it would be verified: the idle-cost measurement repeated with eviction (memory and CPU should stay flat as idle
sessions grow), a test that evicts and wakes a run by message and by timer, and the agent sessions fault evaluation
with eviction enabled and a short threshold, so kills land on evicted, waking and active runs.
