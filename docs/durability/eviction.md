# Evicting idle runs

Status: **Working** (2026-10-02) · Code: `rollout.durable.runner`

A run that waits for a message is a DBOS workflow blocked in a receive. It holds its coroutine chain (program, task,
agent, history) in memory, and on SQLite DBOS polls the database once per second for every waiting receive. The
`DurableRunner` therefore unloads runs that have waited long enough, and loads them again when they have something to
do. An evicted run has not ended: `teardown` does not run and its environment stays.

| Parameter of `DurableRunner` | Default | Meaning |
|---|---|---|
| `evict_after` | `timedelta(minutes=5)` | how long a run waits before it is unloaded; `None` keeps every run resident |
| `eviction_interval` | `5.0` | seconds between looks for runs to evict or wake |

## Mechanism

The run store is the source of truth: `runs.evicted` marks an evicted run, `runs.wake_at` holds its wait's deadline,
`runs.last_activity` the time it was last messaged or woken, and `runs.evictions` counts its evictions. DBOS itself
shows an evicted run as `CANCELLED`.

| Step | What happens |
|---|---|
| Evict | Every `eviction_interval`, the runner looks at the runs resident in its own process. A run whose latest event is a `run.suspended` older than `evict_after` is evicted: the store records `evicted` and `wake_at` (the suspension time plus the wait's timeout, if it has one), DBOS cancels the workflow, and the runner cancels the workflow's asyncio task |
| Wake on a message | `send` delivers to the run's inbox as usual (DBOS keeps messages for cancelled workflows), clears `evicted` and asks DBOS to resume the workflow |
| Wake at the deadline | The same periodic task resumes evicted runs whose `wake_at` has passed. DBOS receive timeouts count from when the wait began, so the replayed receive returns at once |
| Cancel | `cancel` wakes an evicted run, so that it runs `teardown` and ends |
| Restart | `launch` does not load evicted runs; they stay unloaded until woken |

A resumed workflow goes onto DBOS's queue and replays: its recorded steps return their results without executing,
and the run then takes the message or the timeout.

Three details keep this safe:

- **Unloading.** The asyncio task is cancelled with the message `rollout: unload`. The episode loop skips `teardown`
  for that cancellation.
- **Races.** Evicting, delivering and waking a run all hold a lock on that run. Under the lock the runner checks again
  that the `run.suspended` event is still the run's latest and that the run was not messaged or woken since.
- **Runs just woken.** A woken run's latest event is still its old `run.suspended` until it has processed the message
  or timeout. `last_activity` stops the runner from evicting it again in that time. A run woken by a message of a kind
  it does not wait for stays resident until its wait ends.

Waking replays the run's whole recorded history, so its cost grows with the run's length. On SQLite a wake takes
about a second. With several runners, each evicts only the runs resident in its own process, and any of them may
execute a woken run ([several runners](runners.md)).

## Measurements

100 idle conversations of 10 turns each, on SQLite, with a model endpoint that keeps nothing in memory:

| | Python heap | Idle CPU | Asyncio tasks |
|---|---|---|---|
| Resident | +13 MiB (128 KiB each) | 8.9% of a core | 201 |
| Evicted | +4 MiB (37 KiB each) | 0.4% of a core | 2 |

Idle CPU grows linearly with resident runs: at 300 idle conversations it was 36% of a core resident and 0.4% evicted.
Waking a run with 10 or with 200 recorded steps executed none of them again.

## Tests

`tests/durable/test_eviction.py` evicts a run and wakes it by a message (no model call on replay, gapless events),
wakes a run at its wait's deadline, and cancels an evicted run. The agent sessions fault evaluation kills servers
while every session is evicted and again while they wake
([agent sessions](../products/agent-sessions.md#durability-under-faults)).
