# Several runners

Status: **Implemented** (2026-09-29) · Layer: durability · See [durability](README.md), [ADR-0017](../decisions/0017-dbos-substrate.md)

Several `DurableRunner` processes can share one Postgres database. They are peers: any of them accepts messages and
cancellations for any run, and any of them can execute any run. Without a database, a runner keeps its state in SQLite
files and is the only one, as before.

```python
runner = DurableRunner(
    directory,                                  # local files (e.g. environments)
    database="postgresql://…/rollout",          # shared: DBOS's tables and the run store
    runner_id="runner-0",                       # stable across restarts
    providers=…, tool_sets=…, environments=…,   # the same in every runner
)
```

```bash
uv run agents serve --database postgresql://…/sessions --runner-id server-0 --state /shared/sessions --port 8421
uv run agents serve --database postgresql://…/sessions --runner-id server-1 --state /shared/sessions --port 8422
```

Every runner must be configured alike (deployments, model providers, tools, environment backend), since a run can
move to any of them. The agent sessions servers must also share `--state`: environments and blobs are files, so for
now several servers means several processes on one machine, or a shared filesystem.

## Where runs execute

| Event | Where the run executes next |
|---|---|
| Started (a message to a new conversation, or `start`) | the runner that started it |
| Woken after eviction (a message, or its deadline) | whichever runner dequeues it: DBOS's `resume` puts it on DBOS's queue |
| Its runner dies, and is restarted with the same `runner_id` | whichever runner dequeues it: on launch, a runner puts its own unfinished runs on the queue |
| Its runner dies and stays down past `takeover_after` (15 s) | whichever runner dequeues it: a live runner takes over |

The queue hands each run to exactly one runner (DBOS's `ENQUEUED → PENDING` dequeue is atomic), which replays its
recorded steps and continues.

## What is shared and how

| Concern | One runner (SQLite) | Several runners (Postgres) |
|---|---|---|
| DBOS journal (steps, messages) | `dbos.sqlite` | the `dbos` schema; waits are woken by `LISTEN`/`NOTIFY` instead of polling |
| Run store (events, runs, conversations, messages, attempt markers, runners) | `runs.sqlite` | tables in the same database |
| One live run per conversation | an in-process lock | a database lock per conversation (Postgres advisory lock) |
| A message racing eviction | an in-process lock per run | a database lock per run, and the time of the last message kept in the store |
| Eviction | every idle run | only the runs executing in this process, which it can unload |
| Following up a finished run (messages it never consumed) | the runner | the runner the run finished on |
| Coordination store writes | SQLite's single writer | one at a time, by a transaction-scoped advisory lock |
| Coordination relay | one | one leader, elected by a database lock; another takes over when it stops |

The store code is the same for both databases: tables are declared with SQLAlchemy metadata and queries are portable
SQL (`rollout.database`).

## Heartbeats and takeover

Each runner writes a heartbeat every 2 seconds. Each runner also looks for runners whose heartbeat is older than
`takeover_after`; under a lock per dead runner, one of them asks DBOS to put the dead runner's pending workflows back on
the queue, and forgets it. A runner restarted with its id heartbeats again and releases its own unfinished runs as it
launches, so a restart does not wait for a takeover.

The risk is a false takeover: a runner that is alive but stalled for longer than `takeover_after` (a long GC pause, a
suspended VM) would have its runs executed twice at once. The run's recorded steps stay consistent (DBOS records each
step once), but an effect in flight on both could happen twice, except guarded effects, which the attempt markers
stop. Keep `takeover_after` well above any expected stall. Heartbeats compare wall clocks, so runners on different
machines need synchronized clocks.

## A fix on the way

`send` used to mark a message delivered before delivering it; a crash in between lost the message (a retry saw it
marked and did nothing). It now marks it after: a crash in between makes the retry deliver again, which DBOS
deduplicates while the same run is live.

## Verification

The tests start a throwaway Postgres with `pgembed` (a development dependency, no root or Docker needed).

- The in-process integration tests (durable runner, eviction, computer tools, agent sessions, coordination) run on
  both SQLite and Postgres.
- `tests/durable/test_cluster.py` runs two or three runners as separate processes (`tests/durable/peer.py`) on one
  database: messages through different runners reach one run in order, once each, including a new conversation
  messaged through every runner at once and a retry through every runner; a killed runner's run is taken over with no
  model call repeated except the one in flight; a restarted runner releases its runs at once; a command in flight
  when its runner is killed is not run again by the runner that takes over; an evicted run is woken by a message
  through another runner, and by its deadline; cancelling an evicted run through another runner tears it down; and
  messages racing eviction on every runner are all answered, once, in order.
- `tests/agent_sessions/test_sessions.py` runs three `agents serve` processes on one database: the lead's server is
  killed right after creating it and stays down; another server takes the lead over, the relay moves, and the
  fan-out completes with one environment per session.
- `agents faults --servers 3` runs the live fault evaluation against three servers, killing random ones; the first
  killed server stays down past the takeover time. Seed 31 (2026-09-29, `gpt-6-astra`): passed in 101 s with 6 kills
  across all three servers (one taken over), every session evicted and woken, 14 messages each delivered once, 6
  commands none run twice, and 7 of 55 model calls repeated (those in flight at kills).

## Not yet

- Environments on shared storage or behind a network service, so servers can run on several machines. Blobs can
  already live in object storage (`--blobs s3://…`; [local services](../development/local-services.md)).
- Waking other runners through `NOTIFY` when a relay or follow-up has work (they poll every few seconds).
- A follow-up lost if a runner dies between a run finishing and its unconsumed messages reaching the next run (as
  before; the activation queue will close it).
