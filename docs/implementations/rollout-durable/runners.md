# Several runners

Code: `rollout_durable.runner`, `rollout_durable.database`

Several [`DurableRunner`](README.md) processes can share one Postgres database. They are peers: any of them accepts
messages and cancellations for any run, and any of them can execute any run. Without `database`, a runner keeps its
state in SQLite files and is the only one.

```python
runner = DurableRunner(
    directory,                                  # local files
    database="postgresql://…/rollout",          # shared: DBOS's tables and the run store
    runner_id="runner-0",                       # stable across restarts
    providers=…, tool_sets=…, environments=…,   # the same in every runner
)
```

| Parameter of `DurableRunner` | Meaning |
|---|---|
| `database` | A Postgres URL, or a `Database`, shared with other runners. `None` is SQLite in `directory` |
| `runner_id` | This runner's name among them; it is also DBOS's executor id. Left out, it is `runner-` and 12 random hex digits, or `local` on SQLite |
| `heartbeat_interval` | Seconds between heartbeats |
| `takeover_after` | How long a heartbeat may stop before another runner recovers the runs |

Every runner must be configured alike (deployments, model providers, tool sets, environment backend), since a run can
move to any of them.

The [agent sessions](../../products/agent-sessions.md) server takes the same settings as flags:

```bash
uv run agents serve --database postgresql://…/sessions --runner-id server-0 --state /shared/sessions --port 8421
uv run agents serve --database postgresql://…/sessions --runner-id server-1 --state /shared/sessions --port 8422
```

Servers sharing a database must share `--state` too, because environments are directories under it. They run on one
machine or on a shared filesystem. Postgres and object storage for a development machine are in
[local services](../../development/local-services.md).

## Where runs execute

| Event | Where the run executes next |
|---|---|
| Started (a message to a new conversation, or `start`) | The runner that started it |
| Woken after eviction (a message, or its deadline) | Whichever runner dequeues it: DBOS's `resume` puts it on DBOS's queue |
| Its runner dies and is restarted with the same `runner_id` | Whichever runner dequeues it: on launch, a runner puts its own unfinished runs on the queue |
| Its runner dies and stays down past `takeover_after` | Whichever runner dequeues it: a live runner takes over |

The queue hands each run to exactly one runner (DBOS's `ENQUEUED → PENDING` dequeue is atomic), which replays its
recorded steps and continues.

## What is shared and how

| Concern | One runner (SQLite) | Several runners (Postgres) |
|---|---|---|
| DBOS journal (steps, messages) | `dbos.sqlite` | The `dbos` schema; waits are woken by `LISTEN`/`NOTIFY` instead of polling |
| Run store | `runs.sqlite` | Tables in the same database |
| One live run per conversation | An in-process lock | A Postgres advisory lock per conversation |
| A message racing eviction | An in-process lock per run | A Postgres advisory lock per run |
| Eviction | The runs resident in the process | Each runner evicts only the runs resident in its own process |
| Handing a finished run's unconsumed messages on | The runner | The runner the run finished on |
| Agent sessions' coordination store writes | SQLite's single writer | One at a time, by a transaction-scoped advisory lock |
| Agent sessions' coordination relay | One | One delivers, holding a database lock; another takes over when it stops |

## The database

`rollout_durable.database.Database` is a SQL database for stores: `Database.sqlite(path)` for one process,
`Database("postgresql://…")` for many. Store code is the same on both: tables are declared with SQLAlchemy metadata
(`Database.create(metadata)`) and statements are portable SQL with named parameters (`sql`, `fetch_one`, `fetch_all`).

| Member | What it gives |
|---|---|
| `read(query)` | Statements outside a transaction: each sees the latest committed state |
| `write(change, exclusive=name)` | One transaction. Writes with the same `exclusive` name run one at a time across processes. On SQLite every write takes the write lock as its transaction begins |
| `lock(name)` | An async context manager: mutual exclusion across every process using the database, for work that spans several writes and other calls. On Postgres it is an advisory lock, released if the holder's connection drops; on SQLite it is a lock among this process's tasks |
| `shared` | Whether other processes may use the database at the same time (Postgres) |

A store whose tools write deduplicates them by `effect_id` ([effects](../../libraries/rollout/contracts/effects.md)):

| Function | What it does |
|---|---|
| `effects_table(metadata)` | Declares the table `effects` (`effect_id`, `arguments_digest`, `result`) in the store's metadata |
| `recorded(connection, effect_id, arguments_digest, perform)` | Runs `perform` once per effect, inside the caller's transaction, so the write and the record of its result commit together. A repeated effect returns the recorded `ToolResult`. A known `effect_id` with another digest raises `Conflict`. The caller's writes must run one at a time (`write`, with `exclusive=` on Postgres) |

The project assistant's notes and agent sessions' coordination store are built this way. `temporary_postgres` and
`create_database` start a throwaway Postgres server and make a database on one, for tests and evaluations.

## Heartbeats and takeover

Each runner writes a heartbeat to the `runners` table when it launches and every `heartbeat_interval`. Each runner
also looks for runners whose heartbeat is older than `takeover_after`. Under a lock per dead runner, one of them asks
DBOS to put the dead runner's pending workflows back on the queue, and removes its heartbeat. A runner restarted with
its id heartbeats before it recovers, and releases its own unfinished runs as it launches, so a restart does not wait
for a takeover.

A runner that is alive but stalled for longer than `takeover_after` (a long pause, a suspended virtual machine) has
its runs executed twice at once. The recorded steps stay consistent, because DBOS records each step once. An effect
in flight on both happens twice, except guarded effects, which the attempt markers stop
([durable runner](README.md#effects)). Keep `takeover_after` well above any expected stall. Heartbeats compare wall
clocks, so runners on different machines need synchronized clocks.

## Tests

The tests start a throwaway Postgres with `pgembed`, a development dependency that needs neither root nor Docker.

- The in-process integration tests (durable runner, eviction, the database, computer tools, agent sessions,
  coordination) run on both SQLite and Postgres.
- `tests/rollout_durable/test_cluster.py` runs two or three runners as separate processes on one database. Messages
  through different runners reach one run in order, once each, including a new conversation messaged through every
  runner at once and a retry through every runner. A killed runner's run is taken over. A restarted runner releases
  its runs at once. A command in flight when its runner is killed is not run again by the runner that takes over. An
  evicted run is woken by a message through another runner, and by its deadline. Cancelling an evicted run through
  another runner tears it down. Messages racing eviction on every runner are all answered, once, in order.
- `tests/agent_sessions/test_sessions.py` runs three `agents serve` processes on one database. The server holding
  the lead session is killed and stays down; another server takes the session over, the relay moves, and the fan-out
  completes with one environment per session.

## Fault evaluation

`agents faults --servers 3` runs the agent sessions fault evaluation
([agent sessions](../../products/agent-sessions.md#durability-under-faults)) against three servers sharing a throwaway
Postgres. Kills hit random servers, and the first killed server stays down past the takeover time.

| Run | Kills | Messages | Commands (run twice) | Model calls (repeated) | Result |
|---|---|---|---|---|---|
| seed 31, `gpt-6-astra` | 6, across all three servers; one taken over | 14, each delivered once | 6 (0) | 55 (7) | passed in 101 s; every session was evicted and woken |
