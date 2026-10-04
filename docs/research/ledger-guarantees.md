# Ledger guarantees

Code: `rollout_train.ledger`, `rollout_train.database`, `rollout_train.rollouts.scheduler`, `rollout_train.sandboxes`,
`rollout.harness.sandboxes`, `rollout_train.checkpoints`, `rollout_train.loop`, `rollout_durable` · Tests:
`tests/research/test_ledger_guarantees.py`, `tests/research/test_durable_guarantees.py` · See
[checkpoints](../libraries/rollout-train/checkpoints.md#the-ledger), [rollouts](../libraries/rollout-train/rollouts.md),
[training](../libraries/rollout-train/training.md), [sandboxes](../libraries/rollout/sandboxes.md),
[several runners](../implementations/rollout-durable/runners.md)

Processes on one machine or many cooperate through three kinds of shared state:

- a ledger: append-only tables and fences;
- ordinary mutable tables beside it: presence (beats), launches, the registry, run settings and sandbox leases;
- a content-addressed blob store.

This page checks, guarantee by guarantee, what the code promises and whether it keeps the promise. Each section gives
the claim, the mechanism, the assumptions it rests on, and the verdict, with file and line evidence. Where the
guarantee fails, it gives a scenario and a proposed fix. The last section says what a proposed HTTP ledger service would
have to guarantee for all of this to carry over.

The analysis is of `main` at `6ec3721`. Line numbers are of that commit.

**Evidence.** Each violation found has a test marked `xfail(strict=True)`: the suite stays green, and a fix turns the
test into a failure until its mark is removed. Tests that pass show a guarantee holding under real concurrency:

- several processes sharing a `FileLedger`;
- threads, each with a connection pool of its own, on SQLite;
- the same on Postgres. The tests run against a real Postgres 17 started by `pgembed`, in its default READ COMMITTED
  isolation.

Some failures need a crash or a pause at one exact moment. Those tests either build the state that moment leaves (a
line half written, a beat that stopped), or run the other party's step at that moment, through a ledger or store
wrapped to do so. Every step in them is the code's own.

## Findings, ranked

Safety means wrong recorded state, lost data, or two holders of one thing. Liveness means work stalls, is redone or
leaks.

| # | Kind | Finding | Test |
|---|---|---|---|
| 1 | Safety | A runner whose claim lapsed (it paused) still records its episode. The keeper deleted its sandbox when the claim lapsed, so its run fails, and that failure, made by the platform, becomes the episode's outcome and drops the newer attempt's result | `test_a_runner_whose_claim_lapsed_does_not_record_the_episode` |
| 2 | Safety | Two claims of one episode hold at once: a lapsed claim holds again when its runner beats again, and an adoption races a new attempt | `test_a_lapsed_claim_does_not_hold_again_beside_a_newer_attempt`, `test_an_adoption_and_a_new_attempt_never_both_hold` |
| 3 | Safety | Retention can delete a blob that a checkpoint added at the same moment names (content addressing finds it stored and does not write it again) | `test_thin_never_deletes_a_blob_a_concurrent_add_names` |
| 4 | Safety | `FileLedger`: after a writer dies mid-line, the next acknowledged append is lost, and its key then accepts a second, different record | `test_an_append_after_a_torn_line_is_kept` |
| 5 | Safety | Staleness and wall-time limits compare wall clocks. A runner whose clock is 90 s behind looks dead while it plays (finding 1, everywhere at once), and a clock stepped forward ends sandbox leases early | `test_a_live_runner_whose_clock_is_behind_keeps_its_claims`, `test_a_clock_stepped_forward_does_not_end_a_lease_early` |
| 6 | Safety | Durable runner: a run replayed after its terminal event was stored appends a second terminal event (introduced by `42e324d`) | `test_a_run_replayed_after_it_ended_has_one_terminal_event` |
| 7 | Safety | Launch states are overwritten unconditionally: a stop asked for while a run starts is lost, and a stop racing a claim marks a running launch `stopped` | `test_a_stop_asked_for_while_a_run_starts_stops_it`, `test_a_stop_racing_a_claim_never_marks_a_running_launch_stopped` |
| 8 | Safety | Two makers of one suite leave a suite that neither made | `test_two_makers_of_one_suite_leave_one_of_their_suites` |
| 9 | Safety | Two loops of one run (a replaced one that has not noticed yet) share unfenced side effects: the run's directory, the trainer, the bookmark | none (analysis) |
| 10 | Liveness | A runner that dies between taking its fence and adopting (or whose take is applied twice) loses all its durable runs on its next start | `test_a_runner_that_died_while_preparing_adopts_its_runs_when_started_again` |
| 11 | Liveness | `FileLedger`: a crash while taking a fence leaves `fences.json` empty, and the ledger is unusable until repaired by hand | `test_a_crash_while_taking_a_fence_leaves_the_ledger_usable` |
| 12 | Liveness | `FileLedger` blocks the event loop on its lock and rereads whole tables on every append, which can delay beats | none (analysis) |
| 13 | Liveness | The keeper decides from a read and releases afterwards: a lease whose claim was adopted in between is released, and its run is played again | none (analysis) |
| 14 | Documentation | "A table's records … in the order they were appended" does not hold on Postgres for tables that several scopes write | `test_appends_of_two_scopes_to_one_table_get_distinct_positions_on_postgres` |

The [durable runner](#9-the-durable-runner) section lists further problems found by reading `rollout_durable`. Only
finding 6 among them is tested.

## 1. Fencing

**Claim.** "Whoever means to write takes the fence of a scope: a number higher than any taken before. An append that
carries an older fence is refused (`Fenced`), so a process that was replaced, and does not know it yet, cannot write
over its replacement" (`ledger.py:10-12`, checkpoints.md "The ledger").

### `FileLedger` across processes: holds, on one machine's local filesystem

- **Mechanism.** Every operation holds `flock(LOCK_EX)` on `DIRECTORY/.lock` (`ledger.py:133-141`).
  - `take` reads `fences.json`, adds one and writes it back, under the lock (`ledger.py:79-84`).
  - `append` compares its fence with the newest, checks the key, and appends one line, all under the same lock
    (`ledger.py:86-96`).
  - So a take and an append are totally ordered, and an append under fence *n* succeeds only if no take after *n*
    came before it.
- **Evidence.** In `test_processes_sharing_a_ledger_of_files_take_distinct_fences_and_stale_appends_are_refused`,
  four processes take and append 40 times each. Every take gets a number of its own, and the fences of the records,
  in file order, never go down.
- **Assumptions.**
  - The directory is on a local Linux filesystem. `flock` across machines on NFS, and on WSL's `/mnt/c`, is
    unverified, and the docs say a ledger of files serves one machine.
  - No process writes the files except through `FileLedger`.
- **Not durable against crashes.** These do not break fencing, but break the ledger:
  - **A torn line loses the next append (finding 4).** `_read` skips a line that does not parse (`ledger.py:118-123`),
    on the grounds that "a line a dying writer left half written … was never appended". The next `append` opens the
    file in append mode and writes its line directly after the torn bytes, with no newline between
    (`ledger.py:94-95`). Both are now one unparsable line: the new record is dropped, although `append` returned
    True. Its key is free again, so a second, different record is accepted as the first. For a claim, that means two
    runners play one attempt.
    - A torn line needs a writer killed between two `write` system calls of one record (a record larger than the 8 KiB
      buffer), or a full disk.
    - **Fix:** under the lock, if the file does not end in `\n`, truncate it to its last newline (or write a newline)
      before appending; `fsync` the file before returning True.
  - **A crash while taking a fence (finding 11).** `fences.json` is rewritten in place (`write_text`, `ledger.py:83`).
    A crash between the truncate and the write leaves an empty or partial file, so every later `take` and `append`
    raises `JSONDecodeError`. This fails loudly rather than unfencing, because truncated JSON never parses.
    - **Fix:** write a temporary file, `fsync` it, then `os.replace`. `FilePresence`, `FileLeases` and `FileLaunches`
      already write this way (`presence.py:87-89`).
  - Nothing is `fsync`ed. After a power loss, a take may be lost while an append made under the new fence survives.
    The fence would then go back to *n − 1*, and the replaced writer could write again. Unverified: it depends on the
    filesystem's ordering of writes to two files.

### `DatabaseLedger` on SQLite: holds

- **Mechanism.** Every write transaction begins with `BEGIN IMMEDIATE` (`rollout_durable/database.py:111,176-179`),
  which takes SQLite's single write lock. `take` is an upsert that returns the new number (`database.py:122-132`).
  `append` reads the newest fence and inserts in the same transaction (`database.py:134-152`). Writes are therefore
  serial. Reads run outside a transaction and see the latest commit (`rollout_durable/database.py:126-129`).
- **Evidence.** In `test_writers_on_connections_of_their_own_take_distinct_fences_and_stale_appends_are_refused[sqlite]`,
  four threads, each with an engine of its own, take and append 30 times each. Every take gets a number of its own,
  and fences never go down in append order.
- **Assumption.** A local filesystem. WAL mode does not work over a network filesystem.

### `DatabaseLedger` on Postgres, READ COMMITTED: holds

- **Mechanism.**
  - Both `take` and `append` run `SELECT pg_advisory_xact_lock(hash("ledger:" + scope))` first
    (`rollout_durable/database.py:131-137`). The lock is held until the transaction commits.
  - Postgres makes a commit visible before it releases the transaction's locks. So the next holder's `SELECT number`,
    which under READ COMMITTED takes a fresh snapshot per statement, sees any take committed before it got the lock.
  - An append cannot interleave with a take of its scope: it is ordered either before the take (and the take's
    number is higher) or after it (and the append sees the new number and raises `Fenced`).
- **Can two takes return the same number?** No. They are serialized by the advisory lock. Without it, `INSERT … ON
  CONFLICT DO UPDATE` would still lock the row and re-read its latest version.
- **Can an append commit under a stale fence?** No, as long as every writer goes through `DatabaseLedger`. `copy`
  writes fences under another lock (`database.py:452-474`) and is documented to run with nothing writing.
- **Evidence.** The same threaded test, `[postgres]`, passes on a real Postgres.
- **Assumptions.**
  - The connection to Postgres is direct or through a session-mode pooler. Transaction-scoped advisory locks work
    behind a transaction-mode pooler such as pgbouncer, but `Database.lock` (session advisory locks) does not.
  - Two scope names whose 64-bit hashes collide only serialize more. That is harmless.

### First append wins: holds, atomically

- `FileLedger`: the key check and the write happen under the directory lock (`ledger.py:90-96`).
- `DatabaseLedger`: the table's primary key is `(name, key)`, and the insert is `ON CONFLICT (name, key) DO NOTHING`
  (`database.py:40-48,143-150`). This is atomic on both databases, whatever the locks.
- **Callers that ignore the answer.** A caller that loses an append learns nothing about the winner unless it reads
  back:
  - `Checkpoints.add` reads back (`checkpoints.py:203-204`), and so does `reshard` (`resharding.py:82-83`).
  - `make_suite` does not (`evals.py:103-106`), which gives finding 8.
  - `EpisodeRunner._ended` does not (`scheduler.py:494`). That is intended: the loser's episode is dropped.

### Order of records: does not hold on Postgres (finding 14)

- **Claim.** `read` returns "a table's records by key, in the order they were appended" (`ledger.py:52-53`).
- **Mechanism.** Each insert computes `position` as `MAX(position) + 1` for its table (`database.py:146`). It does so
  under the lock of its fence's scope, not of its table.
- **What breaks.** Several scopes write one table: every runner writes `runs/RUN/claims`, `episodes`, `interrupted`
  and `adopted`, and every run writes `checkpoints`. On Postgres, two such appends at once both read the same `MAX`,
  and both get the same position. Ties then come back in any order. Nothing has a unique index on `position`.
- **Evidence.** `test_appends_of_two_scopes_to_one_table_get_distinct_positions_on_postgres` runs the second append
  while the first transaction is open, and gets positions `[1, 1]`.
- **Scope.** SQLite serializes every write, so it is unaffected. Nothing found depends on the order of these tables
  except what is displayed. The same `MAX` scan also makes each append cost O(rows of the table), since there is no
  index on `(name, position)`.
- **Fix.** Take the position from a sequence or an identity column (commit order still differs from position order,
  which matters for streams: see [HTTP](#10-a-proposed-http-ledger-service)). Or lock on `ledger-table:NAME` as well.
  Or say "in an order consistent with each scope's appends".

## 2. Claims and episodes

**Claims** (rollouts.md "What runners write"):

- "two runners never play one attempt";
- "A claim holds while its runner keeps its fence and beats";
- "The first record of an episode is its record";
- at-least-once play: an episode with no record and no claim that holds is open again.

**Mechanism.**

- A claim is an append to `runs/RUN/claims` under `GROUP/EPISODE/ATTEMPT`, under the runner's fence `runners/NAME`
  (`scheduler.py:377-390`).
- `holds` (`scheduler.py:74-90`) says a claim holds if all of these are true:
  - it is not noted in `interrupted`;
  - it was made under its runner's newest fence, or adopted under it;
  - its runner's newest beat is no older than `STALE` = 90 s, unless the reader is that runner.
- `open` reads the fences, the beats and six tables, one after another and not as one snapshot. It offers an
  episode's next attempt when no claim of it holds (`scheduler.py:288-322`).
- The episode's record is an append under `GROUP/EPISODE` (`scheduler.py:490-494`), under the runner's own fence.

**What holds.**

- One attempt, one runner: the claim key is unique, and first append wins.
- One record per episode: the record key is unique, and first append wins.
- Training cannot mix two attempts within one episode. A record names one run's trajectory blob
  (`episodes.py:127-143`), and the recorder's segments are kept by `run_id` (`scheduler.py:474`).
- At-least-once play holds as long as some runner with room keeps serving.

### A zombie's record is accepted (finding 1)

The fence that guards the record is the runner's own, and it does not move when the runner's claim lapses for want
of beats. A runner that was paused (a stop-the-world pause, a suspended VM, a network partition) and resumes therefore
records its episode, although by the scheduler's own rule its claim no longer holds.

1. Runner Z claims `1/1/1`, acquires its sandbox, and stops beating (paused).
2. After 90 s, `holds` says the claim lapsed. The keeper releases Z's lease at its second look and deletes the
   sandbox (`sandboxes.py:212-221`).
3. Runner F sees `1/1` open, claims `1/1/2`, and acquires a new sandbox.
4. Z resumes. Its next sandbox call fails (the sandbox is gone), so the run fails, and Z appends the episode's record
   under `runners/Z`, which is still Z's newest fence. It is the first record.
5. F's completed episode is refused (`append` returns False), and nobody notices.

In a training run, the group's result then counts a failure that the platform caused. In an eval, it counts against
the checkpoint (`evals.py:240-268`). Without sandboxes, Z's completed episode wins instead. That is harmless (it is one
attempt's data), but the run trains on an attempt whose claim had lapsed.

`test_a_runner_whose_claim_lapsed_does_not_record_the_episode` plays exactly this, with the real runner, keeper and
pool. The recorded outcome is `failed: KeyError: 'there is no sandbox …'`.

### Two claims of one episode hold at once (finding 2)

`holds` never asks whether a later attempt exists. There are two ways for two claims of one episode to hold at once:

- **A lapsed claim holds again.** Continue the scenario above. When Z beats again, its claim `1/1/1` holds again,
  beside F's `1/1/2`. A pool beside the ledger then admits Z's key again (`sandboxes.py:182-193`). If Z had not yet
  acquired its sandbox, it gets a fresh one and plays on. Both runners believe, correctly by `holds`, that they hold
  the episode. Test: `test_a_lapsed_claim_does_not_hold_again_beside_a_newer_attempt`.
- **Adoption races a new attempt.** A runner started again takes its fence anew (`scheduler.py:225`). That makes its
  old claims stop holding until it appends their adoption (`scheduler.py:431`). `_adopt` reads the claims, checks that
  no later attempt exists (`scheduler.py:429`), and only then appends. If another runner claims attempt 2 inside that
  window (between the read and the append), both the adoption and attempt 2 stand. Test:
  `test_an_adoption_and_a_new_attempt_never_both_hold`. The window is the whole of `_adopt`, which reads four tables
  per run, for every run.

Consequences: the episode is played twice, and one pool place is used twice. The first record wins as before. If the
adopted run ends first, its record is left out of training (`RESUMED`, `scheduler.py:480-481`), and the trainable
attempt is dropped.

### Fix for findings 1 and 2: a fence per episode

Give each episode a scope of its own, `runs/RUN/episodes/GROUP/EPISODE`.

- To claim, take that scope's fence and append the claim under it. The attempt number can be the fence number.
- Append the episode's record, the `interrupted` note and the adoption under the same fence.
- A newer claim then fences out every older attempt at once:
  - a zombie's record raises `Fenced`;
  - an adoption after a newer claim raises `Fenced`;
  - `holds` needs only "is this attempt the episode's newest fence, and is its runner alive".
- This uses only the ledger's existing operations. The cost is one fence row per episode.

A smaller fix, without fences: a claim holds only if it is its episode's latest attempt, and `_ended` appends only
after reading that its attempt is still the latest. That narrows the window but does not close it.

### Adoption takes "my previous fence" to be "my fence − 1" (finding 10)

`_adopt` judges a claim held if it was made under `self._fence.number - 1`, or adopted under it (`scheduler.py:406,427`).

- A runner that takes its fence and dies before adopting (an out-of-memory kill, a crash in `prepare`) moves its fence
  twice. On its next start, every claim it held is two fences back. It notes them all `LAPSED` and cancels their runs.
- Over HTTP, a retried `take` does the same (see [HTTP](#10-a-proposed-http-ledger-service)).
- Test: `test_a_runner_that_died_while_preparing_adopts_its_runs_when_started_again`.
- This is liveness only: the episodes are played again.
- **Fix:** adopt a claim made or adopted under any earlier fence of this runner, provided it is not noted
  interrupted and no later attempt of its episode exists. Fences of one name only grow, and one incarnation runs at a
  time, so "earlier" is enough; with a fence per episode (above) the check is the episode's fence alone.

### `holds()` and time of check against time of use

- `open` and `holding` read fences, beats and tables at different moments. A stale read can only make a claim look
  lapsed later, or held later, than it is. Duplicates that follow are absorbed by first append wins, except as above.
- `admits` decides, and then `acquire` creates the sandbox (`harness/sandboxes.py:319-347`). A claim that lapses in
  between keeps its sandbox until the keeper's two looks. That is liveness only.

## 3. Clocks

Every timestamp below is `time.time()` on the machine that wrote it.

| Decision | Writer's clock | Reader's clock | Effect of skew or a step | Kind |
|---|---|---|---|---|
| A runner is alive: beat `at` ≥ now − 90 s | runner (`presence.py:84`, `database.py:342`) | whoever judges: other runners, every pool's `admits` and keeper (`presence.py:55-57`) | Writer 90 s behind, or reader stepped 90 s ahead: a live runner's claims lapse. Others play them again, keepers delete its sandboxes, and finding 1 follows for every episode it plays. Writer ahead: a dead runner's claims hold longer | **Safety** (two holders, wrong outcomes) |
| "Two sweeps in a row" | keeper (`sandboxes.py:212-221`) | keeper | none: two looks, counted, with no times compared | — |
| Sandbox wall-time limit: `Lease.ends` ≤ now | pool, at acquire (`harness/sandboxes.py:336,345`) | same pool, at sweep (`harness/sandboxes.py:387,392`) | A step forward ends leases early and deletes sandboxes in use. A step back lets sandboxes overstay | **Safety** (a sandbox deleted under its run) |
| Durable runner takeover: heartbeat older than `takeover_after` | each runner | each runner | A live runner's runs are executed twice at once (documented, runners.md) | **Safety** |
| Durable eviction and waking (`wake_at`, `last_activity`, `recorded_at`) | several machines | evicting runner | Runs evicted or woken early or late | Liveness |
| Launch `at` (ordering of launches asked for) | whoever asked | launcher | Launches started out of order | Liveness |
| Run state shown by the monitor (beat ages) | runner | monitor | Wrong display | Display |

Tests:

- `test_a_live_runner_whose_clock_is_behind_keeps_its_claims` writes a beat stamped by a clock two minutes behind. The
  runner's live claim does not hold.
- `test_a_clock_stepped_forward_does_not_end_a_lease_early` steps the pool's clock forward two hours. A sandbox
  acquired seconds ago, with an hour's limit, is deleted.

WSL2 is prone to both problems: its clock drifts while the host sleeps, and is stepped when it resyncs.

**Fixes.**

- **Liveness by observation, not by comparison.** A reader judges a runner dead when the runner's beat has not changed
  for 90 s by the reader's own `time.monotonic()`. Keep `(runner, at)` → first time seen, in memory. Skew between
  machines then cannot matter, and a step of the reader's clock does not matter either. The price is that a reader
  that starts fresh waits 90 s before it judges anyone dead. Keepers and runners run long, so that is acceptable.
- **Or database time.** `DatabasePresence.beat` writes `at = now()` on the server (Postgres `clock_timestamp()`). The
  staleness test becomes SQL (`now() - at > interval '90 s'`). One clock, so no skew. This is the natural form behind
  an HTTP service.
- **Wall-time limits.** Store the limit as a duration, and judge it from the pool process's monotonic clock. On a pool
  restart, credit the time already used from the database's clock, not the local one.
- Keep `takeover_after` and `STALE` well above the largest expected skew. Alert when a beat's `at` is in the reader's
  future.

## 4. Leases and sandboxes

**Claims** (sandboxes.md "In training: a lease ends with its claim"):

- at most one live sandbox per key;
- a lapsed claim gets no sandbox;
- a runner that dies leaves its sandboxes, which are deleted within two minutes;
- a sandbox no lease names is deleted.

**Mechanism.** `SandboxPool` (`harness/sandboxes.py:275-435`):

- reads its leases once (`_load`, `:429-432`) and then keeps them in memory, under in-process per-key locks;
- names each sandbox by a hash of its key (`handle_of`, `:269-272`), so a retried acquire finds the same sandbox;
- `admits` refuses a key whose claim does not hold, and ends its lease (`:319-322`).

The keeper (`sandboxes.py:202-230`) releases leases found lapsed at two looks 15 s apart. `sweep` deletes sandboxes the
provider has that no lease or making names (`harness/sandboxes.py:401-403`).

**What holds.**

- One sandbox per key, within one pool process: the per-key lock serializes the lookup and the make, and the handle
  is fixed by the key.
- The keeper cannot release a lease acquired after its read. `sweep` releases only keys it found lapsed twice
  (`sandboxes.py:217-219`), and `_making` and `_made` protect sandboxes being made.
- HTTP pools map `NoCapacity`, `LeaseRefused` and `SandboxLost` to 503, 409 and 410, and back
  (`harness/remote.py:151-157,231-233`). Acquire and release are idempotent by key, so a client may retry either.

**What does not.**

- **A rightful holder can lose its sandbox.** The rule is "a lease ends with its claim", so a runner whose claim
  lapsed is by definition not the rightful holder. The problem is that the claim lapses on a clock (section 3) or a
  pause, and the runner carries on (finding 1). A clock step deletes sandboxes under their runs (finding 5).
- **A pool name used by two processes at once.** Pools are named `KIND@HOST/DIRECTORY` (`profile.py:396`). Nothing
  fences a pool's name, so two processes of one run's directory on one host (finding 9) load the same leases and keep
  separate memories of them. Each process's `sweep` deletes sandboxes "no lease names" by its own memory.
  - With a provider whose `held()` lists sandboxes on the machine rather than in the process (Minecraft worlds,
    `worlds.py:133`), one process deletes the other's sandboxes. Unverified with a real provider.
  - **Fix:** the pool takes a fence `pools/NAME` and refuses to sweep once it is fenced out. Or the database holds the
    single truth of the leases, with the per-key lock as an advisory lock.
- **Leaks.** A lease of a run the ledger does not know ends only when released (documented). The leases of a pool name
  that is never opened again (a host renamed, a directory moved) are never swept. Neither are their sandboxes, for a
  provider whose sandboxes outlive the process (a cloud API).
  - **Fix:** a sweep by any keeper of leases whose pool has not beaten for a long time.
- **The keeper decides from a read and releases afterwards (finding 13).** `ended` reads the ledger, and `sweep` then
  releases. An adoption appended in between makes the claim hold again, but the release still happens. The adopted
  run then fails with `SandboxLost`, is noted `LOST`, and is played again. This is liveness only, and needs an
  adoption that took longer than two looks (30 s).
- **Calls are not checked against the lease.** `SandboxPool.call` and the HTTP `/call` operate on whatever sandbox
  the key hashes to, without checking that the key holds a lease (`harness/sandboxes.py:369-374`). A zombie whose
  lease was not yet swept keeps operating. Any client that knows a key can operate its sandbox (see scoped tokens).

## 5. The training loop

**Claims** (`loop.py:16-40`, training.md "Dying and starting again"):

- every action can be taken twice;
- a step is decided before it is taken;
- a checkpoint is made once;
- one loop at a time;
- the settings a step used are recorded, and used again when it is taken again;
- an unfinished eval is finished before anything is decided.

**What holds.**

- **Decided before taken.** The step's record, with its groups, parent, checkpoint id (`makes`, a fresh random id),
  seed, batch manifest and settings, is appended before the trainer is called (`loop.py:370-385`).
- **Made once.** On a retake, the loop looks for `makes` first (`loop.py:389`). If two trainers both produce weights
  for one `makes`, only the first `add` appends. `add` returns the checkpoint that is there (`checkpoints.py:203-204`),
  and a replaced loop's `add` raises `Fenced`. The loser's blobs are written but never named. Content addressing makes
  that harmless (a leak, not a corruption), unless the loser shares the winner's working directory (below).
- **Settings once per step.** `refresh` runs only between steps (`loop.py:459-460`). The settings go into the step's
  record (`loop.py:382`), and a retake applies the recorded ones (`loop.py:397-398`).
- **Scheduled evals.** Steps are serial, and the next step starts only after the previous step's eval
  (`loop.py:435,459-472`). So only the head checkpoint's eval can be unfinished, and a restart finishes it first
  (`loop.py:289-291`). The eval run's name is the same for that step each time it is asked for (`evals.py:150-158`),
  and its groups and results are keyed, so nothing is done twice.

**Assumptions and gaps.**

- **What a retake trains on is recomputed.** The groups come from the record, but the segments are recomputed from
  the episodes with the algorithm and the trainer's budget of the restarted process (`loop.py:322-324,368-369,448`).
  The trainer is not given the recorded batch. A restart with another algorithm setting, or on a trainer with another
  `Budget`, takes the step over different segments than its record and its checkpoint's `batch` say.
  - **Fix:** train from the recorded batch manifest (sources and advantages), loading the segments by source.

**Two loops of one run (finding 9).** The fence stops only ledger appends. A replaced loop that has not yet tried to
append keeps doing everything else:

- **The trainer.** It writes into `DIRECTORY/MAKES` (`loop.py:395-400`). The new loop, retaking the same step,
  removes that directory first (`loop.py:396`) and trains into it while the old loop may still be writing there. Both
  are on one machine when the same run directory is started twice.
- **Hard links.** `kept` hard-links each working file into the blob store and makes it read-only
  (`checkpoints.py:335-345`, `blobs.py:89-103`). A write into a working file after that writes into the blob, and the
  blob no longer matches its hash.
  - Reads through `Checkpoints.files` link the blob into the working directory without checking its hash
    (`checkpoints.py:252-259`), so corrupted weights are served.
  - Read-only mode stops a writer that is not root, but not a trainer running as root (common in containers).
    Unverified with a real trainer.
- **`serve()`.** It deletes everything in the directory but its own checkpoint and that checkpoint's parent
  (`loop.py:209-211`), including the other loop's working and served directories.
- **The bookmark.** `made` moves the profile's bookmark (`profile.py:434-437`). The registry is not fenced, so a late
  old loop moves it back.
- **Fix:** take the run's fence again before each side effect that matters, and check it after. Better, give each
  loop incarnation a working directory of its own (`DIRECTORY/FENCE/…`), copy rather than hard-link files a process
  may still write, and fence registry writes made on a run's behalf.

## 6. Retention and blobs

**Claim.** "A release is appended to the ledger before its blobs are deleted, and a blob is deleted only if no
checkpoint still names it, so this may be repeated after a crash at any point" (`checkpoints.py:209-211`, checkpoints.md).

**Mechanism.** `thin` appends releases, reads every checkpoint, collects the hashes that unreleased checkpoints name,
and deletes the released checkpoints' blobs outside that set (`checkpoints.py:207-233`).

**Crash safety: holds.** A release is recorded before any deletion, and every step is repeatable.

**Atomicity against a concurrent `add`: does not hold (finding 3).** The check and the delete are separate, and `add`
writes nothing when the blob is already there: `_write_once` and `_linked_once` return early (`blobs.py:91-92,107-108`),
and so does `S3BlobStore._put_once` (`store.py:68-70`).

1. `thin` (run X) reads the names. Blob *B* is named only by a checkpoint it just released.
2. `add` (run Y, or a merge, imitation, or a replaced loop) finds *B* stored and appends a checkpoint naming it.
3. `thin` deletes *B*.

Y's checkpoint now names a blob that is gone. `test_thin_never_deletes_a_blob_a_concurrent_add_names` does exactly
this. A file whose bytes recur across checkpoints is what makes it happen: a configuration file, a tokenizer, or a
checkpoint forked from the one being thinned. The same race exists against what `keep` lists:

- `keeping()` reads the runs' starts and the bookmarks before `thin` (`loop.py:218-225,431`);
- a run that starts from a checkpoint after that read, or a bookmark set after it, does not keep it;
- a run's `start` is checked for `weights is None` only when its files are fetched.

**Do datasets, episodes and reshards count as names?**

- No: only checkpoints' `weights` and `state` count (`checkpoints.py:223-225`).
- Today that is safe, because only those blobs are ever deleted, and episode, dataset and batch blobs do not share
  bytes with them.
- A `verbatim` reshard's manifest names the same blobs as its checkpoint's weights (`resharding.py:73`). It loses them
  with its checkpoint, which is then released anyway.
- A future deletion of other blobs would need every manifest counted.

**Fix.**

- Keep a grace period: a blob is deleted only if no checkpoint names it, and it is older than *G* (much longer than any
  `add` takes).
- `put` of a blob that exists refreshes its time: a `utime` for files, a copy-in-place for S3.
- Or put deletion behind one lock (`blobs`) that `add` holds from its first `put` to its append, and that `thin` holds
  from reading the names to its last delete.
- Behind a service, deletion belongs to the service alone (see HTTP).

## 7. Launches and settings

**Claims.**

- A launcher claims a launch asked for. The claim is atomic in both stores: `FileLaunches.claim` under the directory
  lock, and `DatabaseLaunches.claim` with `WHERE state = 'asked'` under an exclusive lock (`launches.py:154-164`,
  `database.py:306-316`). **This holds.**
- "A launch asked to stop is stopped by its launcher" (`launches.py:8`), and `stop` on one not started yet stops it at
  once (`monitor/system.py:262-269`).

**Mechanism.** `note(id, **changes)` overwrites whatever state is there (`launches.py:166-174`, `database.py:318-327`).
The launcher sends the stop signal only to launches it finds in `stopping` (`launcher.py:133-139`).

**What does not hold (finding 7).**

- **The launcher's `running` overwrites a stop.** It writes `running` after starting the process
  (`launcher.py:191`). A stop written between the claim and that write is overwritten, and the run is never signalled.
  Test: `test_a_stop_asked_for_while_a_run_starts_stops_it`.
  - The Ray path checks `state == claimed` before writing `running` (`launcher.py:252-253`), but in a separate read,
    so the same race remains.
- **The monitor's `stop` reads, then writes.** It writes `stopped` for a launch it read as `asked`. A launcher that
  claims and starts it in between leaves a launch marked `stopped`, a final state, whose process runs on unsignalled.
  `_watch` later overwrites it with `ended`. Test: `test_a_stop_racing_a_claim_never_marks_a_running_launch_stopped`.
- **Fix.** Make every transition a compare-and-set in the store, for example `note(id, expect={CLAIMED},
  state=RUNNING)`, which returns the launch as it is when the expectation fails. Encode the state machine once:
  - `asked → claimed | stopped`;
  - `claimed → running | stopping | failed`;
  - `running → stopping | ended | failed`;
  - `stopping → stopped | ended | failed`.
  The launcher, on finding `stopping` where it expected `claimed`, signals the process it just started.

**Run settings: hold.** `want` merges under a lock (`database.py:380-392`). The loop reads the settings between steps
only, and records what each step used. A change written mid-step applies to the next step, as documented.

**The registry.** Names are checked and written under one lock (`database.py:212-223`). Bookmarks are last writer
wins (see finding 9).

## 8. Suites (finding 8)

**Claim.** A suite is "written once under the suite's fence (`suites/NAME`) and never changed" (`evals.py:6-9`).

**Mechanism.** `make_suite` checks that the name is free, takes the fence, then appends the suite's record and each
start under keys of their own (`evals.py:92-106`). `suite_for` calls it on first use.

**What breaks.**

1. Maker A and maker B both find no suite `s`.
2. A takes fence 1, appends the record and start 1.
3. B takes fence 2. A's next append raises `Fenced`.
4. B's record and start 1 already exist: its appends return False, unread. Starts 2 and 3 are B's.

The ledger now holds A's record (A's seeds) with starts `[A1, B2, B3]`, a suite neither made, and B plays its own list.
Test: `test_two_makers_of_one_suite_leave_one_of_their_suites`.

For an environment's own eval data, both makers draw the same starts, so this is harmless. It bites only on
`make_suite` by hand with different rows or seeds, or when environment versions differ.

**Fix:** one record holding the whole suite, or read back after appending and return what the ledger holds.

## 9. The durable runner

Code: `rollout_durable.runner`, `rollout_durable.context`, `rollout_durable.store`. This section was established by
reading the code. Only the first item was run.

**How events are numbered.** Each execution numbers its events from an in-memory counter. The store inserts on the
primary key `(run_id, seq)` with `ON CONFLICT DO NOTHING` (`store.py:323-331`), so the first writer of a `seq` wins.
A replay's events that were already stored are dropped, which is what keeps the log free of duplicates.

- **A second terminal event (finding 6, tested).**
  - What `42e324d` changed: a terminal event takes `MAX(seq) + 1` whenever any event is stored (`context.py:97-102`),
    so that a replay that took another way still ends its stream.
  - What that broke: after the terminal event is stored, `execute` still releases environments and sandboxes and
    finishes the run, before DBOS records the workflow's output (`runner.py:358-371`). A replay that starts in that
    window appends a second terminal event:
    - after a crash;
    - after `close()`, which unloads every run whose `finish_run` has not run (`runner.py:229-239`);
    - from a second executor.
  - If the replay takes another way (its sandbox refused after the release), the second terminal event can say
    something else, and `finish_run` rewrites the outcome.
  - Test: `test_a_run_replayed_after_it_ended_has_one_terminal_event` gets `[(7, run.completed), (8, run.completed)]`.
    `DurableRunHandle.events()` stops at the first terminal event, so an episode runner records the first. The
    store's log, the feed and the outcome may disagree with it.
  - **Fix:** bump `seq` only while no terminal event is stored, and make `finish_run` conditional on the run still
    running.
- **Hooks see replayed events again.** `_recorded` publishes every event, whether or not the store kept it
  (`runner.py:375-377`). **Fix:** have `append` return whether it inserted, and publish only then.
- **Teardown on DBOS control flow (unverified).** The harness skips `teardown` only for an `UNLOAD` cancellation
  (`harness/loop.py:54-59`). DBOS raises `DBOSWorkflowConflictIDError` in the losing executor and
  `DBOSWorkflowCancelledError` on cancellation, and both are `BaseException`s (`dbos/_error.py:370-388`), so
  `teardown` runs in the loser.
- **Eviction writes the store, then DBOS (unverified).** A crash between `store.evict` and `cancel_workflow`
  (`runner.py:458-459`) leaves a pending workflow marked evicted. A later `resume` then puts it on the queue
  alongside the recovered copy: two executors.
- **Unloading mid-effect (unverified).** The cancelled effect's `effect.completed: failed` is stored, and the replay's
  `ok` at the same `seq` is dropped.
- **Two executors at once (documented).** A runner stalled past `takeover_after` is executed twice. Effects in flight
  happen twice, except guarded ones. A loser that runs entirely behind also repeats what lies outside steps: it
  releases environments and sandboxes, and finishes the run.
- **Liveness (unverified).**
  - The eviction loop has no `try` (`runner.py:442-462`): one error stops eviction and deadline waking on that
    runner for good.
  - Waking clears `evicted` before `resume`: a failed resume leaves the run cancelled and not evicted, so it is never
    woken.
  - An exception after the terminal event (a failed environment destroy) skips `finish_run`, and the run stays
    running.

## 10. A proposed HTTP ledger service

Under the proposal, runners use `HttpLedger(url, token)`, which implements `Ledger`, `Presence`, `Leases`, `Launches`
and `DesiredSettings` against a service in front of the database. Every property above holds today because each
operation is one transaction on one database, answered over a connection that either commits or fails. HTTP adds:

- lost responses;
- retries;
- replicas;
- caches;
- a delay between deciding and acting.

Below is what each operation must keep.

### Operations and what each needs

**`take(scope)`.**

- It is not idempotent: a lost response followed by a retry takes two fences.
- Fencing survives that, because the client uses the number it got. But anything that reads meaning into the number
  breaks. `_adopt`'s "previous fence = mine − 1" (finding 10) cuts every durable run short, and `plans` and `starts`
  get gaps.
- The service must accept a client request id. It stores `(scope, request id) → number` in the same transaction as
  the increment, and answers a retry with the stored number.
- The client sends one request id per logical take, and retries with it until it gets an answer.

**`append(table, key, record, fence)`.**

- It is idempotent by key, but the answer is not. If the first attempt committed and its response was lost, the retry
  answers False ("the key is there").
- `_claim` reads False as "another runner claimed this attempt first" (`scheduler.py:387-388`) and does not play it.
  The claim holds as long as the runner beats, so the episode never ends: a stall for the whole group, and so for the
  run.
- The service must answer with whether this request wrote the record. It stores the request id with the record, or
  compares the stored record with the one sent, and returns the stored record when they differ.
- The fence check and the insert must be one transaction under the scope's lock, as today. The check must happen when
  the append is applied, not when the request arrives, so that an append delayed past a newer take is refused.

**Timeouts.**

- A request that timed out on the client may still be applied later.
- The client must never treat a timeout as "not appended" or "not taken". It retries with the same request id until it
  gets a definite answer, or gives up and treats its own state as unknown: it stops claiming, takes its fence again,
  and adopts.
- `Fenced` must be an error of its own (409 with a code), never retried.
- Client deadlines must be well under `STALE`, so that a runner stuck in a request is not mistaken for alive.

**Reads.** Reads that decide something must see every write acknowledged before them:

- `thin`'s names (finding 3 becomes worse with a stale replica);
- `_adopt`'s claims;
- `holding`/`open`;
- `checkpoint(makes)` before a retake;
- `episodes_of`.

Today every read goes to the primary. With replicas, the service must either:

- serve these reads from the primary (linearizable), or
- give each client a session token (the commit position of its last write) and make a replica wait until it has
  applied that position. That gives read-your-writes and monotonic reads per client. Across clients, it is not enough
  for `thin` and `holding`, which must read from the primary.

Two service replicas in front of one primary are fine. Each request is one transaction there, and advisory locks are
transaction-scoped.

**Change streams instead of polling.**

- A stream needs a position in commit order. `position` is neither unique (finding 14) nor in commit order. A
  sequence is assigned at insert, so a row numbered 11 can commit before row 10, and a consumer resuming "after 11"
  never sees 10.
- The service must use one of:
  - positions assigned under a per-table lock in the inserting transaction (serial per table);
  - Postgres logical decoding (commit LSN);
  - a resume rule that re-reads from the oldest position still in flight (`pg_snapshot_xmin`).
- Delivery is then at least once, resumable with an opaque token, and in order per table. Records are immutable and
  keyed, so consumers deduplicate by `(table, key)`.
- A consumer whose token is too old falls back to a full read. It is always safe, because tables only grow.
- Mutable tables (presence, leases, launches, settings) stream as "this row changed, at version *v*". Consumers re-read
  the row and treat the event as a hint.

**Presence and claims by server time.** Beats get `at = now()` on the server, and staleness is judged in SQL against
`now()`. Section 3's safety problems then go away for everything behind the service.

**Synchronous claims.**

- A `claim(runner, runs, room)` call can choose open episodes and insert their claims in one transaction, judged by
  server time. That closes the gaps in `open`/`holds` (section 2), and lets the per-episode fence be taken in the same
  transaction.
- It must itself be idempotent by request id. Otherwise a lost response leaves claims that hold while nobody plays
  them, the same stall as above. The retry returns the claims already made.

**Scoped tokens.**

- Today any holder of any fence may append to any table: nothing ties a fence's scope to the tables it writes, and a
  claim's `runner` field is whatever the writer says.
- The service should bind each token to a role and a name, check that a fence's scope belongs to the token, and check
  that the table belongs to the scope.

| Role | Takes | Appends to | Mutable writes | Reads |
|---|---|---|---|---|
| Runner `NAME` | `runners/NAME` (and per-episode scopes) | `runs/*/claims`, `episodes`, `interrupted`, `adopted`, with `runner = NAME` | its own beat; blob puts | every run's `plans`, `groups`, `results`, `claims`, `episodes`; fences; beats |
| Training loop of run `R` | `runs/R` | `runs/R/*`, `checkpoints`, `checkpoints/released`, `checkpoints/resharding`, `checkpoints/resharded` | the bookmarks its profile names; blob puts | everything |
| Pool `NAME` | `pools/NAME` (proposed) | nothing | its own leases (`pool = NAME`), its beat | claims, fences, beats |
| Launcher `NAME` | nothing | nothing | claim and note launches; `note` only on launches it claimed; its beat | launches |
| Monitor or operator | `suites/*`, `datasets/*`, `merges` | suites, datasets, merged checkpoints | ask and stop launches, run settings, registry | everything |
| Retention | none | none | **blob deletes**: no client role deletes blobs | everything |

Sandbox operations on an HTTP pool should need a token that names the lease's key, so that only the run holding a
lease can operate its sandbox.

**Presigned blob uploads.**

- The service presigns a PUT for the key `sha256/HASH`, and requires the upload's checksum header
  (`x-amz-checksum-sha256`) to equal *HASH*. Otherwise a client can store bytes under a name they do not hash to, and
  readers that link without verifying spread them.
- "The blob exists, skip the upload" must refresh the blob's reference time, as in the retention fix.
- Retention runs inside the service, with the grace period. It never deletes a blob with an unexpired presigned
  upload, or with an upload intent recorded in the last *G*.
- Presigned URLs expire, so a large checkpoint needs a multipart upload, and the record must be appended only after
  the upload completed.

### What the service must guarantee, in one list

1. **Per scope, linearizable fences.** `take` returns a number higher than any before. An append is checked against
   the newest fence and inserted in one transaction under the scope's lock. A take answered to the client is durable.
2. **Unique keys, first append wins, atomically.** The answer says whether this request wrote the record, and returns
   the stored record when it did not.
3. **Idempotency by request id** for `take`, `append`, synchronous claims, launch transitions and beats. A retry with
   the same id gets the first answer.
4. **Durability before acknowledgement.** Commit, with `fsync`, before answering.
5. **Linearizable reads** for every read that decides something (names before deletion, claims, steps, fences, beats),
   or session tokens that give read-your-writes and monotonic reads, with primary-only reads where several clients'
   writes must be seen.
6. **Server time** for beats, staleness, claim lapse and lease expiry.
7. **Compare-and-set** for the mutable tables' state machines (launches, leases), not overwrites.
8. **Streams** in commit order, at least once, resumable, with a full read as fallback.
9. **Authorization** that ties tokens to roles, fence scopes to tables, and a claim's or lease's owner to the token.
10. **Blob deletion only in the service**, with a grace period, and checksummed uploads.

What the client must do:

- send a request id per logical operation, and retry with it until it gets a definite answer;
- never read a timeout as a refusal;
- never retry `Fenced`;
- keep its deadlines under `STALE`;
- on an unknown outcome of a claim, read back by request id rather than move on.

## Unverified

- `flock` across machines (NFS) and on WSL's `/mnt/c`, for `FileLedger`.
- What a power loss does to a `FileLedger` without `fsync` (whether a fence can go back).
- Whether a trainer writes its working files in place after `kept` has hard-linked them, and whether any deployment
  runs trainers as root (which would let such a write reach the blob).
- Pools of one name in two processes with a real provider (Minecraft worlds).
- The durable runner items marked unverified above: teardown on DBOS exceptions, eviction ordering, unloading
  mid-effect, the liveness items. They come from reading the code and DBOS 3.1.0's sources, and were not run.
- Transaction-mode connection poolers in front of Postgres.
