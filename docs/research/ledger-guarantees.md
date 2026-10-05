# Ledger guarantees

**Status: built.** Each guarantee is held by a test. The last section, the HTTP ledger service, is proposed. A design
note: see [Design notes](README.md) for the others.

Code: `rollout_train.ledger`, `rollout_train.database`, `rollout_train.rollouts.scheduler`, `rollout_train.sandboxes`,
`rollout.harness.sandboxes`, `rollout_train.checkpoints`, `rollout_train.loop` · Tests:
`tests/research/test_ledger_guarantees.py` · See
[checkpoints](../libraries/rollout-train/checkpoints.md#the-ledger), [rollouts](../libraries/rollout-train/rollouts.md),
[training](../libraries/rollout-train/training.md), [sandboxes](../libraries/rollout/sandboxes.md)

Processes on one machine or many cooperate through three kinds of shared state:

- a ledger: append-only tables and fences;
- ordinary mutable tables beside it: presence (beats), launches, the registry, run settings and sandbox leases;
- a content-addressed blob store.

This page checks, guarantee by guarantee, what the code promises and whether it keeps the promise. Each section gives
the claim, the mechanism, the assumptions it rests on, and the verdict, with file and line evidence. Where the
guarantee fails, it gives a scenario and a proposed fix. The last section says what the HTTP ledger service, which
every role is to reach the ledger through ([runtime design](runtime-design.md#decisions-after-review)), must guarantee
for all of this to carry over.

The analysis is of `main` at `6ec3721`, and line numbers are of that commit. Findings 1, 2, 3, 4, 5, 7, 8, 10, 11, 12,
13 and 14 have since been fixed, and finding 9 in part: their sections say what the code does now, with line
numbers only where they say what the code did then.

**Evidence.** Each violation still open has a test marked `xfail(strict=True)`: the suite stays green, and a fix turns
the test into a failure until its mark is removed. A fixed violation's test passes, unmarked. Tests that pass show a
guarantee holding under real concurrency:

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

| # | Kind | Finding | Status | Test |
|---|---|---|---|---|
| 1 | Safety | A runner whose claim lapsed (it paused) still records its episode. The keeper deleted its sandbox when the claim lapsed, so its run fails, and that failure, made by the platform, becomes the episode's outcome and drops the newer attempt's result | fixed: a fence per episode, and a claim holds only as its episode's latest attempt ([2](#the-fence-per-episode)) | `test_a_runner_whose_claim_lapsed_does_not_record_the_episode` |
| 2 | Safety | Two claims of one episode hold at once: a lapsed claim holds again when its runner beats again, and an adoption races a new attempt | fixed: as 1 ([2](#the-fence-per-episode)) | `test_a_lapsed_claim_does_not_hold_again_beside_a_newer_attempt`, `test_an_adoption_and_a_new_attempt_never_both_hold` |
| 3 | Safety | Retention can delete a blob that a checkpoint added at the same moment names (content addressing finds it stored and does not write it again) | fixed: a grace period, and a put that finds a blob sets its time ([6](#6-retention-and-blobs)) | `test_thin_never_deletes_a_blob_a_concurrent_add_names` |
| 4 | Safety | `FileLedger`: after a writer dies mid-line, the next acknowledged append is lost, and its key then accepts a second, different record | fixed: the next append removes the unfinished line first, and appends are on disk before they are acknowledged | `test_an_append_after_a_torn_line_is_kept` |
| 5 | Safety | Staleness and wall-time limits compare wall clocks. A runner whose clock is 90 s behind looks dead while it plays (finding 1, everywhere at once), and a clock stepped forward ends sandbox leases early | fixed: beats stamped and aged by the store's clock, time limits on the pool's monotonic clock ([3](#3-clocks)) | `test_a_live_runner_whose_clock_is_behind_keeps_its_claims`, `test_a_clock_stepped_forward_does_not_end_a_lease_early` |
| 7 | Safety | Launch states are overwritten unconditionally: a stop asked for while a run starts is lost, and a stop racing a claim marks a running launch `stopped` | fixed: every state change compares and sets ([7](#7-launches-and-settings)) | `test_a_stop_asked_for_while_a_job_is_made_stops_the_job`, `test_a_stop_racing_a_jobs_start_never_leaves_a_running_launch_stopped` |
| 8 | Safety | Two makers of one suite leave a suite that neither made | fixed: a suite is one record ([8](#8-suites-finding-8)) | `test_two_makers_of_one_suite_leave_one_of_their_suites` |
| 9 | Safety | Two loops of one run (a replaced one that has not noticed yet) share unfenced side effects: the run's directory, the trainer, the bookmark | fixed in part: the loop looks at its fence before acting outside the ledger, and trains in a directory of its own ([5](#5-the-training-loop)) | `test_a_loop_trains_in_a_directory_of_its_own_and_once_replaced_deletes_and_bookmarks_nothing` |
| 10 | Liveness | A runner that dies between taking its fence and adopting (or whose take is applied twice) loses all the runs it would adopt on its next start | fixed: adoption under any earlier fence ([2](#adoption-adopts-under-any-earlier-fence-finding-10)) | `test_a_runner_that_died_while_preparing_adopts_its_runs_when_started_again` |
| 11 | Liveness | `FileLedger`: a crash while taking a fence leaves `fences.json` empty, and the ledger is unusable until repaired by hand | fixed: `fences.json` is replaced whole | `test_a_crash_while_taking_a_fence_leaves_the_ledger_usable` |
| 12 | Liveness | `FileLedger` blocks the event loop on its lock and rereads whole tables on every append, which can delay beats | fixed: it waits in a thread, and keeps each table's keys | `test_a_ledger_of_files_does_not_hold_up_the_event_loop_while_another_holds_its_lock` |
| 13 | Liveness | The keeper decides from a read and releases afterwards: a lease whose claim was adopted in between is released, and its run is played again | fixed: the keeper reads the claim again and ends it in the ledger before releasing ([4](#4-leases-and-sandboxes)) | `test_the_keeper_reads_a_lapsed_claim_again_and_ends_it_in_the_ledger_before_releasing` |
| 14 | Documentation | "A table's records … in the order they were appended" does not hold on Postgres for tables that several scopes write | fixed: an append holds its table's lock while it numbers its record | `test_appends_of_two_scopes_to_one_table_get_distinct_positions_on_postgres` |

Tests of what the fixes added are in `tests/rollout_train/test_episode_fences.py`,
`tests/rollout_train/test_presence.py`, `tests/rollout/harness/test_sandboxes.py`, `tests/rollout/harness/test_blobs.py`
and `tests/rollout_train/training/test_loop.py`.

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
- **Durable against crashes: holds.**
  - **A torn line (finding 4, fixed).** A torn line is what a writer killed between two `write` system calls of one
    record (a record larger than the 8 KiB buffer) leaves, or a full disk. Readers skip a line that does not parse:
    it was never acknowledged. Under the lock, before it writes, an append reads its table's file as far as it grew
    since it last looked. An unfinished last line is cut off (`ftruncate` to the last newline), or, if it holds a
    whole record whose newline was lost, ended with a newline (readers had already counted it). The new line is
    therefore never glued to torn bytes, and the torn record's key stays free only because that record was never
    appended. An append that fails partway (a full disk) cuts off what it wrote before raising.
    - Tests: `test_an_append_after_a_torn_line_is_kept`; in `tests/rollout_train/test_ledger.py`, a file cut short
      mid-line with `os.truncate` and a record whose newline was cut off.
  - **A crash while taking a fence (finding 11, fixed).** `fences.json` is written beside itself to a temporary file,
    put on disk (`fsync`), renamed over the old one (`os.replace`), and the directory is put on disk. A crash at any
    point leaves either the old fences or the new. `test_a_crash_while_taking_a_fence_leaves_the_ledger_usable` kills
    the take after the new fences are written and before the rename: the old fences stand and the ledger goes on.
  - **On disk before acknowledged.** An append calls `fsync` on its table's file before it returns True, and on the
    directory when it made the file. A take is on disk before it returns its number, so a power loss cannot lose a take
    while keeping an append made under the fence it gave: the fence never goes back.
- **Not blocking (finding 12, fixed).** Every operation waits for the lock and does its file work in a thread
  (`asyncio.to_thread`), so a beat is not held up behind another process's append. Each process keeps, per table,
  which file it read (device and inode), how far, and where each key's line begins. An append reads only what other
  processes appended since, or the whole file if it is another file or shorter. A read still reads the whole table.

### `DatabaseLedger` on SQLite: holds

- **Mechanism.** Every write transaction begins with `BEGIN IMMEDIATE` (`rollout_train/sql.py`, `_begin_immediately`),
  which takes SQLite's single write lock. `take` is an upsert that returns the new number (`database.py:122-132`).
  `append` reads the newest fence and inserts in the same transaction (`database.py:134-152`). Writes are therefore
  serial. Reads run outside a transaction and see the latest commit (`Database.read`).
- **Evidence.** In `test_writers_on_connections_of_their_own_take_distinct_fences_and_stale_appends_are_refused[sqlite]`,
  four threads, each with an engine of its own, take and append 30 times each. Every take gets a number of its own,
  and fences never go down in append order.
- **Assumption.** A local filesystem. WAL mode does not work over a network filesystem.

### `DatabaseLedger` on Postgres, READ COMMITTED: holds

- **Mechanism.**
  - Both `take` and `append` run `SELECT pg_advisory_xact_lock(hash("ledger:" + scope))` first
    (`Database.write` in `rollout_train/sql.py`). The lock is held until the transaction commits.
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
  - The ledger takes only transaction-scoped advisory locks, which work directly and behind a pooler in either
    session or transaction mode, such as pgbouncer.
  - Two scope names whose 64-bit hashes collide only serialize more. That is harmless.

### First append wins: holds, atomically

- `FileLedger`: the key check and the write happen under the directory lock (`ledger.py:90-96`).
- `DatabaseLedger`: the table's primary key is `(name, key)`, and the insert is `ON CONFLICT (name, key) DO NOTHING`
  (`database.py:40-48,143-150`). This is atomic on both databases, whatever the locks.
- **What the loser learns.** `append` answers only whether it wrote. `append_returning` answers `Appended(wrote,
  record)`, the record the table holds under the key, from the append itself: `DatabaseLedger` reads the row in the
  same transaction, `FileLedger` reads the key's line by its offset.
  - `Checkpoints.add` reads back, and so does a bridge (`bridges.bridged`), which returns what the first bridging made.
  - `make_suite` uses `append_returning` and plays the winner's suite (finding 8, fixed); `edit_suite` uses it to find its
    number taken, and tries the next.
  - `EpisodeRunner._ended` does not read back. That is intended: the loser's episode is dropped. Most losers are
    refused before that: an older attempt's record raises `Fenced` once a newer attempt took the episode's fence.

### Order of records: holds (finding 14, fixed)

- **Claim.** `read` returns a table's records by key, in the order their appends took effect, whichever scopes made
  them.
- **Mechanism.** Each insert computes `position` as `MAX(position) + 1` for its table. On Postgres, the append takes a
  transaction lock on its table (`pg_advisory_xact_lock(hashtextextended('table:NAME', 0))`) after its scope's lock
  and before the insert, and holds it until it commits. The next append to the table waits for that commit, and its
  insert statement's fresh snapshot (READ COMMITTED) sees it, so positions are unique and in commit order. SQLite runs
  one write at a time. An index on `(name, position)` makes the `MAX` and the ordered read cheap; a database made before
  it gets it when a `DatabaseLedger` opens it.
- **Locks.** Every append takes its scope's lock, then its table's; a take, only its scope's. No transaction takes a
  table's lock and then a scope's, so the two cannot deadlock. Appends to one table, from any scopes, run one at a time.
- **Readers that depend on order.** `checkpoints_in` (oldest first), claim and attempt order, and the monitor's lists
  read tables in position order, which is now commit order on every database. Positions written before (on Postgres,
  possibly tied) read in `(position, key)` order.
- **Evidence.**
  - `test_appends_of_two_scopes_to_one_table_get_distinct_positions_on_postgres` runs the second append while the
    first transaction is open: the second waits, and the positions are `1` for the first to commit and `2` for the
    second.
  - `test_writers_of_scopes_of_their_own_appending_to_one_table_take_positions_one_after_another`: four threads, each
    with a connection pool and a scope of its own, append 25 records each to one table, on SQLite and on Postgres. The
    positions are 1 to 100, and each writer's records read in the order it appended them.

## 2. Claims and episodes

**Claims** (rollouts.md "What runners write"):

- "two runners never play one attempt";
- "A claim holds while it is its episode's latest attempt and its runner keeps its fence and beats";
- "The first record of an episode is its record", and only the attempt that holds the episode's fence can append it;
- at-least-once play: an episode with no record and no claim that holds is open again.

**Mechanism.**

- A claim is an append to `runs/RUN/claims` under `GROUP/EPISODE/ATTEMPT`, under the runner's fence `runners/NAME`.
  The runner whose append succeeded then takes the episode's fence, `runs/RUN/episodes/GROUP/EPISODE`
  ([the fence per episode](#the-fence-per-episode)).
- `Claims.holds` says a claim holds if all of these are true:
  - it is its episode's latest attempt (no claim of a higher attempt exists);
  - it is not noted in `interrupted`;
  - it was made under its runner's newest fence, or adopted under it;
  - its runner's newest beat is no older than `STALE` = 90 s by the store's clock ([clocks](#3-clocks)), unless the
    reader is that runner.
- `open` reads the fences, the beats and the run's tables, one after another and not as one snapshot. It offers an
  episode's next attempt when no claim of it holds.
- The episode's record is an append under `GROUP/EPISODE`, under the episode's fence.

**What holds.**

- One attempt, one runner: the claim key is unique, and first append wins.
- One record per episode: the record key is unique, and first append wins.
- Training cannot mix two attempts within one episode. A record names one run's trajectory blob
  (`episodes.py:127-143`), and the recorder's segments are kept by `run_id` (`scheduler.py:474`).
- At-least-once play holds as long as some runner with room keeps serving.

### A zombie's record (finding 1): refused

At `6ec3721` the fence that guarded the record was the runner's own, and it does not move when the runner's claim
lapses for want of beats. A runner that was paused (a stop-the-world pause, a suspended VM, a network partition) and
resumed therefore recorded its episode, although by the scheduler's own rule its claim no longer held:

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
pool. Now the episode's fence refuses step 4 twice over: the keeper took it before it deleted Z's sandbox (step 2), and
F took it when it claimed attempt 2 (step 3). Z's record raises `Fenced`; Z notes its attempt cut short (`SUPERSEDED`),
and F's completed episode is the record. Without a pool, a paused runner whose episode nobody claimed again holds its
claim again when it beats again, and its record stands: its run was not harmed, and it is the only attempt.

### Two claims of one episode (finding 2): never both hold

At `6ec3721` `holds` never asked whether a later attempt existed. There were two ways for two claims of one episode
to hold at once:

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

The consequences were that the episode was played twice, and one pool place used twice. Now `holds` asks for the
latest attempt, so at most one claim of an episode holds at any moment: once attempt 2's claim is appended, attempt 1's
holds no more, whatever its runner's beats and adoptions say. Both tests pass. In the adoption race, the runner started
again takes the episode's fence before it reads the claims a second time: attempt 2 claimed before that take is seen,
and the claim is not adopted (it is cut short as `LAPSED`); attempt 2 claimed after it takes the fence in turn, and
the adoption, or the adopted run's record, is refused.

### The fence per episode

Each episode has a scope of its own, `runs/RUN/episodes/GROUP/EPISODE` (`episode_scope`).

| Who | Takes it | Appends under it |
|---|---|---|
| The runner whose claim was appended (`_claim`) | right after the claim; only the winner of an attempt takes it | the episode's record (`_ended`) |
| A runner started again (`_adopt`), for each claim it adopts | before it reads its claims a second time | the adoption, then the record |
| A pool's keeper (`ending`), for a lapsed claim that is its episode's latest attempt | after reading the claim again, before releasing its lease | the note that the attempt was cut short (`RELEASED`) |

- **The claim stays under the runner's fence, and the attempt number is the claim's.** Claimers take the episode's
  fence only after their claim was appended. Were the fence taken first, a runner that lost the race for an attempt
  (runners look at the same open episodes, oldest first) would take the fence after the winner's claim and shut the
  winner out; numbering attempts by the fence instead would let every late claimer pre-empt the claim before it. The
  claim's key decides who plays an attempt (first append wins), and the fence decides whose writes about the episode
  count (the last taker's).
- **What closes findings 1 and 2.** "Latest attempt" in `holds` means at most one claim of an episode holds at a
  time. The fence means an older attempt can write nothing that counts once a newer one, or a keeper, has taken it:
  its record and its adoption raise `Fenced`. A runner whose record was refused notes its attempt cut short under its
  own fence (`SUPERSEDED`), so its claim holds no more; that matters when the taker was a keeper racing a new claim,
  which leaves the new claim the latest attempt but unable to record.
- **Interrupts stay under the runner's fence.** Noting one's own attempt cut short only ever ends one's own claim, so
  it needs no fence of the episode, and it then succeeds for every live runner.
- **Records from before the change.** Claims, records and adoptions written before episodes had fences read as before:
  a claim holds by the same rule, a record under a runner's fence counts, and a runner started again takes the
  episode's fence to adopt an old claim like any other. Nothing is migrated.
- **Cost.** One more `take` per claim (an upsert in one transaction: SQLite's single write lock, or one Postgres row
  and a transaction-scoped advisory lock), and one fence row per episode claimed. `fences()` returns every row, so
  each look of each runner, each keeper sweep and each loop's check of its own fence reads one more row per episode
  the ledger has claimed; at hundreds of thousands of episodes, a filter by prefix on `fences()` would be worth
  adding. The monitor leaves episode fences out of the fences it lists.
- **What is left.** An attempt claimed between the keeper's read and its take loses its fence to the keeper: its
  record is refused, and it is played again (liveness). A claimer that dies between its claim and its take leaves a
  claim with no fence taken; that claim holds while its runner beats, and lapses when the runner is started again
  A zombie can still record in the moment between a new claim's append and its
  take, if nobody (no keeper) took the fence before.

### Adoption adopts under any earlier fence (finding 10)

At `6ec3721` `_adopt` judged a claim held if it was made under `self._fence.number - 1`, or adopted under it
(`scheduler.py:406,427`). A runner that took its fence and died before adopting (an out-of-memory kill, a crash in
`prepare`), or whose take was applied twice (a retried request), found its claims two fences back, noted them all
`LAPSED` and cancelled their runs.

Now `_adopt` adopts a run of any claim of its own (made under any earlier fence of its name) that is its episode's
latest attempt, not cut short, of an episode without a record, as long as the episode's fence it takes is still its
when it appends the adoption. Fences of one name only grow, and one incarnation runs at a time, so "earlier" is enough.
`test_a_runner_that_died_while_preparing_adopts_its_runs_when_started_again` passes.

### `holds()` and time of check against time of use

- `open` and `holding` read fences, beats and tables at different moments. A stale read can only make a claim look
  lapsed later, or held later, than it is. Duplicates that follow are absorbed by first append wins, and by the
  episode's fence, which refuses an older attempt's record once a newer attempt took it.
- `admits` decides, and then `acquire` creates the sandbox (`harness/sandboxes.py:319-347`). A claim that lapses in
  between keeps its sandbox until the keeper's two looks. That is liveness only.

## 3. Clocks

At `6ec3721` every timestamp below was `time.time()` on the machine that wrote it.

| Decision | Writer's clock | Reader's clock | Effect of skew or a step | Kind |
|---|---|---|---|---|
| A runner is alive: beat `at` ≥ now − 90 s | runner (`presence.py:84`, `database.py:342`) | whoever judges: other runners, every pool's `admits` and keeper (`presence.py:55-57`) | Writer 90 s behind, or reader stepped 90 s ahead: a live runner's claims lapse. Others play them again, keepers delete its sandboxes, and finding 1 follows for every episode it plays. Writer ahead: a dead runner's claims hold longer | **Safety** (two holders, wrong outcomes). **Fixed**: the store's clock |
| "Two sweeps in a row" | keeper (`sandboxes.py:212-221`) | keeper | none: two looks, counted, with no times compared | — |
| Sandbox wall-time limit: `Lease.ends` ≤ now | pool, at acquire (`harness/sandboxes.py:336,345`) | same pool, at sweep (`harness/sandboxes.py:387,392`) | A step forward ends leases early and deletes sandboxes in use. A step back lets sandboxes overstay | **Safety** (a sandbox deleted under its run). **Fixed**: a duration on the pool's monotonic clock |
| Launch `at` (ordering of launches asked for) | whoever asked | the monitor | Launches listed out of order | Liveness |
| Run state shown by the monitor (beat ages) | runner | monitor | Wrong display | Display. **Fixed**: the store's clock |

WSL2 is prone to both problems: its clock drifts while the host sleeps, and is stepped when it resyncs.

**Staleness by the store's clock.** Two designs were open: database time, or each reader's monotonic observation of a
beat changing. The code uses the store's clock:

- The store stamps a beat when it keeps it (`Beat.at`) and says how old it is when it is read (`Beat.age`), by the
  same clock; `alive(beat)` is `beat.age <= STALE`. The writer's clock is never read.
  - `DatabasePresence` on Postgres stamps and ages in SQL with the server's `clock_timestamp()`: one clock for every
    machine, so skew between runners and readers cannot make a live runner look dead.
  - On SQLite it uses SQLite's own `julianday('now')`, and `FilePresence` the machine's clock: both serve one machine,
    whose clock every writer and reader shares.
- Everything that judges liveness reads the age: `Claims.holds` (so runners' `open`, pools' `admits` and keepers'
  sweeps) and the monitor's run state and machines.
- Why not observation: a reader that starts fresh would have to wait 90 s before it could judge anyone dead, and every
  reader (each runner, each pool, the monitor) would judge separately, so two readers could disagree on whether one
  claim holds. A clock in the store gives every reader one answer at once, and is the form an HTTP service would
  have.
- What is left: a step of the store's own clock (the Postgres server's, or the one machine's for SQLite and files)
  ages every beat at once until each runner beats again (every 15 s). Claims of live runners can then look lapsed for
  that long; with the episode's fence, the cost is attempts played again, not wrong records.

**Time limits on the pool's monotonic clock.** A lease keeps its limit as a duration (`Lease.seconds`, from
`Lease.at`), and `SandboxPool` keeps each lease's end on its own process's `time.monotonic()`. A pool started again
credits each lease with the time it has lasted, by the wall clock, once, as it reads its leases.

Tests:

- `test_a_live_runner_whose_clock_is_behind_keeps_its_claims` beats from a writer whose clock is two minutes behind,
  into SQLite and into Postgres. The claim holds. (A ledger of files has no other machine's clock to be behind.)
- `test_a_database_stamps_and_ages_beats_by_its_own_clock_not_the_writers` beats from clocks an hour ahead and an hour
  behind.
- `test_a_clock_stepped_forward_does_not_end_a_lease_early` steps the pool's wall clock forward two hours. The
  sandbox, acquired seconds before with an hour's limit, stays.
- `test_a_pool_started_again_counts_the_time_its_leases_have_lasted`.

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

The keeper (`keep`) releases leases found lapsed at two looks 15 s apart, after reading each claim again and ending it
in the ledger (`ending`, below). `sweep` deletes sandboxes the provider has that no lease or making names.

**What holds.**

- One sandbox per key, within one pool process: the per-key lock serializes the lookup and the make, and the handle
  is fixed by the key.
- The keeper cannot release a lease acquired after its read. `sweep` releases only keys it found lapsed twice
  (`sandboxes.py:217-219`), and `_making` and `_made` protect sandboxes being made.
- HTTP pools map `NoCapacity`, `LeaseRefused` and `SandboxLost` to 503, 409 and 410, and back
  (`harness/remote.py:151-157,231-233`). Acquire and release are idempotent by key, so a client may retry either.

**What does not.**

- **A rightful holder can lose its sandbox.** The rule is "a lease ends with its claim", so a runner whose claim
  lapsed is by definition not the rightful holder. The claim lapses on the store's clock (section 3) or a pause, and
  the runner carries on; what it records then is refused, because the keeper ends the claim in the ledger before it
  ends the lease (below, and finding 1).
- **A pool name used by two processes at once.** Pools are named `KIND@HOST/DIRECTORY` (`profile.py:396`), so two
  processes of one run's directory on one host (finding 9) load the same leases and keep separate memories of them.
  Each process's `sweep` deletes sandboxes "no lease names" by its own memory, and with a provider whose `held()`
  lists sandboxes on the machine rather than in the process (Minecraft worlds), one process would delete the other's.
  Now each keeper takes the pool's fence (`pools/NAME`) when it starts, looks at it before each sweep, and stops
  sweeping once another process took it (`test_a_keeper_stops_sweeping_once_another_process_keeps_its_pool`). Left:
  a sweep already under way when the other process starts runs to its end; the replaced process still acquires and
  releases for its own runs (which its replaced runner no longer starts).
- **Leaks.** A lease of a run the ledger does not know ends only when released (documented). The leases of a pool name
  that is never opened again (a host renamed, a directory moved) are never swept. Neither are their sandboxes, for a
  provider whose sandboxes outlive the process (a cloud API). A fix would be a sweep, by any keeper, of the leases of
  pools that have not beaten for a long time; pools opened by a profile do not beat today, so it needs that first.
- **The keeper reads the claim again and ends it before releasing (finding 13).** At `6ec3721` `ended` read the
  ledger and `sweep` released afterwards, so an adoption appended in between made the claim hold again while the
  release still happened. Now, for each lease found lapsed at two looks, `ending` reads the claim again just before
  releasing: one that holds again keeps its lease. One that is still its episode's latest attempt is ended in the
  ledger first, by a compare-and-set made of the ledger's own operations: the keeper takes the episode's fence and
  appends the `interrupted` note (`RELEASED`) under it. An adoption appended before the take is seen by the second
  read; an adopter that takes the episode's fence after the keeper's take finds the note when it reads the claims
  again, and does not adopt; one that takes it between the keeper's take and its note gets its adoption, and the
  keeper's note is refused, so the lease stays (and is looked at again). The release follows only a note that stands.
  Test: `test_the_keeper_reads_a_lapsed_claim_again_and_ends_it_in_the_ledger_before_releasing`.
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

**Two loops of one run (finding 9), as at `6ec3721`.** The fence stops only ledger appends. A replaced loop that has
not yet tried to append keeps doing everything else:

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
**What the loop does now.**

- **It looks at its fence before each side effect outside the ledger** (`newest(ledger, fence)`, which raises
  `Fenced`): before it publishes a checkpoint, before `serve()` deletes directories, before it renames a step's
  directory, and before it moves a bookmark (`made`).
- **Its trainer writes into a directory of the loop's own**, `DIRECTORY/making/FENCE/MAKES`. Once the checkpoint is
  appended, the directory is renamed to `DIRECTORY/MAKES`, where the checkpoint's files are looked for. The checkpoint
  is then the loop's own: another loop's add of it under an older fence is refused once this loop took its fence, and
  it was not there when the step began. A loop replaced while its trainer ran writes only under its own fence's
  directory, which its replacement's `serve()` deletes.
- **Linked-out files are checked.** Nothing writes a kept file in place now: a trainer writes into a fresh
  directory, and a kept file is read-only. `FileBlobStore.link` checks a blob before linking it out: its size always,
  and its hash too up to 64 MiB (`CHECKED`), and raises for a blob that does not match
  (`test_a_blob_changed_in_place_is_not_put_in_place`).
- Test: `test_a_loop_trains_in_a_directory_of_its_own_and_once_replaced_deletes_and_bookmarks_nothing`.

**What is left.** A look and the action after it are two steps, so a loop replaced between them still acts once: it
may delete what its replacement just fetched, publish once, or move a bookmark back once. Registry writes are not
fenced. A blob larger than 64 MiB changed in place without changing its size is linked out unnoticed; reading it
through `FileBlobStore.read` still verifies its hash.

## 6. Retention and blobs

**Claim.** A release is appended to the ledger before its blobs are deleted, and a blob is deleted only if nothing
names it and nothing put it in the last `Retention.grace` seconds, so a thinning may be repeated after a crash at any
point, and a checkpoint added at the same moment keeps its blobs (checkpoints.md).

**Mechanism.** `thin` appends releases, reads every checkpoint and what every bridge made
(`checkpoints/resharded`), collects the hashes that unreleased checkpoints' weights and state, and what bridges made of
them, name, and deletes the released checkpoints' other blobs with `delete(reference, unused_for=grace)`. Every put
sets the blob's time to now, whether it writes the blob or finds it.

**Crash safety: holds.** A release is recorded before any deletion, and every step is repeatable.

**Against a concurrent `add`: holds, given the grace (finding 3, fixed).** The race was:

1. `thin` (run X) reads the names. Blob *B* is named only by a checkpoint it just released.
2. `add` (run Y, or a merge, imitation, or a replaced loop) finds *B* stored, writes nothing, and appends a checkpoint
   naming it.
3. `thin` deletes *B*.

A file whose bytes recur across checkpoints is what makes it happen: a configuration file, a tokenizer, or a
checkpoint forked from the one being thinned. Now the put in step 2 sets *B*'s time, so step 3 spares it.

- **Why it holds.** Let `thin` read the names at *t₃* and delete at *t₄*, and let `add` put *B* at *t₂* and append at
  *t₅*. If *t₅* < *t₃*, `thin` sees the name. Otherwise *t₄* − *t₂* < (*t₄* − *t₃*) + (*t₅* − *t₂*): when a thinning's
  span from reading to its last delete, and an add's from its first put to its append, are together shorter than the
  grace (an hour by default), *B* is younger than the grace when `thin` looks, and is spared. A blob spared is deleted
  by a later thinning of any run, which looks at every released checkpoint's blobs again.
- **Why a grace period, not a lock.** One lock spanning an add's puts and its append, and a thinning's read and its
  deletes, would have to live where every writer of a blob store reaches: a file lock serves one machine, a Postgres
  advisory lock would be held by a client for a whole upload of a checkpoint to S3, and an HTTP ledger would need a
  lease of its own. A blob's time lives with the blob, in every store: a file's modification time, an object's
  `LastModified`. It needs nothing beside the store, and it is what a service that alone deletes blobs would keep
  ([HTTP](#10-the-http-ledger-service)).
- **A store of files.** A put that finds a blob's file sets its modification time (`utime`); one that cannot (another
  user's file) writes it again as its own. A file kept by linking (`put_file`) is one inode with the trainer's working
  file, so the put sets the working file's time too; a link made new is set to now as well, since the working file
  keeps the time it was written at. To delete, `thin` looks at the time, moves the file aside (`rename`, atomic) and
  looks again there: a put that found the file in between set the inode's time (the file is linked back, unless a put
  wrote it again meanwhile), and a put after the move finds no file and writes it. Deleting a blob unlinks it from
  the store only: a working copy linked to it keeps its bytes, and a later `put_file` of it links it back.
  - `test_thin_never_deletes_a_blob_a_concurrent_add_names` (on a ledger of files and on SQLite) ages every blob by two
    hours, then runs another run's `add` just before `thin`'s first delete: the shared blob is kept, and the blob only
    the released checkpoint named is deleted.
  - `test_a_put_that_finds_a_blob_as_it_is_being_deleted_keeps_it` runs a put between the look and the move.
- **An S3 store.** A put that finds an object copies it onto itself (`CopyObject` with `MetadataDirective=REPLACE`),
  which sets its `LastModified`, unless it was set in the last minute (`refresh_after`: one copy a minute of a blob put
  often). Ages are judged by the service's clock, the `Date` of its answer. To delete, `thin` asks the object's time
  (`HeadObject`), then deletes it: a put that finds the object between those two requests, and whose add appends before
  that delete, is not seen. The window is one request long; S3's conditional delete does not apply, since copying an
  object onto itself keeps its ETag.

**What is still open.** `keeping()` reads the runs' starts and the bookmarks before `thin` (in `loop.py`). A run that
starts from a checkpoint after that read, or a bookmark set after it, does not keep the checkpoint: it is released,
reads with no weights, and the run that starts from it fails loudly ("was released") when it fetches its files. A
fix needs `thin` to read what keeps checkpoints after appending its releases, and a way to take a release back.

**Do datasets, episodes and bridges' files count as names?**

- What bridges made of checkpoints not released counts: a bridge (not `verbatim`) can write a file that has the bytes
  of a released checkpoint's file, a configuration file say.
- Datasets', episodes' and batches' blobs are never deleted, and do not count: only checkpoints' files are deleted.
  One of them would be lost only if a released checkpoint had a file of exactly its bytes.
- A `verbatim` bridge of a released checkpoint names its weights' blobs; they go with the checkpoint, whose bridged
  files are of no use once it is released.

## 7. Launches and settings

**Mechanism.** `note(id, expect=STATES, **changes)` compares and sets in one step of the store: under the directory
lock for `FileLaunches`, and in one transaction under the `launches` lock for `DatabaseLaunches`, whose `UPDATE` also
says `WHERE state = <the state read>`. The changes are written only if the launch is in a state of `expect` (any, if
none is given) and may go to the state they name. Either way it returns the launch as it then is, and the writer
compares the state it gets with the state it asked for. The moves are written down once (`launches.MOVES`):

- `asked → submitted | running | failed | stopped`;
- `submitted → running | stopping | stopped | ended | failed`;
- `running → stopping | stopped | ended | failed`;
- `stopping → stopped | ended | failed`.

A finished launch goes nowhere, and nothing goes back. `asked → running` is a job's driver that starts before its
submitter has noted the job; `submitted → ended | stopped` and `running → stopped` are a job that ended, or was stopped
from outside, before its driver said so.

- **The submitter** (`submitting.start`). It notes `submitted` and the job expecting `asked`. Refused because a stop
  came while the job was made, it stops the job and notes the job without changing the state; refused because the
  driver already noted `running`, it notes the job alone.
- **The driver** (`jobs.driven`). It notes `running` expecting `asked` or `submitted`; refused because the launch is
  `stopping`, it notes `stopped` and runs nothing. On the way out it notes `ended`, `failed` or `stopped` expecting a
  launch that is going.
- **The monitor and the command line** (`submitting.stopped`). A stop of a launch read as `asked` notes `stopped`
  expecting `asked`; refused (its job was made meanwhile), it notes `stopping` expecting `submitted`, `running` or
  `stopping`, and stops the job.
- Tests: `test_a_stop_asked_for_while_a_job_is_made_stops_the_job` and
  `test_a_stop_racing_a_jobs_start_never_leaves_a_running_launch_stopped` run the races with the real submitter and
  stop; `tests/rollout_train/test_launches.py` checks the moves on both stores, and concurrent stops and starts.

**Run settings: hold.** `want` merges under a lock (`database.py:380-392`). The loop reads the settings between steps
only, and records what each step used. A change written mid-step applies to the next step, as documented.

**The registry.** Names are checked and written under one lock (`database.py:212-223`). Bookmarks are last writer
wins (see finding 9).

## 8. Suites (finding 8)

**Claim.** Each version of a suite is written once under the suite's fence (`suites/NAME`) and never changed, two
makers of one suite at once leave one of their suites whole, and two editors at once leave two whole versions with the
suite's name at the later. **This holds (finding 8, fixed).**

**Mechanism.** `make_suite` checks that the name is free, takes the fence, and appends the whole of version 1 as one
record (`evaluations/SUITE/suite`, key `suite`: its entries, each with its starts in order) with `append_returning`. A maker
whose fence was taken after its own raises `Fenced`; a maker whose append finds a record there returns the suite in that
record, and plays it. With one record, there is nothing for a second maker to finish: the suite in the ledger is one
maker's. `suite_for`, which makes an environment's eval data on first use, tries again after `Fenced` until it finds a
suite of the name. `edit_suite` appends its version under the next number with `append_returning`, under the same fence; an
editor fenced out, or whose number another took, tries the next number (or, told the version it edited, is refused). The
suite's name is ordinary state beside the ledger (the registry's suite names), moved only forward (`point_suite(…,
forward=True)`), so the later of two edits is where it points whichever writes last. A name that points nowhere is its
newest version in the ledger, so an edit that died between its append and the move is still read.

Tests: `test_two_makers_of_one_suite_leave_one_of_their_suites` (the second maker takes the fence after the first and
before its append: the first is fenced out, and the ledger holds the second's suite),
`test_a_maker_that_finds_the_suite_made_meanwhile_plays_that_one`, and in `tests/rollout_train/test_suite_versions.py`
`test_an_edit_makes_a_new_version_and_moves_the_suites_name_to_it` and
`test_a_suites_name_points_to_a_version_beside_the_ledger` (the name moves only forward, in both registries).

## 9. Runs and the ledger

Runs do not survive their process: the `LocalRunner` keeps a run's events in memory. Durability for runs comes from the
ledger. An episode stays open until its record is appended under the episode's fence, so when a run is lost with its
process, a runner started again, or another runner, plays the episode again as a new attempt
([2](#2-claims-and-episodes)), and the sandboxes leased under the lost claim end with it
([4](#4-leases-and-sandboxes)).

## 10. The HTTP ledger service

Every role in a cluster is to reach the ledger through `HttpLedger(url, token)`, which implements `Ledger`, `Presence`,
`Leases`, `Launches` and `DesiredSettings` against a service in front of the database
([runtime design](runtime-design.md#decisions-after-review)); neither is built yet. Every property above holds today
because each operation is one transaction on one database, answered over a connection that either commits or fails.
HTTP adds:

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
  breaks: `plans` and `starts` get gaps. (`_adopt` no longer reads the number: it adopts under any earlier fence.)
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
  compares the stored record with the one sent, and returns the stored record when they differ. The client side of
  that answer exists: `append_returning` returns `Appended(wrote, record)` (both ledgers answer it today; a caller
  that needs the winner, such as `make_suite`, uses it). `append` stays as it
  is, `wrote` alone, for every other caller.
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

- A stream needs a position in commit order. `position` is that, per table: it is assigned under the table's lock in
  the inserting transaction ([order of records](#order-of-records-holds-finding-14-fixed)). A sequence would not be:
  it is assigned at insert, so a row numbered 11 can commit before row 10, and a consumer resuming "after 11" never
  sees 10.
- Other ways, if appends to one table must not wait for each other:
  - Postgres logical decoding (commit LSN);
  - a resume rule that re-reads from the oldest position still in flight (`pg_snapshot_xmin`).
- Delivery is then at least once, resumable with an opaque token, and in order per table. Records are immutable and
  keyed, so consumers deduplicate by `(table, key)`.
- A consumer whose token is too old falls back to a full read. It is always safe, because tables only grow.
- Mutable tables (presence, leases, launches, settings) stream as "this row changed, at version *v*". Consumers re-read
  the row and treat the event as a hint.

**Presence and claims by server time.** `DatabasePresence` already stamps beats with the database's clock and ages them
by it ([clocks](#3-clocks)); a service in front of Postgres keeps that as it is.

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
| Monitor or operator | `suites/*`, `datasets/*`, `merges` | suites' versions, datasets, merged checkpoints | ask and stop launches, run settings, registry (suite names among it) | everything |
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
- Whether the filesystems a `FileLedger` or a store of files runs on keep what `fsync` put on disk across a power loss
  (WSL's virtual disk, network filesystems).
- Clocks for retention's grace: a store of files on a filesystem shared by machines compares a file's time, set by the
  machine that put it, with the thinning machine's clock. Skew much smaller than the grace (an hour) is assumed.
- Whether any deployment runs trainers as root. A trainer no longer shares a working directory with another loop, and
  nothing in the loop writes a kept file in place.
- Pools of one name in two processes with a real provider (Minecraft worlds).
- Transaction-mode connection poolers in front of Postgres.
