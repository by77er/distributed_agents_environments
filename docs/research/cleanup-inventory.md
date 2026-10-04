# Cleanup inventory

What to remove (leftovers of replaced mechanisms, paths the decided design drops, dead code) and what to factor out
(duplicated logic), taken at `0f90a2b`. Each item gives where it is, why it goes, what depends on it, what replaces it,
and roughly how many lines it removes. Items are ranked by payoff over risk within each section; the ranking across
sections is the first table. The last section orders the removals into reviewable commits, each with the tests that
guard it.

Paths are relative to the repository root, except that `RT/` stands for `libraries/rollout-train/src/rollout_train/`,
`R/` for `libraries/rollout/src/rollout/` and `WEB/` for `libraries/rollout-train/web/src/`.

The decisions this inventory assumes:
- Profiles are removed; what they named moves to a cluster config, the run's own settings plus saved presets, and
  environments' own declarations of sandboxes and tools.
- Ray is required.
- One gateway serves every backend; rollout-tinker joins the workspace with only the `tinker` SDK, and tinker-cookbook
  goes.
- Suites are explicit, named bundles; `suite_for` stops freezing an environment's eval data into one.
- Every run records through the gateway.

## Where it stands

The sections below describe the code at `0f90a2b`. Of the commits [section 6](#6-order-of-removal-commits) orders:

| # | State | Made by |
|---|---|---|
| 1–6, 9–13, 15, 17 | made | `dc4ce7a`, `016e3a6`, `4ebaf02`, `5d98c88`, `cac3e19`, `92a9332`, `f9e6b6d`, `ecac1bd`, `77b6519`, `a580c26`, `07d609d`, `30ce139`, `77956aa` |
| 7 | made but for `Control.spawn` and its Java route, which stay | `e8c4f62` |
| 8 | made but for `scripts/train-with-memory-log.sh`, which the Minecraft docs still use | `07f9e78` |
| 18 | in part: `@user_errors`, `--ledger` once, `registry.run_id`; no parent parsers or dispatch table | `ac758ae` |
| 19 | in part: the harness's variables and the hooks rule said once | `151e85b` |
| 21, 22 | made: rollout-tinker joined the workspace with the `tinker` SDK alone, and Ray is a dependency of rollout-train; rollout-verifiers stays a project of its own | `6fdda88`, `b54e4cf` |
| 14, 16, 20, 23–32 | open: the [runtime design](runtime-design.md#8-the-implementation-sequence) carries most of them (profiles, `hosting.py`, the process backend, the in-process bridge path, the file ledger) | |

The appendix's bug is fixed: an eval's start says its suite version as `suite_version` (`54d6827`).

## The fifteen items with the best payoff over risk

| # | Item | Section | Lines | Risk |
|---|---|---|---|---|
| 1 | Lineage readers for tables nothing writes, and their 4,064-line sample fixture | 1.1 | ~4,350 | low |
| 2 | Move test helpers out of `test_*.py` files into support modules (unblocks every removal below) | 3.7 | 0 net | low |
| 3 | Suite and eval compatibility readers (`starts` table, one-entry records, `catalog`, `@1` fallbacks) | 1.2 | ~95 | low |
| 4 | Monitor: one eval scorer instead of three (they disagree today), a `newest()` helper for 22 copies, one ledger read per beat | 3.5 | ~115 | low |
| 5 | The file ledger and its five file stores; one database ledger (SQLite locally); stores no longer optional | 3.1 | ~640 + ~40 branches | medium |
| 6 | Old run-state rule, pre-registry handling, `catalog` launches, `appended()` fallback | 1.3–1.6 | ~75 | low |
| 7 | Small dead code (`starts_in`, `SubjectName`, `machinesPlace`, `Serving.served_by`, `through=`, `Control.spawn`) | 4.1 | ~45 | none–low |
| 8 | Test helper merges (`until` ×12, monitor client ×22, `quickly` ×6, `ThinkingRenderer` ×2, gateway builders ×2) | 3.7 | ~250 | low |
| 9 | CLI: parent parsers, a dispatch table, one error decorator, one start-record header, one run-id resolver | 3.4 | ~120 | low |
| 10 | `recorder` package renamed into the gateway; its tests and `recorder.md` folded in | 1.7 | ~80 docs | low (wide) |
| 11 | Front end: `useTopic`/`useWrite`, `ScoreCell`/`SolvedLegend`, `CheckpointSelect`, `SuiteMatrix` reused, NewRun's own settings parser | 3.6 | ~140 | low |
| 12 | `suite_for` implicit freezing | 2.4 | ~60 | low |
| 13 | tinker-cookbook replaced by our own key remap | 2.5 | ~45 + lock | low |
| 14 | Profiles and the one-process `Platform`, with `rollout engines`/`rollout runner` | 2.1–2.2 | ~1,000 net code, ~300 tests, ~170 TOML | high (it is the spine) |
| 15 | The launcher's process backend and `ray = "auto"` | 2.3 | ~75 code, ~40 tests | medium (a live launcher uses it) |

**Estimated lines removable:** about 8,000 in all. About 4,100 of that is the one JSON fixture. The other ~3,900 lines are
code, tests, TOML and docs, net of the code the profile replacement adds back (about 150 lines of actor code). Another
~800 test lines and ~400 doc lines get rewritten rather than removed.

## What the shared ledger still holds

Read-only queries of `~/.cache/rollout/ledger.db` decide which compatibility readers real data still needs.

- **Runs:** `curriculum-9` (named `qwencraft-1`) and `run_01M43XMSDW6YYRY6DK3GSD4TXZ` (`gsm8k-tinker-base`, a GSM8K eval
  through Tinker). Both are registered.
- **curriculum-9's tables:** only `groups`, `results`, `steps` and `episodes`. It has no `starts`, `ends`, `plans`,
  `claims`, `turns` or `evals` tables. Its steps carry no `settings` and no `suite_version`. Its group records are in
  the current flat format.
- **Checkpoints:** 31, with ids `curriculum-9@1` … `@31` (the `NAME@N` form), 11 rows in `checkpoints/released`, and the
  bookmark `original-ckpt` → `curriculum-9@31`.
- **Datasets:** one, `xymwprlxvxxnvvxy`, made from `curriculum-9@18..29`. It finds its blobs through its own `blobs`
  field.
- **Suites:** `math`, version 1 under key `suite` with `entries`. There is no `evaluations/*/starts` table and no
  `catalog` key anywhere. The eval subject record has `version: "math@1"` and `parts`, the current layout.
- **Empty tables:** `launches`, `run_settings`, `sandboxes` and `suite_names`.
- **Index:** `ledger_records_order` exists.
- **Presence:** three live roles.
  - `launcher/DESKTOP-TRLNC1U/gsm8k` uses **the process backend**. It runs `rollout_verifiers.environments:gsm8k` from
    rollout-verifiers' own project, which has no Ray.
  - A Ray launcher offers the profiles `one-gpu` and `tinker`.
  - A gateway replica runs as `rollout gateway gsm8k_tinker.toml`.
  - `~/.cache/rollout/launch-profiles/gsm8k/gsm8k_tinker.toml` is a symlink into the run worktree.

**So these readers can go:** pre-version suite layouts, the `catalog` key, subjects without a version or parts, evals
records without entries, and launches written with `catalog`.

**Real data still needs these, so keep them:**
- `NAME@N` checkpoint ids and their display (`RT/checkpoints.py:104-118`), unless a migration renames 31 ids and every
  reference to them;
- runs without `starts`/`ends`, read through `RT/monitor/system.py:981`;
- `datasets.where_blobs_are`'s `file://` fallback (`RT/datasets.py:333-342`);
- steps without `settings` (`RT/loop.py:301-302`; it matters only if curriculum-9 is resumed);
- the `"ended"` run state.

---

## 1. Leftovers of replaced mechanisms

### 1.1 Lineage readers for tables nothing writes, and the sample fixture

- **Where:**
  - `RT/monitor/lineage.py:13-27`. Its own docstring says these tables are "proposed in docs/research/policy-dag.md, and
    nothing appends them yet".
  - `:54-57` and `:116-128`: the fixture is loaded and its times shifted.
  - `:84`, `:101-113`.
  - `:207` and `:229`: `runs/RUN/plan` and `runs/RUN/published`.
  - `:362-403`: `workers/*` registered, loaded and unloaded.
  - `:405-470`: `trainers/*` registered, queue and taken.
  - `:150` and `:182-185`: the feed's `routing` notes.
  - `:60-69` and `:130-137`: distillation modes.
  - The fixture itself: `RT/monitor/sample-lineage.json`, 4,064 lines.
  - The `?sample=1` switch: `RT/monitor/app.py:64-65, 134-135`, `RT/monitor/stream.py:125`, `RT/monitor/system.py:667-675`.
  - Front end: `WEB/App.tsx:29`; `WEB/components/ui.tsx:162` (`SampleChip`); `WEB/pages/Checkpoints.tsx:5, 188-202, 359,
    384, 454, 480, 529, 568` (trainer and worker cards, distillation modes, about 27 `sample` references);
    `WEB/api/queries.ts:211` (`useLineage(sample)`).
- **Why:** these read tables no code writes, so real ledgers never show anything in these views. Engine hosts' beats
  and the `runs/RUN/serving` table replaced the "workers" idea. The fixture exists only to make the views visible.
- **Risk:** low.
  - `workers()` also folds in the feed's `published` notes (`:390`, the `serving` set). Keep that, or read engines from
    `serving` and the `engines` beats.
  - Trim `tests/rollout_train/monitor/test_lineage.py:91, 192, 228-240, 237, 277` and the `sample: false` fields in
    `WEB/page.test.tsx`.
  - `docs/research/policy-dag.md` stays: it is the proposal.
  - The committed bundle under `RT/monitor/static/` must be rebuilt in the same commit, as for every front-end change.
- **Replacement:** none. Keep checkpoints, runs, steps, reshards, bookmarks and evaluations.
- **Lines:** about 4,064 JSON, ~200 in lineage.py, ~60 tests and ~80 TSX.

### 1.2 Compatibility readers for old suite and eval layouts

No real record uses any of these layouts (see the ledger section above).

- **Suites written as a record plus a separate `starts` table:**
  - `RT/evals.py:19-21, 38` (docstring), `:98-101` (`suite_table` mentions `starts`), `:298-300`;
  - the `starts` parameter threaded through `versions_of`/`versions_in`/`_versions` (`:394-410`) and through
    `_as_suite`/`_as_entry` (`:417-434`);
  - `suites_among` also matching `/starts` (`:467-470`), and `RT/monitor/lineage.py:527`.
- **One-entry suite record with no `entries`:** `_as_suite` at `:422-424`, and the inferences in `_as_entry` at
  `:436-442`: `chosen` from `held_out`, `eval_data` from the suite name, `environment_version` from `version`.
- **The `catalog` key:** `RT/evals.py:435`.
- **A subject without a version:**
  - `RT/evals.py:123-126`: `played_version` falls back to `NAME@1`.
  - Front end: `WEB/lib/suites.ts:29-44, 49-50` (`versionsOf` invents a `NAME@1` version; `playedVersion ?? @1`),
    `WEB/layout/Tree.tsx:235` and `WEB/pages/Checkpoints.tsx:391`.
  - `WEB/page.test.tsx:482`.
- **An eval without `parts`:** `RT/evals.py:486-489`.
- **A training run's `evals` record without `entries`:** `RT/loop.py:590-597` (`_entries_of`).
- **Step settings without `suite_version`:** `RT/loop.py:305` (`or said[EVALS_SUITE]`).
- **An eval start without a version:** `RT/resuming.py:155` (`or newest.get("suite")`). See also the bug in the
  appendix.
- **`starts_in`:** `RT/evals.py:452-458`, a reader "whichever way the suite was written". Nothing calls it.
- **Docs:** `docs/libraries/rollout-train/evals.md:78-80`.
- **Tests to delete:** `tests/rollout_train/test_suite_versions.py:133-160`
  (`test_a_suite_made_before_versions_reads_as_its_version_1`) and `tests/rollout_train/test_evals.py:110-117`
  (`test_a_suite_made_as_a_catalogs_reads_as_its_environments`).
- **Keep:** `NAME@N` **suite version ids** are the current scheme (`math@1`, and `suite_names.version`), so `versionTag`
  and `suiteName` in `WEB/lib/suites.ts` stay.
- **Optional:** version 1 is stored under the key `suite` while later versions use their number (`RT/evals.py:86-88,
  409`). That is a relic of the time before versions, but changing it means migrating the one `math` record. Low payoff.
- **Lines:** ~50 source, ~40 tests, ~25 web, 3 docs.

### 1.3 The old run-state rule (state by write time)

- **Where:**
  - `RT/monitor/system.py:993-994`, the `else` branch: `RUNNING if quiet < QUIET else IDLE if quiet < SILENT else GONE`.
  - The `SILENT` constant (`:152`) and its docstring (`:140-150`, "a run from before runs said how they ended…").
  - `WEB/api/types.ts:968-970`.
- **Why:** every run now beats and writes `ends`.
- **Risk:** low. `tests/rollout_train/monitor/test_runs.py:52-66` expects a run with no beats, written an hour ago, to
  read `IDLE`; change it to beat, or to expect `GONE`. curriculum-9 already shows as ended.
- **Replacement:** a run with no beat and no end reads `GONE` ("ended"). Keep that state. A run whose `started` is
  within `STALE` reads as running, so a new run does not flash "ended" before its first beat lands.
- **Lines:** ~10.

### 1.4 Handling for runs from before the registry

- **Where:**
  - `RT/registry.py:10-11`, `:309-310`, `:317-318` (`run_of` registers an existing ledger key) and `:356` (the fallback
    `Entry(who, who, 0.0)`).
  - `RT/monitor/system.py:265-269` (`rename` registers first) and `:1330-1333` (`_run_in` falls back to the directory
    name).
  - Docs: `docs/libraries/rollout-train/monitor.md:161`, `checkpoints.md:220`.
- **Why:** both real runs are registered.
- **Tests:** `tests/rollout_train/test_registry.py:123` and `tests/rollout_train/monitor/test_stream.py:111` build runs
  that predate the registry; change them to register.
- **Lines:** ~15 source, ~10 tests.

### 1.5 `catalog` naming in launches

- **Where:** `RT/launches.py:119-129` (`as_asked` renames `catalog` to `environment`) and `RT/monitor/system.py:317`.
- **Why:** the `launches` table is empty, and no front-end code sends `catalog`.
- **Tests:** `tests/rollout_train/test_launches.py:108-110` and `tests/rollout_train/monitor/test_launching.py:135-138`.
- **Replacement:** use `Asked(**fields)` directly.
- **Not leftovers:**
  - the "closed catalog" of events and effects (`R/contracts/events.py`, `effects.py`);
  - the example tool in `docs/guide/tools.md`;
  - Minecraft's `tasks.catalog()`, the task list. Renaming it to `tasks()` is optional and touches about 25 test lines.
- **Lines:** ~20.

### 1.6 The `appended()` fallback for ledgers without `append_returning`

- **Where:** `RT/ledger.py:78-88`, which uses `getattr(ledger, "append_returning", None)`.
- **Why:** every real ledger implements it. Only `tests/rollout_train/test_ledger.py:151-175` (`Plain`) exercises the
  fallback.
- **Replacement:** add `append_returning` to the `Ledger` protocol and delete that test.
- **Lines:** ~30.

### 1.7 The `recorder` package: live code under the old recorder's name

- **What it is:** nothing in `RT/recorder/` is dead. The gateway uses `renderers`, `sampling.sample_turn`, `segments` and
  `compat/{chat,messages,responses,wire}` (`RT/gateway/service.py:55-57`, `RT/gateway/client.py:40-41`,
  `RT/gateway/turns.py:48`). The package name is the leftover.
- **Stale text:**
  - `RT/recorder/sampling.py:3-4` ("the recorder in this process");
  - `RT/testing.py:41, 163-206` (`recording()`, "A runner's recorder");
  - `scripts/generate_reference.py:34` ("The model endpoint for trainable channels");
  - the root `pyproject.toml:55`;
  - `implementations/rollout-tinker/tests/test_engine.py:2`;
  - `docs/libraries/rollout-train/gateway.md:331` and `channels.md:163`.
- **Recorder-era names:** the field `recorder: Recorded` (`RT/rollouts/scheduler.py:24, 265`), the `recorder=`
  parameter of `LocalRunner`, and `ToolCallFormat`, which `RT/recorder/__init__.py` exports but nothing outside uses.
- **Tests:** `tests/rollout_train/recorder/{test_recorder,test_compat,test_sampling}.py` test the gateway; move them to
  `tests/rollout_train/gateway/`. `test_sampling.py:164-200` loads profiles and goes with them.
- **Docs:** `docs/libraries/rollout-train/recorder.md` (119 lines) overlaps `gateway.md` and `harness-endpoint.md`.
- **Replacement:**
  - Move `renderers`, `segments` and `sampling` to `rollout_train.tokens` (or `gateway.sampling`), and `compat` to
    `gateway/apis`.
  - Rename `recorder` to `gateway` in the scheduler and runner, and `recording()` to `gateway_endpoints()`.
  - Fold `recorder.md` into a "sampling and segments" section of `gateway.md`.
  - About 60 import sites change, including rollout-qwen, rollout-gemma and rollout-tinker.
- **Lines:** ~0 net code, ~80 docs.

### 1.8 The `through=` parameter on endpoint addresses

- **Where:** `R/contracts/model_endpoint.py:147-157`, `R/harness/hooks.py:84-85` and `R/testing.py:126-127`.
- **Why:** the in-process recorder honoured `through=`. The only real address provider, `GatewayClient.address`
  (`RT/gateway/client.py:237`), ignores it, because the gateway tells hooks itself (`RT/gateway/service.py:227`).
- **Risk:** low; it changes a protocol signature across rollout, hooks and testing.
- **Lines:** ~8.

### 1.9 A second copy of the gateway's turns: the feed's `sample` lines

- **Where:** `RT/monitor/feed.py:95-110` (`RunFeed.on_sample`) is wired only from `RT/profile.py:391, 442`. The episode
  page then chooses among the feed, the archived events blob and the gateway's turns (`RT/monitor/system.py:779-883`).
  The feed's cancellation rule (`feed.py:62-75`) exists only for the feed.
- **Why:** every run records through the gateway, and its turns are the transcript.
- **Risk:** medium; it changes what the episode page shows for live episodes. Coordinate with the profile removal, which
  removes the wiring anyway.
- **Related, for the designer:** `RT/rollouts/scheduler.py:583-601` still copies the gateway's segments into each
  episode's `trajectories` blob, so every segment is stored twice. curriculum-9 has no gateway turns, so the readers
  (`episodes.loaded`, datasets, imitation) must stay; only the writer could stop.
- **Lines:** ~100 if taken.

### 1.10 Engine protocol stubs left by the custom engine server

- **What is already gone:** the custom engine server (`serve_engines`, `/replicas/…`) was deleted in d6322bf. No route,
  client or doc of it remains.
- **What is left:** the `Engine` protocol (`RT/inference/channel.py:30-68`) still has the colocated-vLLM surface, which
  other engines stub:
  - `RemoteEngine.load_weights` raises `NotImplementedError` (`RT/inference/remote.py:179-183`);
  - `TinkerEngine.load_weights` raises, and its `sleep`/`wake` do nothing
    (`implementations/rollout-tinker/src/rollout_tinker/engine.py:91-98`).
- **Replacement:** a sampling core (`generate`, `load_adapter`, `remove_adapter`, `max_model_len`, `close`) plus optional
  `FullWeights` and `Colocated` capabilities, which `Channel.publish` and `RT/colocated.py:53-60` check for. This fits
  "one gateway for every backend".
- **Lines:** ~20.

### 1.11 Sandbox contract fields only the test fake honours

- **Where:** `R/harness/sandboxes.py:47-102`: `SandboxSpec.process`, `mounts`, `scratch` and `network`; and
  `SandboxLimits.cpus`, `memory_mib` and `processes` (`:75-77`).
- **Who reads them:** only `R/testing.py:186-252` (the fake provider) and `tests/rollout/harness/test_sandboxes.py:262`.
  The one real provider, `environments/minecraft/minecraft_team/worlds.py`, reads `kind`, `parameters`, `slots` and
  `limits.seconds` (`R/harness/sandboxes.py:349`).
- **Decision:** settle this in the environment-declarations design. Keep the fields if a container provider is coming;
  otherwise delete them.
- **Lines:** ~40 contract, ~50 fake.

### 1.12 Already clean

- **Minecraft `begin`/`end` traces:** no code reads or writes them. They were removed in e0ef88c; only curriculum-9's
  event blobs carry them. `minecraft_team/datasets.py:worked` reads both the old and the current form. The only
  cosmetic leftover is `"episode": "e"` in `environments/minecraft/tests/test_datasets.py:23`.
- **The in-process recorder:** no code remains beyond the naming in 1.7 and `through=` in 1.8.
- **CLI flags and environment variables:** a script checked every `add_argument` in every CLI against its reader, and
  every flag is read. The environment variables read are `DISCORD_WEBHOOK_URL`, `TINKER_API_KEY`, `AWS_ENDPOINT_URL`
  and `AGENTS_URL`; all are live. Profile keys go with `profile.py` (section 2).
- **Small text fixes:**
  - `RT/database.py:500`: drop the "in a database made before it had one" clause and keep `CREATE INDEX IF NOT EXISTS`.
  - `RT/monitor/system.py:800`: the "(the record's, once kept)" label fallback; check it before removing.

---

## 2. Paths without Ray, and code that depends on profiles

### 2.1 Entry points today, and what happens to each

| Entry point | What it does today | Fate |
|---|---|---|
| `rollout train PROFILE ENV` (`RT/cli.py:63-146`) | Loads the profile and runs every role in one process through `Platform.start` | Stays as the Ray driver, over run settings (`--preset`, `--set` or a settings document); the profile argument goes |
| `rollout eval PROFILE SUITE` (`:149-219`) | The same, without a trainer; copies its ledger default from the profile | The same |
| `rollout imitate PROFILE` (`:436-522`) | Reads trainer kind and settings, model, renderer, ledger and blobs from the profile; trains in its own process | Stays; settings come from a preset or the run; the trainer runs as a Ray task |
| `rollout env check ENV [--profile --groups]` (`:361-433`) | Scripted check through `LocalRunner`, plus groups played through `Platform` | The scripted check stays; the groups become a run on the cluster; tools and pools come from the environment's declaration (`--tool`/`--pool` only as overrides) |
| `rollout engines PROFILE --run` (`RT/hosting.py:301-316`) | A `Follower` loading checkpoints into vLLM servers on another machine | Goes; an engine actor does this |
| `rollout runner PROFILE` (`RT/hosting.py:319-325`) | `Platform` as a runner only | Goes; a runner actor does this |
| `rollout gateway PROFILE` (`RT/cli.py:252-291`, `RT/gateway/__init__.py:45-76`) | A gateway replica built by `deployed(profile)`; live for Tinker | The role stays, built from the cluster config |
| `rollout pool` / `rollout tools FACTORY` (`RT/cli.py:222-249`, `:975-990`, `:1108-1118`) | HTTP servers for sandbox pools and tool sets | The server code (`R/harness/remote.py`) stays; whether the CLI entry points stay depends on whether pools live outside Ray |
| `rollout launcher --profiles DIR [--ray] [--as-job]` (`RT/cli.py:643-666`, `RT/launcher.py`) | A process or Ray-job backend; offers profiles | The process backend goes; Ray becomes required; presets replace `--profiles`; `--as-job` stays |
| `python -m rollout_train.cli …` | The launcher's child command and the Ray job entrypoint | Stays, as the Ray entrypoint |
| `monitor`, `ledger copy`, `rename`, `pause`, `resume`, `bookmark`, `checkpoints`, `merge`, `suite *`, `dataset *`, `report` | Ledger tools | Stay; `resume` stops needing a profile path; `ledger copy` goes with the file ledger (3.1) |
| `scripts/train-with-memory-log.sh` | Wraps `rollout train PROFILE ENV` with a monitor and memory logs, from before the launcher existed | Delete |
| `minecraft-team server`, `agents`, `project-assistant`, `rollout_tinker.weights` main | Developer and product tools | Unaffected |

### 2.2 Profiles: `RT/profile.py` (712 lines), `Platform` and everything that passes a profile

**Where each profile field goes:**

| Field (`RT/profile.py` line) | Destination |
|---|---|
| `directory` (:188) | Cluster config (the root for runs); a run's own directory stays the run's (the launcher already picks `NAME-ID`) |
| `channels.*.model` (:55) | Run settings or preset (the base model) |
| `.renderer` (:57) | Preset, or a model-to-renderer table in the cluster config |
| `.engine`, `.engines[]` (:59-62: GPU share, `max_num_seqs`, `max_lora_rank`) | Cluster config (hardware); `max_model_len` is arguably a run setting |
| `.thinking_tokens`, `.answer_tokens` (:63-67) | Run settings (launches already change them, `RT/launcher.py:79-81`) |
| `.reshard` (:68) | Cluster config (it follows the engine kind) |
| `.max_lag` (:71) | Run settings (already changeable, `RT/settings.py:45`) |
| `.via`, `.connection`, `REMOTE`/`.routed` (:75-107) | Cluster config |
| `trainer.kind` (:112) | Preset (lora, full or tinker); the cluster config says where it runs |
| `trainer.channel` (:114) | Gone (a run has one trained policy), or a run setting |
| `trainer.start`, `.bookmark` (:116-121) | Run settings (already `Asked.start`/`bookmark`) |
| `trainer.colocated` (:122) | Cluster config (placement) |
| `trainer.settings` (:124) | Run settings plus presets |
| `runner` local or durable (:193) | Cluster config |
| `serve`, `address` (:195-199) | Cluster config (the gateway address); mostly obsolete once every run records through a gateway replica |
| `tools` (:200) | Environment declaration (which tool sets), plus cluster config (URLs of shared services) |
| `pools` (:202) | Environment declaration (sandbox kind and provider), plus cluster config (capacity and URLs) |
| `ledger`, `blobs` (:205-212) | Cluster config |
| `runs_gib`, `training_gib` (:213-216) | Cluster config, or Ray `memory=` resources; with the latter, `require_memory`/`NotEnoughMemory` (:166-183, :693) go |
| `episodes_at_once` (:217) | Run settings, capped by cluster capacity |
| `feed_runs` (:220) | Cluster config, or a fixed default |
| `ray` (:223) | Delete |
| `name` (:225) | The run's own (`Asked.name`) |
| `evals` (`EvalsSpec`, :127-142) | Run settings (already changeable `evals.*`) plus presets |
| `gateway` (`GatewaySpec`, :145-163) | Cluster config |
| `Profile.load`, `_table`, `_only`, the dotted `--set` keys (:234-331; `RT/cli.py:588-599`) | Run-settings validation (`settings.checked`/`applied`) plus merging a preset |

**Every place that loads, offers or passes a profile:**
- **CLI (`RT/cli.py`):**
  - the docstring (`:1-18`);
  - `_train` (`:63-146`; `fixed(described, …)` at `:131`; `started["profile"]`), `_evaluate` (`:149-219`), `_gateway`
    (`:252-291`), `_check` (`:361-433`; tools and pools merged at `:394-396`; groups at `:417-431`) and `_imitate`
    (`:436-522`; `"profile"` at `:508`);
  - the parsers (`:793`, `:796-806`, `:816-820`, `:886`, `:903-915`, `:948-974`, `:992-994`) and the dispatch
    (`:1065-1073`).
- **Launcher and launches:**
  - `RT/launcher.py:3-11`, `:59-106` (`offered`, `_weights`), `:119`, `:157-182` (matches by profile, plus the
    `evals.suite=""` hack), `:188-200` and `:317-323`;
  - `RT/launches.py:3-7, 62-63, 76-77` (`Asked.profile`).
- **Resuming:** `RT/resuming.py:9-15`, `:61`, and `:120-141` (`relaunch` needs `starts.profile`). Also `:185-206`
  (`_offers`, `_profiles`, `_profile_of`); the run's own recorded settings replace all of them.
  - The one `starts` record that carries a profile path points into a deleted worktree, so it already cannot resume.
- **Settings:** `RT/settings.py:4` and `:138-155` (`fixed(profile, trainer)`).
- **Gateway:** `RT/gateway/__init__.py:10, 22-23, 45-76` (`deployed`).
- **Hosting:** all of `RT/hosting.py` (64 lines).
- **Monitor backend:**
  - `RT/monitor/system.py:288-360` (`launches()`/`launch()` check `asked.profile`), `:362-387` (`_checked_evals` falls
    back to the profile's `[evals]`) and `:1003`;
  - `RT/monitor/machines.py:5, 32-33, 82-86, 236`;
  - `RT/monitor/environments.py:14, 378`.
- **Front end:**
  - `WEB/pages/NewRun.tsx:1, 8, 41, 61-73, 84-107, 126, 144-151, 162, 211`;
  - `WEB/components/play.tsx:6, 24-31, 52, 66, 79, 139-141`, `WEB/components/machines.tsx:163-170` and
    `WEB/components/launches.tsx:53`;
  - `WEB/api/types.ts:136, 224-232` (`OfferedProfile`), `:341, 347, 352, 385, 426, 947`;
  - `WEB/lib/settings.ts:1`, `WEB/api/queries.ts:153`, and `WEB/page.test.tsx:25, 232`.
  - `Run.profile` (`types.ts:136`, sent at `RT/monitor/system.py:1003`) and `check.profile` (`types.ts:947`, sent at
    `RT/monitor/environments.py:378`) are read by no page. They can go now.
- **Docstrings that name profiles:**
  - `RT/layout.py:1, 6, 10`, `RT/stores.py:1, 18`, `RT/testing.py:3, 225`, `RT/check.py:9`, `RT/record.py:4`,
    `RT/loop.py:173`, `RT/inference/remote.py:47, 294`, `RT/recorder/renderers.py:7`, `RT/database.py:513`;
  - the `__init__.py` of rollout-vllm, rollout-qwen and rollout-gemma (`:3`);
  - rollout-tinker's `settings.py:1`, `testing.py:20, 393`, `service.py:159` and `trainer.py:80`;
  - `minecraft_team/worlds.py:9, 250` and `libraries/rollout-train/pyproject.toml:4, 16`.
- **Profile files (~170 lines):** `environments/minecraft/profiles/one-gpu.toml` and `tinker.toml`, and
  `implementations/rollout-verifiers/examples/gsm8k_tinker.toml` and `gsm8k_vllm.toml`.
  - They become a cluster config, presets, and an environment declaration for the Minecraft worlds pool.
  - `~/.cache/rollout/launch-profiles/` symlinks into the run worktree; it moves only at a deploy boundary.
- **Tests:**
  - Delete `tests/rollout_train/test_profile.py` (193 lines) and `implementations/rollout-tinker/tests/test_profile.py`
    (110 lines).
    - `test_profile.py:176-193` also loads every ```` ```toml ```` block in `docs/guide/*.md` and every
      `environments/*/profiles/*.toml`, so the docs' TOML is coupled to the profile parser. The replacement should check
      its own configuration examples.
  - Rewrite onto run settings plus a cluster config: `test_full_weights.py` (242 lines), `gateway/test_hosted.py` (163),
    `gateway/test_replicas.py` (139), `gateway/test_on_gpu.py` (96) and `test_machines.py` (118).
  - Edit: `test_launches.py:113-150, 239`, `test_evals.py:464-478`, `test_sandboxes.py:257-277`,
    `test_settings.py:114`, `test_datasets.py:262`, `recorder/test_sampling.py:164-200`,
    `monitor/test_launching.py:32-171`, `test_pausing.py:20`, `test_suite_*.py`,
    `research/test_ledger_guarantees.py:147` and `rollouts/games.py:9`.
- **The one-process `Platform` (`RT/profile.py:377-507`).** It starts engines (`started_engines`, `:637-672`), the
  trainer, an in-process gateway (`_recording`, `:509-527`), a uvicorn server for harnesses (`_serve`, `:626-634`), tool
  sets, pools with lease keepers (`_pools`, `:529-551`), and a local or durable runner (`:456-475`).
  - Ray replaces it with engine, trainer, gateway and runner actors.
  - These parts move to the run driver rather than being deleted: `eval_run`, `bookmarked`, `made`, `publish` and
    `_about`.
- **Code that disappears with profiles:** the "ledger and blobs default to files under the profile directory" logic,
  written five times (`RT/profile.py:341, 406-407`, `RT/cli.py:176-179, 471-473`, `RT/gateway/__init__.py:60-61`,
  `RT/hosting.py:25-30, 284-291`). One cluster-config reader replaces all five (3.4).
- **Risk:** high. `Platform` is the spine of every run, and many tests open it through `Profile.load(...).open()`.
- **Lines:** profile.py ~712 (~150 return as actor code), hosting.py 64, launcher ~110, CLI ~150, resuming ~40,
  gateway `deployed` ~30, front end ~60, TOML ~170, tests ~300 deleted and ~800 rewritten, docs ~400 rewritten plus the
  regenerated reference (`docs/guide/reference.md:4637-4800`; delete the module entry at
  `scripts/generate_reference.py:36`).

### 2.3 The launcher's process backend and `ray = "auto"`

- **Process backend:**
  - Code: `RT/launcher.py:204-234` (`create_subprocess_exec`, `_watch`), `:151-153` (SIGINT to its own process),
    `:304-308` (adopts children by pid), `:129` (`_playing`), `:333-337` (`_tail`), `:326` (`"backend"`) and the
    docstring at `:13-16`.
  - `RT/launches.py:108` (`Launch.pid`) and `RT/monitor/machines.py:87` (`backend`).
  - Front end: `WEB/components/launches.tsx:74` (the pid chip), `WEB/components/machines.tsx:163` (the "on Ray" label)
    and `WEB/api/types.ts:352, 416`.
  - CLI: `RT/cli.py:890-892`, where `--ray` becomes required or comes from the cluster config, and `:1056-1057`.
  - Packaging: the `ray` extra becomes a dependency (`libraries/rollout-train/pyproject.toml:17`; the root
    `pyproject.toml`'s `ray = […]` extra).
  - Docs: `docs/guide/deploying.md:397-497`, `docs/architecture/glossary.md:59` and
    `docs/libraries/rollout-train/evals.md:194`.
- **Tests that use the process path** by patching `asyncio.create_subprocess_exec`:
  - `test_launches.py:159` (the `Process` fake), `:190, 255`, `test_evals.py:232`, `test_suite_versions.py:361`,
    `test_pausing.py:264`, `test_suite_entries.py:234` and `research/test_ledger_guarantees.py:662-709`.
  - They move to the fake `JobSubmissionClient` already in `test_resharding.py:86-110`, which becomes shared.
- **`ray = "auto"` and the in-process reshard fallback:**
  - `RT/profile.py:223-224, 273, 398-400` and `:559-565` (`Platform.reshard` falls back to running `reshard(...)` in
    process);
  - `environments/minecraft/profiles/one-gpu.toml:6` and the docstring at `RT/resharding.py:10-11`;
  - docs: `deploying.md:103, 171, 188, 233-234, 414-416, 481`, `training.md:66`, `checkpoints.md:135-140`,
    `overview.md:66`, `glossary.md:62` and `minecraft-team.md:324`.
- **Keep:**
  - `reshard()` (`RT/resharding.py:57`), the body that `_on_worker` runs;
  - Ray `connect`/`disconnect`, which the driver then calls unconditionally;
  - `RT/ray_cluster.py`.
- **Risk:** medium.
  - **The live GSM8K launcher uses the process backend** from rollout-verifiers' own project, which has no Ray. That
    project, and rollout-tinker, must join the workspace with Ray first.
  - Only `one-gpu.toml` names a `reshard`. The first test in `tests/rollout_train/test_resharding.py` keeps `reshard()`
    covered.
- **Lines:** ~75 code, ~10 front end, ~40 tests (the fakes shrink to the shared one), ~35 docs.

### 2.4 `suite_for`: freezing an environment's eval data into a suite without being asked

- **Where:**
  - The definition: `RT/evals.py:344-369`. `MAKERS` (`:94`) stays, because `edit_suite` uses it.
  - Callers: `RT/cli.py:81, 104-124` (`rollout train` resolving `evals.suite`) and `:164, 178-181`
    (`rollout eval --environment`).
  - `RT/monitor/system.py:324-327` (the freezing branch in `launch`) and `:557`.
  - The export at `RT/__init__.py:14, 35, 82`.
  - `RT/cli.py:356-359`, the `suite list` branch "not played yet".
- **Tests:** `tests/rollout_train/test_evals.py:33, 96-105` and `test_suite_versions.py:31, 218`.
- **Docs:** `evals.md:24, 63-74, 91, 223-224, 260`, `training.md:230-233`, `monitor.md:230-232`, `glossary.md:43`,
  `rollouts.md:205`, `deploying.md:75` and `research/ledger-guarantees.md:612`.
- **Replacement:** `suite_of` plus a clear error, "no suite NAME; make one". `rollout eval --environment` either needs an
  explicit `make_suite` step or goes.
- **Lines:** ~45 code and tests, ~15 docs.

### 2.5 tinker-cookbook

- **The only code import:** `implementations/rollout-tinker/src/rollout_tinker/weights.py:88-98`. `convert()` imports
  `tinker_cookbook.weights.build_lora_adapter`, the key remap that turns Tinker's archive names into PEFT's layout and
  `adapter_config.json`.
  - It runs only when `weights = "peft"` (`settings.py:8`; `trainer.py:366-374` → `downloaded` → `converted`).
  - `converted` (`weights.py:73-85`) is a subprocess with `CUDA_VISIBLE_DEVICES=""`. It exists only because the
    cookbook's converter opens the GPU.
- **Keep:** `fused` (`weights.py:103-151`), which merges the q, k and v adapters into one `in_proj_qkv` adapter. It is
  ours.
- **Packaging:** `implementations/rollout-tinker/pyproject.toml:1-3, 16, 39-41`, i.e. `tinker-cookbook==0.5.7` and the
  `transformers==5.17.0` override. That pin is why the project is locked apart (root `pyproject.toml` workspace
  `exclude`).
- **Replacement:** rename the keys in `convert` with safetensors directly and write `adapter_config.json` (r,
  lora_alpha, target_modules, base model). Then drop the subprocess, the GPU-hiding variable and the transformers
  override. Pin the mapping against the archive names the live test records (`docs/research/thinking-machines.md:584`).
- **Tests:** `implementations/rollout-tinker/tests/test_weights.py` and `testing.py:85`.
- **Docs:** `docs/implementations/rollout-tinker.md:15-18, 119, 197, 208, 278`, `docs/README.md:47, 106` and
  `docs/research/thinking-machines.md` (20 mentions).
- **Lines:** ~25 code, ~20 packaging and docs, plus a separate `uv.lock`.

### 2.6 Recording and ending orphaned engine processes

- **Where:**
  - `R/processes.py:33-64` (`note_processes`, `end_orphans`) and `RT/layout.py:11` (`PROCESSES`);
  - its uses at `RT/profile.py:425-426, 665, 675-690` and `RT/hosting.py:314`;
  - the front end at `WEB/api/types.ts:277` (`processes`, read by machines.tsx);
  - the test `tests/rollout/test_processes.py`.
- **Why:** it exists because a killed driver left engines holding the GPU. Ray supervises actor processes.
- **Keep:** `end_with_parent` (used by `implementations/rollout-lora/src/rollout_lora/worker.py:24` and
  `minecraft_team/paper.py:45`) and `children` (used by `rollout_vllm/engine.py:16`).
- **Risk:** medium. Delete only once engines are Ray actors and Ray's kill has been shown to reap vLLM's children.
- **Lines:** ~50 code, ~15 tests.

### 2.7 What survives

- **`R/local/*` (`LocalRunner`, 737 lines).** It is the in-process program runner, not role orchestration. Its users:
  - the products (`agent_sessions/service.py:155`, `project_assistant/service.py:81`,
    `project_assistant/evaluation/harness.py`);
  - `RT/check.py:228` and `rollout_durable`;
  - about 25 test files;
  - the future runner actor.

  Only its docstring (`R/local/__init__.py:1`, "for the local profile") changes.
- **`RT/colocated.py`.** The trainer shares a GPU with engines in the same process. That is the real one-GPU
  configuration, used by curriculum-9. It is not Ray-less as such. The replacement either puts trainer and engines in
  one actor or adds a sleep and wake protocol between actors. This is a design item, not a removal.
- **`RT/machine.py`** (heartbeat measurements), **`RT/ray_cluster.py`**, and rollout-lora's fresh process per step.

---

## 3. Duplication

### 3.1 Stores beside the ledger: one database ledger, stores always present

**The pairs:**

| Store | File variant | Database variant | Dispatcher |
|---|---|---|---|
| Ledger | `FileLedger`, `RT/ledger.py:98-283` (with `_Index`, `_key_of`, `_replaced`, `_synced_directory`) | `DatabaseLedger`, `RT/database.py:132-233` | `opened` `RT/ledger.py:289`, `of_run` `:300`, `present` `:308` |
| Registry (runs, bookmarks, dataset names, suite names) | `FileRegistry`, `RT/registry.py:170-305` | `DatabaseRegistry`, `RT/database.py:236-339` | `registry_of`, `RT/registry.py:163` |
| Presence | `FilePresence`, `RT/presence.py:93-145` | `DatabasePresence`, `RT/database.py:396-431` | `presence_of`, `RT/presence.py:69` |
| Launches | `FileLaunches`, `RT/launches.py:175-239` | `DatabaseLaunches`, `RT/database.py:342-393` | `launches_of`, `RT/launches.py:163` |
| Desired run settings | `FileDesiredSettings`, `RT/settings.py:175-213` | `DatabaseDesiredSettings`, `RT/database.py:434-459` | `desired_settings_of`, `RT/settings.py:71` |
| Sandbox leases | `FileLeases`, `RT/sandboxes.py:180-223` | `DatabaseLeases`, `RT/database.py:462-496` | `leases_of`, `RT/sandboxes.py:57` |

Checkpoints, datasets and suites are append tables inside the ledger, not stores of their own.

**The copied pattern.**
- Every file variant is a JSON file guarded by an `fcntl` lock on `.lock`, written to `.staged` and renamed into place.
  The `_locked()` context manager appears six times, verbatim: `ledger.py:227`, `registry.py:297`, `presence.py:127`,
  `launches.py:232`, `settings.py:206` and `sandboxes.py:216`. The staged write appears six times too.
- Every dispatcher is `isinstance(ledger, FileLedger)` → the file variant, else `getattr(ledger, attr, None)`.
- 49 call sites go through the dispatchers, and about 40 branches handle `None` (`registry.py:315, 334, 382`,
  `evals.py:388`, `cli.py:638`, `monitor/system.py:613`, …).

**The file ledger is no longer needed.**
- Both real runs point at the database ledger.
- The only file ledgers in `~/.cache/rollout` are scratch directories.
- Files are only the profile's default (`profile.py:341`, `hosting.py:25`, `gateway/__init__.py:60`,
  `cli.py:176, 472, 612`, `monitor/app.py:40`).
- The launcher already needs a database in practice (`cli.py:884`).

**Proposal.**
1. Delete:
   - `FileLedger` and the five file stores;
   - `present()` and the file fallback in `of_run`;
   - the detection at `monitor/app.py:40` and the file branch in `System.ledger` (`monitor/system.py:204`);
   - `database.copy` and `rollout ledger copy` (`RT/database.py:508-554`, `RT/cli.py:615-631, 860-865`). `copy` also
     silently forgets suite names (`database.py:547-553`).
2. Make `DatabaseLedger` the only ledger, with `sqlite:///…` as the local default. Its stores become required attributes
   (`ledger.registry`, `.presence`, `.launches`, `.settings`, `.leases`). The five `*_of` functions and the `None`
   branches go.
3. `DatabaseLedger` needs SQLAlchemy from `rollout_durable.database`. Either make `rollout-durable` a dependency of
   `rollout-train`, or move `Database`, `sql` and `fetch_*` into `rollout-train`.
4. Optional, afterwards: one `KeyedTable(database, table, key, columns)` with `get`, `all`, `put`, `delete` and
   `change(key, fn, exclusive=…)` under the remaining database stores. Leases, desired settings, bookmarks, dataset names
   and suite names collapse into it; launches use `change` with a state precondition; presence keeps its own clock SQL.
   About 90 lines of `database.py:236-496` go.

- **Risk:** medium, mostly test churn.
  - `FileLedger(...)` appears 149 times across tests and the tinker and verifiers tests. One `ledger` fixture in
    `tests/conftest.py`, `DatabaseLedger(f"sqlite:///{tmp_path}/ledger.db")`, replaces them. The existing `database`
    fixture (`tests/conftest.py:38`) already runs on SQLite and Postgres.
  - Delete the file-crash tests (`tests/rollout_train/test_ledger.py:60-138` and the file parts of
    `tests/research/test_ledger_guarantees.py`).
- **Lines:** ~640 source and ~40 `None` branches, plus ~90 with `KeyedTable`.

### 3.2 HTTP client and server boilerplate

1. **The remote tool set and the remote pool are twins** (`R/harness/remote.py`).
   - `RemoteToolSet` (`:74-122`) and `RemotePool` (`:194-259`) have the same `__init__`, the same `_describe`
     (`:100-107` and `:219-226`; both block on a synchronous `httpx.get` inside async code) and the same 500-to-
     `RuntimeError` mapping.
   - `serve` (`:47-71`) and `serve_pool` (`:131-191`) repeat the 500 handler.
   - Proposal: a described-client base and a shared `failed()` response helper. ~45 lines.
2. **Reading an error body:** `RT/inference/remote.py:224-239` (`_answer`, `_object`, `_said`) and
   `RT/gateway/client.py:293-300` (`_error`) both read `{"error": {...}}`. Proposal: one `error_of(response)` in
   `RT/http.py`. ~15 lines.
3. **Retry and backoff:** two loops in `RT/gateway/client.py` (`hosted` at `:121-133`, `_posted` at `:267-285`) and a
   third in `R/harness/model.py:134-146`. Proposal: one `retrying(attempts, backoff, retry_on)`. ~15 lines.
4. **Uvicorn setup, five copies:** `RT/cli.py:242`, `:281-291` (TLS and proxy headers), `:1120`, and
   `RT/profile.py:626-634`. Proposal: `serve(app, listen, *, certificate=None, key=None, proxied=None)`. ~20 lines.
5. **Leave alone:** the products' and Minecraft's own `httpx.AsyncClient` setup (`paper.py:290`, `control.py:17`).

### 3.3 Settings and option parsing

1. **Three ways to turn a profile into dotted keys:** `settings.fixed`/`changeable` (`RT/settings.py:138-164`),
   `launcher.offered` (`RT/launcher.py:73-84`) and `Profile.load`'s overrides (`RT/profile.py:258-263`). Proposal: one
   flat, validated run-settings schema, with `fixed`, `changeable` and `offered` as views of it.
2. **A lossy round trip.**
   - The launcher passes `--set KEY=json.dumps(value)` (`RT/launcher.py:183-186`), and the CLI parses it back as TOML
     (`_setting`, `RT/cli.py:588-599`).
   - TOML has no null and no inline JSON objects, so a null becomes the string `"null"`. That forces two sentinels:
     `evals.suite=""` (`launcher.py:181-182`) and the `"none"` budget (`profile.py:268-270`).
   - Proposal: pass the run's settings as one JSON document (`--settings FILE`, or the launch id). That deletes
     `_setting`, both sentinels and the dotted overlay.
3. **Five whole-number validators:** `RT/settings.py:108-115`, `RT/monitor/system.py:1297-1303` (`_whole`), the
   `ChannelSpec` (`RT/profile.py:82-85`) and `EvalsSpec` checks, and `RT/evals.py:~265`. Proposal: one
   `whole(value, what, least=1, optional=False)` in `RT/settings.py`; the monitor wraps its `ValueError`.
4. **Launch settings checked in two places:** `RT/monitor/system.py:345-350, 364-389` repeat what `_train` checks
   (`RT/cli.py:117-123`). Both should check against the one schema.
5. **Trainer settings repeated:** `TinkerSettings` (`implementations/rollout-tinker/src/rollout_tinker/settings.py:25-101`)
   repeats about 16 `LoraSettings` fields (`implementations/rollout-lora/src/rollout_lora/settings.py:50-125`), plus
   `__post_init__`, `loss` and `rate`. `changeable`/`change` are identical in `rollout_lora/trainer.py:30-38` and
   `rollout_tinker/trainer.py:95-103`.
   - Proposal: a shared objective-settings base. Put it in `rollout_train.trainer`, so that rollout-tinker depends only
     on the `tinker` SDK and the workspace's own libraries.
   - ~60 lines.

### 3.4 CLI commands with near-identical setup (`RT/cli.py`, 1,124 lines)

- **Repeated arguments.** `--ledger` appears 8 times with the same help (`:870-963`) plus 2 literal copies
  (`:849-855`); `--set` 3 times; `--directory` 9; `--name` 8. Proposal: parent parsers (`ledger_arguments`,
  `run_arguments`).
- **Dispatch.** A 120-line `if arguments.command == …` chain (`:1003-1120`), with
  `sys.exit(asyncio.run(until_signalled(work)))` written 7 times. Proposal: `set_defaults(handler=…)` and one runner.
  ~60 lines.
- **Error mapping.** `except (KeyError, ValueError) as error: raise SystemExit(error.args[0]) from None` appears 9 times
  (`:125, 186, 197, 344, 480, 702, 725, 747, 769`). Proposal: one `@user_errors` decorator.
- **The start record.**
  - The `{"host", "process", "started"}` header appears at `RT/cli.py:90, 190, 420-422, 504-508`, `RT/loop.py:205`,
    `RT/evals.py:633` and `RT/check.py:279`.
  - Proposal: `record.start_header(**extra)`.
- **Resolving a run's name or id to its id, four copies:** `RT/hosting.py:33`, `RT/cli.py:560-565, 735-739` and
  `RT/monitor/system.py:269`. Proposal: `registry.run_id(ledger, who)`.
- **Opening the deployment.**
  - Ledger and blobs are opened at `RT/cli.py:176, 471-472, 711`, `RT/hosting.py:25-30`,
    `RT/gateway/__init__.py:60-61` and `RT/profile.py:341, 406`.
  - Routes are built twice (`RT/gateway/__init__.py:66-73`, `RT/profile.py:433-440`), and the keyring is chosen twice
    (`gateway/__init__.py:75`, `profile.py:513-518`).
  - Proposal: one deployment object with `.ledger()`, `.blobs()`, `.routes()` and `.keyring()`, built by the cluster
    config. ~50 lines.
- **`{"kind": "module:name", …}` factory calls, five copies:** `RT/ledger.py:296`, `RT/stores.py:28-31`,
  `RT/cli.py:405, 471` and `RT/profile.py:540`. Proposal: `rollout.names.made(spec, *args)`.
- **A pool beside the ledger, built three times:** `RT/profile.py:535-548` and `RT/cli.py:232-251, 401-406`. Proposal:
  `sandboxes.ledger_pool(provider, name, ledger)` as an async context manager. ~25 lines.
- **Suite-entry building, twice:** `_suite` make and edit (`RT/cli.py:310-347`) duplicates `System.save_suite`/`_entry`
  (`RT/monitor/system.py:426-498`). Proposal: `evals.entry_from(spec, current)`. ~40 lines.
- **Beats.**
  - The channel listing is copied verbatim (`RT/gateway/beats.py:28-35`, `RT/profile.py:616-624`).
  - Four roles hand-roll their beat loops (`RT/launcher.py:310-314`, `RT/following.py:66-74`,
    `RT/rollouts/scheduler.py:355-370`, `RT/sandboxes.py:173-176`); only the gateway uses `presence.beating`.
  - Role names are built ad hoc in five places (`launcher.py:342`, `beats.py:19`, `cli.py:239`, `profile.py:478, 498,
    541`).
  - Proposal: `presence.beating` everywhere, plus `role_name(kind, *parts)` and `channels_about(channels, routes)`.
    ~35 lines.

### 3.5 Monitor backend readers that re-read the same tables

1. **The whole ledger is read once per topic.**
   - `System._tables()` (`RT/monitor/system.py:712-718`) issues one query per table.
   - It is called separately by `snapshot` (`:241`), `_environments_read` (`:424`), `checkpoint_evals` (`:588`), `path`
     (`:597`), `evals` (`:621`), `lineage` (`:671`) and `statistics` (`:680`).
   - Also repeated per topic: `names(registry_of(...))` (7 times), `checkpoints_in` (4 times) and `_beats()`.
   - The hub (`RT/monitor/stream.py:117-144`) reads every watched topic on every beat, so the whole ledger is read 3 to
     6 times every 1.5 s.
   - Proposal: a per-beat ledger view (tables, fences, names, checkpoints, beats) memoised by the hub, plus a bulk
     `DatabaseLedger.read_all()` (`SELECT name, key, record … ORDER BY name, position`) in place of N+1 queries.
   - ~30 lines saved, and most of the monitor's load.
2. **Eval scoring written three times, and they disagree.**
   - `System.evals` (`RT/monitor/system.py:630-665`), `scores._evals` (`RT/monitor/scores.py:119-188`, a superset) and
     `lineage._Reading.evaluations` (`RT/monitor/lineage.py:473-531`), plus `_solved_count` (`system.py:1323-1327`).
   - scores.py counts only episodes where `reported(...)` is truthy (`:150, 196`); system.py and lineage.py count
     `reported(...) is not False` (`:1325`, `:492`). So the same eval shows different solved shares on the Evals page
     and on a checkpoint's page.
   - Proposal: build `System.evals` and lineage's subject summaries from `scores.evals_in`, with one rule for unreported
     episodes. ~60 lines.
3. **The newest record of a table keyed by fence number, 22 copies:**
   - `RT/resuming.py:68, 98`, `RT/datasets.py:337` and `RT/settings.py:97`;
   - `RT/rollouts/scheduler.py:389, 429, 549` and `RT/cli.py:707`;
   - `RT/monitor/scores.py:136`, `environments.py:372` and `lineage.py:211, 392, 418`;
   - `RT/monitor/system.py:216, 525, 630, 837, 924, 981, 1514`.

   Proposal: `record.newest(records) -> Mapping | None`. ~25 lines.
4. **Where a run's blobs are, three copies:** `RT/datasets.py:333-342`, `RT/cli.py:705-710` and
   `RT/monitor/system.py:834-843`. Proposal: use `where_blobs_are` everywhere.
5. **Table names rebuilt by hand:** `RT/monitor/scores.py:130, 144` and lineage's `_EVALUATIONS`. Proposal: use
   `evals.subject_table`/`suite_table`.
6. **Small helpers, three copies:** `_mapping` (`RT/loop.py:647`, `RT/evals.py:473`, `RT/rollouts/scheduler.py:644`).
   Proposal: one `record.mapping`.

### 3.6 Front-end components repeated across pages

| # | What | Where | Consolidation | Lines |
|---|---|---|---|---|
| F1 | The suite grid written twice | `WEB/pages/Checkpoints.tsx:390-450` vs `WEB/components/evals.tsx:104-191` (`SuiteMatrix`) | Render `SuiteMatrix` (a compact prop) on Checkpoints; lineage's `evaluations` payload takes the evals payload's shape (pairs with 3.5.2) | ~50 |
| F2 | Query and mutation boilerplate: 12 `useQuery({queryKey: topics.X().key, queryFn: readJson(...)})` and 5 post-then-invalidate mutations | `WEB/api/queries.ts:102-221` | `useTopic(topic, {enabled})` and `useWrite(path, invalidates)` | ~30 |
| F3 | Checkpoint and bookmark `<select>` (base model, bookmarks, checkpoints newest first) | `WEB/components/play.tsx:97-115`, `WEB/pages/NewRun.tsx:159-183` | `CheckpointSelect` | ~25 |
| F4 | Score cell (solved/played, track bar) and the solved legend, written inline three times though `Legend` exists | `evals.tsx:160-170, 187-190`; `Checkpoints.tsx:338, 420-425, 445-448`; `ui.tsx:123` | `ScoreCell` and `SolvedLegend` in `ui.tsx` | ~15 |
| F5 | NewRun's own settings parser (`typed`/`shown`/`isBudget`), which has drifted: NewRun accepts a `groups_per_step` the settings form refuses | `NewRun.tsx:110-118` vs `WEB/lib/settings.ts:43-66` | `wantedOf`; move the "empty budget means none" rule into `settingOf` | ~10 |
| F6 | "This launcher offers these environments" written 3 times; profile grouping twice | `play.tsx:25-31, 73`, `NewRun.tsx:60-70, 91` | `offers(launcher, environments)` in `lib/`; the grouping goes with profiles | ~10 |
| F7 | `types.ts` (970 lines, 80 types) as one file | `WEB/api/types.ts` | Split by topic (runs, checkpoints, evals, machines, launches) after the removals | 0 |

The built bundle under `RT/monitor/static/assets/` is committed, so every front-end change rebuilds it in the same
commit.

### 3.7 Test helpers duplicated across test files

| # | What | Where | Consolidation | Lines |
|---|---|---|---|---|
| T1 | **Helpers living inside `test_*.py`, imported by other test modules (28 imports).** **Do this first**: deleting `test_profile.py` or the `Process` fake otherwise breaks 5 and 4 other files. | `training/test_loop.py` (`Counting:54`, `Notes:80`, `answering:117`, `Running:243`, `here`, `made_by`, `quickly`; imported by 10 files); `test_profile.py` (`PROFILE:24`, `Steps:49`, `write:63`; 5 files); `test_launches.py` (`Process:159`, `profiles:113`; 4 files); `test_sandboxes.py` (`BOX`, `GATES`, `ask`, `episode_runner`; 3 files); `rollouts/test_scheduler.py`; `test_evals.py`; `test_full_weights.py`; `monitor/test_launching.py`; `rollout/harness/test_hooks.py` | `tests/rollout_train/support.py`, beside `games.py` and `machines.py` | 0 net |
| T2 | `until(condition, seconds)` polling, **12 copies** | `rollout_durable/cluster.py:148`, `rollout_durable/test_eviction.py:72`, `rollout_durable/test_sandboxes.py:60`, `rollout_train/test_adoption.py:96`, `test_sandboxes.py:103`, `test_pausing.py:38`, `test_episode_fences.py:38`, `inference/test_remote.py:31`, `rollouts/test_scheduler.py:76`, `research/test_durable_guarantees.py:57`, `research/test_ledger_guarantees.py:104` | One `until(condition, seconds=10, every=0.01, message=)` taking a sync or async condition, in `rollout.testing` | ~70 |
| T3 | Monitor test client (`ASGITransport(app=create_app(...))` plus `AsyncClient(base_url="http://monitor")`), **~22 copies in 10 files** | `monitor/test_environments`, `test_scores`, `test_launching`, `test_stream`, `test_machines`, `test_monitor`, `test_system`, `test_lineage`; `test_evals:265`; `test_suite_versions:247, 323`; `test_suite_entries:264` | `async with monitor_client(path, beat=0.0)` | ~25 |
| T4 | Inline profile TOML, 10 copies | `test_profile.py:24`, `test_sandboxes.py:257`, `test_datasets.py:262`, `gateway/test_hosted.py:33`, `gateway/test_on_gpu.py:29`, `gateway/test_replicas.py:33`, `test_machines.py:23`, `test_evals.py:478`, `research/test_ledger_guarantees.py:147`, rollout-tinker `tests/test_profile.py:53` | One builder for the replacement's configuration, in `rollout_train.testing`, so ten new copies do not appear | ~150 |
| T5 | Two ways to build a test gateway, each with its own `SECRETS` | `RT/testing.py:41, 163-206` vs `tests/rollout_train/gateway/support.py:19-110` | One `SECRETS` and one `gateway_over` in `rollout_train.testing` | ~20 |
| T6 | `ThinkingRenderer` byte-identical in 2 files | `recorder/test_compat.py:44-58`, `gateway/test_harnesses.py:37-51` | `rollout_train.testing` | ~20 |
| T7 | `quickly` fixture, 6 copies | `test_evals.py:62`, `training/test_loop.py:49`, `test_settings.py:42`, `test_suite_entries.py:55`, `test_suite_versions.py:56`, `monitor/test_environments.py:39` | `tests/rollout_train/conftest.py` | ~20 |
| T8 | Durable test tasks and fake services | `Chat` (`rollout_durable/scenarios.py:106`, `test_durable_runner.py:40`, `test_eviction.py:31`, `rollout/local/test_runner.py:61`, `harness/test_messages.py:106`); `Counting` (`crash_child.py:30`, `scenarios.py:127`, `rollout/local/test_imports.py:54`); `LedgerEndpoint`/`LedgerEnvironments` in `crash_child.py:41` and `guard_child.py:26` shadowing `R/testing.py:116-163` | Import from `scenarios.py`; the child scripts wrap `rollout.testing` with their sleeps | ~60 |
| T9 | `free_port`, 3 copies | `gateway/test_hosted.py:55`, `monitor/test_stream.py:162`, `tests/conftest.py:53-55` | One helper | ~10 |
| T10 | `FileLedger(tmp_path / "ledger")` 149 times, `FileBlobStore(...)` 66 times | everywhere | A `ledger` and a `blobs` fixture (comes with 3.1) | small |

### 3.8 Docs that repeat the same explanation

Each topic gets one home; the other copies become links.

| # | Topic | Copies | Home | Lines |
|---|---|---|---|---|
| D1 | Changeable vs fixed settings; desired settings beside the ledger | `docs/libraries/rollout-train/training.md:206-237`, `monitor.md:220-238, 255-280`, `docs/guide/deploying.md:116-128, 418-426` | training.md | ~30 |
| D2 | Engines elsewhere: request protocol, checkpoint by name, `max_lag`, routing by session hash, the 5-minute wait | `channels.md:124-166`, `deploying.md:221-344` (nearly word for word), `glossary.md:74` | channels.md for the mechanism; deploying keeps the commands | ~40 |
| D3 | Running the gateway: keys, TLS, proxies, health, with and without `url` | `gateway.md:213-236, 329-382`, `deploying.md:362-395` | gateway.md | ~30 |
| D4 | "A gateway in the runner's process tells the runner's hooks; one elsewhere does not", 5 copies | `harness-endpoint.md:41`, `gateway.md:376-378`, `rollout/hooks.md:35`, `rollout/contracts/model-endpoint.md:88`, `deploying.md:383` | gateway.md | ~10 |
| D5 | The harness environment variables, 3 tables | `rollout/sandboxes.md:105-115`, `gateway.md:244-252`, `harness-endpoint.md:12-13` | gateway.md | ~10 |
| D6 | Launchers: what they offer, launch fields, claiming, states | `deploying.md:397-473`, `monitor.md:255-316` | deploying.md, rewritten once for presets | ~40 |
| D7 | Pausing and stopping | `training.md:238-272`, `monitor.md:240-253`, `deploying.md:498-502` | training.md | ~5 |
| D8 | Indexes of pages and packages, 5 copies | `README.md`, `docs/README.md:40-112`, `docs/guide/README.md`, `llms.txt`, `mkdocs.yml` | `docs/README.md` and `llms.txt`; `README.md` links | ~20 |
| D9 | `recorder.md` (see 1.7) | `docs/libraries/rollout-train/recorder.md` | gateway.md | ~80 |

**Docs describing what is being removed** (rewritten with each removal, not merged):
- **Profiles:**
  - `docs/guide/deploying.md` has 58 mentions (`:94-195` above all);
  - `perspectives.md:10, 93-98` and `docs/guide/README.md:7, 24, 35, 75`. The claim at `:7` that tests load every TOML
    block stops being true;
  - `glossary.md:29, 59, 74`;
  - training.md (22 mentions), monitor.md (21), evals.md (19), minecraft-team.md (12), gateway.md (10), rollouts.md (8),
    and channels, checkpoints and datasets (7 each);
  - `docs/implementations/rollout-tinker.md:54-90`, `README.md:4-6` and `llms.txt:5, 28-29, 70`.
- **Ray as optional:** see 2.3.
- **tinker-cookbook and "a project of its own":** see 2.5.
- **`suite_for` freezing:** see 2.4.
- **Compatibility wording:** `monitor.md:161, 238`, `checkpoints.md:220` and `research/ledger-guarantees.md:278`.
- **`docs/guide/reference.md`** (5,702 lines) is generated by `scripts/generate_reference.py`. Regenerate it after each
  docstring change; never edit it by hand.

---

## 4. Dead code

**Method.**
- `uvx vulture --min-confidence 60` over libraries, implementations, environments and products.
- AST scans that follow imports, `__init__` re-exports and `module:name` strings, checking:
  - top-level definitions with no users outside tests;
  - `__all__` exports unused outside their package;
  - modules nothing imports;
  - fields never read.
- A grep of every exported identifier in the front end.
- `ruff --select ARG,ERA` and strict `pyright` on `libraries/`.

**Results.** pyright reports 0 errors, and ruff with the project's rules is clean. Every source module is imported by
code outside tests. Plain dead code is small, about 120 to 160 lines. The large savings are in sections 1 to 3.

### 4.1 No callers at all

| Where | What | Risk | Lines |
|---|---|---|---|
| `RT/evals.py:452-458` | `starts_in` (also a compatibility reader, 1.2) | none | 7 |
| `WEB/components/evals.tsx:32` | `SubjectName` component (`subjectText` is used instead) | none | 3 |
| `WEB/lib/places.ts:43` | `machinesPlace` | none | 1 |
| `RT/serving.py:53-55` | `Serving.served_by`: written (`RT/evals.py:604, 620, 675, 678`, `RT/loop.py:330`) and stored, but never read by Python or TypeScript | low: drop the field and its parameter, or show it | ~10 |
| `R/harness/sandboxes.py:75-77` | `SandboxLimits.cpus`, `memory_mib`, `processes`: declared, never enforced (1.11) | low | 3 + docs |
| `environments/minecraft/minecraft_team/control.py:108-111` | `Control.spawn`, plus the Java route that serves only it (`GroundTruthPlugin.java:141, 650`, `/setup/spawn`) | low | ~4 Python, ~15 Java |
| `R/contracts/model_endpoint.py:147-157`, `R/harness/hooks.py:84-85`, `R/testing.py:126-127` | the `through=` parameter (1.8) | low | ~8 |

### 4.2 Called only by tests

| Where | Verdict |
|---|---|
| `RT/sandboxes.py:135` `sweep()` | It repeats one pass of `keep()`'s loop (`:160-166`). Have `keep` call `sweep`. |
| `RT/ledger.py:78-88` `appended()` fallback | Delete (1.6). |
| `RT/resharding.py:33, 38` `VERBATIM`/`verbatim` | Used by tests and `one-gpu.toml:16`. Decide with the profile removal whether the verbatim layout stays. |
| `implementations/rollout-tinker/src/rollout_tinker/service.py:168` `has_key()` | Keep if the CLI or launcher should check for a key; otherwise delete. |
| `rollout_tinker/service.py:93` `delete_checkpoint_from_tinker_path_async`; `rollout_tinker/weights.py:154` `ranks()`; `minecraft_team/control.py:113` `set_food`; `R/curriculum.py:61` `solved_share` | Keep: test cleanup, an operator helper, harness tests and a public helper for curriculum authors. |
| `rollout_gemma.gemma4`, `rollout_lora.FullTrainer`, `rollout_openai.ApiKey`/`CodexLogin`, `memory.CompactingAgent` | Not dead: reached by `module:name` strings from configuration. How they get named changes with profiles. |

### 4.3 Exports from `__init__` nothing outside the package uses

- **`RT/__init__.py`:** of its ~40 `__all__` names, almost all are used from outside only by tests. Examples:
  `Checkpoints`, `FileLedger`, `Ledger`, `Fence`, `make_suite`, `suite_for`, `suite_of`, `wanted`, `record_serving`,
  `Grpo`, `group_advantages`.
  - `Algorithm`, `Batch`, `Dataset` and `Manifest` are used nowhere outside the package.
  - Trim it to what the docs present as the API. `suite_for` and `FileLedger` go with 2.4 and 3.1.
- **`RT/recorder/__init__.py`:** `ToolCallFormat` is unused outside; the package moves (1.7).
- **Front end:** about 30 exports are used only in their own file. Drop `export`; no lines are saved, but it makes the
  next dead-code scan honest. They include:
  - `lib/scores.ts:11, 24`, `lib/environments.ts:58`, `lib/stored.ts:47`, `lib/format.ts:34`, `lib/machines.ts:71` and
    `lib/suites.ts:18, 58, 71, 96`;
  - `components/control.tsx:12`, `components/play.tsx:25`, `components/charts.tsx:13, 128, 151, 300`,
    `components/machines.tsx:30, 88, 112, 136, 159, 176`, `components/checkpoints.tsx:36`,
    `components/settings.tsx:13` and `components/ui.tsx:65`;
  - about 15 interfaces in `api/types.ts`.
- **Keep:** exports that are signature or protocol types even though only their own package uses them (`rollout_durable`,
  `rollout_openai`, `rollout.contracts`, `rollout.harness`, `rollout.local`, `rollout_train.gateway`, `agent_sessions`,
  `rollout_gemma`).

### 4.4 False positives

Recorded here so nobody chases them:
- protocol and SDK stub parameters (`rollout_tinker/service.py:80-105`, `testing.py:124-257`, `renderers.py:55-56`);
- SQLAlchemy `Table`s registered on metadata (`RT/database.py:55-123`);
- pydantic validators (`R/contracts/content.py:111, 173, 180`, `model_endpoint.py:84, 109`);
- tool methods (`agent_sessions/coordination/tools.py`, `rollout_computers/tools.py`);
- about 115 ruff `ARG` hits, all on callback signatures;
- `ERA001` at `rollout_durable/runner.py:434`, which is a heading comment.

---

## 5. Size

### 5.1 Lines per package (Python, TypeScript and TSX; Markdown for docs)

| Package | Lines | Files |
|---|---|---|
| `libraries/rollout-train/src` | 17,396 | 64 |
| `libraries/rollout-train/web/src` | 8,262 | 49 |
| `libraries/rollout/src` | 5,276 | 36 |
| `environments/minecraft` (code / tests) | 3,026 / 1,615 | 12 / 8 |
| `products/agent-sessions` | 1,783 | 10 |
| `products/project-assistant` | 1,510 | 12 |
| `implementations/rollout-tinker` (src / tests) | 1,411 / 915 | 8 / 8 |
| `implementations/rollout-lora` | 1,361 | 13 |
| `implementations/rollout-durable` | 1,352 | 5 |
| `implementations/rollout-computers` | 632 | 6 |
| `implementations/rollout-verifiers` | 433 | 5 |
| `implementations/rollout-openai` | 407 | 2 |
| `implementations/rollout-s3`, `-vllm`, `-gemma`, `-qwen` | 163, 158, 158, 53 | |
| `tests/` | 18,453 | 112 |
| (of which `tests/rollout_train`) | 11,611 | 56 |
| `scripts/` | 283 | 1 |
| `docs/` (Markdown) | 16,020 | |
| (of which the generated `docs/guide/reference.md`) | 5,702 | |
| Plus the JSON fixture `RT/monitor/sample-lineage.json` | 4,064 | |

### 5.2 Modules that are too large, and how to split them

Splits are pure moves. Do them after the removals, which shrink several of these modules first.

| Module | Lines | Split |
|---|---|---|
| `RT/monitor/system.py` | 1,521 | `System` is a 850-line facade over four concerns. Move them out: `monitor/control.py` (pause, resume, rename, bookmark, launch, stop, save_suite, want; `:252-580`), `monitor/episodes.py` (episode, turns, archived, assembled; `:773-1010`), `monitor/runs.py` (`_run`, `_group`, `_done`, `_state`, `_how_it_ended`, `_runners`; `:1070-1330, 1384-1521`) and `monitor/places.py` (`_Place`, `_Remote`, `_source`). The facade is left at ~300 lines. 3.5 shrinks it first. |
| `RT/cli.py` | 1,124 | `main()` alone is 330 lines of argparse (`:792-1124`). After 3.4, split by command family: `cli/runs.py` (train, eval, imitate, check), `cli/ledger.py` (rename, pause, resume, bookmark, checkpoints, merge), `cli/suites.py`, `cli/datasets.py` and `cli/services.py` (pool, tools, gateway, monitor, launcher). |
| `RT/loop.py` | 649 | `train()` is one 460-line function (`:128-590`) with about 25 nested closures. Make it a `TrainingLoop` class, and move its eval scheduling (`cadence`, `schedule_for`, `pinned`, `schedule_of`, `evaluated_with`; `:276-345`) beside `evals.Schedule`. |
| `RT/evals.py` | 780 | The suite model and its storage (`:98-482`) become `suites.py`; `evaluate()` and `Schedule` (`:550-780`) stay. 1.2 and 2.4 remove about 75 lines first. |
| `RT/rollouts/scheduler.py` | 646 | `EpisodeRunner` is a 380-line class. Claims, plans and scopes (`:92-210`) become `rollouts/claims.py`. |
| `RT/monitor/lineage.py` | 659 | Shrinks by about 200 with 1.1; then no split is needed. |
| `RT/profile.py` | 712 | Deleted (2.2). |
| `RT/database.py` | 554 | Shrinks with 3.1 (`copy` goes) and `KeyedTable`. |
| `environments/minecraft/minecraft_team/tasks.py` | 893 | Tasks and their scoring (`:1-565`) stay; building sites (`Site`, `Built`, `BuildError` and placement; `:569-893`) become `building.py`. `prompts.py` (630) is mostly text and can stay. |
| `WEB/api/types.ts` | 970 | Split by topic (F7). |
| `WEB/pages/Checkpoints.tsx` | 571 | Shrinks by about 130 with 1.1 and F1. |
| `WEB/page.test.tsx` | 613 | One test file per page. |
| `tests/research/test_ledger_guarantees.py` | 792 | Shrinks with 3.1 (file-crash cases) and 2.3 (process fakes). |

---

## 6. Order of removal commits

Each commit is small enough to review, and each names the tests that guard it. "Targeted" means the changed package's
tests. The full suite runs only where shared code moves. Every commit that touches `WEB/` rebuilds the committed bundle
and runs `npm test` and `tsc`. Every commit that changes a docstring regenerates `docs/guide/reference.md`
(`tests/test_docs.py::test_reference_is_current`).

### Phase A: independent of the replacement designs

| # | Commit | Items | Guards |
|---|---|---|---|
| 1 | Move test helpers out of `test_*.py` into `tests/rollout_train/support.py` | T1 | the whole of `tests/rollout_train`, with the same pass count |
| 2 | Delete the lineage readers for unwritten tables, the sample fixture and the `sample` switch | 1.1 | `monitor/test_lineage.py`, `page.test.tsx` |
| 3 | Delete the suite and eval compatibility readers, `starts_in`, and the front end's `@1` fallbacks | 1.2, 4.1 | `test_suite_versions.py`, `test_evals.py`, `monitor/test_scores.py`, `test_suite_entries.py`, `page.test.tsx` |
| 4 | State a run by its beats and ends only | 1.3 | `monitor/test_runs.py`, `monitor/test_system.py` |
| 5 | Delete handling for runs from before the registry | 1.4 | `test_registry.py`, `monitor/test_stream.py` |
| 6 | Delete `catalog` in launches; put `append_returning` in the `Ledger` protocol | 1.5, 1.6 | `test_launches.py`, `monitor/test_launching.py`, `test_ledger.py`, `research/test_ledger_guarantees.py` |
| 7 | Delete small dead code: `SubjectName`, `machinesPlace`, `served_by`, `through=`, `Control.spawn` and its Java route; `keep` calls `sweep` | 4.1, 4.2, 1.8 | `tests/rollout/harness`, `test_sandboxes.py`, `test_evals.py`, Minecraft harness tests, `page.test.tsx` |
| 8 | Delete the monitor's unread `profile` fields and `scripts/train-with-memory-log.sh`; fix the `LocalRunner` docstring | 2.2, 2.1 | `page.test.tsx`, `monitor/test_system.py` |
| 9 | Add `record.newest`, `record.mapping` and `start_header`; use `where_blobs_are` and `subject_table` everywhere | 3.5.3–6, 3.4 | `tests/rollout_train`, all of it (shared code) |
| 10 | One eval scorer with one rule for unreported episodes | 3.5.2 | `monitor/test_scores.py`, `test_lineage.py`, `test_system.py` |
| 11 | Read the ledger once per beat (ledger view plus `read_all`) | 3.5.1 | `monitor/test_system.py`, `test_stream.py`, the `database` fixture tests |
| 12 | Merge test helpers: `until`, `monitor_client`, `quickly`, `ThinkingRenderer`, `free_port`, the gateway builder, the durable fakes | T2, T3, T5–T9 | each touched package's tests |
| 13 | Front end: `useTopic`/`useWrite`, `ScoreCell`/`SolvedLegend`, `CheckpointSelect`, NewRun uses `wantedOf`, Checkpoints reuses `SuiteMatrix`; unexport file-local names | F1–F5, 4.3 | `page.test.tsx`, `machines.test.tsx`, `environments.test.tsx`, `tsc` |
| 14 | Move `recorder` into the gateway (pure move); move its tests; fold `recorder.md` into `gateway.md` | 1.7, D9 | the full suite (many packages import it) |
| 15 | Shared HTTP helpers: described client and `failed()`, `error_of`, `retrying`, `serve` | 3.2 | `tests/rollout/harness/test_sandboxes.py`, `rollouts/test_harness_and_tools.py`, `inference/test_remote.py`, `gateway/` tests |
| 16 | One database ledger: delete `FileLedger`, the file stores, the `*_of` dispatchers and `ledger copy`; `ledger`/`blobs` fixtures | 3.1, T10 | the full suite (shared); `test_ledger.py`, `test_registry.py`, `test_presence.py`, `test_launches.py`, `test_settings.py`, `test_sandboxes.py`, `research/` |
| 17 | `KeyedTable` under the database stores | 3.1.4 | the same, on SQLite and Postgres |
| 18 | CLI: parent parsers, a dispatch table, `@user_errors`, `registry.run_id`, `evals.entry_from` shared with the monitor, `ledger_pool` | 3.4 | `test_check.py`, `test_datasets.py`, `test_registry.py`, `test_pausing.py`, `test_launches.py`, `test_evals.py`, `test_resharding.py`, `monitor/test_environments.py` |
| 19 | Docs: merge D1–D8 | 3.8 | `tests/test_docs.py`, `mkdocs build` |

### Phase B: with the replacements

| # | Commit | Items | Guards |
|---|---|---|---|
| 20 | Explicit suites only: delete `suite_for`; `rollout eval --environment` needs a made suite | 2.4 | `test_evals.py`, `test_suite_versions.py`, `monitor/test_launching.py` |
| 21 | Our own Tinker-to-PEFT key remap; drop tinker-cookbook and the transformers pin | 2.5 | rollout-tinker `tests/test_weights.py`, `test_engine.py`, the live test |
| 22 | rollout-tinker and rollout-verifiers join the workspace; Ray becomes a dependency; trainer settings share a base | 2.3, 3.3.5 | rollout-tinker tests, `tests/rollout_lora`, the verifiers tests |
| 23 | Delete the launcher's process backend (after the live GSM8K launcher moves to Ray); one shared job-client fake | 2.3 | `test_launches.py`, `test_evals.py`, `test_suite_versions.py`, `test_pausing.py`, `test_suite_entries.py`, `test_resharding.py`, `research/test_ledger_guarantees.py` |
| 24 | Reshard on Ray only: delete `ray = "auto"` and the in-process fallback | 2.3 | `test_resharding.py` |
| 25 | Run settings as one JSON document; one `whole()`; delete `_setting` and the sentinels | 3.3.1–4 | `test_settings.py`, `test_launches.py`, `monitor/test_launching.py` |
| 26 | Profiles become a cluster config, run settings and presets, and environment declarations: delete `profile.py`, `Platform`, `hosting.py`, the profile files, the profile lookups in launcher and resuming, and the front end's profile UI; add a deployment object | 2.2, 3.4, F6, T4 | rewritten `test_machines.py`, `test_full_weights.py`, `gateway/test_hosted.py`, `test_replicas.py`, `test_on_gpu.py`, `monitor/test_launching.py`, `test_launches.py`, `page.test.tsx` |
| 27 | The episode page reads transcripts from gateway turns only; delete the feed's `sample` path | 1.9 | `monitor/test_system.py`, `monitor/test_runs.py`, `monitor/test_monitor.py` |
| 28 | Split the engine protocol into a sampling core and capabilities | 1.10 | `inference/test_remote.py`, `test_full_weights.py`, rollout-tinker `test_engine.py`, the colocation tests |
| 29 | Delete orphan-process tracking once engines are Ray actors | 2.6 | `tests/rollout/test_processes.py` (trimmed), `tests/rollout_lora` |
| 30 | Decide the sandbox contract fields with the environment-declarations design | 1.11 | `tests/rollout/harness/test_sandboxes.py` |
| 31 | Rewrite the docs that describe profiles, optional Ray, tinker-cookbook and `suite_for` | 3.8 | `tests/test_docs.py`, `mkdocs build` |
| 32 | Split the large modules (pure moves): `monitor/system.py`, `cli.py`, `loop.train`, `evals.py`, `rollouts/scheduler.py`, Minecraft `tasks.py`, `types.ts`, `page.test.tsx` | 5.2 | the full suite, `npm test` |

---

## Appendix: a bug found on the way

In `RT/evals.py:632-635`, `whole = {**here, **described(first), ...}` lets the environment's `version` (from
`RT/record.py:92`) overwrite `here["version"] = suite.id`. Real data shows it: the GSM8K eval's start record has
`version: "gsm8k 0.1.4, verifiers 0.3.2.dev185, settings 65525de5"`.

The monitor reads `starts.version` as the environment's version, so the overwrite looks intended there. But
`RT/resuming.py:155` reads it as the suite version. Resuming a one-entry eval with no earlier launch would pass the
environment's version string to `suite_of`, get `None`, and ask a launch with a bogus suite. The suite version and the
environment's version need keys of their own.
