// What the monitor's endpoints answer (rollout_train/monitor/system.py, statistics.py, lineage.py).

export type Metrics = Record<string, number | undefined>;

export interface Step {
  step: number;
  groups: number[];
  skipped: number[];
  makes: string | null;
  parent: string | null;
  segments: number | null;
  decided: number | null;
  state: "stepping" | "committed" | "failed" | string;
  error: string | null;
}

export interface Claim {
  group?: number;
  episode: string;
  attempt: number;
  runner: string;
  at: number | null;
}

/** An episode as a group lists it: from the feed while it runs, from the ledger once it ended. */
export interface GroupEpisode {
  run_id: string;
  episode?: string | null;
  state?: string;
  samples?: number | null;
  slots?: string[];
  /** The episode's reward: the mean of its slots' (while it plays, as far as they are assigned). */
  reward?: number | null;
  /** While it plays: each slot's reward so far. */
  rewards?: Record<string, number>;
  updated?: number | null;
  interrupted?: boolean;
  outcome?: string;
  /** Whether it solved its task; null: the task did not say. */
  solved?: boolean | null;
  detail?: string | null;
  sampled?: number;
  info?: Record<string, unknown>;
  labels?: Record<string, string>;
}

/** A group in flight. */
export interface OpenGroup {
  number: number;
  task: string | null;
  title: string | null;
  decided: number | null;
  stage: "waiting" | "playing" | "ended" | "done";
  count: number | null;
  ended: number;
  playing: Claim[];
  episodes: GroupEpisode[];
  step: (Step & Record<string, unknown>) | null;
  error: string | null;
}

/** A group that is done with: its result, and what was done with it. */
export interface DoneLine {
  group: number;
  time: number;
  task: string;
  title: string;
  rollout_seconds: number;
  rewards: number[];
  /** Each episode's; null where none of the group's episodes said whether it solved its task. */
  solved: (boolean | null)[];
  durations: (number | null)[];
  failed: number;
  failures: string[];
  segments_recorded: number;
  segments: number;
  skipped: string | null;
  unlocked: number;
  adapter: string | null;
  step: number | null;
  step_state: string | null;
  depth: number | null;
  update: Metrics | null;
  segments_trained: number;
  error: string | null;
  seconds: number | null;
  episodes?: GroupEpisode[];
}

export interface Throughput {
  at: number;
  tokens_per_second?: number;
  mean_concurrency?: number;
  tokens_per_second_per_stream?: number;
}

export interface Channel {
  channel: string;
  adapter?: string | null;
  version?: number | null;
  published?: number | null;
  throughput: Throughput[];
  /** The run whose engines serve it (as its runners' heartbeats say). */
  run?: string;
}

export interface Played {
  episodes: number;
  outcomes: Record<string, number>;
  playing: number;
  interrupted: number;
  sampled: number;
}

export interface Run {
  /** The run's id: everything kept of it is under it, and it never changes. */
  run: string;
  /** What the run is called (its id, if it was never named). */
  name?: string | null;
  fence: number | null;
  wrote: number | null;
  decided: number;
  open: OpenGroup[];
  done: DoneLine[];
  steps: Step[];
  next: number[];
  channels: Channel[];
  state: RunState;
  ending?: { how: "finished" | "stopped" | "failed"; at: number; detail?: string | null };
  host: string | null;
  address: string | null;
  directory: string | null;
  episodes_at: string | null;
  reached: boolean | null;
  /** The checkpoint it started from (none: the base model). */
  from: string | null;
  started: number | null;
  starts: number;
  written: number | null;
  played?: Played;
  /** What it is: a training run (`run`), or an eval playing a suite (`eval`). */
  kind?: "run" | "eval" | string;
  /** For an eval a training run's schedule asked for: that run, and the step whose checkpoint it plays. */
  by?: string | null;
  by_step?: number | null;
  /** For a run that plays one entry of an eval of several environments: that eval's run. */
  part_of?: string | null;
  /** Whether it is wanted paused (its state says paused once its process has paused it). */
  pause?: boolean;
}

/** A checkpoint: where it came from, what made it, and what is kept of it. */
export interface Checkpoint {
  id: string;
  /** The shortest start of its id that no other checkpoint's has. */
  short: string;
  /** Steps from its base model along its first parents. */
  depth: number;
  /** What it was trained from, then anything it learned from beside; none: from the base model. */
  parents: string[];
  /** What its weights build on: a model, by name; or, for an adapter over a full checkpoint, that checkpoint's id. */
  base: string | null;
  /** What its weights are: an adapter over its base (`lora`), or all of a model's weights (`full`). */
  kind: "lora" | "full" | string;
  run: string | null;
  step: number | null;
  made: number;
  metrics: Metrics;
  weights: { files: number; bytes: number } | null;
  state: { files: number; bytes: number } | null;
  released: number | boolean | null;
  bookmarks: string[];
}

export interface Runner {
  runner: string;
  fence: number | null;
  playing: (Claim & { run: string })[];
  claims: number;
  last: number | null;
}

export interface System {
  at: number;
  name: string;
  directory: string | null;
  ledger_at: string;
  host: string;
  written: number | null;
  runs: Run[];
  checkpoints: Checkpoint[];
  /** Each bookmark's checkpoint, by name. */
  bookmarks: Record<string, string>;
  runners: Runner[];
  channels: Channel[];
  ledger: { fences: Record<string, number>; tables: Record<string, number> };
  kept: { checkpoints: number; episodes: number };
  /** Every registered run's name, by id, and every bookmark's checkpoint, by name. */
  names?: { runs: Record<string, string>; bookmarks: Record<string, string> };
}

export interface Entry {
  id: string;
  name: string;
  created: number;
}

export interface Bookmark {
  name: string;
  checkpoint: string;
  moved: number;
}

export interface Measurement {
  at: number;
  memory: { available: number | null; total: number | null };
  accelerators: { name: string; used: number; total: number; busy: number }[];
  disk: { free: number; total: number } | null;
}

/** A profile a launcher can run: by its name, with the settings a launch may change and their values in it. */
export interface OfferedProfile {
  profile: string;
  path: string;
  /** What it launches: `run` and `eval` with a trainer, `eval` alone without. */
  kinds?: ("run" | "eval")[];
  model: string;
  /** The base models an eval may play with it: its channel's model first, then those the cluster's inference providers
   * of its engine's kind serve. */
  models?: string[];
  /** What its trainer makes: `lora` (adapters) or `full` weights; none where the launcher cannot tell. */
  weights?: string | null;
  /** The published environments it plays: those whose every kind of sandbox it has a pool of. */
  published?: string[];
  settings: Record<string, unknown>;
}

/** What every role on a machine shares: its name among the beats, its machine, whether it beats (within 90 s by the
 * store's clock) and when it last did, by the monitor's clock. */
export interface Role {
  name: string;
  host: string | null;
  alive: boolean;
  at: number | null;
}

/** A channel as a beat says it: what it serves, and what passed through it since the beat before. */
export interface RoleChannel {
  channel: string;
  serving: string | null;
  version: number | null;
  tokens_per_second: number | null;
  mean_concurrency: number | null;
  /** A routed channel's servers: each one's address, what it would sample from now, and how far behind. */
  servers?: { address: string; serving: string | null; version: number | null; behind: number | null; answers?: boolean }[];
}

/** An episode a runner's claim holds. */
export interface Claimed {
  run: string;
  group: number;
  episode: number;
  attempt: number;
  at: number | null;
  run_id: string | null;
}

export interface RunnerRole extends Role {
  /** The run it serves, where it serves one. */
  run: string | null;
  places: number;
  playing: number;
  free: number;
  claims: Claimed[];
  /** Its pools, by name. */
  pools: string[];
  channels: RoleChannel[];
}

/** A sandbox held under a key, as the `sandboxes` table says. */
export interface LeaseHeld {
  key: string;
  kind: string;
  sandbox: string;
  /** The run's episode it was acquired for, where its key names one. */
  run: string | null;
  group?: number;
  episode?: number;
  attempt?: number;
  run_id?: string | null;
  /** Whether that claim holds; none where its key names no claim the ledger has. */
  holds: boolean | null;
  /** When it was made, by the pool's clock. */
  at: number;
  /** How long it may last. */
  seconds: number | null;
  /** Its sandbox is gone. */
  lost: boolean;
}

export interface PoolRole extends Role {
  /** The kind of sandbox it makes. */
  kind: string;
  /** The runner it is within, if it is; else it beats on a machine of its own (or not at all). */
  runner: string | null;
  size: number | null;
  leased: number | null;
  free: number | null;
  leases: LeaseHeld[];
}

/** One of an engine host's engines: its address, what it serves, how far behind what its run wants. */
export interface EngineServed {
  address: string | null;
  serving: string | null;
  version: number | null;
  behind: number | null;
}

export interface EngineChannel extends RoleChannel {
  wanted: string | null;
  behind: number | null;
  error: string | null;
  engines: EngineServed[];
}

export interface EngineRole extends Role {
  follows: string | null;
  channels: EngineChannel[];
}

/** A launch its launcher is playing. */
export interface LaunchGoing {
  id: string;
  state: string;
  at: number;
  updated: number;
  name: string;
  kind: string;
  profile: string;
  environment: string;
  suite: string | null;
}

export interface LauncherRole extends Role {
  profiles: { profile: string; model: string; weights: string | null }[];
  environments: string[];
  launches: LaunchGoing[];
  at_once: number | null;
  playing: number | null;
  backend: string | null;
}

export interface GatewayRole extends Role {
  listen: string | null;
  channels: RoleChannel[];
}

export type RoleKind = "runners" | "pools" | "engines" | "launchers" | "gateways";

/** A machine: alive while any of its roles is, its newest measurements and their history, and its roles. */
export interface Host {
  host: string;
  alive: boolean;
  at: number | null;
  machine: Measurement | null;
  history: { at: number; machine: Measurement }[];
  roles: { kind: RoleKind; name: string; alive: boolean }[];
}

/** Every machine that beats and the roles on it (`/api/machines`). */
export interface Machines {
  now: number;
  hosts: Host[];
  runners: RunnerRole[];
  pools: PoolRole[];
  engines: EngineRole[];
  launchers: LauncherRole[];
  gateways: GatewayRole[];
}

/** What a run is asked to be (`rollout_train.launches.Asked`). */
export interface LaunchAsked {
  profile: string;
  environment: string;
  name: string;
  start?: string | null;
  bookmark?: string | null;
  groups?: number;
  groups_per_step?: number;
  seed?: number;
  settings?: Record<string, unknown>;
  /** A training run (`run`, the default) or an eval (`eval`): one suite played by `start` (none: the base model). */
  kind?: "run" | "eval";
  /** An eval's suite: by name (the version its name points to), or a version by id (`NAME@N`). */
  suite?: string | null;
  /** An eval's episodes of each start; none: the suite's own. */
  episodes?: number | null;
  /** The base model that plays an eval no checkpoint plays; none: the profile's. */
  model?: string | null;
  /** Every other environment it plays: an eval's suite's, a training run's evals' suite's. */
  environments?: string[];
  /** The run it starts again, for a launch that resumes one, and that run's directory, where it runs. */
  resumes?: string | null;
  directory?: string | null;
}

export type LaunchState = "asked" | "claimed" | "running" | "stopping" | "ended" | "failed" | "stopped";

export interface Launch {
  id: string;
  asked: LaunchAsked;
  at: number;
  state: LaunchState;
  launcher: string | null;
  directory: string | null;
  pid: number | null;
  detail: string | null;
  updated: number;
}

/** A launcher alive: what it offers, and whether it has room. */
export interface Launcher {
  launcher: string;
  at: number;
  host?: string;
  profiles: OfferedProfile[];
  environments: string[];
  at_once: number;
  playing: number;
}

export interface Launches {
  launches: Launch[];
  launchers: Launcher[];
}

/** An episode in a feed, summarised. */
export interface FeedRun {
  run_id: string;
  labels: Record<string, string>;
  state: string;
  started: number;
  updated: number;
  rewards: Record<string, number>;
  slots: string[];
  samples: number;
  lines: number;
}

export interface Group extends OpenGroup {
  run: string;
  episodes_at: string | null;
  parameters: Record<string, unknown> | null;
  outcome: DoneLine | null;
  result: DoneLine | null;
  checkpoint: Checkpoint | null;
}

export interface Call {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface PlainMessage {
  role: string;
  text: string;
  reasoning: string;
  calls: Call[];
  results: { id: string; text: string; error: boolean }[];
}

export interface SampleLine {
  kind: "sample";
  slot: string;
  effect_id?: string;
  at: number;
  seconds: number;
  messages: PlainMessage[];
  tools: string[];
  reply: { text: string; reasoning: string; calls: Call[] };
  finish_reason?: string | null;
}

export interface EventLine {
  kind: "event";
  seq: number;
  type: string;
  at: number;
  payload: any;
}

export type Line = SampleLine | EventLine;

export interface Episode {
  run_id: string;
  labels: Record<string, string>;
  state: string | null;
  ended: (Record<string, unknown> & { reward?: number; solved?: boolean | null; sampled?: number; untrained?: string[]; info?: Record<string, unknown> }) | null;
  source: "feed" | "archive" | null;
  lines: Line[];
}

// Statistics
export interface StatisticsGroup {
  group: number;
  task: string | null;
  title: string | null;
  decided: number | null;
  time: number | null;
  rewards: number[];
  solved: (boolean | null)[];
  failed: number;
  segments: number;
  skipped: string | null;
  trained: string | null;
  names: number | null;
}

export interface StatisticsStep {
  step: number;
  checkpoint: string | null;
  decided: number | null;
  made: number | null;
  state: string;
  groups: number;
  segments: number | null;
  metrics: Metrics;
}

export interface StatisticsRun {
  run: string;
  wrote: number | null;
  steps: StatisticsStep[];
  groups: StatisticsGroup[];
  flight: [number, number, number][];
  /** Each channel's measurements: [time, tokens a second, requests at once, tokens generated, requests]. */
  inference: { channel: string; every: number; points: number[][] }[];
}

export interface Statistics {
  now: number;
  runs: StatisticsRun[];
  names?: { runs: Record<string, string> };
}

// The checkpoints as a graph
export interface Life {
  state: string;
  reshard: boolean;
  resharding: number | null;
  resharded: number | null;
  latest_of: string | null;
  workers: Record<string, { since: number; until: number | null }>;
}

export interface LineageCheckpoint {
  id: string;
  short: string;
  depth: number;
  parents: string[];
  base: string | null;
  kind?: string;
  made: number;
  kept: boolean;
  released: number | boolean | null;
  bookmarks: string[];
  metrics: Metrics;
  by: { run: string; name: string; step: number | null } | null;
  life: Life;
}

export interface LineageRun {
  run: string;
  name: string;
  from: string | null;
  checkpoints: string[];
  latest: string | null;
}

export interface QueueEntry {
  run: string;
  step: number;
  makes: string;
  state: "taking" | "queued" | "made" | "failed" | string;
  queued: number | null;
  began: number | null;
}

/** A run's own trainer, whose queue is the run's steps. */
export interface Trainer {
  trainer: string;
  /** What its steps make (`lora` or `full`); none: it has made nothing yet. */
  weights: string | null;
  base: string | null;
  runs: string[];
  colocated: boolean;
  queue: QueueEntry[];
  depth: [number, number, number][];
  groups: [number, number][];
}

/** A run's engines, and the checkpoints they serve now. */
export interface Worker {
  worker: string;
  serving: string[];
}

/** A start of a suite's version: its number in the version, its entry's environment, its row and seed, and what it is in
 * every version that has it (`identity`). */
export interface SuiteStart {
  start: string;
  environment?: string | null;
  task: string;
  seed: number | string;
  title?: string;
  identity?: string;
}

/** How a version's starts were chosen. */
export type Chosen = "eval data" | "rows and seeds" | "starts";

/** One environment of a suite's version: its starts (how many, and how many of the version's come before its first:
 * `offset`), how they were chosen, its episodes of each start and its sampling limits. */
export interface SuiteEntry {
  environment: string | null;
  environment_version: string | null;
  chosen: Chosen;
  eval_data: string | null;
  rows: string[] | null;
  seeds: (number | string)[] | null;
  held_out: boolean;
  episodes: number;
  thinking_tokens: number | null;
  answer_tokens: number | null;
  offset: number;
  starts: number;
}

/** One version of a suite: an eval configuration of one or more environments (its entries), never changed. */
export interface SuiteVersion {
  id: string;
  number: number;
  environments: (string | null)[];
  made: number | null;
  held_out: boolean;
  entries: SuiteEntry[];
  starts: SuiteStart[];
}

/** How a subject did at one entry of the version it played. */
export interface EntryScore {
  environment: string | null;
  /** Its episodes of each start of the entry. */
  episodes?: number;
  played: number;
  /** None: none of its episodes said whether it solved its start. */
  solved: number | null;
  reward: number | null;
}

export interface Suite {
  suite: string;
  /** The version its name points to, by id, and its number. */
  version?: string;
  number?: number;
  /** That version's starts. */
  starts: SuiteStart[];
  /** Every version, oldest first. */
  versions?: SuiteVersion[];
  subjects: {
    subject: string;
    kind: string;
    checkpoint?: string;
    model?: string;
    asked_by?: string;
    /** Its episodes of each start; none where its entries play different numbers. */
    episodes?: number | null;
    /** The version it played, by id, and how many starts that version has. */
    version?: string;
    starts?: number;
    played: number;
    /** None: none of its episodes said whether it solved its start. */
    solved: number | null;
    reward?: number | null;
    results: Record<string, { solved: boolean | null; reward: number; run_id?: string }[]>;
    /** How it did at each entry of the version it played. */
    entries?: EntryScore[];
  }[];
}

/** A suite as the Evals page has it: a suite, with the environments its newest version plays and when it was made. */
export interface EvalSuite extends Suite {
  environments: (string | null)[];
  made: number | null;
}

/** An eval: its run, the suite it plays, the checkpoint that plays it (none: the base model), and how far it got. */
export interface EvalRun {
  run: string;
  name: string;
  suite: string;
  /** The version it played, by id. */
  version?: string;
  checkpoint: string | null;
  started: number | null;
  played: number;
  expected: number;
  solved: number | null;
  done: boolean;
  /** For a suite of several environments, how it did at each (the share solved, where its episodes say). */
  entries?: (EntryScore & { share: number | null })[];
}

export interface Evals {
  suites: EvalSuite[];
  evals: EvalRun[];
}

/** What came from what: `base` (from `base:MODEL` to a line's first checkpoint), `trained` (from a checkpoint to one trained
 * from it), `learned` (to one that learned from it beside). */
export interface Edge {
  kind: "base" | "trained" | "learned";
  from: string;
  to: string;
}

export interface Lineage {
  now: number;
  /** The base models every line grows from. */
  bases: string[];
  checkpoints: LineageCheckpoint[];
  /** Checkpoints something here starts from that this ledger does not have. */
  outside: string[];
  bookmarks: Record<string, string>;
  runs: LineageRun[];
  edges: Edge[];
  trainers: Trainer[];
  workers: Worker[];
}

/** An eval a checkpoint had: the suite, the eval's run, who asked for it (by hand, or a training run's schedule at a
 * step), how far it got and its score. */
export interface CheckpointEval {
  suite: string;
  /** The version it played, by id. */
  version?: string;
  run: string;
  name: string;
  kind: string;
  checkpoint: string | null;
  model: string | null;
  asked_by: "by hand" | "schedule";
  by: string | null;
  step: number | null;
  /** Episodes of each start. */
  episodes: number;
  played: number;
  expected: number;
  /** Episodes solved; none where none of them said whether it solved its start. */
  solved: number | null;
  /** The share solved of those that said. */
  share: number | null;
  reward: number | null;
  started: number | null;
  at: number | null;
  done: boolean;
  /** Its score at each entry of the version it played. */
  entries?: (EntryScore & { share: number | null })[];
}

export interface CheckpointEvals {
  checkpoint: string;
  evals: CheckpointEval[];
}

/** What an eval played: a checkpoint (by id), or a base model (by name). */
export type SubjectKind = "checkpoint" | "model";

/** A subject that has had an eval: what it is, and its evals. */
export interface EvalSubject {
  kind: SubjectKind;
  /** The checkpoint's id, or the base model's name. */
  id: string;
  /** The checkpoint's shortest id, or the base model's name. */
  short: string;
  /** The run and step that made the checkpoint, the run's name, and what it builds on. */
  run: string | null;
  name: string | null;
  step: number | null;
  base: string | null;
  bookmarks: string[];
  /** Its evals, by run, newest first. */
  evals: string[];
  suites: string[];
  /** When its newest eval began. */
  started: number | null;
  /** Its evals still playing. */
  playing: number;
}

export interface EvalSubjects {
  subjects: EvalSubject[];
}

/** A subject's history: every eval it has had, newest first. */
export interface SubjectHistory {
  subject: EvalSubject;
  evals: CheckpointEval[];
}

/** A point's score at a suite, over every eval of it there; and at each of its entries (by environment). */
export interface PathScore {
  reward: number | null;
  /** The share solved; none where its episodes do not say. */
  solved: number | null;
  played: number;
  evals: string[];
  entries?: Record<string, { reward: number | null; solved: number | null; played: number }>;
}

/** A point on a checkpoint's line: the base model (depth 0, no id), then each checkpoint along first parents. */
export interface PathPoint {
  id: string | null;
  short: string;
  depth: number;
  model: string | null;
  run: string | null;
  name: string | null;
  step: number | null;
  kind: "model" | "lora" | "full" | string;
  bookmarks: string[];
  scores: Record<string, PathScore>;
}

export interface Path {
  checkpoint: string;
  points: PathPoint[];
  /** Each version any point was evaluated on: `suite` is its id (what `scores` are keyed by), `label` how it is said. */
  suites: { suite: string; environments: string[]; name?: string; number?: number; label?: string }[];
}

/** One finding of a check: which check, whether it passed, what it said, and whether it passed with something to look
 * at. */
export interface Finding {
  check: string;
  passed: boolean;
  said: string;
  flagged?: boolean;
}

/** Where a published environment's source came from and what its check found. */
export interface PublishedSource {
  /** The version's id: the SHA-256 of its source's zip. */
  version: string;
  source: string;
  ref: string | null;
  commit: string;
  subdirectory: string;
  entry_point: string;
  imported: number;
  check: Finding[];
  passed: boolean;
  dependencies: string[];
}

/** A version of an environment imported from git, as the ledger keeps it. */
export interface EnvironmentVersion {
  /** `NAME@VERSION`: how launches and runs name it. */
  reference: string;
  name: string;
  version: string;
  source: string;
  ref: string | null;
  commit: string;
  subdirectory: string;
  entry_point: string;
  check: Finding[];
  imported: number;
}

/** What an import from git asks for. */
export interface ImportAsked {
  url: string;
  ref: string;
  subdirectory: string;
  entry_point: string;
}

/** An import the monitor made, with where it is: `fetching`, `reading`, `packing`, `storing`, `checking`, `recording`,
 * then `done` (with the version it made) or `refused` (with why). */
export interface ImportGoing extends ImportAsked {
  id: string;
  stage: string;
  started: number;
  ended: number | null;
  error: string | null;
  version: string | null;
}

export interface Imports {
  imports: ImportGoing[];
  /** Whether this monitor imports at all (it was started with a cluster config). */
  importing: boolean;
}

/** An environment the system knows of: offered by a launcher alive, started on by a run, played by a suite, or
 * imported from git. */
export interface KnownEnvironment {
  /** As `module:name`; a published one as `NAME@VERSION`. */
  environment: string;
  /** In a word. */
  name: string;
  /** Its versions seen, in runs' starts and suites' entries. */
  versions: string[];
  /** Whether a launcher alive offers it. */
  offered: boolean;
  /** Its training runs (by id), and the suites with a version that plays it. */
  runs?: string[];
  suites?: string[];
  /** When a run (a training run, an eval or a check) last started on it. */
  used?: number | null;
  /** For one imported from git: where its source came from and what its check found. */
  published?: PublishedSource | null;
}

/** What its results say of an episode: the range its reward falls in, whether it says solved and saturated, and what
 * its duration counts. */
export interface Description {
  rewards: [number | null, number | null];
  solved: boolean;
  saturated: boolean;
  duration: string | null;
  observations: string | null;
}

/** One of an environment's rows, with what the training runs on it played of it. */
export interface EnvironmentRow {
  key: string;
  title: string;
  /** False for a row only its eval data has: training never plays it. */
  trains: boolean;
  groups: number;
  /** Episodes played, failed ones too. */
  played: number;
  /** Of `said` (the episodes fit to train on in runs whose results say), how many solved; none where no run says. */
  solved: number | null;
  said: number;
  reward: number | null;
  /** Its starts in the eval data. */
  held: number;
}

/** A training run on an environment, and what it played. */
export interface EnvironmentRun {
  run: string;
  name: string;
  version: string | null;
  started: number | null;
  groups: number;
  played: number;
  solved: number | null;
  said: number;
}

/** An eval of a version of a suite that plays an environment, and its score at that environment's entry. */
export interface EnvironmentScore {
  run: string;
  name: string;
  suite: string;
  version: string;
  kind: "checkpoint" | "model" | string;
  checkpoint: string | null;
  model: string | null;
  started: number | null;
  done: boolean;
  played: number;
  solved: number | null;
  said: number;
  share: number | null;
  reward: number | null;
}

/** A group a check played: its row, its rewards, and whether every episode scored the same (`flagged`). */
export interface CheckGroup {
  group: number;
  task: string;
  episodes: number | null;
  /** None: not played to its end yet. */
  rewards: number[] | null;
  solved: boolean[] | null;
  failed: number;
  failures: string[];
  flagged: boolean;
  skipped: string | null;
}

/** An environment: what it says of itself where it loads on the monitor's machine, and what was done with it. */
export interface EnvironmentInfo {
  environment: string;
  name: string;
  offered: boolean;
  loads: boolean;
  /** Why it does not load, where it does not. */
  error: string | null;
  /** As it loads; none where it does not. */
  version: string | null;
  /** Seen in runs' starts and suites' entries. */
  versions: string[];
  description: Description | null;
  curriculum: { name: string; own: boolean; start: number | null; reach: number | null } | null;
  rows: EnvironmentRow[];
  /** Each list of eval data, and how many starts it has. */
  evals: Record<string, number>;
  runs: EnvironmentRun[];
  suites: { suite: string; version: string; current: boolean; versions: { id: string; number: number; starts: number; made: number }[] }[];
  scores: EnvironmentScore[];
  check: {
    run: string;
    name: string;
    started: number | null;
    version: string | null;
    ended: { how: string; at: number; detail?: string | null } | null;
    groups: CheckGroup[];
  } | null;
  /** For one imported from git: where its source came from and what its check found. */
  published?: PublishedSource | null;
}

/** A training run's settings, by dotted key: fixed ones, changeable ones as it started, those its newest step used,
 * what is wanted of them, and each step that used other settings than the one before. */
export interface RunSettings {
  run: string;
  kind: string;
  /** The environment it trains on, as `module:name` (its evals' suites are that environment's). */
  environment?: string | null;
  fixed: Record<string, unknown>;
  changeable: Record<string, unknown>;
  now: Record<string, unknown>;
  desired: Record<string, unknown>;
  changed: number | null;
  changes: { step: number; changed: Record<string, unknown> }[];
}

/** Running, idle or paused while its process beats; how it ended, once it said; lost if it stopped beating without
 * saying; ended if it never beat. */
export type RunState = "running" | "idle" | "paused" | "finished" | "stopped" | "failed" | "lost" | "ended";
