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
  reward?: number | null;
  updated?: number | null;
  in_feed?: boolean;
  interrupted?: boolean;
  outcome?: string;
  solved?: boolean;
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
  solved: boolean[];
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
  state: "running" | "idle" | "ended";
  host: string | null;
  address: string | null;
  directory: string | null;
  episodes_at: string | null;
  reached: boolean | null;
  profile: string | null;
  /** The checkpoint it started from (none: the base model). */
  from: string | null;
  started: number | null;
  starts: number;
  written: number | null;
  played?: Played;
  /** What it is: a training run (`run`), or an eval playing a suite (`eval`). */
  kind?: "run" | "eval";
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
  base: string | null;
  run: string | null;
  step: number | null;
  made: number;
  metrics: Metrics;
  weights: { files: number; bytes: number } | null;
  state: { files: number; bytes: number } | null;
  released: number | boolean | null;
  batch: boolean;
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

/** A channel as a heartbeat says it: what it serves, and what passed through it since the beat before. */
export interface BeatChannel {
  channel: string;
  adapter?: string | null;
  version?: number | null;
  requests?: number;
  generated_tokens?: number;
  tokens_per_second?: number;
  tokens_per_second_per_stream?: number;
  mean_concurrency?: number;
}

/** What a runner's (or launcher's) earlier beats measured. */
export interface Beaten {
  at: number;
  machine?: Measurement;
  channels?: BeatChannel[];
  playing?: number;
}

/** A profile a launcher can run: by its name, with the settings a launch may change and their values in it. */
export interface OfferedProfile {
  profile: string;
  path: string;
  model: string;
  settings: Record<string, unknown>;
}

/** A runner's or a launcher's machine, as its heartbeats say: alive or not, what it said last, its recent beats. */
export interface Machine {
  runner: string;
  at: number;
  alive: boolean;
  /** `launcher` for a launcher; nothing for an episode runner. */
  kind?: string;
  host?: string;
  run?: string;
  directory?: string;
  machine?: Measurement;
  processes?: { owner: number; started: { pid: number; name: string; alive: boolean }[] } | null;
  channels?: BeatChannel[];
  places?: number;
  playing?: number;
  at_once?: number;
  profiles?: OfferedProfile[];
  catalogs?: string[];
  history: Beaten[];
}

export interface Machines {
  machines: Machine[];
}

/** What a run is asked to be (`rollout_train.launches.Asked`). */
export interface LaunchAsked {
  profile: string;
  catalog: string;
  name: string;
  start?: string | null;
  bookmark?: string | null;
  groups?: number;
  groups_per_step?: number;
  seed?: number;
  settings?: Record<string, unknown>;
  /** A training run (`run`, the default) or an eval (`eval`): one suite played by `start` (none: the base model). */
  kind?: "run" | "eval";
  suite?: string | null;
  /** An eval's episodes of each start. */
  episodes?: number;
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
  catalogs: string[];
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
  ended: (Record<string, unknown> & { reward?: number; solved?: boolean; sampled?: number; info?: Record<string, unknown> }) | null;
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
  solved: boolean[];
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
  waiting: number;
}

export interface LineageCheckpoint {
  id: string;
  short: string;
  depth: number;
  parents: string[];
  base: string | null;
  made: number;
  kept: boolean;
  released: number | boolean | null;
  bookmarks: string[];
  metrics: Metrics;
  by: { run: string; name: string; kind: string; step: number | null } | null;
  life: Life;
  sample?: boolean;
}

export interface LineageRun {
  run: string;
  name: string;
  kind: string;
  from: string | null;
  teachers: string[];
  data: { sampled_by?: string[]; runs?: string[]; episodes?: number | string };
  objective: string | null;
  mode: string | null;
  /** What its mode means, in words (a distillation's). */
  says: string | null;
  checkpoints: string[];
  latest: string | null;
  sample: boolean;
}

export interface QueueEntry {
  run: string;
  step: number;
  makes: string;
  state: "taking" | "queued" | "made" | "failed" | string;
  queued: number | null;
  began: number | null;
}

export interface Trainer {
  trainer: string;
  weights: string;
  base: string | null;
  runs?: string[];
  colocated?: boolean;
  where?: string;
  implicit: boolean;
  queue: QueueEntry[];
  depth: [number, number, number][];
  groups?: [number, number][];
  sample: boolean;
}

export interface Worker {
  worker: string;
  registered: boolean;
  serving: string[];
  holds?: { run?: string; base?: string };
  adapters?: number;
  share?: string;
  machine?: string;
  accelerators?: string;
  sample: boolean;
}

export interface Suite {
  suite: string;
  starts: { start: string; task: string; seed: number | string; title?: string }[];
  subjects: {
    subject: string;
    kind: string;
    checkpoint?: string;
    model?: string;
    asked_by?: string;
    episodes?: number;
    played: number;
    solved: number;
    reward?: number | null;
    results: Record<string, { solved: boolean; reward: number; run_id?: string }[]>;
  }[];
  sample: boolean;
}

/** A suite as the Evals page has it: a suite, with the catalog it was drawn from and when it was made. */
export interface EvalSuite extends Suite {
  catalog: string | null;
  made: number | null;
}

/** An eval: its run, the suite it plays, the checkpoint that plays it (none: the base model), and how far it got. */
export interface EvalRun {
  run: string;
  name: string;
  suite: string;
  checkpoint: string | null;
  started: number | null;
  played: number;
  expected: number;
  solved: number;
  done: boolean;
}

export interface Evals {
  suites: EvalSuite[];
  evals: EvalRun[];
}

/** What came from what: `base` (from `base:MODEL` to a line's first checkpoint), `trained` (from a checkpoint to one trained
 * from it), `learned` (to one that learned from it beside), `teach` and `start` (to a distillation, by its run). */
export interface Edge {
  kind: "base" | "trained" | "learned" | "teach" | "start";
  from: string;
  to: string;
  mode?: string;
  says?: string;
}

export interface Lineage {
  now: number;
  sample: boolean;
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
  routing: { waiting: Record<string, number>; history: [number, number][] };
  evaluations: Suite[];
}
