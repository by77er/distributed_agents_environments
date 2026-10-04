// What the views work out from a run: its groups by number, the step a group went into, what was done with a group,
// and the episodes a group asked for that have not started.

import type { DoneLine, GroupEpisode, OpenGroup, Run, Step, Checkpoint } from "../api/types";
import { byNumber, mean } from "./format";

export interface GroupEntry {
  number: number;
  task: string | null;
  title: string | null;
  line?: DoneLine;
  open?: OpenGroup;
  episodes: GroupEpisode[];
}

const groupCache = new WeakMap<Run, Map<number, GroupEntry>>();

/** A run's groups by number: those in flight and those done with. */
export function groupsOf(run: Run): Map<number, GroupEntry> {
  let found = groupCache.get(run);
  if (!found) {
    found = new Map<number, GroupEntry>([
      ...run.done.map(line => [line.group, { number: line.group, task: line.task, title: line.title, line, episodes: line.episodes ?? [] }] as const),
      ...run.open.map(group => [group.number, { number: group.number, task: group.task, title: group.title, open: group, episodes: group.episodes }] as const),
    ]);
    groupCache.set(run, found);
  }
  return found;
}

export const stepOf = (run: Run, number: number): Step | undefined =>
  run.steps.find(step => step.groups.includes(number) || step.skipped.includes(number));

/** Groups by number, as briefly as they go: `#3–6`, or `#3 #5`. */
export const range = (numbers: number[]): string =>
  numbers.length
    ? numbers.length > 1 && numbers.at(-1)! - numbers[0] === numbers.length - 1
      ? `#${numbers[0]}–${numbers.at(-1)}`
      : numbers.map(number => `#${number}`).join(" ")
    : "no group";

const KINDS: Record<string, string> = {
  running: "accent", completed: "good", done: "good", failed: "bad", cancelled: "bad", playing: "accent", queued: "warm",
  stepping: "violet", committed: "good", asked: "warm", claimed: "accent", stopping: "warm", ended: "good", stopped: "",
};
/** The color a state is drawn in. */
export const stateKind = (name: string | null | undefined): string => (name ? KINDS[name] ?? "" : "");

/** What was done with a group that is done with, in words, and the color it is said in. */
export function outcomeOf(line: DoneLine): { kind: string; text: string } {
  if (line.update) {
    const moved = Number(line.update.kl_moved ?? 0).toFixed(4);
    return { kind: "moved", text: `step ${line.step} · trained ${line.segments_trained} of ${line.segments_recorded} · moved ${moved}` };
  }
  if (line.error) return { kind: "bad", text: `step ${line.step} failed: ${line.error}` };
  if (line.step_state === "stepping") return { kind: "violet", text: `in step ${line.step}, being taken` };
  if (line.segments) return { kind: "warm", text: "waits for the next step" };
  return { kind: line.failed && !line.rewards.length ? "bad" : "still", text: line.skipped ?? "" };
}

export interface Waiting {
  episode: string;
  waiting: true;
}
export type Asked = GroupEpisode | Waiting;
export const isWaiting = (each: Asked): each is Waiting => "waiting" in each && each.waiting === true;

/** A group's episodes, and a place held for each one asked for that has not started (they start as there is room,
 * whatever group they are of). */
export function asked(group: { episodes: GroupEpisode[]; count: number | null }): Asked[] {
  const have = group.episodes.filter(each => !each.interrupted);
  const numbers = new Set(have.map(each => String(each.episode)));
  const waiting: Waiting[] = [];
  for (let number = 1; waiting.length < (group.count ?? 0) - have.length; number++) {
    if (!numbers.has(String(number))) waiting.push({ episode: String(number), waiting: true });
  }
  return [...group.episodes, ...waiting];
}

/** The square an episode is drawn as: playing, solved, not solved, failed, or not started. */
export function episodeClass(each: Asked): string {
  if (isWaiting(each)) return "waiting";
  if (each.interrupted) return "";
  const ended = each.outcome ?? (each.state && each.state !== "running" ? each.state : null);
  return !ended ? "running" : each.solved ? "solved" : ended !== "completed" ? "failed" : each.solved === false ? "unsolved" : "played";
}

/** A playing episode's reward so far, as an ended one's is: the mean of its slots' (none before any is assigned). */
export const episodeReward = (rewards: Record<string, number>): number | null => mean(Object.values(rewards));

/** Each slot's reward, by slot in their numbers' order, where they differ (none where the slots are rewarded together). */
export const slotRewards = (rewards: Record<string, number> | undefined): [string, number][] => {
  const entries = Object.entries(rewards ?? {}).sort(([a], [b]) => byNumber(a, b));
  return new Set(entries.map(([, value]) => value)).size > 1 ? entries : [];
};

/** Whether any of these says if its episode solved its task: a task may never say, and then nothing of solving is shown. */
export const reported = (solved: (boolean | null | undefined)[]): boolean => solved.some(each => each != null);

/** What a run is called: its name, or its id. */
export const nameOf = (run: { run: string; name?: string | null }): string => run.name || run.run;

/** What the page knows of runs and checkpoints, to say them: a run by its name, and a checkpoint by where it came from
 * (the run that made it and its step) and the shortest start of its id that no other has. */
export interface Known {
  run: (id: string | null | undefined) => string;
  checkpoint: (id: string | null | undefined) => Checkpoint | undefined;
  /** The shortest start of a checkpoint's id (`kpqx`), or "base" for none. */
  short: (id: string | null | undefined) => string;
  /** Where a checkpoint came from: `RUN · S31`, or that it was made outside a run, or that this ledger lacks it. */
  origin: (id: string | null | undefined) => string;
  /** Everything about a checkpoint in a line or two, for under the pointer. */
  title: (id: string | null | undefined) => string;
  /** The bookmarks that name a checkpoint. */
  bookmarks: (id: string | null | undefined) => string[];
  /** What a checkpoint's weights build on, in words: a checkpoint this ledger has (`kpqx (RUN · S3)`), else the
   * model's name (or a checkpoint's id this ledger lacks). */
  base: (base: string | null | undefined) => string;
}

const knownCache = new WeakMap<object, WeakMap<object, Known>>();
const NOTHING: Checkpoint[] = [];
const NO_NAMES: Record<string, string> = {};

export function knownOf(checkpoints: Checkpoint[] | undefined, runs: Record<string, string> | undefined): Known {
  const list = checkpoints ?? NOTHING, names = runs ?? NO_NAMES;
  let byNames = knownCache.get(list);
  if (!byNames) knownCache.set(list, (byNames = new WeakMap()));
  let found = byNames.get(names);
  if (!found) {
    const byId = new Map(list.map(checkpoint => [checkpoint.id, checkpoint]));
    const run = (id: string | null | undefined) => (id ? names[id] ?? id : "");
    const checkpoint = (id: string | null | undefined) => (id ? byId.get(id) : undefined);
    const short = (id: string | null | undefined) => (id ? byId.get(id)?.short ?? id.slice(0, 8) : "base");
    const origin = (id: string | null | undefined) => {
      if (!id) return "the base model";
      const each = byId.get(id);
      if (!each) return "not in this ledger";
      return each.run ? `${run(each.run)}${each.step != null ? ` · S${each.step}` : ""}` : "made outside a run";
    };
    const base = (name: string | null | undefined) =>
      !name ? "the base model" : byId.has(name) ? `${short(name)} (${origin(name)})` : name;
    found = {
      run, checkpoint, short, origin, base,
      bookmarks: id => checkpoint(id)?.bookmarks ?? [],
      title: id => {
        const each = checkpoint(id);
        if (!each) return id ?? "the base model";
        const from = each.parents.length ? each.parents.map(short).join(" + ") : base(each.base);
        return [`${each.id} · depth ${each.depth}${each.kind === "full" ? " · full weights" : ""}`, `${origin(id)}, from ${from}`,
          each.bookmarks.length ? `bookmarks: ${each.bookmarks.join(", ")}` : null].filter(Boolean).join("\n");
      },
    };
    byNames.set(names, found);
  }
  return found;
}

/** A checkpoint's line: its first parents back to the one trained from the base model, oldest first, ending at it (a
 * parent this ledger lacks ends it there). */
export function lineOf(checkpoint: Checkpoint, find: (id: string | null | undefined) => Checkpoint | undefined): Checkpoint[] {
  const line: Checkpoint[] = [];
  for (let each: Checkpoint | undefined = checkpoint; each && !line.includes(each); each = find(each.parents[0])) line.unshift(each);
  return line;
}

/** The checkpoints a run made, oldest first. */
export const madeBy = (checkpoints: Checkpoint[], run: string): Checkpoint[] =>
  checkpoints.filter(checkpoint => checkpoint.run === run).sort((a, b) => a.depth - b.depth || a.made - b.made);

/** The color a run's state is said in. */
export const runKind = (state: string): string =>
  state === "running" ? "good" : state === "idle" ? "warm" : state === "failed" || state === "lost" ? "bad" : "";
