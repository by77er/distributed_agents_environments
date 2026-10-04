// A suite's versions as the page has them: which version each eval played, a suite's subjects grouped by version (and,
// within a version of several environments, by entry) with every start of the versions shown, and what the forms that
// make and edit a suite ask for.

import type { Chosen, EntryScore, Suite, SuiteEntry, SuiteStart, SuiteVersion } from "../api/types";

export type Subject = Suite["subjects"][number];

export const EVAL_DATA: Chosen = "eval data";
export const DRAWN: Chosen = "rows and seeds";
export const GIVEN: Chosen = "starts";
/** An edit that keeps the starts of the version it edits. */
export const SAME = "same";
/** Every version at once, in the grid. */
export const ALL = "all";

/** A version's number, from its id (`NAME@N`); 1 where it says none. */
export function versionNumber(id: string | null | undefined): number {
  const number = Number(id?.split("@").at(-1));
  return Number.isInteger(number) && number > 0 ? number : 1;
}

/** A version, in a word: `v2`. */
export const versionTag = (id: string | null | undefined): string => `v${versionNumber(id)}`;

/** The suite a reference names (`NAME` or `NAME@N`). */
export const suiteName = (reference: string): string => reference.split("@")[0];

/** Every version of a suite, oldest first (a suite the monitor says no versions of: its version 1, of one entry). */
export function versionsOf(suite: Suite & { environments?: (string | null)[] }): SuiteVersion[] {
  if (suite.versions?.length) return suite.versions;
  const environment = suite.environments?.[0] ?? null;
  return [{
    id: `${suite.suite}@1`, number: 1, environments: [environment], made: null, held_out: false, edited_from: null,
    entries: [{
      environment, environment_version: null, chosen: DRAWN, eval_data: null, rows: null, seeds: null, held_out: false,
      episodes: 1, thinking_tokens: null, answer_tokens: null, offset: 0, starts: suite.starts.length,
    }],
    starts: suite.starts,
  }];
}

/** The version a suite's name points to. */
export function currentOf(suite: Suite): SuiteVersion {
  const versions = versionsOf(suite);
  return versions.find(each => each.id === suite.version) ?? versions[versions.length - 1];
}

/** The version a subject played, by id (one from before versions: version 1). */
export const playedVersion = (subject: Subject, suite: string): string => subject.version ?? `${suite}@1`;

/** A version's starts of one entry. */
export const entryStarts = (version: SuiteVersion, entry: SuiteEntry): SuiteStart[] =>
  version.starts.slice(entry.offset, entry.offset + entry.starts);

/** A start, as every version that has it says it: its environment, and what the monitor calls it, else its row and
 * seed. */
export const startIdentity = (start: SuiteStart): string => `${start.environment ?? ""}|${start.identity ?? `${start.task}|${start.seed}`}`;

/** A group of the grid's columns: a version, or one entry of a version of several. `key` is what its columns' cells are
 * found by. */
export interface Block {
  key: string;
  version: SuiteVersion;
  /** The entry, for a version of several; none: every start of the version. */
  entry: SuiteEntry | null;
  subjects: Subject[];
}

/** A version's subjects, as the grid stands them together. */
export interface VersionGroup {
  version: SuiteVersion;
  subjects: Subject[];
}

/** A suite's subjects by the version each played, newest version first: the one picked, or (`ALL`) every version any
 * subject played. Subjects compare within a version only. */
export function versionGroups(suite: Suite, subjects: Subject[], picked: string): VersionGroup[] {
  return [...versionsOf(suite)]
    .filter(version => picked === ALL || version.id === picked)
    .sort((a, b) => b.number - a.number)
    .map(version => ({ version, subjects: subjects.filter(subject => playedVersion(subject, suite.suite) === version.id) }))
    .filter(group => group.subjects.length || group.version.id === picked);
}

/** The grid's column groups: a block for each version shown, and within a version of several environments one for each
 * of its entries, in order. */
export function blocksOf(groups: VersionGroup[]): Block[] {
  return groups.flatMap(({ version, subjects }): Block[] =>
    version.entries.length > 1
      ? version.entries.map(entry => ({ key: `${version.id}#${entry.environment ?? ""}`, version, entry, subjects }))
      : [{ key: version.id, version, entry: null, subjects }]);
}

/** One start of the grid, and its number in each block that has it (by the block's key). */
export interface GridRow {
  key: string;
  start: SuiteStart;
  at: Record<string, string>;
}

/** Every start of some blocks, each once: the newest version's in its order (its entries' in turn), then those only
 * older ones have. */
export function gridRows(blocks: Block[]): GridRow[] {
  const rows = new Map<string, GridRow>();
  for (const block of [...blocks].sort((a, b) => b.version.number - a.version.number)) {
    const starts = block.entry ? entryStarts(block.version, block.entry) : block.version.starts;
    for (const start of starts) {
      const key = startIdentity(start);
      const row = rows.get(key) ?? { key, start, at: {} };
      row.at[block.key] = start.start;
      rows.set(key, row);
    }
  }
  return [...rows.values()];
}

/** How a subject did at a block's starts: at its entry (as the monitor says), or at every start of its version. */
export function blockScore(subject: Subject, block: Block): EntryScore {
  if (!block.entry) return { environment: null, played: subject.played, solved: subject.solved, reward: subject.reward ?? null };
  const place = block.version.entries.indexOf(block.entry);
  return subject.entries?.[place] ?? { environment: block.entry.environment, played: 0, solved: null, reward: null };
}

/** What one entry of a suite's form has, as typed. */
export interface EntryFields {
  environment: string;
  chosen: Chosen | typeof SAME;
  evalData: string;
  /** Row keys; none: every row. */
  rows: string[];
  seeds: string;
  /** One start a line: a row and a seed. */
  starts: string;
  episodes: string;
  thinking: string;
  answer: string;
}

/** Whole numbers, 0 at least, as typed: separated by commas or spaces, a run of them as `1-5`; none where any is not. */
export function wholes(text: string): number[] | null {
  const found: number[] = [];
  for (const part of text.split(/[\s,]+/).filter(Boolean)) {
    const run = /^(\d+)-(\d+)$/.exec(part);
    if (run) {
      const [from, to] = [Number(run[1]), Number(run[2])];
      if (to < from || to - from > 1000) return null;
      for (let each = from; each <= to; each += 1) found.push(each);
    } else if (/^\d+$/.test(part)) found.push(Number(part));
    else return null;
  }
  return found;
}

const whole = (text: string, least = 1): number | null => {
  const value = Number(text.trim());
  return text.trim() !== "" && Number.isInteger(value) && value >= least ? value : null;
};

/** What one entry of a suite's form asks the monitor for, or why a field cannot be what was typed. */
export function entryBody(fields: EntryFields): { body: Record<string, unknown>; errors: Record<string, string> } {
  const errors: Record<string, string> = {};
  const body: Record<string, unknown> = { environment: fields.environment.trim(), chosen: fields.chosen };
  if (!fields.environment.trim()) errors.environment = "which environment";
  if (fields.chosen === EVAL_DATA) {
    if (!fields.evalData.trim()) errors.evalData = "which eval data";
    body.eval_data = fields.evalData.trim();
  } else if (fields.chosen === DRAWN) {
    const seeds = wholes(fields.seeds);
    if (!seeds?.length) errors.seeds = "whole numbers, as 1, 2, 3 or 1-5";
    body.rows = fields.rows.length ? fields.rows : null;
    body.seeds = seeds ?? [];
  } else if (fields.chosen === GIVEN) {
    const lines = fields.starts.split("\n").map(line => line.trim()).filter(Boolean);
    const starts = lines.map(line => line.split(/[\s,]+/)).map(([task, seed, ...more]) => (task && whole(seed ?? "", 0) != null && !more.length ? { task, seed: Number(seed) } : null));
    if (!starts.length || starts.some(each => each == null)) errors.starts = "a row and a seed on each line";
    body.starts = starts.filter(Boolean);
  }
  const episodes = whole(fields.episodes);
  if (episodes == null) errors.episodes = "a whole number, 1 at least";
  body.episodes = episodes;
  for (const [key, field] of [["thinking_tokens", "thinking"], ["answer_tokens", "answer"]] as const) {
    const text = fields[field].trim();
    const value = text ? whole(text) : null;
    if (text && value == null) errors[field] = "a whole number, 1 at least, or empty";
    body[key] = value;
  }
  return { body, errors };
}

/** What a suite's form asks the monitor for (`POST /api/suites/NAME`: its entries, and `base`, the version an edit was
 * made from), or why it cannot be: each entry's errors by field, and the suite's (`suite`). */
export function suiteBody(entries: EntryFields[], base?: number): { body: Record<string, unknown>; errors: Record<string, string>[]; suite: string | null } {
  const made = entries.map(entryBody);
  const environments = entries.map(each => each.environment.trim()).filter(Boolean);
  const twice = environments.find((each, place) => environments.indexOf(each) !== place);
  const suite = !entries.length ? "an environment at least" : twice ? `${twice} is in two entries` : null;
  const body: Record<string, unknown> = { entries: made.map(each => each.body) };
  if (base != null) body.base = base;
  return { body, errors: made.map(each => each.errors), suite };
}

/** An entry of a suite's form filled from one of a version's entries (to edit it: its starts kept unless they are
 * chosen again), or a new one of an environment. */
export function fieldsOf(entry: SuiteEntry | undefined, version: SuiteVersion | undefined, chosen: EntryFields["chosen"] = SAME, environment = ""): EntryFields {
  const starts = entry && version ? entryStarts(version, entry) : [];
  return {
    environment: entry?.environment ?? environment,
    chosen,
    evalData: entry?.eval_data ?? "",
    rows: entry?.chosen === DRAWN ? entry.rows ?? [] : [],
    seeds: entry?.chosen === DRAWN ? (entry.seeds ?? []).join(", ") : "",
    starts: starts.map(start => `${start.task} ${start.seed}`).join("\n"),
    episodes: String(entry?.episodes ?? 1),
    thinking: entry?.thinking_tokens != null ? String(entry.thinking_tokens) : "",
    answer: entry?.answer_tokens != null ? String(entry.answer_tokens) : "",
  };
}

/** An entry's sampling limits, in a few words: a limit it leaves unset is the channel's own (which may be none). */
export const limitsText = (entry: SuiteEntry): string =>
  entry.thinking_tokens == null && entry.answer_tokens == null ? "channel's own"
    : `thinking ${entry.thinking_tokens ?? "channel's"} · answer ${entry.answer_tokens ?? "channel's"}`;

/** How an entry's starts were chosen, in a few words. */
export const chosenText = (entry: SuiteEntry): string =>
  entry.chosen === "eval data" ? `eval data ${entry.eval_data ?? ""}` : entry.chosen === "starts" ? "given starts" : "rows and seeds";
