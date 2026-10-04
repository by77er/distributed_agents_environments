// A suite's versions as the page has them: which version each eval played, a suite's subjects grouped by version with
// every start of the versions shown, and what the forms that make and edit a suite ask for.

import type { Chosen, Suite, SuiteStart, SuiteVersion } from "../api/types";

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

/** Every version of a suite, oldest first (a suite the monitor says no versions of: its version 1). */
export function versionsOf(suite: Suite & { environment?: string | null }): SuiteVersion[] {
  if (suite.versions?.length) return suite.versions;
  return [{
    id: `${suite.suite}@1`, number: 1, environment: suite.environment ?? null, environment_version: null, made: null,
    chosen: DRAWN, eval_data: null, rows: null, seeds: null, held_out: false, episodes: 1, thinking_tokens: null,
    answer_tokens: null, edited_from: null, starts: suite.starts,
  }];
}

/** The version a suite's name points to. */
export function currentOf(suite: Suite): SuiteVersion {
  const versions = versionsOf(suite);
  return versions.find(each => each.id === suite.version) ?? versions[versions.length - 1];
}

/** The version a subject played, by id (one from before versions: version 1). */
export const playedVersion = (subject: Subject, suite: string): string => subject.version ?? `${suite}@1`;

/** A start, as every version that has it says it: what the monitor calls it, else its row and seed. */
export const startIdentity = (start: SuiteStart): string => start.identity ?? `${start.task}|${start.seed}`;

/** One start of the grid, and its number in each version that has it (by the version's id). */
export interface GridRow {
  key: string;
  start: SuiteStart;
  at: Record<string, string>;
}

/** Every start of some versions, each once: the newest version's in its order, then those only older ones have. */
export function gridRows(versions: SuiteVersion[]): GridRow[] {
  const rows = new Map<string, GridRow>();
  for (const version of [...versions].sort((a, b) => b.number - a.number)) {
    for (const start of version.starts) {
      const key = startIdentity(start);
      const row = rows.get(key) ?? { key, start, at: {} };
      row.at[version.id] = start.start;
      rows.set(key, row);
    }
  }
  return [...rows.values()];
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

/** What a suite's form has, as typed. */
export interface SuiteFields {
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

/** What a suite's form asks the monitor for (`POST /api/suites/NAME`), or why a field cannot be what was typed. `base` is
 * the version an edit was made from. */
export function suiteBody(fields: SuiteFields, base?: number): { body: Record<string, unknown>; errors: Record<string, string> } {
  const errors: Record<string, string> = {};
  const body: Record<string, unknown> = { chosen: fields.chosen };
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
  if (base != null) body.base = base;
  return { body, errors };
}

/** A suite's form filled from one of its versions (to edit it: its starts kept unless they are chosen again). */
export function fieldsOf(version: SuiteVersion | undefined, chosen: SuiteFields["chosen"] = SAME): SuiteFields {
  return {
    chosen,
    evalData: version?.eval_data ?? "",
    rows: version?.chosen === DRAWN ? version.rows ?? [] : [],
    seeds: version?.chosen === DRAWN ? (version.seeds ?? []).join(", ") : "",
    starts: version ? version.starts.map(start => `${start.task} ${start.seed}`).join("\n") : "",
    episodes: String(version?.episodes ?? 1),
    thinking: version?.thinking_tokens != null ? String(version.thinking_tokens) : "",
    answer: version?.answer_tokens != null ? String(version.answer_tokens) : "",
  };
}
