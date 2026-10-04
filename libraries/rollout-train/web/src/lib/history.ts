// A subject's history as the page has it: its evals grouped by the version of a suite each played (two versions' scores
// do not compare), each version's environments (apart: environments' rewards do not compare), each eval's score at each,
// and each environment's score over time.

import type { CheckpointEval } from "../api/types";
import { versionNumber } from "./suites";

/** An eval's score at an environment: episodes played, the share solved of those that say (none where none do), and
 * the mean reward. */
export interface Score {
  played: number;
  share: number | null;
  reward: number | null;
}

export interface HistoryGroup {
  /** The version, by id, its suite and its number. */
  version: string;
  suite: string;
  number: number;
  /** How the page says it: the suite's name, and the version where the history has more than one of that suite. */
  label: string;
  /** Its environments, in the order of its entries. */
  environments: (string | null)[];
  /** Its evals, newest first. */
  evals: CheckpointEval[];
}

/** The version an eval played, by id. */
export const versionOf = (each: CheckpointEval): string => each.version ?? `${each.suite}@1`;

/** A subject's evals by the version each played: suites by name, a suite's versions newest first. */
export function historyOf(evals: CheckpointEval[]): HistoryGroup[] {
  const groups = new Map<string, HistoryGroup>();
  for (const each of evals) {
    const version = versionOf(each);
    const group = groups.get(version) ?? { version, suite: each.suite, number: versionNumber(version), label: each.suite, environments: [], evals: [] };
    group.evals.push(each);
    for (const entry of each.entries ?? []) if (!group.environments.includes(entry.environment)) group.environments.push(entry.environment);
    groups.set(version, group);
  }
  const listed = [...groups.values()];
  for (const group of listed) {
    if (!group.environments.length) group.environments.push(null);  // (evals that say no entries: one column)
    group.evals.sort((a, b) => (b.started ?? 0) - (a.started ?? 0));
    if (listed.filter(each => each.suite === group.suite).length > 1) group.label = `${group.suite} v${group.number}`;
  }
  return listed.sort((a, b) => a.suite.localeCompare(b.suite) || b.number - a.number);
}

/** An eval's score at each of its group's environments: its entry's there, or, for an eval that says no entries, its
 * whole score at the first and nothing at the others. */
export function scoresOf(group: HistoryGroup, each: CheckpointEval): (Score | null)[] {
  const entries = each.entries ?? [];
  if (!entries.length) return group.environments.map((_, place) => (place ? null : { played: each.played, share: each.share, reward: each.reward }));
  return group.environments.map(environment => {
    const entry = entries.find(found => found.environment === environment);
    return entry ? { played: entry.played, share: entry.share, reward: entry.reward } : null;
  });
}

export interface HistorySeries {
  environment: string | null;
  /** The share solved where any eval's episodes there say, else the mean reward. */
  measure: "solved" | "reward";
  /** [when the eval began, its score], oldest first. */
  points: [number, number][];
}

/** Each environment's score over time in a group: an eval's share solved there (else its mean reward), at when it
 * began; evals that played nothing there are left out. */
export function historySeries(group: HistoryGroup): HistorySeries[] {
  const scored = group.evals.filter(each => each.started != null).map(each => ({ at: each.started!, scores: scoresOf(group, each) }));
  return group.environments.map((environment, place) => {
    const here = scored.flatMap(({ at, scores }) => (scores[place]?.played ? [{ at, score: scores[place]! }] : []));
    const measure: "solved" | "reward" = here.some(({ score }) => score.share != null) ? "solved" : "reward";
    const points = here
      .map(({ at, score }) => [at, measure === "solved" ? score.share : score.reward] as [number, number | null])
      .filter((point): point is [number, number] => point[1] != null)
      .sort((a, b) => a[0] - b[0]);
    return { environment, measure, points };
  });
}
