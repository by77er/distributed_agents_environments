// Environments as the page says them: by a readable name, with `module:name` (a published one's `NAME@VERSION`) beside
// it where it matters; what an environment's page works out from what the monitor says of it: each row's share solved
// and the rows together, its evals by who played them, and what its newest check found; and a published one's source,
// its check, and an import's stage.

import type { CheckGroup, Description, EnvironmentInfo, EnvironmentRow, EnvironmentScore, Finding, KnownEnvironment, PublishedSource } from "../api/types";

/** Names an environment's object is often given, which say nothing of it. */
const GENERIC = new Set(["environment", "env", "environments", "main"]);

/** A published environment's `NAME@VERSION`: its name and version; none for a built-in one's `module:name`. */
export function publishedParts(environment: string): { name: string; version: string } | null {
  const found = /^([^:@]+)@([0-9a-f]{64})$/.exec(environment);
  return found ? { name: found[1], version: found[2] } : null;
}

/** A version's id, in the few characters the monitor shows of it. */
export const shortVersion = (version: string): string => version.slice(0, 12);

/** An environment's `module:name`, in a word: its object's name, or its package's where that name says nothing (as the
 * monitor names it); a published one's `NAME@VERSION` with the version's first few characters. */
export function readable(environment: string | null | undefined): string {
  if (!environment) return "–";
  const published = publishedParts(environment);
  if (published) return `${published.name}@${shortVersion(published.version)}`;
  const [module, attribute] = environment.split(":");
  if (attribute && !GENERIC.has(attribute.toLowerCase())) return attribute;
  return module.split(".")[0] || environment;
}

/** What the pickers list: the environments the system knows, and any named here it does not (an old suite's, say), each
 * with a name no other has (its module's beside a name two share). */
export function pickable(known: KnownEnvironment[], also: (string | null | undefined)[] = []): KnownEnvironment[] {
  const listed = [...known];
  for (const each of also) {
    if (each && !listed.some(environment => environment.environment === each)) {
      listed.push({ environment: each, name: readable(each), versions: [], offered: false });
    }
  }
  const named = (name: string) => listed.filter(each => each.name === name).length;
  return listed
    .map(each => (named(each.name) > 1 ? { ...each, name: `${each.name} (${each.environment.split(":")[0]})` } : each))
    .sort((a, b) => Number(b.offered) - Number(a.offered) || a.name.localeCompare(b.name));
}

/** The share of a row's (or a run's) episodes that solved it, of those whose results say; none where none say. */
export const solvedShare = (counts: { solved: number | null; said: number }): number | null =>
  counts.solved != null && counts.said ? counts.solved / counts.said : null;

/** Every row's counts together: groups, episodes played, and solved of those that say. */
export function rowTotals(rows: EnvironmentRow[]): { groups: number; played: number; solved: number | null; said: number } {
  const said = rows.filter(row => row.solved != null);
  return {
    groups: rows.reduce((sum, row) => sum + row.groups, 0),
    played: rows.reduce((sum, row) => sum + row.played, 0),
    solved: said.length ? said.reduce((sum, row) => sum + (row.solved ?? 0), 0) : null,
    said: said.reduce((sum, row) => sum + row.said, 0),
  };
}

/** Its rows played and its rows held out: how many rows any run played, and how many have eval starts. */
export const rowCounts = (rows: EnvironmentRow[]): { rows: number; played: number; held: number; evalOnly: number } => ({
  rows: rows.filter(row => row.trains).length,
  played: rows.filter(row => row.groups > 0).length,
  held: rows.filter(row => row.held > 0).length,
  evalOnly: rows.filter(row => !row.trains).length,
});

/** Who played an eval: a checkpoint by its id, or a model by its name. */
export const subjectKey = (score: EnvironmentScore): string => (score.checkpoint ? score.checkpoint : `model:${score.model ?? ""}`);

/** Its evals, by who played them: each checkpoint (or model) once, with its evals newest first, those evaluated most
 * lately first. */
export function bySubject(scores: EnvironmentScore[]): { key: string; checkpoint: string | null; model: string | null; scores: EnvironmentScore[] }[] {
  const groups = new Map<string, EnvironmentScore[]>();
  for (const score of [...scores].sort((a, b) => (b.started ?? 0) - (a.started ?? 0))) {
    const key = subjectKey(score);
    groups.set(key, [...(groups.get(key) ?? []), score]);
  }
  return [...groups.entries()].map(([key, listed]) => ({ key, checkpoint: listed[0].checkpoint, model: listed[0].model, scores: listed }));
}

/** What a check found: its groups played to their end, those whose episodes all scored the same (which teach a
 * group-relative update nothing), and in words. */
export function checkFound(groups: CheckGroup[]): { played: number; flagged: number; says: string } {
  const played = groups.filter(group => group.rewards != null);
  const flagged = played.filter(group => group.flagged).length;
  const says = !played.length ? "no group played yet"
    : flagged === played.length ? `every episode of each of the ${played.length} groups scored the same: training would take no step`
      : `${played.length - flagged} of ${played.length} groups have something to teach${flagged ? `; in ${flagged}, every episode scored the same` : ""}`;
  return { played: played.length, flagged, says };
}

/** A description's reward range, as `[0, 1]` (an open side as ∞). */
export const rangeText = (description: Description): string => {
  const [low, high] = description.rewards;
  return `[${low == null ? "−∞" : low}, ${high == null ? "∞" : high}]`;
};

/** The versions of an environment to say: the one it loads as first, then the others seen. */
export const versionsText = (found: Pick<EnvironmentInfo, "version" | "versions">): string[] =>
  [...new Set([...(found.version ? [found.version] : []), ...found.versions])];

/** Where a published environment's source came from, in a line: its repository (without the scheme or `.git`), and its
 * subdirectory there. */
export const sourceText = (published: Pick<PublishedSource, "source" | "subdirectory">): string =>
  `${published.source.replace(/^[a-z+]+:\/\//, "").replace(/\.git$/, "")}${published.subdirectory ? `/${published.subdirectory}` : ""}`;

/** What a check found, in a few words: how many checks passed, and how many of those with something to look at. */
export function checkText(findings: Finding[]): string {
  const passed = findings.filter(each => each.passed).length;
  const flagged = findings.filter(each => each.passed && each.flagged).length;
  if (passed < findings.length) return `${findings.length - passed} of ${findings.length} failed`;
  return `${passed} passed${flagged ? `, ${flagged} flagged` : ""}`;
}

/** What an import is doing, by its stage. */
export const STAGES: Record<string, string> = {
  fetching: "fetching", reading: "reading", packing: "packing", storing: "storing", checking: "checking on Ray",
  recording: "recording", done: "imported", refused: "refused",
};

export const stageText = (stage: string | null | undefined): string => (stage ? STAGES[stage] ?? stage : "starting");
