// Settings as forms have them: typed in as text, read as the values a profile or a run takes.

export const GROUPS_PER_STEP = "groups_per_step";
export const EVALS_SUITE = "evals.suite";
export const EVALS_EVERY = "evals.every";
export const EVALS_EPISODES = "evals.episodes";
/** Settings that are whole numbers, 1 at least. */
const WHOLE = new Set([GROUPS_PER_STEP, EVALS_EVERY, EVALS_EPISODES]);

/** A setting's value as typed: a number, a boolean, JSON (a list, a table, a quoted string), or else the text. */
export function typed(text: string): unknown {
  const trimmed = text.trim();
  if (trimmed === "") return "";
  if (trimmed === "true" || trimmed === "false") return trimmed === "true";
  if (/^-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(trimmed)) return Number(trimmed);
  if (/^[[{"]/.test(trimmed)) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return text;
    }
  }
  return text;
}

/** A value as a form shows it. */
export const shown = (value: unknown): string => (value == null ? "" : typeof value === "string" ? value : JSON.stringify(value));

/** One setting's value from what was typed, or why it cannot be: a whole number of 1 at least where one is needed (the
 * evals' episodes may be empty: the suite's own), a suite's name or none for the evals' suite, else as `typed` reads it
 * (empty: none). */
export function settingOf(key: string, text: string): { value: unknown } | { error: string } {
  if (key === EVALS_SUITE) return { value: text.trim() || null };
  if (key === EVALS_EPISODES && !text.trim()) return { value: null };
  if (WHOLE.has(key)) {
    const value = Number(text.trim());
    return Number.isInteger(value) && value >= 1 && text.trim() !== "" ? { value } : { error: "a whole number, 1 at least" };
  }
  const value = typed(text);
  return { value: value === "" ? null : value };
}

/** What a settings form asks for: each field whose value differs from the run's (`current`: what it uses or is wanted
 * to), and why any field cannot be what was typed. */
export function wantedOf(fields: Record<string, string>, current: Record<string, unknown>): { settings: Record<string, unknown>; errors: Record<string, string> } {
  const settings: Record<string, unknown> = {}, errors: Record<string, string> = {};
  for (const [key, text] of Object.entries(fields)) {
    const read = settingOf(key, text);
    if ("error" in read) errors[key] = read.error;
    else if (shown(read.value) !== shown(current[key] ?? null)) settings[key] = read.value;
  }
  return { settings, errors };
}

/** The choice of no evals, beside the suites (no suite's name says `@`). */
export const NO_EVALS = "@none";

/** The evals a new run makes, as launch settings: a suite (`suite`, by name), every how many steps and how many episodes
 * of each start (empty: the suite's own); or none (`NO_EVALS`), said so. Nothing chosen, or a field that cannot be what
 * was typed, says why. */
export function evalsSettings(suite: string, every: string, episodes: string): { settings: Record<string, unknown>; errors: Record<string, string> } {
  if (!suite.trim()) return { settings: {}, errors: { [EVALS_SUITE]: "a suite, or none" } };
  if (suite === NO_EVALS) return { settings: { [EVALS_SUITE]: null }, errors: {} };
  const asked = wantedOf({ [EVALS_SUITE]: suite, [EVALS_EVERY]: every, [EVALS_EPISODES]: episodes }, {});
  return { settings: { [EVALS_EPISODES]: null, ...asked.settings }, errors: asked.errors };
}
