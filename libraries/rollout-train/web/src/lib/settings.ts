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

/** One setting's value from what was typed, or why it cannot be: a whole number of 1 at least where one is needed, a
 * suite's name or none for the evals' suite, else as `typed` reads it (empty: none). */
export function settingOf(key: string, text: string): { value: unknown } | { error: string } {
  if (key === EVALS_SUITE) return { value: text.trim() || null };
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

/** The evals a new run makes, as launch settings: none without a suite; else the suite, every how many steps and how
 * many episodes of each start, or why those cannot be. */
export function evalsSettings(suite: string, every: string, episodes: string): { settings: Record<string, unknown>; errors: Record<string, string> } {
  if (!suite.trim()) return { settings: {}, errors: {} };
  return wantedOf({ [EVALS_SUITE]: suite, [EVALS_EVERY]: every, [EVALS_EPISODES]: episodes }, {});
}
