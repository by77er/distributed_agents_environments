// Environments as the page says them: by a readable name, with `module:name` beside it where it matters.

import type { KnownEnvironment } from "../api/types";

/** Names an environment's object is often given, which say nothing of it. */
const GENERIC = new Set(["environment", "env", "environments", "main"]);

/** An environment's `module:name`, in a word: its object's name, or its package's where that name says nothing (as the
 * monitor names it). */
export function readable(environment: string | null | undefined): string {
  if (!environment) return "–";
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
