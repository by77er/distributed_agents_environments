// Reading the monitor's endpoints. Each answer's version (its ETag) is kept beside it: asked again, the monitor is
// told which version the page has, and answers 304 when nothing changed, so the page keeps what it has.

export class NotFound extends Error {}

interface Kept {
  etag: string;
  data: unknown;
}

const kept = new Map<string, Kept>();

/** The version the page has of what a path answers, as the stream names versions (without the ETag's quoting). */
export const versionHeld = (path: string): string | null => kept.get(path)?.etag.replace(/^W\/"|"$/g, "") ?? null;

export async function readJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const held = kept.get(path);
  const answer = await fetch(path, { signal, headers: held ? { "If-None-Match": held.etag } : {} });
  if (answer.status === 304 && held) return held.data as T;
  if (answer.status === 404) throw new NotFound(path);
  if (!answer.ok) throw new Error(`${path}: ${answer.status}`);
  const data = (await answer.json()) as T;
  const etag = answer.headers.get("ETag");
  if (etag) kept.set(path, { etag, data });
  return data;
}
