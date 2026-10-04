// What the page reads, each a query of its own: read when a view needs it, kept while it is shown, and read again
// when the stream says its topic changed (`stream.ts`). An answer that changed only in part keeps the parts that did
// not (TanStack Query's structural sharing), so only the views over what changed draw again.

import { QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { readJson } from "./client";
import type { Bookmark, Entry, Episode, FeedRun, Group, Lineage, Machine, Statistics, System } from "./types";
import { type Known, knownOf } from "../lib/model";
import { setServerTime } from "../lib/now";

/** A topic (as the stream names it), the query that holds it, and the path it is read from. */
export interface Topic {
  topic: string;
  key: readonly unknown[];
  path: string;
}

export const topics = {
  system: (): Topic => ({ topic: "system", key: ["system"], path: "api/system" }),
  feeds: (): Topic => ({ topic: "feeds", key: ["feeds"], path: "api/runs" }),
  machine: (): Topic => ({ topic: "machine", key: ["machine"], path: "api/machine" }),
  statistics: (): Topic => ({ topic: "statistics", key: ["statistics"], path: "api/statistics" }),
  versions: (sample: boolean): Topic => ({
    topic: sample ? "versions/sample" : "versions",
    key: ["versions", sample],
    path: `api/versions${sample ? "?sample=1" : ""}`,
  }),
  group: (run: string, number: number): Topic => ({
    topic: `group/${run}/${number}`,
    key: ["group", run, number],
    path: `api/groups/${encodeURIComponent(run)}/${number}`,
  }),
  episode: (id: string): Topic => ({ topic: `episode/${id}`, key: ["episode", id], path: `api/episodes/${encodeURIComponent(id)}` }),
};

export function newQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // The stream says when to read again; reading every half minute besides is for a stream that was cut.
        staleTime: Infinity,
        refetchInterval: 30_000,
        refetchOnWindowFocus: false,
        retry: (count, error) => error.name !== "NotFound" && count < 3,
      },
    },
  });
}

const systemQuery = {
  queryKey: topics.system().key,
  queryFn: async ({ signal }: { signal: AbortSignal }) => {
    const system = await readJson<System>(topics.system().path, signal);
    setServerTime(system.at);
    return system;
  },
};

export const useSystem = () => useQuery(systemQuery);

/** What the page knows of runs and versions, to say them (`Known`); a view over it draws again only when a version, a
 * bookmark or a run's name changes. */
export function useKnown(): Known {
  const { data: versions } = useQuery({ ...systemQuery, select: (system: System) => system.versions });
  const { data: runs } = useQuery({ ...systemQuery, select: (system: System) => system.names?.runs });
  return knownOf(versions, runs);
}

/** Change a name in the registry; every page open on the monitor hears of it through its stream, this one at once. */
function useWrite<Asked, Answer>(write: (asked: Asked) => Promise<Answer>) {
  const client = useQueryClient();
  return useMutation({ mutationFn: write, onSuccess: () => client.invalidateQueries({ queryKey: topics.system().key }) });
}

async function asked<T>(path: string, method: string, body?: unknown): Promise<T> {
  const answer = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  const said = (await answer.json()) as T & { error?: string };
  if (!answer.ok) throw new Error(said.error ?? `the monitor answered ${answer.status}`);
  return said;
}

/** Call a run by a new name. */
export const useRename = () =>
  useWrite(async ({ id, name }: { id: string; name: string }) => (await asked<{ entry: Entry }>("api/rename", "POST", { id, name })).entry);

/** Make a bookmark name a version, or move it there. */
export const useBookmark = () =>
  useWrite(async ({ name, version }: { name: string; version: string }) =>
    (await asked<{ bookmark: Bookmark }>("api/bookmarks", "POST", { name, version })).bookmark);

/** Take a bookmark away (the version stays). */
export const useUnbookmark = () =>
  useWrite(async (name: string) => asked<{ unbookmarked: string }>(`api/bookmarks/${encodeURIComponent(name)}`, "DELETE"));

export const useFeeds = () =>
  useQuery({ queryKey: topics.feeds().key, queryFn: ({ signal }) => readJson<FeedRun[]>(topics.feeds().path, signal) });

export const useMachine = () =>
  useQuery({ queryKey: topics.machine().key, queryFn: ({ signal }) => readJson<Machine>(topics.machine().path, signal) });

export const useStatistics = () =>
  useQuery({
    queryKey: topics.statistics().key,
    queryFn: async ({ signal }) => {
      const figures = await readJson<Statistics>(topics.statistics().path, signal);
      setServerTime(figures.now);
      return figures;
    },
  });

export const useLineage = (sample: boolean) =>
  useQuery({ queryKey: topics.versions(sample).key, queryFn: ({ signal }) => readJson<Lineage>(topics.versions(sample).path, signal) });

export const useGroup = (run: string, number: number) =>
  useQuery({
    queryKey: topics.group(run, number).key,
    queryFn: ({ signal }) => readJson<Group>(topics.group(run, number).path, signal),
  });

/** An episode's lines, read as they grow: each reading asks only for the lines after those the page has. */
export function useEpisode(id: string, enabled = true) {
  const client = useQueryClient();
  const topic = topics.episode(id);
  return useQuery({
    queryKey: topic.key,
    enabled: enabled && Boolean(id),
    structuralSharing: false,  // (the lines it had are kept as they were: only the new ones are new)
    queryFn: async ({ signal }) => {
      const known = client.getQueryData<Episode>(topic.key);
      const after = known ? known.lines.length : 0;
      const more = await readJson<Episode>(`${topic.path}?after=${after}`, signal);
      if (known && more.source === known.source) return { ...more, lines: more.lines.length ? [...known.lines, ...more.lines] : known.lines };
      if (known && after) return readJson<Episode>(topic.path, signal);  // (read from elsewhere now: from the start)
      return more;
    },
  });
}
