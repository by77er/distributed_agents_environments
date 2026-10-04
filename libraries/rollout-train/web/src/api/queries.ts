// What the page reads, each a query of its own: read when a view needs it, kept while it is shown, and read again
// when the stream says its topic changed (`stream.ts`). An answer that changed only in part keeps the parts that did
// not (TanStack Query's structural sharing), so only the views over what changed draw again.

import { QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { readJson } from "./client";
import type { Bookmark, CheckpointEvals, Entry, EnvironmentInfo, Episode, Evals, FeedRun, Group, KnownEnvironment, Launch, LaunchAsked, Launches, Lineage, Machines, Path, RunSettings, Statistics, System } from "./types";
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
  machines: (): Topic => ({ topic: "machines", key: ["machines"], path: "api/machines" }),
  launches: (): Topic => ({ topic: "launches", key: ["launches"], path: "api/launches" }),
  statistics: (): Topic => ({ topic: "statistics", key: ["statistics"], path: "api/statistics" }),
  evals: (): Topic => ({ topic: "evals", key: ["evals"], path: "api/evals" }),
  checkpoints: (sample: boolean): Topic => ({
    topic: sample ? "checkpoints/sample" : "checkpoints",
    key: ["checkpoints", sample],
    path: `api/checkpoints${sample ? "?sample=1" : ""}`,
  }),
  checkpointEvals: (id: string): Topic => ({ topic: `checkpoint-evals/${id}`, key: ["checkpoint-evals", id], path: `api/checkpoints/${encodeURIComponent(id)}/evals` }),
  path: (id: string): Topic => ({ topic: `path/${id}`, key: ["path", id], path: `api/checkpoints/${encodeURIComponent(id)}/path` }),
  settings: (run: string): Topic => ({ topic: `settings/${run}`, key: ["settings", run], path: `api/runs/${encodeURIComponent(run)}/settings` }),
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

/** What the page knows of runs and checkpoints, to say them (`Known`); a view over it draws again only when a checkpoint, a
 * bookmark or a run's name changes. */
export function useKnown(): Known {
  const { data: checkpoints } = useQuery({ ...systemQuery, select: (system: System) => system.checkpoints });
  const { data: runs } = useQuery({ ...systemQuery, select: (system: System) => system.names?.runs });
  return knownOf(checkpoints, runs);
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

/** Make a bookmark name a checkpoint, or move it there. */
export const useBookmark = () =>
  useWrite(async ({ name, checkpoint }: { name: string; checkpoint: string }) =>
    (await asked<{ bookmark: Bookmark }>("api/bookmarks", "POST", { name, checkpoint })).bookmark);

/** Take a bookmark away (the checkpoint stays). */
export const useUnbookmark = () =>
  useWrite(async (name: string) => asked<{ unbookmarked: string }>(`api/bookmarks/${encodeURIComponent(name)}`, "DELETE"));

export const useFeeds = () =>
  useQuery({ queryKey: topics.feeds().key, queryFn: ({ signal }) => readJson<FeedRun[]>(topics.feeds().path, signal) });

export const useMachines = () =>
  useQuery({ queryKey: topics.machines().key, queryFn: ({ signal }) => readJson<Machines>(topics.machines().path, signal) });

export const useLaunches = () =>
  useQuery({ queryKey: topics.launches().key, queryFn: ({ signal }) => readJson<Launches>(topics.launches().path, signal) });

export const useEvals = (enabled = true) =>
  useQuery({ queryKey: topics.evals().key, enabled, queryFn: ({ signal }) => readJson<Evals>(topics.evals().path, signal) });

/** What the suites' forms need of an environment (none where it does not load on the monitor's machine). */
export const useEnvironment = (name: string) =>
  useQuery({
    queryKey: ["environment", name],
    enabled: Boolean(name),
    staleTime: Infinity,
    refetchInterval: false,
    retry: false,
    queryFn: ({ signal }) => readJson<EnvironmentInfo>(`api/environments/${encodeURIComponent(name)}`, signal),
  });

/** Every environment the system knows of, for the pickers: offered by a launcher alive, started on, or played by a
 * suite (read again now and then: launchers come and go). */
export const useEnvironments = () =>
  useQuery({
    queryKey: ["environments"],
    refetchInterval: 15_000,
    queryFn: async ({ signal }) => (await readJson<{ environments: KnownEnvironment[] }>("api/environments", signal)).environments,
  });

/** Make a suite, or its next version (which its name then points to). */
export function useSaveSuite(name: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (body: Record<string, unknown>) =>
      asked<{ suite: string; version: string; number: number }>(`api/suites/${encodeURIComponent(name)}`, "POST", body),
    onSuccess: () => client.invalidateQueries({ queryKey: topics.evals().key }),
  });
}

/** Ask for a run or an eval: a launcher alive that offers its profile starts it. */
export function useLaunch() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (launch: LaunchAsked) => (await asked<{ launch: Launch }>("api/launches", "POST", launch)).launch,
    onSuccess: () => client.invalidateQueries({ queryKey: topics.launches().key }),
  });
}

/** Ask a launch to stop: one not started is stopped at once; a run going is stopped by its launcher. */
export function useStop() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => (await asked<{ launch: Launch }>(`api/launches/${encodeURIComponent(id)}/stop`, "POST")).launch,
    onSuccess: () => client.invalidateQueries({ queryKey: topics.launches().key }),
  });
}

export const useStatistics = () =>
  useQuery({
    queryKey: topics.statistics().key,
    queryFn: async ({ signal }) => {
      const figures = await readJson<Statistics>(topics.statistics().path, signal);
      setServerTime(figures.now);
      return figures;
    },
  });

/** Every eval a checkpoint had. */
export const useCheckpointEvals = (id: string | null | undefined) =>
  useQuery({ queryKey: topics.checkpointEvals(id ?? "").key, enabled: Boolean(id), queryFn: ({ signal }) => readJson<CheckpointEvals>(topics.checkpointEvals(id!).path, signal) });

/** A checkpoint's line from the base model, with each point's scores. */
export const usePath = (id: string | null | undefined) =>
  useQuery({ queryKey: topics.path(id ?? "").key, enabled: Boolean(id), queryFn: ({ signal }) => readJson<Path>(topics.path(id!).path, signal) });

/** A training run's settings, and what is wanted of them. */
export const useRunSettings = (run: string) =>
  useQuery({ queryKey: topics.settings(run).key, queryFn: ({ signal }) => readJson<RunSettings>(topics.settings(run).path, signal) });

/** Want some of a run's changeable settings from its next step on. */
export function useWantSettings(run: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (settings: Record<string, unknown>) => asked<{ desired: unknown }>(topics.settings(run).path, "POST", { settings }),
    onSuccess: () => client.invalidateQueries({ queryKey: topics.settings(run).key }),
  });
}

export const useLineage = (sample: boolean) =>
  useQuery({ queryKey: topics.checkpoints(sample).key, queryFn: ({ signal }) => readJson<Lineage>(topics.checkpoints(sample).path, signal) });

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
