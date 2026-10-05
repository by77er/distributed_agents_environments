// What the page reads, each a query of its own: read when a view needs it, kept while it is shown, and read again
// when the stream says its topic changed (`stream.ts`). An answer that changed only in part keeps the parts that did
// not (TanStack Query's structural sharing), so only the views over what changed draw again.

import { keepPreviousData, QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { readJson } from "./client";
import type { Bookmark, Checked, CheckpointEvals, Entry, EnvironmentInfo, EnvironmentVersion, Episode, EvalSubjects, Evals, FeedRun, Group, ImportAsked, Imports, KnownEnvironment, Launch, LaunchAsking, Launches, Lineage, Machines, Offers, Path, Preset, Presets, PresetVersions, RunSettings, SettingFinding, Statistics, SubjectHistory, SubjectKind, System } from "./types";
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
  offers: (): Topic => ({ topic: "offers", key: ["offers"], path: "api/offers" }),
  presets: (): Topic => ({ topic: "presets", key: ["presets"], path: "api/presets" }),
  preset: (name: string): Topic => ({ topic: `preset/${name}`, key: ["preset", name], path: `api/presets/${encodeURIComponent(name)}` }),
  statistics: (): Topic => ({ topic: "statistics", key: ["statistics"], path: "api/statistics" }),
  evals: (): Topic => ({ topic: "evals", key: ["evals"], path: "api/evals" }),
  evalSubjects: (): Topic => ({ topic: "eval-subjects", key: ["eval-subjects"], path: "api/evals/subjects" }),
  history: (kind: SubjectKind, id: string): Topic => ({ topic: `history/${kind}/${id}`, key: ["history", kind, id], path: `api/evals/${kind}/${encodeURIComponent(id)}` }),
  environments: (): Topic => ({ topic: "environments", key: ["environments"], path: "api/environments" }),
  environment: (name: string): Topic => ({ topic: `environment/${name}`, key: ["environment", name], path: `api/environments/${encodeURIComponent(name)}` }),
  imports: (): Topic => ({ topic: "imports", key: ["imports"], path: "api/environments/imports" }),
  checkpoints: (): Topic => ({ topic: "checkpoints", key: ["checkpoints"], path: "api/checkpoints" }),
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

/** A topic's body, read from its path. */
function useTopic<T>(topic: Topic, enabled = true) {
  return useQuery({ queryKey: topic.key, enabled, queryFn: ({ signal }) => readJson<T>(topic.path, signal) });
}

/** Change something on the monitor, then read again the topics it changes (by default where everything stands); every
 * page open on the monitor hears of it through its stream, this one at once. */
function useWrite<Asked, Answer>(write: (asked: Asked) => Promise<Answer>, changes: Topic[] = [topics.system()]) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: write,
    onSuccess: () => Promise.all(changes.map(topic => client.invalidateQueries({ queryKey: topic.key }))),
  });
}

/** A launch the monitor refused: why, and the findings that refuse it, each with the setting it is about. */
export class Refused extends Error {
  constructor(message: string, readonly refusals: SettingFinding[], readonly notes: SettingFinding[]) {
    super(message);
    this.name = "Refused";
  }
}

async function asked<T>(path: string, method: string, body?: unknown): Promise<T> {
  const answer = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  const said = (await answer.json()) as T & { error?: string; refusals?: SettingFinding[]; notes?: SettingFinding[] };
  if (said.refusals && answer.status === 422) throw new Refused(said.error ?? "refused", said.refusals, said.notes ?? []);
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

export const useFeeds = () => useTopic<FeedRun[]>(topics.feeds());

export const useMachines = () => useTopic<Machines>(topics.machines());

export const useLaunches = () => useTopic<Launches>(topics.launches());

/** What a run can be asked for on the monitor's cluster. */
export const useOffers = () => useTopic<Offers>(topics.offers());

export const useEvals = (enabled = true) => useTopic<Evals>(topics.evals(), enabled);

/** What the suites' forms and the new run's form need of an environment: its version, rows and eval data (an error where
 * it does not load on the monitor's machine). */
export const useEnvironment = (name: string) =>
  useQuery({
    queryKey: ["environment-loaded", name],
    enabled: Boolean(name),
    staleTime: Infinity,
    refetchInterval: false,
    retry: false,
    queryFn: async ({ signal }) => {
      const found = await readJson<EnvironmentInfo>(topics.environment(name).path, signal);
      if (!found.loads) throw new Error(found.error ?? `${name} does not load here`);
      return found;
    },
  });

/** An environment's page: what it says of itself and what was done with it. */
export const useEnvironmentPage = (name: string) => useTopic<EnvironmentInfo>(topics.environment(name));

/** Every environment the system knows of: offered by the cluster, started on, or played by a suite (read again now and
 * then besides, for the pickers on pages that do not watch it: imports come and go). */
export const useEnvironments = () =>
  useQuery({
    queryKey: topics.environments().key,
    refetchInterval: 15_000,
    queryFn: async ({ signal }) => (await readJson<{ environments: KnownEnvironment[] }>(topics.environments().path, signal)).environments,
  });

/** The imports from git the monitor made, each with its stage; read every second while `watching` (an import is under
 * way here). */
export const useImports = (watching = false) =>
  useQuery({
    queryKey: topics.imports().key,
    refetchInterval: watching ? 1000 : 30_000,
    queryFn: ({ signal }) => readJson<Imports>(topics.imports().path, signal),
  });

/** Import an environment from git: the monitor fetches, checks and records it, and answers with the version once it is
 * recorded (one imported before, the same source, is answered at once), or with why it was refused. */
export const useImport = () =>
  useWrite(
    async (body: ImportAsked) => asked<{ version: EnvironmentVersion; existing: boolean }>("api/environments/import", "POST", body),
    [topics.environments(), topics.imports(), topics.offers()],
  );

/** Make a suite, or its next version (which its name then points to). */
export const useSaveSuite = (name: string) =>
  useWrite(async (body: Record<string, unknown>) =>
    asked<{ suite: string; version: string; number: number }>(`api/suites/${encodeURIComponent(name)}`, "POST", body), [topics.evals()]);

/** Ask for a run or an eval: the monitor checks its settings against the cluster and submits its job (a `Refused` error
 * holds the findings that refuse it). */
export const useLaunch = () =>
  useWrite(async (launch: LaunchAsking) => (await asked<{ launch: Launch }>("api/launches", "POST", launch)).launch, [topics.launches()]);

/** What the monitor says of a run's settings as the New run form has them now (none: nothing to check yet): its
 * refusals and notes, what it trains, one step's spend and its environment's slots. The answer before is kept while the
 * next is read. */
export const useCheck = (launch: LaunchAsking | null) =>
  useQuery({
    queryKey: ["check", launch],
    enabled: launch != null,
    staleTime: Infinity,
    refetchInterval: false,
    retry: false,
    placeholderData: keepPreviousData,
    queryFn: () => asked<Checked>("api/launches/check", "POST", launch),
  });

export const usePresets = () => useTopic<Presets>(topics.presets());

/** A preset's versions, oldest first. */
export const usePreset = (name: string) => useTopic<PresetVersions>(topics.preset(name), Boolean(name));

/** Save settings as a preset's next version. */
export const useSavePreset = () =>
  useWrite(
    async ({ name, settings, note }: { name: string; settings: Record<string, unknown>; note?: string }) =>
      (await asked<{ preset: Preset }>(`api/presets/${encodeURIComponent(name)}`, "POST", { settings, note: note ?? "" })).preset,
    [topics.presets(), topics.offers()],
  );

/** Delete a preset (its versions stay readable by number). */
export const useDeletePreset = () =>
  useWrite(async (name: string) => asked<{ deleted: string }>(`api/presets/${encodeURIComponent(name)}`, "DELETE"), [topics.presets(), topics.offers()]);

/** Ask a launch to stop: one whose job was not made is stopped at once; a job going is asked to stop. */
export const useStop = () =>
  useWrite(async (id: string) => (await asked<{ launch: Launch }>(`api/launches/${encodeURIComponent(id)}/stop`, "POST")).launch, [topics.launches()]);

/** Pause a run, or resume it: in place while its driver is there, else its job submitted again with its settings. */
export const useRunControl = (run: string) =>
  useWrite(async (action: "pause" | "resume") => asked<unknown>(`api/runs/${encodeURIComponent(run)}/${action}`, "POST"), [topics.system(), topics.launches()]);

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
export const useCheckpointEvals = (id: string | null | undefined) => useTopic<CheckpointEvals>(topics.checkpointEvals(id ?? ""), Boolean(id));

/** Every subject (a checkpoint or a base model) that has had an eval. */
export const useEvalSubjects = () => useTopic<EvalSubjects>(topics.evalSubjects());

/** Every eval a subject has had. */
export const useHistory = (kind: SubjectKind, id: string) => useTopic<SubjectHistory>(topics.history(kind, id));

/** A checkpoint's line from the base model, with each point's scores. */
export const usePath = (id: string | null | undefined) => useTopic<Path>(topics.path(id ?? ""), Boolean(id));

/** A training run's settings, and what is wanted of them. */
export const useRunSettings = (run: string) => useTopic<RunSettings>(topics.settings(run));

/** Want some of a run's changeable settings from its next step on. */
export const useWantSettings = (run: string) =>
  useWrite(async (settings: Record<string, unknown>) => asked<{ desired: unknown }>(topics.settings(run).path, "POST", { settings }), [topics.settings(run)]);

export const useLineage = () => useTopic<Lineage>(topics.checkpoints());

export const useGroup = (run: string, number: number) => useTopic<Group>(topics.group(run, number));

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
