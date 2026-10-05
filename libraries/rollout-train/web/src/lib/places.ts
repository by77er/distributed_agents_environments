// Places: every view has an address after the page's `#`, so it can be linked to and comes back on reload. Each is
// on one of the pages: the runs, the checkpoints, the evals, the environments, the statistics, the machines.

import { useLocation } from "react-router-dom";
import type { SubjectKind } from "../api/types";

export type Page = "runs" | "checkpoints" | "evals" | "environments" | "statistics" | "machines";

export type Place =
  | { page: "runs"; kind: "runs" }
  | { page: "runs"; kind: "run"; run: string }
  | { page: "runs"; kind: "step"; run: string; number: number }
  | { page: "runs"; kind: "group"; run: string; number: number }
  | { page: "runs"; kind: "episode"; id: string; slot: string | null }
  | { page: "runs"; kind: "outside" }
  | { page: "runs"; kind: "launch" }
  | { page: "runs"; kind: "presets" }
  | { page: "runs"; kind: "preset"; preset: string }
  | { page: "checkpoints"; kind: "checkpoints" }
  | { page: "checkpoints"; kind: "checkpoint"; id: string }
  | { page: "checkpoints"; kind: "base"; model: string }
  | { page: "evals"; kind: "evals" }
  | { page: "evals"; kind: "suite"; suite: string }
  | { page: "evals"; kind: "eval"; run: string }
  | { page: "evals"; kind: "subject"; subject: SubjectKind; id: string }
  | { page: "environments"; kind: "environments" }
  | { page: "environments"; kind: "environment"; environment: string }
  | { page: "statistics"; kind: "statistics"; section: string | null }
  | { page: "machines"; kind: "machines" }
  | { page: "machines"; kind: "host"; host: string; role: string | null };

export const runPlace = (run: string) => `/run/${encodeURIComponent(run)}`;
export const groupPlace = (run: string, number: number) => `${runPlace(run)}/group/${number}`;
export const stepPlace = (run: string, number: number) => `${runPlace(run)}/step/${number}`;
export const episodePlace = (id: string, slot?: string | null) =>
  `/episode/${encodeURIComponent(id)}${slot ? `/${encodeURIComponent(slot)}` : ""}`;
export const launchPlace = "/runs/new";
/** The new run's form, with an environment chosen. */
export const launchOn = (environment: string) => `${launchPlace}?environment=${encodeURIComponent(environment)}`;
/** The New run form with a base model chosen. */
export const launchFor = (model: string) => `${launchPlace}?model=${encodeURIComponent(model)}`;
export const presetsPlace = "/presets";
export const presetPlace = (preset: string) => `${presetsPlace}/${encodeURIComponent(preset)}`;
export const checkpointPlace = (id: string) => `/checkpoint/${encodeURIComponent(id)}`;
/** A base model, a root of the checkpoints' graph, by name. */
export const basePlace = (model: string) => `/base/${encodeURIComponent(model)}`;
export const evalsPlace = "/evals";
export const suitePlace = (suite: string) => `/evals/${encodeURIComponent(suite)}`;
export const evalPlace = (run: string) => `/eval/${encodeURIComponent(run)}`;
/** A subject's history: a checkpoint's by id, a base model's by name. */
export const subjectPlace = (kind: SubjectKind, id: string) => `/evals/${kind}/${encodeURIComponent(id)}`;
export const environmentsPlace = "/environments";
export const environmentPlace = (environment: string) => `/environment/${encodeURIComponent(environment)}`;
export const statisticsPlace = (section?: string | null) => `/statistics${section ? `/${section}` : ""}`;
export const hostPlace = (host: string, role?: string | null) =>
  `/machines/${encodeURIComponent(host)}${role ? `/${encodeURIComponent(role)}` : ""}`;

export const PAGES: [Page, string, string][] = [
  ["runs", "Runs", "/runs"],
  ["checkpoints", "Checkpoints", "/checkpoints"],
  ["evals", "Evals", "/evals"],
  ["environments", "Environments", "/environments"],
  ["statistics", "Statistics", "/statistics"],
  ["machines", "Machines", "/machines"],
];

export function placeOf(pathname: string): Place {
  const parts = pathname.replace(/^\/+/, "").split("/").map(part => decodeURIComponent(part));
  if (parts[0] === "run" && parts[2] === "group") return { page: "runs", kind: "group", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run" && parts[2] === "step") return { page: "runs", kind: "step", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run" && parts[1]) return { page: "runs", kind: "run", run: parts[1] };
  if (parts[0] === "episode" && parts[1]) return { page: "runs", kind: "episode", id: parts[1], slot: parts[2] || null };
  if (parts[0] === "episodes") return { page: "runs", kind: "outside" };
  if (parts[0] === "runs" && parts[1] === "new") return { page: "runs", kind: "launch" };
  if (parts[0] === "presets") return parts[1] ? { page: "runs", kind: "preset", preset: parts[1] } : { page: "runs", kind: "presets" };
  if (parts[0] === "checkpoints") return { page: "checkpoints", kind: "checkpoints" };
  if (parts[0] === "checkpoint" && parts[1]) return { page: "checkpoints", kind: "checkpoint", id: parts[1] };
  if (parts[0] === "base" && parts[1]) return { page: "checkpoints", kind: "base", model: parts.slice(1).join("/") };
  if (parts[0] === "eval" && parts[1]) return { page: "evals", kind: "eval", run: parts[1] };
  if (parts[0] === "evals" && (parts[1] === "checkpoint" || parts[1] === "model") && parts[2]) return { page: "evals", kind: "subject", subject: parts[1], id: parts.slice(2).join("/") };
  if (parts[0] === "evals") return parts[1] ? { page: "evals", kind: "suite", suite: parts[1] } : { page: "evals", kind: "evals" };
  if (parts[0] === "environment" && parts[1]) return { page: "environments", kind: "environment", environment: parts[1] };
  if (parts[0] === "environments") return { page: "environments", kind: "environments" };
  if (parts[0] === "machines") return parts[1] ? { page: "machines", kind: "host", host: parts[1], role: parts[2] || null } : { page: "machines", kind: "machines" };
  if (parts[0] === "statistics") return { page: "statistics", kind: "statistics", section: parts[1] || null };
  return { page: "runs", kind: "runs" };
}

export const usePlace = (): Place => placeOf(useLocation().pathname);
