// Places: every view has an address after the page's `#`, so it can be linked to and comes back on reload. Each is
// on one of the pages: the runs, the checkpoints, the evals, the statistics, the machines.

import { useLocation } from "react-router-dom";

export type Page = "runs" | "checkpoints" | "evals" | "statistics" | "machines";

export type Place =
  | { page: "runs"; kind: "runs" }
  | { page: "runs"; kind: "run"; run: string }
  | { page: "runs"; kind: "step"; run: string; number: number }
  | { page: "runs"; kind: "group"; run: string; number: number }
  | { page: "runs"; kind: "episode"; id: string; slot: string | null }
  | { page: "runs"; kind: "outside" }
  | { page: "runs"; kind: "launch" }
  | { page: "checkpoints"; kind: "checkpoints"; sample: boolean }
  | { page: "checkpoints"; kind: "checkpoint"; id: string }
  | { page: "evals"; kind: "evals" }
  | { page: "evals"; kind: "suite"; suite: string }
  | { page: "evals"; kind: "eval"; run: string }
  | { page: "statistics"; kind: "statistics"; section: string | null }
  | { page: "machines"; kind: "machines" }
  | { page: "machines"; kind: "host"; host: string; role: string | null };

export const runPlace = (run: string) => `/run/${encodeURIComponent(run)}`;
export const groupPlace = (run: string, number: number) => `${runPlace(run)}/group/${number}`;
export const stepPlace = (run: string, number: number) => `${runPlace(run)}/step/${number}`;
export const episodePlace = (id: string, slot?: string | null) =>
  `/episode/${encodeURIComponent(id)}${slot ? `/${encodeURIComponent(slot)}` : ""}`;
export const launchPlace = "/runs/new";
export const checkpointPlace = (id: string) => `/checkpoint/${encodeURIComponent(id)}`;
export const checkpointsPlace = (sample: boolean) => `/checkpoints${sample ? "/sample" : ""}`;
export const evalsPlace = "/evals";
export const suitePlace = (suite: string) => `/evals/${encodeURIComponent(suite)}`;
export const evalPlace = (run: string) => `/eval/${encodeURIComponent(run)}`;
export const statisticsPlace = (section?: string | null) => `/statistics${section ? `/${section}` : ""}`;
export const machinesPlace = "/machines";
export const hostPlace = (host: string, role?: string | null) =>
  `/machines/${encodeURIComponent(host)}${role ? `/${encodeURIComponent(role)}` : ""}`;

export const PAGES: [Page, string, string][] = [
  ["runs", "Runs", "/runs"],
  ["checkpoints", "Checkpoints", "/checkpoints"],
  ["evals", "Evals", "/evals"],
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
  // (`versions` and `version` are what these pages were called: links to them still open them)
  if (parts[0] === "checkpoints" || parts[0] === "versions") return { page: "checkpoints", kind: "checkpoints", sample: parts[1] === "sample" };
  if ((parts[0] === "checkpoint" || parts[0] === "version") && parts[1]) return { page: "checkpoints", kind: "checkpoint", id: parts[1] };
  if (parts[0] === "eval" && parts[1]) return { page: "evals", kind: "eval", run: parts[1] };
  if (parts[0] === "evals") return parts[1] ? { page: "evals", kind: "suite", suite: parts[1] } : { page: "evals", kind: "evals" };
  if (parts[0] === "machines") return parts[1] ? { page: "machines", kind: "host", host: parts[1], role: parts[2] || null } : { page: "machines", kind: "machines" };
  // (the machines were a section of the statistics, and `system` before that: links to them open the machines)
  if (parts[0] === "system" || (parts[0] === "statistics" && parts[1] === "machines")) return { page: "machines", kind: "machines" };
  if (parts[0] === "statistics") return { page: "statistics", kind: "statistics", section: parts[1] || null };
  return { page: "runs", kind: "runs" };
}

export const usePlace = (): Place => placeOf(useLocation().pathname);
