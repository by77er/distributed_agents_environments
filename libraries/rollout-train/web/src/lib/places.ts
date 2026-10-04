// Places: every view has an address after the page's `#`, so it can be linked to and comes back on reload. Each is
// on one of the three pages: the runs, the versions, the statistics.

import { useLocation } from "react-router-dom";

export type Page = "runs" | "versions" | "statistics";

export type Place =
  | { page: "runs"; kind: "runs" }
  | { page: "runs"; kind: "run"; run: string }
  | { page: "runs"; kind: "step"; run: string; number: number }
  | { page: "runs"; kind: "group"; run: string; number: number }
  | { page: "runs"; kind: "episode"; id: string; slot: string | null }
  | { page: "runs"; kind: "outside" }
  | { page: "runs"; kind: "launch" }
  | { page: "versions"; kind: "versions"; sample: boolean }
  | { page: "versions"; kind: "version"; id: string }
  | { page: "statistics"; kind: "statistics"; section: string | null };

export const runPlace = (run: string) => `/run/${encodeURIComponent(run)}`;
export const groupPlace = (run: string, number: number) => `${runPlace(run)}/group/${number}`;
export const stepPlace = (run: string, number: number) => `${runPlace(run)}/step/${number}`;
export const episodePlace = (id: string, slot?: string | null) =>
  `/episode/${encodeURIComponent(id)}${slot ? `/${encodeURIComponent(slot)}` : ""}`;
export const launchPlace = "/runs/new";
export const versionPlace = (id: string) => `/version/${encodeURIComponent(id)}`;
export const versionsPlace = (sample: boolean) => `/versions${sample ? "/sample" : ""}`;
export const statisticsPlace = (section?: string | null) => `/statistics${section ? `/${section}` : ""}`;

export const PAGES: [Page, string, string][] = [
  ["runs", "Runs", "/runs"],
  ["versions", "Versions", "/versions"],
  ["statistics", "Statistics", "/statistics"],
];

export function placeOf(pathname: string): Place {
  const parts = pathname.replace(/^\/+/, "").split("/").map(part => decodeURIComponent(part));
  if (parts[0] === "run" && parts[2] === "group") return { page: "runs", kind: "group", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run" && parts[2] === "step") return { page: "runs", kind: "step", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run" && parts[1]) return { page: "runs", kind: "run", run: parts[1] };
  if (parts[0] === "episode" && parts[1]) return { page: "runs", kind: "episode", id: parts[1], slot: parts[2] || null };
  if (parts[0] === "episodes") return { page: "runs", kind: "outside" };
  if (parts[0] === "runs" && parts[1] === "new") return { page: "runs", kind: "launch" };
  if (parts[0] === "versions") return { page: "versions", kind: "versions", sample: parts[1] === "sample" };
  if (parts[0] === "version" && parts[1]) return { page: "versions", kind: "version", id: parts[1] };
  if (parts[0] === "statistics") return { page: "statistics", kind: "statistics", section: parts[1] || null };
  if (parts[0] === "system") return { page: "statistics", kind: "statistics", section: "machines" };  // (the machines are a section of the statistics)
  return { page: "runs", kind: "runs" };
}

export const usePlace = (): Place => placeOf(useLocation().pathname);
