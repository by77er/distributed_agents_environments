// A checkpoint's line as a chart: each suite's score at every point from the base model, by depth, and where the line
// changes run or what its weights are.

import type { Path, PathPoint } from "../api/types";
import type { Series } from "../components/charts";

export interface PathChart {
  /** One line for each suite: [depth, score] where the point was evaluated on it. */
  series: (Series & { suite: string; measure: "solved" | "reward" })[];
  /** What each depth is called: a bookmark that names it, else its shortest id ("base" for the base model). */
  labels: Map<number, string>;
  /** Where the line enters a run or changes what its weights are (full, LoRA). */
  marks: { x: number; label: string }[];
  /** Whether every line is a share solved (else a mean reward, or a mix). */
  solved: boolean;
}

/** A point's name on the chart. */
export const pointLabel = (point: PathPoint): string => (point.id == null ? "base" : point.bookmarks[0] ?? point.short);

const weightsName = (kind: string): string => (kind === "full" ? "full" : kind === "lora" ? "LoRA" : kind);

/** The chart of a checkpoint's line: each suite's share solved at each point where its episodes say, else its mean
 * reward; the points named by bookmark or short id; a mark where the line enters a run (or leaves runs: a merge made
 * outside one) and where its weights change between full and LoRA. */
export function pathChart(path: Path, colorOf: (place: number) => string = place => `var(--series-${(place % 8) + 1})`): PathChart {
  const points = path.points;
  const series = path.suites.map(({ suite, label }, place) => {
    const scored = points.filter(point => point.scores[suite]);
    const measure: "solved" | "reward" = scored.some(point => point.scores[suite].solved != null) ? "solved" : "reward";
    const values = scored
      .map(point => [point.depth, measure === "solved" ? point.scores[suite].solved : point.scores[suite].reward] as [number, number | null])
      .filter((each): each is [number, number] => each[1] != null);
    return { name: label ?? suite, suite: label ?? suite, measure, color: colorOf(place), points: values };
  }).filter(each => each.points.length);
  const labels = new Map(points.map(point => [point.depth, pointLabel(point)]));
  const marks: { x: number; label: string }[] = [];
  points.forEach((point, place) => {
    const before = points[place - 1];
    if (!before || point.id == null) return;
    const said: string[] = [];
    if (point.run !== before.run) said.push(point.run ? point.name ?? point.run : "outside a run");
    if (before.id != null && point.kind !== before.kind) said.push(weightsName(point.kind));
    if (said.length) marks.push({ x: point.depth, label: said.join(" · ") });
  });
  return { series, labels, marks, solved: series.length > 0 && series.every(each => each.measure === "solved") };
}
