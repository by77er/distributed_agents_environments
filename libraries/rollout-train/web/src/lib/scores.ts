// A checkpoint's line as a chart: each suite's score (each environment's, for a suite of several) at every point from the
// base model, by depth, and where the line changes run or what its weights are.

import type { Path, PathPoint, PathScore } from "../api/types";
import type { Series } from "../components/charts";
import { readable } from "./environments";

/** A score at a point: the share solved (none where its episodes do not say) and the mean reward. */
type Scored = { solved: number | null; reward: number | null };

export interface PathChart {
  /** One line for each suite, and for a suite of several environments one for each: [depth, score] where the point was
   * evaluated on it. */
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
 * reward, and for a suite of several environments each environment's apart (their rewards do not compare); the points
 * named by bookmark or short id; a mark where the line enters a run (or leaves runs: a merge made outside one) and where
 * its weights change between full and LoRA. */
export function pathChart(path: Path, colorOf: (place: number) => string = place => `var(--series-${(place % 8) + 1})`): PathChart {
  const points = path.points;
  const lines = path.suites.flatMap(({ suite, label, environments }) =>
    (environments?.length ?? 0) > 1
      ? environments.map(environment => ({ suite, name: `${label ?? suite} · ${readable(environment)}`, of: (score: PathScore): Scored | undefined => score.entries?.[environment] }))
      : [{ suite, name: label ?? suite, of: (score: PathScore): Scored | undefined => score }]);
  const series = lines.map(({ suite, name, of }, place) => {
    const scored: [number, Scored][] = [];
    for (const point of points) {
      const found = point.scores[suite] ? of(point.scores[suite]) : undefined;
      if (found) scored.push([point.depth, found]);
    }
    const measure: "solved" | "reward" = scored.some(([, score]) => score.solved != null) ? "solved" : "reward";
    const values = scored
      .map(([depth, score]) => [depth, measure === "solved" ? score.solved : score.reward] as [number, number | null])
      .filter((each): each is [number, number] => each[1] != null);
    return { name, suite: name, measure, color: colorOf(place), points: values };
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
