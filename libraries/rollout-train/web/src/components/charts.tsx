// Charts, drawn to scale in SVG: round ticks on one axis each way, a line or column for each series in its own color,
// and under the pointer a rule with every series' value there (the hover is the chart's own state: pointing at one
// chart draws only it again).

import { memo, useCallback, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import type { DoneLine, Run } from "../api/types";
import { clock, day, figure, mean, span, tick } from "../lib/format";
import { outcomeOf, stepOf } from "../lib/model";
import { groupPlace } from "../lib/places";

/** The width an element has to draw in, as it changes. */
export function useWidth<T extends HTMLElement>(fallback = 600): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => setWidth(Math.max(240, Math.floor(element.clientWidth)));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/** Draws its child at the width it is given. */
export function Sized({ children, fallback }: { children: (width: number) => ReactNode; fallback?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(fallback);
  return <div ref={ref} style={{ minWidth: 0 }}>{children(width)}</div>;
}

/** Where a spark's values are drawn, from the lowest of them (or 0) at the bottom to the highest (or 0) at the top, and
 * the height of 0. */
export function sparkPoints(values: number[], width: number, height: number): { points: number[][]; zero: number } {
  const low = Math.min(0, ...values), high = Math.max(0, ...values), range = high - low || 1e-9;
  const step = (width - 6) / Math.max(1, values.length - 1), y = (value: number) => height - 3 - ((value - low) / range) * (height - 10);
  return { points: values.map((value, index) => [3 + index * step, y(value)]), zero: y(0) };
}

export const Spark = memo(function Spark({ values, kind, width, height, fill }: { values: number[]; kind: string; width: number; height: number; fill?: boolean }) {
  if (!values.length) return <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" />;
  const { points, zero } = sparkPoints(values, width, height);
  const [x, y] = points.at(-1)!;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img">
      <line x1={0} x2={width} y1={zero + 0.5} y2={zero + 0.5} className="s-grid" />
      {fill ? <polygon className={`${kind.replace("s-", "f-")}-soft`} points={[`3,${zero}`, ...points.map(point => point.join(",")), `${x},${zero}`].join(" ")} /> : null}
      {values.length > 1 ? <polyline className={kind} points={points.map(point => point.join(",")).join(" ")} /> : null}
      <circle cx={x} cy={y} r={3} className={kind.replace("s-", "f-")} />
    </svg>
  );
});

/** Every group the run is done with: each episode's reward as a dot, the group's mean as a rule, and under the axis what
 * was done with the group; groups stand in the order of their steps, and a rule parts one step's from the next. A
 * column opens its group. */
export const RewardsChart = memo(function RewardsChart({ run, width }: { run: Run; width: number }) {
  const navigate = useNavigate();
  // (in the order of the steps they went into, and within a step by number: a group may finish, and be trained on,
  // before one decided earlier; those toward the next step come last)
  const order = (line: DoneLine) => { const step = stepOf(run, line.group); return step ? run.steps.indexOf(step) : run.steps.length; };
  const lines = [...run.done].sort((a, b) => order(a) - order(b) || a.group - b.group);
  const left = 32, top = 10, plot = 120, band = top + plot + 10, height = band + 28;
  const count = Math.max(lines.length, 12), column = (width - left - 6) / count;
  const ys = scale(lines.flatMap(line => line.rewards), { count: 2 });  // (from the lowest reward, or 0, to the highest)
  const y = (value: number) => top + plot - ((value - ys.low) / (ys.high - ys.low)) * plot;
  const every = Math.max(1, Math.ceil(34 / column)), mark = Math.max(3, Math.min(9, column - 3)), spread = Math.min(3.4, column / 6);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label="rewards by group">
      {ys.ticks.map(value => (
        <g key={value}>
          <line x1={left} x2={width} y1={y(value)} y2={y(value)} className="s-grid" />
          <text x={left - 8} y={y(value) + 3} textAnchor="end">{tick(value)}</text>
        </g>
      ))}
      {lines.map((line, index) => {
        const x = left + (index + 0.5) * column;
        const step = stepOf(run, line.group), before = index ? stepOf(run, lines[index - 1].group) : step;
        const did = line.update ? "f-accent" : line.error || (line.failed && !line.rewards.length) ? "f-bad"
          : line.step_state === "stepping" ? "f-violet" : line.segments ? "f-warm" : "f-hollow";
        const average = mean(line.rewards);
        return (
          <g key={line.group}>
            <rect x={x - column / 2} y={top - 4} width={column} height={height - top} className="f-none column" onClick={() => navigate(groupPlace(run.run, line.group))}>
              <title>{`#${line.group} ${line.task} · ${line.title}\n${line.rewards.map(figure).join(" ") || "no episode"}${line.failed ? ` (${line.failed} failed)` : ""}\n${outcomeOf(line).text}`}</title>
            </rect>
            {step !== before ? <line x1={x - column / 2} x2={x - column / 2} y1={top - 4} y2={height} className="s-grid" pointerEvents="none" /> : null}
            {line.rewards.map((value, place) => (
              <circle key={place} cx={x + (place - (line.rewards.length - 1) / 2) * spread} cy={y(value)} r={Math.min(3.2, Math.max(1.6, column / 5))}
                className={line.solved[place] ? "f-good" : line.solved[place] === false ? "f-quiet" : "f-played"} pointerEvents="none" />
            ))}
            {average != null ? <line x1={x - column * 0.34} x2={x + column * 0.34} y1={y(average)} y2={y(average)} className="s-ink" pointerEvents="none" /> : null}
            <rect x={x - mark / 2} y={band} width={mark} height={mark} rx={2} className={did} pointerEvents="none" />
            {index % every === 0 ? <text x={x} y={band + 23} textAnchor="middle">{String(line.group)}</text> : null}
          </g>
        );
      })}
    </svg>
  );
});

export const BarChart = memo(function BarChart({ values, labels, width, height, onBar }: { values: number[]; labels: string[]; width: number; height: number; onBar?: (index: number) => void }) {
  const top = Math.max(...values, 1e-9), column = width / Math.max(values.length, 12), base = height - 16;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img">
      <line x1={0} x2={width} y1={base + 0.5} y2={base + 0.5} className="s-grid" />
      {values.map((value, index) => {
        const tall = Math.max(1, (value / top) * (base - 6));
        return (
          <g key={index}>
            <rect x={index * column + column * 0.18} y={base - tall} width={column * 0.64} height={tall} rx={3} className="f-violet"
              style={onBar ? { cursor: "pointer" } : undefined} onClick={onBar ? () => onBar(index) : undefined}>
              <title>{`${labels[index]}: ${figure(value)}`}</title>
            </rect>
            {values.length <= 24 || index % Math.ceil(values.length / 24) === 0
              ? <text x={index * column + column / 2} y={height - 3} textAnchor="middle">{labels[index]}</text> : null}
          </g>
        );
      })}
    </svg>
  );
});

// Scales and ticks
const DAY = 86400;
export const TIME_STEPS = [60, 300, 900, 1800, 3600, 7200, 10800, 21600, 43200, DAY, 2 * DAY, 7 * DAY];

/** A scale from the values to round ends, with its ticks: `zero` takes it down (or up) to 0; `min` and `max` fix an end. */
export function scale(values: number[], { zero = true, min, max, count = 4 }: { zero?: boolean; min?: number; max?: number; count?: number } = {}) {
  const finite = values.filter(Number.isFinite);
  let low = min ?? Math.min(...finite, ...(zero ? [0] : []));
  let high = max ?? Math.max(...finite, ...(zero ? [0] : []));
  if (!finite.length && min == null && max == null) [low, high] = [0, 1];
  if (!(high > low)) {
    const pad = Math.abs(high) * 0.1 || 1;
    if (min == null) low -= zero && low === 0 ? 0 : pad;
    if (max == null) high += pad;
  }
  const raw = (high - low) / count, power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(each => each * power).find(each => each >= raw * 0.999) ?? raw;
  if (min == null) low = Math.floor(low / step + 1e-9) * step;
  if (max == null) high = Math.ceil(high / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let value = Math.ceil(low / step - 1e-9) * step; value <= high + step * 1e-6; value += step) ticks.push(+value.toPrecision(12));
  return { low, high, ticks, step };
}

/** Ticks on local clock times (or days), about `count` of them. */
export function timeTicks(low: number, high: number, count: number): [number, string][] {
  const step = TIME_STEPS.find(each => (high - low) / each <= count) ?? 7 * DAY, offset = new Date(low * 1000).getTimezoneOffset() * 60;
  const ticks: [number, string][] = [];
  for (let at = Math.ceil((low - offset) / step) * step + offset; at <= high; at += step) {
    ticks.push([at, step >= DAY || (at - offset) % DAY === 0 ? day(at) : clock(at)]);
  }
  return ticks;
}

export interface Series {
  name: string;
  color: string;
  points: number[][];
  until?: number | null;
  scatter?: number[][];
}

interface LineChartProps {
  series: Series[];
  width: number;
  height?: number;
  time?: boolean;
  y?: { zero?: boolean; min?: number; max?: number };
  stepped?: boolean;
  rules?: { value: number; label: string }[];
  /** Places along x to mark with a rule across the chart and a word at its top. */
  marks?: { x: number; label: string }[];
  label: string;
  /** The span of x drawn at least (beside the points' own). */
  domain?: [number, number];
  format?: (value: number) => string;
  xFormat?: (value: number) => string;
  yTick?: (value: number) => string;
  dots?: boolean;
  gap?: number;
}

/** A line for each series over x (a group's place, a step, or a time with `time`), on one y scale. `stepped` holds each
 * value until the next (a count); `rules` are reference lines, `marks` places along x. */
export const LineChart = memo(function LineChart({ series, width, height = 210, time = false, y = {}, stepped = false, rules = [], marks = [], domain, label, format = figure, xFormat = figure, yTick, dots, gap = Infinity }: LineChartProps) {
  const left = 50, right = 14, top = marks.length ? 34 : 12, bottom = 26;
  const rows: number[] = [];  // (the row each mark's word is on)
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const shown = series.map(each => ({ ...each, points: each.points.filter(point => Number.isFinite(point[1])) })).filter(each => each.points.length);
  const points = shown.flatMap(each => [...each.points, ...(each.scatter ?? [])]).filter(point => Number.isFinite(point[1]));
  let x0 = Math.min(...points.map(point => point[0]), ...marks.map(mark => mark.x), ...(domain ? [domain[0]] : []));
  let x1 = Math.max(...points.map(point => point[0]), ...shown.map(each => each.until ?? -Infinity), ...marks.map(mark => mark.x), ...(domain ? [domain[1]] : []));
  if (!(x1 > x0)) { x0 -= time ? 1800 : 1; x1 += time ? 1800 : 1; }
  const ys = scale([...points.map(point => point[1]), ...rules.map(rule => rule.value)], y);
  const X = (value: number) => left + ((value - x0) / (x1 - x0)) * (width - left - right);
  const Y = (value: number) => top + ((ys.high - value) / (ys.high - ys.low)) * (height - top - bottom);
  const xs = [...new Set(shown.flatMap(each => each.points.map(point => point[0])))].sort((a, b) => a - b);
  const onMove = useCallback((event: React.MouseEvent) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || !xs.length) return;
    const at = x0 + (((event.clientX - box.left) * width) / box.width - left) / (width - left - right) * (x1 - x0);
    setHover(xs.reduce((best, each) => (Math.abs(each - at) < Math.abs(best - at) ? each : best), xs[0]));
  }, [xs, x0, x1, width]);
  if (!points.length) {
    return <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={label} className="chart"><text x={width / 2} y={height / 2} textAnchor="middle">nothing to draw yet</text></svg>;
  }
  const decimals = Math.max(0, -Math.floor(Math.log10(ys.step) + 1e-9));  // (as many as the ticks' step needs)
  const labelOf = yTick ?? ((value: number) => (Math.abs(value) >= 1e4 ? tick(value) : value.toFixed(decimals)));
  const across = Math.max(2, Math.floor((width - left - right) / (time ? 96 : 64)));
  const xTicks: [number, string][] = time ? timeTicks(x0, x1, across)
    : scale([x0, x1], { zero: false, min: x0, max: x1, count: across }).ticks.filter(Number.isInteger).map(value => [value, xFormat(value)]);
  const valueAt = (each: Series, x: number): number | null => {
    if (stepped) {
      const before = each.points.filter(point => point[0] <= x).at(-1);
      return before && (each.until == null || x <= each.until || x === before[0]) ? before[1] : null;
    }
    return each.points.find(point => point[0] === x)?.[1] ?? null;
  };
  return (
    <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={label} className="chart">
      {ys.ticks.map(value => (
        <g key={`y${value}`}>
          <line x1={left} x2={width - right} y1={Y(value)} y2={Y(value)} className="s-grid" />
          <text x={left - 7} y={Y(value) + 3.5} textAnchor="end">{labelOf(value)}</text>
        </g>
      ))}
      {xTicks.map(([value, text]) => (
        <g key={`x${value}`}>
          <line x1={X(value)} x2={X(value)} y1={height - bottom} y2={height - bottom + 4} className="s-grid" />
          <text x={X(value)} y={height - bottom + 15} textAnchor="middle">{text}</text>
        </g>
      ))}
      <line x1={left} x2={width - right} y1={height - bottom + 0.5} y2={height - bottom + 0.5} className="s-axis" />
      {marks.map((mark, place) => {
        // (a word too close to the one before goes on the row under it)
        const near = place > 0 && X(mark.x) - X(marks[place - 1].x) < 130, row = near && rows[place - 1] === 0 ? 1 : 0;
        rows[place] = row;
        const at = 10 + row * 12, end = X(mark.x) > width - right - 130;
        return (
          <g key={`m${mark.x}`} className="mark-x">
            <line x1={X(mark.x)} x2={X(mark.x)} y1={at + 3} y2={height - bottom} className="s-rule" />
            <text x={X(mark.x) + (end ? -4 : 4)} y={at} textAnchor={end ? "end" : "start"}>{mark.label}</text>
          </g>
        );
      })}
      {rules.map(rule => (
        <g key={rule.label}>
          <line x1={left} x2={width - right} y1={Y(rule.value)} y2={Y(rule.value)} className="s-rule" />
          <text x={width - right - 2} y={Y(rule.value) - 4} textAnchor="end">{rule.label}</text>
        </g>
      ))}
      {shown.map(each => {
        const line = each.points;
        let path = `M ${X(line[0][0])} ${Y(line[0][1])}`;
        line.slice(1).forEach(([x, value], place) => {  // (a line breaks where measurements stopped for longer than `gap`)
          path += stepped ? ` H ${X(x)} V ${Y(value)}` : `${x - line[place][0] > gap ? " M" : " L"} ${X(x)} ${Y(value)}`;
        });
        if (stepped && each.until != null && each.until > line.at(-1)![0]) path += ` H ${X(each.until)}`;
        const marked = dots ?? line.length <= 40 ? line : [line.at(-1)!];
        return (
          <g key={each.name}>
            {(each.scatter ?? []).filter(point => Number.isFinite(point[1])).map(([x, value], place) => (
              <circle key={`s${place}`} cx={X(x)} cy={Y(value)} r={2.2} style={{ fill: each.color }} className="scatter" />
            ))}
            <path d={path} className="series" style={{ stroke: each.color }} />
            {marked.map(([x, value]) => <circle key={`d${x}`} cx={X(x)} cy={Y(value)} r={3} className="dot-series" style={{ fill: each.color }} />)}
          </g>
        );
      })}
      {hover != null ? (() => {
        const rows = shown.map(each => [each, valueAt(each, hover)] as const).filter((row): row is readonly [Series, number] => row[1] != null);
        const lines: [Series | null, string][] = [[null, time ? `${day(hover)} ${clock(hover)}` : xFormat(hover)], ...rows.map(([each, value]) => [each, `${each.name}  ${format(value)}`] as [Series, string])];
        const wide = Math.max(...lines.map(([, text]) => text.length)) * 6.3 + 26, tall = lines.length * 15 + 8;
        const tipX = X(hover) + 10 + wide > width - right ? X(hover) - 10 - wide : X(hover) + 10, tipY = top + 2;
        return (
          <g className="hover" pointerEvents="none">
            <line x1={X(hover)} x2={X(hover)} y1={top} y2={height - bottom} className="s-cross" />
            {rows.map(([each, value]) => <circle key={each.name} cx={X(hover)} cy={Y(value)} r={4} className="dot-series" style={{ fill: each.color }} />)}
            <rect x={tipX} y={tipY} width={wide} height={tall} rx={3} className="tip" />
            {lines.map(([each, text], place) => (
              <g key={place}>
                {each ? <rect x={tipX + 8} y={tipY + 9 + place * 15} width={8} height={8} rx={1} style={{ fill: each.color }} /> : null}
                <text x={tipX + (each ? 21 : 8)} y={tipY + 16 + place * 15} className={each ? "tip-text" : "tip-head"}>{text}</text>
              </g>
            ))}
          </g>
        );
      })() : null}
      <rect x={left} y={top} width={width - left - right} height={height - top - bottom} className="f-none" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
    </svg>
  );
});

export interface Span {
  from: number;
  to: number;
  values: number[];
}

/** Columns over time, one for each span, stacked by series (a value per series in each), on one scale; each column
 * says what it holds when pointed at. */
export const ColumnChart = memo(function ColumnChart({ spans, series, width, height = 190, label, format = figure }: { spans: Span[]; series: { name: string; color: string }[]; width: number; height?: number; label: string; format?: (value: number) => string }) {
  const left = 50, right = 14, top = 12, bottom = 26;
  if (!spans.length) {
    return <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={label} className="chart"><text x={width / 2} y={height / 2} textAnchor="middle">nothing to draw yet</text></svg>;
  }
  const x0 = spans[0].from, x1 = spans.at(-1)!.to;
  const ys = scale(spans.map(each => each.values.reduce((sum, value) => sum + value, 0)), {});
  const X = (value: number) => left + ((value - x0) / (x1 - x0)) * (width - left - right);
  const Y = (value: number) => top + ((ys.high - value) / (ys.high - ys.low)) * (height - top - bottom);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={label} className="chart">
      {ys.ticks.map(value => (
        <g key={`y${value}`}>
          <line x1={left} x2={width - right} y1={Y(value)} y2={Y(value)} className="s-grid" />
          <text x={left - 7} y={Y(value) + 3.5} textAnchor="end">{tick(value)}</text>
        </g>
      ))}
      {timeTicks(x0, x1, Math.max(2, Math.floor((width - left - right) / 96))).map(([value, text]) => (
        <g key={`x${value}`}>
          <line x1={X(value)} x2={X(value)} y1={height - bottom} y2={height - bottom + 4} className="s-grid" />
          <text x={X(value)} y={height - bottom + 15} textAnchor="middle">{text}</text>
        </g>
      ))}
      {spans.map(each => {
        const x = X(each.from) + 1, wide = Math.max(1, X(each.to) - X(each.from) - 2);
        let base = 0;
        return (
          <g key={each.from} className="column-stack">
            {each.values.map((value, place) => {
              if (!value) return null;
              const y1 = Y(base), y2 = Y(base + value);
              const bar = <rect key={place} x={x} y={y2} width={wide} height={Math.max(1, y1 - y2 - (base ? 1 : 0))} style={{ fill: series[place].color }} />;
              base += value;
              return bar;
            })}
            <rect x={X(each.from)} y={top} width={X(each.to) - X(each.from)} height={height - top - bottom} className="f-none">
              <title>{[`${day(each.from)} ${clock(each.from)} to ${clock(each.to)}`, ...series.map((one, place) => (each.values[place] ? `${one.name}: ${format(each.values[place])}` : null)).filter(Boolean)].join("\n")}</title>
            </rect>
          </g>
        );
      })}
      <line x1={left} x2={width - right} y1={height - bottom + 0.5} y2={height - bottom + 0.5} className="s-axis" />
    </svg>
  );
});

/** A count of things that happened at times, in spans of a round length (about `count` of them between `from` and `to`),
 * for each series. */
export function spansOf(times: number[][], from: number, to: number, count = 48): { spans: Span[]; step: number } {
  const step = TIME_STEPS.find(each => (to - from) / each <= count) ?? 7 * DAY, offset = new Date(from * 1000).getTimezoneOffset() * 60;
  const start = Math.floor((from - offset) / step) * step + offset, spans: Span[] = [];
  for (let at = start; at <= to; at += step) spans.push({ from: at, to: at + step, values: times.map(() => 0) });
  times.forEach((list, place) => { for (const at of list) { const bucket = spans[Math.floor((at - start) / step)]; if (bucket) bucket.values[place] += 1; } });
  return { spans, step };
}

/** A count over time, drawn as steps up to now: a trainer's queue (what waits, over what is being taken), and the groups
 * of a run that wait toward a step. */
export const QueueChart = memo(function QueueChart({ depth, groups, now, width, height }: { depth: [number, number, number][]; groups: [number, number][]; now: number; width: number; height: number }) {
  const times = [...depth.map(point => point[0]), ...groups.map(point => point[0])];
  if (!times.length) return <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" preserveAspectRatio="none" className="queue-chart" />;
  const start = Math.min(...times), top = Math.max(1, ...depth.map(point => point[1] + point[2]), ...groups.map(point => point[1]));
  const left = 18, base = height - 14;
  const x = (at: number) => left + ((at - start) / Math.max(1, now - start)) * (width - left - 4), y = (value: number) => base - (value / top) * (base - 6);
  const stepped = <T extends number[]>(points: T[], value: (point: T) => number) => {
    let path = "";
    points.forEach((point, place) => {
      const next = points[place + 1]?.[0] ?? now;
      path += `${place ? " L" : "M"} ${x(point[0])} ${y(value(point))} L ${x(next)} ${y(value(point))}`;
    });
    return path;
  };
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" preserveAspectRatio="none" className="queue-chart">
      <line x1={left} x2={width} y1={base + 0.5} y2={base + 0.5} className="s-grid" />
      <text x={left - 5} y={y(top) + 3} textAnchor="end">{String(top)}</text>
      <text x={left - 5} y={base + 3} textAnchor="end">0</text>
      <text x={left} y={height - 2}>{`${span(now - start)} ago`}</text>
      <text x={width - 4} y={height - 2} textAnchor="end">now</text>
      {depth.length ? (
        <>
          <path d={`${stepped(depth, point => point[2])} L ${x(now)} ${base} L ${x(depth[0][0])} ${base} Z`} className="f-violet-soft" />
          <path d={stepped(depth, point => point[2])} className="s-violet" />
          <path d={stepped(depth, point => point[1] + point[2])} className="s-warm" />
        </>
      ) : null}
      {groups.length ? <path d={stepped(groups, point => point[1])} className="s-accent dash" /> : null}
    </svg>
  );
});
