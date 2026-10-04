// Every checkpoint, as a graph: each base model a root, and under it a lane for each run with the checkpoints it made from
// the left (folded to the ones that matter until it is opened); a run that starts from another's checkpoint hangs under
// that run's lane, and the lines between lanes say what came from what. Below it the distillations, each with what its
// mode means in words; the trainers with their queues, the inference workers with what each serves, and evaluations.
// What no run writes yet comes from the sample fixture, when it is asked for, and is marked so.

import { memo, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useLineage, useSystem } from "../api/queries";
import type { Lineage, LineageRun, LineageCheckpoint, Suite, Trainer, Worker } from "../api/types";
import { QueueChart, Sized } from "../components/charts";
import { Marks } from "../components/checkpoints";
import { anySolved } from "../components/evals";
import { Card, Empty, Head, Kpi, Kpis, Mark, SampleChip, SectionTitle, solvedClass, Spec, Specs, Table, Twist } from "../components/ui";
import { clock, figure, mean, span } from "../lib/format";
import { runPlace, checkpointPlace, checkpointsPlace, suitePlace } from "../lib/places";
import { useFolds } from "../lib/stored";

const LANE = 78, COLUMN = 62, PAD = 34;
const OUTSIDE = "(outside a run)";
const short = (name: string | null | undefined) => (name ? String(name).split("/").at(-1) : "–");
const modeKind = (mode: string | null | undefined) => (mode === "on-policy" ? "violet" : mode === "off-policy" ? "warm" : "accent");
const lifeKind = (state: string) => ({ serving: "good", "rolling out": "warm", resharding: "violet", resharded: "accent" } as Record<string, string>)[state] ?? "";
const ago = (lineage: Lineage, at: number | null | undefined) => (at ? span(Math.max(0, lineage.now - at)) : "–");

interface Index {
  checkpoints: Map<string, LineageCheckpoint>;
  runs: Map<string, LineageRun>;
  scores: Map<string, Suite["subjects"][number] & { suite: string; starts: number }>;
  shortOf: (id: string | null | undefined) => string;
  runOf: (id: string | null | undefined) => string;
}

/** What each checkpoint is, wherever it is drawn: where it came from, the run that made it, its evaluations. */
function indexOf(lineage: Lineage): Index {
  const checkpoints = new Map(lineage.checkpoints.map(checkpoint => [checkpoint.id, checkpoint])), runs = new Map(lineage.runs.map(run => [run.run, run]));
  const scores: Index["scores"] = new Map();
  for (const suite of lineage.evaluations) for (const subject of suite.subjects) {
    if (subject.checkpoint && !scores.has(subject.checkpoint)) scores.set(subject.checkpoint, { ...subject, suite: suite.suite, starts: suite.starts.length });
  }
  return {
    checkpoints, runs, scores,
    shortOf: id => (id ? checkpoints.get(id)?.short ?? String(id).slice(0, 12) : "the base model"),
    runOf: id => (id ? runs.get(id)?.name ?? id : ""),
  };
}

interface Lane {
  key: string;
  base?: string;
  outside?: string[];
  run?: LineageRun;
  checkpoints: LineageCheckpoint[];
  depth: number;
}

/** The lanes, in order: a lane for each base model (its root), and under it each run whose first checkpoint was trained
 * from it; under a run, each run that starts from one of its checkpoints (a fork, or a distillation's start). Checkpoints
 * this ledger does not have, but that something here starts from, are in a lane of their own at the top. */
function lanesOf(lineage: Lineage, index: Index): Lane[] {
  const byRun = new Map<string, LineageCheckpoint[]>();
  for (const checkpoint of lineage.checkpoints) {
    const key = checkpoint.by?.run ?? OUTSIDE;
    byRun.set(key, [...(byRun.get(key) ?? []), checkpoint]);
  }
  for (const run of lineage.runs) if (!byRun.has(run.run)) byRun.set(run.run, []);
  for (const line of byRun.values()) line.sort((a, b) => a.depth - b.depth || a.made - b.made);
  const outside = new Set(lineage.outside);
  const source = (key: string): string => {  // (the lane a run's lane hangs under)
    const first = byRun.get(key)![0], run = index.runs.get(key);
    const from = run?.kind === "distill" ? run.from ?? run.teachers[0] : first?.parents[0] ?? run?.from;
    if (!from) return `base:${first?.base ?? lineage.bases[0] ?? "the base model"}`;
    if (outside.has(from)) return "outside";
    const owner = index.checkpoints.get(from)?.by?.run ?? OUTSIDE;
    return owner === key ? `base:${first?.base ?? "the base model"}` : owner;
  };
  const children = new Map<string, string[]>();
  for (const key of byRun.keys()) children.set(source(key), [...(children.get(source(key)) ?? []), key]);
  const made = (key: string) => byRun.get(key)![0]?.made ?? Infinity;
  const lanes: Lane[] = [], seen = new Set<string>();
  const visit = (key: string, depth: number) => {
    for (const child of [...(children.get(key) ?? [])].sort((a, b) => made(a) - made(b))) {
      if (seen.has(child)) continue;
      seen.add(child);
      lanes.push({ key: child, run: index.runs.get(child), checkpoints: byRun.get(child)!, depth });
      visit(child, depth + 1);
    }
  };
  if (outside.size) { lanes.push({ key: "outside", outside: [...outside], checkpoints: [], depth: 0 }); visit("outside", 1); }
  const bases = [...new Set([...lineage.bases, ...[...children.keys()].filter(key => key.startsWith("base:")).map(key => key.slice(5))])];
  for (const base of bases) { lanes.push({ key: `base:${base}`, base, checkpoints: [], depth: 0 }); visit(`base:${base}`, 1); }
  for (const key of byRun.keys()) if (!seen.has(key)) { lanes.push({ key, run: index.runs.get(key), checkpoints: byRun.get(key)!, depth: 1 }); seen.add(key); }
  return lanes;
}

type Item =
  | { kind: "base"; name: string }
  | { kind: "checkpoint"; name: string; outside?: boolean }
  | { kind: "distill"; run: LineageRun }
  | { kind: "gap"; count: number; names: string[] };

const itemId = (item: Item) => item.kind === "checkpoint" ? `v:${item.name}` : item.kind === "distill" ? `d:${item.run.run}` : item.kind === "base" ? `b:${item.name}` : `g:${item.names[0]}`;
const edgeFrom = (edge: Lineage["edges"][number]) => (edge.from.startsWith("base:") ? `b:${edge.from.slice(5)}` : `v:${edge.from}`);
const edgeTo = (edge: Lineage["edges"][number]) => (edge.kind === "teach" || edge.kind === "start" ? `d:${edge.to}` : `v:${edge.to}`);

/** A lane's items, left to right: a base's root; or a run's checkpoints, the distillation that made them (before the first
 * it made), and, while the lane is folded, a gap for each stretch of checkpoints that nothing points at. */
function itemsOf(lane: Lane, open: boolean, anchors: Set<string>): Item[] {
  if (lane.base) return [{ kind: "base", name: lane.base }];
  if (lane.outside) return lane.outside.map(name => ({ kind: "checkpoint", name, outside: true }));
  const items: Item[] = [], checkpoints = lane.checkpoints;
  let hidden: string[] = [];
  const flush = () => { if (hidden.length) items.push({ kind: "gap", count: hidden.length, names: hidden }); hidden = []; };
  if (lane.run?.kind === "distill") items.push({ kind: "distill", run: lane.run });
  checkpoints.forEach((checkpoint, place) => {
    const shown = open || place === 0 || place === checkpoints.length - 1 || anchors.has(checkpoint.id) || checkpoint.bookmarks.length
      || !["written", "superseded"].includes(checkpoint.life.state);
    if (shown) { flush(); items.push({ kind: "checkpoint", name: checkpoint.id }); } else hidden.push(checkpoint.id);
  });
  flush();
  return items;
}

/** What an edge between lanes says, in words. */
function said(edge: Lineage["edges"][number], index: Index): string {
  switch (edge.kind) {
    case "base": return `${index.shortOf(edge.to)} was trained from the base model ${edge.from.slice(5)}`;
    case "trained": return `${index.shortOf(edge.to)} was trained from ${index.shortOf(edge.from)}: a fork`;
    case "learned": return `${index.shortOf(edge.to)} also learned from ${index.shortOf(edge.from)}`;
    case "teach": return `${index.shortOf(edge.from)} teaches ${index.runOf(edge.to)} — ${edge.says ?? edge.mode ?? ""}`;
    default: return `${index.runOf(edge.to)} starts from ${index.shortOf(edge.from)} — ${edge.says ?? edge.mode ?? ""}`;
  }
}

const LineageGraph = memo(function LineageGraph({ lineage, index, lanes, room, inLedger }: { lineage: Lineage; index: Index; lanes: Lane[]; room: number; inLedger: Set<string> }) {
  const navigate = useNavigate();
  const [folds, fold] = useFolds();
  // (what an edge between lanes points at; an edge along a lane, from a checkpoint to the next its run made, pins nothing)
  const anchors = new Set<string>(lineage.edges.filter(edge => edge.kind !== "trained").flatMap(edge => [edge.from, edge.kind === "learned" || edge.kind === "base" ? edge.to : null]).filter((each): each is string => Boolean(each)));
  for (const name of index.scores.keys()) anchors.add(name);
  for (const edge of lineage.edges) {
    if (edge.kind === "trained" && index.checkpoints.get(edge.from)?.by?.run !== index.checkpoints.get(edge.to)?.by?.run) { anchors.add(edge.from); anchors.add(edge.to); }
  }
  const laid = lanes.map(lane => ({ ...lane, open: Boolean(folds[`lane:${lane.key}`]), items: [] as Item[] }));
  for (const lane of laid) lane.items = itemsOf(lane, lane.open, anchors);
  // Columns: every item stands right of what it comes from, in its lane and across lanes (longest path).
  const before = new Map<string, string[]>(), where = new Map<string, { item: Item; row: number }>();
  laid.forEach((lane, row) => lane.items.forEach((item, place) => {
    where.set(itemId(item), { item, row });
    before.set(itemId(item), place ? [itemId(lane.items[place - 1])] : []);
  }));
  const crossing = (edge: Lineage["edges"][number]) =>
    where.has(edgeFrom(edge)) && where.has(edgeTo(edge)) && where.get(edgeFrom(edge))!.row !== where.get(edgeTo(edge))!.row;
  for (const edge of lineage.edges) if (crossing(edge)) before.get(edgeTo(edge))!.push(edgeFrom(edge));
  const column = new Map<string, number>(), visiting = new Set<string>();
  const columnOf = (id: string): number => {
    if (column.has(id)) return column.get(id)!;
    if (visiting.has(id)) return 0;  // (a cycle cannot happen: a checkpoint is made after what it comes from)
    visiting.add(id);
    const found = Math.max(0, ...before.get(id)!.map(each => columnOf(each) + 1));
    visiting.delete(id);
    column.set(id, found);
    return found;
  };
  for (const id of where.keys()) columnOf(id);
  const columns = Math.max(0, ...column.values()) + 1;
  const width = Math.max(room - 252, PAD * 2 + (columns - 1) * COLUMN + 90), height = laid.length * LANE;
  const x = (id: string) => PAD + column.get(id)! * COLUMN, y = (row: number) => row * LANE + LANE / 2 + 8;
  const kindOfCheckpoint = (name: string) => {
    const checkpoint = index.checkpoints.get(name), run = checkpoint?.by ? index.runs.get(checkpoint.by.run) : undefined;
    return run?.kind === "distill" ? modeKind(run.mode) : "accent";
  };
  return (
    <div className="dag">
      <div className="dag-labels">
        {laid.map(lane => {
          const toggle = () => fold(`lane:${lane.key}`, !lane.open);
          if (lane.outside) {
            return <div key={lane.key} className="lane-label"><span /><b className="muted">Outside this ledger</b><small>{lane.outside.length} checkpoint{lane.outside.length === 1 ? "" : "s"} something here starts from</small></div>;
          }
          if (lane.base) {
            return <div key={lane.key} className="lane-label base-label"><span /><b title={lane.base}>{short(lane.base)}</b><small>base model</small></div>;
          }
          const run = lane.run, first = lane.checkpoints[0], from = first?.parents[0];
          const says = run?.kind === "distill" ? run.says ?? run.mode ?? "distilled"
            : from && index.checkpoints.get(from)?.by?.run !== lane.key
              ? `forked from ${index.shortOf(from)}${index.checkpoints.get(from)?.by ? ` (${index.checkpoints.get(from)!.by!.name})` : ""}` : "from the base model";
          const sample = lane.checkpoints.some(checkpoint => checkpoint.sample) || run?.sample;
          const name = run?.name ?? (lane.key === OUTSIDE ? "made outside a run" : lane.key);
          return (
            <div key={lane.key} className={`lane-label${run?.kind === "distill" ? " distill-label" : ""}`} style={{ paddingLeft: 4 + Math.min(lane.depth, 3) * 10 }} onClick={toggle} title={lane.open ? "fold the lane" : "show every checkpoint"}>
              <Twist open={lane.open} onToggle={toggle} />
              <b title={run ? `id: ${run.run}` : undefined}>{run && !sample && inLedger.has(run.run) ? <Link to={runPlace(run.run)} onClick={event => event.stopPropagation()}>{name}</Link> : name}</b>
              <small className="says">{lane.checkpoints.length} checkpoints · {says}</small>
              <span className="lane-tags">{run?.kind === "distill" ? <span className={`chip t-${modeKind(run.mode)}`}>{run.mode}</span> : null}{sample ? <SampleChip /> : null}</span>
            </div>
          );
        })}
      </div>
      <div className="dag-frame">
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label="checkpoints as a graph, from their base models" className="dag-drawing">
          {laid.map((lane, row) => <rect key={lane.key} x={0} y={row * LANE} width={width} height={LANE} className={`lane-band${row % 2 ? " odd" : ""}${lane.checkpoints.some(checkpoint => checkpoint.sample) ? " sample" : ""}`} />)}
          {/* Along each lane: a line from item to item, in the color of what made the later one. */}
          {laid.map((lane, row) => lane.items.map((item, place) => {
            if (!place) return null;
            const from = lane.items[place - 1];
            const kind = item.kind === "gap" || from.kind === "gap" || lane.outside ? "quiet dash" : item.kind === "checkpoint" ? kindOfCheckpoint(item.name) : "quiet";
            return <line key={`${lane.key}${itemId(item)}`} x1={x(itemId(from))} x2={x(itemId(item))} y1={y(row)} y2={y(row)} className={kind.split(" ").map(each => (each === "dash" ? "dash" : `s-${each}`)).join(" ")} />;
          }))}
          {/* Between lanes: from a base model to the first checkpoint of a line, a fork, a teacher, a distillation's start. */}
          {lineage.edges.map((edge, place) => {
            if (!crossing(edge)) return null;
            const from = where.get(edgeFrom(edge))!, to = where.get(edgeTo(edge))!;
            const x1 = x(edgeFrom(edge)), y1 = y(from.row), x2 = x(itemId(to.item)) - (to.item.kind === "distill" ? 9 : 7), y2 = y(to.row);
            const middle = x1 + Math.max(18, (x2 - x1) * 0.55);
            const kind = edge.kind === "base" || edge.kind === "trained" ? "s-quiet" : edge.kind === "learned" ? "s-quiet dash" : `s-${modeKind(edge.mode)}${edge.kind === "start" ? " dash" : ""}`;
            return <path key={`e${place}`} d={`M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2} ${y2}`} className={`edge ${kind}`}><title>{said(edge, index)}</title></path>;
          })}
          {/* The items, over the lines; and above each run's lane, its name where its stretch begins. */}
          {laid.map((lane, row) => lane.items.map((item, place) => {
            const id = itemId(item), cx = x(id), cy = y(row);
            if (item.kind === "base") {
              return (
                <g key={id}>
                  <g className="base"><rect x={cx - 8} y={cy - 8} width={16} height={16} rx={3} className="f-quiet" /><title>{`the base model ${item.name}: every line here grows from it`}</title></g>
                </g>
              );
            }
            if (item.kind === "gap") {
              return (
                <g key={id} className="gap" onClick={() => fold(`lane:${lane.key}`, true)}>
                  <rect x={cx - 15} y={cy - 9} width={30} height={18} rx={9} />
                  <text x={cx} y={cy + 3.5} textAnchor="middle">+{item.count}</text>
                  <title>{`${item.count} more checkpoints: ${index.shortOf(item.names[0])} to ${index.shortOf(item.names.at(-1))} (open the lane)`}</title>
                </g>
              );
            }
            if (item.kind === "distill") {
              const run = item.run, kind = modeKind(run.mode);
              return (
                <g key={id}>
                  <g className="distill">
                    <path d={`M ${cx - 9} ${cy} L ${cx} ${cy - 9} L ${cx + 9} ${cy} L ${cx} ${cy + 9} Z`} className={`f-${kind}`} />
                    <title>{`${run.name}: distil ${run.teachers.map(index.shortOf).join(" + ")}${run.from ? `, from ${index.shortOf(run.from)}` : ""}\n${run.says ?? ""}\nobjective ${run.objective ?? "–"}`}</title>
                  </g>
                  <text x={cx} y={cy + 23} textAnchor="middle" className={`t-${kind}`}>{run.mode}</text>
                </g>
              );
            }
            const checkpoint = index.checkpoints.get(item.name);
            if (!checkpoint) {  // (a checkpoint this ledger does not have)
              return (
                <g key={id}>
                  <circle cx={cx} cy={cy} r={6} className="dot-outside"><title>{`${item.name}: not in this ledger`}</title></circle>
                  <text x={cx} y={cy + 22} textAnchor="middle" className="v">{item.name.length > 10 ? `…${item.name.slice(-9)}` : item.name}</text>
                </g>
              );
            }
            const life = checkpoint.life, workers = Object.entries(life.workers).filter(([, held]) => held.until == null).map(([worker]) => worker);
            const score = index.scores.get(item.name);
            const real = !checkpoint.sample && inLedger.has(checkpoint.id);
            const share = score?.solved != null ? score.solved / Math.max(1, score.played) : null;  // (none: its task does not say)
            const scoreKind = share == null ? "quiet" : share >= 0.6 ? "good" : share >= 0.35 ? "warm" : "bad";
            const scoreText = score ? (score.solved == null ? figure(score.reward) : `${score.solved}/${score.played}`) + (score.played < score.starts ? "…" : "") : "";
            const first = place === (lane.items[0]?.kind === "distill" ? 1 : 0);
            return (
              <g key={id}>
                {first && lane.run ? <text x={cx - 6} y={cy - 26} className="stretch">{lane.run.name}</text> : null}
                <g className={`checkpoint${real ? " link" : ""}`} onClick={() => { if (real) navigate(checkpointPlace(checkpoint.id)); }}>
                  {["serving", "rolling out", "resharding"].includes(life.state) ? <circle cx={cx} cy={cy} r={10.5} className={`ring ring-${lifeKind(life.state)}`} /> : null}
                  <circle cx={cx} cy={cy} r={6} className={checkpoint.kept ? `dot-${kindOfCheckpoint(item.name)}` : "dot-released"} />
                  <text x={cx} y={cy + 22} textAnchor="middle" className="v">{checkpoint.by?.step != null ? `S${checkpoint.by.step}` : checkpoint.short}</text>
                  {checkpoint.bookmarks.length ? <text x={cx} y={cy - 13} textAnchor="middle" className="bookmark">{checkpoint.bookmarks.join(", ")}</text> : null}
                  {score ? <text x={cx} y={cy - (checkpoint.bookmarks.length ? 24 : 13)} textAnchor="middle" className={`score t-${scoreKind}`}>{scoreText}</text> : null}
                  <title>{[
                    `${checkpoint.id} · depth ${checkpoint.depth}${checkpoint.kind === "full" ? " · full weights" : ""}${checkpoint.sample ? " (sample)" : ""}`,
                    `made ${clock(checkpoint.made)}${checkpoint.by ? ` by ${checkpoint.by.name}${checkpoint.by.step != null ? ` at step ${checkpoint.by.step}` : ""}` : ""}, from ${checkpoint.parents.map(index.shortOf).join(" + ") || (checkpoint.base && index.checkpoints.has(checkpoint.base) ? index.shortOf(checkpoint.base) : `the base model ${checkpoint.base ?? ""}`)}`,
                    checkpoint.bookmarks.length ? `bookmarks: ${checkpoint.bookmarks.join(", ")}` : null,
                    `${life.state}${workers.length ? ` on ${workers.join(", ")}` : ""}${life.waiting ? ` · ${life.waiting} requests waiting` : ""}${life.latest_of ? ` · ${index.runOf(life.latest_of)}'s latest` : ""}`,
                    checkpoint.metrics.kl_moved != null ? `moved ${checkpoint.metrics.kl_moved.toFixed(4)} from its parent` : null,
                    checkpoint.kept ? "its weights are kept" : "released: its weights are gone, its record stays",
                    score ? `${score.suite}: ${score.solved == null ? `mean reward ${figure(score.reward)} over ${score.played}` : `solved ${score.solved} of ${score.played}`}${score.played < score.starts ? ` (${score.starts - score.played} starts to play)` : ""}` : null,
                  ].filter(Boolean).join("\n")}</title>
                </g>
              </g>
            );
          }))}
        </svg>
      </div>
    </div>
  );
});

/** Every distillation, and what its mode means: whose samples the student is trained on, and who scores them. */
function Distillations({ lineage, index }: { lineage: Lineage; index: Index }) {
  const distills = lineage.runs.filter(run => run.kind === "distill");
  if (!distills.length) return null;
  return (
    <div className="distills">
      {distills.map(run => (
        <div key={run.run} className={`tile rail ${modeKind(run.mode)}`}>
          <header>
            <span className="diamond" style={{ background: `var(--${modeKind(run.mode)})` }} />
            <b>{run.name}</b><span className="what">{run.mode}</span>{run.sample ? <SampleChip /> : null}
          </header>
          <p className="says-text">{run.says ?? "a distillation"}</p>
          <div className="facts">
            <span>teachers <b className="mono">{run.teachers.map(index.shortOf).join(" + ") || "–"}</b></span>
            <span>student starts from <b className="mono">{index.shortOf(run.from)}</b></span>
            <span>objective <b>{run.objective ?? "–"}</b></span>
            {run.data.runs ? <span>samples from <b>{run.data.runs.map(index.runOf).join(", ")}</b>{run.data.episodes ? ` (${run.data.episodes})` : ""}</span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function TrainerTile({ trainer, lineage, index }: { trainer: Trainer; lineage: Lineage; index: Index }) {
  const taking = trainer.queue.filter(entry => entry.state === "taking"), queued = trainer.queue.filter(entry => entry.state === "queued");
  const done = trainer.queue.filter(entry => entry.state === "made" || entry.state === "failed").slice(-3).reverse();
  const waited = trainer.queue.filter(entry => entry.began && entry.queued).map(entry => entry.began! - entry.queued!);
  return (
    <div className={`tile rail ${taking.length ? "violet" : queued.length ? "warm" : ""}`}>
      <header><b>{trainer.trainer}</b><span className="what">{trainer.weights === "full" ? "full weights" : trainer.weights === "lora" ? "LoRA" : "nothing made yet"}{trainer.base ? ` on ${index.checkpoints.has(trainer.base) ? index.shortOf(trainer.base) : short(trainer.base)}` : ""}</span>{trainer.sample ? <SampleChip /> : null}</header>
      <div className="facts">
        <span>{trainer.implicit ? "trains for " : trainer.weights === "full" ? "dedicated to " : "any adapter of its base: "}<b>{(trainer.runs ?? []).map(index.runOf).join(", ")}</b></span>
        {trainer.colocated ? <span>shares the engines' accelerator: they sleep while it steps</span> : trainer.where ? <span>{trainer.where}</span> : null}
        {trainer.implicit ? <span>not registered: the run's own, from its profile</span> : null}
      </div>
      <div className="cells four">
        <div className={`cell ${taking.length ? "violet" : ""}`}><span>taking</span><b>{taking.length}</b><small className="mono">{taking[0] ? index.shortOf(taking[0].makes) : "idle"}</small></div>
        <div className={`cell ${queued.length ? "warm" : "waiting"}`}><span>queued</span><b>{queued.length}</b><small>{queued.length ? `oldest ${ago(lineage, queued[0].queued)}` : "none"}</small></div>
        <div className="cell"><span>waited</span><b>{waited.length && !trainer.implicit ? span(mean(waited)) : "–"}</b><small>{trainer.implicit ? "not recorded" : "mean, queued to taken"}</small></div>
        <div className="cell good"><span>made</span><b>{trainer.queue.filter(entry => entry.state === "made").length}</b><small>checkpoints</small></div>
      </div>
      <div>
        <QueueChart depth={trainer.depth} groups={trainer.groups ?? []} now={lineage.now} width={420} height={92} />
        <div className="legend">
          <span><i style={{ background: "var(--violet)" }} />being taken</span><span><i style={{ background: "var(--warm)" }} />with those queued</span>
          {trainer.groups?.length ? <span><i className="rule" style={{ background: "var(--accent)" }} />groups waiting toward a step</span> : null}
        </div>
      </div>
      <div className="members queue">
        {[...taking, ...queued, ...done].map(entry => (
          <div key={`${entry.run}${entry.step}`} className="member">
            <b>S{entry.step}</b><span className="what">{index.runOf(entry.run)}</span>
            <Mark state={entry.state === "taking" ? "stepping" : entry.state === "queued" ? "queued" : entry.state === "made" ? "committed" : "failed"}>
              {entry.state === "taking" ? `taking, ${ago(lineage, entry.began)}` : entry.state === "queued" ? `queued ${ago(lineage, entry.queued)}` : entry.state}
            </Mark>
            <span className="faint mono" title={entry.makes}>→ {index.shortOf(entry.makes)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function WorkerTile({ worker, lineage, index }: { worker: Worker; lineage: Lineage; index: Index }) {
  const holds = worker.holds?.run ? `${index.runOf(worker.holds.run)}'s line, full weights` : worker.holds?.base ? `${worker.serving.length} of ${worker.adapters ?? "?"} adapter slots` : "the run's engines";
  const waiting = lineage.routing.waiting ?? {};
  return (
    <div className={`tile rail ${worker.serving.length ? "good" : ""}`}>
      <header><b>{worker.worker}</b><span className="what">{holds}</span>{worker.share ? <Mark state={worker.share === "evaluations" ? "queued" : ""}>{worker.share}</Mark> : null}{worker.sample ? <SampleChip /> : null}</header>
      {worker.machine ? <div className="facts"><span>{worker.machine}</span><span>{worker.accelerators}</span>{worker.holds?.base ? <span>{short(worker.holds.base)}</span> : null}</div> : null}
      <div className="chips">
        {worker.serving.length ? worker.serving.map(name => (
          <span key={name} className="chip mono" title={`${name}: ${waiting[name] ?? 0} requests waiting for it`}>{index.shortOf(name)}{waiting[name] ? <b className="waits"> {waiting[name]} waiting</b> : null}</span>
        )) : <span className="none">serves nothing</span>}
      </div>
    </div>
  );
}

/** A checkpoint's way to the engines, as stages: written, resharded (full weights only), rolling out, serving. */
function Way({ checkpoint }: { checkpoint: LineageCheckpoint }) {
  const life = checkpoint.life, at = ({ written: 0, resharding: 1, resharded: 1, "rolling out": 2, serving: 3, superseded: 4 } as Record<string, number>)[life.state] ?? 0;
  const stages = ["written", life.reshard ? "resharded" : "no reshard", "rolling", "serving"];
  return (
    <div className="stages way">
      {stages.map((name, place) => (
        <div key={name} className={[place < at || (place === at && life.state === "resharded") ? "done" : place === at ? `now${life.state === "resharding" || life.state === "rolling out" ? " active" : ""}` : "",
          place === 1 && !life.reshard ? "skipped" : ""].join(" ")}>
          <span>{place === at && life.state === "resharding" ? "resharding" : name}</span>
        </div>
      ))}
    </div>
  );
}

function SuiteCard({ suite, index, order }: { suite: Suite; index: Index; order: (checkpoint: LineageCheckpoint) => number }) {
  const subjects = [...suite.subjects].sort((a, b) => {
    const place = (subject: Suite["subjects"][number]) => {
      const checkpoint = subject.checkpoint ? index.checkpoints.get(subject.checkpoint) : undefined;
      return checkpoint ? order(checkpoint) * 1000 + checkpoint.depth : subject.checkpoint ? 9e8 : 1e9;
    };
    return place(a) - place(b);
  });
  const said = anySolved(subjects);
  return (
    <section className="card">
      <header><h2>Evaluation · {suite.sample ? suite.suite : <Link to={suitePlace(suite.suite)}>{suite.suite}</Link>}</h2><span>{suite.starts.length} fixed starts (row and seed), played by {subjects.length} subjects{suite.sample ? <> <SampleChip /></> : null}</span></header>
      <div className="body">
        <div className="table">
          <table className="evals">
            <thead>
              <tr>
                <th>start</th>
                {subjects.map(subject => (
                  <th key={subject.subject} className="subject">
                    <div title={subject.checkpoint ?? subject.subject}>{subject.checkpoint ? index.shortOf(subject.checkpoint) : subject.model ?? subject.subject}</div>
                    <small>{subject.kind === "model" ? short(subject.subject.split(".").at(-1)) : subject.asked_by === "by hand" ? "" : "scheduled"}</small>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="total">
                <td>{said ? "solved" : "mean reward"}</td>
                {subjects.map(subject => (
                  <td key={subject.subject} className="n">
                    {subject.solved == null ? <b>{figure(subject.reward)}</b> : <b>{subject.solved}/{subject.played}</b>}
                    {subject.played < suite.starts.length ? <small className="faint"> {subject.solved == null ? `${subject.played} of ${suite.starts.length}` : `of ${suite.starts.length}`}</small> : null}
                    {subject.solved == null ? null : <div className="track"><i style={{ width: `${((100 * subject.solved) / Math.max(1, suite.starts.length)).toFixed(1)}%` }} /></div>}
                  </td>
                ))}
              </tr>
              {suite.starts.map(start => (
                <tr key={start.start}>
                  <td className="key" title={start.title ?? ""}>{start.task} · {start.seed}</td>
                  {subjects.map(subject => {
                    const played = subject.results[start.start] ?? [];
                    return (
                      <td key={subject.subject} className="cell-result">
                        {played.length ? played.map((each, place) => <i key={place} className={solvedClass(each.solved)} title={`${subject.subject} on ${start.task} seed ${start.seed}: reward ${figure(each.reward)}${each.solved ? ", solved" : ""}`} />)
                          : <i className="unplayed" title="not played yet" />}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="legend">
          {said ? <><span><i style={{ background: "var(--good)" }} />solved</span><span><i style={{ background: "var(--line-strong)" }} />not solved</span></> : <span><i style={{ background: "var(--quiet)" }} />played</span>}
          <span><i className="hollow" />not played yet</span>
        </div>
      </div>
    </section>
  );
}

export function Checkpoints({ sample }: { sample: boolean }) {
  const { data: lineage } = useLineage(sample);
  const { data: system } = useSystem();
  const [folds, , many] = useFolds();
  const index = useMemo(() => (lineage ? indexOf(lineage) : null), [lineage]);
  const lanes = useMemo(() => (lineage && index ? lanesOf(lineage, index) : []), [lineage, index]);
  const inLedger = useMemo(() => new Set([...(system?.runs ?? []).map(run => run.run), ...(system?.checkpoints ?? []).map(checkpoint => checkpoint.id)]), [system?.runs, system?.checkpoints]);
  if (!lineage || !index) return <Empty>Reading the checkpoints…</Empty>;
  const opened = lanes.filter(lane => folds[`lane:${lane.key}`]).length;
  const all = (open: boolean) => many(Object.fromEntries(lanes.map(lane => [`lane:${lane.key}`, open])));
  const order = new Map(lanes.map((lane, place) => [lane.key, place]));
  const laneOf = (checkpoint: LineageCheckpoint) => checkpoint.by?.run ?? OUTSIDE;
  const heads = new Set(lanes.filter(lane => lane.checkpoints.length).map(lane => lane.checkpoints.at(-1)!.id));
  const moving = lineage.checkpoints.filter(checkpoint => !["superseded", "written"].includes(checkpoint.life.state) || (checkpoint.life.state === "written" && heads.has(checkpoint.id)))
    .sort((a, b) => (order.get(laneOf(a)) ?? 0) - (order.get(laneOf(b)) ?? 0) || b.depth - a.depth);
  const waiting = lineage.routing.waiting ?? {}, totalWaiting = Object.values(waiting).reduce((sum, count) => sum + count, 0);
  return (
    <>
      <Head title="Checkpoints" sub="Every checkpoint grows from a base model, along what it was trained from; each run makes a line of them, and a run that starts from another's checkpoint forks there. Below: what distillations do, what trains the checkpoints, what serves them, and how they play a fixed suite.">
        <Specs>
          <Spec label="base models">{lineage.bases.length}</Spec>
          <Spec label="checkpoints">{lineage.checkpoints.length}</Spec>
          <Spec label="runs">{lineage.runs.length}</Spec>
          <Spec label="bookmarks" kind="accent">{Object.keys(lineage.bookmarks).length}</Spec>
          <Spec label="distillations" kind="warm">{lineage.runs.filter(run => run.kind === "distill").length}</Spec>
          <Spec label="trainers" kind="violet">{lineage.trainers.length}</Spec>
          <Spec label="workers" kind="accent">{lineage.workers.length}</Spec>
          <Spec label="suites">{lineage.evaluations.length}</Spec>
        </Specs>
        <div className="segmented">
          <Link to={checkpointsPlace(false)} className={`seg${sample ? "" : " current"}`}>The ledger</Link>
          <Link to={checkpointsPlace(true)} className={`seg${sample ? " current" : ""}`}>With the sample fixture</Link>
        </div>
      </Head>
      {sample ? (
        <div className="tile rail warm notice">
          <header><b>Sample fixture</b><SampleChip /></header>
          <p className="muted small">Everything marked sample comes from rollout_train/monitor/sample-lineage.json: the tables proposed in docs/research/policy-dag.md (a run's plan, trainers and their queues, resharding, inference workers and their loads, the router's waiting requests, evaluation suites). No run writes them yet. The rest is the ledger's, and the runs' feeds'.</p>
        </div>
      ) : null}
      <section className="card">
        <header>
          <h2>Lineage</h2>
          <span>a lane per run under its base model, its checkpoints from the left; {opened ? `${opened} open` : "folded to the checkpoints something points at"} · <button type="button" className="linkish" onClick={() => all(true)}>open all</button> · <button type="button" className="linkish" onClick={() => all(false)}>fold all</button></span>
        </header>
        {lineage.checkpoints.length || lineage.bases.length ? <Sized fallback={1100}>{width => <LineageGraph lineage={lineage} index={index} lanes={lanes} room={width} inLedger={inLedger} />}</Sized> : <Empty>The ledger has no checkpoint yet.</Empty>}
        <div className="legend dag-legend">
          <span><i className="rule" style={{ background: "var(--quiet)" }} />trained from (a base model, or a checkpoint of another run: a fork)</span>
          <span><i style={{ background: "var(--accent)" }} />trained by a run on its own groups</span>
          <span><i style={{ background: "var(--warm)" }} />distilled off-policy: on its teachers' samples</span>
          <span><i style={{ background: "var(--violet)" }} />distilled on-policy: it samples, its teachers score each token</span>
          <span><b className="t-accent">name</b> a bookmark</span>
          <span><i className="hollow" />released (weights deleted)</span><span><i className="ring-good" />serving</span>
          <span><i className="ring-warm" />rolling out</span><span><i className="ring-violet" />resharding</span>
          {lineage.evaluations.length ? <span><b className="t-good">9/16</b> solved of the suite played</span> : null}
        </div>
      </section>
      {lineage.runs.some(run => run.kind === "distill") ? (
        <>
          <SectionTitle title="Distillations" note="a student learns from teachers; its mode says whose samples it is trained on" />
          <Distillations lineage={lineage} index={index} />
        </>
      ) : null}
      <SectionTitle title="Trainers" note="finished groups collect into a step; a step waits in its trainer's queue" />
      <div className="tiles wide-tiles">{lineage.trainers.map(trainer => <TrainerTile key={trainer.trainer} trainer={trainer} lineage={lineage} index={index} />)}</div>
      <SectionTitle title="Serving" note="a request names an exact checkpoint, or a run's latest; the router sends it to a worker that has it" />
      <Kpis>
        <Kpi label="Requests waiting" value={lineage.routing.history.length ? String(totalWaiting) : "–"} note={lineage.routing.history.length ? "by the checkpoint they name" : "the router notes none"} />
        <Kpi label="Serving" value={String(moving.filter(checkpoint => checkpoint.life.state === "serving").length)} note="checkpoints on some worker" />
        <Kpi label="Rolling out" value={String(moving.filter(checkpoint => checkpoint.life.state === "rolling out").length)} note="a run's latest, not yet on every worker" />
        <Kpi label="Resharding" value={String(moving.filter(checkpoint => ["resharding", "resharded"].includes(checkpoint.life.state)).length)} note="full weights, for the engines' layout" />
        <Kpi label="Workers" value={String(lineage.workers.length)} note={`${lineage.workers.filter(worker => worker.registered).length} registered`} />
      </Kpis>
      <Card title="On their way to the engines, and served" note="newest first in each run's line">
        <Table
          heads={[["checkpoint"], ["made by"], ["way"], ["workers"], ["waiting", "n"], ["latest of"], ["made", "n"]]}
          keys={moving.map(checkpoint => checkpoint.id)}
          rows={moving.map(checkpoint => {
            const workers = Object.entries(checkpoint.life.workers);
            return [
              <span><b className="mono" title={checkpoint.id}>{checkpoint.short}</b> <Marks names={checkpoint.bookmarks} />{checkpoint.sample ? <> <SampleChip /></> : null}</span>,
              checkpoint.by ? `${checkpoint.by.name}${checkpoint.by.step != null ? ` S${checkpoint.by.step}` : ""}` : "–",
              <Way checkpoint={checkpoint} />,
              <div className="chips">
                {workers.length ? workers.map(([worker, served]) => (
                  <span key={worker} className={`chip${served.until == null ? " on" : " off"}`} title={served.until == null ? `serving since ${clock(served.since)}` : `served ${clock(served.since)} to ${clock(served.until)}`}>{worker}</span>
                )) : <span className="none">{checkpoint.life.state === "resharding" ? "files being rewritten" : "on no worker"}</span>}
              </div>,
              checkpoint.life.waiting ? { text: String(checkpoint.life.waiting), kind: "bad" } : "0",
              checkpoint.life.latest_of ? index.runOf(checkpoint.life.latest_of) : "–",
              ago(lineage, checkpoint.made),
            ];
          })}
          to={moving.map(checkpoint => (inLedger.has(checkpoint.id) ? checkpointPlace(checkpoint.id) : null))}
        />
      </Card>
      <div className="tiles">{lineage.workers.map(worker => <WorkerTile key={worker.worker} worker={worker} lineage={lineage} index={index} />)}</div>
      <Card title="Runs and distillations" note="what made each line">
        <Table
          heads={[["run"], ["kind"], ["from"], ["teachers"], ["trains on"], ["objective"], ["checkpoints"], ["latest"]]}
          keys={lineage.runs.map(run => run.run)}
          rows={lineage.runs.map(run => [
            <span><b title={`id: ${run.run}`}>{run.name}</b>{run.sample ? <> <SampleChip /></> : null}</span>,
            run.kind === "distill" ? <Mark state={modeKind(run.mode) === "violet" ? "stepping" : "queued"}>distil, {run.mode}</Mark> : "train",
            index.shortOf(run.from), run.teachers.map(index.shortOf).join(", ") || "–",
            run.kind === "distill" ? `${run.says ?? ""}${run.data.runs ? ` (from ${run.data.runs.map(index.runOf).join(", ")})` : ""}` : "its own groups",
            run.objective ?? "policy gradient",
            run.checkpoints.length ? `${index.shortOf(run.checkpoints[0])}–${index.shortOf(run.checkpoints.at(-1))}` : "–",
            run.latest ? index.shortOf(run.latest) : "–",
          ])}
          to={lineage.runs.map(run => (inLedger.has(run.run) && !run.sample ? runPlace(run.run) : null))}
        />
      </Card>
      <SectionTitle title="Evaluations" note={lineage.evaluations.length ? "a fixed suite of starts, played by checkpoints and by other models" : ""} />
      {lineage.evaluations.length ? lineage.evaluations.map(suite => <SuiteCard key={suite.suite} suite={suite} index={index} order={checkpoint => order.get(laneOf(checkpoint)) ?? 99} />) : <Empty>No evaluation suite is in the ledger.</Empty>}
    </>
  );
}
