// Every checkpoint, as a graph: each base model a root, and under it a lane for each run with the checkpoints it made from
// the left (folded to the ones that matter until it is opened); a run that starts from another's checkpoint hangs under
// that run's lane, and the lines between lanes say what came from what. Below it each run's trainer with its queue of
// steps, and what the runs' engines serve.

import { memo, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useLineage, useSystem } from "../api/queries";
import type { Lineage, LineageRun, LineageCheckpoint, Trainer, Worker } from "../api/types";
import { QueueChart, Sized } from "../components/charts";
import { Marks } from "../components/checkpoints";
import { Card, Empty, Head, Kpi, Kpis, Mark, SectionTitle, Spec, Specs, Table, Twist } from "../components/ui";
import { clock, span } from "../lib/format";
import { runPlace, checkpointPlace } from "../lib/places";
import { useFolds } from "../lib/stored";

const LANE = 78, COLUMN = 62, PAD = 34;
const OUTSIDE = "(outside a run)";
const short = (name: string | null | undefined) => (name ? String(name).split("/").at(-1) : "–");
const lifeKind = (state: string) => ({ serving: "good", resharding: "violet", resharded: "accent" } as Record<string, string>)[state] ?? "";
const ago = (lineage: Lineage, at: number | null | undefined) => (at ? span(Math.max(0, lineage.now - at)) : "–");

interface Index {
  checkpoints: Map<string, LineageCheckpoint>;
  runs: Map<string, LineageRun>;
  shortOf: (id: string | null | undefined) => string;
  runOf: (id: string | null | undefined) => string;
}

/** What each checkpoint is, wherever it is drawn: where it came from, the run that made it. */
function indexOf(lineage: Lineage): Index {
  const checkpoints = new Map(lineage.checkpoints.map(checkpoint => [checkpoint.id, checkpoint])), runs = new Map(lineage.runs.map(run => [run.run, run]));
  return {
    checkpoints, runs,
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
 * from it; under a run, each run that starts from one of its checkpoints (a fork). Checkpoints
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
    const from = first?.parents[0] ?? run?.from;
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
  | { kind: "gap"; count: number; names: string[] };

const itemId = (item: Item) => item.kind === "checkpoint" ? `v:${item.name}` : item.kind === "base" ? `b:${item.name}` : `g:${item.names[0]}`;
const edgeFrom = (edge: Lineage["edges"][number]) => (edge.from.startsWith("base:") ? `b:${edge.from.slice(5)}` : `v:${edge.from}`);
const edgeTo = (edge: Lineage["edges"][number]) => `v:${edge.to}`;

/** A lane's items, left to right: a base's root; or a run's checkpoints, and, while the lane is folded, a gap for each
 * stretch of checkpoints that nothing points at. */
function itemsOf(lane: Lane, open: boolean, anchors: Set<string>): Item[] {
  if (lane.base) return [{ kind: "base", name: lane.base }];
  if (lane.outside) return lane.outside.map(name => ({ kind: "checkpoint", name, outside: true }));
  const items: Item[] = [], checkpoints = lane.checkpoints;
  let hidden: string[] = [];
  const flush = () => { if (hidden.length) items.push({ kind: "gap", count: hidden.length, names: hidden }); hidden = []; };
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
    default: return `${index.shortOf(edge.to)} also learned from ${index.shortOf(edge.from)}`;
  }
}

const LineageGraph = memo(function LineageGraph({ lineage, index, lanes, room, inLedger }: { lineage: Lineage; index: Index; lanes: Lane[]; room: number; inLedger: Set<string> }) {
  const navigate = useNavigate();
  const [folds, fold] = useFolds();
  // (what an edge between lanes points at; an edge along a lane, from a checkpoint to the next its run made, pins nothing)
  const anchors = new Set<string>(lineage.edges.filter(edge => edge.kind !== "trained").flatMap(edge => [edge.from, edge.kind === "learned" || edge.kind === "base" ? edge.to : null]).filter((each): each is string => Boolean(each)));
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
  return (
    <div className="dag">
      <div className="dag-labels">
        {laid.map(lane => {
          const toggle = () => fold(`lane:${lane.key}`, !lane.open);
          if (lane.outside) {
            return <div key={lane.key} className="lane-label"><span /><b className="muted">Outside this ledger</b><small>{lane.outside.length} checkpoint{lane.outside.length === 1 ? "" : "s"}</small></div>;
          }
          if (lane.base) {
            return <div key={lane.key} className="lane-label base-label"><span /><b title={lane.base}>{short(lane.base)}</b><small>base model</small></div>;
          }
          const run = lane.run, first = lane.checkpoints[0], from = first?.parents[0];
          const says = from && index.checkpoints.get(from)?.by?.run !== lane.key
            ? `forked from ${index.shortOf(from)}${index.checkpoints.get(from)?.by ? ` (${index.checkpoints.get(from)!.by!.name})` : ""}` : "from the base model";
          const name = run?.name ?? (lane.key === OUTSIDE ? "made outside a run" : lane.key);
          return (
            <div key={lane.key} className="lane-label" style={{ paddingLeft: 4 + Math.min(lane.depth, 3) * 10 }} onClick={toggle} title={lane.open ? "fold the lane" : "show every checkpoint"}>
              <Twist open={lane.open} onToggle={toggle} />
              <b title={run ? `id: ${run.run}` : undefined}>{run && inLedger.has(run.run) ? <Link to={runPlace(run.run)} onClick={event => event.stopPropagation()}>{name}</Link> : name}</b>
              <small className="says">{lane.checkpoints.length} checkpoints · {says}</small>
            </div>
          );
        })}
      </div>
      <div className="dag-frame">
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label="checkpoints as a graph, from their base models" className="dag-drawing">
          {laid.map((lane, row) => <rect key={lane.key} x={0} y={row * LANE} width={width} height={LANE} className={`lane-band${row % 2 ? " odd" : ""}`} />)}
          {/* Along each lane: a line from item to item. */}
          {laid.map((lane, row) => lane.items.map((item, place) => {
            if (!place) return null;
            const from = lane.items[place - 1];
            const kind = item.kind === "gap" || from.kind === "gap" || lane.outside ? "quiet dash" : item.kind === "checkpoint" ? "accent" : "quiet";
            return <line key={`${lane.key}${itemId(item)}`} x1={x(itemId(from))} x2={x(itemId(item))} y1={y(row)} y2={y(row)} className={kind.split(" ").map(each => (each === "dash" ? "dash" : `s-${each}`)).join(" ")} />;
          }))}
          {/* Between lanes: from a base model to the first checkpoint of a line, a fork, a merge's other parents. */}
          {lineage.edges.map((edge, place) => {
            if (!crossing(edge)) return null;
            const from = where.get(edgeFrom(edge))!, to = where.get(edgeTo(edge))!;
            const x1 = x(edgeFrom(edge)), y1 = y(from.row), x2 = x(itemId(to.item)) - 7, y2 = y(to.row);
            const middle = x1 + Math.max(18, (x2 - x1) * 0.55);
            const kind = edge.kind === "learned" ? "s-quiet dash" : "s-quiet";
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
            const real = inLedger.has(checkpoint.id);
            const first = place === 0;
            return (
              <g key={id}>
                {first && lane.run ? <text x={cx - 6} y={cy - 26} className="stretch">{lane.run.name}</text> : null}
                <g className={`checkpoint${real ? " link" : ""}`} onClick={() => { if (real) navigate(checkpointPlace(checkpoint.id)); }}>
                  {["serving", "resharding"].includes(life.state) ? <circle cx={cx} cy={cy} r={10.5} className={`ring ring-${lifeKind(life.state)}`} /> : null}
                  <circle cx={cx} cy={cy} r={6} className={checkpoint.kept ? "dot-accent" : "dot-released"} />
                  <text x={cx} y={cy + 22} textAnchor="middle" className="v">{checkpoint.by?.step != null ? `S${checkpoint.by.step}` : checkpoint.short}</text>
                  {checkpoint.bookmarks.length ? <text x={cx} y={cy - 13} textAnchor="middle" className="bookmark">{checkpoint.bookmarks.join(", ")}</text> : null}
                  <title>{[
                    `${checkpoint.id} · depth ${checkpoint.depth}${checkpoint.kind === "full" ? " · full weights" : ""}`,
                    `made ${clock(checkpoint.made)}${checkpoint.by ? ` by ${checkpoint.by.name}${checkpoint.by.step != null ? ` at step ${checkpoint.by.step}` : ""}` : ""}, from ${checkpoint.parents.map(index.shortOf).join(" + ") || (checkpoint.base && index.checkpoints.has(checkpoint.base) ? index.shortOf(checkpoint.base) : `the base model ${checkpoint.base ?? ""}`)}`,
                    checkpoint.bookmarks.length ? `bookmarks: ${checkpoint.bookmarks.join(", ")}` : null,
                    `${life.state}${workers.length ? ` on ${workers.join(", ")}` : ""}${life.latest_of ? ` · ${index.runOf(life.latest_of)}'s latest` : ""}`,
                    checkpoint.metrics.kl_moved != null ? `moved ${checkpoint.metrics.kl_moved.toFixed(4)} from its parent` : null,
                    checkpoint.kept ? "its weights are kept" : "released: its weights are gone, its record stays",
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

function TrainerTile({ trainer, lineage, index }: { trainer: Trainer; lineage: Lineage; index: Index }) {
  const taking = trainer.queue.filter(entry => entry.state === "taking"), queued = trainer.queue.filter(entry => entry.state === "queued");
  const done = trainer.queue.filter(entry => entry.state === "made" || entry.state === "failed").slice(-3).reverse();
  return (
    <div className={`tile rail ${taking.length ? "violet" : queued.length ? "warm" : ""}`}>
      <header><b>{trainer.trainer}</b><span className="what">{trainer.weights === "full" ? "full weights" : trainer.weights === "lora" ? "LoRA" : "nothing made yet"}{trainer.base ? ` on ${index.checkpoints.has(trainer.base) ? index.shortOf(trainer.base) : short(trainer.base)}` : ""}</span></header>
      <div className="facts">
        <span>trains for <b>{trainer.runs.map(index.runOf).join(", ")}</b></span>
        {trainer.colocated ? <span>shares the engines' accelerator</span> : null}
      </div>
      <div className="cells three">
        <div className={`cell ${taking.length ? "violet" : ""}`}><span>taking</span><b>{taking.length}</b><small className="mono">{taking[0] ? index.shortOf(taking[0].makes) : "idle"}</small></div>
        <div className={`cell ${queued.length ? "warm" : "waiting"}`}><span>queued</span><b>{queued.length}</b><small>{queued.length ? `oldest ${ago(lineage, queued[0].queued)}` : ""}</small></div>
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

function WorkerTile({ worker, index }: { worker: Worker; index: Index }) {
  return (
    <div className={`tile rail ${worker.serving.length ? "good" : ""}`}>
      <header><b>{worker.worker}</b><span className="what">what its run published</span></header>
      <div className="chips">
        {worker.serving.length ? worker.serving.map(name => (
          <span key={name} className="chip mono" title={name}>{index.shortOf(name)}</span>
        )) : <span className="none">serves nothing</span>}
      </div>
    </div>
  );
}

/** A checkpoint's way to the engines, as stages: written, resharded (full weights only), serving. */
function Way({ checkpoint }: { checkpoint: LineageCheckpoint }) {
  const life = checkpoint.life, at = ({ written: 0, resharding: 1, resharded: 1, serving: 2, superseded: 3 } as Record<string, number>)[life.state] ?? 0;
  const stages = ["written", life.reshard ? "resharded" : "no reshard", "serving"];
  return (
    <div className="stages way">
      {stages.map((name, place) => (
        <div key={name} className={[place < at || (place === at && life.state === "resharded") ? "done" : place === at ? `now${life.state === "resharding" ? " active" : ""}` : "",
          place === 1 && !life.reshard ? "skipped" : ""].join(" ")}>
          <span>{place === at && life.state === "resharding" ? "resharding" : name}</span>
        </div>
      ))}
    </div>
  );
}

export function Checkpoints() {
  const { data: lineage } = useLineage();
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
  return (
    <>
      <Head title="Checkpoints">
        <Specs>
          <Spec label="base models">{lineage.bases.length}</Spec>
          <Spec label="checkpoints">{lineage.checkpoints.length}</Spec>
          <Spec label="runs">{lineage.runs.length}</Spec>
          <Spec label="bookmarks" kind="accent">{Object.keys(lineage.bookmarks).length}</Spec>
          <Spec label="trainers" kind="violet">{lineage.trainers.length}</Spec>
          <Spec label="engines" kind="accent">{lineage.workers.length}</Spec>
        </Specs>
      </Head>
      <section className="card">
        <header>
          <h2>Lineage</h2>
          <span>{opened ? `${opened} open` : "folded"} · <button type="button" className="linkish" onClick={() => all(true)}>open all</button> · <button type="button" className="linkish" onClick={() => all(false)}>fold all</button></span>
        </header>
        {lineage.checkpoints.length || lineage.bases.length ? <Sized fallback={1100}>{width => <LineageGraph lineage={lineage} index={index} lanes={lanes} room={width} inLedger={inLedger} />}</Sized> : <Empty>The ledger has no checkpoint yet.</Empty>}
        <div className="legend dag-legend">
          <span><i className="rule" style={{ background: "var(--quiet)" }} />trained from</span>
          <span><i style={{ background: "var(--accent)" }} />trained on its own groups</span>
          <span><b className="t-accent">name</b> a bookmark</span>
          <span><i className="hollow" />released</span><span><i className="ring-good" />serving</span>
          <span><i className="ring-violet" />resharding</span>
        </div>
      </section>
      <SectionTitle title="Trainers" />
      <div className="tiles wide-tiles">{lineage.trainers.map(trainer => <TrainerTile key={trainer.trainer} trainer={trainer} lineage={lineage} index={index} />)}</div>
      <SectionTitle title="Serving" />
      <Kpis>
        <Kpi label="Serving" value={String(moving.filter(checkpoint => checkpoint.life.state === "serving").length)} />
        <Kpi label="Resharding" value={String(moving.filter(checkpoint => ["resharding", "resharded"].includes(checkpoint.life.state)).length)} />
        <Kpi label="Engines" value={String(lineage.workers.length)} />
      </Kpis>
      <Card title="On their way to the engines, and served">
        <Table
          heads={[["checkpoint"], ["made by"], ["way"], ["engines"], ["latest of"], ["made", "n"]]}
          keys={moving.map(checkpoint => checkpoint.id)}
          rows={moving.map(checkpoint => {
            const workers = Object.entries(checkpoint.life.workers);
            return [
              <span><b className="mono" title={checkpoint.id}>{checkpoint.short}</b> <Marks names={checkpoint.bookmarks} /></span>,
              checkpoint.by ? `${checkpoint.by.name}${checkpoint.by.step != null ? ` S${checkpoint.by.step}` : ""}` : "–",
              <Way checkpoint={checkpoint} />,
              <div className="chips">
                {workers.length ? workers.map(([worker, served]) => (
                  <span key={worker} className={`chip${served.until == null ? " on" : " off"}`} title={served.until == null ? `serving since ${clock(served.since)}` : `served ${clock(served.since)} to ${clock(served.until)}`}>{worker}</span>
                )) : <span className="none">{checkpoint.life.state === "resharding" ? "files being rewritten" : "on no engine"}</span>}
              </div>,
              checkpoint.life.latest_of ? index.runOf(checkpoint.life.latest_of) : "–",
              ago(lineage, checkpoint.made),
            ];
          })}
          to={moving.map(checkpoint => (inLedger.has(checkpoint.id) ? checkpointPlace(checkpoint.id) : null))}
        />
      </Card>
      <div className="tiles">{lineage.workers.map(worker => <WorkerTile key={worker.worker} worker={worker} index={index} />)}</div>
      <Card title="Runs">
        <Table
          heads={[["run"], ["from"], ["checkpoints"], ["latest"]]}
          keys={lineage.runs.map(run => run.run)}
          rows={lineage.runs.map(run => [
            <b title={`id: ${run.run}`}>{run.name}</b>,
            index.shortOf(run.from),
            run.checkpoints.length ? `${index.shortOf(run.checkpoints[0])}–${index.shortOf(run.checkpoints.at(-1))}` : "–",
            run.latest ? index.shortOf(run.latest) : "–",
          ])}
          to={lineage.runs.map(run => (inLedger.has(run.run) ? runPlace(run.run) : null))}
        />
      </Card>
    </>
  );
}
