// The pieces every view is made of: a datasheet's facts under a title, figures, cards, tables, meters, a group's
// stages, its episodes as cells, and an agent's badge.

import { memo, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { DoneLine } from "../api/types";
import { figure, slotHue } from "../lib/format";
import { type Asked, episodeClass, isWaiting, stateKind } from "../lib/model";

/** A fact about the thing shown, as a datasheet gives it: a label and a value, set side by side under the title. */
export const Spec = ({ label, kind = "", children }: { label: string; kind?: string; children: ReactNode }) => (
  <span className={`spec ${kind}`}>
    <span className="k">{label}</span>
    <span className="v">{children}</span>
  </span>
);
export const Specs = ({ children }: { children: ReactNode }) => <div className="specs">{children}</div>;

/** A state: a square in its color, then its name. */
export const Mark = ({ state, children }: { state: string | null | undefined; children?: ReactNode }) => (
  <span className={`mark ${stateKind(state)}`}>
    <i />
    {children ?? state}
  </span>
);

export const Kpi = ({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) => (
  <div className="kpi">
    <span>{label}</span>
    <b title={typeof value === "string" ? value : undefined}>{value}</b>
    {note ? <small>{note}</small> : null}
  </div>
);
export const Kpis = ({ children, style }: { children: ReactNode; style?: React.CSSProperties }) => (
  <div className="kpis" style={style}>{children}</div>
);

/** A card: its title, with a note beside it, over its body; with neither, the body alone (where a section's title above
 * it says what it is). */
export const Card = ({ title, note, children, className = "" }: { title?: ReactNode; note?: ReactNode; children?: ReactNode; className?: string }) => (
  <section className={`card ${className}`}>
    {title || note ? (
      <header>
        <h2>{title}</h2>
        {note ? <span>{note}</span> : null}
      </header>
    ) : null}
    <div className="body">{children}</div>
  </section>
);

export const SectionTitle = ({ title, note, id }: { title: string; note?: ReactNode; id?: string }) => (
  <div className="section-title" id={id}>
    <h2>{title}</h2>
    {note ? <span>{note}</span> : null}
  </div>
);

export const Empty = ({ children }: { children: ReactNode }) => <div className="empty">{children}</div>;

export const Head = ({ title, sub, children }: { title: ReactNode; sub?: ReactNode; children?: ReactNode }) => (
  <div className="head">
    <h1>{title}</h1>
    {sub ? <div className="sub">{sub}</div> : null}
    {children}
  </div>
);

type Cell = ReactNode | { text: ReactNode; kind?: string };
const isKinded = (cell: Cell): cell is { text: ReactNode; kind?: string } =>
  typeof cell === "object" && cell !== null && !Array.isArray(cell) && "text" in cell;

/** A table: headings (a name, and `n` for a column of numbers), rows of cells, and where a row leads, if anywhere. */
export function Table({ heads, rows, to, keys }: { heads: [string, string?][]; rows: Cell[][]; to?: (string | null)[]; keys?: (string | number)[] }) {
  const navigate = useNavigate();
  return (
    <div className="table">
      <table>
        <thead>
          <tr>{heads.map(([name, kind], index) => <th key={index} className={kind}>{name}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const place = to?.[index];
            return (
              <tr key={keys?.[index] ?? index} className={place ? "link" : undefined} onClick={place ? () => navigate(place) : undefined}>
                {row.map((cell, column) => {
                  const kind = [heads[column]?.[1], isKinded(cell) ? cell.kind : undefined].filter(Boolean).join(" ") || undefined;
                  return <td key={column} className={kind}>{isKinded(cell) ? cell.text : cell ?? ""}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** How much of something is used, as a bar: one nearly full is warm, then bad, unless being full is what it is for
 * (`warns` false: a runner's places, a pool's sandboxes). */
export function Meter({ name, used, total, says, warns = true }: { name: string; used: number; total: number; says: string; warns?: boolean }) {
  const share = total ? Math.min(used / total, 1) : 0;
  return (
    <div className="meter">
      <div><span>{name}</span><b>{says}</b></div>
      <div className={`track${!warns ? "" : share > 0.93 ? " bad" : share > 0.85 ? " warm" : ""}`}><i style={{ width: `${(100 * share).toFixed(1)}%` }} /></div>
    </div>
  );
}

export const Pairs = ({ entries }: { entries: [string, ReactNode][] }) => (
  <dl className="pairs">
    {entries.map(([key, value]) => [<dt key={`k${key}`}>{key}</dt>, <dd key={`v${key}`}>{value}</dd>])}
  </dl>
);

/** A share as a bar and a percentage. */
export const Share = ({ value, kind = "" }: { value: number | null; kind?: string }) =>
  value == null ? <span className="faint">–</span> : (
    <span className="share">
      <span className={`track ${kind}`}><i style={{ width: `${(100 * value).toFixed(1)}%` }} /></span>
      <b>{Math.round(100 * value)}%</b>
    </span>
  );

export const Legend = ({ items }: { items: { name: string; color?: string; className?: string }[] }) => (
  <div className="legend">
    {items.map(item => <span key={item.name}><i className={item.className} style={item.color ? { background: item.color } : undefined} />{item.name}</span>)}
  </div>
);

export const Twist = ({ open, has = true, onToggle }: { open: boolean; has?: boolean; onToggle: () => void }) => (
  <button
    type="button"
    className={`twist${open ? " open" : ""}${has ? "" : " none"}`}
    aria-label={open ? "collapse" : "expand"}
    aria-expanded={open}
    onClick={event => { event.preventDefault(); event.stopPropagation(); onToggle(); }}
  >
    <svg width={10} height={10} viewBox="0 0 10 10"><path d="M3 1.5 L7 5 L3 8.5" fill="none" stroke="currentColor" strokeWidth={1.6} /></svg>
  </button>
);

/** An agent's badge: its slot's number (`agent-2` → 2), or, for a slot with no number, its initial, in a color its whole
 * name gives it. */
export const Avatar = memo(function Avatar({ name }: { name: string }) {
  const number = name.match(/(\d+)$/)?.[1];
  return <span className="avatar" title={name} style={{ background: `hsl(${slotHue(name)} 55% 46%)` }}>{number ?? name.slice(0, 1)}</span>;
});

/** An episode's square: solved, not solved, or ended where its task does not say. */
export const solvedClass = (solved: boolean | null | undefined): string => (solved == null ? "played" : solved ? "solved" : "unsolved");

export const Dots = memo(function Dots({ line }: { line: DoneLine }) {
  return (
    <span className="dots">
      {line.rewards.map((_, index) => <i key={`r${index}`} className={solvedClass(line.solved[index])} />)}
      {Array.from({ length: line.failed }, (_, index) => <i key={`f${index}`} className="failed" />)}
    </span>
  );
});

export const EpisodeDots = ({ episodes }: { episodes: Asked[] }) => (
  <span className="dots">{episodes.map((each, index) => <i key={isWaiting(each) ? `w${each.episode}` : each.run_id ?? index} className={episodeClass(each)} />)}</span>
);

/** A group's stages, up to its result; what is trained on it is its step's (a step is shown as a thing of its own). */
const STAGES = ["asked", "claimed", "played", "recorded"];
const AT: Record<string, [number, string]> = { waiting: [1, "to claim"], playing: [2, ""], ended: [3, "recording"], done: [4, ""] };
export function Stages({ stage, ended, count }: { stage: string; ended: number; count: number | null }) {
  const [at, waits] = AT[stage] ?? [0, ""];
  const says = stage === "playing" ? `playing ${ended}/${count ?? "?"}` : waits;
  return (
    <div className="stages">
      {STAGES.map((name, index) => (
        <div key={name} className={index < at ? "done" : index === at ? `now${stage === "playing" ? " active" : ""}` : ""}>
          <span>{index === at ? says : name}</span>
        </div>
      ))}
    </div>
  );
}

/** A group's episodes as a strip of cells: each with its state as a bar along its top, and its reward (or, while it
 * plays, how many samples its agents have taken). */
export const Cells = ({ episodes }: { episodes: Asked[] }) => (
  <div className="cells">
    {episodes.map(episode => isWaiting(episode) ? (
      <div key={`w${episode.episode}`} className="cell waiting"><span>E{episode.episode}</span><b className="faint">–</b><small>not started</small></div>
    ) : (
      <div key={episode.run_id} className={`cell ${episode.interrupted ? "" : stateKind(episode.state)}`}>
        <span>E{episode.episode ?? "?"}</span>
        {episode.outcome ? <b>{figure(episode.reward)}</b> : <b className="faint">{episode.samples ?? 0}</b>}
        <small>{episode.interrupted ? "interrupted" : episode.outcome ? (episode.solved ? "solved" : episode.outcome) : "samples"}</small>
      </div>
    ))}
  </div>
);

/** A link styled as the thing it opens (a tile, a member row). */
export const Tile = ({ to, className = "", children }: { to?: string; className?: string; children: ReactNode }) =>
  to ? <Link to={to} className={`tile ${className}`}>{children}</Link> : <div className={`tile ${className}`}>{children}</div>;
