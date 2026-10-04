// How a suite's subjects are said and drawn wherever they appear: who played (a checkpoint, or the base model), and
// how each did at each start of the suite.

import { Link } from "react-router-dom";
import { useKnown, useSystem } from "../api/queries";
import type { Suite } from "../api/types";
import { figure, percent } from "../lib/format";
import type { Known } from "../lib/model";
import { evalPlace, runPlace } from "../lib/places";
import { CheckpointTag } from "./checkpoints";
import { solvedClass } from "./ui";

export type Subject = Suite["subjects"][number];
export type SuiteStart = Suite["starts"][number];

/** The share of its episodes a subject solved (none: it played none yet, or its task does not say). */
export const shareOf = (subject: Subject): number | null =>
  subject.played && subject.solved != null ? subject.solved / subject.played : null;

/** The share of a start's episodes a subject solved (none where none of them says). */
export function startShare(subject: Subject, start: SuiteStart): number | null {
  const played = subject.results[start.start] ?? [];
  return played.some(each => each.solved != null) ? played.filter(each => each.solved).length / played.length : null;
}

/** Whether any subject's episodes say if they solved their starts. */
export const anySolved = (subjects: Subject[]): boolean => subjects.some(subject => subject.solved != null);

/** Who played: the checkpoint (opening its page), or the base model by its name. */
export const SubjectName = ({ subject }: { subject: Subject }) =>
  subject.kind === "model" ? <CheckpointTag id={null} base={subject.model} /> : <CheckpointTag id={subject.checkpoint} bare />;

/** Who played, in a few words: the checkpoint's shortest id, or the base model's name. */
export const subjectText = (subject: Subject, known: Known): string =>
  subject.kind === "model" ? `base ${subject.model?.split("/").at(-1) ?? "model"}` : known.short(subject.checkpoint);

/** A start, as its row and seed. */
export const startName = (start: SuiteStart) => `${start.task} · ${start.seed}`;

/** A subject's episodes at a start, each solved or not; a hollow square for a start not played yet. */
export function Played({ subject, start }: { subject: Subject; start: SuiteStart }) {
  const played = subject.results[start.start] ?? [];
  if (!played.length) return <i className="unplayed" title="not played yet" />;
  return (
    <>
      {played.map((each, place) => (
        <i key={place} className={solvedClass(each.solved)} title={`${startName(start)}: reward ${figure(each.reward)}${each.solved ? ", solved" : ""}`} />
      ))}
    </>
  );
}

/** A suite's subjects in columns, grouped: the base model first, then each run's checkpoints (and those made outside a
 * run), runs in the order their checkpoints grew, and within a run by depth. */
export interface Column {
  key: string;
  /** The run, or `base` for the base model, or `outside` for checkpoints made outside a run. */
  label: string;
  run: string | null;
  subjects: Subject[];
}

export function columnsOf(subjects: Subject[], known: Known): Column[] {
  const groups = new Map<string, Column>();
  const at = (subject: Subject) => known.checkpoint(subject.checkpoint);
  for (const subject of subjects) {
    const checkpoint = at(subject);
    const key = subject.kind === "model" ? "base" : checkpoint?.run ? `run:${checkpoint.run}` : "outside";
    const label = key === "base" ? "base model" : checkpoint?.run ? known.run(checkpoint.run) : "outside a run";
    const group = groups.get(key) ?? { key, label, run: checkpoint?.run ?? null, subjects: [] };
    group.subjects.push(subject);
    groups.set(key, group);
  }
  const depth = (subject: Subject) => at(subject)?.depth ?? 0, made = (subject: Subject) => at(subject)?.made ?? 0;
  const order = (a: Subject, b: Subject) => depth(a) - depth(b) || made(a) - made(b) || a.subject.localeCompare(b.subject);
  for (const group of groups.values()) group.subjects.sort(order);
  const first = (group: Column) => group.subjects[0];
  return [...groups.values()].sort((a, b) => (a.key === "base" ? -1 : b.key === "base" ? 1 : order(first(a), first(b))));
}

/** A column's heading: the checkpoint (its short id, what its weights are, its bookmarks) and its step, or the base
 * model's name; under it, how many episodes of each start (opening the eval). */
function SubjectHead({ subject, runs }: { subject: Subject; runs: Set<string> }) {
  const known = useKnown();
  const checkpoint = known.checkpoint(subject.checkpoint);
  const each = `${subject.episodes ?? 1} per start`;
  const scheduled = subject.asked_by && subject.asked_by !== "by hand" ? " · scheduled" : "";  // (asked for by a run)
  return (
    <th className="subject">
      <div>{subject.kind === "model" ? <span className="nowrap" title={subject.model ?? ""}>{subject.model?.split("/").at(-1) ?? "base model"}</span> : <CheckpointTag id={subject.checkpoint} bare />}</div>
      <small>
        {checkpoint?.step != null ? `S${checkpoint.step} · ` : ""}
        {runs.has(subject.subject) ? <Link to={evalPlace(subject.subject)} className="linkish">{each}</Link> : each}{scheduled}
      </small>
    </th>
  );
}

/** Every subject of a suite, start by start: what each solved, and in all. Columns stand by run, then depth; a column
 * opens the eval that played it. */
export function SuiteMatrix({ suite, subjects }: { suite: Suite; subjects: Subject[] }) {
  const { data: system } = useSystem();
  const known = useKnown();
  const runs = new Set((system?.runs ?? []).map(run => run.run)), said = anySolved(subjects);
  const columns = columnsOf(subjects, known), ordered = columns.flatMap(column => column.subjects);
  return (
    <>
      <div className="table">
        <table className="evals">
          <thead>
            <tr className="groups">
              <th rowSpan={2}>start</th>
              {columns.map(column => (
                <th key={column.key} colSpan={column.subjects.length} className="group">
                  {column.run && runs.has(column.run) ? <Link to={runPlace(column.run)} className="linkish">{column.label}</Link> : column.label}
                </th>
              ))}
            </tr>
            <tr>{ordered.map(subject => <SubjectHead key={subject.subject} subject={subject} runs={runs} />)}</tr>
          </thead>
          <tbody>
            <tr className="total">
              <td>{said ? "solved" : "mean reward"}</td>
              {ordered.map(subject => {
                const expected = suite.starts.length * (subject.episodes ?? 1);
                return (
                  <td key={subject.subject} className="n" title={`mean reward ${figure(subject.reward)}`}>
                    {subject.solved == null ? <b>{figure(subject.reward)}</b> : <b>{subject.solved}/{subject.played}</b>}
                    {subject.played < expected ? <small className="faint"> {subject.solved == null ? `${subject.played} of ${expected}` : `of ${expected}`}</small> : null}
                    {subject.solved == null ? null : <div className="track"><i style={{ width: `${((100 * subject.solved) / Math.max(1, expected)).toFixed(1)}%` }} /></div>}
                  </td>
                );
              })}
            </tr>
            {suite.starts.map(start => (
              <tr key={start.start}>
                <td className="key" title={start.title ?? ""}>{startName(start)}</td>
                {ordered.map(subject => <td key={subject.subject} className="cell-result"><Played subject={subject} start={start} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend">
        {said ? <><span><i style={{ background: "var(--good)" }} />solved</span><span><i style={{ background: "var(--line-strong)" }} />not solved</span></> : <span><i style={{ background: "var(--quiet)" }} />played</span>}
        <span><i className="hollow" />not played yet</span>
      </div>
    </>
  );
}

/** A share, or a dash for none. */
export const shareText = (share: number | null): string => (share == null ? "–" : percent(share));
