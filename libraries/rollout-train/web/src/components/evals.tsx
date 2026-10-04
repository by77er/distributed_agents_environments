// How a suite's subjects are said and drawn wherever they appear: who played (a checkpoint, or the base model), and
// how each did at each start of the suite.

import { Link } from "react-router-dom";
import { useSystem } from "../api/queries";
import type { Suite } from "../api/types";
import { figure, percent } from "../lib/format";
import type { Known } from "../lib/model";
import { runPlace } from "../lib/places";
import { CheckpointTag } from "./checkpoints";

export type Subject = Suite["subjects"][number];
export type SuiteStart = Suite["starts"][number];

/** The share of its episodes a subject solved (none: it played none yet). */
export const shareOf = (subject: Subject): number | null => (subject.played ? subject.solved / subject.played : null);

/** The share of a start's episodes a subject solved. */
export function startShare(subject: Subject, start: SuiteStart): number | null {
  const played = subject.results[start.start] ?? [];
  return played.length ? played.filter(each => each.solved).length / played.length : null;
}

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
        <i key={place} className={each.solved ? "solved" : "unsolved"} title={`${startName(start)}: reward ${figure(each.reward)}${each.solved ? ", solved" : ""}`} />
      ))}
    </>
  );
}

/** Every subject of a suite, start by start: what each solved, and in all. A subject opens the eval that played it. */
export function SuiteMatrix({ suite, subjects }: { suite: Suite; subjects: Subject[] }) {
  const { data: system } = useSystem();
  const runs = new Set((system?.runs ?? []).map(run => run.run));
  return (
    <>
      <div className="table">
        <table className="evals">
          <thead>
            <tr>
              <th>start</th>
              {subjects.map(subject => (
                <th key={subject.subject} className="subject">
                  <div><SubjectName subject={subject} /></div>
                  <small>{runs.has(subject.subject) ? <Link to={runPlace(subject.subject)} className="linkish">{subject.episodes ?? 1} a start</Link> : `${subject.episodes ?? 1} a start`}</small>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="total">
              <td>solved</td>
              {subjects.map(subject => {
                const expected = suite.starts.length * (subject.episodes ?? 1);
                return (
                  <td key={subject.subject} className="n" title={`mean reward ${figure(subject.reward)}`}>
                    <b>{subject.solved}/{subject.played}</b>{subject.played < expected ? <small className="faint"> of {expected}</small> : null}
                    <div className="track"><i style={{ width: `${((100 * subject.solved) / Math.max(1, expected)).toFixed(1)}%` }} /></div>
                  </td>
                );
              })}
            </tr>
            {suite.starts.map(start => (
              <tr key={start.start}>
                <td className="key" title={start.title ?? ""}>{startName(start)}</td>
                {subjects.map(subject => <td key={subject.subject} className="cell-result"><Played subject={subject} start={start} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend"><span><i style={{ background: "var(--good)" }} />solved</span><span><i style={{ background: "var(--line-strong)" }} />not solved</span><span><i className="hollow" />not played yet</span></div>
    </>
  );
}

/** A share, or a dash for none. */
export const shareText = (share: number | null): string => (share == null ? "–" : percent(share));
