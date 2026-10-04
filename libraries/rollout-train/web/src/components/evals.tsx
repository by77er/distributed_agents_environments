// How a suite's subjects are said and drawn wherever they appear: who played (a checkpoint, or the base model), the
// version of the suite it played, and how each did at each start of it.

import { Link } from "react-router-dom";
import { useKnown, useSystem } from "../api/queries";
import type { Suite } from "../api/types";
import { figure, percent } from "../lib/format";
import type { Known } from "../lib/model";
import { evalPlace, runPlace } from "../lib/places";
import { readable } from "../lib/environments";
import { ALL, type Block, blockScore, blocksOf, currentOf, gridRows, type Subject, versionGroups, versionTag } from "../lib/suites";
import { CheckpointTag } from "./checkpoints";
import { solvedClass } from "./ui";

export type { Subject };
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
function SubjectHead({ subject, runs, episodes, className = "" }: { subject: Subject; runs: Set<string>; episodes?: number; className?: string }) {
  const known = useKnown();
  const checkpoint = known.checkpoint(subject.checkpoint);
  const each = `${episodes ?? subject.episodes ?? 1} per start`;
  const scheduled = subject.asked_by && subject.asked_by !== "by hand" ? " · scheduled" : "";  // (asked for by a run)
  return (
    <th className={`subject${className}`}>
      <div>{subject.kind === "model" ? <span className="nowrap" title={subject.model ?? ""}>{subject.model?.split("/").at(-1) ?? "base model"}</span> : <CheckpointTag id={subject.checkpoint} bare />}</div>
      <small>
        {checkpoint?.step != null ? `S${checkpoint.step} · ` : ""}
        {runs.has(subject.subject) ? <Link to={evalPlace(subject.subject)} className="linkish">{each}</Link> : each}{scheduled}
      </small>
    </th>
  );
}

/** Every subject of a suite, start by start: what each solved, and in all. Columns stand by the version each subject
 * played (newest first, a version's columns marked off from the next), within a version of several environments by its
 * entry (a column group each, with each subject's total at that entry), then by run and depth; a row is a start, empty
 * under a group that does not have it. A column opens the eval that played it. `picked` is a version's id, or `ALL`. */
export function SuiteMatrix({ suite, subjects, picked = ALL }: { suite: Suite; subjects: Subject[]; picked?: string }) {
  const { data: system } = useSystem();
  const known = useKnown();
  const runs = new Set((system?.runs ?? []).map(run => run.run)), said = anySolved(subjects);
  const current = currentOf(suite).id;
  const groups = versionGroups(suite, subjects, picked).filter(group => group.subjects.length);
  const blocks = blocksOf(groups);
  const several = groups.length > 1, entries = blocks.some(block => block.entry != null);
  const rows = gridRows(blocks);
  const stood = blocks.map(block => ({ block, columns: columnsOf(block.subjects, known) }));
  const ordered = stood.flatMap(({ block, columns }, at) =>
    columns.flatMap(column => column.subjects).map((subject, place) => ({
      subject, block, edge: place > 0 ? "" : several && (at === 0 || stood[at - 1].block.version.id !== block.version.id) ? " version-start" : entries && at > 0 ? " entry-start" : "",
    })));
  const edgeOf = (block: Block, place: number) => ordered.find(each => each.block.key === block.key && place === 0)?.edge ?? "";
  const heads = 2 + (several ? 1 : 0) + (entries ? 1 : 0);
  return (
    <>
      <div className="table">
        <table className="evals">
          <thead>
            {several ? (
              <tr className="groups versions">
                <th rowSpan={heads}>start</th>
                {groups.map(group => (
                  <th key={group.version.id} colSpan={group.subjects.length * Math.max(1, group.version.entries.length > 1 ? group.version.entries.length : 1)} className="group version-start">
                    {versionTag(group.version.id)}{group.version.id === current ? " · newest" : ""}
                  </th>
                ))}
              </tr>
            ) : null}
            {entries ? (
              <tr className="groups entries">
                {several ? null : <th rowSpan={heads}>start</th>}
                {blocks.map(block => (
                  <th key={block.key} colSpan={block.subjects.length} className={`group${edgeOf(block, 0)}`} title={block.entry?.environment ?? ""}>
                    {block.entry ? readable(block.entry.environment) : ""}
                  </th>
                ))}
              </tr>
            ) : null}
            <tr className="groups">
              {several || entries ? null : <th rowSpan={2}>start</th>}
              {stood.flatMap(({ block, columns }) => columns.map((column, place) => (
                <th key={`${block.key}:${column.key}`} colSpan={column.subjects.length} className={`group${place === 0 ? edgeOf(block, 0) : ""}`}>
                  {column.run && runs.has(column.run) ? <Link to={runPlace(column.run)} className="linkish">{column.label}</Link> : column.label}
                </th>
              )))}
            </tr>
            <tr>{ordered.map(({ subject, block, edge }) => <SubjectHead key={`${block.key}:${subject.subject}`} subject={subject} runs={runs} episodes={blockScore(subject, block).episodes} className={edge} />)}</tr>
          </thead>
          <tbody>
            <tr className="total">
              <td>{said ? "solved" : "mean reward"}</td>
              {ordered.map(({ subject, block, edge }) => {
                const score = blockScore(subject, block);
                const starts = block.entry ? block.entry.starts : subject.starts || block.version.starts.length;
                const each = block.entry ? (score.episodes ?? subject.episodes ?? block.entry.episodes) : (subject.episodes ?? 1);
                const expected = starts * each;
                return (
                  <td key={`${block.key}:${subject.subject}`} className={`n${edge}`} title={`mean reward ${figure(score.reward)}`}>
                    {score.solved == null ? <b>{figure(score.reward)}</b> : <b>{score.solved}/{score.played}</b>}
                    {score.played < expected ? <small className="faint"> {score.solved == null ? `${score.played} of ${expected}` : `of ${expected}`}</small> : null}
                    {score.solved == null ? null : <div className="track"><i style={{ width: `${((100 * score.solved) / Math.max(1, expected)).toFixed(1)}%` }} /></div>}
                  </td>
                );
              })}
            </tr>
            {rows.map(row => (
              <tr key={row.key}>
                <td className="key" title={[row.start.title, row.start.environment].filter(Boolean).join(" · ")}>{startName(row.start)}</td>
                {ordered.map(({ subject, block, edge }) => {
                  const number = row.at[block.key];
                  return number == null
                    ? <td key={`${block.key}:${subject.subject}`} className={`cell-result absent${edge}`} title={`not in ${versionTag(block.version.id)}${block.entry ? ` ${readable(block.entry.environment)}` : ""}`} />
                    : <td key={`${block.key}:${subject.subject}`} className={`cell-result${edge}`}><Played subject={subject} start={{ ...row.start, start: number }} /></td>;
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
    </>
  );
}

/** Each environment's share solved (else its mean reward) of an eval of several, in a few words; none for an eval of
 * one. */
export const entriesText = (entries: { share: number | null; reward: number | null }[] | undefined): string | null =>
  (entries?.length ?? 0) > 1 ? entries!.map(each => (each.share != null ? percent(each.share) : figure(each.reward))).join(" · ") : null;

/** A share, or a dash for none. */
export const shareText = (share: number | null): string => (share == null ? "–" : percent(share));
