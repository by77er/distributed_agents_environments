// The hierarchy, on the left, for the page shown: the runs (each run, its steps, their groups, their episodes and
// the episodes' rollouts), the checkpoints, the suites and the checkpoints evaluated (each with its evals), the
// environments with their runs and suites, the statistics' sections and the training runs drawn, or the machines and
// the roles on each. What is folded is remembered in this browser.

import { memo, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useEnvironments, useEpisode, useEvalSubjects, useEvals, useFeeds, useKnown, useSystem } from "../api/queries";
import type { EvalRun, GroupEpisode, Run, System } from "../api/types";
import { Avatar, Dots, EpisodeDots, Twist } from "../components/ui";
import { entriesText, shareText } from "../components/evals";
import { byNumber, figure, mean } from "../lib/format";
import { asked, episodeClass, groupsOf, madeBy, nameOf, range, reported } from "../lib/model";
import { environmentPlace, episodePlace, evalPlace, groupPlace, type Place, runPlace, statisticsPlace, stepPlace, checkpointPlace, subjectPlace, suitePlace } from "../lib/places";
import { type Folds, useFolds, useStored } from "../lib/stored";
import { versionTag } from "../lib/suites";
import { RunDot, running, useRunColor } from "./runs";
import { MachinesTree } from "./MachinesTree";

/** A row of the tree: it opens its place. */
export function Node({ to, current, className = "", children }: { to: string; current?: boolean; className?: string; children: ReactNode }) {
  const navigate = useNavigate();
  return (
    <div
      className={`node ${className}${current ? " current" : ""}`}
      role="link"
      tabIndex={0}
      onClick={() => navigate(to)}
      onKeyDown={event => { if (event.key === "Enter") navigate(to); }}
    >
      {children}
    </div>
  );
}

export function Tree({ place }: { place: Place }) {
  const { data: system } = useSystem();
  if (!system) return null;
  return (
    <nav className="tree" aria-label="hierarchy">
      {place.page === "checkpoints" ? <CheckpointsTree place={place} system={system} /> : place.page === "statistics" ? <StatisticsTree place={place} system={system} />
        : place.page === "evals" ? <EvalsTree place={place} /> : place.page === "machines" ? <MachinesTree place={place} />
          : place.page === "environments" ? <EnvironmentsTree place={place} system={system} /> : <RunsTree place={place} system={system} />}
    </nav>
  );
}

function RunsTree({ place, system }: { place: Place; system: System }) {
  const [folds, fold] = useFolds();
  const { data: feeds } = useFeeds();
  const episodeId = place.kind === "episode" ? place.id : "";
  const { data: episode } = useEpisodeLabels(episodeId);
  const showing = place.kind === "episode" ? episode : undefined;
  const others = (feeds ?? []).filter(run => !run.labels.run).length;
  const runs = system.runs.filter(run => run.kind !== "eval");  // (an eval's run is under the evals)
  return (
    <>
      {runs.map(run => (
        <RunBranch key={run.run} run={run} only={runs.length === 1} place={place} folds={folds} fold={fold} showing={showing} host={system.host} />
      ))}
      <Node to="/episodes" current={place.kind === "outside"}>
        <span className="name">Episodes outside a run</span>
        <span className="tag">{others}</span>
      </Node>
    </>
  );
}

/** The labels of the episode shown, if one is (to open the run and group it is of). */
function useEpisodeLabels(id: string): { data: Record<string, string> | undefined } {
  const { data } = useEpisode(id, Boolean(id));
  return { data: id ? data?.labels : undefined };
}

interface BranchProps {
  run: Run;
  only: boolean;
  place: Place;
  folds: Folds;
  fold: (key: string, open: boolean) => void;
  showing: Record<string, string> | undefined;
  host: string;
}

const RunBranch = memo(function RunBranch({ run, only, place, folds, fold, showing, host }: BranchProps) {
  const known = useKnown();
  const runKey = `run:${run.run}`;
  const here = "run" in place ? place.run : undefined;
  const mine = here === run.run || showing?.run === run.run;
  const runOpen = folds[runKey] ?? (mine || only);
  const groups = groupsOf(run);
  const inGroup = (number: number) =>
    (place.kind === "group" && place.run === run.run && place.number === number) || (showing?.run === run.run && Number(showing?.group) === number);
  const groupRows = (number: number, skipped: boolean) => {
    const group = groups.get(number);
    if (!group) return null;
    const key = `group:${run.run}:${number}`, open = folds[key] ?? inGroup(number);
    const line = group.line;
    const tag = group.open ? group.open.stage : skipped ? "skipped" : !line?.rewards.length ? "failed"
      : reported(line.solved) ? `${line.solved.filter(Boolean).length}/${line.rewards.length}` : figure(mean(line.rewards));
    const sorted = [...group.episodes].sort((a, b) => byNumber(String(a.episode), String(b.episode)));
    return (
      <div key={`g${number}`}>
        <Node to={groupPlace(run.run, number)} current={place.kind === "group" && place.run === run.run && place.number === number} className="group">
          <Twist open={open} has={group.episodes.length > 0} onToggle={() => fold(key, !open)} />
          <span className="num">#{number}</span>
          <span className="name">{group.task}</span>
          {group.line && !group.episodes.length ? <Dots line={group.line} /> : <EpisodeDots episodes={group.open ? asked(group.open) : group.episodes} />}
          <span className="tag">{tag}</span>
        </Node>
        {open && sorted.length ? (
          <div className="children">{sorted.map(each => <EpisodeRow key={each.run_id} each={each} place={place} folds={folds} fold={fold} />)}</div>
        ) : null}
      </div>
    );
  };
  const children: ReactNode[] = [];
  if (runOpen) {
    if (run.next.length) {
      const key = `next:${run.run}`, open = folds[key] ?? true;
      children.push(
        <Node key="next" to={runPlace(run.run)} className="step">
          <Twist open={open} onToggle={() => fold(key, !open)} />
          <span className="num">next</span>
          <span className="name">toward a step</span>
          <span className="tag">{run.next.length} group{run.next.length === 1 ? "" : "s"}</span>
        </Node>,
      );
      if (open) children.push(<div key="next-children" className="children">{[...run.next].reverse().map(number => groupRows(number, false))}</div>);
    }
    const newest = run.steps.at(-1)?.step;
    for (const step of [...run.steps].reverse()) {
      const key = `step:${run.run}:${step.step}`, members = [...step.groups, ...step.skipped];
      const current = place.kind === "step" && place.run === run.run && place.number === step.step;
      const open = folds[key] ?? (step.step === newest || members.some(inGroup) || current);
      children.push(
        <Node key={`s${step.step}`} to={stepPlace(run.run, step.step)} current={current} className="step">
          <Twist open={open} has={members.length > 0} onToggle={() => fold(key, !open)} />
          <span className="num">S{step.step}</span>
          <span className="name mono">{step.state === "committed" ? known.short(step.makes) : `${known.short(step.makes)} ${step.state}`}</span>
          <span className="tag">{range(step.groups)}</span>
        </Node>,
      );
      if (open) {
        children.push(
          <div key={`s${step.step}-children`} className="children">
            {[...step.groups].reverse().map(number => groupRows(number, false))}
            {[...step.skipped].reverse().map(number => groupRows(number, true))}
          </div>,
        );
      }
    }
  }
  return (
    <>
      <Node to={runPlace(run.run)} current={place.kind === "run" && place.run === run.run}>
        <Twist open={runOpen} onToggle={() => fold(runKey, !runOpen)} />
        <RunDot run={run} host={host} />
        <span className="name" title={`id: ${run.run}`}>{nameOf(run)}</span>
        <span className="tag">{running(run, host)}</span>
      </Node>
      {runOpen ? <div className="children">{children}</div> : null}
    </>
  );
});

function EpisodeRow({ each, place, folds, fold }: { each: GroupEpisode; place: Place; folds: Folds; fold: (key: string, open: boolean) => void }) {
  const key = `episode:${each.run_id}`, slots = [...(each.slots ?? [])].sort(byNumber);
  const isHere = place.kind === "episode" && place.id === each.run_id;
  const open = folds[key] ?? (isHere && Boolean(place.kind === "episode" && place.slot));
  return (
    <>
      <Node to={episodePlace(each.run_id)} current={isHere && place.kind === "episode" && !place.slot} className="episode">
        <Twist open={open} has={slots.length > 0} onToggle={() => fold(key, !open)} />
        <span className="num">E{each.episode ?? "?"}</span>
        <span className="dots"><i className={episodeClass(each)} /></span>
        <span className="name">{each.interrupted ? "interrupted" : each.outcome ?? (each.state === "running" ? `${each.samples ?? 0} samples` : each.state)}</span>
        <span className="tag">{each.outcome || each.state === "completed" ? figure(each.reward) : ""}</span>
      </Node>
      {open && slots.length ? (
        <div className="children">
          {slots.map(slot => (
            <Node key={slot} to={episodePlace(each.run_id, slot)} current={isHere && place.kind === "episode" && place.slot === slot} className="rollout">
              <Avatar name={slot} />
              <span className="name">rollout {slot}</span>
            </Node>
          ))}
        </div>
      ) : null}
    </>
  );
}

/** The checkpoints: those bookmarks name, and each run's newest. */
function CheckpointsTree({ place, system }: { place: Place; system: System }) {
  const known = useKnown();
  const heads = system.runs.map(run => madeBy(system.checkpoints, run.run).at(-1)).filter(each => each !== undefined);
  const bookmarked = system.checkpoints.filter(checkpoint => checkpoint.bookmarks.length);
  const row = (id: string, tag: string, key: string) => (
    <Node key={key} to={checkpointPlace(id)} current={place.kind === "checkpoint" && place.id === id}>
      <span className="name mono" title={known.title(id)}>{known.short(id)}</span>
      <span className="tag">{tag}</span>
    </Node>
  );
  return (
    <>
      {bookmarked.length ? <div className="label">Bookmarks</div> : null}
      {bookmarked.map(checkpoint => row(checkpoint.id, checkpoint.bookmarks.join(", "), `b${checkpoint.id}`))}
      {heads.length ? <div className="label">Each run's newest</div> : null}
      {heads.map(checkpoint => row(checkpoint.id, known.origin(checkpoint.id), `h${checkpoint.id}`))}
      {system.checkpoints.length ? null : <div className="empty">No checkpoint yet.</div>}
    </>
  );
}

/** How an eval did, in a few words: each environment's share solved (else its mean reward), or how far it has got. */
const evalTag = (each: EvalRun): string =>
  each.done ? entriesText(each.entries) ?? shareText(each.played && each.solved != null ? each.solved / each.played : null) : `${each.played}/${each.expected}`;

/** The suites, each opening to its evals, newest first: who played it and how it went; then the checkpoints and base
 * models evaluated, each opening its history and opening to its evals, newest first. */
function EvalsTree({ place }: { place: Place }) {
  const { data: evals } = useEvals();
  const { data: subjects } = useEvalSubjects();
  const [folds, fold] = useFolds();
  const known = useKnown();
  if (!evals) return null;
  if (!evals.suites.length) return <div className="empty">No suite yet.</div>;
  const shown = place.kind === "eval" ? evals.evals.find(each => each.run === place.run)?.suite : place.kind === "suite" ? place.suite : null;
  const byRun = new Map(evals.evals.map(each => [each.run, each]));
  const evaluated = subjects?.subjects ?? [];
  return (
    <>
      <div className="label">Suites</div>
      {evals.suites.map(suite => {
        const key = `suite:${suite.suite}`, open = folds[key] ?? suite.suite === shown;
        const played = evals.evals.filter(each => each.suite === suite.suite).sort((a, b) => (b.started ?? 0) - (a.started ?? 0));
        const differ = new Set(played.map(each => each.version ?? `${suite.suite}@1`)).size > 1;  // (each says its version then)
        return (
          <div key={suite.suite}>
            <Node to={suitePlace(suite.suite)} current={place.kind === "suite" && place.suite === suite.suite}>
              <Twist open={open} has={played.length > 0} onToggle={() => fold(key, !open)} />
              <span className="name">{suite.suite}</span>
              <span className="tag">{played.length}</span>
            </Node>
            {open && played.length ? (
              <div className="children">
                {played.map(each => (
                  <Node key={each.run} to={evalPlace(each.run)} current={place.kind === "eval" && place.run === each.run}>
                    <span className={`dot${each.done ? "" : " alive"}`} />
                    <span className="name" title={each.name}>{each.checkpoint ? known.short(each.checkpoint) : "base model"}</span>
                    {differ ? <span className="version">{versionTag(each.version)}</span> : null}
                    <span className="tag">{evalTag(each)}</span>
                  </Node>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
      {evaluated.length ? <div className="label">Checkpoints</div> : null}
      {evaluated.map(subject => {
        const key = `subject:${subject.kind}:${subject.id}`;
        const here = place.kind === "subject" && place.subject === subject.kind && (place.id === subject.id || (subject.kind === "checkpoint" && subject.id.startsWith(place.id)));
        const open = folds[key] ?? here;
        const played = subject.evals.map(run => byRun.get(run)).filter(each => each !== undefined);
        return (
          <div key={key}>
            <Node to={subjectPlace(subject.kind, subject.id)} current={here}>
              <Twist open={open} has={played.length > 0} onToggle={() => fold(key, !open)} />
              {subject.playing ? <span className="dot alive" /> : null}
              <span className={`name${subject.kind === "checkpoint" ? " mono" : ""}`} title={subject.kind === "checkpoint" ? known.title(subject.id) : subject.id}>
                {subject.kind === "checkpoint" ? known.short(subject.id) : `base ${subject.id.split("/").at(-1)}`}
              </span>
              <span className="tag">{subject.kind === "checkpoint" ? subject.bookmarks.join(", ") || known.origin(subject.id) : ""}</span>
            </Node>
            {open && played.length ? (
              <div className="children">
                {played.map(each => (
                  <Node key={each.run} to={evalPlace(each.run)} current={place.kind === "eval" && place.run === each.run}>
                    <span className={`dot${each.done ? "" : " alive"}`} />
                    <span className="name" title={each.name}>{each.suite}</span>
                    {(evals.suites.find(suite => suite.suite === each.suite)?.versions?.length ?? 1) > 1 ? <span className="version">{versionTag(each.version)}</span> : null}
                    <span className="tag">{evalTag(each)}</span>
                  </Node>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </>
  );
}

/** The environments, each opening to its training runs and the suites that play it. */
function EnvironmentsTree({ place, system }: { place: Place; system: System }) {
  const { data: known } = useEnvironments();
  const [folds, fold] = useFolds();
  if (!known) return null;
  if (!known.length) return <div className="empty">No environment yet.</div>;
  const shown = place.kind === "environment" ? place.environment : null;
  return (
    <>
      {known.map(each => {
        const key = `environment:${each.environment}`, open = folds[key] ?? each.environment === shown;
        const runs = (each.runs ?? []).map(id => system.runs.find(run => run.run === id)).filter(run => run !== undefined);
        const suites = each.suites ?? [];
        return (
          <div key={each.environment}>
            <Node to={environmentPlace(each.environment)} current={each.environment === shown}>
              <Twist open={open} has={runs.length + suites.length > 0} onToggle={() => fold(key, !open)} />
              <span className={`dot${each.offered ? " alive" : ""}`} title={each.offered ? "offered by a launcher alive" : "no launcher alive offers it"} />
              <span className="name" title={each.environment}>{each.name}</span>
              <span className="tag">{runs.length || ""}</span>
            </Node>
            {open && runs.length + suites.length ? (
              <div className="children">
                {runs.map(run => (
                  <Node key={run.run} to={runPlace(run.run)}>
                    <RunDot run={run} host={system.host} />
                    <span className="name" title={`id: ${run.run}`}>{nameOf(run)}</span>
                    <span className="tag">{running(run, system.host)}</span>
                  </Node>
                ))}
                {suites.map(suite => (
                  <Node key={`suite:${suite}`} to={suitePlace(suite)}>
                    <span className="num">suite</span>
                    <span className="name">{suite}</span>
                  </Node>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </>
  );
}

export const SECTIONS: [string, string][] = [
  ["outcomes", "Outcomes"], ["rows", "Rows"], ["steps", "Steps"], ["pace", "Pace"], ["queue", "Queue"], ["inference", "Inference"], ["ledger", "Ledger"],
];

/** The runs left out of the statistics, as this browser remembers them. */
export const useHidden = () => useStored<string[]>("monitor.hidden", []);

/** The statistics: its sections, and the training runs drawn (each in its color; a click leaves it out or takes it back). */
function StatisticsTree({ place, system }: { place: Place; system: System }) {
  const [hidden, setHidden] = useHidden();
  const colorOf = useRunColor();
  const runs = system.runs.filter(run => run.kind !== "eval");  // (evals are on the evals page)
  const toggle = (run: string) => setHidden(hidden.includes(run) ? hidden.filter(each => each !== run) : [...hidden, run]);
  return (
    <>
      {SECTIONS.map(([key, name]) => (
        <Node key={key} to={statisticsPlace(key)} current={place.kind === "statistics" && place.section === key}><span className="name">{name}</span></Node>
      ))}
      <div className="label">Runs · {runs.filter(run => !hidden.includes(run.run)).length} of {runs.length}</div>
      {runs.map(run => {
        const off = hidden.includes(run.run);
        return (
          <div
            key={run.run}
            className={`node${off ? " off" : ""}`}
            role="checkbox"
            tabIndex={0}
            aria-checked={!off}
            title={off ? "draw this run" : "leave this run out"}
            onClick={() => toggle(run.run)}
            onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggle(run.run); } }}
          >
            <span className="swatch" style={{ background: colorOf(run.run) }} />
            <span className="name">{nameOf(run)}</span>
            <span className="tag">{running(run, system.host)}</span>
          </div>
        );
      })}
    </>
  );
}
