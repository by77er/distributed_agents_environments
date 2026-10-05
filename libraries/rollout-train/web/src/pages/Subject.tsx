// A subject's evals: every eval a checkpoint or a base model has had, by the version of a suite each played (two
// versions' scores do not compare), with each environment's score over time; for a checkpoint, each suite's score along
// its line from the base model.

import { Link } from "react-router-dom";
import { useEvals, useHistory, useKnown } from "../api/queries";
import type { CheckpointEval, SubjectKind } from "../api/types";
import { LineChart, Sized } from "../components/charts";
import { Marks } from "../components/checkpoints";
import { PathCard } from "../components/scores";
import { Card, Empty, Head, Legend, Mark, Spec, Specs, Table } from "../components/ui";
import { readable } from "../lib/environments";
import { clock, figure, percent } from "../lib/format";
import { type HistoryGroup, historyOf, historySeries, type Score, scoresOf } from "../lib/history";
import { checkpointPlace, evalPlace, runPlace, stepPlace, suitePlace } from "../lib/places";
import { versionsOf, versionTag } from "../lib/suites";
import { Ago } from "../layout/runs";

export function SubjectView({ kind, id }: { kind: SubjectKind; id: string }) {
  const { data: history, isError } = useHistory(kind, id);
  if (isError && !history) return <Empty>There is no {kind === "checkpoint" ? "checkpoint" : "base model"} {id}.</Empty>;
  if (!history) return <Empty>Reading the evals…</Empty>;
  const subject = history.subject, groups = historyOf(history.evals);
  return (
    <>
      <Head title={kind === "checkpoint" ? <span><span className="mono" title={subject.id}>{subject.short}</span> <Marks names={subject.bookmarks} /></span> : subject.id}>
        <Specs>
          {kind === "checkpoint" ? <Spec label="checkpoint"><Link to={checkpointPlace(subject.id)}>{subject.id}</Link></Spec> : null}
          {subject.run ? <Spec label="made by"><Link to={runPlace(subject.run)} title={`id: ${subject.run}`}>{subject.name ?? subject.run}</Link></Spec> : null}
          {subject.run && subject.step != null ? <Spec label="step" kind="violet"><Link to={stepPlace(subject.run, subject.step)}>S{subject.step}</Link></Spec> : null}
          <Spec label="evals">{history.evals.length}</Spec>
          <Spec label="suites">{subject.suites.length}</Spec>
          {subject.playing ? <Spec label="playing" kind="good">{subject.playing}</Spec> : null}
        </Specs>
      </Head>
      {kind === "checkpoint" ? <PathCard checkpoint={subject.id} /> : null}
      {groups.length ? groups.map(group => <VersionCard key={group.version} group={group} />) : <Empty>No eval yet.</Empty>}
    </>
  );
}

const shareCell = (score: Score | null) => (score == null || score.share == null ? "–" : percent(score.share));
const rewardCell = (score: Score | null) => (score == null ? "–" : figure(score.reward));

/** Who asked for an eval: a training run's schedule at a step (opening the run), or someone by hand. */
function AskedBy({ each }: { each: CheckpointEval }) {
  const known = useKnown();
  if (each.asked_by !== "schedule" || !each.by) return <>by hand</>;
  return <Link to={runPlace(each.by)} className="linkish" onClick={event => event.stopPropagation()}>{known.run(each.by)}{each.step != null ? ` · S${each.step}` : ""}</Link>;
}

/** The evals of one version of a suite, newest first, with each environment's score over time above them. */
export function VersionCard({ group }: { group: HistoryGroup }) {
  const { data: evals } = useEvals();
  const suite = evals?.suites.find(each => each.suite === group.suite);
  const tagged = (suite ? versionsOf(suite).length : group.number) > 1;
  const several = group.environments.length > 1;
  const series = historySeries(group).map((each, place) => ({ ...each, place })).filter(each => each.points.length);
  const solved = series.length > 0 && series.every(each => each.measure === "solved");
  const lines = series.map(each => ({
    name: `${several ? readable(each.environment) : group.label}${solved ? "" : each.measure === "solved" ? " (solved)" : " (mean reward)"}`,
    color: `var(--series-${(each.place % 8) + 1})`,
    points: each.points,
  }));
  const charted = series.some(each => each.points.length > 1);
  const heads: [string, string?][] = [["eval"]];
  for (const environment of group.environments) {
    const name = several ? `${readable(environment)} ` : "";
    heads.push([`${name}solved`, "n"], [several ? `${name}reward` : "mean reward", "n"]);
  }
  heads.push(["episodes", "n"], ["asked by"], ["started"], ["state"]);
  return (
    <Card title={<><Link to={suitePlace(group.suite)}>{group.suite}</Link>{tagged ? <span className="tag-version">{versionTag(group.version)}</span> : null}</>}>
      {charted ? (
        <>
          <Sized>{width => (
            <LineChart series={lines} width={width} height={190} time dots label={`${group.label} over time`}
              y={solved ? { min: 0, max: 1 } : { zero: false }} format={solved ? percent : figure} yTick={solved ? percent : undefined} />
          )}</Sized>
          {lines.length > 1 ? <Legend items={lines} /> : null}
          <div style={{ height: 14 }} />
        </>
      ) : null}
      <Table
        heads={heads}
        keys={group.evals.map(each => each.run)}
        rows={group.evals.map(each => [
          <b title={`id: ${each.run}`}>{each.name}</b>,
          ...scoresOf(group, each).flatMap(score => [shareCell(score), rewardCell(score)]),
          <span title={`${each.episodes} of each start`}>{each.played === each.expected ? each.played : `${each.played} of ${each.expected}`}</span>,
          <AskedBy each={each} />,
          each.started ? <span title={clock(each.started)}><Ago at={each.started} /> ago</span> : "–",
          <Mark state={each.done ? "ended" : "running"}>{each.done ? "done" : "playing"}</Mark>,
        ])}
        to={group.evals.map(each => evalPlace(each.run))}
      />
    </Card>
  );
}
