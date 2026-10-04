// A training run: where it started from and where it is now, how its groups went, what is in flight, every group's
// rewards, its steps and the tasks it played.

import { memo } from "react";
import { Link } from "react-router-dom";
import { useKnown, useSystem } from "../api/queries";
import { Rename } from "../components/Rename";
import type { OpenGroup, Run as RunData, Step, Version } from "../api/types";
import { RewardsChart, Sized } from "../components/charts";
import { Card, Cells, Dots, Empty, Head, Kpi, Kpis, Legend, Mark, SectionTitle, Spec, Specs, Stages, Table, Tile } from "../components/ui";
import { byNumber, figure, mean, shareOf, span } from "../lib/format";
import { asked, type GroupEntry, groupsOf, madeBy, nameOf, range, stateKind } from "../lib/model";
import { groupPlace, stepPlace } from "../lib/places";
import { VersionTag } from "../components/versions";
import { Ago, Elsewhere, running, Wrote } from "../layout/runs";

export function Run({ name }: { name: string }) {
  const { data: system } = useSystem();
  if (!system) return <Empty>Reading the run…</Empty>;
  const run = system.runs.find(each => each.run === name);
  if (!run) return <Empty>There is no run {name}.</Empty>;
  const made = madeBy(system.versions, run.run);  // (the versions it made, oldest first)
  return (
    <>
      <RunHead run={run} made={made} host={system.host} />
      <Elsewhere run={run} />
      <RunFigures run={run} made={made} />
      <InFlight run={run} />
      <Card title="Rewards by group" note="each dot an episode; rewards are each task's own">
        {run.done.length ? (
          <>
            <Sized>{width => <RewardsChart run={run} width={width} />}</Sized>
            <Legend items={[
              { name: "solved", color: "var(--good)" }, { name: "not solved", color: "var(--faint)" }, { name: "mean", className: "rule" },
              { name: "trained on", color: "var(--accent)" }, { name: "in the step being taken", color: "var(--violet)" },
              { name: "waits for a step", color: "var(--warm)" }, { name: "skipped", className: "hollow" }, { name: "no episode, or the step failed", color: "var(--bad)" },
            ]} />
          </>
        ) : <Empty>No group is done with yet.</Empty>}
      </Card>
      <div className="cols">
        <StepsCard run={run} />
        <TasksCard run={run} />
      </div>
    </>
  );
}

const RunHead = memo(function RunHead({ run, made, host }: { run: RunData; made: Version[]; host: string }) {
  const channel = run.channels.find(each => each.adapter) ?? run.channels[0];
  const from = run.from ?? run.steps[0]?.parent ?? null, newest = made.at(-1);
  return (
    <Head title={<>Run <Rename id={run.run} name={nameOf(run)} /></>}>
      <Specs>
        <Spec label="state" kind={run.state === "running" ? "good" : run.state === "idle" ? "warm" : ""}>{running(run, host)} · <Wrote run={run} /></Spec>
        {newest?.base ? <Spec label="base model">{newest.base}</Spec> : null}
        <Spec label="from"><VersionTag id={from} /></Spec>
        {newest ? <Spec label="now" kind="violet"><VersionTag id={newest.id} bare /></Spec> : null}
        {channel?.adapter ? <Spec label="serving" kind="accent"><VersionTag id={channel.adapter} bare /></Spec> : null}
        <Spec label="id">{run.run}</Spec>
        <Spec label="fence">{run.fence ?? "–"}</Spec>
        <Spec label="directory">{run.directory ?? "–"}</Spec>
        {run.starts > 1 ? <Spec label="started">{run.starts} times</Spec> : null}
      </Specs>
    </Head>
  );
});

/** From what it started, where it is now, and how its groups went, early and late. */
const RunFigures = memo(function RunFigures({ run, made }: { run: RunData; made: Version[] }) {
  const known = useKnown();
  const from = run.from ?? run.steps[0]?.parent ?? null, newest = made.at(-1);
  const trained = run.done.filter(line => line.update).length, last = run.done.at(-1);
  const committed = run.steps.filter(step => step.state === "committed").length;
  const channel = run.channels.find(each => each.adapter) ?? run.channels[0];
  const throughput = channel?.throughput.at(-1);
  const solvedOf = (lines: RunData["done"]) => lines.flatMap(line => line.solved);
  const outcomes = solvedOf(run.done), half = Math.ceil(run.done.length / 2);
  const all = run.done.filter(line => line.solved.length && line.solved.every(Boolean)).length;
  const none = run.done.filter(line => line.solved.length && !line.solved.some(Boolean)).length;
  return (
    <Kpis>
      <Kpi label="Started from" value={from ? known.short(from) : "base"} note={from ? known.origin(from) : made[0]?.base ?? "the base model"} />
      <Kpi label="Now" value={newest ? newest.short : "–"} note={newest ? `depth ${newest.depth} · ${made.filter(each => each.weights).length} of ${made.length} versions kept` : "no step committed"} />
      <Kpi label="Steps" value={`${run.steps.length}`} note={`${committed} committed · ${run.next.length} waiting`} />
      <Kpi label="Groups done" value={`${run.done.length}`} note={`${trained} trained on, of ${run.decided} decided`} />
      <Kpi label="Groups solved" value={`${all} · ${run.done.length - all - none} · ${none}`} note="all · some · none solved" />
      <Kpi label="Episodes" value={`${outcomes.length}`} note={`${outcomes.filter(Boolean).length} solved · ${shareOf(outcomes)}`} />
      <Kpi label="Solved, early → late" value={run.done.length > 1 ? `${shareOf(solvedOf(run.done.slice(0, half)))} → ${shareOf(solvedOf(run.done.slice(half)))}` : "–"}
        note={run.done.length > 1 ? `groups 1–${half}, then the ${run.done.length - half} after` : ""} />
      <Kpi label="Mean reward" value={figure(mean(run.done.flatMap(line => line.rewards)))} note="over every episode done" />
      <Kpi label="Rows unlocked" value={last ? `${last.unlocked}` : "–"} note="of the catalog" />
      <Kpi label="Inference" value={throughput ? `${figure(throughput.tokens_per_second)} tok/s` : "–"} note={throughput ? `${figure(throughput.mean_concurrency)} requests at once` : "no measurement yet"} />
    </Kpis>
  );
});

function Member({ run, number, groups }: { run: RunData; number: number; groups: Map<number, GroupEntry> }) {
  const line = groups.get(number)?.line;
  return (
    <Link to={groupPlace(run.run, number)} className="member">
      <b>#{number}</b><span className="what">{groups.get(number)?.task ?? ""}</span>
      {line ? <Dots line={line} /> : <span />}
      <span className="faint">{line?.rewards.length ? line.rewards.map(figure).join(" ") : ""}</span>
    </Link>
  );
}

const InFlight = memo(function InFlight({ run }: { run: RunData }) {
  const groups = groupsOf(run), stepping = run.steps.filter(step => step.state === "stepping");
  const waiting = run.next.filter(number => groups.get(number)?.line);
  const note = [`${run.open.length} groups playing`, stepping.length ? `step ${stepping.map(step => step.step).join(", ")} being taken` : null].filter(Boolean).join(" · ");
  return (
    <>
      <SectionTitle title="In flight" note={note} />
      {run.open.length || stepping.length || waiting.length ? (
        <div className="tiles">
          {stepping.map(step => <SteppingTile key={`s${step.step}`} run={run} step={step} groups={groups} />)}
          {waiting.length ? (
            <div className="tile rail warm">
              <header><b>Toward the next step</b><span className="what">{waiting.length} recorded, waiting for a step</span></header>
              <div className="members">{waiting.map(number => <Member key={number} run={run} number={number} groups={groups} />)}</div>
            </div>
          ) : null}
          {run.open.map(group => <OpenTile key={group.number} run={run.run} group={group} />)}
        </div>
      ) : <Empty>Nothing is in flight.</Empty>}
    </>
  );
});

function SteppingTile({ run, step, groups }: { run: RunData; step: Step; groups: Map<number, GroupEntry> }) {
  const known = useKnown();
  return (
    <Tile to={stepPlace(run.run, step.step)} className="rail violet">
      <header><b>Step {step.step}</b><span className="what">→ <span className="mono">{known.short(step.makes)}</span> · {step.segments ?? "?"} segments</span><Mark state="stepping"><Ago at={step.decided} /></Mark></header>
      <div className="members">{step.groups.map(number => <Member key={number} run={run} number={number} groups={groups} />)}</div>
    </Tile>
  );
}

const OpenTile = memo(function OpenTile({ run, group }: { run: string; group: OpenGroup }) {
  return (
    <Tile to={groupPlace(run, group.number)}>
      <header><b>#{group.number}</b><span className="what">{group.task} · {group.title ?? ""}</span><span className="faint small"><Ago at={group.decided} otherwise="" /></span></header>
      <Stages stage={group.stage} ended={group.ended} count={group.count} />
      {group.count ? <Cells episodes={asked(group)} /> : null}
    </Tile>
  );
});

const StepsCard = memo(function StepsCard({ run }: { run: RunData }) {
  const known = useKnown();
  const recent = [...run.steps].reverse().slice(0, 10), groups = groupsOf(run);
  return (
    <Card title="Steps" note="newest first">
      <Table
        heads={[["step"], ["made"], ["groups"], ["solved"], ["segments", "n"], ["moved", "n"], ["took", "n"]]}
        keys={recent.map(step => step.step)}
        rows={recent.map(step => {
          const version = known.version(step.makes);
          const solved = step.groups.map(number => groups.get(number)?.line).filter(Boolean).flatMap(line => line!.solved);
          return [
            { text: `S${step.step}`, kind: "key" },
            step.state === "committed" ? <span className="mono" title={known.title(step.makes)}>{known.short(step.makes)}</span> : { text: step.state, kind: stateKind(step.state) },
            <span>{range(step.groups)}{step.skipped.length ? <span className="faint"> + {step.skipped.length} skipped</span> : null}</span>,
            solved.length ? `${solved.filter(Boolean).length}/${solved.length}` : "–",
            figure(step.segments),
            version?.metrics.kl_moved?.toFixed(4) ?? "–",
            span(version?.metrics.update_seconds ?? version?.metrics.seconds),
          ];
        })}
        to={recent.map(step => stepPlace(run.run, step.step))}
      />
    </Card>
  );
});

const TasksCard = memo(function TasksCard({ run }: { run: RunData }) {
  const tasks = new Map<string, { title: string; groups: number; trained: number; last: RunData["done"][number] }>();
  for (const line of run.done) {
    const task = tasks.get(line.task) ?? { title: line.title, groups: 0, trained: 0, last: line };
    task.groups += 1;
    task.trained += line.update ? 1 : 0;
    task.last = line;
    tasks.set(line.task, task);
  }
  const played = [...tasks].sort(([a], [b]) => byNumber(a, b));
  return (
    <Card title="Tasks played" note={`${played.length} rows`}>
      <Table
        heads={[["task"], ["groups", "n"], ["trained", "n"], ["last rewards"], ["solved", "n"]]}
        keys={played.map(([key]) => key)}
        rows={played.map(([key, task]) => [
          { text: key, kind: "key" }, task.groups, task.trained, task.last.rewards.map(figure).join(" ") || "–",
          `${task.last.solved.filter(Boolean).length}/${task.last.rewards.length}`,
        ])}
        to={played.map(([, task]) => groupPlace(run.run, task.last.group))}
      />
    </Card>
  );
});
