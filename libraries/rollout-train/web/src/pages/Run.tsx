// A training run: where it started from and where it is now, how its groups went, what is in flight, every group's
// rewards, its steps, the tasks it played, each suite's score along its line, and its settings; while it waits in the
// queue, its place there and why. An eval's run has a page of its own (`EvalRun`).

import { memo } from "react";
import { Link } from "react-router-dom";
import { useKnown, useLaunches, useSystem } from "../api/queries";
import { useQueued } from "../components/queue";
import { launchState } from "../lib/launches";
import { RunControls } from "../components/control";
import { Rename } from "../components/Rename";
import type { OpenGroup, Run as RunData, Step, Checkpoint } from "../api/types";
import { RewardsChart, Sized } from "../components/charts";
import { Card, Cells, Dots, Empty, Head, Kpi, Kpis, Legend, Mark, SectionTitle, Spec, Specs, Stages, Table, Tile } from "../components/ui";
import { byNumber, figure, mean, shareOf, span } from "../lib/format";
import { asked, type GroupEntry, groupsOf, learnableText, madeBy, nameOf, nothingText, range, reported, runKind, stateKind, towardStep } from "../lib/model";
import { groupPlace, stepPlace } from "../lib/places";
import { BaseName, CheckpointTag } from "../components/checkpoints";
import { PathCard } from "../components/scores";
import { RunSettingsSection } from "../components/settings";
import { Ago, Elsewhere, running, Wrote } from "../layout/runs";
import { EvalRun } from "./EvalRun";

export function Run({ name }: { name: string }) {
  const { data: system } = useSystem();
  if (!system) return <Empty>Reading the run…</Empty>;
  const run = system.runs.find(each => each.run === name);
  if (!run) return <NotStarted id={name} />;
  if (run.kind === "eval") return <EvalRun run={run.run} />;
  const made = madeBy(system.checkpoints, run.run);  // (the checkpoints it made, oldest first)
  return (
    <>
      <RunHead run={run} made={made} host={system.host} />
      <Elsewhere run={run} />
      <RunFigures run={run} made={made} />
      <InFlight run={run} />
      <Card title="Rewards by group">
        {run.done.length ? (
          <>
            <Sized>{width => <RewardsChart run={run} width={width} />}</Sized>
            <Legend items={[
              ...(reported(run.done.flatMap(line => line.solved)) ? [{ name: "solved", color: "var(--good)" }, { name: "not solved", color: "var(--faint)" }] : []),
              { name: "mean", className: "rule" },
              { name: "trained on", color: "var(--accent)" }, { name: "in the step being taken", color: "var(--violet)" },
              { name: "waits for a step", color: "var(--warm)" }, { name: "skipped", className: "hollow" }, { name: "no episode, or the step failed", color: "var(--bad)" },
            ]} />
          </>
        ) : <Empty>No group done yet.</Empty>}
      </Card>
      <div className="cols">
        <StepsCard run={run} />
        <TasksCard run={run} />
      </div>
      {made.length ? <PathCard checkpoint={made.at(-1)!.id} /> : null}
      <RunSettingsSection run={run.run} />
    </>
  );
}

/** A run asked for whose run has not started yet: its state (where it is in the queue, and why it waits), when it was
 * asked for, and its id. */
function NotStarted({ id }: { id: string }) {
  const { data: launched } = useLaunches();
  const queued = useQueued(id);
  const launch = launched?.launches?.find(each => each.run === id);
  if (!launch) return <Empty>There is no run {id}.</Empty>;
  const said = launchState(launch);
  return (
    <Head title={`Run ${launch.asked.name}`}>
      <Specs>
        <Spec label="state" kind={queued ? "warm" : said.kind}>
          {queued ? `waiting · ${queued}` : said.reason ? `${said.state}: ${said.reason}` : said.state}
        </Spec>
        <Spec label="asked"><Ago at={launch.at} /> ago</Spec>
        <Spec label="id">{id}</Spec>
      </Specs>
    </Head>
  );
}

const RunHead = memo(function RunHead({ run, made, host }: { run: RunData; made: Checkpoint[]; host: string }) {
  const known = useKnown();
  const queued = useQueued(run.run);
  const from = run.from ?? run.steps[0]?.parent ?? null, newest = made.at(-1), many = run.channels.length > 1;
  return (
    <Head title={<span className="head-with-action"><span>Run <Rename id={run.run} name={nameOf(run)} /></span><RunControls run={run} /></span>}>
      <Specs>
        {queued ? <Spec label="state" kind="warm">waiting · {queued}</Spec> : (
          <Spec label="state" kind={runKind(run.state)}>{running(run, host)}{run.ending?.detail ? `: ${run.ending.detail}` : ""} · <Wrote run={run} /></Spec>
        )}
        {newest?.base ? <Spec label={known.checkpoint(newest.base) ? "over" : "base model"}><BaseName base={newest.base} /></Spec> : null}
        <Spec label="from"><CheckpointTag id={from} /></Spec>
        {newest ? <Spec label="now" kind="violet"><CheckpointTag id={newest.id} bare /></Spec> : null}
        {run.channels.map(channel => (
          <Spec key={channel.channel} label={many ? `serving on ${channel.channel}` : "serving"} kind="accent">
            {channel.adapter ? <CheckpointTag id={channel.adapter} bare /> : "the model its engines started with"}
          </Spec>
        ))}
        <Spec label="id">{run.run}</Spec>
        <Spec label="fence">{run.fence ?? "–"}</Spec>
        <Spec label="directory">{run.directory ?? "–"}</Spec>
        {run.starts > 1 ? <Spec label="started">{run.starts} times</Spec> : null}
      </Specs>
    </Head>
  );
});

/** From what it started, where it is now, and how its groups went, early and late. */
const RunFigures = memo(function RunFigures({ run, made }: { run: RunData; made: Checkpoint[] }) {
  const known = useKnown();
  const from = run.from ?? run.steps[0]?.parent ?? null, newest = made.at(-1);
  const trained = run.done.filter(line => line.update).length, last = run.done.at(-1);
  const committed = run.steps.filter(step => step.state === "committed").length, toward = towardStep(run);
  // (each channel's newest measurement, summed: every channel the run serves)
  const measured = run.channels.map(channel => channel.throughput.at(-1)).filter(each => each != null);
  const sum = (key: "tokens_per_second" | "mean_concurrency") => measured.reduce((total, each) => total + (each[key] ?? 0), 0);
  const solvedOf = (lines: RunData["done"]) => lines.flatMap(line => line.solved.map(Boolean));
  const outcomes = solvedOf(run.done), half = Math.ceil(run.done.length / 2), said = reported(run.done.flatMap(line => line.solved));
  const all = run.done.filter(line => line.solved.length && line.solved.every(Boolean)).length;
  const none = run.done.filter(line => line.solved.length && !line.solved.some(Boolean)).length;
  return (
    <Kpis>
      <Kpi label="Started from" value={from ? known.short(from) : "base"} note={from ? known.origin(from) : <BaseName base={made[0]?.base} short />} />
      <Kpi label="Now" value={newest ? newest.short : "–"} note={newest ? `depth ${newest.depth} · ${made.filter(each => each.weights).length} of ${made.length} kept` : ""} />
      <Kpi label="Steps" value={`${run.steps.length}`} note={`${committed} committed · ${toward.learnable.length}${toward.perStep != null ? ` of ${toward.perStep}` : ""} toward the next`} />
      <Kpi label="Groups done" value={`${run.done.length}`} note={`${trained} trained on, of ${run.decided} decided`} />
      {said ? <Kpi label="Groups solved" value={`${all} · ${run.done.length - all - none} · ${none}`} note="all · some · none solved" /> : null}
      <Kpi label="Episodes" value={`${outcomes.length}`} note={said ? `${outcomes.filter(Boolean).length} solved · ${shareOf(outcomes)}` : ""} />
      {said ? <Kpi label="Solved, early → late" value={run.done.length > 1 ? `${shareOf(solvedOf(run.done.slice(0, half)))} → ${shareOf(solvedOf(run.done.slice(half)))}` : "–"} /> : null}
      <Kpi label="Mean reward" value={figure(mean(run.done.flatMap(line => line.rewards)))} />
      <Kpi label="Rows unlocked" value={last ? `${last.unlocked}` : "–"} />
      <Kpi label="Inference" value={measured.length ? `${figure(sum("tokens_per_second"))} tok/s` : "–"} note={measured.length ? `${figure(sum("mean_concurrency"))} requests at once` : ""} />
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
  const toward = towardStep(run), waiting = toward.learnable.length + toward.nothing.length, nothing = nothingText(toward);
  const note = [`${run.open.length} groups playing`, stepping.length ? `step ${stepping.map(step => step.step).join(", ")} being taken` : null].filter(Boolean).join(" · ");
  return (
    <>
      <SectionTitle title="In flight" note={note} />
      {run.open.length || stepping.length || waiting ? (
        <div className="tiles">
          {stepping.map(step => <SteppingTile key={`s${step.step}`} run={run} step={step} groups={groups} />)}
          {waiting ? (
            <div className="tile rail warm">
              <header><b>Toward the next step</b><span className="what">{learnableText(toward)}</span></header>
              {toward.learnable.length ? <div className="members">{toward.learnable.map(number => <Member key={number} run={run} number={number} groups={groups} />)}</div> : null}
              {nothing ? <div className="small muted">{nothing} · {range(toward.nothing)}</div> : null}
            </div>
          ) : null}
          {run.open.map(group => <OpenTile key={group.number} run={run.run} group={group} />)}
        </div>
      ) : <Empty>Nothing in flight.</Empty>}
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
    <Card title="Steps">
      <Table
        heads={[["step"], ["made"], ["groups"], ["solved", "n"], ["segments", "n"], ["moved", "n"], ["took", "n"]]}
        keys={recent.map(step => step.step)}
        rows={recent.map(step => {
          const checkpoint = known.checkpoint(step.makes);
          const solved = step.groups.map(number => groups.get(number)?.line).filter(Boolean).flatMap(line => line!.solved);
          return [
            { text: `S${step.step}`, kind: "key" },
            step.state === "committed" ? <span className="mono" title={known.title(step.makes)}>{known.short(step.makes)}</span> : { text: step.state, kind: stateKind(step.state) },
            <span>{range(step.groups)}{step.skipped.length ? <span className="faint"> + {step.skipped.length} skipped</span> : null}</span>,
            reported(solved) ? `${solved.filter(Boolean).length}/${solved.length}` : "–",
            figure(step.segments),
            checkpoint?.metrics.kl_moved?.toFixed(4) ?? "–",
            span(checkpoint?.metrics.update_seconds ?? checkpoint?.metrics.seconds),
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
          reported(task.last.solved) ? `${task.last.solved.filter(Boolean).length}/${task.last.rewards.length}` : "–",
        ])}
        to={played.map(([, task]) => groupPlace(run.run, task.last.group))}
      />
    </Card>
  );
});
