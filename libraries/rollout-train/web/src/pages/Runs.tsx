// Every run of the ledger, running ones first, each with how it is going; and the episodes no run asked for.

import { memo } from "react";
import { Link } from "react-router-dom";
import { useFeeds, useKnown, useLaunches, useSystem } from "../api/queries";
import type { Run } from "../api/types";
import { Spark } from "../components/charts";
import { Card, Empty, Head, Mark, Spec, Specs, Tile } from "../components/ui";
import { clock, figure, mean } from "../lib/format";
import { episodeReward, nameOf, reported, runKind, slotRewards, stateKind } from "../lib/model";
import { Marks } from "../components/checkpoints";
import { episodePlace, launchPlace, runPlace } from "../lib/places";
import { LaunchList } from "../components/launches";
import { RunDot, running, Wrote } from "../layout/runs";

export function Runs() {
  const { data: system } = useSystem();
  const { data: feeds } = useFeeds();
  const { data: launched } = useLaunches();
  if (!system) return <Empty>Reading the runs…</Empty>;
  const others = (feeds ?? []).filter(run => !run.labels.run);
  const runs = system.runs.filter(run => run.kind !== "eval");  // (evals are on their own page)
  const launches = (launched?.launches ?? []).filter(each => each.asked.kind !== "eval");
  const states = (["running", "paused", "idle", "finished", "stopped", "failed", "lost", "ended"] as const).map(name => [name, runs.filter(run => run.state === name).length] as const).filter(([, count]) => count);
  return (
    <>
      <Head title={<span className="head-with-action">Runs<Link to={launchPlace} className="action">New run</Link></span>}>
        {states.length ? (
          <Specs>
            {states.map(([name, count]) => <Spec key={name} label={name} kind={runKind(name)}>{count}</Spec>)}
          </Specs>
        ) : null}
      </Head>
      {launches.length ? <LaunchList launches={launches} system={system} /> : null}
      {runs.length ? <div className="tiles wide-tiles">{runs.map(run => <RunTile key={run.run} run={run} host={system.host} />)}</div> : <Empty>No run yet.</Empty>}
      {others.length ? <Card title={<Link to="/episodes" className="linkish">Episodes outside a run</Link>} note={String(others.length)} /> : null}
    </>
  );
}

const RunTile = memo(function RunTile({ run, host }: { run: Run; host: string }) {
  const known = useKnown();
  const recent = run.done.slice(-12), solved = recent.flatMap(line => line.solved), rewards = recent.flatMap(line => line.rewards);
  const committed = run.steps.filter(step => step.state === "committed").length;
  const head = run.steps.findLast(step => step.state === "committed")?.makes;
  const from = run.from ?? run.steps[0]?.parent ?? null;
  return (
    <Tile to={runPlace(run.run)} className={`rail ${runKind(run.state)}`}>
      <header><RunDot run={run} host={host} /><b title={`id: ${run.run}`}>{nameOf(run)}</b><span className="what">from {from ? `${known.short(from)} (${known.origin(from)})` : known.base(known.checkpoint(head)?.base)}</span><span className="faint small">{running(run, host)}</span></header>
      <div className="cells four">
        <div className={`cell ${run.open.length ? "accent" : "waiting"}`}><span>in flight</span><b>{run.open.length}</b><small>{run.next.length} toward a step</small></div>
        <div className="cell"><span>groups done</span><b>{run.done.length}</b><small>of {run.decided} decided</small></div>
        <div className="cell violet"><span>steps</span><b>{committed}</b><small className="mono">{head ? known.short(head) : "none yet"}{head ? <> <Marks names={known.bookmarks(head)} /></> : null}</small></div>
        {reported(solved) ? <div className="cell good"><span>solved</span><b>{`${Math.round((100 * solved.filter(Boolean).length) / solved.length)}%`}</b><small>of the last {recent.length} groups</small></div>
          : <div className="cell"><span>mean reward</span><b>{figure(mean(rewards))}</b><small>of the last {recent.length} groups</small></div>}
      </div>
      {run.done.length > 1 ? (
        <div>
          <Spark values={run.done.map(line => mean(line.rewards) ?? 0)} kind="s-accent" width={420} height={40} fill />
          <div className="small muted">mean reward by group</div>
        </div>
      ) : null}
      <div className="facts">
        <span><Wrote run={run} /></span>
        {run.host ? <span>on <b>{run.host}</b></span> : null}
        {run.episodes_at === "here" ? null : <span className={run.reached === false || !run.episodes_at ? "t-warm" : ""}>{run.episodes_at ? `episodes on ${run.episodes_at}` : "ledger only"}</span>}
      </div>
    </Tile>
  );
});

export function Outside() {
  const { data: feeds } = useFeeds();
  const others = (feeds ?? []).filter(run => !run.labels.run);
  return (
    <>
      <Head title="Episodes outside a run" />
      {others.length ? (
        <div className="tiles">
          {others.map(run => (
            <Tile key={run.run_id} to={episodePlace(run.run_id)} className={`rail ${stateKind(run.state)}`}>
              <header><b>{run.labels.title ?? run.labels.task ?? run.run_id.slice(-8)}</b><span className="what" /><Mark state={run.state} /></header>
              <div className="big">{Object.values(run.rewards).length ? figure(episodeReward(run.rewards)) : <span className="faint">…</span>}</div>
              <div className="facts">{slotRewards(run.rewards).map(([slot, value]) => <span key={slot}>{slot} <b>{figure(value)}</b></span>)}<span><b>{run.samples}</b> samples</span><span><b>{run.slots.length}</b> slots</span><span>{clock(run.started)}</span></div>
            </Tile>
          ))}
        </div>
      ) : <Empty>None.</Empty>}
    </>
  );
}
