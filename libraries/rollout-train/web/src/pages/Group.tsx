// A step (one update of the policy, over the groups it covers), and a group (a task, a start and its episodes).

import { memo } from "react";
import { Link } from "react-router-dom";
import { useGroup, useKnown, useSystem } from "../api/queries";
import { CheckpointTag } from "../components/checkpoints";
import type { GroupEpisode, Metrics } from "../api/types";
import { NotFound } from "../api/client";
import { Card, Dots, Empty, Head, Kpi, Kpis, Mark, Pairs, SectionTitle, Spec, Specs, Stages, Tile } from "../components/ui";
import { clock, figure, mean, span, tokens } from "../lib/format";
import { asked, groupsOf, isWaiting, outcomeOf, range, reported, slotRewards, stateKind, stepOf } from "../lib/model";
import { episodePlace, groupPlace, stepPlace } from "../lib/places";
import { Ago, Elsewhere } from "../layout/runs";

const UPDATE = ["kl_moved", "kl_floor", "loss", "clip_fraction", "mean_mismatch", "mean_weight", "truncated_fraction", "optimizer_steps", "tokens", "longest_segment_tokens", "peak_gpu_gib"];
const updatePairs = (metrics: Metrics): [string, string][] => [
  ...UPDATE.filter(key => metrics[key] !== undefined).map(key => [key.replaceAll("_", " "), figure(metrics[key])] as [string, string]),
  ["took", span(metrics.update_seconds ?? metrics.seconds)],
];

export function StepView({ run: name, number }: { run: string; number: number }) {
  const { data: system } = useSystem();
  const known = useKnown();
  if (!system) return <Empty>Reading the step…</Empty>;
  const run = system.runs.find(each => each.run === name), step = run?.steps.find(each => each.step === number);
  if (!run || !step) return <Empty>There is no step {number}.</Empty>;
  const checkpoint = known.checkpoint(step.makes), metrics = checkpoint?.metrics, groups = groupsOf(run);
  const solved = step.groups.map(each => groups.get(each)?.line).filter(Boolean).flatMap(line => line!.solved);
  return (
    <>
      <Head title={`Step ${step.step}`}>
        <Specs>
          <Spec label="makes" kind="violet">{step.makes ? <CheckpointTag id={step.makes} bare /> : "–"}</Spec>
          <Spec label="from"><CheckpointTag id={step.parent} /></Spec>
          <Spec label="state" kind={stateKind(step.state)}>{step.state}</Spec>
          <Spec label="decided">{clock(step.decided) || "–"}</Spec>
        </Specs>
      </Head>
      <Kpis>
        <Kpi label="Groups" value={`${step.groups.length}`} note={`${range(step.groups)}${step.skipped.length ? ` · ${step.skipped.length} gave nothing to train on` : ""}`} />
        {reported(solved) ? <Kpi label="Solved" value={`${solved.filter(Boolean).length} of ${solved.length}`} /> : null}
        <Kpi label="Segments" value={figure(step.segments)} />
        <Kpi label="Moved" value={metrics?.kl_moved != null ? metrics.kl_moved.toFixed(4) : "–"} />
        <Kpi label="Took" value={metrics ? span(metrics.update_seconds ?? metrics.seconds) : step.state === "stepping" && step.decided ? <Ago at={step.decided} /> : "–"} />
      </Kpis>
      <SectionTitle title="Groups" />
      <div className="tiles">
        {[...step.groups, ...step.skipped].map(each => {
          const group = groups.get(each), line = group?.line, skipped = step.skipped.includes(each);
          return (
            <Tile key={each} to={groupPlace(run.run, each)} className={`rail ${skipped ? "" : line?.rewards.length ? "good" : "bad"}`}>
              <header><b>#{each}</b><span className="what">{group?.task ?? ""} · {group?.title ?? ""}</span></header>
              <div className="big">{line?.rewards.length ? figure(mean(line.rewards)) : <span className="faint">–</span>}</div>
              <div className="facts">
                {line ? <Dots line={line} /> : null}
                {line?.rewards.length ? <span>{line.rewards.map(figure).join(" ")}</span> : null}
                {skipped ? <span>{line?.skipped ?? "nothing to train on"}</span> : line ? <span><b>{line.segments ?? "–"}</b> segments</span> : null}
              </div>
            </Tile>
          );
        })}
      </div>
      <Card title="The update" note={checkpoint ? `${checkpoint.short}, from ${checkpoint.parents.length ? checkpoint.parents.map(known.short).join(" + ") : known.base(checkpoint.base)}` : step.state === "failed" ? "the step failed" : "being taken"}>
        {metrics ? <Pairs entries={updatePairs(metrics)} /> : step.error ? <p className="error-text">{step.error}</p> : <p className="muted">Being taken.</p>}
      </Card>
    </>
  );
}

export function GroupView({ run: name, number }: { run: string; number: number }) {
  const { data: system } = useSystem();
  const { data: group, error } = useGroup(name, number);
  const known = useKnown();
  if (error instanceof NotFound) return <Empty>There is no group {number}.</Empty>;
  if (!group || !system) return <Empty>Reading the group…</Empty>;
  const { outcome, checkpoint, result } = group;
  const rewards = result?.rewards ?? group.episodes.filter(each => each.outcome === "completed").map(each => each.reward ?? 0);
  const run = system.runs.find(each => each.run === group.run), step = run && stepOf(run, group.number);
  const kept = step && !step.groups.includes(group.number);
  const metrics = checkpoint?.metrics ?? outcome?.update;
  const runners = [...new Set(group.playing.map(claim => claim.runner))];
  return (
    <>
      <Head title={`Group #${group.number}`}>
        <Specs>
          <Spec label="task">{group.task}</Spec>
          <Spec label="row">{group.title ?? ""}</Spec>
          <Spec label="stage" kind={stateKind(group.stage)}>{group.stage}</Spec>
          {outcome ? <Spec label="outcome" kind={outcome.update ? "good" : outcome.error ? "bad" : ""}>{outcomeOf(outcome).text.split(" · ")[0]}</Spec> : null}
        </Specs>
      </Head>
      {run ? <Elsewhere run={{ ...run, episodes_at: group.episodes_at }} /> : null}
      <Kpis>
        <Kpi label="Mean reward" value={rewards.length ? figure(mean(rewards)) : "–"} note={rewards.map(figure).join("  ")} />
        {result ? reported(result.solved) ? <Kpi label="Solved" value={`${result.solved.filter(Boolean).length} of ${result.solved.length}`} note={result.failed ? `${result.failed} failed` : ""} />
          : result.failed ? <Kpi label="Failed" value={String(result.failed)} note={`of ${result.failed + result.rewards.length}`} /> : null
          : reported(group.episodes.map(each => each.solved)) ? <Kpi label="Solved" value={`${group.episodes.filter(each => each.solved).length} of ${group.count ?? "?"}`} /> : null}
        <Kpi label="Played for" value={result ? span(result.rollout_seconds) : group.decided ? <Ago at={group.decided} /> : "–"} note={group.decided ? `decided ${clock(group.decided)}` : ""} />
        <Kpi label="To train on" value={result ? `${result.segments}` : "–"} note={result ? `of ${result.segments_recorded} segments` : ""} />
        <Kpi label="Step" value={step ? <Link to={stepPlace(group.run, step.step)}>S{step.step} → {known.short(step.makes)}</Link> : "–"}
          note={step ? (kept ? "decided after it; nothing of it trained" : `over ${range(step.groups)}`) : run?.next.includes(group.number) ? "toward the next step" : ""} />
      </Kpis>
      {group.stage !== "done" ? <Card title="Stage"><Stages stage={group.stage} ended={group.ended} count={group.count} /></Card> : null}
      <SectionTitle title="Episodes" note={group.count ? `${group.count} asked for · ${runners.length ? `${runners.join(", ")} playing` : `${group.ended} ended`}` : ""} />
      {group.episodes.length || group.count ? (
        <div className="tiles">
          {asked(group).map(episode => isWaiting(episode) ? (
            <div key={`w${episode.episode}`} className="tile rail waiting">
              <header><b>Episode {episode.episode}</b><span className="what" /><Mark state="">not started</Mark></header>
              <div className="big"><span className="faint">–</span></div>
            </div>
          ) : <EpisodeTile key={episode.run_id} episode={episode} />)}
        </div>
      ) : <Empty>No episode has started.</Empty>}
      <div className="cols">
        <Card title="What was done" note={outcome ? outcomeOf(outcome).text : "nothing yet"}>
          {metrics ? (
            <Pairs entries={[
              ["checkpoint", checkpoint ? <CheckpointTag id={checkpoint.id} bare /> : outcome?.adapter ? <CheckpointTag id={outcome.adapter} bare /> : "–"],
              ["from", <CheckpointTag id={checkpoint?.parents[0] ?? group.step?.parent} />],
              ...updatePairs(metrics),
            ]} />
          ) : group.step ? (
            <Pairs entries={[
              ["makes", group.step.makes ? <CheckpointTag id={group.step.makes} bare /> : "–"], ["from", <CheckpointTag id={group.step.parent} />], ["segments", figure(group.step.segments)],
              ["decided", group.step.decided ? <><Ago at={group.step.decided} /> ago</> : "–"],
            ]} />
          ) : result?.skipped ? <p className="muted">{result.skipped}</p>
            : result ? <p className="muted">Waiting for a step.</p>
              : <p className="muted">Playing.</p>}
        </Card>
        <Card title="Start">
          <Pairs entries={Object.entries(group.parameters ?? {}).map(([key, value]) => [key, figure(value)])} />
        </Card>
      </div>
      {result?.failures?.length ? (
        <Card title="Why episodes failed" note={`${result.failures.length}`}>
          {result.failures.map((failure, index) => <p key={index} className="error-text">{failure}</p>)}
        </Card>
      ) : null}
    </>
  );
}

const EpisodeTile = memo(function EpisodeTile({ episode }: { episode: GroupEpisode }) {
  const info = (episode.info ?? {}) as Record<string, unknown>;
  const bySlot = slotRewards(episode.rewards);  // (while it plays, where its slots are not rewarded together)
  return (
    <Tile to={episodePlace(episode.run_id)} className={`rail ${episode.interrupted ? "" : stateKind(episode.state)}`}>
      <header>
        <b>Episode {episode.episode ?? "?"}</b><span className="what" />
        <Mark state={episode.interrupted ? "" : episode.state ?? "running"}>{episode.interrupted ? "interrupted" : episode.state ?? "running"}</Mark>
      </header>
      <div className="big">{episode.outcome || episode.reward != null ? figure(episode.reward) : <span className="faint">…</span>}</div>
      <div className="facts">
        {bySlot.map(([slot, value]) => <span key={slot}>{slot} <b>{figure(value)}</b></span>)}
        {info.turns != null ? <span><b>{String(info.turns)}</b> turns</span> : episode.samples != null ? <span><b>{episode.samples}</b> samples</span> : null}
        {info.duration != null ? <span>duration <b>{figure(info.duration)}</b></span> : null}
        {info.ended ? <span>ended by <b>{String(info.ended)}</b></span> : null}
        {episode.sampled ? <span><b>{tokens(episode.sampled)}</b> tokens</span> : null}
        {episode.solved ? <span className="moved">solved</span> : null}
      </div>
      {episode.detail ? <div className="error-text">{episode.detail.slice(0, 240)}</div> : null}
    </Tile>
  );
});
