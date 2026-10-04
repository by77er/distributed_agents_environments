// Statistics across every run: each run in its own color, read from the ledger (and from its runners' heartbeats for
// its engines); then what each channel serves now, and the ledger.

import { memo, useMemo } from "react";
import { useKnown, useStatistics, useSystem } from "../api/queries";
import type { StatisticsGroup, StatisticsRun, System } from "../api/types";
import { ColumnChart, LineChart, type Series, Sized, spansOf } from "../components/charts";
import { Card, Dots, Empty, Head, Kpi, Kpis, Legend, SectionTitle, Share, Table } from "../components/ui";
import { bytes, clock, figure, mean, percent, span, tickSpan } from "../lib/format";
import { reported } from "../lib/model";
import { groupPlace } from "../lib/places";
import { useStored } from "../lib/stored";
import { SECTIONS, useHidden } from "../layout/Tree";
import { Ago, useRunColor } from "../layout/runs";

const DAY = 86400;
const LastAgo = ({ at }: { at: number }) => <><Ago at={at} /> ago</>;
const sectionName = (key: string) => SECTIONS.find(([each]) => each === key)![1];
/** The share of the groups' episodes that solved their rows (none where none of them says). */
const solvedShare = (groups: StatisticsGroup[]): number | null => {
  if (!reported(groups.flatMap(group => group.solved))) return null;
  const solved = groups.reduce((sum, group) => sum + group.solved.filter(Boolean).length, 0);
  const played = groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0);
  return played ? solved / played : null;
};
const meanReward = (groups: StatisticsGroup[]) => mean(groups.flatMap(group => group.rewards));
const STEP_FIGURES: [string, string, string][] = [
  ["kl_moved", "KL moved", "from the checkpoint before"], ["kl_floor", "KL floor", "the update's noise floor"], ["clip_fraction", "Clip fraction", "of tokens clipped"],
  ["mean_mismatch", "Mean mismatch", "sampler against trainer"], ["mean_weight", "Mean weight", "the off-policy correction"], ["truncated_fraction", "Truncated fraction", "of weights truncated"],
  ["loss", "Loss", "the objective"], ["seconds", "Step time", "the update, start to end"], ["start_seconds", "Start time", "before the first optimizer step"],
];
const done = (run: StatisticsRun) => run.groups.filter(group => group.time != null).sort((a, b) => a.time! - b.time!);

/** Two charts side by side where there is room, one above the other where there is not. */
const Halves = ({ children }: { children: React.ReactNode }) => <div className="cols">{children}</div>;

const Legendary = ({ series }: { series: { name: string; color: string }[] }) => (series.length > 1 ? <Legend items={series} /> : null);

export function Statistics() {
  const { data: figures } = useStatistics();
  const { data: system } = useSystem();
  const known = useKnown();
  const [hidden] = useHidden();
  const colorOf = useRunColor();
  const [stretch, setStretch] = useStored<number>("monitor.window", 8);
  const runs = useMemo(() => (figures?.runs ?? []).filter(run => !hidden.includes(run.run)).sort((a, b) => a.run.localeCompare(b.run)), [figures, hidden]);
  if (!figures || !system) return <Empty>Reading the statistics…</Empty>;
  const runName = (id: string) => figures.names?.runs[id] ?? known.run(id);  // (a run by its name, as the figures were read)
  const colored = (run: StatisticsRun) => ({ name: runName(run.run), color: colorOf(run.run) });
  const life = new Map(system.runs.map(run => [run.run, run]));
  const every = runs.flatMap(done), said = reported(every.flatMap(group => group.solved));
  const title = (key: string, name: string) => <SectionTitle id={`section-${key}`} title={name} />;
  if (!runs.length) {
    return (
      <>
        <Head title="Statistics" />
        <Empty>{figures.runs.length ? "Every run is left out." : "No run yet."}</Empty>
        {title("ledger", sectionName("ledger"))}
        <LedgerCard system={system} />
      </>
    );
  }

  // Totals
  const episodes = every.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), failed = every.reduce((sum, group) => sum + group.failed, 0);
  const lastDay = every.filter(group => group.time! > figures.now - DAY);
  const steps = runs.flatMap(run => run.steps);

  // Outcomes over groups: a rolling solve rate and mean reward, a line for each run.
  const rolling = (groups: StatisticsGroup[], value: (span: StatisticsGroup[]) => number | null): number[][] =>
    groups.map((_, place) => [place + 1, value(groups.slice(Math.max(0, place + 1 - stretch), place + 1)) ?? NaN]);
  const solveSeries: Series[] = runs.map(run => ({ ...colored(run), points: rolling(done(run), solvedShare), scatter: done(run).map((group, place) => [place + 1, solvedShare([group]) ?? NaN]) }));
  const rewardSeries: Series[] = runs.map(run => ({ ...colored(run), points: rolling(done(run), meanReward), scatter: done(run).map((group, place) => [place + 1, mean(group.rewards) ?? NaN]) }));

  // Results by row
  const rows = new Map<string, { task: string; title: string | null; runs: Set<string>; groups: (StatisticsGroup & { run: string })[]; best: number | null }>();
  for (const run of runs) for (const group of done(run)) {
    const key = String(group.task);
    const row = rows.get(key) ?? { task: key, title: group.title, runs: new Set<string>(), groups: [], best: null };
    row.runs.add(run.run);
    row.groups.push({ ...group, run: run.run });
    for (const reward of group.rewards) row.best = row.best == null ? reward : Math.max(row.best, reward);
    rows.set(key, row);
  }
  const byRow = [...rows.values()].sort((a, b) => a.task.localeCompare(b.task, undefined, { numeric: true }));
  const named = every.filter(group => group.names != null);
  const sizes = [...new Set(named.map(group => group.names!))].sort((a, b) => a - b);

  // Steps: each statistic of the update over the steps, a line for each run.
  const valueOf = (metrics: Record<string, number | undefined>, key: string) => (key === "seconds" ? metrics.update_seconds ?? metrics.seconds : metrics[key]);
  const present = STEP_FIGURES.filter(([key]) => steps.some(step => valueOf(step.metrics, key) != null));

  // Pace: episodes and groups an hour, and what was done with each group.
  const from = Math.min(...every.map(group => group.time!)), to = Math.max(...every.map(group => group.time!)) + 1;
  const paced = (count: (group: StatisticsGroup) => number) => spansOf(runs.map(run => done(run).flatMap(group => Array.from({ length: count(group) }, () => group.time!))), from, to);
  const episodeSpans = every.length ? paced(group => group.rewards.length + group.failed) : { spans: [], step: 3600 };
  const groupSpans = every.length ? paced(() => 1) : { spans: [], step: 3600 };
  const perHour = (spans: typeof episodeSpans.spans, step: number) => spans.map(each => ({ ...each, values: each.values.map(value => (value * 3600) / step) }));
  const kinds: [string, string, string][] = [["trained", "trained on", "var(--accent)"], ["stepping", "in a step being taken", "var(--violet)"], ["waiting", "waiting for a step", "var(--warm)"],
    ["skipped", "nothing to train on", "var(--line-strong)"], ["failed", "no episode, or its step failed", "var(--bad)"], ["flight", "in flight", "var(--faint)"]];
  const kindOf = (group: StatisticsGroup) => (group.time == null ? "flight" : group.trained === "committed" ? "trained" : group.trained === "failed" || !group.rewards.length ? "failed"
    : group.trained === "stepping" ? "stepping" : group.segments ? "waiting" : "skipped");
  const reasons = new Map<string, number>();
  for (const run of runs) for (const group of done(run)) if (group.skipped) reasons.set(group.skipped, (reasons.get(group.skipped) ?? 0) + 1);

  // In flight and waiting, over time; inference, from each run's runners' heartbeats.
  const until = (run: StatisticsRun) => (life.get(run.run)?.state === "running" ? figures.now : run.wrote);
  const measured = runs.filter(run => run.inference.length);
  const channelSeries = (place: number): Series[] => measured.flatMap(run => run.inference.map(channel => ({
    name: run.inference.length > 1 ? `${runName(run.run)} · ${channel.channel}` : runName(run.run), color: colorOf(run.run),
    points: channel.points.map(point => [point[0], point[place]]),
  })));
  const quiet = 10 * 60 * Math.max(1, ...measured.flatMap(run => run.inference.map(channel => channel.every)));  // (engines idle that long: no line across)
  const legend = runs.map(colored);

  return (
    <>
      <Head title="Statistics" />
      <Kpis>
        <Kpi label="Runs" value={String(runs.length)} note={`${runs.filter(run => life.get(run.run)?.state === "running").length} running`} />
        <Kpi label="Groups done" value={every.length.toLocaleString()} note={`${runs.reduce((sum, run) => sum + run.groups.filter(group => group.time == null).length, 0)} in flight`} />
        <Kpi label="Episodes" value={episodes.toLocaleString()} note={`${failed} failed`} />
        {said ? <Kpi label="Solved" value={percent(solvedShare(every) ?? 0)} /> : null}
        <Kpi label="Steps" value={String(steps.filter(step => step.state === "committed").length)} note={`${steps.filter(step => step.state === "failed").length} failed`} />
        <Kpi label="Last day" value={`${lastDay.length} groups`} note={`${lastDay.reduce((sum, group) => sum + group.rewards.length + group.failed, 0)} episodes`} />
      </Kpis>

      {title("outcomes", sectionName("outcomes"))}
      <div className="segmented">
        {[4, 8, 16, 32].map(size => <button type="button" key={size} className={`seg${size === stretch ? " current" : ""}`} onClick={() => setStretch(size)}>{size} groups</button>)}
      </div>
      <Halves>
        {said ? (
          <Card title="Solve rate">
            <Sized>{width => <LineChart series={solveSeries} width={width} label="solve rate over groups" y={{ min: 0, max: 1 }} yTick={percent} format={percent} xFormat={value => `group ${value}`} dots={false} />}</Sized>
            <Legendary series={legend} />
          </Card>
        ) : null}
        <Card title="Mean reward">
          <Sized>{width => <LineChart series={rewardSeries} width={width} label="mean reward over groups" format={figure} xFormat={value => `group ${value}`} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
      </Halves>

      {title("rows", sectionName("rows"))}
      <Card title="Rows" note={`${byRow.length}`}>
        <Table
          heads={[["row"], ["title"], ...(runs.length > 1 ? [["runs", "n"] as [string, string]] : []), ["groups", "n"], ["episodes", "n"], ["solved"], ["mean", "n"], ["best", "n"], ["last rewards"], ["last", "n"]]}
          keys={byRow.map(row => row.task)}
          rows={byRow.map(row => {
            const last = row.groups.at(-1)!;
            return [
              { text: row.task ?? "–", kind: "key" }, <span className="muted">{row.title ?? ""}</span>, ...(runs.length > 1 ? [row.runs.size] : []), row.groups.length,
              row.groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), <Share value={solvedShare(row.groups)} kind="good" />,
              figure(meanReward(row.groups)), figure(row.best),
              <span><Dots line={{ rewards: last.rewards, solved: last.solved, failed: last.failed } as never} /> <span className="faint small">{last.rewards.map(figure).join(" ")}</span></span>,
              <LastAgo at={last.time!} />,
            ];
          })}
          to={byRow.map(row => { const last = row.groups.at(-1)!; return groupPlace(last.run, last.group); })}
        />
      </Card>
      {named.length ? (
        <Card title="By the names a start gives">
          <Table heads={[["names", "n"], ["groups", "n"], ["episodes", "n"], ["solved"], ["mean reward", "n"]]} keys={sizes}
            rows={sizes.map(size => {
              const groups = named.filter(group => group.names === size);
              return [size, groups.length, groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), <Share value={solvedShare(groups)} kind="good" />, figure(meanReward(groups))];
            })} />
        </Card>
      ) : null}

      {title("steps", "Steps")}
      {present.length ? (
        <div className="multiples">
          {present.map(([key, name, note]) => {
            const seconds = key.endsWith("seconds");
            return (
              <section key={key} className="card small-card">
                <header title={note}><h2>{name}</h2></header>
                <div className="body">
                  <Sized fallback={300}>{width => (
                    <LineChart series={runs.map(run => ({ ...colored(run), points: run.steps.filter(step => valueOf(step.metrics, key) != null).map(step => [step.step, valueOf(step.metrics, key)!]) }))}
                      width={width} height={150} label={`${name} over steps`} y={{ zero: seconds || ["clip_fraction", "truncated_fraction", "kl_moved"].includes(key) }}
                      yTick={seconds ? tickSpan : undefined} format={seconds ? (value => span(value)) : value => Number(value).toPrecision(4)} xFormat={value => `step ${value}`} />
                  )}</Sized>
                </div>
              </section>
            );
          })}
        </div>
      ) : <Empty>No step yet.</Empty>}

      {title("pace", "Pace")}
      <Halves>
        <Card title="Episodes an hour" note={`in spans of ${tickSpan(episodeSpans.step)}`}>
          <Sized>{width => <ColumnChart spans={perHour(episodeSpans.spans, episodeSpans.step)} series={legend} width={width} label="episodes an hour" format={value => `${figure(+value.toFixed(1))} an hour`} />}</Sized>
          <Legendary series={legend} />
        </Card>
        <Card title="Groups an hour" note={`in spans of ${tickSpan(groupSpans.step)}`}>
          <Sized>{width => <ColumnChart spans={perHour(groupSpans.spans, groupSpans.step)} series={legend} width={width} label="groups an hour" format={value => `${figure(+value.toFixed(2))} an hour`} />}</Sized>
          <Legendary series={legend} />
        </Card>
      </Halves>
      <Halves>
        <Card title="What was done with each group">
          <div className="fates">
            {runs.map(run => {
              const counts = new Map(kinds.map(([key]) => [key, 0]));
              for (const group of run.groups) counts.set(kindOf(group), counts.get(kindOf(group))! + 1);
              return (
                <div key={run.run} className="fate">
                  <span className="fate-name"><i className="swatch" style={{ background: colorOf(run.run) }} />{runName(run.run)}</span>
                  <div className="stack">{kinds.filter(([key]) => counts.get(key)).map(([key, name, color]) => <i key={key} style={{ flex: counts.get(key), background: color }} title={`${name}: ${counts.get(key)}`} />)}</div>
                  <span className="fate-count">{kinds.filter(([key]) => counts.get(key)).map(([key]) => `${counts.get(key)} ${key}`).join(" · ")}</span>
                </div>
              );
            })}
            <Legend items={kinds.map(([, name, color]) => ({ name, color }))} />
          </div>
        </Card>
        <Card title="Why groups gave nothing to train on" note={`${[...reasons.values()].reduce((sum, count) => sum + count, 0)} groups skipped`}>
          {reasons.size ? <Table heads={[["reason"], ["groups", "n"]]} rows={[...reasons].sort((a, b) => b[1] - a[1]).map(([reason, count]) => [reason, count])} keys={[...reasons.keys()]} /> : <Empty>None skipped.</Empty>}
        </Card>
      </Halves>

      {title("queue", sectionName("queue"))}
      <Halves>
        <Card title="Groups in flight">
          <Sized>{width => <LineChart series={runs.map(run => ({ ...colored(run), points: run.flight.map(point => [point[0], point[1]]), until: until(run) }))} width={width} time stepped label="groups in flight" format={String} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
        <Card title="Groups waiting for a step">
          <Sized>{width => <LineChart series={runs.map(run => ({ ...colored(run), points: run.flight.map(point => [point[0], point[2]]), until: until(run) }))} width={width} time stepped label="groups waiting for a step" format={String} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
      </Halves>

      {title("inference", "Inference")}
      {measured.length ? (
        <Halves>
          <Card title="Tokens a second">
            <Sized>{width => <LineChart series={channelSeries(1)} width={width} time label="tokens a second" format={value => `${figure(value)} tok/s`} dots={false} gap={quiet} />}</Sized>
            <Legendary series={channelSeries(1)} />
          </Card>
          <Card title="Requests at once">
            <Sized>{width => <LineChart series={channelSeries(2)} width={width} time label="requests at once" format={figure} dots={false} gap={quiet} />}</Sized>
            <Legendary series={channelSeries(2)} />
          </Card>
        </Halves>
      ) : <Empty>No measurement yet.</Empty>}

      <ChannelCards system={system} />

      {title("ledger", sectionName("ledger"))}
      <LedgerCard system={system} />
    </>
  );
}

/** What each run's channels serve now and how fast. */
const ChannelCards = memo(function ChannelCards({ system }: { system: System }) {
  const known = useKnown();
  if (!system.channels.length) return null;
  return (
    <div className="cols" style={{ marginTop: 22 }}>
      {system.channels.map(channel => {
        const last = channel.throughput.at(-1);
        return (
          <Card key={`${channel.run}${channel.channel}`} title={`Channel ${channel.channel}`} note={`${known.run(channel.run)} · ${channel.adapter ? `serving ${known.short(channel.adapter)} (${known.origin(channel.adapter)}) since ${clock(channel.published)}` : "serving the model its engines started with"}`}>
            {last ? (
              <Kpis>
                <Kpi label="tokens a second" value={figure(last.tokens_per_second)} />
                <Kpi label="requests at once" value={figure(last.mean_concurrency)} />
                <Kpi label="each" value={`${figure(last.tokens_per_second_per_stream)} tok/s`} />
              </Kpis>
            ) : <p className="muted">No request has been measured yet.</p>}
          </Card>
        );
      })}
    </div>
  );
});

/** The ledger: the runners that took their fences in it and the claims each made, the fences and the tables. */
const LedgerCard = memo(function LedgerCard({ system }: { system: System }) {
  return (
    <Card title="Ledger" note={`${system.ledger_at} · ${bytes(system.kept.checkpoints)} of checkpoints and ${bytes(system.kept.episodes)} of episodes kept`}>
      {system.runners.length ? (
        <>
          <Table heads={[["runner"], ["fence", "n"], ["claims", "n"], ["playing", "n"], ["last claimed", "n"]]} keys={system.runners.map(runner => runner.runner)}
            rows={system.runners.map(runner => [runner.runner, String(runner.fence ?? "–"), String(runner.claims), String(runner.playing.length), runner.last ? <><Ago at={runner.last} /> ago</> : "–"])} />
          <div style={{ height: 14 }} />
        </>
      ) : null}
      <Table heads={[["scope"], ["fence", "n"]]} rows={Object.entries(system.ledger.fences)} keys={Object.keys(system.ledger.fences)} />
      <div style={{ height: 14 }} />
      <Table heads={[["table"], ["records", "n"]]} rows={Object.entries(system.ledger.tables)} keys={Object.keys(system.ledger.tables)} />
    </Card>
  );
});
