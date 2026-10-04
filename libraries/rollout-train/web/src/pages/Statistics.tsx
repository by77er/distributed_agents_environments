// Statistics across every run: each run in its own color, read from the ledger (and from its runners' heartbeats for
// its engines); then the machines that run things, as their heartbeats say, the engines, the runners and the ledger.

import { memo, useMemo } from "react";
import { useKnown, useMachines, useStatistics, useSystem } from "../api/queries";
import type { Machine, StatisticsGroup, StatisticsRun, System } from "../api/types";
import { ColumnChart, LineChart, type Series, Sized, spansOf } from "../components/charts";
import { Card, Dots, Empty, Head, Kpi, Kpis, Legend, Meter, SectionTitle, Share, Table } from "../components/ui";
import { bytes, clock, figure, mean, percent, span, tick, tickSpan } from "../lib/format";
import { groupPlace } from "../lib/places";
import { useStored } from "../lib/stored";
import { SECTIONS, useHidden } from "../layout/Tree";
import { Ago, useRunColor } from "../layout/runs";

const DAY = 86400;
const sectionName = (key: string) => SECTIONS.find(([each]) => each === key)![1];
const solvedShare = (groups: StatisticsGroup[]): number | null => {
  const solved = groups.reduce((sum, group) => sum + group.solved.filter(Boolean).length, 0);
  const played = groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0);
  return played ? solved / played : null;
};
const meanReward = (groups: StatisticsGroup[]) => mean(groups.flatMap(group => group.rewards));
const STEP_FIGURES: [string, string, string][] = [
  ["kl_moved", "KL moved", "from the version before"], ["kl_floor", "KL floor", "the update's noise floor"], ["clip_fraction", "Clip fraction", "of tokens clipped"],
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
  const every = runs.flatMap(done);
  const title = (key: string, name: string, note: string) => <SectionTitle id={`section-${key}`} title={name} note={note} />;
  if (!runs.length) {
    return (
      <>
        <Head title="Statistics" />
        <Empty>{figures.runs.length ? "Every run is left out: choose some in the sidebar." : "The ledger has no run yet."}</Empty>
        <MachineSection system={system} />
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
      <Head title="Statistics" sub="Each run in its own color, read from the ledger (and from its runners' heartbeats for its engines). Leave runs out in the sidebar." />
      <Kpis>
        <Kpi label="Runs" value={String(runs.length)} note={`${runs.filter(run => life.get(run.run)?.state === "running").length} running`} />
        <Kpi label="Groups done" value={every.length.toLocaleString()} note={`${runs.reduce((sum, run) => sum + run.groups.filter(group => group.time == null).length, 0)} in flight`} />
        <Kpi label="Episodes" value={episodes.toLocaleString()} note={`${failed} failed`} />
        <Kpi label="Solved" value={every.length ? percent(solvedShare(every) ?? 0) : "–"} note="of the episodes played" />
        <Kpi label="Steps" value={String(steps.filter(step => step.state === "committed").length)} note={`${steps.filter(step => step.state === "failed").length} failed`} />
        <Kpi label="Last day" value={`${lastDay.length} groups`} note={`${lastDay.reduce((sum, group) => sum + group.rewards.length + group.failed, 0)} episodes`} />
      </Kpis>

      {title("outcomes", sectionName("outcomes"), `groups in the order their results were written; the line the mean of the last ${stretch}, a dot each group`)}
      <div className="segmented">
        {[4, 8, 16, 32].map(size => <button type="button" key={size} className={`seg${size === stretch ? " current" : ""}`} onClick={() => setStretch(size)}>{size} groups</button>)}
      </div>
      <Halves>
        <Card title="Solve rate" note="the share of a group's episodes that solved its row">
          <Sized>{width => <LineChart series={solveSeries} width={width} label="solve rate over groups" y={{ min: 0, max: 1 }} yTick={percent} format={percent} xFormat={value => `group ${value}`} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
        <Card title="Mean reward" note="of the episodes that completed; each row's rewards are its own">
          <Sized>{width => <LineChart series={rewardSeries} width={width} label="mean reward over groups" format={figure} xFormat={value => `group ${value}`} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
      </Halves>

      {title("rows", sectionName("rows"), "every row the runs drawn have played")}
      <Card title="Rows" note={`${byRow.length} rows; the last rewards are of the row's newest group`}>
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
        <Card title="By the names a start gives" note="groups whose start lists names, by how many">
          <Table heads={[["names", "n"], ["groups", "n"], ["episodes", "n"], ["solved"], ["mean reward", "n"]]} keys={sizes}
            rows={sizes.map(size => {
              const groups = named.filter(group => group.names === size);
              return [size, groups.length, groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), <Share value={solvedShare(groups)} kind="good" />, figure(meanReward(groups))];
            })} />
        </Card>
      ) : null}

      {title("steps", "Steps", "the trainer's statistics, a point for each step")}
      {present.length ? (
        <div className="multiples">
          {present.map(([key, name, note]) => {
            const seconds = key.endsWith("seconds");
            return (
              <section key={key} className="card small-card">
                <header><h2>{name}</h2><span>{note}</span></header>
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
      ) : <Empty>No step has made a version yet.</Empty>}

      {title("pace", "Pace", "counted when each group's result was written")}
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
        <Card title="What was done with each group" note="every group decided, by run">
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

      {title("queue", sectionName("queue"), "how many groups were being played, and how many waited for a step, over time")}
      <Halves>
        <Card title="Groups in flight" note="decided, their result not written">
          <Sized>{width => <LineChart series={runs.map(run => ({ ...colored(run), points: run.flight.map(point => [point[0], point[1]]), until: until(run) }))} width={width} time stepped label="groups in flight" format={String} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
        <Card title="Groups waiting for a step" note="recorded with something to train on, no step begun over them">
          <Sized>{width => <LineChart series={runs.map(run => ({ ...colored(run), points: run.flight.map(point => [point[0], point[2]]), until: until(run) }))} width={width} time stepped label="groups waiting for a step" format={String} dots={false} />}</Sized>
          <Legendary series={legend} />
        </Card>
      </Halves>

      {title("inference", "Inference", "each run's engines, as its feed has them: a measurement a minute while they are busy")}
      {measured.length ? (
        <Halves>
          <Card title="Tokens a second" note="generated, across requests">
            <Sized>{width => <LineChart series={channelSeries(1)} width={width} time label="tokens a second" format={value => `${figure(value)} tok/s`} dots={false} gap={quiet} />}</Sized>
            <Legendary series={channelSeries(1)} />
          </Card>
          <Card title="Requests at once" note="on average over each minute">
            <Sized>{width => <LineChart series={channelSeries(2)} width={width} time label="requests at once" format={figure} dots={false} gap={quiet} />}</Sized>
            <Legendary series={channelSeries(2)} />
          </Card>
        </Halves>
      ) : <Empty>No runner of these runs has measured its engines yet (a runner says what its channels served in each heartbeat).</Empty>}

      <MachineSection system={system} />
    </>
  );
}

const LastAgo = ({ at }: { at: number }) => <><Ago at={at} /> ago</>;

const gib = (value: number) => value / 2 ** 30;

/** Every runner's and launcher's machine, as its heartbeats say (memory, accelerators and disk, now and over its
 * recent beats; its engines' processes and channels), the engines, the runners in the ledger, the fences and tables. */
const MachineSection = memo(function MachineSection({ system }: { system: System }) {
  const { data: found } = useMachines();
  const known = useKnown();
  const machines = found?.machines ?? [];
  return (
    <>
      <SectionTitle id="section-machines" title="Machines" note="each runner's and launcher's machine, as its heartbeats say; the engines, the runners in the ledger, and the ledger" />
      {found === undefined ? <Empty>Reading the heartbeats…</Empty>
        : machines.length ? <div className="cols">{machines.map(machine => <MachineCard key={machine.runner} machine={machine} />)}</div>
        : <Empty>No runner or launcher has beaten yet: a run's runner beats every 15 s while it runs.</Empty>}
      <div className="cols">
        {system.channels.map(channel => {
          const last = channel.throughput.at(-1);
          return (
            <Card key={`${channel.run}${channel.channel}`} title={`Channel ${channel.channel}`} note={`${known.run(channel.run)} · ${channel.adapter ? `serving ${known.short(channel.adapter)} (${known.origin(channel.adapter)}) since ${clock(channel.published)}` : "serving the base model"}`}>
              {last ? (
                <Kpis style={{ marginBottom: 12 }}>
                  <Kpi label="tokens a second" value={figure(last.tokens_per_second)} />
                  <Kpi label="requests at once" value={figure(last.mean_concurrency)} />
                  <Kpi label="each" value={`${figure(last.tokens_per_second_per_stream)} tok/s`} />
                </Kpis>
              ) : <p className="muted">No request has been measured yet.</p>}
            </Card>
          );
        })}
        {system.runners.map(runner => (
          <Card key={runner.runner} title={`Runner ${runner.runner}`} note={runner.last ? <>last claimed <Ago at={runner.last} /> ago</> : "has claimed nothing"}>
            <Kpis style={{ marginBottom: 12 }}>
              <Kpi label="playing" value={String(runner.playing.length)} />
              <Kpi label="claims" value={String(runner.claims)} />
              <Kpi label="fence" value={String(runner.fence ?? "–")} />
            </Kpis>
            {runner.playing.length ? (
              <Table heads={[["run"], ["group", "n"], ["episode", "n"], ["attempt", "n"], ["since", "n"]]}
                rows={runner.playing.map(claim => [known.run(claim.run), `#${claim.group}`, `E${claim.episode}`, String(claim.attempt), <Ago at={claim.at} />])}
                to={runner.playing.map(claim => (claim.group != null ? groupPlace(claim.run, claim.group) : null))} />
            ) : null}
          </Card>
        ))}
        <Card title="Ledger" note={`${system.ledger_at} · ${bytes(system.kept.versions)} of versions and ${bytes(system.kept.episodes)} of episodes kept`}>
          <Table heads={[["scope"], ["fence", "n"]]} rows={Object.entries(system.ledger.fences)} keys={Object.keys(system.ledger.fences)} />
          <div style={{ height: 14 }} />
          <Table heads={[["table"], ["records", "n"]]} rows={Object.entries(system.ledger.tables)} keys={Object.keys(system.ledger.tables)} />
        </Card>
      </div>
    </>
  );
});

/** One machine, as a runner's or a launcher's heartbeats say: alive or not, its meters now, and its memory and
 * accelerators over its recent beats; its engines' processes and what its channels serve. */
const MachineCard = memo(function MachineCard({ machine }: { machine: Machine }) {
  const known = useKnown();
  const latest = machine.machine;
  const history = machine.history.filter(each => each.machine).map(each => ({ ...each.machine!, at: each.at }));
  const accelerators = (latest?.accelerators ?? []).map((each, place) => ({ name: each.name, color: `var(--series-${place + 1})`, place }));
  const launcher = machine.kind === "launcher";
  const what = launcher
    ? `launcher · ${machine.profiles?.length ?? 0} profile${machine.profiles?.length === 1 ? "" : "s"} · ${machine.playing ?? 0} of ${machine.at_once ?? 1} playing`
    : `runner${machine.run ? ` of ${known.run(machine.run)}` : ""} · ${machine.playing ?? 0} of ${machine.places ?? "?"} places`;
  return (
    <Card className={`machine${machine.alive ? "" : " stale"}`}
      title={<span className="machine-title"><span className={`dot ${machine.alive ? "alive" : "gone"}`} />{machine.host ?? machine.runner}</span>}
      note={<>{what} · {machine.alive ? "beat" : "gone: last beat"} <LastAgo at={machine.at} /></>}>
      <div className="small faint mono machine-name" title={machine.runner}>{machine.runner}</div>
      {latest ? (
        <>
          {latest.memory.total ? <Meter name="memory" used={latest.memory.total - (latest.memory.available ?? 0)} total={latest.memory.total} says={`${bytes(latest.memory.available)} available of ${bytes(latest.memory.total)}`} /> : null}
          {latest.accelerators.map(each => <Meter key={each.name} name={each.name} used={each.used} total={each.total} says={`${bytes(each.used)} of ${bytes(each.total)} · ${Math.round(100 * each.busy)}% busy`} />)}
          {latest.disk ? <Meter name="disk" used={latest.disk.total - latest.disk.free} total={latest.disk.total} says={`${bytes(latest.disk.free)} free`} /> : null}
        </>
      ) : <p className="muted small">Its beats measure nothing.</p>}
      {history.length > 1 ? (
        <Sized>{width => (
          <LineChart width={width} height={150} time label="memory in use" dots={false} yTick={value => `${tick(value)} GiB`} format={value => `${value.toFixed(1)} GiB`}
            series={[
              ...(history.some(each => each.memory.total) ? [{ name: "memory", color: "var(--quiet)", points: history.filter(each => each.memory.total).map(each => [each.at, gib(each.memory.total! - (each.memory.available ?? 0))] as [number, number]) }] : []),
              ...accelerators.map(each => ({ ...each, points: history.filter(one => one.accelerators[each.place]).map(one => [one.at, gib(one.accelerators[each.place].used)] as [number, number]) })),
            ]} />
        )}</Sized>
      ) : null}
      {history.length > 1 && accelerators.length ? (
        <Sized>{width => (
          <LineChart width={width} height={120} time label="accelerators busy" dots={false} y={{ min: 0, max: 1 }} yTick={percent} format={percent}
            series={accelerators.map(each => ({ ...each, points: history.filter(one => one.accelerators[each.place]).map(one => [one.at, one.accelerators[each.place].busy] as [number, number]) }))} />
        )}</Sized>
      ) : null}
      {history.length > 1 ? <Legendary series={[...(history.some(each => each.memory.total) ? [{ name: "memory", color: "var(--quiet)" }] : []), ...accelerators]} /> : null}
      {history.length > 1 ? <p className="small muted" style={{ margin: 0 }}>Its newest {history.length} beats cover {span((latest?.at ?? machine.at) - history[0].at)}.</p> : null}
      {machine.channels?.length ? (
        <Table heads={[["channel"], ["serving"], ["tok/s", "n"], ["at once", "n"]]}
          rows={machine.channels.map(channel => [channel.channel, channel.adapter ? `${known.short(channel.adapter)} · ${known.origin(channel.adapter)}` : "base model", figure(channel.tokens_per_second), figure(channel.mean_concurrency)])}
          keys={machine.channels.map(channel => channel.channel)} />
      ) : null}
      {machine.processes?.started.length ? (
        <Table heads={[["process"], ["pid", "n"], ["state"]]}
          rows={machine.processes.started.map(each => [each.name, String(each.pid), { text: each.alive ? "running" : "gone", kind: each.alive ? "t-good" : "t-bad" }])}
          keys={machine.processes.started.map(each => each.pid)} />
      ) : null}
      {launcher && machine.profiles?.length ? (
        <div className="facts">{machine.profiles.map(profile => <span key={profile.profile} className="chip">{profile.profile}</span>)}{(machine.catalogs ?? []).map(each => <span key={each} className="chip mono">{each}</span>)}</div>
      ) : null}
    </Card>
  );
});
