// An eval: one version of a suite played by one checkpoint (or the base model), nothing trained. Who played, the suite
// and its version, who asked for it, its score (at each environment, for a suite of several), and how it did at each
// start of that version.

import { Link } from "react-router-dom";
import { useEvals, useKnown, useSystem } from "../api/queries";
import type { SuiteStart } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { Played, startName, type Subject } from "../components/evals";
import { RunControls } from "../components/control";
import { Rename } from "../components/Rename";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { clock, figure, mean, percent, span } from "../lib/format";
import { readable } from "../lib/environments";
import { nameOf, runKind } from "../lib/model";
import { stateOf } from "../layout/runs";
import { runPlace, stepPlace, suitePlace } from "../lib/places";
import { entryStarts, playedVersion, versionsOf, versionTag } from "../lib/suites";

export function EvalRun({ run }: { run: string }) {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const known = useKnown();
  if (!evals || !system) return <Empty>Reading the eval…</Empty>;
  const entry = system.runs.find(each => each.run === run);
  if (entry?.part_of) return <EvalRun run={entry.part_of} />;  // (a run that plays one environment of an eval: the eval)
  const listed = evals.evals.find(each => each.run === run);
  if (!entry && !listed) return <Empty>There is no eval {run}.</Empty>;
  const suite = evals.suites.find(each => each.suite === listed?.suite);
  const subject = suite?.subjects.find(each => each.subject === run);
  const version = suite ? versionsOf(suite).find(each => each.id === (subject ? playedVersion(subject, suite.suite) : listed?.version)) : undefined;
  const starts = version?.starts ?? suite?.starts ?? [];
  const played = Object.values(subject?.results ?? {}).flat();
  const said = played.filter(each => each.solved != null);
  const share = said.length ? said.filter(each => each.solved).length / said.length : null;
  const rewards = played.map(each => each.reward).filter(each => each != null);
  const ended = (entry != null && !["running", "idle", "paused"].includes(entry.state)) || listed?.done;
  const by = entry?.by, step = entry?.by_step;
  const checkpoint = listed?.checkpoint ?? null;
  const name = entry ? nameOf(entry) : listed?.name ?? run;
  const several = (version?.entries.length ?? 0) > 1;
  return (
    <>
      <Head title={<span className="head-with-action"><span>Eval <Rename id={run} name={name} /></span>{entry && !listed?.done ? <RunControls run={entry} /> : null}</span>}>
        <Specs>
          <Spec label="state" kind={ended ? "" : entry ? runKind(entry.state) : "good"}>{listed?.done ? "done" : entry ? stateOf(entry) : "–"}</Spec>
          <Spec label="played by">{checkpoint ? <CheckpointTag id={checkpoint} /> : <CheckpointTag id={null} base={subject?.model} />}</Spec>
          <Spec label="suite">{listed ? <><Link to={suitePlace(listed.suite)}>{listed.suite}</Link> {versionTag(version?.id ?? listed.version)}</> : "–"}</Spec>
          <Spec label="asked by">
            {by ? <><Link to={runPlace(by)}>{known.run(by)}</Link>{step != null ? <> · <Link to={stepPlace(by, step)}>S{step}</Link></> : null}</> : "by hand"}
          </Spec>
          <Spec label="started">{clock(listed?.started ?? entry?.started)}</Spec>
          <Spec label="id">{run}</Spec>
          {entry?.directory ? <Spec label="directory">{entry.directory}</Spec> : null}
        </Specs>
      </Head>
      <Kpis>
        {several && version ? version.entries.map((each, place) => {
          const score = subject?.entries?.[place];
          const solved = score && score.solved != null && score.played ? score.solved / score.played : null;
          return <Kpi key={each.environment ?? place} label={readable(each.environment)} value={solved == null ? figure(score?.reward) : percent(solved)}
            note={solved == null ? "mean reward" : `mean reward ${figure(score?.reward)}`} />;
        }) : (
          <>
            <Kpi label="Solved" value={share == null ? "–" : percent(share)} note={share == null ? "" : `${said.filter(each => each.solved).length} of ${said.length}`} />
            <Kpi label="Mean reward" value={figure(mean(rewards))} />
          </>
        )}
        <Kpi label="Played" value={listed ? `${listed.played} of ${listed.expected}` : String(played.length)} note={subject?.episodes ? `${subject.episodes} per start` : ""} />
        <Kpi label="Took" value={listed?.done && listed.started && entry?.written ? span(entry.written - listed.started) : "–"} />
      </Kpis>
      {several && version && suite && subject ? version.entries.map(each => (
        <Card key={each.environment ?? each.offset} title={readable(each.environment)} note={each.environment ?? ""}>
          <ByStart subject={subject} starts={entryStarts(version, each)} />
        </Card>
      )) : (
        <Card title="By start">
          {suite && subject ? <ByStart subject={subject} starts={starts} /> : <p className="muted">Nothing played yet.</p>}
        </Card>
      )}
    </>
  );
}

/** How a subject did at each of some starts: its episodes, how many solved and their mean reward. */
function ByStart({ subject, starts }: { subject: Subject; starts: SuiteStart[] }) {
  return (
    <Table
      heads={[["start"], ["episodes"], ["solved", "n"], ["mean reward", "n"]]}
      keys={starts.map(start => start.start)}
      rows={starts.map(start => {
        const here = subject.results[start.start] ?? [];
        const solved = here.filter(each => each.solved != null);
        return [
          <span title={start.title ?? ""}>{startName(start)}</span>,
          { text: <Played subject={subject} start={start} />, kind: "cell-result" },
          solved.length ? `${solved.filter(each => each.solved).length}/${solved.length}` : "–",
          figure(mean(here.map(each => each.reward))),
        ];
      })}
    />
  );
}
