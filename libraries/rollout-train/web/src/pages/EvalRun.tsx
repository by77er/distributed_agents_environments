// An eval: one suite played by one checkpoint (or the base model), nothing trained. Who played, the suite, who asked
// for it, its score, and how it did at each start.

import { Link } from "react-router-dom";
import { useEvals, useKnown, useSystem } from "../api/queries";
import { CheckpointTag } from "../components/checkpoints";
import { Played, startName } from "../components/evals";
import { Rename } from "../components/Rename";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { clock, figure, mean, percent, span } from "../lib/format";
import { nameOf } from "../lib/model";
import { runPlace, stepPlace, suitePlace } from "../lib/places";

export function EvalRun({ run }: { run: string }) {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const known = useKnown();
  if (!evals || !system) return <Empty>Reading the eval…</Empty>;
  const entry = system.runs.find(each => each.run === run);
  const listed = evals.evals.find(each => each.run === run);
  if (!entry && !listed) return <Empty>There is no eval {run}.</Empty>;
  const suite = evals.suites.find(each => each.suite === listed?.suite);
  const subject = suite?.subjects.find(each => each.subject === run);
  const played = Object.values(subject?.results ?? {}).flat();
  const said = played.filter(each => each.solved != null);
  const share = said.length ? said.filter(each => each.solved).length / said.length : null;
  const rewards = played.map(each => each.reward).filter(each => each != null);
  const ended = (entry != null && !["running", "idle"].includes(entry.state)) || listed?.done;
  const by = entry?.by, step = entry?.by_step;
  const checkpoint = listed?.checkpoint ?? null;
  const name = entry ? nameOf(entry) : listed?.name ?? run;
  return (
    <>
      <Head title={<>Eval <Rename id={run} name={name} /></>}>
        <Specs>
          <Spec label="state" kind={ended ? "" : "good"}>{listed?.done ? "done" : entry?.state ?? "–"}</Spec>
          <Spec label="played by">{checkpoint ? <CheckpointTag id={checkpoint} /> : <CheckpointTag id={null} base={subject?.model} />}</Spec>
          <Spec label="suite">{listed ? <Link to={suitePlace(listed.suite)}>{listed.suite}</Link> : "–"}</Spec>
          <Spec label="asked by">
            {by ? <><Link to={runPlace(by)}>{known.run(by)}</Link>{step != null ? <> · <Link to={stepPlace(by, step)}>S{step}</Link></> : null}</> : "by hand"}
          </Spec>
          <Spec label="started">{clock(listed?.started ?? entry?.started)}</Spec>
          <Spec label="id">{run}</Spec>
          {entry?.directory ? <Spec label="directory">{entry.directory}</Spec> : null}
        </Specs>
      </Head>
      <Kpis>
        <Kpi label="Solved" value={share == null ? "–" : percent(share)} note={share == null ? "" : `${said.filter(each => each.solved).length} of ${said.length}`} />
        <Kpi label="Mean reward" value={figure(mean(rewards))} />
        <Kpi label="Played" value={listed ? `${listed.played} of ${listed.expected}` : String(played.length)} note={subject?.episodes ? `${subject.episodes} per start` : ""} />
        <Kpi label="Took" value={listed?.done && listed.started && entry?.written ? span(entry.written - listed.started) : "–"} />
      </Kpis>
      <Card title="By start">
        {suite && subject ? (
          <Table
            heads={[["start"], ["episodes"], ["solved", "n"], ["mean reward", "n"]]}
            keys={suite.starts.map(start => start.start)}
            rows={suite.starts.map(start => {
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
        ) : <p className="muted">Nothing played yet.</p>}
      </Card>
    </>
  );
}
