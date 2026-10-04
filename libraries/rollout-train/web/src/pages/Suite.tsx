// A suite: its starts, every subject that played it start by start, two subjects compared, and the form that asks a
// launcher to play it with a checkpoint (an eval: nothing trained).

import { useState } from "react";
import { Link } from "react-router-dom";
import { useEvals, useKnown, useLaunches, useSystem } from "../api/queries";
import type { EvalSuite } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { anySolved, Played, shareOf, shareText, startName, startShare, type Subject, subjectText, SuiteMatrix } from "../components/evals";
import { LaunchList } from "../components/launches";
import { PlayForm } from "../components/play";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { Ago } from "../layout/runs";
import { clock, figure } from "../lib/format";
import { evalPlace, evalsPlace } from "../lib/places";
import { NoLauncher } from "./NewRun";

export function Suite({ name }: { name: string }) {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const { data: launched } = useLaunches();
  if (!evals || !system) return <Empty>Reading the suite…</Empty>;
  const suite = evals.suites.find(each => each.suite === name);
  if (!suite) return <Empty>There is no suite {name}. <Link to={evalsPlace} className="linkish">Every suite</Link></Empty>;
  const playing = evals.evals.filter(each => each.suite === name && !each.done);
  const launches = (launched?.launches ?? []).filter(each => each.asked.kind === "eval" && each.asked.suite === name);
  const said = anySolved(suite.subjects), score = (subject: Subject) => (said ? shareOf(subject) : subject.reward ?? null);
  const subjects = [...suite.subjects].sort((a, b) => (score(b) ?? -Infinity) - (score(a) ?? -Infinity));
  const rows = [...new Set(suite.starts.map(start => start.task))];
  const seeds = [...new Set(suite.starts.map(start => String(start.seed)))];
  return (
    <>
      <Head title={suite.suite}>
        <Specs>
          <Spec label="environment">{suite.environment ?? "–"}</Spec>
          <Spec label="rows">{rows.join(", ")}</Spec>
          <Spec label="seeds">{seeds.join(", ")}</Spec>
          {suite.made ? <Spec label="made">{clock(suite.made)}</Spec> : null}
        </Specs>
      </Head>
      <Kpis>
        <Kpi label="Starts" value={String(suite.starts.length)} />
        <Kpi label="Played by" value={String(subjects.length)} />
        <Kpi label="Best" value={subjects[0] ? (said ? shareText(shareOf(subjects[0])) : figure(subjects[0].reward)) : "–"} note={subjects[0] ? <SubjectLabel subject={subjects[0]} /> : ""} />
        <Kpi label="Playing" value={String(playing.length)} note={playing.map(each => each.name).join(", ")} />
      </Kpis>
      {launched ? (launched.launchers.length ? <PlayForm suite={suite} suites={evals.suites} launchers={launched.launchers} system={system} title="Run this suite" /> : <NoLauncher ledger={system.ledger_at} />) : null}
      {launches.length ? <LaunchList launches={launches} system={system} /> : null}
      {playing.length ? (
        <Card title="Playing now">
          <Table
            heads={[["eval"], ["played by"], ["played", "n"], ["started"]]}
            keys={playing.map(each => each.run)}
            rows={playing.map(each => [<b>{each.name}</b>, <CheckpointTag id={each.checkpoint} link={false} />, `${each.played}/${each.expected}`, each.started ? <><Ago at={each.started} /> ago</> : "–"])}
            to={playing.map(each => evalPlace(each.run))}
          />
        </Card>
      ) : null}
      <Card title="Start by start">
        {subjects.length ? <SuiteMatrix suite={suite} subjects={subjects} /> : <p className="muted">None played yet.</p>}
      </Card>
      {subjects.length > 1 && said ? <Compare suite={suite} subjects={subjects} /> : null}
    </>
  );
}

const SubjectLabel = ({ subject }: { subject: Subject }) => {
  const known = useKnown();
  return <>{subjectText(subject, known)}{subject.kind === "model" ? "" : ` · ${known.origin(subject.checkpoint)}`}</>;
};

/** Two subjects, start by start: where each solved more than the other. */
function Compare({ suite, subjects }: { suite: EvalSuite; subjects: Subject[] }) {
  const known = useKnown();
  const [first, setFirst] = useState(subjects[0].subject);
  const [second, setSecond] = useState(subjects[1].subject);
  const a = subjects.find(each => each.subject === first) ?? subjects[0];
  const b = subjects.find(each => each.subject === second) ?? subjects[1];
  const compared = suite.starts.map(start => ({ start, a: startShare(a, start), b: startShare(b, start) }));
  const both = compared.filter(each => each.a != null && each.b != null);
  const better = both.filter(each => each.a! > each.b!), worse = both.filter(each => each.a! < each.b!);
  const differ = [...better, ...worse];
  const option = (subject: Subject) => (
    <option key={subject.subject} value={subject.subject}>{subjectText(subject, known)}{subject.kind === "model" ? "" : ` · ${known.origin(subject.checkpoint)}`} · {shareText(shareOf(subject))}</option>
  );
  return (
    <Card title="Compared">
      <div className="field-row">
        <label className="field"><span>This</span><select value={a.subject} onChange={event => setFirst(event.target.value)}>{subjects.map(option)}</select></label>
        <label className="field"><span>against</span><select value={b.subject} onChange={event => setSecond(event.target.value)}>{subjects.map(option)}</select></label>
      </div>
      <p className="muted">
        At the {both.length} starts both played, <b>{subjectText(a, known)}</b> solved more at <b className="t-good">{better.length}</b>,
        less at <b className="t-bad">{worse.length}</b>, and as much at {both.length - differ.length}.
      </p>
      {differ.length ? (
        <Table
          heads={[["start"], [subjectText(a, known)], [subjectText(b, known)], ["difference", "n"]]}
          keys={differ.map(each => each.start.start)}
          rows={differ.map(each => [
            <span title={each.start.title ?? ""}>{startName(each.start)}</span>,
            { text: <Played subject={a} start={each.start} />, kind: "cell-result" },
            { text: <Played subject={b} start={each.start} />, kind: "cell-result" },
            { text: `${each.a! > each.b! ? "+" : ""}${Math.round(100 * (each.a! - each.b!))}%`, kind: each.a! > each.b! ? "t-good" : "t-bad" },
          ])}
        />
      ) : null}
    </Card>
  );
}
