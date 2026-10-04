// A suite: the version its name points to (an eval configuration), every subject that played any of its versions start
// by start, two subjects of one version compared, the form that edits it (a new version), and the form that asks a
// launcher to play it with a checkpoint (an eval: nothing trained).

import { useState } from "react";
import { Link } from "react-router-dom";
import { useEvals, useKnown, useLaunches, useSystem } from "../api/queries";
import type { EvalSuite, SuiteVersion } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { anySolved, Played, shareOf, shareText, startName, startShare, type Subject, subjectText, SuiteMatrix } from "../components/evals";
import { LaunchList } from "../components/launches";
import { PlayForm } from "../components/play";
import { SuiteForm } from "../components/suites";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { Ago } from "../layout/runs";
import { clock, figure } from "../lib/format";
import { evalPlace, evalsPlace } from "../lib/places";
import { ALL, currentOf, playedVersion, suiteName, versionsOf, versionTag } from "../lib/suites";
import { NoLauncher } from "./NewRun";

/** A version's sampling limits, in a few words (none: the channel's own). */
export const limitsText = (version: SuiteVersion): string =>
  [version.thinking_tokens != null ? `thinking ${version.thinking_tokens}` : "", version.answer_tokens != null ? `answer ${version.answer_tokens}` : ""].filter(Boolean).join(" · ");

/** How a version's starts were chosen, in a few words. */
export const chosenText = (version: SuiteVersion): string =>
  version.chosen === "eval data" ? `eval data ${version.eval_data ?? ""}` : version.chosen === "starts" ? "given starts" : "rows and seeds";

export function Suite({ name }: { name: string }) {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const { data: launched } = useLaunches();
  const [picked, setPicked] = useState(ALL);
  const [editing, setEditing] = useState(false);
  if (!evals || !system) return <Empty>Reading the suite…</Empty>;
  const suite = evals.suites.find(each => each.suite === name);
  if (!suite) return <Empty>There is no suite {name}. <Link to={evalsPlace} className="linkish">Every suite</Link></Empty>;
  const versions = versionsOf(suite), current = currentOf(suite);
  const shown = versions.find(each => each.id === picked) ?? current;
  const playing = evals.evals.filter(each => each.suite === name && !each.done);
  const launches = (launched?.launches ?? []).filter(each => each.asked.kind === "eval" && suiteName(each.asked.suite ?? "") === name);
  const said = anySolved(suite.subjects), score = (subject: Subject) => (said ? shareOf(subject) : subject.reward ?? null);
  const ranked = [...suite.subjects].sort((a, b) => (score(b) ?? -Infinity) - (score(a) ?? -Infinity));
  const compared = ranked.filter(subject => playedVersion(subject, name) === shown.id);  // (subjects compare within a version)
  const listed = picked === ALL ? ranked : compared;
  const rows = [...new Set(shown.starts.map(start => start.task))];
  const seeds = [...new Set(shown.starts.map(start => String(start.seed)))];
  const limits = limitsText(shown);
  const action = <button type="button" className="action" onClick={() => setEditing(!editing)}>{editing ? "Close" : "Edit"}</button>;
  return (
    <>
      <Head title={<span className="head-with-action">{suite.suite}{action}</span>}>
        <Specs>
          <Spec label="version">{versionTag(shown.id)}{shown.id === current.id ? (versions.length > 1 ? ` of ${versions.length}` : "") : ` · newest ${versionTag(current.id)}`}</Spec>
          <Spec label="environment">{shown.environment ?? "–"}{shown.environment_version ? ` · ${shown.environment_version}` : ""}</Spec>
          <Spec label="starts">{chosenText(shown)}</Spec>
          <Spec label="rows">{rows.join(", ")}</Spec>
          <Spec label="seeds">{seeds.join(", ")}</Spec>
          <Spec label="episodes per start">{shown.episodes}</Spec>
          {limits ? <Spec label="limits">{limits}</Spec> : null}
          {shown.held_out ? <Spec label="held out" kind="good">yes</Spec> : null}
          {shown.made ? <Spec label="made">{clock(shown.made)}</Spec> : null}
        </Specs>
      </Head>
      {editing ? <SuiteForm key={current.id} title={`Edit ${suite.suite}`} name={suite.suite} version={current} onDone={() => setEditing(false)} onCancel={() => setEditing(false)} /> : null}
      <Kpis>
        <Kpi label="Starts" value={String(shown.starts.length)} note={versionTag(shown.id)} />
        <Kpi label="Played by" value={String(listed.length)} />
        <Kpi label="Best" value={compared[0] ? (said ? shareText(shareOf(compared[0])) : figure(compared[0].reward)) : "–"} note={compared[0] ? <SubjectLabel subject={compared[0]} /> : versionTag(shown.id)} />
        <Kpi label="Playing" value={String(playing.length)} note={playing.map(each => each.name).join(", ")} />
      </Kpis>
      {launched ? (launched.launchers.length ? <PlayForm suite={suite} suites={evals.suites} launchers={launched.launchers} system={system} title="Run this suite" /> : <NoLauncher ledger={system.ledger_at} />) : null}
      {launches.length ? <LaunchList launches={launches} system={system} /> : null}
      {playing.length ? (
        <Card title="Playing now">
          <Table
            heads={[["eval"], ["version"], ["played by"], ["played", "n"], ["started"]]}
            keys={playing.map(each => each.run)}
            rows={playing.map(each => [<b>{each.name}</b>, versionTag(each.version), <CheckpointTag id={each.checkpoint} link={false} />, `${each.played}/${each.expected}`, each.started ? <><Ago at={each.started} /> ago</> : "–"])}
            to={playing.map(each => evalPlace(each.run))}
          />
        </Card>
      ) : null}
      <Card title="Start by start" note={versions.length > 1 ? <VersionPicker suite={suite} picked={picked} onPick={setPicked} /> : undefined}>
        {listed.length ? <SuiteMatrix suite={suite} subjects={listed} picked={picked} /> : <p className="muted">None played {versionTag(shown.id)} yet.</p>}
      </Card>
      {compared.length > 1 && said ? <Compare key={shown.id} version={shown} subjects={compared} /> : null}
    </>
  );
}

/** Which version the grid shows: every version, or one. */
function VersionPicker({ suite, picked, onPick }: { suite: EvalSuite; picked: string; onPick: (version: string) => void }) {
  const current = currentOf(suite);
  const played = (id: string) => suite.subjects.filter(subject => playedVersion(subject, suite.suite) === id).length;
  return (
    <select className="picker" value={picked} onChange={event => onPick(event.target.value)} aria-label="version">
      <option value={ALL}>every version</option>
      {[...versionsOf(suite)].reverse().map(version => (
        <option key={version.id} value={version.id}>{versionTag(version.id)}{version.id === current.id ? " (newest)" : ""} · {version.starts.length} starts · {played(version.id)} played</option>
      ))}
    </select>
  );
}

const SubjectLabel = ({ subject }: { subject: Subject }) => {
  const known = useKnown();
  return <>{subjectText(subject, known)}{subject.kind === "model" ? "" : ` · ${known.origin(subject.checkpoint)}`}</>;
};

/** Two subjects of one version, start by start: where each solved more than the other. */
function Compare({ version, subjects }: { version: SuiteVersion; subjects: Subject[] }) {
  const known = useKnown();
  const [first, setFirst] = useState(subjects[0].subject);
  const [second, setSecond] = useState(subjects[1].subject);
  const a = subjects.find(each => each.subject === first) ?? subjects[0];
  const b = subjects.find(each => each.subject === second) ?? subjects[1];
  const compared = version.starts.map(start => ({ start, a: startShare(a, start), b: startShare(b, start) }));
  const both = compared.filter(each => each.a != null && each.b != null);
  const better = both.filter(each => each.a! > each.b!), worse = both.filter(each => each.a! < each.b!);
  const differ = [...better, ...worse];
  const option = (subject: Subject) => (
    <option key={subject.subject} value={subject.subject}>{subjectText(subject, known)}{subject.kind === "model" ? "" : ` · ${known.origin(subject.checkpoint)}`} · {shareText(shareOf(subject))}</option>
  );
  return (
    <Card title="Compared" note={versionTag(version.id)}>
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
