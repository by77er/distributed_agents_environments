// A suite: the version its name points to (an eval configuration of one or more environments, its entries), every subject
// that played any of its versions start by start, two subjects of one version compared, the form that edits it (a new
// version), and the form that asks for an eval of it by a checkpoint or a base model (nothing trained).

import { useState } from "react";
import { Link } from "react-router-dom";
import { useEvals, useKnown, useLaunches, useOffers, useSystem } from "../api/queries";
import type { EvalSuite, SuiteEntry, SuiteVersion } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { anySolved, Played, shareOf, shareText, startName, startShare, type Subject, subjectText, SuiteMatrix } from "../components/evals";
import { Starting } from "../components/launches";
import { PlayForm } from "../components/play";
import { SuiteForm } from "../components/suites";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { Ago } from "../layout/runs";
import { clock, figure } from "../lib/format";
import { readable } from "../lib/environments";
import { evalPlace, evalsPlace } from "../lib/places";
import { ALL, chosenText, currentOf, entryStarts, limitsText, playedVersion, suiteName, versionsOf, versionTag } from "../lib/suites";

export function Suite({ name }: { name: string }) {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const { data: launched } = useLaunches();
  const { data: offers } = useOffers();
  const [picked, setPicked] = useState(ALL);
  const [editing, setEditing] = useState(false);
  if (!evals || !system) return <Empty>Reading the suite…</Empty>;
  const suite = evals.suites.find(each => each.suite === name);
  if (!suite) return <Empty>There is no suite {name}. <Link to={evalsPlace} className="linkish">Every suite</Link></Empty>;
  const versions = versionsOf(suite), current = currentOf(suite);
  const shown = versions.find(each => each.id === picked) ?? current;
  const playing = evals.evals.filter(each => each.suite === name && !each.done);
  const launches = (launched?.launches ?? []).filter(each => each.asked.kind === "eval" && suiteName(String(each.asked.settings["eval.suite"] ?? "")) === name);
  const said = anySolved(suite.subjects), score = (subject: Subject) => (said ? shareOf(subject) : subject.reward ?? null);
  const ranked = [...suite.subjects].sort((a, b) => (score(b) ?? -Infinity) - (score(a) ?? -Infinity));
  const compared = ranked.filter(subject => playedVersion(subject) === shown.id);  // (subjects compare within a version)
  const listed = picked === ALL ? ranked : compared;
  const one = shown.entries.length === 1;  // (a best of several environments would weigh one's rewards against another's)
  const action = <button type="button" className="action" onClick={() => setEditing(!editing)}>{editing ? "Close" : "Edit"}</button>;
  return (
    <>
      <Head title={<span className="head-with-action">{suite.suite}{action}</span>}>
        <Specs>
          <Spec label="version">{versionTag(shown.id)}{shown.id === current.id ? (versions.length > 1 ? ` of ${versions.length}` : "") : ` · newest ${versionTag(current.id)}`}</Spec>
          {one ? <EntrySpecs version={shown} entry={shown.entries[0]} /> : <Spec label="environments">{shown.entries.length}</Spec>}
          {shown.held_out ? <Spec label="held out" kind="good">yes</Spec> : null}
          {shown.made ? <Spec label="made">{clock(shown.made)}</Spec> : null}
        </Specs>
      </Head>
      {one ? null : <Entries version={shown} />}
      {editing ? <SuiteForm key={current.id} title={`Edit ${suite.suite}`} name={suite.suite} version={current} onDone={() => setEditing(false)} onCancel={() => setEditing(false)} /> : null}
      <Kpis>
        <Kpi label="Starts" value={String(shown.starts.length)} note={versionTag(shown.id)} />
        <Kpi label="Played by" value={String(listed.length)} />
        {one ? <Kpi label="Best" value={compared[0] ? (said ? shareText(shareOf(compared[0])) : figure(compared[0].reward)) : "–"} note={compared[0] ? <SubjectLabel subject={compared[0]} /> : versionTag(shown.id)} /> : null}
        <Kpi label="Playing" value={String(playing.length)} note={playing.map(each => each.name).join(", ")} />
      </Kpis>
      {offers ? <PlayForm suite={suite} suites={evals.suites} offers={offers} system={system} title="Run this suite" /> : null}
      <Starting launches={launches} system={system} titled />
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

/** One entry of a version, as the head of a suite of one environment says it. */
function EntrySpecs({ version, entry }: { version: SuiteVersion; entry: SuiteEntry }) {
  const starts = entryStarts(version, entry);
  return (
    <>
      <Spec label="environment"><span title={entry.environment ?? ""}>{readable(entry.environment)}</span>{entry.environment_version ? ` · ${entry.environment_version}` : ""}</Spec>
      <Spec label="starts">{chosenText(entry)}</Spec>
      <Spec label="rows">{[...new Set(starts.map(start => start.task))].join(", ")}</Spec>
      <Spec label="seeds">{[...new Set(starts.map(start => String(start.seed)))].join(", ")}</Spec>
      <Spec label="episodes per start">{entry.episodes}</Spec>
      <Spec label="limits">{limitsText(entry)}</Spec>
    </>
  );
}

/** A version's entries, one environment each. */
function Entries({ version }: { version: SuiteVersion }) {
  return (
    <Card title="Environments" note={versionTag(version.id)}>
      <Table
        heads={[["environment"], ["version"], ["starts"], ["rows"], ["seeds"], ["episodes", "n"], ["limits"]]}
        keys={version.entries.map(entry => entry.environment ?? String(entry.offset))}
        rows={version.entries.map(entry => {
          const starts = entryStarts(version, entry);
          return [
            <span title={entry.environment ?? ""}><b>{readable(entry.environment)}</b>{entry.held_out ? <small className="faint"> held out</small> : null}</span>,
            entry.environment_version ?? "–",
            `${entry.starts} · ${chosenText(entry)}`,
            [...new Set(starts.map(start => start.task))].join(", "),
            [...new Set(starts.map(start => String(start.seed)))].join(", "),
            entry.episodes,
            limitsText(entry),
          ];
        })}
      />
    </Card>
  );
}

/** Which version the grid shows: every version, or one. */
function VersionPicker({ suite, picked, onPick }: { suite: EvalSuite; picked: string; onPick: (version: string) => void }) {
  const current = currentOf(suite);
  const played = (id: string) => suite.subjects.filter(subject => playedVersion(subject) === id).length;
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
