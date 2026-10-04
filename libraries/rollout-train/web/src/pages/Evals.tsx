// The evals: every suite (an eval configuration, kept in versions) with the subject that does best at the version its
// name points to, the form that makes a new suite, every eval (one version of a suite played by one checkpoint, nothing
// trained) newest first, and the evals asked for from the page.

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useEvals, useKnown, useLaunches, useSystem } from "../api/queries";
import type { EvalSuite } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { anySolved, entriesText, shareOf, shareText, type Subject, subjectText } from "../components/evals";
import { LaunchList } from "../components/launches";
import { SuiteForm } from "../components/suites";
import { Card, Empty, Head, Mark, Spec, Specs, Table, Tile } from "../components/ui";
import { Ago } from "../layout/runs";
import { readable } from "../lib/environments";
import { clock, figure } from "../lib/format";
import { evalPlace, suitePlace } from "../lib/places";
import { currentOf, playedVersion, versionsOf, versionTag } from "../lib/suites";

/** How to make a suite from the command line, for a ledger with none. */
export function MakeSuite({ ledger }: { ledger: string }) {
  return (
    <Card title="Make a suite">
      <pre className="command">{`rollout suite make NAME --environment module:name --rows ROW,ROW --seeds 1,2,3 --ledger ${ledger}`}</pre>
    </Card>
  );
}

export function Evals() {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const { data: launched } = useLaunches();
  const navigate = useNavigate();
  const [making, setMaking] = useState(false);
  if (!evals || !system) return <Empty>Reading the evals…</Empty>;
  const going = evals.evals.filter(each => !each.done).length;
  const launches = (launched?.launches ?? []).filter(each => each.asked.kind === "eval");
  const several = new Set(evals.suites.filter(each => versionsOf(each).length > 1).map(each => each.suite));
  const open = making || !evals.suites.length;
  return (
    <>
      <Head title={<span className="head-with-action">Evals{evals.suites.length ? <button type="button" className="action" onClick={() => setMaking(!making)}>{making ? "Close" : "New suite"}</button> : null}</span>}>
        <Specs>
          <Spec label="suites">{evals.suites.length}</Spec>
          <Spec label="evals">{evals.evals.length}</Spec>
          {going ? <Spec label="playing" kind="good">{going}</Spec> : null}
        </Specs>
      </Head>
      {open ? <SuiteForm title="New suite" onDone={version => navigate(suitePlace(version.split("@")[0]))} onCancel={evals.suites.length ? () => setMaking(false) : undefined} /> : null}
      {launches.length ? <LaunchList launches={launches} system={system} /> : null}
      {evals.suites.length ? (
        <div className="tiles">{evals.suites.map(suite => <SuiteTile key={suite.suite} suite={suite} />)}</div>
      ) : null}
      <Card title="Every eval">
        {evals.evals.length ? (
          <Table
            heads={[["eval"], ["suite"], ["played by"], ["played", "n"], ["solved", "n"], ["started"], ["state"]]}
            keys={evals.evals.map(each => each.run)}
            rows={evals.evals.map(each => [
              <b title={`id: ${each.run}`}>{each.name}</b>,
              <><Link to={suitePlace(each.suite)} className="linkish" onClick={event => event.stopPropagation()}>{each.suite}</Link>{several.has(each.suite) ? <span className="tag-version">{versionTag(each.version)}</span> : null}</>,
              <CheckpointTag id={each.checkpoint} link={false} />,
              `${each.played}/${each.expected}`,
              entriesText(each.entries) ?? shareText(each.played && each.solved != null ? each.solved / each.played : null),
              each.started ? <><Ago at={each.started} /> ago</> : "–",
              <Mark state={each.done ? "ended" : "running"}>{each.done ? "done" : "playing"}</Mark>,
            ])}
            to={evals.evals.map(each => evalPlace(each.run))}
          />
        ) : <p className="muted">None yet.</p>}
      </Card>
    </>
  );
}

function SuiteTile({ suite }: { suite: EvalSuite }) {
  const known = useKnown();
  const current = currentOf(suite), versions = versionsOf(suite);
  const subjects = suite.subjects.filter(subject => playedVersion(subject, suite.suite) === current.id);  // (they compare)
  const said = anySolved(subjects), score = (subject: Subject) => (said ? shareOf(subject) : subject.reward) ?? -Infinity;
  const one = current.entries.length === 1;  // (a best of several environments would weigh one's rewards against another's)
  const best = one ? [...subjects].filter(each => each.played).sort((a, b) => score(b) - score(a))[0] : undefined;
  const environments = (suite.environments ?? current.environments).map(readable);
  return (
    <Tile to={suitePlace(suite.suite)} className="rail accent">
      <header><b>{suite.suite}{versions.length > 1 ? <span className="tag-version">{versionTag(current.id)}</span> : null}</b><span className="what" title={(suite.environments ?? []).join(", ")}>{environments.join(" · ")}</span></header>
      <div className="cells three">
        <div className="cell"><span>starts</span><b>{current.starts.length}</b><small>{new Set(current.starts.map(start => start.task)).size} rows</small></div>
        <div className="cell"><span>played by</span><b>{suite.subjects.length}</b><small>{versions.length > 1 ? `${subjects.length} on ${versionTag(current.id)}` : "subjects"}</small></div>
        {one ? <div className={`cell ${best ? "good" : ""}`}><span>best</span><b>{best ? (said ? shareText(shareOf(best)) : figure(best.reward)) : "–"}</b><small>{best ? subjectText(best, known) : ""}</small></div>
          : <div className="cell"><span>environments</span><b>{current.entries.length}</b><small /></div>}
      </div>
      <div className="facts">{current.made ? <span>made {clock(current.made)}</span> : null}{suite.sample ? <span>sample</span> : null}</div>
    </Tile>
  );
}
