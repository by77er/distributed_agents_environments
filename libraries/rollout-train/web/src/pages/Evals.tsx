// The evals: every suite (a frozen list of starts) with the subject that does best at it, every eval (one suite
// played by one checkpoint, nothing trained) newest first, and the evals asked for from the page.

import { Link } from "react-router-dom";
import { useEvals, useKnown, useLaunches, useSystem } from "../api/queries";
import type { EvalSuite } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { anySolved, shareOf, shareText, type Subject, subjectText } from "../components/evals";
import { LaunchList } from "../components/launches";
import { Card, Empty, Head, Mark, Spec, Specs, Table, Tile } from "../components/ui";
import { Ago } from "../layout/runs";
import { clock, figure } from "../lib/format";
import { runPlace, suitePlace } from "../lib/places";

/** How to make a suite, for a ledger with none (or to make another). */
export function MakeSuite({ ledger }: { ledger: string }) {
  return (
    <Card title="Make a suite" note="a suite is never changed: make another for other starts">
      <p className="muted" style={{ marginTop: 0 }}>A suite draws each row's start once for each seed, from a catalog a launcher offers, and keeps them in the ledger.</p>
      <pre className="command">{`rollout suite make NAME --catalog module:name --rows ROW,ROW --seeds 1,2,3 --ledger ${ledger}`}</pre>
    </Card>
  );
}

export function Evals() {
  const { data: evals } = useEvals();
  const { data: system } = useSystem();
  const { data: launched } = useLaunches();
  if (!evals || !system) return <Empty>Reading the evals…</Empty>;
  const going = evals.evals.filter(each => !each.done).length;
  const launches = (launched?.launches ?? []).filter(each => each.asked.kind === "eval");
  return (
    <>
      <Head title="Evals" sub="A suite is a frozen list of starts, each a catalog row's start with a seed of its own. An eval plays one suite with one checkpoint (or the base model), trains on nothing, and records how it did at each start, so that checkpoints compare start for start.">
        <Specs>
          <Spec label="suites">{evals.suites.length}</Spec>
          <Spec label="evals">{evals.evals.length}</Spec>
          {going ? <Spec label="playing" kind="good">{going}</Spec> : null}
        </Specs>
      </Head>
      {launches.length ? <LaunchList launches={launches} system={system} /> : null}
      {evals.suites.length ? (
        <div className="tiles">{evals.suites.map(suite => <SuiteTile key={suite.suite} suite={suite} />)}</div>
      ) : <MakeSuite ledger={system.ledger_at} />}
      <Card title="Every eval" note="newest first; one opens its run">
        {evals.evals.length ? (
          <Table
            heads={[["eval"], ["suite"], ["played by"], ["played", "n"], ["solved", "n"], ["started"], ["state"]]}
            keys={evals.evals.map(each => each.run)}
            rows={evals.evals.map(each => [
              <b title={`id: ${each.run}`}>{each.name}</b>,
              <Link to={suitePlace(each.suite)} className="linkish" onClick={event => event.stopPropagation()}>{each.suite}</Link>,
              <CheckpointTag id={each.checkpoint} link={false} />,
              `${each.played}/${each.expected}`,
              shareText(each.played && each.solved != null ? each.solved / each.played : null),
              each.started ? <><Ago at={each.started} /> ago</> : "–",
              <Mark state={each.done ? "ended" : "running"}>{each.done ? "done" : "playing"}</Mark>,
            ])}
            to={evals.evals.map(each => runPlace(each.run))}
          />
        ) : <p className="muted">No eval has been played yet: open a suite to run it with a checkpoint.</p>}
      </Card>
    </>
  );
}

function SuiteTile({ suite }: { suite: EvalSuite }) {
  const known = useKnown();
  const said = anySolved(suite.subjects), score = (subject: Subject) => (said ? shareOf(subject) : subject.reward) ?? -Infinity;
  const best = [...suite.subjects].filter(each => each.played).sort((a, b) => score(b) - score(a))[0];
  return (
    <Tile to={suitePlace(suite.suite)} className="rail accent">
      <header><b>{suite.suite}</b><span className="what">{suite.catalog ?? ""}</span></header>
      <div className="cells">
        <div className="cell"><span>starts</span><b>{suite.starts.length}</b><small>{new Set(suite.starts.map(start => start.task)).size} rows</small></div>
        <div className="cell"><span>played by</span><b>{suite.subjects.length}</b><small>subjects</small></div>
        <div className={`cell ${best ? "good" : ""}`}><span>best</span><b>{best ? (said ? shareText(shareOf(best)) : figure(best.reward)) : "–"}</b><small>{best ? subjectText(best, known) : "none yet"}</small></div>
      </div>
      <div className="facts">{suite.made ? <span>made {clock(suite.made)}</span> : null}{suite.sample ? <span>sample</span> : null}</div>
    </Tile>
  );
}
