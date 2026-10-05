// The environments: every one the system knows of (offered by the cluster, trained on, played by a suite, or
// imported from git, with its source and what its check found), with the form that imports one; and one environment's
// page: what it says of itself where it loads on the monitor's machine (its version, description, rows, eval data and
// curriculum; a published one's as its import's check recorded it, with its source), each row with what the training
// runs played of it, its runs, its suites and the evals of them at its entry, and its newest check; with the forms that
// start a run on it and make a suite of it.

import { Fragment, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useEnvironmentPage, useEnvironments, useImports, useSystem } from "../api/queries";
import type { EnvironmentInfo, PublishedSource } from "../api/types";
import { CheckpointTag } from "../components/checkpoints";
import { ImportForm } from "../components/environments";
import { shareText } from "../components/evals";
import { SuiteForm } from "../components/suites";
import { Card, Empty, Head, Mark, Share, Spec, Specs, Table } from "../components/ui";
import { Ago } from "../layout/runs";
import { bySubject, checkFound, checkText, rangeText, rowCounts, rowTotals, shortVersion, solvedShare, sourceText, versionsText } from "../lib/environments";
import { figure } from "../lib/format";
import { runKind } from "../lib/model";
import { environmentPlace, environmentsPlace, evalPlace, launchOn, runPlace, suitePlace } from "../lib/places";
import { versionTag } from "../lib/suites";

export function Environments() {
  const { data: known } = useEnvironments();
  const { data: imports } = useImports();
  const navigate = useNavigate();
  const [importing, setImporting] = useState(false);
  if (!known) return <Empty>Reading the environments…</Empty>;
  const offered = known.filter(each => each.offered).length;
  const able = imports?.importing ?? true;
  const action = (
    <button type="button" className="action" onClick={() => setImporting(!importing)} disabled={!able && !importing}
      title={able ? undefined : "this monitor was started without a cluster config to import with (rollout monitor --cluster)"}>
      {importing ? "Close" : "Import from git"}
    </button>
  );
  return (
    <>
      <Head title={<span className="head-with-action">Environments{action}</span>}>
        {known.length ? (
          <Specs>
            <Spec label="environments">{known.length}</Spec>
            <Spec label="offered" kind={offered ? "good" : ""}>{offered}</Spec>
          </Specs>
        ) : null}
      </Head>
      {importing ? <ImportForm onDone={version => navigate(environmentPlace(version.reference))} onCancel={() => setImporting(false)} /> : null}
      {known.length ? (
        <Card>
        <Table
          heads={[["environment"], ["offered"], ["versions"], ["source"], ["check", "nowrap"], ["runs", "n"], ["suites", "n"], ["last used", "nowrap"]]}
          keys={known.map(each => each.environment)}
          rows={known.map(each => [
            <span className="named-line"><b>{each.name}</b><small className="mono faint">{each.published ? each.published.entry_point : each.environment}</small></span>,
            each.offered ? <Mark state="running">offered</Mark> : <span className="faint">no</span>,
            each.versions.join(", ") || "–",
            each.published ? <span className="mono" title={each.published.source}>{sourceText(each.published)} <span className="faint">@ {each.published.commit.slice(0, 7)}</span></span> : <span className="faint">–</span>,
            each.published ? <span className={each.published.passed ? "t-good" : "t-bad"}>{checkText(each.published.check)}</span> : <span className="faint">–</span>,
            each.runs?.length ?? 0,
            each.suites?.length ?? 0,
            each.used ? <><Ago at={each.used} /> ago</> : "–",
          ])}
          to={known.map(each => environmentPlace(each.environment))}
        />
        </Card>
      ) : <Empty>No environment yet: none is offered by the cluster, trained on, played by a suite, or imported.</Empty>}
    </>
  );
}

export function Environment({ name }: { name: string }) {
  const { data: found, error } = useEnvironmentPage(name);
  const navigate = useNavigate();
  const [making, setMaking] = useState(false);
  if (error && !found) return <Empty>No environment {name} is known here. <Link to={environmentsPlace} className="linkish">Every environment</Link></Empty>;
  if (!found) return <Empty>Reading the environment…</Empty>;
  const description = found.description, curriculum = found.curriculum, published = found.published;
  const actions = (
    <>
      <Link to={launchOn(found.environment)} className="action">New run</Link>
      <button type="button" className="action" onClick={() => setMaking(!making)}>{making ? "Close" : "New suite"}</button>
    </>
  );
  return (
    <>
      <Head title={<span className="head-with-action">{found.name}{actions}</span>}>
        <Specs>
          {published ? (
            <>
              <Spec label="source"><span className="mono" title={published.source}>{sourceText(published)}</span></Spec>
              <Spec label="commit"><span className="mono" title={published.commit}>{published.commit.slice(0, 12)}</span>{published.ref ? ` · ${published.ref}` : ""}</Spec>
              <Spec label="entry point"><span className="mono">{published.entry_point}</span></Spec>
              <Spec label="version"><span className="mono" title={published.version}>{shortVersion(published.version)}</span></Spec>
              <Spec label="imported"><Ago at={published.imported} /> ago</Spec>
            </>
          ) : <Spec label="module:name"><span className="mono">{found.environment}</span></Spec>}
          <Spec label="offered" kind={found.offered ? "good" : ""}>{found.offered ? "yes" : "the cluster does not offer it"}</Spec>
          {found.loads ? null : <Spec label="here" kind="warm"><span title={found.error ?? undefined}>does not load</span></Spec>}
          <Spec label={found.loads ? "version" : "versions seen"}>{versionsText(found).map((version, index) => (
            <Fragment key={version}>{index ? ", " : ""}{index === 0 && found.version ? <b>{version}</b> : version}</Fragment>
          ))}{versionsText(found).length ? null : "–"}</Spec>
          {description ? <Spec label="rewards">{rangeText(description)}</Spec> : null}
          {description ? <Spec label="solved">{description.solved ? "reported" : "not reported"}</Spec> : null}
          {description?.duration ? <Spec label="duration">{description.duration}</Spec> : null}
          {curriculum ? <Spec label="curriculum">{curriculum.own ? `its own (${curriculum.name})` : "generic"}</Spec> : null}
        </Specs>
      </Head>
      {making ? (
        <SuiteForm title="New suite" environment={found.environment} onDone={version => navigate(suitePlace(version.split("@")[0]))} onCancel={() => setMaking(false)} />
      ) : null}
      <Rows found={found} />
      <EvalData found={found} />
      <Runs found={found} />
      <Scores found={found} />
      <Check found={found} />
      {published ? <Imported published={published} /> : null}
    </>
  );
}

function Imported({ published }: { published: PublishedSource }) {
  return (
    <Card title="Import check" note={checkText(published.check)}>
      <Table
        heads={[["check"], [""], ["said"]]}
        keys={published.check.map(each => each.check)}
        rows={published.check.map(each => [
          <span className="mono">{each.check}</span>,
          each.passed ? (each.flagged ? { text: "flagged", kind: "t-warm" } : { text: "ok", kind: "t-good" }) : { text: "failed", kind: "t-bad" },
          each.said,
        ])}
      />
      {published.dependencies.length ? <p className="muted small mono">{published.dependencies.join(" · ")}</p> : null}
    </Card>
  );
}

function Rows({ found }: { found: EnvironmentInfo }) {
  const totals = rowTotals(found.rows), counts = rowCounts(found.rows);
  const note = `${counts.played} of ${counts.rows} played${counts.held ? ` · ${counts.held} held out` : ""}`;
  return (
    <Card title="Rows" note={found.rows.length ? note : undefined}>
      {found.rows.length ? (
        <Table
          heads={[["row"], ["title"], ["groups", "n"], ["episodes", "n"], ["solved"], ["mean reward", "n"], ["eval starts", "n"]]}
          keys={found.rows.map(row => row.key)}
          rows={[
            ...found.rows.map(row => [
              <span className="mono">{row.key}{row.trains ? null : <small className="faint"> eval only</small>}</span>,
              row.title === row.key ? "" : row.title,
              row.groups || <span className="faint">–</span>,
              row.played || <span className="faint">–</span>,
              <Share value={solvedShare(row)} kind="good" />,
              row.reward == null ? <span className="faint">–</span> : figure(row.reward),
              row.held || <span className="faint">–</span>,
            ]),
            [{ text: <b>every row</b>, kind: "nowrap" }, "", totals.groups, totals.played, <Share value={solvedShare(totals)} kind="good" />, "", ""],
          ]}
        />
      ) : <p className="muted">{found.loads ? "It has no rows." : "No run played it yet."}</p>}
    </Card>
  );
}

function EvalData({ found }: { found: EnvironmentInfo }) {
  const data = Object.entries(found.evals);
  return (
    <div className="cols">
      <Card title="Eval data">
        {data.length ? (
          <Table heads={[["name"], ["starts", "n"]]} keys={data.map(([each]) => each)} rows={data.map(([each, starts]) => [<span className="mono">{each}</span>, starts])} />
        ) : <p className="muted">{found.loads ? "None." : "Not known: it does not load here."}</p>}
      </Card>
      <Card title="Suites">
        {found.suites.length ? (
          <Table
            heads={[["suite"], ["versions"], ["starts", "n"]]}
            keys={found.suites.map(each => each.suite)}
            rows={found.suites.map(each => [
              <b>{each.suite}</b>,
              <span>{each.versions.map(version => (
                <span key={version.id} className={`tag-version${version.id === each.version ? " current" : ""}`} title={version.id === each.version ? "the version its name points to" : undefined}>{versionTag(version.id)}</span>
              ))}{each.current ? null : <small className="faint"> not in {versionTag(each.version)}</small>}</span>,
              each.versions.at(-1)?.starts ?? 0,
            ])}
            to={found.suites.map(each => suitePlace(each.suite))}
          />
        ) : <p className="muted">None.</p>}
      </Card>
    </div>
  );
}

function Runs({ found }: { found: EnvironmentInfo }) {
  const { data: system } = useSystem();
  const stateOf = (run: string) => system?.runs.find(each => each.run === run)?.state;
  return (
    <Card title="Runs">
      {found.runs.length ? (
        <Table
          heads={[["run"], ["state"], ["version"], ["groups", "n"], ["episodes", "n"], ["solved"], ["started"]]}
          keys={found.runs.map(run => run.run)}
          rows={found.runs.map(run => {
            const state = stateOf(run.run);
            return [
              <b title={`id: ${run.run}`}>{run.name}</b>,
              state ? <Mark state={state} /> : "–",
              run.version ?? "–",
              run.groups,
              run.played,
              <Share value={solvedShare(run)} kind={state ? runKind(state) || "good" : "good"} />,
              run.started ? <><Ago at={run.started} /> ago</> : "–",
            ];
          })}
          to={found.runs.map(run => runPlace(run.run))}
        />
      ) : <p className="muted">None trained on it yet.</p>}
    </Card>
  );
}

function Scores({ found }: { found: EnvironmentInfo }) {
  const subjects = bySubject(found.scores);
  const rows = subjects.flatMap(subject => subject.scores.map((score, index) => ({ subject, score, first: index === 0 })));
  return (
    <Card title="Evals">
      {rows.length ? (
        <Table
          heads={[["played by"], ["eval"], ["suite"], ["episodes", "n"], ["solved", "n"], ["mean reward", "n"], ["started"]]}
          keys={rows.map(({ score }) => score.run)}
          rows={rows.map(({ subject, score, first }) => [
            first ? <CheckpointTag id={subject.checkpoint} base={subject.model} link={false} /> : "",
            <span title={`id: ${score.run}`}>{score.name}{score.done ? null : <small className="faint"> playing</small>}</span>,
            <span>{score.suite}<span className="tag-version">{versionTag(score.version)}</span></span>,
            score.played,
            shareText(score.share),
            figure(score.reward),
            score.started ? <><Ago at={score.started} /> ago</> : "–",
          ])}
          to={rows.map(({ score }) => evalPlace(score.run))}
        />
      ) : <p className="muted">No suite with it was played yet.</p>}
    </Card>
  );
}

function Check({ found }: { found: EnvironmentInfo }) {
  const check = found.check;
  if (!check) return null;
  const said = checkFound(check.groups);
  const note = (
    <><Link to={runPlace(check.run)} className="linkish">{check.name}</Link>{check.started ? <> · <Ago at={check.started} /> ago</> : null}{check.version ? ` · version ${check.version}` : ""}</>
  );
  return (
    <Card title="Newest check" note={note}>
      <p className={said.flagged === said.played && said.played ? "t-bad" : said.flagged ? "t-warm" : "muted"}>{said.says}{check.ended?.how === "failed" ? ` · it failed: ${check.ended.detail ?? ""}` : ""}</p>
      {check.groups.length ? (
        <Table
          heads={[["group", "n"], ["row"], ["rewards"], ["failed", "n"], [""]]}
          keys={check.groups.map(group => group.group)}
          rows={check.groups.map(group => [
            `#${group.group}`,
            <span className="mono">{group.task}</span>,
            group.rewards ? group.rewards.map(reward => figure(reward)).join(" ") || "none" : <span className="faint">playing</span>,
            group.failed ? <span title={group.failures.join("; ")}>{group.failed}</span> : <span className="faint">–</span>,
            group.flagged ? { text: "every episode scored the same", kind: "t-warm" } : "",
          ])}
        />
      ) : null}
    </Card>
  );
}
