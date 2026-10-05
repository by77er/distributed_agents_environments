// A base model, a root of the checkpoints' graph: the runs trained from it (and those forked from theirs), every eval it
// had by the version of a suite each played, and the form that asks for another.

import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useEvals, useHistory, useLaunches, useLineage, useOffers, useSystem } from "../api/queries";
import type { System } from "../api/types";
import { Starting } from "../components/launches";
import { PlayForm, providersOf } from "../components/play";
import { isHosted } from "../lib/form";
import { Card, Empty, Head, Spec, Specs, Table } from "../components/ui";
import { historyOf } from "../lib/history";
import { launchFor, runPlace } from "../lib/places";
import { indexOf, type Lane, lanesOf } from "./Checkpoints";
import { MakeSuite } from "./Evals";
import { VersionCard } from "./Subject";

/** The lanes of runs that hang under a base model's lane: those trained from it, and those forked from theirs. */
export function trainedFrom(lanes: Lane[], model: string): Lane[] {
  const at = lanes.findIndex(lane => lane.base === model);
  if (at < 0) return [];
  const next = lanes.findIndex((lane, place) => place > at && lane.depth === 0);
  return lanes.slice(at + 1, next < 0 ? undefined : next).filter(lane => lane.run);
}

export function Base({ model }: { model: string }) {
  const { data: lineage } = useLineage();
  const { data: system } = useSystem();
  const { data: history, isLoading } = useHistory("model", model);  // (none, for a model the ledger knows nothing of)
  const index = useMemo(() => (lineage ? indexOf(lineage) : null), [lineage]);
  const lanes = useMemo(() => (lineage && index ? trainedFrom(lanesOf(lineage, index), model) : []), [lineage, index, model]);
  if (!lineage || !index || !system) return <Empty>Reading the base model…</Empty>;
  const evals = history?.evals ?? [], groups = historyOf(evals);
  const inLedger = new Set(system.runs.map(run => run.run));
  const runs = lanes.map(lane => lane.run!);
  return (
    <>
      <Head title={model}>
        <Specs>
          <Spec label="runs">{runs.length}</Spec>
          <Spec label="evals">{evals.length}</Spec>
          <Spec label="suites">{history?.subject.suites.length ?? 0}</Spec>
          {history?.subject.playing ? <Spec label="playing" kind="good">{history.subject.playing}</Spec> : null}
        </Specs>
      </Head>
      <Card title="Runs">
        {runs.length ? (
          <Table
            heads={[["run"], ["from"], ["checkpoints"], ["latest"]]}
            keys={runs.map(run => run.run)}
            rows={runs.map(run => [
              <b title={`id: ${run.run}`}>{run.name}</b>,
              run.from ? index.shortOf(run.from) : "the base model",
              run.checkpoints.length ? `${index.shortOf(run.checkpoints[0])}–${index.shortOf(run.checkpoints.at(-1))}` : "–",
              run.latest ? index.shortOf(run.latest) : "–",
            ])}
            to={runs.map(run => (inLedger.has(run.run) ? runPlace(run.run) : null))}
          />
        ) : <p className="muted">None.</p>}
      </Card>
      {groups.length ? groups.map(group => <VersionCard key={group.version} group={group} />) : isLoading ? <Empty>Reading the evals…</Empty> : null}
      <PlayIt model={model} system={system} />
    </>
  );
}

/** Ask for an eval of a suite by the base model, or a training run from it; and the evals of it asked for so. */
function PlayIt({ model, system }: { model: string; system: System }) {
  const { data: evals } = useEvals();
  const { data: launched } = useLaunches();
  const { data: offers } = useOffers();
  if (!evals || !launched || !offers) return null;
  const serving = providersOf(offers, model);
  const hosted = serving.length > 0 && serving.every(isHosted);  // (a hosted API's model is played, never trained)
  const launches = launched.launches.filter(each =>
    each.asked.kind === "eval" && !each.asked.settings.start && each.asked.settings["channels.policy.model"] === model);
  return (
    <>
      {!evals.suites.length ? <MakeSuite ledger={system.ledger_at} />
        : <PlayForm model={model} suites={evals.suites} offers={offers} system={system} title="Run an eval" />}
      {offers.cluster && !hosted ?<p><Link to={launchFor(model)} className="linkish">Train from {model}</Link></p> : null}
      <Starting launches={launches} system={system} titled />
    </>
  );
}
