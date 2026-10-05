// Runs asked for from the page: those whose run has not appeared yet, drawn as tiles of their own, and what was asked
// of a run that has, for its run's tile. The words for their states are a run's (`lib/launches`).

import { memo } from "react";
import { Link } from "react-router-dom";
import { useOffers, useStop } from "../api/queries";
import type { Launch, Preset, Run, System } from "../api/types";
import { Ago } from "../layout/runs";
import { publishedParts, readable } from "../lib/environments";
import { span } from "../lib/format";
import { changedSettings, GOING, type LaunchShown, launchState, loopChanged } from "../lib/launches";
import { evalPlace, runPlace, suitePlace } from "../lib/places";
import { suiteName, versionTag } from "../lib/suites";
import { CheckpointTag } from "./checkpoints";
import { useQueued } from "./queue";
import { SectionTitle, Tile } from "./ui";

/** The runs asked for whose run has not appeared yet: asked, starting, waiting, or failed before it started. A run that
 * exists is drawn by its own tile, with what was asked of it (`Asked`). */
export function Starting({ launches, system, titled = false }: { launches: Launch[]; system: System; titled?: boolean }) {
  const { data: offers } = useOffers();
  const known = new Set(system.runs.map(run => run.run));
  const waiting = launches.filter(each => !each.run || !known.has(each.run));
  if (!waiting.length) return null;
  const presets = offers?.presets ?? [];
  const tiles = waiting.map(launch => (
    <LaunchTile key={launch.id} launch={launch} system={system} preset={presets.find(each => each.id === launch.asked.preset)} />
  ));
  return titled ? (
    <>
      <SectionTitle id="section-starting" title="Starting" note={String(waiting.length)} />
      <div className="tiles wide-tiles launches">{tiles}</div>
    </>
  ) : <>{tiles}</>;
}

/** The launch that made each run, by the run's id. */
export function launchesByRun(launches: Launch[]): Map<string, Launch> {
  const by = new Map<string, Launch>();
  for (const launch of launches) if (launch.run && !by.has(launch.run)) by.set(launch.run, launch);
  return by;
}

/** What was asked of a run that a run's tile shows: its preset and changed settings, why it waits (and where it is in
 * the queue) or failed, and the button that stops it while it goes. */
export function Asked({ launch, run }: { launch: Launch; run: Run }) {
  const stop = useStop();
  const { data: offers } = useOffers();
  const preset = (offers?.presets ?? []).find(each => each.id === launch.asked.preset);
  const said = launchState(launch, run);
  const going = GOING.has(launch.state);
  const queued = useQueued(going ? launch.run : null);
  const changed = changedSettings(launch.asked.settings ?? {}, preset);
  return (
    <>
      {launch.asked.preset || changed.length ? (
        <div className="facts wraps settings-changed">
          {launch.asked.preset ? <span title="preset">{launch.asked.preset}</span> : null}
          {changed.map(each => <span key={each.key} title={each.key}>{each.text}</span>)}
        </div>
      ) : null}
      {said.state === "failed" && launch.detail ? (
        <details className="launch-detail" onClick={event => event.stopPropagation()}>
          <summary>why it failed</summary>
          <pre>{launch.detail}</pre>
        </details>
      ) : queued || said.reason ? <div className="small muted" title={launch.detail ?? undefined}>{queued ?? said.reason}</div> : null}
      {going && launch.state !== "stopping" ? (
        <div className="launch-actions">
          <button type="button" className="danger" disabled={stop.isPending}
            onClick={event => { event.preventDefault(); event.stopPropagation(); stop.mutate(launch.id); }}>
            {stop.isPending ? "stopping…" : "stop"}
          </button>
          {stop.isError ? <span className="error-text small">{stop.error.message}</span> : null}
        </div>
      ) : null}
    </>
  );
}

const DOT: Record<LaunchShown["state"], string> = {
  asked: "idle", submitted: "idle", waiting: "idle", running: "alive", paused: "paused", stopping: "idle", finished: "", stopped: "",
  failed: "gone", lost: "gone",
};

/** An environment by its readable name; a published version as its name and a chip of its version's first characters. */
function EnvironmentName({ environment }: { environment: string }) {
  const published = publishedParts(environment);
  if (!published) return <span title={environment}>{readable(environment)}</span>;
  return <span title={environment}>{published.name} <span className="tag-version">{published.version.slice(0, 8)}</span></span>;
}

const LaunchTile = memo(function LaunchTile({ launch, system, preset }: { launch: Launch; system: System; preset?: Preset }) {
  const stop = useStop();
  const asked = launch.asked;
  const settings = asked.settings ?? {};
  const run = launch.run ? system.runs.find(each => each.run === launch.run) : undefined;
  const said = launchState(launch, run);
  const going = GOING.has(launch.state);
  const queued = useQueued(going ? launch.run : null);
  const environment =typeof settings.environment === "string" ? settings.environment : null;
  const start = typeof settings.start === "string" ? settings.start : null;
  const suite = typeof settings["eval.suite"] === "string" ? settings["eval.suite"] : null;
  const episodes = settings["eval.episodes"];
  const model = typeof settings["channels.policy.model"] === "string" ? settings["channels.policy.model"] : null;
  const changed = changedSettings(settings, preset).filter(each => !(asked.kind === "eval" && each.key === "channels.policy.model"));
  const loop = asked.kind === "train" ? loopChanged(settings) : [];
  const place = run ? (run.kind === "eval" ? evalPlace(run.run) : runPlace(run.run)) : null;
  return (
    <Tile className={`rail ${said.kind}`}>
      <header>
        <span className={`dot ${DOT[said.state]}`} title={said.state} />
        <b>{place ? <Link to={place} className="linkish">{asked.name}</Link> : asked.name}</b>
        <span className="what">{environment ? <EnvironmentName environment={environment} /> : null}</span>
        <span className="faint small" title={launch.job ? `job ${launch.job}` : undefined}>{said.state === "submitted" ? "starting" : said.state}</span>
      </header>
      <div className="facts wraps">
        {asked.kind === "eval" ? (
          <span>
            plays {suite ? <><Link to={suitePlace(suiteName(suite))} className="linkish">{suiteName(suite)}</Link>{suite.includes("@") ? <span className="tag-version">{versionTag(suite)}</span> : null}</> : "its suite"}
            {" "}with <CheckpointTag id={start} base={model} />
          </span>
        ) : start ? <span>from <CheckpointTag id={start} /></span> : null}
        {asked.kind === "eval" && typeof episodes === "number" ? <span>{episodes} episode{episodes === 1 ? "" : "s"} a start</span> : null}
        {typeof settings.bookmark === "string" && settings.bookmark ? <span>carries <span className="chip bookmark">{settings.bookmark}</span></span> : null}
        {loop.map(each => <span key={each}>{each}</span>)}
        {asked.resumes ? <span>resumes</span> : null}
        {asked.preset ? <span title="preset">{asked.preset}</span> : null}
      </div>
      {changed.length ? (
        <div className="facts wraps settings-changed">
          {changed.map(each => <span key={each.key} title={each.key}>{each.text}</span>)}
        </div>
      ) : null}
      <div className="facts">
        {going ? <span>asked <Ago at={launch.at} /> ago</span> : <span>took {span(Math.max(0, launch.updated - launch.at))} · <Ago at={launch.updated} /> ago</span>}
      </div>
      {said.state === "failed" && launch.detail ? (
        <details className="launch-detail">
          <summary>why it failed</summary>
          <pre>{launch.detail}</pre>
        </details>
      ) : queued || said.reason ? <div className="small muted" title={launch.detail ?? undefined}>{queued ?? said.reason}</div> : null}
      {going && launch.state !== "stopping" ? (
        <div className="launch-actions">
          <button type="button" className="danger" disabled={stop.isPending} onClick={() => stop.mutate(launch.id)}>
            {stop.isPending ? "stopping…" : launch.state === "asked" ? "cancel" : "stop"}
          </button>
          {stop.isError ? <span className="error-text small">{stop.error.message}</span> : null}
        </div>
      ) : null}
    </Tile>
  );
});
