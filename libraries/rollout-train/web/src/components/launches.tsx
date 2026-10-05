// The runs and evals asked for from the page, and how each goes, in the words a run's tile uses (`lib/launches`).

import { memo, useState } from "react";
import { Link } from "react-router-dom";
import { useOffers, useStop } from "../api/queries";
import type { Launch, Preset, System } from "../api/types";
import { Ago } from "../layout/runs";
import { publishedParts, readable } from "../lib/environments";
import { span } from "../lib/format";
import { changedSettings, GOING, type LaunchShown, launchState, loopChanged } from "../lib/launches";
import { evalPlace, runPlace, suitePlace } from "../lib/places";
import { suiteName, versionTag } from "../lib/suites";
import { CheckpointTag } from "./checkpoints";
import { SectionTitle, Tile } from "./ui";

const FINISHED_SHOWN = 6;

/** Every launch going, and the newest that finished. */
export function LaunchList({ launches, system }: { launches: Launch[]; system: System }) {
  const [all, setAll] = useState(false);
  const { data: offers } = useOffers();
  const going = launches.filter(each => GOING.has(each.state));
  const finished = launches.filter(each => !GOING.has(each.state));
  const shown = [...going, ...(all ? finished : finished.slice(0, FINISHED_SHOWN))];
  const hidden = finished.length - (all ? finished.length : Math.min(finished.length, FINISHED_SHOWN));
  const presets = offers?.presets ?? [];
  return (
    <>
      <SectionTitle id="section-launches" title="Launches" note={`${going.length} going · ${finished.length} finished`} />
      <div className="tiles wide-tiles launches">
        {shown.map(launch => <LaunchTile key={launch.id} launch={launch} system={system} preset={presets.find(each => each.id === launch.asked.preset)} />)}
      </div>
      {hidden > 0 || all ? (
        <button type="button" className="linkish small more" onClick={() => setAll(!all)}>{all ? "show fewer" : `and ${hidden} more finished`}</button>
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
  const environment = typeof settings.environment === "string" ? settings.environment : null;
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
        <span className="faint small" title={launch.job ? `job ${launch.job}` : undefined}>{said.state}</span>
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
      ) : said.reason ? <div className="small muted" title={launch.detail ?? undefined}>{said.reason}</div> : null}
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
