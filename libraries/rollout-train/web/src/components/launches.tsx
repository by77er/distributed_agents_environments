// The runs and evals asked for from the page, and how each goes: asked, claimed by a launcher, running, and how it
// finished.

import { memo, useState } from "react";
import { Link } from "react-router-dom";
import { useStop } from "../api/queries";
import type { Launch, System } from "../api/types";
import { Ago } from "../layout/runs";
import { span } from "../lib/format";
import { evalPlace, runPlace, suitePlace } from "../lib/places";
import { suiteName, versionTag } from "../lib/suites";
import { Mark, SectionTitle, Tile } from "./ui";
import { CheckpointTag } from "./checkpoints";

const GOING = new Set(["asked", "claimed", "running", "stopping"]);
const FINISHED_SHOWN = 6;

/** Every launch going, and the newest that finished. */
export function LaunchList({ launches, system }: { launches: Launch[]; system: System }) {
  const [all, setAll] = useState(false);
  const going = launches.filter(each => GOING.has(each.state));
  const finished = launches.filter(each => !GOING.has(each.state));
  const shown = [...going, ...(all ? finished : finished.slice(0, FINISHED_SHOWN))];
  const hidden = finished.length - (all ? finished.length : Math.min(finished.length, FINISHED_SHOWN));
  return (
    <>
      <SectionTitle id="section-launches" title="Launches" note={`${going.length} going · ${finished.length} finished`} />
      <div className="tiles launches">
        {shown.map(launch => <LaunchTile key={launch.id} launch={launch} system={system} />)}
      </div>
      {hidden > 0 || all ? (
        <button type="button" className="linkish small more" onClick={() => setAll(!all)}>{all ? "show fewer" : `and ${hidden} more finished`}</button>
      ) : null}
    </>
  );
}

const rail: Record<string, string> = { asked: "warm", claimed: "accent", running: "accent", stopping: "warm", ended: "good", failed: "bad", stopped: "" };

const LaunchTile = memo(function LaunchTile({ launch, system }: { launch: Launch; system: System }) {
  const stop = useStop();
  const asked = launch.asked;
  // (its run, once it is in the ledger: the one called what the launch named it, started from its directory)
  const run = system.runs.find(each => (launch.directory && each.directory === launch.directory) || each.name === asked.name);
  const changed = Object.entries(asked.settings ?? {});
  return (
    <Tile className={`rail ${rail[launch.state] ?? ""}`}>
      <header>
        <b>{run ? <Link to={run.kind === "eval" ? evalPlace(run.run) : runPlace(run.run)} className="linkish">{asked.name}</Link> : asked.name}</b>
        <Mark state={launch.state} />
      </header>
      <div className="facts wraps"><span>{asked.profile}</span><span className="mono small">{asked.environment}</span></div>
      {asked.kind === "eval" ? (
        <div className="facts">
          <span>plays <Link to={suitePlace(suiteName(asked.suite ?? ""))} className="linkish">{suiteName(asked.suite ?? "")}</Link>{asked.suite?.includes("@") ? <span className="tag-version">{versionTag(asked.suite)}</span> : null} with <CheckpointTag id={asked.start ?? null} /></span>
          <span>{asked.episodes ?? 1} episode{(asked.episodes ?? 1) === 1 ? "" : "s"} a start</span>
        </div>
      ) : (
        <div className="facts">
          <span>from <CheckpointTag id={asked.start ?? null} /></span>
          {asked.bookmark ? <span>carries <span className="chip bookmark">{asked.bookmark}</span></span> : null}
          <span>{asked.groups ?? 100} groups · {asked.groups_per_step ?? 4} a step · seed {asked.seed ?? 0}</span>
        </div>
      )}
      {changed.length ? (
        <div className="facts settings-changed">
          {changed.map(([key, value]) => <span key={key} className="mono small"><b>{key}</b> = {JSON.stringify(value)}</span>)}
        </div>
      ) : null}
      <div className="facts">
        <span>asked <Ago at={launch.at} /> ago</span>
        {launch.launcher ? <span>by <b>{launch.launcher}</b></span> : <span className="t-warm">no launcher has claimed it</span>}
        {launch.pid ? <span className="mono">pid {launch.pid}</span> : null}
        {GOING.has(launch.state) ? null : <span>took {span(Math.max(0, launch.updated - launch.at))}</span>}
      </div>
      {launch.directory ? <div className="mono small faint launch-directory" title={launch.directory}>{launch.directory}</div> : null}
      {launch.detail ? (
        launch.state === "failed" ? (
          <details className="launch-detail">
            <summary>why it failed</summary>
            <pre>{launch.detail}</pre>
          </details>
        ) : launch.detail !== launch.state ? <div className="small muted">{launch.detail}</div> : null
      ) : null}
      {GOING.has(launch.state) && launch.state !== "stopping" ? (
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
