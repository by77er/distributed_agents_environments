// The frame every place is shown in: the pages along the top, the hierarchy on the left, where one is in a bar over
// the content, and whether the monitor is heard from. It stays put as places change and as what they show is read
// again: only the views over what changed draw.

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { type Topic, topics, useEpisode, useKnown, useSystem } from "../api/queries";
import { useConnection, useStream } from "../api/stream";
import { span } from "../lib/format";
import { nameOf, stepOf } from "../lib/model";
import { useNow } from "../lib/now";
import { episodePlace, groupPlace, PAGES, type Place, runPlace, stepPlace, usePlace } from "../lib/places";
import { Tree } from "./Tree";

/** The topics the place shown needs the monitor to say it changed. */
function watched(place: Place): Topic[] {
  const found = [topics.system(), topics.feeds()];
  if (place.kind === "group") found.push(topics.group(place.run, place.number));
  if (place.kind === "episode") found.push(topics.episode(place.id));
  if (place.kind === "statistics") found.push(topics.statistics(), topics.machines());
  if (place.kind === "runs" || place.kind === "launch") found.push(topics.launches());
  if (place.kind === "checkpoints") found.push(topics.checkpoints(place.sample));
  return found;
}

export function Shell({ children }: { children: ReactNode }) {
  const place = usePlace();
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const main = useRef<HTMLElement>(null);
  useStream(watched(place));
  const { data: system } = useSystem();
  // A new place opens at its top (or at the section it names); what is read again keeps where one is.
  useEffect(() => {
    setOpen(false);
    const section = place.kind === "statistics" ? place.section : null;
    const target = section ? document.getElementById(`section-${section}`) : null;
    if (target) target.scrollIntoView({ block: "start" });
    else main.current?.scrollTo({ top: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);
  return (
    <div className={`shell${open ? " open" : ""}`}>
      <header className="top">
        <button type="button" className="menu" aria-label="show the sidebar" onClick={() => setOpen(!open)}>☰</button>
        <Link className="brand" to="/runs">
          <div className="mark-logo">R</div>
          <div style={{ minWidth: 0 }}><b>Runs monitor</b><small>{system?.ledger_at ?? "connecting…"}</small></div>
        </Link>
        <nav className="pages" aria-label="pages">
          {PAGES.map(([page, name, to]) => {
            const count = page === "runs" ? system?.runs.length : page === "checkpoints" ? system?.checkpoints.length : null;
            return (
              <Link key={page} to={to} className={place.page === page ? "current" : undefined} aria-current={place.page === page ? "page" : undefined}>
                {name}{count != null ? <span className="count">{count}</span> : null}
              </Link>
            );
          })}
        </nav>
        <Live />
      </header>
      <div className="frame">
        <aside><Tree place={place} /></aside>
        <div className="content">
          <div className="bar"><Crumbs place={place} /></div>
          <main ref={main}>
            <div className={`page${place.kind === "episode" ? " wide" : ""}`}>{children}</div>
          </main>
        </div>
      </div>
    </div>
  );
}

/** Whether runs are running, when anything was last written, and whether the monitor is heard from. */
function Live() {
  const { data: system, isError } = useSystem();
  const connection = useConnection();
  const now = useNow();
  if (isError && !system) return <div className="live"><span className="dot gone" />cannot reach the monitor</div>;
  if (!system) return <div className="live">connecting…</div>;
  const busy = system.runs.filter(run => run.state === "running").length;
  return (
    <div className="live" title={connection === "live" ? "the monitor says when anything changes" : "not hearing from the monitor: trying again"}>
      <span className={`dot ${connection === "lost" ? "gone" : busy ? "alive" : ""}`} />
      {connection === "lost" ? "reconnecting · " : ""}
      {busy} of {system.runs.length} run{system.runs.length === 1 ? "" : "s"} running
      {system.written ? ` · wrote ${span(Math.max(0, now - system.written))} ago` : ""}
    </div>
  );
}

/** Where one is: the page, then each place above the one shown. */
function Crumbs({ place }: { place: Place }) {
  const { data: system } = useSystem();
  const { data: episode } = useEpisode(place.kind === "episode" ? place.id : "", place.kind === "episode");
  const known = useKnown();
  const [, pageName, pageTo] = PAGES.find(([page]) => page === place.page)!;
  const crumbs: [string, string][] = [[pageName, pageTo]];
  const runName = (key: string) => {
    const run = system?.runs.find(each => each.run === key);
    return run ? nameOf(run) : key;
  };
  const stepCrumb = (run: string, number: number): [string, string][] => {
    const found = system?.runs.find(each => each.run === run);
    const step = found && stepOf(found, number);
    return step ? [[`Step ${step.step}`, stepPlace(run, step.step)]] : [];
  };
  if (place.kind === "run") crumbs.push([`Run ${runName(place.run)}`, runPlace(place.run)]);
  else if (place.kind === "step") crumbs.push([`Run ${runName(place.run)}`, runPlace(place.run)], [`Step ${place.number}`, ""]);
  else if (place.kind === "group") crumbs.push([`Run ${runName(place.run)}`, runPlace(place.run)], ...stepCrumb(place.run, place.number), [`Group #${place.number}`, ""]);
  else if (place.kind === "episode") {
    const labels = episode?.labels ?? {};
    if (labels.run && labels.group) {
      crumbs.push([`Run ${runName(labels.run)}`, runPlace(labels.run)], ...stepCrumb(labels.run, Number(labels.group)),
        [`Group #${Number(labels.group)}`, groupPlace(labels.run, Number(labels.group))]);
    } else crumbs.push(["Episodes outside a run", "/episodes"]);
    crumbs.push([labels.episode ? `Episode ${labels.episode}` : "Episode", place.slot ? episodePlace(place.id) : ""]);
    if (place.slot) crumbs.push([`Rollout ${place.slot}`, ""]);
  } else if (place.kind === "outside") crumbs.push(["Episodes outside a run", ""]);
  else if (place.kind === "launch") crumbs.push(["New run", ""]);
  else if (place.kind === "checkpoint") crumbs.push([`Checkpoint ${known.short(place.id)}`, ""]);
  else if (place.kind === "checkpoints" && place.sample) crumbs.push(["Sample fixture", ""]);
  return (
    <div className="crumbs">
      {crumbs.map(([name, to], index) => (
        <span key={index} className="crumb">
          {index ? <span className="sep">/</span> : null}
          {index === crumbs.length - 1 || !to ? <b>{name}</b> : <Link to={to}>{name}</Link>}
        </span>
      ))}
    </div>
  );
}
