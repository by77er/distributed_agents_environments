// The frame every place is shown in: the pages along the top, the hierarchy on the left, where one is in a bar over
// the content, and whether the monitor is heard from. It stays put as places change and as what they show is read
// again: only the views over what changed draw.

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { type Topic, topics, useEpisode, useEvals, useKnown, useSystem } from "../api/queries";
import type { System } from "../api/types";
import { useConnection, useStream } from "../api/stream";
import { readable } from "../lib/environments";
import { madeBy, nameOf, stepOf } from "../lib/model";
import { episodePlace, groupPlace, PAGES, type Place, presetsPlace, runPlace, stepPlace, suitePlace, usePlace } from "../lib/places";
import { Tree } from "./Tree";

/** The topics the place shown needs the monitor to say it changed. */
function watched(place: Place, system: System | undefined): Topic[] {
  const found = [topics.system(), topics.feeds()];
  if (place.kind === "run") {
    const newest = system ? madeBy(system.checkpoints, place.run).at(-1) : undefined;
    found.push(topics.settings(place.run), ...(newest ? [topics.path(newest.id)] : []), topics.launches(), topics.queue());
  }
  if (place.kind === "checkpoint") {
    const id = system?.checkpoints.find(each => each.id === place.id || each.id.startsWith(place.id))?.id ?? place.id;
    found.push(topics.checkpointEvals(id), topics.path(id), topics.launches(), topics.offers());
  }
  if (place.kind === "base") found.push(topics.checkpoints(), topics.history("model", place.model), topics.launches(), topics.offers(), topics.evals());
  if (place.kind === "launch") found.push(topics.evals());
  if (place.kind === "presets") found.push(topics.presets());
  if (place.kind === "preset") found.push(topics.preset(place.preset));
  if (place.kind === "group") found.push(topics.group(place.run, place.number));
  if (place.kind === "episode") found.push(topics.episode(place.id));
  if (place.kind === "statistics") found.push(topics.statistics());
  if (place.page === "machines") found.push(topics.machines());
  if (place.kind === "machines" || place.kind === "runs") found.push(topics.queue());
  if (place.kind === "runs" || place.kind === "launch") found.push(topics.launches(), topics.offers());
  if (place.kind === "checkpoints") found.push(topics.checkpoints());
  if (place.page === "evals" || place.kind === "checkpoint") found.push(topics.evals());
  if (place.page === "evals") found.push(topics.launches(), topics.offers(), topics.evalSubjects());
  if (place.kind === "subject") {
    const id = place.subject === "checkpoint" ? system?.checkpoints.find(each => each.id === place.id || each.id.startsWith(place.id))?.id ?? place.id : place.id;
    found.push(topics.history(place.subject, place.id), ...(place.subject === "checkpoint" ? [topics.path(id)] : []));
  }
  if (place.page === "environments") found.push(topics.environments());
  if (place.kind === "environments") found.push(topics.imports());
  if (place.kind === "environment") found.push(topics.environment(place.environment));
  return found;
}

export function Shell({ children }: { children: ReactNode }) {
  const place = usePlace();
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const main = useRef<HTMLElement>(null);
  const { data: system } = useSystem();
  useStream(watched(place, system));
  // A new place opens at its top (or at the section it names); what is read again keeps where one is.
  useEffect(() => {
    setOpen(false);
    const section = place.kind === "statistics" ? place.section : null;
    const role = place.kind === "host" ? place.role : null;
    const target = section ? document.getElementById(`section-${section}`) : role ? document.getElementById(`role-${role}`) : null;
    if (target) target.scrollIntoView({ block: "start" });
    else main.current?.scrollTo({ top: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);
  return (
    <div className={`shell${open ? " open" : ""}`}>
      <header className="top">
        <button type="button" className="menu" aria-label="show the sidebar" onClick={() => setOpen(!open)}>☰</button>
        <Link className="brand" to="/runs" title={system?.ledger_at ?? undefined} aria-label="Rollout">
          <span className="wordmark" aria-hidden="true"><span className="speed"><i /><i /><i /></span>ROLLOUT</span>
        </Link>
        <nav className="pages" aria-label="pages">
          {PAGES.map(([page, name, to]) => {
            const count = page === "runs" ? system?.runs.filter(run => run.kind !== "eval").length : page === "checkpoints" ? system?.checkpoints.length
              : page === "evals" ? system?.runs.filter(run => run.kind === "eval" && !run.part_of).length : null;
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

/** Nothing while the monitor is heard from; a notice while it is not. */
function Live() {
  const { data: system, isError } = useSystem();
  const connection = useConnection();
  if (isError && !system) return <div className="live"><span className="dot gone" />cannot reach the monitor</div>;
  if (!system) return <div className="live">connecting…</div>;
  if (connection === "lost") return <div className="live" title="not hearing from the monitor: trying again"><span className="dot gone" />reconnecting</div>;
  return null;
}

/** Where one is: the page, then each place above the one shown. */
/** The tab's title: the place shown, and where it is when that alone says little ("Step 3" of which run). */
export function titleOf(crumbs: [string, string][]): string {
  const [last] = crumbs[crumbs.length - 1];
  const within = crumbs.length > 2 && /^(Step|Group|Episode|Rollout) /.test(last) ? crumbs[1][0] : null;
  return [last, within, "Rollout"].filter(Boolean).join(" · ");
}

function Crumbs({ place }: { place: Place }) {
  const { data: system } = useSystem();
  const { data: episode } = useEpisode(place.kind === "episode" ? place.id : "", place.kind === "episode");
  const { data: evals } = useEvals(place.kind === "eval");
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
  else if (place.kind === "presets") crumbs.push(["Presets", ""]);
  else if (place.kind === "preset") crumbs.push(["Presets", presetsPlace], [`Preset ${place.preset}`, ""]);
  else if (place.kind === "checkpoint") crumbs.push([`Checkpoint ${known.short(place.id)}`, ""]);
  else if (place.kind === "base") crumbs.push([`Base model ${place.model}`, ""]);
  else if (place.kind === "suite") crumbs.push([`Suite ${place.suite}`, ""]);
  else if (place.kind === "subject") crumbs.push([place.subject === "checkpoint" ? `Checkpoint ${known.short(place.id)}` : `Base model ${place.id}`, ""]);
  else if (place.kind === "host") crumbs.push([place.host, ""]);
  else if (place.kind === "environment") crumbs.push([readable(place.environment), ""]);
  else if (place.kind === "eval") {
    const played = evals?.evals.find(each => each.run === place.run);
    if (played) crumbs.push([`Suite ${played.suite}`, suitePlace(played.suite)]);
    crumbs.push([`Eval ${played?.name ?? runName(place.run)}`, ""]);
  }
  const title = titleOf(crumbs);
  useEffect(() => { document.title = title; }, [title]);
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
