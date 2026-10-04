// A version: where it came from (its parents, its base model, the run and step that made it), the bookmarks that name
// it (made, moved and taken away here), its line back to the base model, and what grew from it.

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useBookmark, useKnown, useSystem, useUnbookmark } from "../api/queries";
import type { Version as VersionData } from "../api/types";
import { BarChart, Sized } from "../components/charts";
import { Marks } from "../components/versions";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { bytes, clock, figure, span } from "../lib/format";
import { runPlace, stepPlace, versionPlace } from "../lib/places";

export function Version({ id }: { id: string }) {
  const { data: system } = useSystem();
  const known = useKnown();
  const navigate = useNavigate();
  if (!system) return <Empty>Reading the version…</Empty>;
  const version = system.versions.find(each => each.id === id) ?? system.versions.find(each => each.id.startsWith(id));
  if (!version) return <Empty>There is no version {id}.</Empty>;
  const children = system.versions.filter(each => each.parents.includes(version.id));
  const line: VersionData[] = [];  // (its first parents, back to the base model)
  for (let each: VersionData | undefined = version; each; each = known.version(each.parents[0])) line.unshift(each);
  const serving = system.channels.filter(channel => channel.adapter === version.id);
  const size = (version.weights?.bytes ?? 0) + (version.state?.bytes ?? 0);
  const placeOf = (each: VersionData) => (each.run && each.step != null && system.runs.some(run => run.run === each.run) ? stepPlace(each.run, each.step) : versionPlace(each.id));
  const related: [VersionData | undefined, string, string][] = [
    ...version.parents.map((parent, place) => [known.version(parent), place ? "learned from" : "trained from", parent] as [VersionData | undefined, string, string]),
    ...children.map(each => [each, each.parents[0] === version.id ? "trained from it" : "learned from it", each.id] as [VersionData, string, string]),
  ];
  return (
    <>
      <Head title={<span className="mono" title={version.id}>{version.short}</span>}
        sub={<>{known.origin(version.id)} · depth {version.depth} · from {version.parents.length ? version.parents.map(known.short).join(" + ") : version.base ?? "the base model"}</>}>
        <Specs>
          <Spec label="id">{version.id}</Spec>
          <Spec label="base model">{version.base ?? "–"}</Spec>
          {version.run ? <Spec label="made by"><Link to={runPlace(version.run)} title={`id: ${version.run}`}>{known.run(version.run)}</Link></Spec> : <Spec label="made by">outside a run</Spec>}
          {version.run && version.step != null ? <Spec label="step" kind="violet"><Link to={stepPlace(version.run, version.step)}>S{version.step}</Link></Spec> : null}
          {serving.length ? <Spec label="served on" kind="accent">{serving.map(channel => channel.channel).join(", ")}</Spec> : null}
          <Spec label="made">{clock(version.made)}</Spec>
        </Specs>
      </Head>
      <Bookmarks version={version} all={system.bookmarks} />
      <Kpis>
        <Kpi label="Depth" value={String(version.depth)} note="steps from its base model" />
        <Kpi label="Moved" value={figure(version.metrics.kl_moved)} note="KL from its parent" />
        <Kpi label="Loss" value={figure(version.metrics.loss)} />
        <Kpi label="Took" value={span(version.metrics.update_seconds ?? version.metrics.seconds) || "–"} note={version.metrics.segments != null ? `${figure(version.metrics.segments)} segments` : ""} />
        <Kpi label="Kept" value={version.weights ? bytes(size) : "released"} note={version.weights ? "weights and trainer state" : "its record stays"} />
        <Kpi label="Grown from it" value={String(children.length)} note={children.length ? children.map(each => each.short).join(", ") : "nothing yet"} />
      </Kpis>
      <Card title="Its line" note={`from ${line[0]?.base ?? "the base model"}, first parent by first parent: how far each step moved it`}>
        {line.length > 1 ? (
          <Sized>{width => <BarChart values={line.map(each => each.metrics.kl_moved ?? 0)} labels={line.map(each => each.short)} width={width} height={150} onBar={index => navigate(placeOf(line[index]))} />}</Sized>
        ) : <p className="muted">It was trained from the base model{version.base ? ` ${version.base}` : ""}.</p>}
      </Card>
      <Card title="What it came from, and what grew from it">
        {related.length ? (
          <Table
            heads={[["version"], ["is"], ["made"], ["by"], ["moved", "n"], ["state"]]}
            keys={related.map(([, is, key]) => `${is}${key}`)}
            rows={related.map(([each, is, key]) => each ? [
              <span><b className="mono" title={known.title(each.id)}>{each.short}</b> <Marks names={each.bookmarks} /></span>, is, clock(each.made),
              known.origin(each.id), each.metrics.kl_moved?.toFixed(4) ?? "–", each.weights ? "kept" : { text: "released", kind: "still" },
            ] : [<span className="mono" title={key}>{key.slice(0, 12)}</span>, is, "–", "not in this ledger", "–", "–"])}
            to={related.map(([each]) => (each ? versionPlace(each.id) : null))}
          />
        ) : <p className="muted">It was trained from the base model, and nothing has grown from it yet.</p>}
      </Card>
    </>
  );
}

/** The bookmarks that name a version, and the controls to name it with another (a new bookmark, or one moved here from
 * the version it names now) or to take one away. */
function Bookmarks({ version, all }: { version: VersionData; all: Record<string, string> }) {
  const [name, setName] = useState("");
  const bookmark = useBookmark(), unbookmark = useUnbookmark();
  const known = useKnown();
  const elsewhere = name.trim() && all[name.trim()] && all[name.trim()] !== version.id ? all[name.trim()] : null;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    bookmark.mutate({ name: name.trim(), version: version.id }, { onSuccess: () => setName("") });
  };
  const failed = bookmark.error ?? unbookmark.error;
  return (
    <section className="card bookmarks">
      <header><h2>Bookmarks</h2><span>names for this version; a run told to carry one moves it to each version it makes</span></header>
      <div className="body">
        <div className="chips">
          {version.bookmarks.length ? version.bookmarks.map(each => (
            <span key={each} className="chip bookmark removable">
              {each}
              <button type="button" aria-label={`take the bookmark ${each} away`} title="take this bookmark away (the version stays)"
                disabled={unbookmark.isPending} onClick={() => unbookmark.mutate(each)}>×</button>
            </span>
          )) : <span className="none">none: it is known by where it came from</span>}
        </div>
        <form className="rename bookmark-form" onSubmit={submit}>
          <input value={name} onChange={event => setName(event.target.value)} placeholder="a bookmark's name" aria-label="a bookmark to name this version" spellCheck={false}
            list="bookmark-names" />
          <datalist id="bookmark-names">{Object.keys(all).map(each => <option key={each} value={each} />)}</datalist>
          <button type="submit" disabled={bookmark.isPending || !name.trim()}>{bookmark.isPending ? "saving…" : elsewhere ? "move it here" : "bookmark"}</button>
          {elsewhere ? <span className="faint small">it names {known.short(elsewhere)} ({known.origin(elsewhere)}) now</span> : null}
          {failed ? <span className="error-text small">{failed.message}</span> : null}
        </form>
      </div>
    </section>
  );
}
