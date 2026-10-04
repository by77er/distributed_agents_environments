// A checkpoint: where it came from (its parents, its base model, the run and step that made it), the bookmarks that name
// it (made, moved and taken away here), its line back to the base model, what grew from it, and the suites it played.

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useBookmark, useEvals, useKnown, useSystem, useUnbookmark } from "../api/queries";
import type { Checkpoint as CheckpointData } from "../api/types";
import { BarChart, Sized } from "../components/charts";
import { BaseName, Marks } from "../components/checkpoints";
import { Card, Empty, Head, Kpi, Kpis, Spec, Specs, Table } from "../components/ui";
import { bytes, clock, figure, span } from "../lib/format";
import { evalsPlace, runPlace, stepPlace, checkpointPlace, suitePlace } from "../lib/places";
import { shareOf, shareText } from "../components/evals";

export function Checkpoint({ id }: { id: string }) {
  const { data: system } = useSystem();
  const known = useKnown();
  const navigate = useNavigate();
  if (!system) return <Empty>Reading the checkpoint…</Empty>;
  const checkpoint = system.checkpoints.find(each => each.id === id) ?? system.checkpoints.find(each => each.id.startsWith(id));
  if (!checkpoint) return <Empty>There is no checkpoint {id}.</Empty>;
  const children = system.checkpoints.filter(each => each.parents.includes(checkpoint.id));
  const line: CheckpointData[] = [];  // (its first parents, back to the base model)
  for (let each: CheckpointData | undefined = checkpoint; each; each = known.checkpoint(each.parents[0])) line.unshift(each);
  const serving = system.channels.filter(channel => channel.adapter === checkpoint.id);
  const size = (checkpoint.weights?.bytes ?? 0) + (checkpoint.state?.bytes ?? 0);
  const placeOf = (each: CheckpointData) => (each.run && each.step != null && system.runs.some(run => run.run === each.run) ? stepPlace(each.run, each.step) : checkpointPlace(each.id));
  const related: [CheckpointData | undefined, string, string][] = [
    ...checkpoint.parents.map((parent, place) => [known.checkpoint(parent), place ? "learned from" : "trained from", parent] as [CheckpointData | undefined, string, string]),
    ...children.map(each => [each, each.parents[0] === checkpoint.id ? "trained from it" : "learned from it", each.id] as [CheckpointData, string, string]),
  ];
  return (
    <>
      <Head title={<span className="mono" title={checkpoint.id}>{checkpoint.short}</span>}>
        <Specs>
          <Spec label="id">{checkpoint.id}</Spec>
          <Spec label="weights">{checkpoint.kind === "full" ? "full" : "LoRA adapter"}</Spec>
          <Spec label={known.checkpoint(checkpoint.base) ? "over" : "base model"}>{checkpoint.base ? <BaseName base={checkpoint.base} /> : "–"}</Spec>
          {checkpoint.run ? <Spec label="made by"><Link to={runPlace(checkpoint.run)} title={`id: ${checkpoint.run}`}>{known.run(checkpoint.run)}</Link></Spec> : <Spec label="made by">outside a run</Spec>}
          {checkpoint.run && checkpoint.step != null ? <Spec label="step" kind="violet"><Link to={stepPlace(checkpoint.run, checkpoint.step)}>S{checkpoint.step}</Link></Spec> : null}
          {serving.length ? <Spec label="served on" kind="accent">{serving.map(channel => channel.channel).join(", ")}</Spec> : null}
          <Spec label="made">{clock(checkpoint.made)}</Spec>
        </Specs>
      </Head>
      <Bookmarks checkpoint={checkpoint} all={system.bookmarks} />
      <Kpis>
        <Kpi label="Depth" value={String(checkpoint.depth)} />
        <Kpi label="Moved" value={figure(checkpoint.metrics.kl_moved)} />
        <Kpi label="Loss" value={figure(checkpoint.metrics.loss)} />
        <Kpi label="Took" value={span(checkpoint.metrics.update_seconds ?? checkpoint.metrics.seconds) || "–"} note={checkpoint.metrics.segments != null ? `${figure(checkpoint.metrics.segments)} segments` : ""} />
        <Kpi label="Kept" value={checkpoint.weights ? bytes(size) : "released"} />
        <Kpi label="Grown from it" value={String(children.length)} note={children.map(each => each.short).join(", ")} />
      </Kpis>
      <Card title="Its line" note={`from ${known.base(line[0]?.base)}`}>
        {line.length > 1 ? (
          <Sized>{width => <BarChart values={line.map(each => each.metrics.kl_moved ?? 0)} labels={line.map(each => each.short)} width={width} height={150} onBar={index => navigate(placeOf(line[index]))} />}</Sized>
        ) : <p className="muted">It was trained from {checkpoint.base ? <BaseName base={checkpoint.base} /> : "the base model"}.</p>}
      </Card>
      <Card title="What it came from, and what grew from it">
        {related.length ? (
          <Table
            heads={[["checkpoint"], ["is"], ["made"], ["by"], ["moved", "n"], ["state"]]}
            keys={related.map(([, is, key]) => `${is}${key}`)}
            rows={related.map(([each, is, key]) => each ? [
              <span><b className="mono" title={known.title(each.id)}>{each.short}</b> <Marks names={each.bookmarks} /></span>, is, clock(each.made),
              known.origin(each.id), each.metrics.kl_moved?.toFixed(4) ?? "–", each.weights ? "kept" : { text: "released", kind: "still" },
            ] : [<span className="mono" title={key}>{key.slice(0, 12)}</span>, is, "–", "not in this ledger", "–", "–"])}
            to={related.map(([each]) => (each ? checkpointPlace(each.id) : null))}
          />
        ) : <p className="muted">None.</p>}
      </Card>
      <Played id={checkpoint.id} />
    </>
  );
}

/** Every suite a checkpoint played, and how it did at each: one opens the suite, beside every other subject's. */
function Played({ id }: { id: string }) {
  const { data: evals } = useEvals();
  const played = (evals?.suites ?? []).flatMap(suite =>
    suite.subjects.filter(subject => subject.checkpoint === id).map(subject => ({ suite, subject })));
  return (
    <Card title="Evals" note={<Link to={evalsPlace} className="linkish">play one with it</Link>}>
      {played.length ? (
        <Table
          heads={[["suite"], ["starts", "n"], ["played", "n"], ["solved", "n"], ["mean reward", "n"]]}
          keys={played.map(({ subject }) => subject.subject)}
          rows={played.map(({ suite, subject }) => [
            <b>{suite.suite}</b>, String(suite.starts.length), `${subject.played} of ${suite.starts.length * (subject.episodes ?? 1)}`,
            shareText(shareOf(subject)), figure(subject.reward),
          ])}
          to={played.map(({ suite }) => suitePlace(suite.suite))}
        />
      ) : <p className="muted">None yet.</p>}
    </Card>
  );
}

/** The bookmarks that name a checkpoint, and the controls to name it with another (a new bookmark, or one moved here from
 * the checkpoint it names now) or to take one away. */
function Bookmarks({ checkpoint, all }: { checkpoint: CheckpointData; all: Record<string, string> }) {
  const [name, setName] = useState("");
  const bookmark = useBookmark(), unbookmark = useUnbookmark();
  const known = useKnown();
  const elsewhere = name.trim() && all[name.trim()] && all[name.trim()] !== checkpoint.id ? all[name.trim()] : null;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    bookmark.mutate({ name: name.trim(), checkpoint: checkpoint.id }, { onSuccess: () => setName("") });
  };
  const failed = bookmark.error ?? unbookmark.error;
  return (
    <section className="card bookmarks">
      <header><h2>Bookmarks</h2></header>
      <div className="body">
        <div className="chips">
          {checkpoint.bookmarks.length ? checkpoint.bookmarks.map(each => (
            <span key={each} className="chip bookmark removable">
              {each}
              <button type="button" aria-label={`take the bookmark ${each} away`} title="take this bookmark away (the checkpoint stays)"
                disabled={unbookmark.isPending} onClick={() => unbookmark.mutate(each)}>×</button>
            </span>
          )) : <span className="none">none</span>}
        </div>
        <form className="rename bookmark-form" onSubmit={submit}>
          <input value={name} onChange={event => setName(event.target.value)} placeholder="a bookmark's name" aria-label="a bookmark to name this checkpoint" spellCheck={false}
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
