// The queue: a bar for each resource the cluster gives runs, its capacity split into each admitted run's share (in the
// run's color, linked to it), the runs that wait, in the queue's order, with what each asks for, how long it has
// waited and why, and the pods the platform rents on RunPod (outside the queue's capacity), each with its run.

import { memo } from "react";
import { Link } from "react-router-dom";
import { useKnown, useQueue } from "../api/queries";
import type { Queue, QueuePending, QueuePod } from "../api/types";
import { Ago, useRunColor } from "../layout/runs";
import { bytes } from "../lib/format";
import { amount, type Bar, barOf, cell, pendingOf, queued, RESOURCES, waitsFor } from "../lib/queue";
import { runPlace } from "../lib/places";
import { Card, Empty, SectionTitle } from "./ui";

const percent = (share: number) => `${(100 * share).toFixed(3)}%`;

/** What a bar says beside its name: what is used of its capacity. */
function used(bar: Bar): string {
  if (bar.resource === "memory") return `${bytes(bar.used)} of ${bytes(bar.capacity)}`;
  return `${+bar.used.toPrecision(3)} of ${amount(bar.resource, bar.capacity)}`;
}

/** A resource's bar: each admitted run's segment, sized by its share of the capacity, in its color, linked to it. */
export const QueueBar = memo(function QueueBar({ bar, name }: { bar: Bar; name: string }) {
  const colorOf = useRunColor();
  const known = useKnown();
  return (
    <div className="queue-bar">
      <div className="queue-bar-head"><span>{name}</span><b>{used(bar)}</b></div>
      <div className="queue-track">
        {bar.segments.map((segment, place) => {
          const said = `${segment.run ? known.run(segment.run) : segment.name} · ${amount(bar.resource, segment.amount)}`;
          const style = { left: percent(segment.start), width: percent(segment.width), background: segment.run ? colorOf(segment.run) : "var(--faint)" };
          return segment.run ? (
            <Link key={`${segment.run}:${place}`} to={runPlace(segment.run)} className="queue-segment" style={style} title={said} aria-label={said} />
          ) : <span key={`${segment.name}:${place}`} className="queue-segment" style={style} title={said} />;
        })}
        {bar.other ? (
          <span className="queue-segment other" style={{ left: percent(bar.other.start), width: percent(bar.other.width) }}
            title={`held outside the runs · ${amount(bar.resource, bar.other.width * bar.capacity)}`} />
        ) : null}
      </div>
    </div>
  );
});

/** What a run asks for, in words: `1 GPU · 2 CPUs · 4.0 GiB`. */
const asks = (pending: QueuePending) =>
  (["gpu", "cpu", "memory"] as const).filter(key => pending.requests[key]).map(key => amount(key, pending.requests[key]!)).join(" · ") || "–";

/** The runs that wait, in the queue's order: each a row of its place, its name (linked to it), what it asks for, how
 * long it has waited and why; on a narrow page, each a block. */
export function PendingList({ queue }: { queue: Queue }) {
  const known = useKnown();
  const name = (run: string | null) => (run ? known.run(run) : "another job");
  return (
    <div className="queue-pending" role="table" aria-label="waiting">
      <div className="queue-row head" role="row">
        <span role="columnheader" className="n">#</span><span role="columnheader">run</span>
        {RESOURCES.map(([key, label]) => <span key={key} role="columnheader" className="n wide">{label}</span>)}
        <span role="columnheader" className="n">waited</span><span role="columnheader">why</span>
      </div>
      {queue.pending.map(pending => (
        <div key={pending.workload ?? pending.run ?? pending.position} className="queue-row" role="row">
          <span role="cell" className="n position">{pending.position}</span>
          <span role="cell" className="run">
            {pending.run ? <Link to={runPlace(pending.run)}>{known.run(pending.run)}</Link> : pending.name}
            <span className="asks narrow">{asks(pending)}</span>
          </span>
          {RESOURCES.map(([key]) => <span key={key} role="cell" className="n wide">{cell(key, pending.requests[key])}</span>)}
          <span role="cell" className="n waited"><Ago at={pending.since} /></span>
          <span role="cell" className="why" title={pending.reason ?? undefined}>{waitsFor(pending, queue, name)}</span>
        </div>
      ))}
    </div>
  );
}

/** Where a run waits in the queue and why, in words (`2nd in the queue · waits for 1 GPU: in use by run alpha`); none
 * where it does not wait there. */
export function useQueued(run: string | null | undefined): string | null {
  const { data: queue } = useQueue();
  const known = useKnown();
  const pending = run ? pendingOf(queue, run) : undefined;
  if (!queue || !pending) return null;
  return queued(pending, queue, each => (each ? known.run(each) : "another job"));
}

/** The queue's section: what the runs hold, and who waits; nothing where neither Kueue nor Ray says anything. */
export function QueueSection() {
  const { data: queue } = useQueue();
  if (!queue || queue.source == null) return null;
  const bars = RESOURCES.map(([key, label]) => [barOf(queue, key), label] as const).filter((each): each is [Bar, string] => each[0] != null);
  return (
    <>
      <SectionTitle id="section-queue" title="Queue" />
      {queue.error ? <div className="error-text">{queue.error}</div> : null}
      <div className="cols">
        <Card title="Held">
          {bars.length ? <div className="queue-bars">{bars.map(([bar, label]) => <QueueBar key={bar.resource} bar={bar} name={label} />)}</div> : null}
          {!bars.length && queue.admitted.length ? <Holding queue={queue} /> : null}
          {queue.admitted.length ? null : <Empty>No runs hold resources.</Empty>}
        </Card>
        <Card title="Waiting">
          {queue.pending.length ? <PendingList queue={queue} /> : <Empty>Nothing waits.</Empty>}
        </Card>
      </div>
      {queue.pods?.length ? <Card title="Pods"><PodList pods={queue.pods} /></Card> : null}
    </>
  );
}

/** The pods the platform rents on RunPod: each its run (linked to it; `warm` for one no run holds), GPU, price, and
 * what its run has been charged for it. */
export function PodList({ pods }: { pods: QueuePod[] }) {
  const known = useKnown();
  const colorOf = useRunColor();
  return (
    <div className="queue-holding" role="table" aria-label="pods">
      {pods.map(each => (
        <div key={each.pod} className="queue-row" role="row">
          <span className="swatch" style={{ background: each.run ? colorOf(each.run) : "var(--faint)" }} />
          <span role="cell">{each.run ? <Link to={runPlace(each.run)}>{known.run(each.run)}</Link> : each.state === "idle" ? "warm" : each.state}</span>
          <span role="cell" title={each.pod}>{each.gpu}</span>
          <span role="cell" className="n">${Number(each.price.toFixed(2))}/h</span>
          <span role="cell" className="n">{each.run ? `$${each.spent.toFixed(2)}` : <Ago at={each.since} />}</span>
        </div>
      ))}
    </div>
  );
}

/** What each admitted run holds, where the capacity is not known (no bar to split). */
function Holding({ queue }: { queue: Queue }) {
  const known = useKnown();
  const colorOf = useRunColor();
  return (
    <div className="queue-holding">
      {queue.admitted.map(each => (
        <div key={each.workload ?? each.run ?? each.name} className="queue-row">
          <span className="swatch" style={{ background: each.run ? colorOf(each.run) : "var(--faint)" }} />
          {each.run ? <Link to={runPlace(each.run)}>{known.run(each.run)}</Link> : <span>{each.name}</span>}
          <span className="n">{(["gpu", "cpu", "memory"] as const).filter(key => each.requests[key]).map(key => amount(key, each.requests[key]!)).join(" · ")}</span>
        </div>
      ))}
    </div>
  );
}
