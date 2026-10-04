// How a run is said wherever it appears: whether it is running (its process is there and writes), idle (there, and
// quiet) or ended, on which host when not this one, when it last wrote, its color, and where its episodes are read.

import { useCallback } from "react";
import { useSystem } from "../api/queries";
import type { Run } from "../api/types";
import { span } from "../lib/format";
import { useNow } from "../lib/now";

export const running = (run: Run, host: string): string => `${run.state}${run.host && run.host !== host ? ` on ${run.host}` : ""}`;

/** How long ago the run last wrote, counting up. */
export function Wrote({ run }: { run: { written: number | null } }) {
  const now = useNow();
  return <>{run.written ? `wrote ${span(Math.max(0, now - run.written))} ago` : "wrote nothing yet"}</>;
}

/** How long ago something happened, counting up. */
export function Ago({ at, otherwise = "–" }: { at: number | null | undefined; otherwise?: string }) {
  const now = useNow();
  return <>{at ? span(Math.max(0, now - at)) : otherwise}</>;
}

export const RunDot = ({ run, host }: { run: Run; host: string }) => (
  <span className={`dot ${run.state === "running" ? "alive" : run.state === "idle" ? "idle" : run.state === "failed" || run.state === "lost" ? "gone" : ""}`} title={running(run, host)} />
);

/** A run's color: the categorical slots in order, by the run's place among every run (so it keeps its color whatever
 * is drawn); a run past the eighth is drawn in gray. */
export function useRunColor(): (name: string) => string {
  const { data: system } = useSystem();
  const names = (system?.runs ?? []).map(run => run.run).sort().join("\n");
  return useCallback((name: string) => {
    const place = names.split("\n").indexOf(name);
    return place >= 0 && place < 8 ? `var(--series-${place + 1})` : "var(--faint)";
  }, [names]);
}

/** Where a run's episodes are read: its directory on this machine (nothing to say), the monitor on its own machine
 * (said, with a link, and whether it answers), or nowhere (what is shown is the ledger's). */
export function Elsewhere({ run }: { run: Run }) {
  if (run.episodes_at === "here") return null;
  const there = run.episodes_at;
  const address = there ? <a className="linkish" href={there} target="_blank" rel="noopener noreferrer">{there}</a> : null;
  if (there && run.reached !== false) {
    return (
      <div className="tile rail accent notice">
        <header><b>Episodes from the monitor on its machine</b><span className="what">{run.host ?? ""}</span></header>
        <p className="muted small">Asked of {address}.</p>
      </div>
    );
  }
  return (
    <div className="tile rail warm notice">
      <header><b>Read from the ledger alone</b><span className="what">{run.directory ?? ""}</span></header>
      <p className="muted small">
        {there ? <>The monitor on its machine ({address}) does not answer.</>
          : `Its directory is not on this machine${run.host ? ` (${run.host})` : ""}, and its start names no monitor.`}
      </p>
    </div>
  );
}
