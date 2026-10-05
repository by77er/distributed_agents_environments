// Pause, resume and stop a run from its page, the buttons shown by where it is: a run going can be paused, one paused
// resumed in place, one stopped, failed or lost submitted again with its settings, and one a launch plays stopped.

import { useLaunches, useRunControl, useStop } from "../api/queries";
import type { Launch, Run } from "../api/types";
import { GOING as LAUNCHING } from "../lib/launches";

const GOING = new Set(["running", "idle", "paused"]);
const RESUMABLE = new Set(["stopped", "failed", "lost", "ended"]);

/** The launch going that plays a run. */
const launchOf = (launches: Launch[] | undefined, run: Run): Launch | undefined =>
  launches?.find(each => LAUNCHING.has(each.state) && (each.run === run.run || each.asked.resumes === run.run));

export function RunControls({ run }: { run: Run }) {
  const { data: launched } = useLaunches();
  const control = useRunControl(run.run), stop = useStop();
  const launch = launchOf(launched?.launches, run);
  const going = GOING.has(run.state);
  const busy = control.isPending || stop.isPending;
  const buttons: { label: string; act: () => void; danger?: boolean }[] = [];
  if (going && !run.pause) buttons.push({ label: "pause", act: () => control.mutate("pause") });
  if (going && run.pause) buttons.push({ label: "resume", act: () => control.mutate("resume") });
  if (!going && !launch && !run.by && RESUMABLE.has(run.state)) buttons.push({ label: "resume", act: () => control.mutate("resume") });
  if (launch && launch.state !== "stopping") buttons.push({ label: "stop", act: () => stop.mutate(launch.id), danger: true });
  const error = control.error ?? stop.error;
  if (!buttons.length && !error) return null;
  return (
    <span className="run-controls">
      {buttons.map(each => (
        <button key={each.label} type="button" className={each.danger ? "action danger" : "action"} disabled={busy} onClick={each.act}>{each.label}</button>
      ))}
      {error ? <span className="error-text small">{error.message}</span> : null}
    </span>
  );
}
