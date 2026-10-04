// A training run's settings: the changeable ones, which it takes from its next step on once changed here, and the
// fixed ones, shown as they are.

import { useState } from "react";
import { useEvals, useRunSettings, useWantSettings } from "../api/queries";
import type { RunSettings } from "../api/types";
import { EVALS_SUITE, GROUPS_PER_STEP, shown, wantedOf } from "../lib/settings";
import { Card, Pairs, SectionTitle, Table } from "./ui";

const FIRST = [GROUPS_PER_STEP, EVALS_SUITE, "evals.every", "evals.episodes"];

/** The changeable settings in the order a form shows them: the loop's, the evals', then the trainer's by name. */
export const changeableKeys = (settings: RunSettings): string[] =>
  [...FIRST.filter(key => key in settings.changeable), ...Object.keys(settings.changeable).filter(key => !FIRST.includes(key)).sort()];

/** A run's settings, where its start says them: a form for the changeable ones, and the fixed ones as they are. */
export function RunSettingsSection({ run }: { run: string }) {
  const { data: settings } = useRunSettings(run);
  if (!settings || !Object.keys(settings.changeable).length) return null;
  return (
    <>
      <SectionTitle id="section-settings" title="Settings" />
      <div className="cols">
        <SettingsForm settings={settings} />
        <Card title="Fixed">
          <Pairs entries={Object.entries(settings.fixed).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => [key, <span className="mono">{shown(value) || "–"}</span>])} />
        </Card>
      </div>
    </>
  );
}

function SettingsForm({ settings }: { settings: RunSettings }) {
  const want = useWantSettings(settings.run);
  const { data: evals } = useEvals();
  const [fields, setFields] = useState<Record<string, string>>({});
  const current = { ...settings.now, ...settings.desired };  // (what it is wanted to use, else what it uses)
  const keys = changeableKeys(settings);
  const { settings: wanted, errors } = wantedOf(fields, current);
  const count = Object.keys(wanted).length;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    want.mutate(wanted, { onSuccess: () => setFields({}) });
  };
  const suites = (evals?.suites ?? []).map(each => each.suite);  // (a run plays each environment of its evals' suite on its channel)
  return (
    <form onSubmit={submit}>
      <Card title="Changeable">
        <div className="settings-grid">
          {keys.map(key => {
            const value = fields[key] ?? shown(current[key]);
            const pending = key in settings.desired && shown(settings.desired[key]) !== shown(settings.now[key]);
            const changed = key in wanted;
            return (
              <label key={key} className={`setting${changed || pending ? " changed" : ""}`}>
                <span className="mono">{key}</span>
                {key === EVALS_SUITE ? (
                  <select value={value} onChange={event => setFields({ ...fields, [key]: event.target.value })}>
                    <option value="">none</option>
                    {[...new Set([...suites, ...(value ? [value] : [])])].map(each => <option key={each} value={each}>{each}</option>)}
                  </select>
                ) : (
                  <input value={value} onChange={event => setFields({ ...fields, [key]: event.target.value })} placeholder="not set" spellCheck={false} />
                )}
                {errors[key] ? <small className="error-text">{errors[key]}</small>
                  : pending && !changed ? <small className="faint">from its next step · now {shown(settings.now[key]) || "none"}</small> : null}
              </label>
            );
          })}
        </div>
        <div className="launch-submit">
          <button type="submit" disabled={want.isPending || !count || Object.keys(errors).length > 0}>{want.isPending ? "saving…" : "Save"}</button>
          {count ? <button type="button" className="linkish" onClick={() => setFields({})}>reset</button> : null}
          {want.isError ? <span className="error-text">{want.error.message}</span> : null}
        </div>
        {settings.changes.length ? (
          <Table
            heads={[["step"], ["changed"]]}
            keys={settings.changes.map(change => change.step)}
            rows={settings.changes.map(change => [
              { text: `S${change.step}`, kind: "key" },
              <span className="mono small">{Object.entries(change.changed).map(([key, value]) => `${key} = ${shown(value) || "none"}`).join(" · ")}</span>,
            ])}
          />
        ) : null}
      </Card>
    </form>
  );
}
