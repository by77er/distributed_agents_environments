// Picking an environment: a select of those the system knows, each by its name, with its `module:name` under the select.
// And importing one from git: its URL, ref, subdirectory and entry point, with the import's stage while the monitor
// makes it, or why it was refused.

import { useState } from "react";
import { useEnvironments, useImport, useImports } from "../api/queries";
import type { EnvironmentVersion } from "../api/types";
import { pickable, stageText } from "../lib/environments";
import { Card } from "./ui";

/** The form that imports an environment from git; `onDone` is given the version once the monitor recorded it. */
export function ImportForm({ onDone, onCancel }: { onDone: (version: EnvironmentVersion) => void; onCancel: () => void }) {
  const importing = useImport();
  const [url, setUrl] = useState("");
  const [ref, setRef] = useState("");
  const [subdirectory, setSubdirectory] = useState("");
  const [entry, setEntry] = useState("");
  const { data: imports } = useImports(importing.isPending);
  const going = importing.isPending ? imports?.imports.find(each => each.url === url.trim() && each.ended == null) : undefined;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const asked = { url: url.trim(), ref: ref.trim(), subdirectory: subdirectory.trim(), entry_point: entry.trim() };
    importing.mutate(asked, { onSuccess: made => onDone(made.version) });
  };
  return (
    <form onSubmit={submit} className="import-form">
      <Card title="Import from git">
        <div className="fields">
          <label className="field">
            <span>URL</span>
            <input value={url} onChange={event => setUrl(event.target.value)} required spellCheck={false} disabled={importing.isPending} />
          </label>
          <div className="field-row">
            <label className="field">
              <span>Ref</span>
              <input value={ref} onChange={event => setRef(event.target.value)} placeholder="default branch" spellCheck={false} disabled={importing.isPending} />
            </label>
            <label className="field">
              <span>Subdirectory</span>
              <input value={subdirectory} onChange={event => setSubdirectory(event.target.value)} spellCheck={false} disabled={importing.isPending} />
            </label>
            <label className="field">
              <span>Entry point</span>
              <input value={entry} onChange={event => setEntry(event.target.value)} placeholder="module:name" spellCheck={false} disabled={importing.isPending} />
            </label>
          </div>
        </div>
        <div className="launch-submit">
          <button type="submit" disabled={importing.isPending || !url.trim()}>{importing.isPending ? `${stageText(going?.stage)}…` : "Import"}</button>
          <button type="button" className="linkish" onClick={onCancel}>cancel</button>
          {importing.isError ? <span className="error-text" role="alert">{importing.error.message}</span> : null}
        </div>
      </Card>
    </form>
  );
}

interface PickerProps {
  value: string;
  onChange: (environment: string) => void;
  /** For a launch: those no launcher alive offers cannot be picked. */
  launching?: boolean;
  /** Environments to list besides those the system knows (the one picked, say). */
  also?: (string | null | undefined)[];
  /** Environments not to list (those another entry has). */
  without?: string[];
  label?: string;
}

/** A select of environments: offered by a launcher alive first, then those none offers (marked so). */
export function EnvironmentPicker({ value, onChange, launching = false, also = [], without = [], label = "environment" }: PickerProps) {
  const { data: known } = useEnvironments();
  const listed = pickable(known ?? [], [...also, value]).filter(each => each.environment === value || !without.includes(each.environment));
  const offered = listed.filter(each => each.offered), others = listed.filter(each => !each.offered);
  const option = (each: (typeof listed)[number], disabled = false) => (
    <option key={each.environment} value={each.environment} disabled={disabled} title={each.environment}>{each.name}</option>
  );
  return (
    <>
      <select value={value} onChange={event => onChange(event.target.value)} aria-label={label} required>
        {value ? null : <option value="" disabled>{known ? "choose" : "reading…"}</option>}
        {offered.length && others.length ? <optgroup label="offered">{offered.map(each => option(each))}</optgroup> : offered.map(each => option(each))}
        {others.length ? <optgroup label="no launcher offers">{others.map(each => option(each, launching))}</optgroup> : null}
      </select>
      {value ? <small className="mono" title={value}>{value}</small> : null}
    </>
  );
}
