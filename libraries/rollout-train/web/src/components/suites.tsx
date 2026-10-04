// The form that makes a suite, or its next version: its name (a new suite's), and its entries, each an environment with
// how its starts are chosen, the episodes of each start and the limits of its episodes. Entries are added and taken away.

import { useRef, useState } from "react";
import { useEnvironment, useSaveSuite } from "../api/queries";
import type { SuiteVersion } from "../api/types";
import { readable } from "../lib/environments";
import { DRAWN, EVAL_DATA, type EntryFields, fieldsOf, GIVEN, SAME, suiteBody, versionTag } from "../lib/suites";
import { EnvironmentPicker } from "./environments";
import { Card } from "./ui";

/** Rows shown as boxes to tick; an environment with more is given its rows' keys as text. */
const TICKED = 40;

interface FormProps {
  title: string;
  /** The version edited (its suite's newest): the form makes the next. None: a new suite. */
  version?: SuiteVersion;
  name?: string;
  /** A new suite's first entry's environment. */
  environment?: string;
  onDone?: (version: string) => void;
  onCancel?: () => void;
}

export function SuiteForm({ title, version, name: fixedName, environment, onDone, onCancel }: FormProps) {
  const editing = version != null;
  const [name, setName] = useState(fixedName ?? "");
  const made = useRef(0);
  const keyed = (fields: EntryFields) => ({ key: (made.current += 1), fields });
  const [entries, setEntries] = useState<{ key: number; fields: EntryFields }[]>(() =>
    (version ? version.entries.map(entry => fieldsOf(entry, version)) : [fieldsOf(undefined, undefined, DRAWN, environment)]).map(keyed));
  const save = useSaveSuite(name.trim());
  const { body, errors, suite } = suiteBody(entries.map(each => each.fields), version?.number);
  const change = (key: number, fields: EntryFields) => setEntries(entries.map(each => (each.key === key ? { key, fields } : each)));
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    save.mutate(body, { onSuccess: made => onDone?.(made.version) });
  };
  const ready = name.trim() && !suite && errors.every(each => !Object.keys(each).length);
  return (
    <form onSubmit={submit} className="suite-form">
      <Card title={title} note={editing ? `${versionTag(version.id)} → v${version.number + 1}` : undefined}>
        <div className="fields">
          {editing ? null : (
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={event => setName(event.target.value)} required spellCheck={false} />
            </label>
          )}
          {entries.map(({ key, fields }, place) => (
            <EntryForm key={key} fields={fields} errors={errors[place]} version={version}
              others={entries.filter(each => each.key !== key).map(each => each.fields.environment)}
              onChange={changed => change(key, changed)}
              onRemove={entries.length > 1 ? () => setEntries(entries.filter(each => each.key !== key)) : undefined} />
          ))}
          <div>
            <button type="button" className="linkish" onClick={() => setEntries([...entries, keyed(fieldsOf(undefined, undefined, DRAWN))])}>+ environment</button>
          </div>
        </div>
        <div className="launch-submit">
          <button type="submit" disabled={save.isPending || !ready}>{save.isPending ? "saving…" : editing ? "Save version" : "Make the suite"}</button>
          {onCancel ? <button type="button" className="linkish" onClick={onCancel}>cancel</button> : null}
          {suite && entries.length ? <span className="error-text">{suite}</span> : null}
          {save.isError ? <span className="error-text">{save.error.message}</span> : null}
        </div>
      </Card>
    </form>
  );
}

interface EntryProps {
  fields: EntryFields;
  errors: Record<string, string>;
  /** The version edited, whose entry of the same environment an entry may keep the starts of. */
  version?: SuiteVersion;
  /** The other entries' environments (an environment is in a version once). */
  others: string[];
  onChange: (fields: EntryFields) => void;
  onRemove?: () => void;
}

/** One entry: its environment, its starts, episodes and limits. */
function EntryForm({ fields, errors, version, others, onChange, onRemove }: EntryProps) {
  const environment = fields.environment.trim();
  const { data: described, error: unloaded } = useEnvironment(environment);
  const [rowsText, setRowsText] = useState(fields.rows.join(", "));
  const before = version?.entries.find(each => each.environment === environment);
  const many = (described?.rows.length ?? 0) > TICKED;
  const evalData = Object.keys(described?.evals ?? {});
  const set = (change: Partial<EntryFields>) => onChange({ ...fields, ...change });
  const pick = (picked: string) => {
    const kept = version?.entries.find(each => each.environment === picked);
    onChange(kept ? fieldsOf(kept, version) : { ...fieldsOf(undefined, undefined, DRAWN, picked), episodes: fields.episodes });
    setRowsText(kept?.rows?.join(", ") ?? "");
  };
  return (
    <fieldset className="entry">
      <legend>
        {environment ? readable(environment) : "environment"}
        {onRemove ? <button type="button" className="linkish" onClick={onRemove} aria-label={`take ${readable(environment)} away`}>remove</button> : null}
      </legend>
      <div className="field-row">
        <label className="field">
          <span>Environment</span>
          <EnvironmentPicker value={environment} onChange={pick} without={others} also={version?.environments ?? []} />
          {unloaded && environment ? <small className="error-text">does not load here</small> : null}
        </label>
        <label className="field">
          <span>Starts</span>
          <select value={fields.chosen} onChange={event => set({ chosen: event.target.value as EntryFields["chosen"], evalData: fields.evalData || evalData[0] || "" })}>
            {before && version ? <option value={SAME}>same as {versionTag(version.id)} ({before.starts})</option> : null}
            <option value={EVAL_DATA} disabled={!evalData.length}>eval data</option>
            <option value={DRAWN}>rows and seeds</option>
            <option value={GIVEN}>starts</option>
          </select>
        </label>
        <label className="field">
          <span>Episodes per start</span>
          <input type="number" min={1} value={fields.episodes} onChange={event => set({ episodes: event.target.value })} />
          {errors.episodes ? <small className="error-text">{errors.episodes}</small> : null}
        </label>
        <label className="field">
          <span>Thinking tokens</span>
          <input type="number" min={1} value={fields.thinking} onChange={event => set({ thinking: event.target.value })} placeholder="channel's" />
          {errors.thinking ? <small className="error-text">{errors.thinking}</small> : null}
        </label>
        <label className="field">
          <span>Answer tokens</span>
          <input type="number" min={1} value={fields.answer} onChange={event => set({ answer: event.target.value })} placeholder="channel's" />
          {errors.answer ? <small className="error-text">{errors.answer}</small> : null}
        </label>
      </div>
      {fields.chosen === EVAL_DATA ? (
        <label className="field">
          <span>Eval data</span>
          <select value={fields.evalData} onChange={event => set({ evalData: event.target.value })}>
            {evalData.map(each => <option key={each} value={each}>{each} · {described?.evals[each]} starts</option>)}
          </select>
        </label>
      ) : null}
      {fields.chosen === DRAWN ? (
        <>
          <div className="field">
            <span>Rows</span>
            {many ? (
              <input value={rowsText} onChange={event => { setRowsText(event.target.value); set({ rows: event.target.value.split(/[\s,]+/).filter(Boolean) }); }}
                placeholder="every row" spellCheck={false} />
            ) : (
              <div className="ticks">
                {(described?.rows ?? []).map(row => (
                  <label key={row.key} className="tick" title={row.title}>
                    <input type="checkbox" checked={fields.rows.includes(row.key)}
                      onChange={event => set({ rows: event.target.checked ? [...fields.rows, row.key] : fields.rows.filter(each => each !== row.key) })} />
                    {row.key}
                  </label>
                ))}
                {!fields.rows.length ? <small className="faint">every row</small> : null}
              </div>
            )}
          </div>
          <label className="field">
            <span>Seeds</span>
            <input value={fields.seeds} onChange={event => set({ seeds: event.target.value })} placeholder="1, 2, 3" spellCheck={false} />
            {errors.seeds && fields.seeds.trim() ? <small className="error-text">{errors.seeds}</small> : null}
          </label>
        </>
      ) : null}
      {fields.chosen === GIVEN ? (
        <label className="field">
          <span>Starts</span>
          <textarea rows={Math.min(10, Math.max(3, fields.starts.split("\n").length))} value={fields.starts} onChange={event => set({ starts: event.target.value })} placeholder="row seed" spellCheck={false} />
          {errors.starts && fields.starts.trim() ? <small className="error-text">{errors.starts}</small> : null}
        </label>
      ) : null}
    </fieldset>
  );
}
