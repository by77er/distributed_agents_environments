// The form that makes a suite, or its next version: its name and environment (a new suite's), how its starts are
// chosen, the episodes of each start, and the limits of its evals' channel.

import { useState } from "react";
import { useEnvironment, useSaveSuite } from "../api/queries";
import type { SuiteVersion } from "../api/types";
import { DRAWN, EVAL_DATA, fieldsOf, GIVEN, SAME, type SuiteFields, suiteBody, versionTag } from "../lib/suites";
import { Card } from "./ui";

/** Rows shown as boxes to tick; an environment with more is given its rows' keys as text. */
const TICKED = 40;

interface FormProps {
  title: string;
  /** The version edited (its suite's newest): the form makes the next. None: a new suite. */
  version?: SuiteVersion;
  name?: string;
  /** Environments to offer a new suite (`module:name`). */
  environments?: string[];
  onDone?: (version: string) => void;
  onCancel?: () => void;
}

export function SuiteForm({ title, version, name: fixedName, environments = [], onDone, onCancel }: FormProps) {
  const editing = version != null;
  const [name, setName] = useState(fixedName ?? "");
  const [environment, setEnvironment] = useState(version?.environment ?? environments[0] ?? "");
  const [fields, setFields] = useState<SuiteFields>(() => fieldsOf(version, editing ? SAME : DRAWN));
  const [rowsText, setRowsText] = useState((version?.rows ?? []).join(", "));
  const { data: described, error: unloaded } = useEnvironment(environment.trim());
  const save = useSaveSuite(name.trim());
  const many = (described?.rows.length ?? 0) > TICKED;
  const rows = many ? rowsText.split(/[\s,]+/).filter(Boolean) : fields.rows;
  const { body, errors } = suiteBody({ ...fields, rows }, version?.number);
  const set = (change: Partial<SuiteFields>) => setFields({ ...fields, ...change });
  const evalData = Object.keys(described?.evals ?? {});
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    save.mutate(editing ? body : { ...body, environment: environment.trim() }, { onSuccess: made => onDone?.(made.version) });
  };
  const ready = name.trim() && environment.trim() && !Object.keys(errors).length;
  return (
    <form onSubmit={submit} className="suite-form">
      <Card title={title} note={editing ? `${versionTag(version.id)} → v${version.number + 1}` : undefined}>
        <div className="fields">
          {editing ? null : (
            <div className="field-row">
              <label className="field">
                <span>Name</span>
                <input value={name} onChange={event => setName(event.target.value)} required spellCheck={false} />
              </label>
              <label className="field">
                <span>Environment</span>
                <input value={environment} onChange={event => setEnvironment(event.target.value)} placeholder="module:name" list="suite-environments" required spellCheck={false} />
                <datalist id="suite-environments">{environments.map(each => <option key={each} value={each} />)}</datalist>
                {unloaded && environment.trim() ? <small className="error-text">does not load here</small> : null}
              </label>
            </div>
          )}
          <div className="field-row">
            <label className="field">
              <span>Starts</span>
              <select value={fields.chosen} onChange={event => set({ chosen: event.target.value as SuiteFields["chosen"], evalData: fields.evalData || evalData[0] || "" })}>
                {editing ? <option value={SAME}>same as {versionTag(version.id)} ({version.starts.length})</option> : null}
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
                  <input value={rowsText} onChange={event => setRowsText(event.target.value)} placeholder="every row" spellCheck={false} />
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
        </div>
        <div className="launch-submit">
          <button type="submit" disabled={save.isPending || !ready}>{save.isPending ? "saving…" : editing ? "Save version" : "Make the suite"}</button>
          {onCancel ? <button type="button" className="linkish" onClick={onCancel}>cancel</button> : null}
          {save.isError ? <span className="error-text">{save.error.message}</span> : null}
        </div>
      </Card>
    </form>
  );
}
