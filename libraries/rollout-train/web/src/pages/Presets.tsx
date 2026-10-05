// Presets: named run settings in versions. The list of every preset's newest version, and a preset's page: each
// version's settings with what it changed from the one before, an editor that saves the next version, and deleting it.

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDeletePreset, usePreset, usePresets, useSavePreset } from "../api/queries";
import type { Preset } from "../api/types";
import { Card, Empty, Head, Table } from "../components/ui";
import { Ago } from "../layout/runs";
import { launchPlace, presetPlace, presetsPlace } from "../lib/places";
import { diffKeys } from "../lib/form";
import { shown, typed } from "../lib/settings";

export function Presets() {
  const { data } = usePresets();
  if (!data) return <Empty>Reading the presets…</Empty>;
  return (
    <>
      <Head title={<span className="head-with-action">Presets<Link to={launchPlace} className="action">New run</Link></span>} />
      {data.presets.length ? (
        <Table
          heads={[["preset"], ["version"], ["settings", "n"], ["note"], ["saved"]]}
          rows={data.presets.map(each => [<Link to={presetPlace(each.name)} className="linkish">{each.name}</Link>, `v${each.version}${each.versions > 1 ? ` of ${each.versions}` : ""}`, Object.keys(each.settings).length, each.note || "–", <><Ago at={each.saved} /> ago</>])}
          to={data.presets.map(each => presetPlace(each.name))}
          keys={data.presets.map(each => each.name)}
        />
      ) : <Empty>{data.keeps ? "No preset yet." : "This ledger keeps no presets beside it."}</Empty>}
    </>
  );
}

interface Row {
  key: string;
  value: string;
}

const rowsOf = (preset: Preset): Row[] => Object.keys(preset.settings).sort().map(key => ({ key, value: shown(preset.settings[key]) }));

export function PresetPage({ name }: { name: string }) {
  const { data, error } = usePreset(name);
  const navigate = useNavigate();
  const remove = useDeletePreset();
  const [chosen, setChosen] = useState<number | null>(null);
  const [editing, setEditing] = useState<Row[] | null>(null);
  if (error) return <Empty>There is no preset {name}.</Empty>;
  if (!data) return <Empty>Reading the preset…</Empty>;
  const newest = data.versions.at(-1)!;
  const version = data.versions.find(each => each.version === chosen) ?? newest;
  const before = data.versions.filter(each => each.version < version.version).at(-1);
  const changed = before ? diffKeys(before.settings, version.settings) : new Set<string>();
  const removed = before ? Object.keys(before.settings).filter(key => !(key in version.settings)).sort() : [];
  const deleting = () => {
    if (window.confirm(`Delete the preset ${name}? Its versions stay readable by number.`)) remove.mutate(name, { onSuccess: () => navigate(presetsPlace) });
  };
  return (
    <>
      <Head title={<span className="head-with-action">{name}
        <button type="button" className="action" onClick={() => setEditing(rowsOf(version))} disabled={editing != null}>Edit</button>
        <button type="button" className="action danger" onClick={deleting} disabled={remove.isPending}>Delete</button>
      </span>} />
      {remove.isError ? <p className="error-text" role="alert">{remove.error.message}</p> : null}
      {editing ? (
        <Editor name={name} rows={editing} next={newest.version + 1} onRows={setEditing} onDone={() => { setEditing(null); setChosen(null); }} />
      ) : (
        <Card title={
          <select className="picker" value={version.version} onChange={event => setChosen(Number(event.target.value))} aria-label="version">
            {[...data.versions].reverse().map(each => <option key={each.version} value={each.version}>v{each.version}{each.note ? ` · ${each.note}` : ""}</option>)}
          </select>
        } note={<><Ago at={version.saved} /> ago</>}>
          <Table
            heads={[["setting"], ["value"]]}
            rows={Object.keys(version.settings).sort().map(key => [{ text: <span className="mono">{key}</span>, kind: changed.has(key) ? "changed" : undefined }, <span className="mono">{shown(version.settings[key]) || "none"}</span>])}
            keys={Object.keys(version.settings).sort()}
          />
          {removed.length ? <p className="small muted">removed: <span className="mono">{removed.join(", ")}</span></p> : null}
        </Card>
      )}
    </>
  );
}

/** The editor of a preset's settings: a row a setting; saving makes the next version. */
function Editor({ name, rows, next, onRows, onDone }: { name: string; rows: Row[]; next: number; onRows: (rows: Row[]) => void; onDone: () => void }) {
  const save = useSavePreset();
  const [note, setNote] = useState("");
  const keys = rows.map(row => row.key.trim()).filter(Boolean);
  const twice = keys.find((key, at) => keys.indexOf(key) !== at);
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const settings = Object.fromEntries(rows.filter(row => row.key.trim()).map(row => [row.key.trim(), row.value.trim() === "" ? null : typed(row.value)]));
    save.mutate({ name, settings, note: note.trim() }, { onSuccess: onDone });
  };
  return (
    <form onSubmit={submit}>
      <Card title={`v${next}`}>
        <div className="settings-grid">
          {rows.map((row, index) => (
            <div key={index} className="setting extra">
              <input className="mono" value={row.key} onChange={event => onRows(rows.map((each, at) => (at === index ? { ...each, key: event.target.value } : each)))} placeholder="KEY" spellCheck={false} aria-label="setting" />
              <input value={row.value} onChange={event => onRows(rows.map((each, at) => (at === index ? { ...each, value: event.target.value } : each)))} placeholder="none" spellCheck={false} aria-label={`${row.key} value`} />
              <button type="button" className="linkish" onClick={() => onRows(rows.filter((_, at) => at !== index))} aria-label={`remove ${row.key}`}>remove</button>
            </div>
          ))}
          <button type="button" className="linkish" onClick={() => onRows([...rows, { key: "", value: "" }])}>+ add a setting</button>
        </div>
        <label className="field">
          <span>Note</span>
          <input value={note} onChange={event => setNote(event.target.value)} spellCheck={false} />
        </label>
        <div className="launch-submit">
          <button type="submit" disabled={save.isPending || twice != null}>{save.isPending ? "saving…" : `Save v${next}`}</button>
          <button type="button" className="linkish" onClick={onDone}>cancel</button>
          {twice ? <span className="error-text">{twice} is given twice</span> : null}
          {save.isError ? <span className="error-text" role="alert">{save.error.message}</span> : null}
        </div>
      </Card>
    </form>
  );
}
