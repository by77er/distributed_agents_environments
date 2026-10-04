// A new training run, asked for from the page: a launcher alive on a training machine offers its profiles (each with
// the settings a launch may change), and starts the run asked for on one of them.

import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useKnown, useLaunch, useLaunches, useSystem } from "../api/queries";
import type { Launcher, OfferedProfile } from "../api/types";
import { Card, Empty, Head, SectionTitle } from "../components/ui";
import { Ago } from "../layout/runs";
import { nameOf } from "../lib/model";

/** A setting's value as typed: a number, a boolean, JSON (a list, a table, a quoted string), or else the text. */
export function typed(text: string): unknown {
  const trimmed = text.trim();
  if (trimmed === "") return "";
  if (trimmed === "true" || trimmed === "false") return trimmed === "true";
  if (/^-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(trimmed)) return Number(trimmed);
  if (/^[[{"]/.test(trimmed)) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return text;
    }
  }
  return text;
}

const shown = (value: unknown): string =>
  value == null ? "" : typeof value === "string" ? value : JSON.stringify(value);

/** What weights are, in a word or two: full, or a LoRA adapter. */
const weightsText = (kind: string): string => (kind === "full" ? "full weights" : kind === "lora" ? "LoRA" : kind);

export function NewRun() {
  const { data: launched } = useLaunches();
  const { data: system } = useSystem();
  if (!launched || !system) return <Empty>Reading what can be launched…</Empty>;
  const launchers = launched.launchers;
  return (
    <>
      <Head title="New run" />
      {launchers.length ? <Form launchers={launchers} /> : <NoLauncher ledger={system.ledger_at} />}
    </>
  );
}

export function NoLauncher({ ledger }: { ledger: string }) {
  const command = `rollout launcher --ledger ${ledger.includes("://") ? ledger : `${ledger}`} --profiles DIR --environment module:name --runs DIR`;
  return (
    <Card title="No launcher is alive">
      <pre className="command">{command}</pre>
    </Card>
  );
}

interface Row {
  key: string;
  value: string;
}

function Form({ launchers }: { launchers: Launcher[] }) {
  const navigate = useNavigate();
  const launch = useLaunch();
  const known = useKnown();
  const { data: system } = useSystem();
  const offered = useMemo(() => {
    const byName = new Map<string, { profile: OfferedProfile; launchers: Launcher[] }>();
    for (const launcher of launchers) {
      for (const profile of launcher.profiles ?? []) {
        const found = byName.get(profile.profile);
        if (found) found.launchers.push(launcher);
        else byName.set(profile.profile, { profile, launchers: [launcher] });
      }
    }
    return [...byName.values()];
  }, [launchers]);
  const environments = useMemo(() => [...new Set(launchers.flatMap(each => each.environments ?? []))], [launchers]);
  const [profileName, setProfileName] = useState(offered[0]?.profile.profile ?? "");
  const chosen = offered.find(each => each.profile.profile === profileName) ?? offered[0];
  const [environment, setEnvironment] = useState(environments[0] ?? "");
  const [name, setName] = useState("");
  const [start, setStart] = useState("");
  const [bookmark, setBookmark] = useState("");
  const [groups, setGroups] = useState("100");
  const [perStep, setPerStep] = useState("4");
  const [seed, setSeed] = useState("0");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [rows, setRows] = useState<Row[]>([]);
  const defaults = chosen?.profile.settings ?? {};
  const settingKeys = Object.keys(defaults).filter(key => key !== "trainer.start" && key !== "trainer.bookmark").sort();
  const room = chosen ? chosen.launchers.some(each => (each.playing ?? 0) < (each.at_once ?? 1)) : false;
  const checkpoints = [...(system?.checkpoints ?? [])].sort((a, b) => b.made - a.made);
  const bookmarks = Object.keys(system?.bookmarks ?? {}).sort();
  const full = chosen?.profile.weights === "full";  // (a trainer of every weight starts from full weights, not an adapter)
  const startable = (id: string, profile = chosen?.profile) => {
    const checkpoint = known.checkpoint(id);
    return checkpoint?.weights != null && !(profile?.weights === "full" && checkpoint.kind !== "full");
  };

  const changed = (): Record<string, unknown> => {
    const settings: Record<string, unknown> = {};
    for (const [key, text] of Object.entries(edits)) {
      if (!(key in defaults)) continue;
      const value = typed(text);
      if (shown(value) !== shown(defaults[key])) settings[key] = value;
    }
    for (const row of rows) if (row.key.trim()) settings[row.key.trim()] = typed(row.value);
    return settings;
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!chosen) return;
    launch.mutate(
      {
        profile: chosen.profile.profile, environment, name: name.trim(), start: start || null, bookmark: bookmark.trim() || null,
        groups: Number(groups), groups_per_step: Number(perStep), seed: Number(seed), settings: changed(),
      },
      { onSuccess: () => navigate("/runs") },
    );
  };

  const count = Object.keys(changed()).length;
  return (
    <form className="launch-form" onSubmit={submit}>
      <div className="cols">
        <Card title="The run">
          <div className="fields">
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={event => setName(event.target.value)} placeholder="a name for the run" required spellCheck={false} />
            </label>
            <label className="field">
              <span>Profile</span>
              <select value={chosen?.profile.profile ?? ""} onChange={event => {
                const next = offered.find(each => each.profile.profile === event.target.value)?.profile;
                const from = start && system?.bookmarks[start] ? system.bookmarks[start] : start;
                setProfileName(event.target.value); setEdits({});
                if (from && !startable(from, next)) setStart("");
              }}>
                {offered.map(each => <option key={each.profile.profile} value={each.profile.profile}>{each.profile.profile} · {each.profile.model}{each.profile.weights ? ` · ${weightsText(each.profile.weights)}` : ""}</option>)}
              </select>
              <small>offered by {chosen?.launchers.map(each => each.launcher).join(", ")}{room ? "" : " · all busy: it waits for room"}</small>
            </label>
            <label className="field">
              <span>Environment</span>
              {environments.length ? (
                <select value={environment} onChange={event => setEnvironment(event.target.value)}>
                  {environments.map(each => <option key={each} value={each}>{each}</option>)}
                </select>
              ) : <input value={environment} onChange={event => setEnvironment(event.target.value)} placeholder="module:name" required spellCheck={false} />}
            </label>
            <label className="field">
              <span>Starts from</span>
              <select value={start} onChange={event => setStart(event.target.value)}>
                <option value="">the base model{chosen ? ` (${chosen.profile.model})` : ""}</option>
                {bookmarks.length ? (
                  <optgroup label="Bookmarks">
                    {bookmarks.map(mark => {
                      const id = system!.bookmarks[mark], checkpoint = known.checkpoint(id);
                      return (
                        <option key={`b${mark}`} value={mark} disabled={checkpoint !== undefined && !startable(id)}>
                          {mark} · {known.origin(id)} · {known.short(id)}{checkpoint ? ` · ${weightsText(checkpoint.kind)}` : ""}
                        </option>
                      );
                    })}
                  </optgroup>
                ) : null}
                {checkpoints.length ? (
                  <optgroup label="Checkpoints, newest first">
                    {checkpoints.map(checkpoint => (
                      <option key={checkpoint.id} value={checkpoint.id} disabled={!startable(checkpoint.id)}>
                        {known.origin(checkpoint.id)} · {checkpoint.short} · {weightsText(checkpoint.kind)}{checkpoint.bookmarks.length ? ` [${checkpoint.bookmarks.join(", ")}]` : ""}{checkpoint.weights == null ? " (released)" : ""}
                      </option>
                    ))}
                  </optgroup>
                ) : null}
              </select>
              {full ? <small>its trainer trains every weight: an adapter is merged first (rollout merge)</small> : null}
            </label>
            <label className="field">
              <span>Carries a bookmark</span>
              <input value={bookmark} onChange={event => setBookmark(event.target.value)} placeholder="optional: moved to each checkpoint it makes" spellCheck={false} />
            </label>
            <div className="field-row">
              <label className="field"><span>Groups</span><input type="number" min={1} value={groups} onChange={event => setGroups(event.target.value)} /></label>
              <label className="field"><span>Groups a step</span><input type="number" min={1} value={perStep} onChange={event => setPerStep(event.target.value)} /></label>
              <label className="field"><span>Seed</span><input type="number" value={seed} onChange={event => setSeed(event.target.value)} /></label>
            </div>
          </div>
        </Card>
        <Card title="Settings" note={count ? `${count} changed` : ""}>
          <div className="settings-grid">
            {settingKeys.map(key => {
              const value = edits[key] ?? shown(defaults[key]);
              const isChanged = key in edits && shown(typed(edits[key])) !== shown(defaults[key]);
              return (
                <label key={key} className={`setting${isChanged ? " changed" : ""}`}>
                  <span className="mono">{key}</span>
                  <input value={value} onChange={event => setEdits({ ...edits, [key]: event.target.value })} placeholder="not set" spellCheck={false} />
                  {isChanged ? <button type="button" className="linkish" onClick={() => { const next = { ...edits }; delete next[key]; setEdits(next); }}>reset to {shown(defaults[key]) || "not set"}</button> : null}
                </label>
              );
            })}
            {settingKeys.length ? null : <p className="muted small">This profile offers no settings.</p>}
          </div>
          <SectionTitle title="More trainer settings" />
          <div className="settings-grid">
            {rows.map((row, index) => (
              <div key={index} className="setting extra">
                <input className="mono" value={row.key} onChange={event => setRows(rows.map((each, at) => (at === index ? { ...each, key: event.target.value } : each)))} placeholder="trainer.KEY" spellCheck={false} aria-label="setting" />
                <input value={row.value} onChange={event => setRows(rows.map((each, at) => (at === index ? { ...each, value: event.target.value } : each)))} placeholder="value" spellCheck={false} aria-label="value" />
                <button type="button" className="linkish" onClick={() => setRows(rows.filter((_, at) => at !== index))} aria-label="remove this setting">remove</button>
              </div>
            ))}
            <button type="button" className="linkish" onClick={() => setRows([...rows, { key: "trainer.", value: "" }])}>+ add a setting</button>
          </div>
        </Card>
      </div>
      <div className="launch-submit">
        <button type="submit" disabled={launch.isPending || !name.trim() || !environment}>{launch.isPending ? "asking…" : "Launch the run"}</button>
        <Link to="/runs" className="linkish">cancel</Link>
        {launch.isError ? <span className="error-text">{launch.error.message}</span> : null}
        {!launch.isError ? <span className="small faint">launchers alive: {launchers.map(each => <span key={each.launcher}>{each.launcher} (beat <Ago at={each.at} /> ago, {each.playing ?? 0} of {each.at_once ?? 1} playing) </span>)}</span> : null}
      </div>
      {system && system.runs.some(run => nameOf(run) === name.trim()) ? <p className="error-text small">another run is called {name.trim()}</p> : null}
    </form>
  );
}
