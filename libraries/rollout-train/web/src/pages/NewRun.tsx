// A new training run, asked for from the page: its settings start from a preset's (each a default the form may
// change), with the environment the cluster offers, where it starts, and the evals it makes of its checkpoints: a suite,
// or none, said so. The monitor checks the settings against the cluster and submits the run's job; what it refuses is
// said beside the field it is about.

import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Refused, useEvals, useKnown, useLaunch, useOffers, useSystem } from "../api/queries";
import type { EvalSuite, Offers, SettingFinding } from "../api/types";
import { EnvironmentPicker } from "../components/environments";
import { Card, Empty, Head, SectionTitle } from "../components/ui";
import { nameOf } from "../lib/model";
import { EVALS_EPISODES, EVALS_EVERY, EVALS_SUITE, evalsSettings, NO_EVALS, shown, typed } from "../lib/settings";
import { currentOf, versionTag } from "../lib/suites";

export { typed };
const EVALS = [EVALS_SUITE, EVALS_EVERY, EVALS_EPISODES];
/** Settings the form has a field of its own for. */
const OWN = new Set(["environment", "start", "bookmark", "groups", "groups_per_step", "seed", "name", "kind", ...EVALS]);
const LOOP: Record<string, number> = { groups: 100, groups_per_step: 4, seed: 0 };

/** What weights are, in a word or two: full, or a LoRA adapter. */
const weightsText = (kind: string): string => (kind === "full" ? "full weights" : kind === "lora" ? "LoRA" : kind);

/** Whether a setting is a channel's thinking or answer budget, which a launch may set to "none" (no budget). */
const isBudget = (key: string): boolean => /^channels\.[^.]+\.(thinking|answer)_tokens$/.test(key);
/** What an unset setting is: "none" for a channel's budget, else "not set". */
const unsetText = (key: string): string => (isBudget(key) ? "none" : "not set");

export function NewRun() {
  const { data: offers } = useOffers();
  const { data: system } = useSystem();
  if (!offers || !system) return <Empty>Reading what can be asked for…</Empty>;
  return (
    <>
      <Head title="New run" />
      {offers.cluster ? <Form offers={offers} /> : <NoCluster />}
    </>
  );
}

export function NoCluster() {
  return <Empty>This monitor asks for no runs without a cluster config: <span className="mono">rollout monitor --cluster</span>.</Empty>;
}

interface Row {
  key: string;
  value: string;
}

/** The refusals said of one field. */
function Said({ refusals, keys }: { refusals: SettingFinding[]; keys: string[] }) {
  const found = refusals.filter(each => keys.includes(each.key));
  return found.length ? <small className="error-text">{found.map(each => each.reason).join("; ")}</small> : null;
}

function Form({ offers }: { offers: Offers }) {
  const navigate = useNavigate();
  const launch = useLaunch();
  const known = useKnown();
  const { data: system } = useSystem();
  const { data: evals } = useEvals();
  const [asked] = useSearchParams();  // (`?environment=module:name`, `?model=NAME`: the form opened from that page)
  const model = asked.get("model");
  const matching = model ? offers.presets.find(each => each.settings["channels.policy.model"] === model) : undefined;
  const environments = useMemo(() => offers.environments.map(each => each.environment), [offers]);
  const [presetId, setPresetId] = useState(matching?.id ?? (model ? "" : offers.presets[0]?.id ?? ""));
  const preset = offers.presets.find(each => each.id === presetId);
  const defaults = preset?.settings ?? {};
  const [environment, setEnvironment] = useState(asked.get("environment") ?? (typeof defaults.environment === "string" ? defaults.environment : environments[0] ?? ""));
  const [name, setName] = useState("");
  const [start, setStart] = useState("");
  const [bookmark, setBookmark] = useState("");
  const [loop, setLoop] = useState<Record<string, string>>({});
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [rows, setRows] = useState<Row[]>(model && !matching ? [{ key: "channels.policy.model", value: model }] : []);
  const [suite, setSuite] = useState<string | null>(null);  // (none chosen: the preset's)
  const [every, setEvery] = useState<string | null>(null);
  const [episodes, setEpisodes] = useState<string | null>(null);
  const offered = new Set(environments);
  const playable = (each: EvalSuite) => currentOf(each).environments.every(environment => !environment || offered.has(environment));
  const suites = (evals?.suites ?? []).filter(playable);
  const chosenSuite = suite ?? shown(defaults[EVALS_SUITE]);
  const evalsAsked = evalsSettings(chosenSuite, every ?? (shown(defaults[EVALS_EVERY]) || "1"), episodes ?? shown(defaults[EVALS_EPISODES]));
  const evaluating = Boolean(chosenSuite) && chosenSuite !== NO_EVALS;
  const suiteEpisodes = suites.find(each => each.suite === chosenSuite);
  const settingKeys = Object.keys(defaults).filter(key => !OWN.has(key)).sort();
  const checkpoints = [...(system?.checkpoints ?? [])].sort((a, b) => b.made - a.made);
  const bookmarks = Object.keys(system?.bookmarks ?? {}).sort();
  const trainer = offers.trainers.find(each => each.name === defaults["trainer.provider"]);
  const full = trainer?.produces === "full";  // (a trainer of every weight starts from full weights, not an adapter)
  const startable = (id: string, produces = trainer?.produces) => {
    const checkpoint = known.checkpoint(id);
    return checkpoint?.weights != null && !(produces === "full" && checkpoint.kind !== "full");
  };
  const loopValue = (key: string) => loop[key] ?? shown(defaults[key] ?? LOOP[key]);
  const refusals = launch.error instanceof Refused ? launch.error.refusals : [];
  const fielded = new Set(["name", "environment", "start", "bookmark", "groups", "groups_per_step", "seed", ...EVALS, ...settingKeys, ...rows.map(row => row.key.trim())]);
  const elsewhere = refusals.filter(each => !fielded.has(each.key));

  const changed = (): Record<string, unknown> => {
    const settings: Record<string, unknown> = {};
    for (const [key, text] of Object.entries(edits)) {
      if (!(key in defaults)) continue;
      const value = !text.trim() && isBudget(key) ? null : typed(text);  // (a budget emptied: none, no budget)
      if (shown(value) !== shown(defaults[key])) settings[key] = value;
    }
    for (const row of rows) if (row.key.trim()) settings[row.key.trim()] = typed(row.value);
    return { ...settings, ...evalsAsked.settings };  // (the evals, always said: a suite, or none)
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const settings = changed();
    if (start) settings.start = start;
    if (bookmark.trim()) settings.bookmark = bookmark.trim();
    for (const key of Object.keys(LOOP)) {
      const value = Number(loopValue(key));
      if (shown(value) !== shown(defaults[key] ?? LOOP[key])) settings[key] = value;
    }
    launch.mutate({ kind: "train", name: name.trim(), environment, preset: preset?.id ?? null, settings }, { onSuccess: () => navigate("/runs") });
  };

  const count = Object.keys(changed()).filter(key => !EVALS.includes(key)).length;
  return (
    <form className="launch-form" onSubmit={submit}>
      <div className="cols">
        <Card title="The run">
          <div className="fields">
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={event => setName(event.target.value)} placeholder="a name for the run" required spellCheck={false} />
              <Said refusals={refusals} keys={["name"]} />
            </label>
            <label className="field">
              <span>Preset</span>
              <select value={presetId} onChange={event => {
                const next = offers.presets.find(each => each.id === event.target.value);
                const produces = offers.trainers.find(each => each.name === next?.settings["trainer.provider"])?.produces;
                const from = start && system?.bookmarks[start] ? system.bookmarks[start] : start;
                setPresetId(event.target.value); setEdits({}); setLoop({}); setSuite(null); setEvery(null); setEpisodes(null);
                if (from && !startable(from, produces)) setStart("");
              }}>
                <option value="">no preset</option>
                {offers.presets.map(each => {
                  const model = each.settings["channels.policy.model"];
                  const produces = offers.trainers.find(trainer => trainer.name === each.settings["trainer.provider"])?.produces;
                  return <option key={each.id} value={each.id}>{each.id}{typeof model === "string" ? ` · ${model}` : ""}{produces ? ` · ${weightsText(produces)}` : ""}</option>;
                })}
              </select>
            </label>
            <label className="field">
              <span>Environment</span>
              <EnvironmentPicker value={environment} onChange={picked => { setEnvironment(picked); setSuite(null); }} launching={environments.length > 0} also={environments} />
              <Said refusals={refusals} keys={["environment"]} />
            </label>
            <label className="field">
              <span>Starts from</span>
              <select value={start} onChange={event => setStart(event.target.value)}>
                <option value="">the base model{typeof defaults["channels.policy.model"] === "string" ? ` (${defaults["channels.policy.model"]})` : ""}</option>
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
              <Said refusals={refusals} keys={["start"]} />
            </label>
            <label className="field">
              <span>Carries a bookmark</span>
              <input value={bookmark} onChange={event => setBookmark(event.target.value)} placeholder="optional: moved to each checkpoint it makes" spellCheck={false} />
              <Said refusals={refusals} keys={["bookmark"]} />
            </label>
            <div className="field-row">
              <label className="field"><span>Groups</span><input type="number" min={1} value={loopValue("groups")} onChange={event => setLoop({ ...loop, groups: event.target.value })} /><Said refusals={refusals} keys={["groups"]} /></label>
              <label className="field"><span>Groups a step</span><input type="number" min={1} value={loopValue("groups_per_step")} onChange={event => setLoop({ ...loop, groups_per_step: event.target.value })} /><Said refusals={refusals} keys={["groups_per_step"]} /></label>
              <label className="field"><span>Seed</span><input type="number" value={loopValue("seed")} onChange={event => setLoop({ ...loop, seed: event.target.value })} /><Said refusals={refusals} keys={["seed"]} /></label>
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
                  <input value={value} onChange={event => setEdits({ ...edits, [key]: event.target.value })} placeholder={unsetText(key)} spellCheck={false} />
                  {isChanged ? <button type="button" className="linkish" onClick={() => { const next = { ...edits }; delete next[key]; setEdits(next); }}>reset to {shown(defaults[key]) || unsetText(key)}</button> : null}
                  <Said refusals={refusals} keys={[key]} />
                </label>
              );
            })}
          </div>
          <SectionTitle title="Evals" />
          <div className="field-row evals-fields">
            <label className="field">
              <span>Suite</span>
              <select value={chosenSuite} onChange={event => setSuite(event.target.value)} required>
                {chosenSuite ? null : <option value="">choose…</option>}
                <option value={NO_EVALS}>none</option>
                {suites.map(each => <option key={each.suite} value={each.suite}>{each.suite} · {versionTag(currentOf(each).id)} · {currentOf(each).starts.length} starts</option>)}
                {chosenSuite && chosenSuite !== NO_EVALS && !suites.some(each => each.suite === chosenSuite) ? <option value={chosenSuite}>{chosenSuite}</option> : null}
              </select>
              {evalsAsked.errors[EVALS_SUITE] ? <small className="error-text">{evalsAsked.errors[EVALS_SUITE]}</small> : null}
              <Said refusals={refusals} keys={[EVALS_SUITE]} />
            </label>
            <label className="field">
              <span>Every N steps</span>
              <input type="number" min={1} value={every ?? (shown(defaults[EVALS_EVERY]) || "1")} onChange={event => setEvery(event.target.value)} disabled={!evaluating} />
              {evalsAsked.errors[EVALS_EVERY] ? <small className="error-text">{evalsAsked.errors[EVALS_EVERY]}</small> : null}
              <Said refusals={refusals} keys={[EVALS_EVERY]} />
            </label>
            <label className="field">
              <span>Episodes per start</span>
              <input type="number" min={1} value={episodes ?? shown(defaults[EVALS_EPISODES])} onChange={event => setEpisodes(event.target.value)} disabled={!evaluating}
                placeholder={suiteEpisodes && new Set(currentOf(suiteEpisodes).entries.map(entry => entry.episodes)).size === 1 ? String(currentOf(suiteEpisodes).entries[0].episodes) : "suite's"} />
              {evalsAsked.errors[EVALS_EPISODES] ? <small className="error-text">{evalsAsked.errors[EVALS_EPISODES]}</small> : null}
              <Said refusals={refusals} keys={[EVALS_EPISODES]} />
            </label>
          </div>
          <SectionTitle title="More settings" />
          <div className="settings-grid">
            {rows.map((row, index) => (
              <div key={index} className="setting extra">
                <input className="mono" value={row.key} onChange={event => setRows(rows.map((each, at) => (at === index ? { ...each, key: event.target.value } : each)))} placeholder="KEY" spellCheck={false} aria-label="setting" />
                <input value={row.value} onChange={event => setRows(rows.map((each, at) => (at === index ? { ...each, value: event.target.value } : each)))} placeholder="value" spellCheck={false} aria-label="value" />
                <button type="button" className="linkish" onClick={() => setRows(rows.filter((_, at) => at !== index))} aria-label="remove this setting">remove</button>
                <Said refusals={refusals} keys={[row.key.trim()]} />
              </div>
            ))}
            <button type="button" className="linkish" onClick={() => setRows([...rows, { key: "trainer.", value: "" }])}>+ add a setting</button>
          </div>
        </Card>
      </div>
      <div className="launch-submit">
        <button type="submit" disabled={launch.isPending || !name.trim() || !environment || Object.keys(evalsAsked.errors).length > 0}>{launch.isPending ? "asking…" : "Launch the run"}</button>
        <Link to="/runs" className="linkish">cancel</Link>
        {launch.isError && !refusals.length ? <span className="error-text">{launch.error.message}</span> : null}
      </div>
      {elsewhere.length ? (
        <ul className="error-text small refusals">
          {elsewhere.map(each => <li key={`${each.key}:${each.reason}`}><span className="mono">{each.key}</span>: {each.reason}</li>)}
        </ul>
      ) : null}
      {system && system.runs.some(run => nameOf(run) === name.trim()) ? <p className="error-text small">another run is called {name.trim()}</p> : null}
    </form>
  );
}
