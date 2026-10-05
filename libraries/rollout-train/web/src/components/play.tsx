// The form that asks for an eval: a version of a suite played by a checkpoint or a base model (nothing trained), with
// the channel settings of a preset. On a suite's page, who plays is chosen; on a checkpoint's or a base model's, the
// suite. The version is the newest unless another is chosen.

import { useState } from "react";
import { useKnown, useLaunch } from "../api/queries";
import type { EvalSuite, Offers, Preset, SuiteVersion, System } from "../api/types";
import { readable } from "../lib/environments";
import { currentOf, versionsOf, versionTag } from "../lib/suites";
import { Card } from "./ui";

/** A name no run has: the one wanted, else it with the first number after it that no run has. */
export function free(wanted: string, taken: Set<string>): string {
  if (!taken.has(wanted)) return wanted;
  let number = 2;
  while (taken.has(`${wanted} (${number})`)) number += 1;
  return `${wanted} (${number})`;
}

/** The base models the cluster's inference providers serve, each once, in the order they offer them. */
export const baseModelsOf = (offers: Offers): string[] => [...new Set(offers.inference.flatMap(provider => provider.models.map(each => each.model)))];

/** The settings of a preset an eval takes: its channels' and slots', and how many episodes it plays at once (an eval
 * trains nothing, so a training run's settings are left out). */
export const evalSettingsOf = (preset: Preset | undefined): Record<string, unknown> =>
  Object.fromEntries(Object.entries(preset?.settings ?? {}).filter(([key]) => /^(channels|slots)\.|^(self_judging|episodes_at_once)$/.test(key)));

/** The channel settings an eval of a base model plays with: the preset's, with the model; where the preset's provider
 * does not serve it, the first provider that does (and that model's renderer, where one is named). */
export function playedBy(offers: Offers, preset: Preset | undefined, model: string | undefined): Record<string, unknown> {
  const settings = evalSettingsOf(preset);
  if (model == null) return settings;
  settings["channels.policy.model"] = model;
  const serves = (name: unknown) => offers.inference.some(each => each.name === name && each.models.some(offered => offered.model === model));
  if (!serves(settings["channels.policy.provider"])) {
    const provider = offers.inference.find(each => each.models.some(offered => offered.model === model));
    if (provider) settings["channels.policy.provider"] = provider.name;
    const renderer = provider?.models.find(each => each.model === model)?.renderers[0];
    if (renderer) settings["channels.policy.renderer"] = renderer;
  }
  return settings;
}

/** How the Played by picker says a base model (a checkpoint or a bookmark is said by itself). */
const BASE = "base:";

/** The environments a version of a suite plays (its newest, unless another is given). */
const environmentsOf = (suite: EvalSuite | undefined, version?: SuiteVersion): string[] =>
  ((version ?? (suite ? currentOf(suite) : undefined))?.environments ?? []).filter((each): each is string => Boolean(each));

interface PlayProps {
  suites: EvalSuite[];
  offers: Offers;
  system: System;
  /** The suite played, where the page is a suite's: who plays is chosen. */
  suite?: EvalSuite;
  /** The checkpoint that plays, where the page is a checkpoint's: the suite is chosen. */
  subject?: string;
  /** The base model that plays, where the page is a base model's: the suite is chosen. */
  model?: string;
  title: string;
}

/** Ask for an eval of a version of a suite with a checkpoint or a base model, so many episodes of each start (by
 * default the version's). */
export function PlayForm({ suites, offers, system, suite: fixedSuite, subject: fixedSubject, model: fixedModel, title }: PlayProps) {
  const launch = useLaunch();
  const known = useKnown();
  const offered = new Set(offers.environments.map(each => each.environment));
  const plays = (suite: EvalSuite) => environmentsOf(suite).every(each => offered.has(each));
  // (by default the first suite whose environments the cluster offers)
  const [suiteName, setSuiteName] = useState(fixedSuite?.suite ?? (suites.find(plays) ?? suites[0])?.suite ?? "");
  const suite = fixedSuite ?? suites.find(each => each.suite === suiteName) ?? suites[0];
  const [presetId, setPreset] = useState<string | null>(null);
  const [chosenSubject, setSubject] = useState("");
  const [episodes, setEpisodes] = useState("");
  const [versions, setVersions] = useState<Record<string, string>>({});  // (the version chosen of each suite)
  const [name, setName] = useState("");
  const [asked, setAsked] = useState<string | null>(null);
  const checkpoints = [...system.checkpoints].sort((a, b) => b.made - a.made);
  const bookmarks = Object.keys(system.bookmarks ?? {}).sort();
  const taken = new Set(Object.values(system.names?.runs ?? {}).concat(system.runs.map(run => run.name ?? run.run)));
  const every = suite ? [...versionsOf(suite)].reverse() : [];
  const version = (suite && every.find(each => each.id === versions[suite.suite])) ?? (suite ? currentOf(suite) : undefined);
  const bases = baseModelsOf(offers);
  // (who plays: the page's checkpoint or base model, else the one chosen, else the first base model offered)
  const picked = chosenSubject || (bases[0] ? `${BASE}${bases[0]}` : "");
  const model = fixedModel ?? (fixedSubject == null && picked.startsWith(BASE) ? picked.slice(BASE.length) : undefined);
  const subject = fixedSubject ?? (model == null ? picked : "");
  // (the preset chosen; by default one whose trained channel is the base model played, else the first)
  const preset = presetId === "" ? undefined
    : offers.presets.find(each => each.id === presetId) ?? offers.presets.find(each => model != null && each.settings["channels.policy.model"] === model) ?? offers.presets[0];
  const said = model != null ? model.split("/").at(-1) : system.bookmarks[subject] ? subject : known.short(subject);
  const tag = suite && every.length > 1 && version ? ` ${versionTag(version.id)}` : "";
  const named = name.trim() || free(`${suite?.suite ?? "suite"}${tag} on ${said}`, taken);
  const own = [...new Set((version?.entries ?? []).map(entry => entry.episodes))];  // (each entry's episodes of each start)
  const count = episodes.trim() ? Number(episodes) : own.length === 1 ? own[0] : 1;
  const total = (version?.entries ?? []).reduce((sum, entry) => sum + entry.starts * (episodes.trim() ? Math.max(1, count || 1) : entry.episodes), 0);
  const playing = environmentsOf(suite, version);
  const unoffered = playing.filter(each => !offered.has(each));
  const served = model == null || bases.includes(model);
  const missing = !offers.cluster ? "this monitor asks for no runs (rollout monitor --cluster)"
    : !served ? `the cluster offers no provider of ${model}`
      : unoffered.length ? `the cluster does not offer ${unoffered.map(readable).join(", ")}` : null;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!suite || !version || missing) return;
    const settings: Record<string, unknown> = { ...playedBy(offers, preset, model), "eval.suite": version.id };
    if (subject) settings.start = subject;
    if (episodes.trim()) settings["eval.episodes"] = count;
    launch.mutate(
      { kind: "eval", name: named, environment: playing[0] ?? null, preset: null, settings },
      { onSuccess: made => { setAsked(made.asked.name); setName(""); } },
    );
  };
  if (!suite) return null;
  return (
    <form onSubmit={submit}>
      <Card title={title} note={missing ?? undefined}>
        <div className="fields">
          <div className="field-row">
            {fixedSuite ? (
              <label className="field">
                <span>Played by</span>
                <select value={picked} onChange={event => setSubject(event.target.value)}>
                  {bases.length ? (
                    <optgroup label="Base models">
                      {bases.map(each => <option key={`m${each}`} value={`${BASE}${each}`}>{each}</option>)}
                    </optgroup>
                  ) : null}
                  {bookmarks.length ? (
                    <optgroup label="Bookmarks">
                      {bookmarks.map(mark => <option key={`b${mark}`} value={mark}>{mark} · {known.origin(system.bookmarks[mark])} · {known.short(system.bookmarks[mark])}</option>)}
                    </optgroup>
                  ) : null}
                  {checkpoints.length ? (
                    <optgroup label="Checkpoints, newest first">
                      {checkpoints.map(checkpoint => (
                        <option key={checkpoint.id} value={checkpoint.id} disabled={checkpoint.weights == null}>
                          {known.origin(checkpoint.id)} · {checkpoint.short}{checkpoint.bookmarks.length ? ` [${checkpoint.bookmarks.join(", ")}]` : ""}{checkpoint.weights == null ? " (released)" : ""}
                        </option>
                      ))}
                    </optgroup>
                  ) : null}
                </select>
              </label>
            ) : (
              <label className="field">
                <span>Suite</span>
                <select value={suite.suite} onChange={event => setSuiteName(event.target.value)}>
                  {suites.map(each => <option key={each.suite} value={each.suite}>{each.suite} · {each.starts.length} starts</option>)}
                </select>
              </label>
            )}
            <label className="field">
              <span>Version</span>
              <select value={version?.id ?? ""} onChange={event => setVersions({ ...versions, [suite.suite]: event.target.value })} disabled={every.length < 2}>
                {every.map((each, place) => <option key={each.id} value={each.id}>{versionTag(each.id)}{place === 0 ? " (newest)" : ""} · {each.starts.length} starts</option>)}
              </select>
            </label>
            <label className="field">
              <span>Episodes per start</span>
              <input type="number" min={1} value={episodes} onChange={event => setEpisodes(event.target.value)} placeholder={own.length === 1 ? String(own[0]) : "each entry's"} />
              <small>{total} in all</small>
            </label>
          </div>
          <div className="field-row">
            <label className="field">
              <span>Preset</span>
              <select value={preset?.id ?? ""} onChange={event => setPreset(event.target.value)}>
                <option value="">no preset</option>
                {offers.presets.map(each => <option key={each.id} value={each.id}>{each.id}{typeof each.settings["channels.policy.model"] === "string" ? ` · ${each.settings["channels.policy.model"]}` : ""}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={event => setName(event.target.value)} placeholder={named} spellCheck={false} />
            </label>
          </div>
        </div>
        <div className="launch-submit">
          <button type="submit" disabled={launch.isPending || missing != null || !Number.isInteger(count) || count < 1}>{launch.isPending ? "asking…" : title}</button>
          {launch.isError ? <span className="error-text">{launch.error.message}</span> : asked ? <span className="small muted">asked for {asked}</span> : null}
        </div>
      </Card>
    </form>
  );
}
