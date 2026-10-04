// The form that asks a launcher to play a version of a suite with a checkpoint (an eval: nothing trained): on a suite's
// page, the checkpoint is chosen; on a checkpoint's, the suite. The version is the newest unless another is chosen.

import { useState } from "react";
import { useKnown, useLaunch } from "../api/queries";
import type { EvalSuite, Launcher, OfferedProfile, SuiteVersion, System } from "../api/types";
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

/** The environments a version of a suite plays (its newest, unless another is given). */
const environmentsOf = (suite: EvalSuite | undefined, version?: SuiteVersion): string[] =>
  ((version ?? (suite ? currentOf(suite) : undefined))?.environments ?? []).filter((each): each is string => Boolean(each));

/** The launchers that can play a version of a suite (its newest, unless another is given): those that offer each of its
 * environments (those that name none play whatever they are asked); and the profiles they offer, each once. */
export function offeredFor(suite: EvalSuite | undefined, launchers: Launcher[], version?: SuiteVersion): OfferedProfile[] {
  const wanted = environmentsOf(suite, version);
  const able = launchers.filter(each => !each.environments?.length || (wanted.length > 0 && wanted.every(environment => each.environments.includes(environment))));
  const byName = new Map<string, OfferedProfile>();
  for (const launcher of able) for (const profile of launcher.profiles ?? []) if (!byName.has(profile.profile)) byName.set(profile.profile, profile);
  return [...byName.values()];
}

interface PlayProps {
  suites: EvalSuite[];
  launchers: Launcher[];
  system: System;
  /** The suite played, where the page is a suite's: the checkpoint is chosen. */
  suite?: EvalSuite;
  /** The checkpoint that plays, where the page is a checkpoint's: the suite is chosen. */
  subject?: string;
  title: string;
}

/** Ask a launcher to play a version of a suite with a checkpoint (or the base model), so many episodes of each start
 * (by default the version's). */
export function PlayForm({ suites, launchers, system, suite: fixedSuite, subject: fixedSubject, title }: PlayProps) {
  const launch = useLaunch();
  const known = useKnown();
  // (by default the first suite a launcher alive can play)
  const [suiteName, setSuiteName] = useState(fixedSuite?.suite ?? (suites.find(each => offeredFor(each, launchers).length) ?? suites[0])?.suite ?? "");
  const suite = fixedSuite ?? suites.find(each => each.suite === suiteName) ?? suites[0];
  const [profile, setProfile] = useState("");
  const [chosenSubject, setSubject] = useState("");
  const [episodes, setEpisodes] = useState("");
  const [versions, setVersions] = useState<Record<string, string>>({});  // (the version chosen of each suite)
  const [name, setName] = useState("");
  const [asked, setAsked] = useState<string | null>(null);
  const subject = fixedSubject ?? chosenSubject;
  const checkpoints = [...system.checkpoints].sort((a, b) => b.made - a.made);
  const bookmarks = Object.keys(system.bookmarks ?? {}).sort();
  const taken = new Set(Object.values(system.names?.runs ?? {}).concat(system.runs.map(run => run.name ?? run.run)));
  const said = subject ? (system.bookmarks[subject] ? subject : known.short(subject)) : "base";
  const every = suite ? [...versionsOf(suite)].reverse() : [];
  const version = (suite && every.find(each => each.id === versions[suite.suite])) ?? (suite ? currentOf(suite) : undefined);
  const offered = offeredFor(suite, launchers, version);
  const chosen = offered.find(each => each.profile === profile) ?? offered[0];
  const tag = suite && every.length > 1 && version ? ` ${versionTag(version.id)}` : "";
  const named = name.trim() || free(`${suite?.suite ?? "suite"}${tag} on ${said}`, taken);
  const own = [...new Set((version?.entries ?? []).map(entry => entry.episodes))];  // (each entry's episodes of each start)
  const count = episodes.trim() ? Number(episodes) : own.length === 1 ? own[0] : 1;
  const total = (version?.entries ?? []).reduce((sum, entry) => sum + entry.starts * (episodes.trim() ? Math.max(1, count || 1) : entry.episodes), 0);
  const playing = environmentsOf(suite, version);
  const unoffered = playing.filter(each => !launchers.some(launcher => !launcher.environments?.length || launcher.environments.includes(each)));
  const missing = playing.length ? (unoffered.length ? unoffered : playing).map(readable).join(", ") : "its environments";
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!chosen || !suite || !version) return;
    launch.mutate(
      { kind: "eval", suite: version.id, profile: chosen.profile, environment: playing[0] ?? "", name: named, start: subject || null, episodes: episodes.trim() ? count : null },
      { onSuccess: made => { setAsked(made.asked.name); setName(""); } },
    );
  };
  if (!suite) return null;
  if (!offered.length && fixedSuite) {
    return (
      <Card title={title} note={`no launcher alive offers ${missing}`}>
        <pre className="command">{`rollout launcher …${playing.length ? playing.map(each => ` --environment ${each}`).join("") : " --environment module:name"}`}</pre>
      </Card>
    );
  }
  return (
    <form onSubmit={submit}>
      <Card title={title} note={offered.length ? undefined : `no launcher alive offers ${missing}`}>
        <div className="fields">
          <div className="field-row">
            {fixedSuite ? (
              <label className="field">
                <span>Played by</span>
                <select value={chosenSubject} onChange={event => setSubject(event.target.value)}>
                  <option value="">the base model{chosen ? ` (${chosen.model})` : ""}</option>
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
              <span>Profile</span>
              <select value={chosen?.profile ?? ""} onChange={event => setProfile(event.target.value)}>
                {offered.map(each => <option key={each.profile} value={each.profile}>{each.profile} · {each.model}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={event => setName(event.target.value)} placeholder={named} spellCheck={false} />
            </label>
          </div>
        </div>
        <div className="launch-submit">
          <button type="submit" disabled={launch.isPending || !chosen || !Number.isInteger(count) || count < 1}>{launch.isPending ? "asking…" : title}</button>
          {launch.isError ? <span className="error-text">{launch.error.message}</span> : asked ? <span className="small muted">asked for {asked}</span> : null}
        </div>
      </Card>
    </form>
  );
}
