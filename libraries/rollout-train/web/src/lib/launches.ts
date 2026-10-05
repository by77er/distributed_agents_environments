// A launch as the page says it: its state in the run's words once its run exists (running, paused, finished, stopped,
// failed, lost), before that asked, submitted or waiting (with the reason); and the settings it changed, each with its
// meaning, leaving out what the defaults or its preset say.

import type { Launch, Preset, Run } from "../api/types";

/** A launch's state as the page says it, the color it is drawn in, and the reason it waits, where it waits. */
export interface LaunchShown {
  state: "asked" | "submitted" | "waiting" | "running" | "paused" | "stopping" | "finished" | "stopped" | "failed" | "lost";
  kind: string;
  /** What it waits for, in a few words; the full detail beside it. */
  reason: string | null;
}

/** The states of a launch that is going. */
export const GOING = new Set(["asked", "submitted", "running", "stopping"]);

const COLOR: Record<LaunchShown["state"], string> = {
  asked: "warm", submitted: "accent", waiting: "warm", running: "good", paused: "violet", stopping: "warm",
  finished: "good", stopped: "", failed: "bad", lost: "bad",
};

/** What a run waits for, in a few words, from the detail its launch says: GPUs the run's actors asked Ray for and do
 * not have, else the detail itself. */
export function waitingFor(detail: string): string {
  const gpus = [...detail.matchAll(/\(([\d.]+) GPU/g)].map(found => Number(found[1]));
  if (gpus.length) {
    const wanted = gpus.reduce((sum, each) => sum + each, 0);
    return `no GPU free (${+wanted.toPrecision(3)} wanted)`;
  }
  return detail.replace(/^waits for /, "");
}

const shown = (state: LaunchShown["state"], reason: string | null = null): LaunchShown => ({ state, kind: COLOR[state], reason });

/** A launch's state in one vocabulary (the module's comment): what the launch table and its job's status say, in the
 * run's words once the run exists and goes. */
export function launchState(launch: Launch, run?: Run): LaunchShown {
  if (launch.state === "ended") return shown("finished");
  if (launch.state === "failed") return shown("failed");
  if (launch.state === "stopped") return shown("stopped");
  if (launch.state === "stopping") return shown("stopping");
  if (launch.state === "asked") return shown("asked");
  const waits = launch.detail != null && launch.detail.startsWith("waits for");
  if (launch.state === "submitted") {
    if (launch.detail && launch.detail !== "submitted") return shown("waiting", waitingFor(launch.detail));
    return shown("submitted");
  }
  if (waits) return shown("waiting", waitingFor(launch.detail!));
  if (run) {
    if (run.state === "paused" || run.pause) return shown("paused");
    if (run.state === "lost") return shown("lost");
  }
  return shown("running");
}

/** What a setting is called, where the page has a word for it. */
const LABELS: Record<string, string> = {
  episodes_at_once: "episodes at once",
  max_lag: "lag",
  "evals.every": "evals every",
  "evals.episodes": "eval episodes",
  "eval.episodes": "episodes a start",
  "trainer.provider": "trainer",
  "trainer.model": "trains",
  "trainer.rank": "rank",
  "trainer.learning_rate": "learning rate",
  "trainer.segment_tokens": "segment tokens",
  "trainer.segments_per_step": "segments a step",
  "trainer.tokens_per_step": "tokens a step",
  "limits.spend": "spend limit",
  "objective.preset": "objective",
  self_judging: "self-judging",
};

/** Settings the tile says otherwise (its title, its facts) or that every launch has. */
const OWN = new Set(["kind", "name", "environment", "start", "bookmark", "groups", "groups_per_step", "seed", "eval.suite"]);

/** Run settings' defaults, as the schema has them: a setting at its default is left out. */
export const DEFAULTS: Record<string, unknown> = {
  groups: 100, groups_per_step: 4, seed: 0, episodes_at_once: 6, max_lag: 1, "evals.every": 1, "evals.episodes": null,
  "limits.spend": null, self_judging: false, "objective.preset": "default", "trainer.channel": "policy",
};

/** A setting's name in words: the page's word for it, a channel's setting by its channel, else the key. */
export function settingLabel(key: string): string {
  if (LABELS[key]) return LABELS[key];
  const channel = /^channels\.([^.]+)\.(.+)$/.exec(key);
  if (channel) {
    const [, name, setting] = channel;
    const word = { provider: "inference", providers: "inference", model: "model", renderer: "renderer", thinking_tokens: "thinking tokens",
      answer_tokens: "answer tokens", replicas: "replicas", mode: "serves", checkpoint: "checkpoint", follows: "follows", lag: "lag" }[setting];
    return word ? (name === "policy" ? word : `${name} ${word}`) : key;
  }
  const slot = /^slots\.(.+)$/.exec(key);
  if (slot) return `${slot[1]} slot`;
  if (key.startsWith("trainer.")) return key.slice("trainer.".length).replaceAll("_", " ");
  return key;
}

/** A setting's value as the page writes it: numbers readably (a small rate in exponent form), true and false as yes
 * and no, none as none, lists joined. */
export function settingValue(value: unknown): string {
  if (value == null) return "none";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString("en");
    if (Math.abs(value) < 0.001) return value.toExponential();
    return String(+value.toPrecision(4));
  }
  if (Array.isArray(value)) return value.map(settingValue).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** One changed setting as the tile says it: its words, the key in its title. */
export interface SettingShown {
  key: string;
  text: string;
}

/** The settings a launch asked for that differ from the defaults and from its preset's (`preset`, where the page knows
 * them), each in words: "no evals" for no suite, a budget's none as no budget. */
export function changedSettings(settings: Record<string, unknown>, preset?: Preset): SettingShown[] {
  const found: SettingShown[] = [];
  for (const key of Object.keys(settings).sort()) {
    const value = settings[key];
    if (OWN.has(key)) continue;
    if (preset && key in preset.settings && JSON.stringify(preset.settings[key]) === JSON.stringify(value)) continue;
    if (key in DEFAULTS && JSON.stringify(DEFAULTS[key]) === JSON.stringify(value ?? null)) continue;
    if (!(key in DEFAULTS) && value == null && !key.startsWith("evals.") && !/_tokens$/.test(key)) continue;
    if (key === "evals.suite") {
      found.push({ key, text: value == null ? "no evals" : `evals on ${String(value)}` });
      continue;
    }
    if (/^channels\.[^.]+\.(thinking|answer)_tokens$/.test(key) && (value == null || value === "none")) {
      found.push({ key, text: `no ${settingLabel(key).replace(/ tokens$/, "")} budget` });
      continue;
    }
    if (key === "limits.spend" && typeof value === "number") {
      found.push({ key, text: `spend up to $${settingValue(value)}` });
      continue;
    }
    found.push({ key, text: `${settingLabel(key)} ${settingValue(value)}` });
  }
  return found;
}

/** A launch's groups, groups a step and seed, each only where it differs from the default. */
export function loopChanged(settings: Record<string, unknown>): string[] {
  const said: string[] = [];
  const groups = settings.groups, perStep = settings.groups_per_step, seed = settings.seed;
  if (groups != null && groups !== DEFAULTS.groups) said.push(`${settingValue(groups)} groups`);
  if (perStep != null && perStep !== DEFAULTS.groups_per_step) said.push(`${settingValue(perStep)} a step`);
  if (seed != null && seed !== DEFAULTS.seed) said.push(`seed ${settingValue(seed)}`);
  return said;
}
