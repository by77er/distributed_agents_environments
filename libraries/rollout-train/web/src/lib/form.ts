// The New run form's choices, worked out from what the cluster offers (`/api/offers`): which weights, trainers, models
// and providers fit what is chosen so far, and why one that cannot work cannot (it is shown disabled, with the reason);
// what follows from a choice upstream (`settled`); the objective's presets by family and the components a family
// accepts; which settings the form has a field for; and the settings a launch asks for.

import type { ObjectiveComponent, ObjectivePreset, OfferedProvider, OfferedTrainer, Offers, Preset, Weights } from "../api/types";

export const WEIGHTS: Weights[] = ["lora", "full"];
export const WEIGHT_LABELS: Record<Weights, string> = { lora: "LoRA", full: "Full weights" };
/** Weights in words, as a reason says them. */
export const weightsText = (weights: string): string => (weights === "full" ? "full weights" : "a LoRA");

/** A choice in a picker, and why it can never work, where it cannot. */
export interface Choice {
  value: string;
  label: string;
  why: string | null;
}

/** What a picker shows of a choice: its label, and the reason it is disabled after it. */
export const choiceText = (choice: Choice): string => (choice.why ? `${choice.label} — ${choice.why}` : choice.label);

/** The first choice that can work: the one given, if it can, else the first that can. */
export function pick(choices: Choice[], wanted?: unknown): string | undefined {
  const fits = choices.filter(each => each.why == null);
  return fits.find(each => each.value === wanted)?.value ?? fits[0]?.value;
}

/** The trained channel. */
export const trainedOf = (settings: Record<string, unknown>): string => String(settings["trainer.channel"] ?? "policy");

/** Whether a trainer trains each kind of weights: neither can work where no trainer trains it. */
export const weightChoices = (offers: Offers): Choice[] =>
  WEIGHTS.map(weights => ({
    value: weights,
    label: WEIGHT_LABELS[weights],
    why: offers.trainers.some(each => each.weights.includes(weights)) ? null : `no trainer here trains ${weightsText(weights)}`,
  }));

/** The trainers, each disabled where it trains the other kind of weights. */
export const trainerChoices = (offers: Offers, weights: string | null): Choice[] =>
  offers.trainers.map(each => ({
    value: each.name,
    label: each.name,
    why: weights && !each.weights.includes(weights as Weights) ? `trains ${weightsText(each.weights[0] ?? "")} only` : null,
  }));

/** The renderer families of a model, as the providers that serve it (or a model quantized from it) say. */
export function familiesOf(offers: Offers, model: string): string[] {
  const found = new Set<string>();
  for (const provider of offers.inference) {
    for (const each of provider.models) if (each.model === model || each.base === model) each.families.forEach(family => found.add(family));
  }
  return [...found].sort();
}

/** A trainer's models: those of the environment's model family first, the others said to be of another family (where
 * the families of both are known). */
export function modelChoices(offers: Offers, trainer: OfferedTrainer | undefined, environmentFamilies: string[] = []): Choice[] {
  if (!trainer) return [];
  const other = (model: string) => {
    const families = familiesOf(offers, model);
    return environmentFamilies.length > 0 && families.length > 0 && !families.some(each => environmentFamilies.includes(each));
  };
  return [...trainer.models]
    .sort((a, b) => Number(other(a)) - Number(other(b)))
    .map(model => ({ value: model, label: other(model) ? `${model} (another family)` : model, why: null }));
}

/** Why a provider cannot serve a run's checkpoints of these weights, if it cannot. */
export function servesWhy(provider: OfferedProvider, weights: string): string | null {
  if (provider.weights.includes(weights as Weights)) return null;
  if (weights === "lora") return "serves no adapters";
  return provider.kind === "tinker" ? "Tinker's sampler serves only checkpoints Tinker trained" : "cannot reload full weights in place";
}

/** How a channel's provider is used: for the trained channel, for one that follows it (both serve the run's
 * checkpoints), or for one that serves a fixed model. */
export type Role = "trained" | "follows" | "fixed";

/** The providers for a channel, each disabled where it cannot serve it: the run's weights, the trained channel's needs
 * (the exact tokens it sampled, and their logprobs), and a bridge from the trainer's format. */
export function providerChoices(offers: Offers, role: Role, trainer: string | undefined, weights: string | null): Choice[] {
  return offers.inference.map(provider => {
    let why: string | null = null;
    if (role !== "fixed" && weights) why = servesWhy(provider, weights);
    if (!why && role === "trained" && (!provider.capabilities.token_exact || !provider.capabilities.sampled_logprobs)) {
      why = "returns text, not the sampled tokens and their logprobs";
    }
    if (!why && role !== "fixed" && trainer) why = offers.pairs.find(each => each.trainer === trainer && each.inference === provider.name)?.refused ?? null;
    return { value: provider.name, label: provider.name, why };
  });
}

/** The bridges from a trainer's checkpoints to what a provider loads, where there is work to do. */
export function bridgeOf(offers: Offers, trainer: string | undefined, provider: string | undefined): string[] {
  const pair = offers.pairs.find(each => each.trainer === trainer && each.inference === provider);
  return (pair?.bridge ?? []).filter(each => each !== "none");
}

/** A provider's models; for a channel serving the run's checkpoints, only the trainer's model or one quantized from it. */
export function channelModelChoices(provider: OfferedProvider | undefined, trainerModel: string | undefined, serving: boolean): Choice[] {
  return (provider?.models ?? []).map(each => ({
    value: each.model,
    label: each.model,
    why: serving && trainerModel && each.model !== trainerModel && each.base !== trainerModel ? `neither ${trainerModel} nor quantized from it` : null,
  }));
}

/** The renderers that say they render a model (or the model it was quantized from); none for a hosted API, which
 * renders messages itself. */
export const renderersOf = (provider: OfferedProvider | undefined, model: unknown): string[] =>
  provider?.models.find(each => each.model === model)?.renderers ?? [];

/** The renderer a channel's model takes: the one said where it still renders the model, else the first that does;
 * none where none does (the check then says why) or the provider renders messages itself. */
export function rendererFor(provider: OfferedProvider | undefined, model: unknown, said: unknown): string | undefined {
  const renderers = renderersOf(provider, model);
  if (typeof said === "string" && renderers.includes(said)) return said;
  return renderers[0];
}

/** What follows from a choice upstream: a trainer that trains the weights, a model it trains, and for the trained
 * channel a provider that can serve it, a model it serves and a renderer, each kept where it still fits. */
export function settled(offers: Offers, settings: Record<string, unknown>): Record<string, unknown> {
  const next = { ...settings };
  const weights = (next.weights as string | undefined) ?? null;
  const trainerName = pick(trainerChoices(offers, weights), next["trainer.provider"]);
  if (trainerName !== undefined) next["trainer.provider"] = trainerName;
  const trainer = offers.trainers.find(each => each.name === next["trainer.provider"]);
  if (!weights && trainer) next.weights = trainer.weights[0];
  if (trainer && !trainer.models.includes(String(next["trainer.model"]))) next["trainer.model"] = trainer.models[0];
  const channel = trainedOf(next);
  const providerName = pick(providerChoices(offers, "trained", trainer?.name, (next.weights as string) ?? null), next[`channels.${channel}.provider`]);
  if (providerName !== undefined) next[`channels.${channel}.provider`] = providerName;
  const provider = offers.inference.find(each => each.name === next[`channels.${channel}.provider`]);
  const model = pick(channelModelChoices(provider, next["trainer.model"] as string | undefined, true), next[`channels.${channel}.model`]);
  if (model !== undefined) next[`channels.${channel}.model`] = model;
  next[`channels.${channel}.renderer`] = rendererFor(provider, next[`channels.${channel}.model`], next[`channels.${channel}.renderer`]);
  return next;
}

/** The objective's presets, by family, each disabled where the trainer takes no objective of its family. */
export function objectiveGroups(offers: Offers, trainer: OfferedTrainer | undefined): { family: string; choices: Choice[] }[] {
  const objectives = offers.objectives;
  if (!objectives) return [];
  return objectives.families.map(family => ({
    family,
    choices: objectives.presets.filter(each => each.family === family).map(each => ({
      value: each.name,
      label: each.name,
      why: trainer && !trainer.families.includes(family) ? `the ${trainer.name} trainer takes no ${family} objective` : null,
    })),
  })).filter(group => group.choices.length > 0);
}

export const objectivePreset = (offers: Offers, name: unknown): ObjectivePreset | undefined =>
  offers.objectives?.presets.find(each => each.name === (name ?? "default"));

/** The components an objective's family accepts. */
export const componentsOf = (offers: Offers, family: string | undefined): ObjectiveComponent[] =>
  (offers.objectives?.components ?? []).filter(each => family != null && each.families.includes(family));

/** A schema key's default, as the cluster's monitor says it. */
export const defaultOf = (offers: Offers, key: string): unknown => offers.schema?.find(each => each.key === key)?.default;

/** The settings of a channel the form shows a field for. */
const CHANNEL_FIELDS = ["provider", "providers", "model", "renderer", "mode", "follows", "lag", "thinking_tokens", "answer_tokens"];
const OWN = new Set([
  "environment", "weights", "trainer.provider", "trainer.model", "start", "bookmark", "name", "groups", "groups_per_step",
  "episodes_at_once", "max_lag", "evals.suite", "evals.every", "evals.episodes", "limits.spend", "self_judging", "objective.preset",
]);

/** Whether the form has a field for a setting (where a refusal of it is said): its own, an objective's component, a
 * trainer's declared setting, a setting of a channel or a slot it shows. */
export function fielded(key: string, shown: { channels: string[]; slots: string[]; trainer: string[] }): boolean {
  if (OWN.has(key) || key.startsWith("objective.") || shown.trainer.includes(key)) return true;
  const channel = /^channels\.([^.]+)\.(.+)$/.exec(key);
  if (channel) return shown.channels.includes(channel[1]) && CHANNEL_FIELDS.includes(channel[2]);
  const slot = /^slots\.(.+)$/.exec(key);
  return slot != null && shown.slots.includes(slot[1]);
}

/** The settings a launch asks for: those the form holds (an empty field left out), and, for a key its preset sets that
 * the form cleared, none (so the preset's does not come back). */
export function launchSettings(settings: Record<string, unknown>, preset?: Preset): Record<string, unknown> {
  const said: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(settings)) if (value !== undefined && value !== "" && key !== "name" && key !== "kind") said[key] = value;
  for (const key of Object.keys(preset?.settings ?? {})) if (!(key in said) && key !== "name") said[key] = null;
  return said;
}

/** The settings a preset fills the form with: its own, over the environment chosen where it names none, and what it
 * trains where it does not say (its trainer's). */
export function filled(offers: Offers, preset: Preset, current: Record<string, unknown>): Record<string, unknown> {
  const settings: Record<string, unknown> = { ...preset.settings };
  delete settings.name;
  if (settings.environment == null && current.environment != null) settings.environment = current.environment;
  const trainer = offers.trainers.find(each => each.name === settings["trainer.provider"]);
  if (settings.weights == null && trainer) settings.weights = trainer.weights[0];
  if (settings["trainer.model"] == null && trainer) {
    const model = settings[`channels.${trainedOf(settings)}.model`];
    const provider = offers.inference.find(each => each.name === settings[`channels.${trainedOf(settings)}.provider`]);
    const base = provider?.models.find(each => each.model === model)?.base;
    settings["trainer.model"] = base && trainer.models.includes(base) ? base : model;
  }
  return settings;
}

/** Whether two settings' values are the same. */
export const same = (a: unknown, b: unknown): boolean => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** The keys whose values differ between two settings, or that only the second has. */
export const diffKeys = (before: Record<string, unknown>, after: Record<string, unknown>): Set<string> =>
  new Set(Object.keys(after).filter(key => !(key in before) || !same(before[key], after[key])));
