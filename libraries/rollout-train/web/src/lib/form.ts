// The New run form's choices, worked out from what the cluster offers (`/api/offers`): which weights, trainers, models
// and providers fit what is chosen so far, and why one that cannot work cannot (it is shown disabled, with the reason);
// what follows from a choice upstream (`settled`); the objective's presets by family and the components a family
// accepts; which settings the form has a field for; and the settings a launch asks for.

import type {
  CheckpointsAt, ObjectiveComponent, ObjectivePreset, OfferedPair, OfferedPods, OfferedProvider, OfferedTrainer, Offers, Preset, StoreSaid, Weights,
} from "../api/types";

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

/** Why a trainer cannot train these weights, if it cannot. */
const weightsWhy = (trainer: OfferedTrainer, weights: string | null): string | null =>
  weights && !trainer.weights.includes(weights as Weights) ? `trains ${weightsText(trainer.weights[0] ?? "")} only` : null;

/** The trainers, each disabled where it trains the other kind of weights; for training apart (`separate`), also one
 * that trains only beside its provider. */
export const trainerChoices = (offers: Offers, weights: string | null, mode?: Mode): Choice[] =>
  offers.trainers.map(each => {
    let why = weightsWhy(each, weights);
    const apart = () => each.separate !== false && providerChoices(offers, "trained", each.name, weights, "separate").some(choice => choice.why == null);
    if (!why && mode === "separate" && !apart()) {
      why = each.colocate_with ? `trains only beside ${each.colocate_with}` : "nothing here serves its checkpoints";
    }
    return { value: each.name, label: each.name, why };
  });

/** Whether one machine trains and samples (a trainer and its trained channel's provider that share it), or a trainer
 * and a provider apart. */
export type Mode = "together" | "separate";
export const MODES: Mode[] = ["together", "separate"];
export const MODE_LABELS: Record<Mode, string> = { together: "Together", separate: "Separate" };

/** The pair of a trainer and a provider, as the offers say it. */
export const pairOf = (offers: Offers, trainer: unknown, provider: unknown): OfferedPair | undefined =>
  offers.pairs.find(each => each.trainer === trainer && each.inference === provider);

/** Whether a run's trainer and its trained channel's provider share one machine. */
export const modeOf = (offers: Offers, settings: Record<string, unknown>): Mode =>
  pairOf(offers, settings["trainer.provider"], settings[`channels.${trainedOf(settings)}.provider`])?.together ? "together" : "separate";

/** A machine choice's value: its trainer and provider. */
const machineValue = (pair: OfferedPair): string => `${pair.trainer}\n${pair.inference}`;
const machineOf = (value: string): [string, string] => {
  const [trainer, provider] = value.split("\n");
  return [trainer, provider];
};

/** A machine in a few words: its provider's name, and its pods (`NVIDIA H100 80GB HBM3 · $2.69 an hour`) or its
 * GPUs. */
export function machineText(provider: OfferedProvider | undefined, name: string): string {
  if (provider?.pods) return `${name}: ${podsOf(provider.pods)}`;
  const gpus = provider?.gpus ?? 0;
  return gpus ? `${name}: ${gpus} GPU${gpus === 1 ? "" : "s"}` : name;
}

/** The machines that train and sample together: each pair that shares one and can work, named by its machine (and its
 * trainer, where the machine has several); one whose trainer trains the other weights is disabled. */
export function machineChoices(offers: Offers, weights: string | null): Choice[] {
  const together = offers.pairs.filter(each => each.together && !each.refused);
  return together.map(pair => {
    const trainer = offers.trainers.find(each => each.name === pair.trainer);
    const provider = offers.inference.find(each => each.name === pair.inference);
    const several = together.filter(each => each.inference === pair.inference).length > 1;
    const label = `${machineText(provider, pair.inference)}${several ? ` (${pair.trainer})` : ""}`;
    return { value: machineValue(pair), label, why: trainer ? weightsWhy(trainer, weights) : null };
  });
}

/** The machine a run trains and samples on, as its machine choice's value; none where its pair shares none. */
export const machineChosen = (offers: Offers, settings: Record<string, unknown>): string | undefined => {
  const pair = pairOf(offers, settings["trainer.provider"], settings[`channels.${trainedOf(settings)}.provider`]);
  return pair?.together ? machineValue(pair) : undefined;
};

/** The settings a machine choice fills: its trainer, and the trained channel's provider. */
export const machineSettings = (settings: Record<string, unknown>, value: string): Record<string, unknown> => {
  const [trainer, provider] = machineOf(value);
  return { "trainer.provider": trainer, [`channels.${trainedOf(settings)}.provider`]: provider };
};

/** Together or separate, each disabled where nothing here trains these weights that way. */
export function modeChoices(offers: Offers, weights: string | null): Choice[] {
  const together = machineChoices(offers, weights);
  const apart = trainerChoices(offers, weights, "separate");
  return MODES.map(mode => {
    const fits = (mode === "together" ? together : apart).some(each => each.why == null);
    const why = fits ? null : mode === "together" ? "no machine here trains and samples" : "no trainer here trains apart";
    return { value: mode, label: MODE_LABELS[mode], why };
  });
}

/** A blob store in a few words: a named one by its name and bucket (`r2 (rollout)`), the cluster's own as its bucket,
 * a store of files as its directory. */
export function storeText(store: StoreSaid): string {
  const place = store.bucket ?? (store.directory ? `files in ${store.directory}` : store.kind);
  if (store.name) return `${store.name} (${place})`;
  return store.bucket ? `the cluster's bucket (${store.bucket})` : place;
}

/** Where a run's checkpoints go, in a few words: the store, or Tinker, and where a bridge writes its copies. */
export function checkpointsText(at: CheckpointsAt): string {
  const store = storeText(at.store);
  const copies = at.bridges.length && at.bridged ? storeText(at.bridged) : null;
  if (at.tinker) return copies ? `Tinker, bridged to ${copies}` : "Tinker";
  return copies && copies !== store ? `${store}, bridged to ${copies}` : store;
}

/** The same, in full, as a tooltip says it: each store's kind and prefix, and the bridges. */
export function checkpointsTitle(at: CheckpointsAt): string {
  const full = (store: StoreSaid) => [store.kind, store.bucket, store.prefix, store.directory].filter(Boolean).join(" · ");
  const said = [`${at.tinker ? "pointers to Tinker's archive in" : "written to"} ${full(at.store)}`];
  if (at.bridges.length && at.bridged) said.push(`${at.bridges.join(" → ")} into ${full(at.bridged)}`);
  return said.join("\n");
}

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

/** Whether a provider is a hosted API: it takes messages and returns text, with no exact tokens or logprobs. */
export const isHosted = (provider: OfferedProvider | undefined): boolean =>
  provider != null && provider.capabilities.token_exact === false && provider.capabilities.sampled_logprobs === false;

/** Why a hosted API cannot serve a channel of this role, where it cannot: what it samples is never trained on, and it
 * serves no checkpoint. */
export const HOSTED_TRAINED = "a hosted API: it returns text, not the exact tokens it sampled or their logprobs, so nothing it samples is trained on";
export const HOSTED_FOLLOWS = "a hosted API: it serves no checkpoint";

/** The providers for a channel, each disabled where it cannot serve it: the run's weights, the trained channel's needs
 * (the exact tokens it sampled, and their logprobs), and a bridge from the trainer's format. A hosted API serves only a
 * fixed model. Training apart (`separate`), the trained channel's provider is not one that shares the trainer's
 * machine. */
export function providerChoices(offers: Offers, role: Role, trainer: string | undefined, weights: string | null, mode?: Mode): Choice[] {
  return offers.inference.map(provider => {
    let why: string | null = null;
    if (role !== "fixed" && isHosted(provider)) why = role === "trained" ? HOSTED_TRAINED : HOSTED_FOLLOWS;
    if (!why && role !== "fixed" && weights) why = servesWhy(provider, weights);
    if (!why && role === "trained" && (!provider.capabilities.token_exact || !provider.capabilities.sampled_logprobs)) {
      why = "returns text, not the sampled tokens and their logprobs";
    }
    const pair = trainer ? pairOf(offers, trainer, provider.name) : undefined;
    if (!why && role === "trained" && mode === "separate" && pair?.together) why = `shares ${trainer}'s machine`;
    if (!why && role !== "fixed" && trainer) why = pair?.refused ?? null;
    return { value: provider.name, label: provider.name, why };
  });
}

/** A model's catalog prices, in a few words (`$2 in · $10 out a million tokens`); empty where the catalog prices none. */
export function pricesOf(provider: OfferedProvider | undefined, model: unknown): string {
  const cost = provider?.models.find(each => each.model === model)?.cost ?? {};
  if (cost.input == null && cost.output == null) return "";
  const dollars = (value: number) => `$${Number(value.toFixed(4))}`;
  const said = [cost.input != null ? `${dollars(cost.input)} in` : "", cost.output != null ? `${dollars(cost.output)} out` : ""];
  return `${said.filter(Boolean).join(" · ")} a million tokens`;
}

/** What a RunPod provider's pods are, in words: `NVIDIA H100 80GB HBM3 · $2.69 an hour`. */
export function podsOf(pods: OfferedPods | null | undefined): string {
  if (!pods) return "";
  const gpu = pods.gpu_count > 1 ? `${pods.gpu_count}× ${pods.gpu}` : pods.gpu;
  return pods.price != null ? `${gpu} · $${Number(pods.price.toFixed(4))} an hour` : gpu;
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
 * channel a provider that can serve it, a model it serves and a renderer, each kept where it still fits. Together
 * (`mode`, else as the settings are), the trainer and the trained channel's provider are a machine's that trains the
 * weights; separate, a trainer that trains apart and a provider that does not share its machine. Where nothing here
 * trains the weights in that mode and something does in the other, the other is taken. */
export function settled(offers: Offers, settings: Record<string, unknown>, mode?: Mode): Record<string, unknown> {
  const next = { ...settings };
  const weights = (next.weights as string | undefined) ?? null;
  const channel = trainedOf(next);
  const modes = modeChoices(offers, weights);
  let wanted = mode ?? modeOf(offers, next);
  const other: Mode = wanted === "together" ? "separate" : "together";
  if (modes.find(each => each.value === wanted)?.why != null && modes.find(each => each.value === other)?.why == null) wanted = other;
  const machine = wanted === "together" ? pick(machineChoices(offers, weights), machineChosen(offers, next)) : undefined;
  if (machine !== undefined) Object.assign(next, machineSettings(next, machine));
  else {
    const said = next["trainer.provider"];
    const trainerName = pick(trainerChoices(offers, weights, "separate"), said) ?? pick(trainerChoices(offers, weights), said);
    if (trainerName !== undefined) next["trainer.provider"] = trainerName;
  }
  const trainer = offers.trainers.find(each => each.name === next["trainer.provider"]);
  if (!weights && trainer) next.weights = trainer.weights[0];
  if (trainer && !trainer.models.includes(String(next["trainer.model"]))) next["trainer.model"] = trainer.models[0];
  if (machine === undefined) {
    const trains = (next.weights as string) ?? null, said = next[`channels.${channel}.provider`];
    const providerName = pick(providerChoices(offers, "trained", trainer?.name, trains, "separate"), said) ?? pick(providerChoices(offers, "trained", trainer?.name, trains), said);
    if (providerName !== undefined) next[`channels.${channel}.provider`] = providerName;
  }
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
  "environment", "weights", "trainer.provider", "trainer.model", "start", "bookmark", "name", "groups", "groups_per_step", "group_size",
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
