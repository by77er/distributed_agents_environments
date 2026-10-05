// A new training run, asked for from the page, in the order its choices decide each other: the environment, LoRA or
// full weights, the trainer and its model, inference for each channel, the objective, budgets, evals, the spend limit
// and the name. Every choice comes from what the cluster offers (`/api/offers`); one that can never work is disabled
// with its reason. A preset fills the form, and the form saves itself as one. The monitor checks the settings as they
// change, and what it refuses is said beside the field it is about.

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Refused, useCheck, useEvals, useKnown, useLaunch, useOffers, useSavePreset, useSystem } from "../api/queries";
import type { Checked, EvalSuite, LaunchAsking, OfferedTrainer, Offers, SettingFinding, System } from "../api/types";
import { Card, Empty, Head } from "../components/ui";
import { publishedParts, readable } from "../lib/environments";
import {
  bridgeOf, channelModelChoices, type Choice, choiceText, componentsOf, defaultOf, fielded, filled, launchSettings, modelChoices, objectiveGroups,
  objectivePreset, pick, pricesOf, providerChoices, rendererFor, renderersOf, type Role, same, servesWhy, settled, trainedOf, trainerChoices,
  weightChoices,
} from "../lib/form";
import { presetsPlace } from "../lib/places";
import { EVALS_EPISODES, EVALS_EVERY, EVALS_SUITE, evalsSettings, NO_EVALS, shown, typed } from "../lib/settings";
import { currentOf, versionTag } from "../lib/suites";

export { typed };

/** How long the form waits after a change before the monitor checks it, in milliseconds. */
const CHECK_AFTER = 400;

export function NewRun() {
  const { data: offers } = useOffers();
  const { data: system } = useSystem();
  if (!offers || !system) return <Empty>Reading what can be asked for…</Empty>;
  return (
    <>
      <Head title="New run" />
      {offers.cluster ? <Form offers={offers} system={system} /> : <NoCluster />}
    </>
  );
}

export function NoCluster() {
  return <Empty>This monitor asks for no runs without a cluster config: <span className="mono">rollout monitor --cluster</span>.</Empty>;
}

/** A value that follows `value` once it has stayed the same for `delay` milliseconds. */
function useSettledValue<T>(value: T, delay: number): T {
  const [held, setHeld] = useState(value);
  const said = JSON.stringify(value);
  useEffect(() => {
    const timer = setTimeout(() => setHeld(JSON.parse(said) as T), delay);
    return () => clearTimeout(timer);
  }, [said, delay]);
  return held;
}

/** The refusals said of one field. */
function Said({ refusals, keys }: { refusals: SettingFinding[]; keys: string[] }) {
  const found = refusals.filter(each => keys.includes(each.key));
  return found.length ? <small className="error-text" role="alert">{found.map(each => each.reason).join("; ")}</small> : null;
}

interface FieldProps {
  /** None where the card's title says it. */
  label?: string;
  /** The settings it holds, whose refusals it says. */
  keys: string[];
  refusals: SettingFinding[];
  /** Set by the preset chosen, as it set it. */
  marked?: boolean;
  children: ReactNode;
  note?: ReactNode;
}

function Field({ label, keys, refusals, marked = false, children, note }: FieldProps) {
  return (
    <label className={`field${marked ? " from-preset" : ""}`}>
      {label ? <span title={marked ? "as the preset sets it" : undefined}>{label}</span> : null}
      {children}
      {note}
      <Said refusals={refusals} keys={keys} />
    </label>
  );
}

/** A select of choices: one that can never work is disabled, its reason after its name and in its tooltip. */
function Picker({ value, choices, onChange, label, empty }: { value: unknown; choices: Choice[]; onChange: (value: string) => void; label: string; empty?: string }) {
  const current = typeof value === "string" ? value : "";
  return (
    <select value={current} onChange={event => onChange(event.target.value)} aria-label={label}>
      {current && choices.some(each => each.value === current) ? null : <option value="">{empty ?? "choose"}</option>}
      {choices.map(each => <option key={each.value} value={each.value} disabled={each.why != null} title={each.why ?? undefined}>{choiceText(each)}</option>)}
    </select>
  );
}

const asNumber = (text: string): number | undefined => (text.trim() === "" ? undefined : Number(text));
const text = (value: unknown): string => (value == null ? "" : String(value));

/** A renderer by its function's name (`rollout_qwen:qwen35` is `qwen35`). */
const rendererName = (reference: string): string => reference.split(":").pop() ?? reference;

function Form({ offers, system }: { offers: Offers; system: System }) {
  const navigate = useNavigate();
  const launch = useLaunch();
  const savePreset = useSavePreset();
  const known = useKnown();
  const { data: evals } = useEvals();
  const [asked] = useSearchParams();  // (`?environment=module:name`, `?model=NAME`: the form opened from that page)
  const [presetId, setPresetId] = useState("");
  const preset = offers.presets.find(each => each.id === presetId);
  const [settings, setSettings] = useState<Record<string, unknown>>(() => {
    const environment = asked.get("environment") ?? offers.environments[0]?.environment;
    const model = asked.get("model");
    const trainer = model ? offers.trainers.find(each => each.models.includes(model)) : undefined;
    return settled(offers, { environment, "evals.suite": null, ...(trainer ? { "trainer.provider": trainer.name, "trainer.model": model, weights: trainer.weights[0] } : {}) });
  });
  const [name, setName] = useState("");
  const [saving, setSaving] = useState<string | null>(null);  // (the name the settings are being saved as a preset under)
  const set = (changes: Record<string, unknown>) => setSettings(current => ({ ...current, ...changes }));
  const setUpstream = (changes: Record<string, unknown>) => setSettings(current => settled(offers, { ...current, ...changes }));
  const unset = (keys: string[]) => setSettings(current => Object.fromEntries(Object.entries(current).filter(([key]) => !keys.includes(key))));

  const trained = trainedOf(settings);
  const weights = (settings.weights as string | undefined) ?? null;
  const trainer = offers.trainers.find(each => each.name === settings["trainer.provider"]);
  const environment = typeof settings.environment === "string" ? settings.environment : "";
  const environmentFamilies = offers.environments.find(each => each.environment === environment)?.families ?? [];
  const marked = (...keys: string[]) => preset != null && keys.some(key => key in preset.settings && same(preset.settings[key], settings[key]));

  const asking: LaunchAsking = { kind: "train", name: name.trim(), environment: environment || null, preset: preset?.id ?? null, settings: launchSettings(settings, preset) };
  const checking = useSettledValue(asking, CHECK_AFTER);
  const check = useCheck(checking.environment ? checking : null);
  const checked: Checked | undefined = check.data;
  const submitted = launch.error instanceof Refused ? launch.error.refusals : [];
  const refusals = [...(checked?.refusals ?? []), ...submitted.filter(each => !(checked?.refusals ?? []).some(other => other.key === each.key && other.reason === each.reason))];

  // The channels besides the trained one: each slot of the environment that is not trained or judges, and any the
  // settings name.
  const declared = checked?.environment;
  const slotNames = [...new Set([...(declared?.untrained ?? []), ...(declared?.judges ?? [])])].sort();
  const channelOf = (slot: string) => String(settings[`slots.${slot}`] ?? slot);
  const named = Object.keys(settings).map(key => /^channels\.([^.]+)\./.exec(key)?.[1]).filter((each): each is string => each != null && each !== trained);
  const others = [...new Set([...slotNames.map(channelOf), ...named])];
  const trainerKeys = (trainer?.settings ?? []).map(each => each.key);
  const shownKeys = { channels: [trained, ...others], slots: slotNames, trainer: trainerKeys };
  const elsewhere = refusals.filter(each => !fielded(each.key, shownKeys));

  const offered = new Set(offers.environments.map(each => each.environment));
  const playable = (each: EvalSuite) => currentOf(each).environments.every(played => !played || offered.has(played));
  const suites = (evals?.suites ?? []).filter(playable);
  const suite = settings[EVALS_SUITE] == null ? NO_EVALS : String(settings[EVALS_SUITE]);
  const evalsAsked = evalsSettings(suite, text(settings[EVALS_EVERY] ?? 1), text(settings[EVALS_EPISODES]));

  const choosePreset = (id: string) => {
    setPresetId(id);
    const next = offers.presets.find(each => each.id === id);
    if (next) setSettings(current => filled(offers, next, current));
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    launch.mutate({ ...asking, settings: { ...asking.settings, ...evalsAsked.settings } }, { onSuccess: () => navigate("/runs") });
  };

  const save = () => {
    if (!saving?.trim()) return;
    const kept = launchSettings(settings);
    savePreset.mutate({ name: saving.trim(), settings: kept }, { onSuccess: made => { setSaving(null); setPresetId(made.id); } });
  };

  const notes = (checked?.notes ?? []).filter(each => !each.reason.startsWith("one step's spend cannot be estimated"));
  return (
    <form className="launch-form" onSubmit={submit}>
      <div className="preset-bar">
        <label className="field">
          <span>Preset</span>
          <select value={presetId} onChange={event => choosePreset(event.target.value)} aria-label="preset">
            <option value="">no preset</option>
            {offers.presets.map(each => <option key={each.id} value={each.id}>{each.id}</option>)}
          </select>
        </label>
        {saving == null ? (
          <button type="button" className="action" onClick={() => setSaving(preset?.name ?? "")}>Save as preset</button>
        ) : (
          <span className="preset-save">
            <input value={saving} onChange={event => setSaving(event.target.value)} placeholder="preset name" aria-label="preset name" spellCheck={false} autoFocus />
            <button type="button" className="action" disabled={!saving.trim() || savePreset.isPending} onClick={save}>{savePreset.isPending ? "saving…" : "Save"}</button>
            <button type="button" className="linkish" onClick={() => { setSaving(null); savePreset.reset(); }}>cancel</button>
          </span>
        )}
        <Link to={presetsPlace} className="linkish">Presets</Link>
        {savePreset.isError ? <span className="error-text" role="alert">{savePreset.error.message}</span> : null}
      </div>

      <Card title="Environment">
        <Field keys={["environment"]} refusals={refusals} marked={marked("environment")}>
          <EnvironmentChoice offers={offers} value={environment} onChange={picked => set({ environment: picked })} />
        </Field>
      </Card>

      <Card title="Weights">
        <WeightsChoice offers={offers} value={weights} onChange={picked => setUpstream({ weights: picked })} marked={marked("weights")} />
        <Said refusals={refusals} keys={["weights"]} />
      </Card>

      <Card title="Trainer">
        <TrainerStep offers={offers} system={system} known={known} settings={settings} trainer={trainer} weights={weights} families={environmentFamilies}
          refusals={refusals} marked={marked} set={set} setUpstream={setUpstream} unset={unset} />
      </Card>

      <Card title="Inference">
        <div className="fields">
          <ChannelFields offers={offers} settings={settings} channel={trained} role="trained" trainer={trainer} weights={weights} refusals={refusals} marked={marked}
            set={set} setUpstream={setUpstream} />
          {others.map(channel => {
            const slot = slotNames.find(each => channelOf(each) === channel);
            const role: Role = settings[`channels.${channel}.mode`] === "follows" ? "follows" : "fixed";
            return (
              <ChannelFields key={channel} offers={offers} settings={settings} channel={channel} slot={slot} role={role} trainer={trainer} weights={weights}
                judges={slot != null && (declared?.judges ?? []).includes(slot)} refusals={refusals} marked={marked} set={set} setUpstream={setUpstream} />
            );
          })}
        </div>
      </Card>

      <Card title="Objective">
        <ObjectiveStep offers={offers} settings={settings} trainer={trainer} refusals={refusals} marked={marked} set={set} unset={unset} />
      </Card>

      <Card title="Budgets">
        <div className="field-row">
          {([
            ["Groups", "groups"], ["Groups a step", "groups_per_step"], ["Episodes a group", "group_size"], ["Episodes at once", "episodes_at_once"],
            ["Thinking tokens", `channels.${trained}.thinking_tokens`], ["Answer tokens", `channels.${trained}.answer_tokens`], ["Max lag", "max_lag"],
          ] as const).map(([label, key]) => (
            <Field key={key} label={label} keys={[key]} refusals={refusals} marked={marked(key)}>
              <input type="number" min={key === "max_lag" ? 0 : 1} value={text(settings[key])} placeholder={shown(defaultOf(offers, key.replace(/^channels\.[^.]+\./, "channels.*."))) || "none"}
                onChange={event => set({ [key]: (key.endsWith("_tokens") || key === "group_size") && !event.target.value.trim() ? null : asNumber(event.target.value) })} />
            </Field>
          ))}
        </div>
      </Card>

      <Card title="Evals">
        <div className="field-row evals-fields">
          <Field label="Suite" keys={[EVALS_SUITE]} refusals={refusals} marked={marked(EVALS_SUITE)}>
            <select value={suite} onChange={event => set({ [EVALS_SUITE]: event.target.value === NO_EVALS ? null : event.target.value })} aria-label="suite">
              <option value={NO_EVALS}>none</option>
              {suites.map(each => <option key={each.suite} value={each.suite}>{each.suite} · {versionTag(currentOf(each).id)} · {currentOf(each).starts.length} starts</option>)}
              {suite !== NO_EVALS && !suites.some(each => each.suite === suite) ? <option value={suite}>{suite}</option> : null}
            </select>
          </Field>
          <Field label="Every N steps" keys={[EVALS_EVERY]} refusals={refusals} marked={marked(EVALS_EVERY)}
            note={evalsAsked.errors[EVALS_EVERY] ? <small className="error-text">{evalsAsked.errors[EVALS_EVERY]}</small> : null}>
            <input type="number" min={1} value={text(settings[EVALS_EVERY])} placeholder="1" disabled={suite === NO_EVALS} onChange={event => set({ [EVALS_EVERY]: asNumber(event.target.value) })} />
          </Field>
          <Field label="Episodes per start" keys={[EVALS_EPISODES]} refusals={refusals} marked={marked(EVALS_EPISODES)}>
            <input type="number" min={1} value={text(settings[EVALS_EPISODES])} placeholder="suite's" disabled={suite === NO_EVALS} onChange={event => set({ [EVALS_EPISODES]: asNumber(event.target.value) })} />
          </Field>
        </div>
      </Card>

      <Card title="Spend">
        <Field label="Limit ($)" keys={["limits.spend"]} refusals={refusals} marked={marked("limits.spend")} note={<SpendNote checked={checked} />}>
          <input type="number" min={0} step="any" value={text(settings["limits.spend"])} placeholder="none" onChange={event => set({ "limits.spend": asNumber(event.target.value) })} />
        </Field>
      </Card>

      <Card title="Name">
        <div className="fields">
          <Field keys={["name"]} refusals={refusals}>
            <input value={name} onChange={event => setName(event.target.value)} required spellCheck={false} aria-label="name" />
          </Field>
          <Field label="Bookmark" keys={["bookmark"]} refusals={refusals} marked={marked("bookmark")}>
            <input value={text(settings.bookmark)} onChange={event => set({ bookmark: event.target.value.trim() || undefined })} placeholder="optional" spellCheck={false} />
          </Field>
        </div>
      </Card>

      <div className="launch-submit">
        <button type="submit" disabled={launch.isPending || !name.trim() || !environment || Object.keys(evalsAsked.errors).length > 0}>{launch.isPending ? "asking…" : "Launch the run"}</button>
        <Link to="/runs" className="linkish">cancel</Link>
        {check.isFetching ? <span className="small faint">checking…</span> : null}
        {launch.isError && !(launch.error instanceof Refused) ? <span className="error-text" role="alert">{launch.error.message}</span> : null}
        {check.isError ? <span className="error-text" role="alert">{check.error.message}</span> : null}
      </div>
      {notes.length ? (
        <ul className="small muted refusals notes">
          {notes.map(each => <li key={`${each.key}:${each.reason}`}>{each.reason}</li>)}
        </ul>
      ) : null}
      {elsewhere.length ? (
        <ul className="error-text small refusals" role="alert">
          {elsewhere.map(each => <li key={`${each.key}:${each.reason}`}><span className="mono">{each.key}</span>: {each.reason}</li>)}
        </ul>
      ) : null}
    </form>
  );
}

/** The environments the cluster offers, built in and imported, each by its readable name. */
function EnvironmentChoice({ offers, value, onChange }: { offers: Offers; value: string; onChange: (environment: string) => void }) {
  const builtIn = offers.environments.filter(each => !each.published);
  const imported = offers.environments.filter(each => each.published);
  const option = (environment: string) => {
    const published = publishedParts(environment);
    return <option key={environment} value={environment} title={environment}>{published ? `${published.name} · ${published.version.slice(0, 8)}` : readable(environment)}</option>;
  };
  return (
    <>
      <select value={value} onChange={event => onChange(event.target.value)} aria-label="environment" required>
        {value ? null : <option value="" disabled>choose</option>}
        {builtIn.length && imported.length ? <optgroup label="built in">{builtIn.map(each => option(each.environment))}</optgroup> : builtIn.map(each => option(each.environment))}
        {imported.length ? <optgroup label="imported">{imported.map(each => option(each.environment))}</optgroup> : null}
        {value && !offers.environments.some(each => each.environment === value) ? option(value) : null}
      </select>
      {value ? <small className="mono" title={value}>{value}</small> : null}
    </>
  );
}

/** LoRA or full weights: one no trainer trains is disabled, with why. */
function WeightsChoice({ offers, value, onChange, marked }: { offers: Offers; value: string | null; onChange: (weights: string) => void; marked: boolean }) {
  return (
    <div className={`segmented${marked ? " from-preset" : ""}`} role="radiogroup" aria-label="weights">
      {weightChoices(offers).map(each => (
        <label key={each.value} className={each.why ? "disabled" : undefined} title={each.why ?? undefined}>
          <input type="radio" name="weights" value={each.value} checked={value === each.value} disabled={each.why != null} onChange={() => onChange(each.value)} />
          <span>{each.label}</span>
          {each.why ? <small>{each.why}</small> : null}
        </label>
      ))}
    </div>
  );
}

interface StepProps {
  offers: Offers;
  settings: Record<string, unknown>;
  refusals: SettingFinding[];
  marked: (...keys: string[]) => boolean;
  set: (changes: Record<string, unknown>) => void;
}

function TrainerStep({ offers, system, known, settings, trainer, weights, families, refusals, marked, set, setUpstream, unset }: StepProps & {
  system: System; known: ReturnType<typeof useKnown>; trainer: OfferedTrainer | undefined; weights: string | null; families: string[];
  setUpstream: (changes: Record<string, unknown>) => void; unset: (keys: string[]) => void;
}) {
  const checkpoints = [...system.checkpoints].sort((a, b) => b.made - a.made);
  const bookmarks = Object.keys(system.bookmarks ?? {}).sort();
  const start = text(settings.start);
  const startWhy = (id: string): string | null => {
    const checkpoint = known.checkpoint(id);
    if (!checkpoint) return null;
    if (checkpoint.weights == null) return "its weights were deleted";
    if (weights === "full" && checkpoint.kind !== "full") return "an adapter: merge it first";
    if (weights === "lora" && checkpoint.kind === "full" && trainer?.format === "tinker") return "Tinker trains only from checkpoints Tinker made";
    return null;
  };
  const changed = trainer?.settings.filter(each => settings[each.key] !== undefined && !same(settings[each.key], each.default)) ?? [];
  const [open, setOpen] = useState(false);
  const opened = open || trainer?.settings.some(each => refusals.some(refusal => refusal.key === each.key));
  return (
    <div className="fields">
      <div className="field-row">
        <Field label="Trainer" keys={["trainer.provider"]} refusals={refusals} marked={marked("trainer.provider")}
          note={trainer ? <small>{trainer.kind} · {trainer.allocation}</small> : null}>
          <Picker value={settings["trainer.provider"]} choices={trainerChoices(offers, weights)} onChange={picked => setUpstream({ "trainer.provider": picked })} label="trainer" />
        </Field>
        <Field label="Model" keys={["trainer.model"]} refusals={refusals} marked={marked("trainer.model")}>
          <Picker value={settings["trainer.model"]} choices={modelChoices(offers, trainer, families)} onChange={picked => setUpstream({ "trainer.model": picked })} label="model" />
        </Field>
        <Field label="Starts from" keys={["start"]} refusals={refusals} marked={marked("start")}>
          <select value={start} onChange={event => set({ start: event.target.value || undefined })} aria-label="starts from">
            <option value="">the base model</option>
            {bookmarks.length ? (
              <optgroup label="Bookmarks">
                {bookmarks.map(mark => {
                  const why = startWhy(system.bookmarks[mark]);
                  return <option key={`b${mark}`} value={mark} disabled={why != null} title={why ?? undefined}>{mark} · {known.short(system.bookmarks[mark])}{why ? ` — ${why}` : ""}</option>;
                })}
              </optgroup>
            ) : null}
            {checkpoints.length ? (
              <optgroup label="Checkpoints">
                {checkpoints.map(checkpoint => {
                  const why = startWhy(checkpoint.id);
                  return <option key={checkpoint.id} value={checkpoint.id} disabled={why != null} title={why ?? undefined}>{known.origin(checkpoint.id)} · {checkpoint.short}{why ? ` — ${why}` : ""}</option>;
                })}
              </optgroup>
            ) : null}
          </select>
        </Field>
      </div>
      {trainer?.settings.length ? (
        <details className="advanced" open={opened} onToggle={event => setOpen((event.target as HTMLDetailsElement).open)}>
          <summary>Advanced{changed.length ? <span className="faint"> · {changed.length} changed</span> : null}</summary>
          <div className="settings-grid">
            {trainer.settings.map(spec => {
              const isChanged = changed.includes(spec);
              return (
                <label key={spec.key} className={`setting${isChanged ? " changed" : ""}${marked(spec.key) ? " from-preset" : ""}`}>
                  <span className="mono">{spec.key.slice("trainer.".length)}</span>
                  <input value={shown(settings[spec.key])} placeholder={shown(spec.default) || "none"} spellCheck={false}
                    onChange={event => (event.target.value.trim() ? set({ [spec.key]: typed(event.target.value) }) : unset([spec.key]))} />
                  <Said refusals={refusals} keys={[spec.key]} />
                </label>
              );
            })}
          </div>
        </details>
      ) : null}
    </div>
  );
}

function ChannelFields({ offers, settings, channel, slot, role, trainer, weights, judges = false, refusals, marked, set, setUpstream }: StepProps & {
  channel: string; slot?: string; role: Role; trainer: OfferedTrainer | undefined; weights: string | null; judges?: boolean;
  setUpstream: (changes: Record<string, unknown>) => void;
}) {
  const key = (setting: string) => `channels.${channel}.${setting}`;
  const providerName = settings[key("provider")];
  const provider = offers.inference.find(each => each.name === providerName);
  const serving = role !== "fixed";
  const providers = providerChoices(offers, role, trainer?.name, weights);
  const models = channelModelChoices(provider, settings["trainer.model"] as string | undefined, serving);
  const renderers = renderersOf(provider, settings[key("model")]);
  const bridge = serving ? bridgeOf(offers, trainer?.name, provider?.name) : [];
  const trained = role === "trained";
  const bound = slot ? { [`slots.${slot}`]: channel } : {};
  const chooseProvider = (picked: string) => {
    if (trained) return setUpstream({ [key("provider")]: picked });
    const next = offers.inference.find(each => each.name === picked);
    const model = pick(channelModelChoices(next, settings["trainer.model"] as string | undefined, serving), settings[key("model")]);
    set({ ...bound, [key("provider")]: picked, [key("model")]: model, [key("renderer")]: rendererFor(next, model, settings[key("renderer")]) });
  };
  const chooseMode = (mode: string) => {
    if (mode === "follows") {
      const fits = provider && weights ? servesWhy(provider, weights) == null : true;
      set({ ...bound, [key("mode")]: "follows", [key("follows")]: trainedOf(settings), ...(fits ? {} : { [key("provider")]: pick(providerChoices(offers, "follows", trainer?.name, weights)) }) });
    } else set({ ...bound, [key("mode")]: "fixed", [key("follows")]: undefined, [key("lag")]: undefined });
  };
  return (
    <fieldset className="entry">
      <legend>{slot && slot !== channel ? `${slot} · ${channel}` : channel}</legend>
      <div className="field-row">
        <Field label="Provider" keys={[key("provider"), key("providers")]} refusals={refusals} marked={marked(key("provider"))}
          note={provider ? <small>{provider.kind} · {provider.allocation}{bridge.length ? ` · ${bridge.join(" → ")}` : ""}{provider.allocation === "metered" && pricesOf(provider, settings[key("model")]) ? ` · ${pricesOf(provider, settings[key("model")])}` : ""}</small> : null}>
          <Picker value={providerName} choices={providers} onChange={chooseProvider} label={`${channel} provider`} />
        </Field>
        <Field label="Model" keys={[key("model")]} refusals={refusals} marked={marked(key("model"))}>
          <Picker value={settings[key("model")]} choices={models}
            onChange={picked => set({ ...bound, [key("model")]: picked, [key("renderer")]: rendererFor(provider, picked, settings[key("renderer")]) })} label={`${channel} model`} />
        </Field>
        <Field label="Renderer" keys={[key("renderer")]} refusals={refusals} marked={marked(key("renderer"))}>
          {renderers.length > 1 ? (
            <select value={text(settings[key("renderer")])} onChange={event => set({ [key("renderer")]: event.target.value })} aria-label={`${channel} renderer`}>
              {renderers.map(each => <option key={each} value={each}>{rendererName(each)}</option>)}
            </select>
          ) : (
            <span className="field-value" aria-label={`${channel} renderer`} title={text(settings[key("renderer")])}>
              {provider && provider.capabilities.token_exact === false ? "the provider's own" : renderers.length ? rendererName(renderers[0]) : "none renders this model"}
            </span>
          )}
        </Field>
        {trained ? null : (
          <Field label="Serves" keys={[key("mode"), key("follows")]} refusals={refusals} marked={marked(key("mode"))}>
            <select value={role} onChange={event => chooseMode(event.target.value)} aria-label={`${channel} serves`}>
              <option value="fixed">a fixed model</option>
              <option value="follows">follows {trainedOf(settings)}</option>
            </select>
          </Field>
        )}
        {role === "follows" ? (
          <Field label="Lag" keys={[key("lag")]} refusals={refusals} marked={marked(key("lag"))}>
            <input type="number" min={0} value={text(settings[key("lag")])} placeholder="0" onChange={event => set({ [key("lag")]: asNumber(event.target.value) })} />
          </Field>
        ) : null}
      </div>
      {judges && role === "follows" ? (
        <label className="tick">
          <input type="checkbox" checked={settings.self_judging === true} onChange={event => set({ self_judging: event.target.checked || undefined })} />
          self-judging
        </label>
      ) : null}
      {judges ? <Said refusals={refusals} keys={["self_judging"]} /> : null}
      {slot ? <Said refusals={refusals} keys={[`slots.${slot}`]} /> : null}
    </fieldset>
  );
}

function ObjectiveStep({ offers, settings, trainer, refusals, marked, set, unset }: StepProps & { trainer: OfferedTrainer | undefined; unset: (keys: string[]) => void }) {
  const chosen = objectivePreset(offers, settings["objective.preset"]);
  const components = componentsOf(offers, chosen?.family);
  const groups = objectiveGroups(offers, trainer);
  const changed = components.filter(each => settings[`objective.${each.key}`] !== undefined);
  const refused = refusals.some(each => each.key.startsWith("objective.") && each.key !== "objective.preset");
  const [open, setOpen] = useState(false);
  const value = String(settings["objective.preset"] ?? "default");
  return (
    <div className="fields">
      <Field label="Preset" keys={["objective.preset"]} refusals={refusals} marked={marked("objective.preset")} note={chosen ? <small>{chosen.family}</small> : null}>
        <select value={value} onChange={event => { const keep = components.map(each => `objective.${each.key}`); unset(keep); set({ "objective.preset": event.target.value }); }} aria-label="objective">
          {groups.map(group => (
            <optgroup key={group.family} label={group.family}>
              {group.choices.map(each => <option key={each.value} value={each.value} disabled={each.why != null} title={each.why ?? undefined}>{choiceText(each)}</option>)}
            </optgroup>
          ))}
        </select>
      </Field>
      {components.length ? (
        <details className="advanced" open={open || refused} onToggle={event => setOpen((event.target as HTMLDetailsElement).open)}>
          <summary>Advanced{changed.length ? <span className="faint"> · {changed.length} changed</span> : null}</summary>
          <div className="settings-grid">
            {components.map(component => {
              const key = `objective.${component.key}`;
              const given = settings[key];
              const presetValue = chosen?.components[component.key];
              const choices = component.choices.length ? component.choices : component.types.includes("bool") ? ["true", "false"] : [];
              return (
                <label key={key} className={`setting${given !== undefined ? " changed" : ""}${marked(key) ? " from-preset" : ""}`} title={component.says}>
                  <span className="mono">{component.key}</span>
                  {choices.length ? (
                    <select value={given === undefined ? "" : String(given)} onChange={event => (event.target.value === "" ? unset([key]) : set({ [key]: typed(event.target.value) }))} aria-label={component.key}>
                      <option value="">{shown(presetValue)}</option>
                      {choices.filter(each => each !== String(presetValue)).map(each => <option key={each} value={each}>{each}</option>)}
                    </select>
                  ) : (
                    <input value={shown(given)} placeholder={shown(presetValue) || "none"} spellCheck={false} aria-label={component.key}
                      onChange={event => (event.target.value.trim() ? set({ [key]: typed(event.target.value) }) : unset([key]))} />
                  )}
                  <Said refusals={refusals} keys={[key]} />
                </label>
              );
            })}
          </div>
        </details>
      ) : null}
    </div>
  );
}

/** One step's spend on the run's metered parts, or why it cannot be estimated yet; nothing where nothing is metered. */
function SpendNote({ checked }: { checked: Checked | undefined }) {
  const spend = checked?.spend;
  const parts = useMemo(() => Object.entries(spend?.parts ?? {}), [spend]);
  if (!spend) return null;
  if (spend.dollars == null) return <small>can't be estimated yet: {spend.why}</small>;
  if (!parts.length) return null;
  return (
    <small title={parts.map(([part, dollars]) => `${part}: $${dollars.toFixed(2)}`).join("\n")}>
      ≈ ${spend.dollars.toFixed(2)} a step{parts.length > 1 ? ` (${parts.map(([part, dollars]) => `${part} $${dollars.toFixed(2)}`).join(", ")})` : ` on ${parts[0][0]}`}
    </small>
  );
}
