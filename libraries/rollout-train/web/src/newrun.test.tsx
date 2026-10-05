import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Checked, OfferedProvider, OfferedTrainer, Offers, Preset, PresetVersions, Presets as PresetList, System } from "./api/types";
import {
  checkpointsText, fielded, filled, HOSTED_FOLLOWS, HOSTED_TRAINED, launchSettings, machineChoices, modeChoices, modeOf, pricesOf, providerChoices, settled,
  trainerChoices, weightChoices,
} from "./lib/form";
import { NewRun } from "./pages/NewRun";
import { PresetPage, Presets } from "./pages/Presets";

const GSM8K = "rollout_verifiers.environments:gsm8k";
const QWEN = "Qwen/Qwen3.5-4B", SMALL = "Qwen/Qwen3-0.6B";
const exact = { token_exact: true, sampled_logprobs: true };

const trainer = (name: string, kind: string, weights: "lora" | "full", models: string[]): OfferedTrainer => ({
  name, kind, produces: weights, format: kind === "tinker" ? "tinker" : weights === "full" ? "full" : "peft", models, gpus: 0, colocate_with: null,
  segment_tokens: null, cost: {}, families: ["policy_gradient", "preference"], allocation: kind === "tinker" ? "metered" : "scheduled", concurrency: null,
  weights: [weights], separate: true, settings: [{ key: "trainer.rank", types: ["int"], default: 32, changeable: false }, { key: "trainer.learning_rate", types: ["float"], default: 1e-4, changeable: true }],
});
const provider = (name: string, kind: string, weights: ("lora" | "full")[], models: string[], capabilities: Record<string, unknown> = exact): OfferedProvider => ({
  name, kind, gpus: 0, replicas: 1, capabilities, allocation: kind === "vllm" ? "scheduled" : "metered", concurrency: null, weights,
  models: models.map(model => ({ model, context: 8192, base: null, max_lora_rank: 64, cost: {}, renderers: model === QWEN ? ["rollout_qwen:qwen35"] : [], families: [] })),
});
const preset: Preset = {
  name: "gsm8k-tinker", version: 2, id: "gsm8k-tinker@2", note: "", saved: 1,
  settings: {
    environment: GSM8K, "trainer.provider": "tinker-lora", "channels.policy.provider": "local-vllm", "channels.policy.model": QWEN,
    "channels.policy.renderer": "rollout_qwen:qwen35", "channels.policy.thinking_tokens": 1024, "limits.spend": 2,
  },
};
const offers: Offers = {
  cluster: "home", kinds: ["train"], environments: [{ environment: GSM8K, published: false, families: ["rollout_qwen"] }],
  trainers: [trainer("tinker-lora", "tinker", "lora", [QWEN]), trainer("local-full", "full", "full", [SMALL])],
  inference: [
    provider("local-vllm", "vllm", ["lora", "full"], [QWEN, SMALL]), provider("tinker", "tinker", ["lora"], [QWEN]),
    provider("openai", "api", [], ["gpt-5"], { token_exact: false, sampled_logprobs: false }),
  ],
  pairs: [
    { trainer: "tinker-lora", inference: "local-vllm", together: false, bridge: ["peft-from-tinker"] }, { trainer: "tinker-lora", inference: "tinker", together: false, bridge: ["none"] },
    { trainer: "tinker-lora", inference: "openai", together: false, bridge: null, refused: "the OpenAI Responses API returns text" },
    { trainer: "local-full", inference: "local-vllm", together: false, bridge: ["full-reload"] },
    { trainer: "local-full", inference: "tinker", together: false, bridge: null, refused: "Tinker samples only checkpoints Tinker trained: there is no upload" },
    { trainer: "local-full", inference: "openai", together: false, bridge: null, refused: "the OpenAI Responses API returns text" },
  ],
  sandboxes: {}, presets: [preset], capacity: null,
  objectives: {
    families: ["policy_gradient", "preference", "likelihood", "distillation"],
    presets: [
      { name: "default", family: "policy_gradient", source: "", says: "", components: { "clip.low": 0.2 } },
      { name: "dpo", family: "preference", source: "", says: "", components: { "preference.beta": 0.1 } },
    ],
    components: [
      { key: "clip.low", types: ["float"], families: ["policy_gradient"], changeable: true, says: "The ratio's lower bound", choices: [], least: 0, above: false },
      { key: "preference.beta", types: ["float"], families: ["preference"], changeable: true, says: "The scale", choices: [], least: 0, above: true },
    ],
  },
  schema: [{ key: "groups", types: ["int"], default: 100, changeable: false, says: "", choices: [], least: 1 }],
};
const system = { runs: [], checkpoints: [], bookmarks: {}, names: { runs: {}, bookmarks: {} } } as unknown as System;

// The same, with machines that train and sample together: a local GPU's colocated trainer and engines, and a RunPod
// host pod with the trainer that takes its steps there.
const H100 = { gpu: "NVIDIA H100 80GB HBM3", gpu_types: ["NVIDIA H100 80GB HBM3"], gpu_count: 1, price: 2.69, cloud: "SECURE", regions: [], max_pods: 1, idle_stop: 600, host: null };
const NO_UPLOAD = "Tinker samples only checkpoints Tinker trained: there is no upload", TEXT = "the OpenAI Responses API returns text";
const paired: Offers = {
  ...offers,
  trainers: [
    ...offers.trainers, { ...trainer("local-lora", "lora", "lora", [QWEN]), gpus: 1, colocate_with: "local-vllm" },
    { ...trainer("h100-lora", "runpod-trainer", "lora", [QWEN]), colocate_with: "h100", separate: false, pods: H100 },
  ],
  inference: [{ ...offers.inference[0], gpus: 1 }, ...offers.inference.slice(1), { ...provider("h100", "runpod-host", ["lora"], [QWEN]), allocation: "scheduled", pods: H100 }],
  pairs: [
    ...offers.pairs,
    { trainer: "tinker-lora", inference: "h100", together: false, bridge: ["peft-from-tinker"] },
    { trainer: "local-full", inference: "h100", together: false, bridge: null, refused: "the provider cannot reload full weights in place" },
    { trainer: "local-lora", inference: "local-vllm", together: true, bridge: ["verbatim"] },
    { trainer: "local-lora", inference: "tinker", together: false, bridge: null, refused: NO_UPLOAD },
    { trainer: "local-lora", inference: "openai", together: false, bridge: null, refused: TEXT },
    { trainer: "local-lora", inference: "h100", together: false, bridge: ["verbatim"] },
    { trainer: "h100-lora", inference: "local-vllm", together: false, bridge: ["verbatim"] },
    { trainer: "h100-lora", inference: "tinker", together: false, bridge: null, refused: NO_UPLOAD },
    { trainer: "h100-lora", inference: "openai", together: false, bridge: null, refused: TEXT },
    { trainer: "h100-lora", inference: "h100", together: true, bridge: ["verbatim"] },
  ],
};
const LOCAL_MACHINE = "local-lora\nlocal-vllm", HOST_MACHINE = "h100-lora\nh100";
const bucket = (name: string | null, named: string) => ({ name, kind: "rollout_s3:S3BlobStore", bucket: named, prefix: "blobs/", directory: null });

const answer = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
let checked: Checked;
const asked: { path: string; method: string; body: Record<string, unknown> }[] = [];
const fetched = vi.fn(async (path: string, init?: RequestInit) => {
  const method = init?.method ?? "GET";
  const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
  asked.push({ path, method, body });
  if (path === "api/launches/check") return answer(checked);
  if (path === "api/launches") return answer({ launch: { id: "launch_1", asked: { name: body.name } } });
  if (path.startsWith("api/presets/") && method === "POST") return answer({ preset: { ...preset, name: "mine", id: "mine@1", version: 1 } });
  if (path.startsWith("api/presets/") && method === "DELETE") return answer({ deleted: "gsm8k-tinker" });
  return answer({});
});

function Where() {
  return <output data-testid="where">{useLocation().pathname}</output>;
}

function shown(children: React.ReactNode, at = "/runs/new", more: (client: ReturnType<typeof newQueryClient>) => void = () => {}) {
  const client = newQueryClient();
  client.setQueryData(topics.system().key, system);
  client.setQueryData(topics.offers().key, offers);
  client.setQueryData(topics.evals().key, { suites: [], evals: [] });
  more(client);
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[at]}>{children}<Where /></MemoryRouter></QueryClientProvider>);
}

const select = (label: string) => screen.getByRole("combobox", { name: label }) as HTMLSelectElement;
const option = (label: string, name: string | RegExp) => within(select(label)).getByRole("option", { name }) as HTMLOptionElement;
const fieldOf = (element: HTMLElement) => element.closest(".field") as HTMLElement;

beforeEach(() => {
  asked.length = 0;
  checked = { refusals: [], notes: [], settings: {}, preset: null, weights: "lora", spend: { dollars: 0, parts: {}, why: "" }, environment: { slots: ["policy"], untrained: [], judges: [] } };
  vi.stubGlobal("fetch", fetched);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("the New run form's choices", () => {
  it("filter trainers and providers by the weights, each that cannot work disabled with why", () => {
    expect(weightChoices(offers).map(each => each.why)).toEqual([null, null]);
    expect(weightChoices({ ...offers, trainers: [offers.trainers[0]] })[1].why).toBe("no trainer here trains full weights");
    expect(trainerChoices(offers, "full").map(each => [each.value, each.why])).toEqual([["tinker-lora", "trains a LoRA only"], ["local-full", null]]);
    expect(providerChoices(offers, "trained", "local-full", "full").map(each => [each.value, each.why])).toEqual([
      ["local-vllm", null], ["tinker", "Tinker's sampler serves only checkpoints Tinker trained"], ["openai", HOSTED_TRAINED],
    ]);
    expect(providerChoices(offers, "trained", "tinker-lora", "lora").map(each => each.why)).toEqual([null, null, HOSTED_TRAINED]);
    expect(providerChoices(offers, "follows", "tinker-lora", "lora").map(each => each.why)).toEqual([null, null, HOSTED_FOLLOWS]);
    expect(providerChoices(offers, "fixed", "tinker-lora", "lora").map(each => each.why)).toEqual([null, null, null]);  // (a judge, say)
    expect(providerChoices(offers, "fixed", "tinker-lora", "lora").map(each => each.why)).toEqual([null, null, null]);  // (a judge may be any)
  });

  it("follow a choice upstream, keeping what still fits", () => {
    const lora = settled(offers, { environment: GSM8K });
    expect([lora.weights, lora["trainer.provider"], lora["trainer.model"], lora["channels.policy.provider"], lora["channels.policy.model"], lora["channels.policy.renderer"]])
      .toEqual(["lora", "tinker-lora", QWEN, "local-vllm", QWEN, "rollout_qwen:qwen35"]);
    const full = settled(offers, { ...lora, weights: "full" });
    expect([full["trainer.provider"], full["trainer.model"], full["channels.policy.provider"], full["channels.policy.model"]]).toEqual(["local-full", SMALL, "local-vllm", SMALL]);
    expect(settled(offers, { ...lora, "channels.policy.provider": "tinker" })["channels.policy.provider"]).toBe("tinker");  // (it fits: kept)
  });

  it("say which settings have a field, and ask for a preset's cleared keys as none", () => {
    const shownKeys = { channels: ["policy", "judge"], slots: ["judge"], trainer: ["trainer.rank"] };
    expect(["trainer.rank", "channels.judge.model", "slots.judge", "objective.clip.low", "limits.spend"].every(key => fielded(key, shownKeys))).toBe(true);
    expect(["trainer.wings", "channels.policy.replicas", "channels.rival.model", "seed"].some(key => fielded(key, shownKeys))).toBe(false);
    expect(launchSettings({ groups: 12, bookmark: "", start: undefined, "channels.policy.thinking_tokens": null }, preset)).toEqual({
      groups: 12, "channels.policy.thinking_tokens": null, environment: null, "trainer.provider": null, "channels.policy.provider": null,
      "channels.policy.model": null, "channels.policy.renderer": null, "limits.spend": null,
    });
    const from = filled(offers, preset, {});
    expect([from.weights, from["trainer.model"], from["limits.spend"]]).toEqual(["lora", QWEN, 2]);
  });
});

describe("training and sampling together or separate", () => {
  it("offers together only the pairs that share a machine, each named by it", () => {
    expect(machineChoices(paired, "lora").map(each => [each.value, each.label, each.why])).toEqual([
      [LOCAL_MACHINE, "local-vllm: 1 GPU", null], [HOST_MACHINE, "h100: NVIDIA H100 80GB HBM3 · $2.69 an hour", null],
    ]);
    expect(machineChoices(paired, "full").map(each => each.why)).toEqual(["trains a LoRA only", "trains a LoRA only"]);
    expect(modeChoices(paired, "full").map(each => [each.value, each.why])).toEqual([["together", "no machine here trains and samples"], ["separate", null]]);
    expect(modeChoices(offers, "lora")[0].why).toBe("no machine here trains and samples");  // (no pair shares one)
  });

  it("offers separate a trainer and a provider apart, from the pairs that can work", () => {
    expect(trainerChoices(paired, "lora", "separate").map(each => [each.value, each.why])).toEqual([
      ["tinker-lora", null], ["local-full", "trains full weights only"], ["local-lora", null], ["h100-lora", "trains only beside h100"],
    ]);
    expect(providerChoices(paired, "trained", "local-lora", "lora", "separate").map(each => [each.value, each.why])).toEqual([
      ["local-vllm", "shares local-lora's machine"], ["tinker", NO_UPLOAD], ["openai", HOSTED_TRAINED], ["h100", null],
    ]);
    expect(providerChoices(paired, "trained", "tinker-lora", "lora", "separate").filter(each => each.why == null).map(each => each.value))
      .toEqual(["local-vllm", "tinker", "h100"]);  // (Tinker's trainer with vLLM here, through a bridge; or Tinker's sampler; or a pod)
  });

  it("fills the trainer and the trained channel's provider from the mode chosen", () => {
    const start = settled(paired, { environment: GSM8K });
    expect([modeOf(paired, start), start["trainer.provider"], start["channels.policy.provider"]]).toEqual(["separate", "tinker-lora", "local-vllm"]);
    const together = settled(paired, start, "together");
    expect([modeOf(paired, together), together["trainer.provider"], together["channels.policy.provider"], together["channels.policy.model"]])
      .toEqual(["together", "local-lora", "local-vllm", QWEN]);
    const apart = settled(paired, together, "separate");
    expect([modeOf(paired, apart), apart["trainer.provider"], apart["channels.policy.provider"]]).toEqual(["separate", "local-lora", "h100"]);
    const full = settled(paired, { ...together, weights: "full" }, "together");  // (no machine trains full weights: separate)
    expect([modeOf(paired, full), full["trainer.provider"], full["channels.policy.provider"]]).toEqual(["separate", "local-full", "local-vllm"]);
  });

  it("chooses a machine together, and a trainer and a provider separate, in the form", async () => {
    shown(<NewRun />, "/runs/new", client => client.setQueryData(topics.offers().key, paired));
    expect((screen.getByRole("radio", { name: "Separate" }) as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByRole("combobox", { name: "machine" })).toBeNull();
    expect(option("trainer", /h100-lora/).textContent).toBe("h100-lora — trains only beside h100");
    fireEvent.click(screen.getByRole("radio", { name: "Together" }));
    expect(select("machine").value).toBe(LOCAL_MACHINE);
    expect(within(select("machine")).getAllByRole("option").map(each => each.textContent)).toEqual(["local-vllm: 1 GPU", "h100: NVIDIA H100 80GB HBM3 · $2.69 an hour"]);
    expect(screen.queryByRole("combobox", { name: "trainer" })).toBeNull();
    expect(screen.getByLabelText("policy provider").textContent).toBe("local-vllm");
    fireEvent.change(select("machine"), { target: { value: HOST_MACHINE } });
    expect(screen.getByLabelText("policy provider").textContent).toBe("h100");
    fireEvent.change(screen.getByRole("textbox", { name: "name" }), { target: { value: "on the pod" } });
    fireEvent.click(screen.getByRole("button", { name: "Launch the run" }));
    await waitFor(() => expect(asked.some(each => each.path === "api/launches")).toBe(true));
    const settings = asked.find(each => each.path === "api/launches")!.body.settings as Record<string, unknown>;
    expect([settings["trainer.provider"], settings["channels.policy.provider"], settings["channels.policy.model"]]).toEqual(["h100-lora", "h100", QWEN]);
    fireEvent.click(screen.getByRole("radio", { name: "Separate" }));
    expect(select("trainer").value).toBe("tinker-lora");  // (h100-lora trains only beside its host)
    expect(option("policy provider", /^h100/).disabled).toBe(false);
  });

  it("says where the run's checkpoints go, beside the mode", async () => {
    checked = { ...checked, checkpoints: { store: bucket("r2", "rollout"), tinker: false, bridges: [], bridged: null } };
    shown(<NewRun />);
    await waitFor(() => expect(screen.getByText("Checkpoints: r2 (rollout)")).toBeTruthy());
    expect(screen.getByText("Checkpoints: r2 (rollout)").closest(".mode-bar")).toBeTruthy();
    const local = bucket(null, "local");
    expect(checkpointsText({ store: local, tinker: false, bridges: [], bridged: null })).toBe("the cluster's bucket (local)");
    expect(checkpointsText({ store: local, tinker: true, bridges: [], bridged: null })).toBe("Tinker");
    expect(checkpointsText({ store: local, tinker: true, bridges: ["peft-from-tinker"], bridged: local })).toBe("Tinker, bridged to the cluster's bucket (local)");
    const files = { name: null, kind: "files", bucket: null, prefix: null, directory: "/var/blobs" };
    expect(checkpointsText({ store: files, tinker: false, bridges: [], bridged: null })).toBe("files in /var/blobs");
  });
});

describe("the New run form", () => {
  it("asks LoRA or full weights first, and offers only the trainers and providers that fit", () => {
    shown(<NewRun />);
    expect((screen.getByRole("radio", { name: "LoRA" }) as HTMLInputElement).checked).toBe(true);
    expect(select("trainer").value).toBe("tinker-lora");
    expect(option("trainer", /local-full/).disabled).toBe(true);
    expect(option("trainer", /local-full/).textContent).toBe("local-full — trains full weights only");
    fireEvent.click(screen.getByRole("radio", { name: "Full weights" }));
    expect(select("trainer").value).toBe("local-full");
    expect(select("model").value).toBe(SMALL);
    expect(option("trainer", /tinker-lora/).disabled).toBe(true);
    const tinker = option("policy provider", /^tinker/);
    expect(tinker.disabled).toBe(true);
    expect(tinker.textContent).toBe("tinker — Tinker's sampler serves only checkpoints Tinker trained");
    expect(select("policy provider").value).toBe("local-vllm");
    expect(select("policy model").value).toBe(SMALL);
  });

  it("disables a choice that can never work, with its reason as visible text and a tooltip", () => {
    shown(<NewRun />);
    const openai = option("policy provider", /openai/);  // (a hosted API, on the trained channel)
    expect(openai.disabled).toBe(true);
    expect(openai.title).toBe(HOSTED_TRAINED);
    expect(openai.textContent).toBe(`openai — ${HOSTED_TRAINED}`);
    expect(fieldOf(select("policy provider")).textContent).toContain("peft-from-tinker");  // (the pair's bridge)
  });

  it("is filled by a preset, marking the fields it set, and saves itself as one", async () => {
    shown(<NewRun />);
    fireEvent.change(select("preset"), { target: { value: "gsm8k-tinker@2" } });
    expect(select("trainer").value).toBe("tinker-lora");
    expect((screen.getByRole("spinbutton", { name: "Thinking tokens" }) as HTMLInputElement).value).toBe("1024");
    expect(fieldOf(select("trainer")).classList.contains("from-preset")).toBe(true);
    expect(fieldOf(screen.getByRole("spinbutton", { name: "Groups" })).classList.contains("from-preset")).toBe(false);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Thinking tokens" }), { target: { value: "512" } });
    expect(fieldOf(screen.getByRole("spinbutton", { name: "Thinking tokens" })).classList.contains("from-preset")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Save as preset" }));
    fireEvent.change(screen.getByRole("textbox", { name: "preset name" }), { target: { value: "mine" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(asked.some(each => each.path === "api/presets/mine" && each.method === "POST")).toBe(true));
    const saved = asked.find(each => each.path === "api/presets/mine")!.body.settings as Record<string, unknown>;
    expect([saved["trainer.provider"], saved["channels.policy.thinking_tokens"], saved.weights]).toEqual(["tinker-lora", 512, "lora"]);
  });

  it("says each refusal beside its field, the notes and the spend near the submit button, live", async () => {
    checked = {
      ...checked,
      refusals: [
        { rule: "rank", key: "trainer.rank", reason: "trainer.rank 32 times 3 is 96, above provider local-vllm's max_lora_rank 64", refuses: true },
        { rule: "models", key: "channels.policy.model", reason: "provider local-vllm does not serve it", refuses: true },
        { rule: "settings", key: "channels.policy.replicas", reason: "too many replicas", refuses: true },
      ],
      notes: [{ rule: "capacity", key: "trainer.provider", reason: "the run needs 1 GPUs, and 0 are free: it waits", refuses: false }],
      spend: { dollars: 0.42, parts: { "tinker-lora": 0.42 }, why: "" },
    };
    shown(<NewRun />);
    await waitFor(() => expect(screen.getByText("provider local-vllm does not serve it")).toBeTruthy());
    expect(fieldOf(screen.getByText("provider local-vllm does not serve it"))).toBe(fieldOf(select("policy model")));
    expect(screen.getByText(/above provider local-vllm's max_lora_rank 64/).closest(".setting")?.textContent).toContain("rank");
    expect(screen.getByText("the run needs 1 GPUs, and 0 are free: it waits").closest("ul")?.classList.contains("notes")).toBe(true);
    expect([...document.querySelectorAll("ul.refusals:not(.notes) li")].map(each => each.textContent)).toEqual(["channels.policy.replicas: too many replicas"]);
    expect(screen.getByText("≈ $0.42 a step on tinker-lora")).toBeTruthy();
    const body = asked.find(each => each.path === "api/launches/check")!.body;
    expect([body.kind, body.environment, (body.settings as Record<string, unknown>).weights]).toEqual(["train", GSM8K, "lora"]);
  });

  it("says why a step's spend cannot be estimated yet", async () => {
    checked = { ...checked, spend: { dollars: null, parts: {}, why: "the environment does not say how many turns an episode takes" } };
    shown(<NewRun />);
    await waitFor(() => expect(screen.getByText("can't be estimated yet: the environment does not say how many turns an episode takes")).toBeTruthy());
  });

  it("binds the environment's other slots to channels of their own, a judge following the run with self-judging", async () => {
    checked = { ...checked, environment: { slots: ["policy", "judge"], untrained: ["judge"], judges: ["judge"] } };
    shown(<NewRun />);
    await waitFor(() => expect(select("judge provider")).toBeTruthy());
    expect(option("judge provider", /openai/).disabled).toBe(false);  // (a fixed judge may be any)
    fireEvent.change(select("judge provider"), { target: { value: "openai" } });
    fireEvent.change(select("judge serves"), { target: { value: "follows" } });
    expect(select("judge provider").value).toBe("local-vllm");  // (one following the run serves its LoRA)
    fireEvent.click(screen.getByRole("checkbox", { name: "self-judging" }));
    fireEvent.change(screen.getByRole("textbox", { name: "name" }), { target: { value: "judged" } });
    fireEvent.click(screen.getByRole("button", { name: "Launch the run" }));
    await waitFor(() => expect(asked.some(each => each.path === "api/launches")).toBe(true));
    const settings = asked.find(each => each.path === "api/launches")!.body.settings as Record<string, unknown>;
    expect([settings["slots.judge"], settings["channels.judge.mode"], settings["channels.judge.follows"], settings.self_judging]).toEqual(["judge", "follows", "policy", true]);
  });

  it("binds a judge to a hosted API, shown metered with its model's prices and no renderer of its own", async () => {
    checked = { ...checked, environment: { slots: ["policy", "judge"], untrained: ["judge"], judges: ["judge"] } };
    const priced = { model: "gpt-5", context: 400_000, base: null, max_lora_rank: null, cost: { input: 1.25, output: 10 }, renderers: [], families: [] };
    const hosted = { ...offers, inference: offers.inference.map(each => (each.name === "openai" ? { ...each, models: [priced] } : each)) };
    shown(<NewRun />, "/runs/new", client => client.setQueryData(topics.offers().key, hosted));
    await waitFor(() => expect(select("judge provider")).toBeTruthy());
    fireEvent.change(select("judge provider"), { target: { value: "openai" } });
    expect(fieldOf(select("judge provider")).textContent).toContain("api · metered · $1.25 in · $10 out a million tokens");
    expect(screen.getByLabelText("judge renderer").textContent).toBe("the provider's own");
    expect(pricesOf(hosted.inference[2], "gpt-5")).toBe("$1.25 in · $10 out a million tokens");
    expect(pricesOf(hosted.inference[0], QWEN)).toBe("");  // (nothing priced: placed, not metered)
  });

  it("posts the run's settings and goes to the runs", async () => {
    shown(<NewRun />);
    fireEvent.change(screen.getByRole("textbox", { name: "name" }), { target: { value: "gsm8k-tinker-4b" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Limit ($)" }), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Launch the run" }));
    await waitFor(() => expect(screen.getByTestId("where").textContent).toBe("/runs"));
    const launched = asked.find(each => each.path === "api/launches")!.body;
    expect([launched.kind, launched.name, launched.environment]).toEqual(["train", "gsm8k-tinker-4b", GSM8K]);
    const settings = launched.settings as Record<string, unknown>;
    expect([settings["trainer.provider"], settings.weights, settings["channels.policy.provider"], settings["limits.spend"], settings["evals.suite"]])
      .toEqual(["tinker-lora", "lora", "local-vllm", 2, null]);
  });
});

describe("the presets page", () => {
  const versions: PresetVersions = {
    name: "gsm8k-tinker",
    versions: [{ ...preset, version: 1, id: "gsm8k-tinker@1", note: "first", settings: { ...preset.settings, "limits.spend": 1 } }, preset],
  };
  const listed: PresetList = { presets: [{ ...preset, versions: 2 }], keeps: true };

  it("lists every preset's newest version", () => {
    shown(<Presets />, "/presets", client => client.setQueryData(topics.presets().key, listed));
    const link = screen.getByRole("link", { name: "gsm8k-tinker" });
    expect(link.getAttribute("href")).toBe("/presets/gsm8k-tinker");
    expect(link.closest("tr")?.textContent).toContain("v2 of 2");
  });

  it("shows a version's settings with what it changed, saves an edit as the next version, and deletes it", async () => {
    vi.stubGlobal("confirm", () => true);
    shown(<PresetPage name="gsm8k-tinker" />, "/presets/gsm8k-tinker", client => client.setQueryData(topics.preset("gsm8k-tinker").key, versions));
    expect(select("version").value).toBe("2");
    expect(screen.getByText("limits.spend").closest("td")?.classList.contains("changed")).toBe(true);
    expect(screen.getByText("trainer.provider").closest("td")?.classList.contains("changed")).toBe(false);
    fireEvent.change(select("version"), { target: { value: "1" } });
    expect(screen.getByText("limits.spend").closest("td")?.classList.contains("changed")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByRole("textbox", { name: "limits.spend value" }), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Save v3" }));
    await waitFor(() => expect(asked.some(each => each.path === "api/presets/gsm8k-tinker" && each.method === "POST")).toBe(true));
    const saved = asked.find(each => each.method === "POST")!.body;
    expect((saved.settings as Record<string, unknown>)["limits.spend"]).toBe(3);
    expect((saved.settings as Record<string, unknown>)["trainer.provider"]).toBe("tinker-lora");
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(screen.getByTestId("where").textContent).toBe("/presets"));
    expect(asked.some(each => each.path === "api/presets/gsm8k-tinker" && each.method === "DELETE")).toBe(true);
  });
});
