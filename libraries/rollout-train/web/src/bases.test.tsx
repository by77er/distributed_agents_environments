import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Checkpoint, EvalSuite, Launches, Lineage, LineageCheckpoint, OfferedProvider, Offers, Preset, SubjectHistory, SuiteVersion, System } from "./api/types";
import { baseModelsOf, evalSettingsOf, playedBy, providersOf, spendText } from "./components/play";
import { basePlace, placeOf } from "./lib/places";
import { Base } from "./pages/Base";
import { Checkpoints } from "./pages/Checkpoints";
import { Suite } from "./pages/Suite";
import { Tree } from "./layout/Tree";

const A = "kpqxlmnoprstuvwx", B = "vwxyzabcdefghijk";

const drawn = (id: string, short: string, base: string, run: string, name: string): LineageCheckpoint => ({
  id, short, depth: 1, parents: [], base, kind: "lora", made: 1, kept: true, released: null, bookmarks: [], metrics: {},
  by: { run, name, step: 1 }, life: { state: "written", reshard: false, resharding: null, resharded: null, latest_of: null, workers: {} },
});

const lineage: Lineage = {
  now: 100, bases: ["org/base-a", "org/base-b"], outside: [], bookmarks: {}, trainers: [], workers: [],
  checkpoints: [drawn(A, "kpqx", "org/base-a", "run_1", "first"), drawn(B, "vwxy", "org/base-b", "run_2", "second")],
  runs: [{ run: "run_1", name: "first", from: null, checkpoints: [A], latest: A }, { run: "run_2", name: "second", from: null, checkpoints: [B], latest: B }],
  edges: [{ kind: "base", from: "base:org/base-a", to: A }, { kind: "base", from: "base:org/base-b", to: B }],
};

const checkpoint = (id: string, short: string, base: string, run: string): Checkpoint => ({
  id, short, depth: 1, parents: [], base, kind: "lora", run, step: 1, made: 1, metrics: {}, weights: { files: 1, bytes: 1 },
  state: null, released: null, bookmarks: [],
});

const system: System = {
  at: 100, name: "runs", directory: null, ledger_at: "/ledger", host: "here", written: 10, runs: [], bookmarks: {}, runners: [], channels: [],
  checkpoints: [checkpoint(A, "kpqx", "org/base-a", "run_1"), checkpoint(B, "vwxy", "org/base-b", "run_2")],
  ledger: { fences: {}, tables: {} }, kept: { checkpoints: 0, episodes: 0 }, names: { runs: { run_1: "first", run_2: "second" }, bookmarks: {} },
};

const version: SuiteVersion = {
  id: "words-v1@1", number: 1, environments: ["games:words"], made: 1, held_out: false,
  entries: [{ environment: "games:words", environment_version: "1", chosen: "rows and seeds", eval_data: null, rows: null, seeds: null, held_out: false,
    episodes: 1, thinking_tokens: null, answer_tokens: null, offset: 0, starts: 1 }],
  starts: [{ start: "1", environment: "games:words", task: "say-yes", seed: 1, title: "say-yes", identity: "say-yes|1" }],
};
const suite: EvalSuite = { suite: "words-v1", version: version.id, number: 1, environments: ["games:words"], made: 1, starts: version.starts, versions: [version], subjects: [] };

const provider = (name: string, models: string[], renderers: string[] = []): OfferedProvider => ({
  name, kind: "vllm", gpus: 1, replicas: 1, capabilities: {}, allocation: "scheduled", concurrency: null, weights: ["lora", "full"],
  models: models.map(model => ({ model, context: 4096, base: null, max_lora_rank: 32, cost: {}, renderers, families: [] })),
});
const preset = (name: string, provider: string, model: string): Preset => ({
  name, version: 1, id: `${name}@1`, note: "", saved: 1,
  settings: { "trainer.provider": "lora", "trainer.rank": 32, "channels.policy.provider": provider, "channels.policy.model": model,
    "channels.policy.renderer": "rollout_qwen:qwen3" },
});
const offers: Offers = {
  cluster: "here", kinds: ["train", "eval"], environments: [{ environment: "games:words", published: false }], trainers: [], pairs: [],
  inference: [provider("local", ["org/base-a", "org/base-b"]), provider("far", ["org/elsewhere"], ["rollout_qwen:qwen35"])],
  sandboxes: {}, presets: [preset("vllm", "local", "org/base-a"), preset("tinker", "far", "org/elsewhere")], capacity: null,
};
const launches: Launches = { launches: [], submits: true };

function Where() {
  return <output data-testid="where">{useLocation().pathname}</output>;
}

// (the launch asked for, and the launches read again after it)
const fetched = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(async (_, init) => new Response(
  JSON.stringify(init?.method === "POST" ? { launch: { id: "launch_1", asked: { name: "asked" } } } : launches),
  { headers: { "Content-Type": "application/json" } },
));

function shown(children: React.ReactNode, at: string, more: (client: ReturnType<typeof newQueryClient>) => void = () => {}) {
  const client = newQueryClient();
  client.setQueryData(topics.system().key, system);
  client.setQueryData(topics.checkpoints().key, lineage);
  client.setQueryData(topics.launches().key, launches);
  client.setQueryData(topics.offers().key, offers);
  client.setQueryData(topics.evals().key, { suites: [suite], evals: [] });
  more(client);
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[at]}>{children}<Where /></MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetched);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  fetched.mockClear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("a base model in the checkpoints' graph", () => {
  it("is a place of its own, by name", () => {
    expect(basePlace("org/base-b")).toBe("/base/org%2Fbase-b");
    expect(placeOf("/base/org%2Fbase-b")).toEqual({ page: "checkpoints", kind: "base", model: "org/base-b" });
    expect(placeOf("/base/org/base-b")).toEqual({ page: "checkpoints", kind: "base", model: "org/base-b" });
  });

  it("is selected as a checkpoint is: by a click, or by Enter once it has the focus", () => {
    shown(<Checkpoints />, "/checkpoints");
    const base = screen.getByRole("link", { name: "the base model org/base-b" });
    expect(base.getAttribute("tabindex")).toBe("0");
    fireEvent.click(base);
    expect(placeOf(screen.getByTestId("where").textContent!)).toEqual({ page: "checkpoints", kind: "base", model: "org/base-b" });
    fireEvent.keyDown(screen.getByRole("link", { name: "the base model org/base-a" }), { key: "Enter" });
    expect(screen.getByTestId("where").textContent).toBe(basePlace("org/base-a"));
    fireEvent.keyDown(screen.getByRole("link", { name: "the checkpoint kpqx" }), { key: " " });
    expect(screen.getByTestId("where").textContent).toBe(`/checkpoint/${A}`);
  });
});

describe("the checkpoints' sidebar", () => {
  it("lists every base model the graph has: those trained from, then those only evals have played", () => {
    const evaluated = { ...lineage, bases: [...lineage.bases, "org/evaluated"] };
    shown(<Tree place={{ page: "checkpoints", kind: "checkpoints" }} />, "/checkpoints", client => client.setQueryData(topics.checkpoints().key, evaluated));
    const label = screen.getByText("Base models");
    const rows = [...label.parentElement!.querySelectorAll(".node")].filter(row => row.querySelector(".tag") === null).map(row => row.textContent);
    expect(rows).toEqual(["base-a", "base-b", "evaluated"]);
  });
});

describe("a base model's page", () => {
  const history: SubjectHistory = {
    subject: { kind: "model", id: "org/base-b", short: "org/base-b", run: null, name: null, step: null, base: null, bookmarks: [], evals: [], suites: [], started: null, playing: 0 },
    evals: [],
  };

  it("lists the runs trained from it, and asks for an eval of it with a preset's channel settings", async () => {
    shown(<Base model="org/base-b" />, basePlace("org/base-b"), client => client.setQueryData(topics.history("model", "org/base-b").key, history));
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("org/base-b");
    const runs = screen.getByRole("heading", { name: "Runs" }).closest("section")!;
    expect(runs.textContent).toContain("second");
    expect(runs.textContent).not.toContain("first");
    const form = screen.getByRole("button", { name: "Run an eval" }).closest("form")!;
    expect(form.querySelector<HTMLInputElement>("input[placeholder='words-v1 on base-b']")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Train from org/base-b" }).getAttribute("href")).toBe("/runs/new?model=org%2Fbase-b");
    fireEvent.click(screen.getByRole("button", { name: "Run an eval" }));
    await waitFor(() => expect(fetched).toHaveBeenCalledWith("api/launches", expect.objectContaining({ method: "POST" })));
    const body = JSON.parse(String(fetched.mock.calls.find(([path]) => path === "api/launches")![1]!.body));
    expect(body).toEqual({
      kind: "eval", name: "words-v1 on base-b", environment: "games:words", preset: null,
      settings: { "eval.suite": "words-v1@1", "channels.policy.provider": "local", "channels.policy.model": "org/base-b",
        "channels.policy.renderer": "rollout_qwen:qwen3" },
    });  // (the preset's trainer settings left out: an eval trains nothing)
  });

  it("says when the cluster offers no provider of it", () => {
    shown(<Base model="org/nowhere" />, basePlace("org/nowhere"));
    expect(screen.getByText("the cluster offers no provider of org/nowhere")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Run an eval" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("is a root with no lanes when the cluster offers it and nothing has used it, and asks for its first eval", () => {
    const offered = { ...lineage, bases: [...lineage.bases, "org/unused"] };
    shown(<Base model="org/unused" />, basePlace("org/unused"), client => {
      client.setQueryData(topics.checkpoints().key, offered);
      client.setQueryData(topics.offers().key, { ...offers, inference: [...offers.inference, provider("more", ["org/unused"])] });
    });
    expect(screen.getByRole("heading", { name: "Runs" }).closest("section")!.textContent).toContain("None.");
    expect((screen.getByRole("button", { name: "Run an eval" }) as HTMLButtonElement).disabled).toBe(false);
    cleanup();
    shown(<Tree place={{ page: "checkpoints", kind: "checkpoints" }} />, "/checkpoints", client => client.setQueryData(topics.checkpoints().key, offered));
    const rows = [...screen.getByText("Base models").parentElement!.querySelectorAll(".node")].filter(row => row.querySelector(".tag") === null);
    expect(rows.map(row => row.textContent)).toEqual(["base-a", "base-b", "unused"]);
  });
});

describe("a suite's eval form", () => {
  it("offers the cluster's base models to play it, and a preset whose model is the one chosen", () => {
    const { container } = shown(<Suite name="words-v1" />, "/evals/words-v1");
    const group = screen.getByRole("group", { name: "Base models" });
    expect([...group.querySelectorAll("option")].map(option => option.textContent)).toEqual(["org/base-a", "org/base-b", "org/elsewhere"]);
    const [played, , chosen] = [...container.querySelectorAll("form select")] as HTMLSelectElement[];
    expect(played.value).toBe("base:org/base-a");
    expect(chosen.value).toBe("vllm@1");
    fireEvent.change(played, { target: { value: "base:org/elsewhere" } });
    expect(chosen.value).toBe("tinker@1");
  });
});

describe("an eval's settings", () => {
  it("are a preset's channel settings, with the base model played and a provider that serves it", () => {
    expect(baseModelsOf(offers)).toEqual(["org/base-a", "org/base-b", "org/elsewhere"]);
    expect(evalSettingsOf(offers.presets[0])).toEqual({
      "channels.policy.provider": "local", "channels.policy.model": "org/base-a", "channels.policy.renderer": "rollout_qwen:qwen3",
    });
    expect(playedBy(offers, offers.presets[0], "org/elsewhere")).toEqual({
      "channels.policy.provider": "far", "channels.policy.model": "org/elsewhere", "channels.policy.renderer": "rollout_qwen:qwen35",
    });
    expect(playedBy(offers, undefined, undefined)).toEqual({});
  });
});

describe("an eval of a hosted model", () => {
  const hosted: OfferedProvider = {
    name: "anthropic", kind: "api", gpus: 0, replicas: 1, capabilities: { token_exact: false, sampled_logprobs: false },
    allocation: "metered", concurrency: 16, weights: [],
    models: [{ model: "claude-haiku-4-5-20251001", context: 200_000, base: null, max_lora_rank: null, cost: { input: 1, cached_input: 0.1, output: 5 },
      renderers: [], families: [] }],
  };
  const served = { ...offers, inference: [...offers.inference, hosted, { ...provider("also", ["org/base-a"]) }] };
  const checked = { refusals: [], notes: [], settings: {}, preset: null, weights: null, environment: null,
    spend: { dollars: 0.42, parts: { anthropic: 0.42 }, why: "", per: "eval" as const } };
  const posted: { path: string; body: Record<string, unknown> }[] = [];
  const answering = vi.fn(async (path: string, init?: RequestInit) => {
    if (init?.method === "POST") posted.push({ path, body: JSON.parse(String(init.body)) as Record<string, unknown> });
    const answer = path === "api/launches/check" ? checked : init?.method === "POST" ? { launch: { id: "launch_1", asked: { name: "asked" } } } : launches;
    return new Response(JSON.stringify(answer), { headers: { "Content-Type": "application/json" } });
  });

  beforeEach(() => {
    posted.length = 0;
    vi.stubGlobal("fetch", answering);
  });

  it("plays on a provider that serves it, a hosted API naming no renderer", () => {
    expect(providersOf(served, "org/base-a").map(each => each.name)).toEqual(["local", "also"]);
    expect(playedBy(served, offers.presets[0], "org/base-a", "also")["channels.policy.provider"]).toBe("also");
    expect(playedBy(served, offers.presets[0], "claude-haiku-4-5-20251001")).toEqual({
      "channels.policy.provider": "anthropic", "channels.policy.model": "claude-haiku-4-5-20251001",
    });  // (the preset's renderer is for its own model, and a hosted API renders messages itself)
    expect(spendText(undefined)).toBeNull();
    expect(spendText({ ...checked, spend: { dollars: null, parts: {}, why: "the suite's starts are not known here" } }))
      .toBe("spend can't be estimated yet: the suite's starts are not known here");
  });

  it("is asked for from the suite's page as metered, with its estimated spend and a limit", async () => {
    shown(<Suite name="words-v1" />, "/evals/words-v1", client => client.setQueryData(topics.offers().key, served));
    const played = screen.getByRole("group", { name: "Base models" }).closest("select")!;
    expect([...played.querySelectorAll("option")].map(option => option.textContent)).toContain("claude-haiku-4-5-20251001");
    fireEvent.change(played, { target: { value: "base:claude-haiku-4-5-20251001" } });
    const chosen = screen.getByRole("combobox", { name: "provider" }) as HTMLSelectElement;
    expect([...chosen.options].map(option => option.textContent)).toEqual(["anthropic · metered"]);
    await waitFor(() => expect(chosen.closest(".field")!.textContent).toContain("≈ $0.42 for the eval at most"));
    expect(chosen.closest(".field")!.textContent).toContain("$1 in · $5 out a million tokens");
    fireEvent.change(screen.getByRole("spinbutton", { name: "limit" }), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Run this suite" }));
    await waitFor(() => expect(posted.some(each => each.path === "api/launches")).toBe(true));
    const settings = posted.find(each => each.path === "api/launches")!.body.settings as Record<string, unknown>;
    expect(settings).toEqual({
      "eval.suite": "words-v1@1", "channels.policy.provider": "anthropic", "channels.policy.model": "claude-haiku-4-5-20251001",
      "limits.spend": 2,
    });
    const check = posted.find(each => each.path === "api/launches/check")!.body;
    expect([check.kind, (check.settings as Record<string, unknown>)["channels.policy.provider"]]).toEqual(["eval", "anthropic"]);
  });

  it("asks for no check and no limit on a placed provider, and offers no training from a hosted model", () => {
    shown(<Base model="org/base-b" />, basePlace("org/base-b"), client => client.setQueryData(topics.offers().key, served));
    expect(screen.queryByRole("spinbutton", { name: "limit" })).toBeNull();
    expect(posted.filter(each => each.path === "api/launches/check")).toEqual([]);
    cleanup();
    shown(<Base model="claude-haiku-4-5-20251001" />, basePlace("claude-haiku-4-5-20251001"), client => client.setQueryData(topics.offers().key, served));
    expect(screen.queryByRole("link", { name: /Train from/ })).toBeNull();
    expect(screen.getByRole("spinbutton", { name: "limit" })).toBeTruthy();
  });
});
