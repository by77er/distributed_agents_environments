import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Checkpoint, EvalSuite, Launcher, Launches, Lineage, LineageCheckpoint, OfferedProfile, SubjectHistory, SuiteVersion, System } from "./api/types";
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

const offered = (profile: string, model: string, models?: string[]): OfferedProfile => ({ profile, path: `/profiles/${profile}.toml`, model, models, settings: {} });
const launcher: Launcher = {
  launcher: "launcher/far", at: 99, environments: ["games:words"], at_once: 1, playing: 0,
  profiles: [offered("vllm", "org/base-a", ["org/base-a", "org/base-b"]), offered("tinker", "org/elsewhere")],
};
const launches: Launches = { launches: [], launchers: [launcher] };

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

  it("lists the runs trained from it, and asks for an eval it plays with a profile that offers it", async () => {
    shown(<Base model="org/base-b" />, basePlace("org/base-b"), client => client.setQueryData(topics.history("model", "org/base-b").key, history));
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("org/base-b");
    const runs = screen.getByRole("heading", { name: "Runs" }).closest("section")!;
    expect(runs.textContent).toContain("second");
    expect(runs.textContent).not.toContain("first");
    const form = screen.getByRole("button", { name: "Run an eval" }).closest("form")!;
    const profiles = [...form.querySelectorAll("option")].filter(option => option.closest("select")?.value === "vllm").map(option => option.textContent);
    expect(profiles).toEqual(["vllm · org/base-a"]);  // (not tinker's: it does not offer org/base-b)
    expect(form.querySelector<HTMLInputElement>("input[placeholder='words-v1 on base-b']")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Run an eval" }));
    await waitFor(() => expect(fetched).toHaveBeenCalledWith("api/launches", expect.objectContaining({ method: "POST" })));
    const body = JSON.parse(String(fetched.mock.calls.find(([path]) => path === "api/launches")![1]!.body));
    expect(body).toMatchObject({ kind: "eval", suite: "words-v1@1", profile: "vllm", model: "org/base-b", start: null, name: "words-v1 on base-b" });
  });

  it("says when no launcher alive offers it", () => {
    shown(<Base model="org/nowhere" />, basePlace("org/nowhere"));
    expect(screen.getByText("no launcher alive offers org/nowhere")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Run an eval" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("a suite's eval form", () => {
  it("offers the launchers' base models to play it, and the profiles that offer the one chosen", () => {
    const { container } = shown(<Suite name="words-v1" />, "/evals/words-v1");
    const group = screen.getByRole("group", { name: "Base models" });
    expect([...group.querySelectorAll("option")].map(option => option.textContent)).toEqual(["org/base-a", "org/base-b", "org/elsewhere"]);
    const [played, , profile] = [...container.querySelectorAll("form select")] as HTMLSelectElement[];
    expect(played.value).toBe("base:org/base-a");
    fireEvent.change(played, { target: { value: "base:org/elsewhere" } });
    expect([...profile.options].map(option => option.value)).toEqual(["tinker"]);
    fireEvent.change(played, { target: { value: A } });
    expect([...profile.options].map(option => option.value)).toEqual(["vllm", "tinker"]);
  });
});
