import { QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Launch, Offers, Run, System, Checkpoint, Evals, EvalSuite, Path, PathPoint, PlainMessage, SampleLine, SuiteEntry, SuiteVersion } from "./api/types";
import { scale, sparkPoints } from "./components/charts";
import { RunControls } from "./components/control";
import { columnsOf, type Subject } from "./components/evals";
import { pickable, readable } from "./lib/environments";
import { slotHue } from "./lib/format";
import { episodeClass, knownOf, lineOf, reported } from "./lib/model";
import { running } from "./layout/runs";
import { titleOf } from "./layout/Shell";
import { placeOf } from "./lib/places";
import { pathChart } from "./lib/scores";
import { evalsSettings, NO_EVALS, settingOf, wantedOf } from "./lib/settings";
import { ALL, blocksOf, DRAWN, entryBody, fieldsOf, GIVEN, gridRows, limitsText, SAME, suiteBody, versionGroups, versionTag, wholes } from "./lib/suites";
import { mapRows, resultOf, samplesOf, seenOf } from "./pages/Episode";
import { Runs } from "./pages/Runs";
import { Suite } from "./pages/Suite";

const run = (name: string, done: number): Run => ({
  run: name, name, fence: 1, wrote: 10, decided: done, open: [], next: [], channels: [], state: "ended", host: "here",
  address: null, directory: `/runs/${name}`, episodes_at: "here", reached: null, from: null, started: 1,
  starts: 1, written: 10, steps: [],
  done: Array.from({ length: done }, (_, index) => ({
    group: index + 1, time: 10 + index, task: "t003", title: "chests", rollout_seconds: 5, rewards: [1, 0], solved: [true, false],
    durations: [1, 2], failed: 0, failures: [], segments_recorded: 4, segments: 4, skipped: null, unlocked: 3, adapter: null,
    step: null, step_state: null, depth: null, update: null, segments_trained: 0, error: null, seconds: 5,
  })),
});

const system = (runs: Run[]): System => ({
  at: 100, name: "runs", directory: null, ledger_at: "/ledger", host: "here", written: 10, runs, checkpoints: [], bookmarks: {}, runners: [],
  channels: [], ledger: { fences: {}, tables: {} }, kept: { checkpoints: 0, episodes: 0 }, names: { runs: {}, bookmarks: {} },
});

describe("places", () => {
  it("are read from the address after the #, as the page's links write them", () => {
    expect(placeOf("/run/a%2Fb/group/3")).toEqual({ page: "runs", kind: "group", run: "a/b", number: 3 });
    expect(placeOf("/run/x/step/2")).toEqual({ page: "runs", kind: "step", run: "x", number: 2 });
    expect(placeOf("/episode/r_1/agent-2")).toEqual({ page: "runs", kind: "episode", id: "r_1", slot: "agent-2" });
    expect(placeOf("/checkpoints")).toEqual({ page: "checkpoints", kind: "checkpoints" });
    expect(placeOf("/checkpoint/kpqx")).toEqual({ page: "checkpoints", kind: "checkpoint", id: "kpqx" });
    expect(placeOf("/runs/new")).toEqual({ page: "runs", kind: "launch" });
    expect(placeOf("/evals")).toEqual({ page: "evals", kind: "evals" });
    expect(placeOf("/evals/words-v1")).toEqual({ page: "evals", kind: "suite", suite: "words-v1" });
    expect(placeOf("/eval/run_1")).toEqual({ page: "evals", kind: "eval", run: "run_1" });
    expect(placeOf("")).toEqual({ page: "runs", kind: "runs" });
  });
});

const checkpoint = (id: string, short: string, run: string | null, step: number | null, parents: string[] = [], bookmarks: string[] = []): Checkpoint => ({
  id, short, depth: parents.length + 1, parents, base: "Qwen/Qwen3.5-9B", kind: "lora", run, step, made: 1, metrics: {}, weights: null, state: null,
  released: null, bookmarks,
});

describe("checkpoints", () => {
  it("are said by where they came from and the shortest start of their id, runs by their names", () => {
    const checkpoints = [checkpoint("kpqxlmnoprstuvwx", "kpqx", "run_1", 3, [], ["good"]), checkpoint("klmnopqrstuvwxyz", "klmn", null, null, ["kpqxlmnoprstuvwx"])];
    const runs = { run_1: "first" }, known = knownOf(checkpoints, runs);
    expect([known.run("run_1"), known.run("other"), known.short("kpqxlmnoprstuvwx"), known.short(null), known.short("zzzzzzzzzzzz")])
      .toEqual(["first", "other", "kpqx", "base", "zzzzzzzz"]);
    expect([known.origin("kpqxlmnoprstuvwx"), known.origin("klmnopqrstuvwxyz"), known.origin("zzzz"), known.origin(null)])
      .toEqual(["first · S3", "made outside a run", "not in this ledger", "the base model"]);
    expect(known.bookmarks("kpqxlmnoprstuvwx")).toEqual(["good"]);
    expect(known.title("klmnopqrstuvwxyz")).toBe("klmnopqrstuvwxyz · depth 2\nmade outside a run, from kpqx");
    expect(knownOf(checkpoints, runs)).toBe(known);  // (the same answer, the same helpers: nothing draws again)
  });

  it("say what a checkpoint builds on: a full checkpoint this ledger has as a checkpoint, else the model by name", () => {
    const merged = { ...checkpoint("mnopqrstuvwxyzkl", "mnop", "run_1", 4), kind: "full" };
    const adapter = { ...checkpoint("nopqrstuvwxyzklm", "nopq", "run_2", 1), base: merged.id };
    const known = knownOf([merged, adapter], { run_1: "first" });
    expect([known.base(adapter.base), known.base("Qwen/Qwen3.5-9B"), known.base(null)]).toEqual(["mnop (first · S4)", "Qwen/Qwen3.5-9B", "the base model"]);
    expect(known.title(adapter.id)).toBe("nopqrstuvwxyzklm · depth 1\nrun_2 · S1, from mnop (first · S4)");
    expect(known.title(merged.id).split("\n")[0]).toBe("mnopqrstuvwxyzkl · depth 1 · full weights");
  });
});

describe("the tab's title", () => {
  it("names the place shown, and the run it is in when that alone says little", () => {
    expect(titleOf([["Runs", "/runs"]])).toBe("Runs · Rollout");
    expect(titleOf([["Runs", "/runs"], ["Run e2e-stacked", "/run/x"]])).toBe("Run e2e-stacked · Rollout");
    expect(titleOf([["Runs", "/runs"], ["Run e2e-stacked", "/run/x"], ["Step 3", ""]])).toBe("Step 3 · Run e2e-stacked · Rollout");
    expect(titleOf([["Checkpoints", "/checkpoints"], ["Checkpoint volr", ""]])).toBe("Checkpoint volr · Rollout");
  });
});

describe("an episode's rollouts", () => {
  const sample = (slot: string, tools: string[], text: string): SampleLine => ({
    kind: "sample", slot, at: 0, seconds: 1, messages: [], tools, reply: { text, reasoning: "", calls: [] },
  });
  const message = (role: string, text: string, results: PlainMessage["results"] = []): PlainMessage => ({ role, text, reasoning: "", calls: [], results });

  it("are every agent's turns, with an acting agent's samples offered no tools folded under its turn before", () => {
    const summary = sample("agent-1", [], "what happened so far");
    const { slots, beside, folded } = samplesOf([
      sample("agent-1", ["mine"], "go"), summary, sample("agent-1", ["mine"], "dig"), sample("policy", [], "candle"),
    ]);
    expect([...slots].map(([slot, turns]) => [slot, turns.map(turn => turn.reply.text)])).toEqual([
      ["agent-1", ["go", "dig"]], ["policy", ["candle"]],
    ]);
    expect(beside.get(slots.get("agent-1")![0])).toEqual([summary]);
    expect(folded).toBe(1);
  });

  it("are in their slots' numbers' order", () => {
    const { slots } = samplesOf(["agent-10", "agent-2", "agent-1"].map(slot => sample(slot, [], "hello")));
    expect([...slots.keys()]).toEqual(["agent-1", "agent-2", "agent-10"]);
  });

  it("say what came back: each call's result, or the words the next turn's context added", () => {
    const first: SampleLine = { ...sample("policy", ["look"], ""), messages: [message("system", "rules"), message("user", "a word?")] };
    const called = { id: "c1", name: "look", arguments: {} }, other = { id: "c2", name: "look", arguments: { far: true } };
    const next: SampleLine = {
      ...first,
      messages: [...first.messages, message("assistant", "candle"), message("user", "close: try again"), message("user", "turn 2")],
    };
    expect(resultOf(first, next)).toEqual({ text: "close: try again\n\nturn 2", error: false });
    // (a context written anew, a summary in place of what came before: the words after its newest reply)
    const rewritten: SampleLine = { ...first, messages: [message("user", "so far: candle"), message("assistant", "candle"), message("user", "close")] };
    expect(resultOf(first, rewritten)).toEqual({ text: "close", error: false });
    const results: SampleLine = { ...first, messages: [...first.messages, message("tool", "", [{ id: "c1", text: "a tree", error: false }, { id: "c2", text: "a hill", error: true }])] };
    expect([resultOf(first, results, called), resultOf(first, results, other)]).toEqual([{ id: "c1", text: "a tree", error: false }, { id: "c2", text: "a hill", error: true }]);
    expect(resultOf(first, undefined)).toBeNull();
  });

  it("show what an agent saw last: what came back since its newest reply, tool results too", () => {
    const messages = [message("system", "rules"), message("user", "start"), message("assistant", "look"), message("tool", "", [{ id: "c1", text: "a tree", error: false }])];
    expect(seenOf(messages)).toEqual([messages[3]]);
    expect(seenOf(messages.slice(0, 2))).toEqual([messages[1]]);
  });

  it("draw a map only under the line a map opens with", () => {
    expect(mapRows("3 1 4 1 5 9 2 6")).toEqual([false]);
    expect(mapRows("Map of what you have seen within 6 blocks…\ny=64 (your feet):\n-3 # # . . @ . ?\nOn the map: d diamond_ore.\n-3 # # . . @ . ?"))
      .toEqual([false, false, true, false, false]);
  });
});

describe("what is drawn", () => {
  it("scales rewards below zero from their lowest", () => {
    const { points, zero } = sparkPoints([-2, 1, -0.5], 100, 40);
    expect(points.every(([, y]) => y >= 0 && y <= 40)).toBe(true);
    expect(points[0][1]).toBeGreaterThan(zero);  // (below 0: under the line of 0)
    const ys = scale([-3, 2], { count: 2 });
    expect([ys.low <= -3, ys.high >= 2]).toEqual([true, true]);
  });

  it("colors each agent by its whole name", () => {
    expect(slotHue("red1")).not.toBe(slotHue("blue1"));
    const hues = ["agent-1", "agent-2", "agent-3", "agent-4"].map(slotHue);
    for (const [place, each] of hues.entries()) for (const other of hues.slice(place + 1)) expect(Math.min(Math.abs(each - other), 360 - Math.abs(each - other))).toBeGreaterThan(20);
    expect(slotHue("agent-2")).toBe(slotHue("agent-2"));
  });

  it("says nothing of solving where no episode said whether it solved its task", () => {
    expect([reported([null, null]), reported([null, false]), reported([])]).toEqual([false, true, false]);
    expect(episodeClass({ run_id: "r", outcome: "completed", solved: null })).toBe("played");
    expect(episodeClass({ run_id: "r", outcome: "completed", solved: false })).toBe("unsolved");
  });
});

describe("a page read again", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("[]", { headers: { "Content-Type": "application/json" } })));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("draws again only what changed: another run's tile stays the element it was", async () => {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system([run("alpha", 3), run("beta", 2)]));
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Runs /></MemoryRouter>
      </QueryClientProvider>,
    );
    const alpha = screen.getByText("alpha").closest("a")!, beta = screen.getByText("beta").closest("a")!;
    expect(beta.textContent).toContain("of 2 decided");
    const alphaText = alpha.innerHTML;
    // The monitor answers again: beta has a group more; alpha is as it was (a new object, equal in every part).
    act(() => { client.setQueryData(topics.system().key, system([run("alpha", 3), run("beta", 3)])); });
    await waitFor(() => expect(screen.getByText("beta").closest("a")!.textContent).toContain("of 3 decided"));
    expect(screen.getByText("alpha").closest("a")).toBe(alpha);
    expect(alpha.innerHTML).toBe(alphaText);
  });

  it("shows a run whose task never says whether it solved by its rewards", () => {
    const client = newQueryClient(), quiet = run("quiet", 2);
    client.setQueryData(topics.system().key, system([{ ...quiet, done: quiet.done.map(line => ({ ...line, rewards: [-1, 0.5], solved: [null, null] })) }]));
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Runs /></MemoryRouter>
      </QueryClientProvider>,
    );
    const tile = screen.getByText("quiet").closest("a")!;
    expect(tile.textContent).toContain("mean reward-0.25");
    expect(tile.textContent).not.toContain("solved");
  });

  it("leaves the evals' runs to the evals page", () => {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system([run("alpha", 3), { ...run("words on kpqx", 2), kind: "eval" }]));
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Runs /></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(screen.getByText("alpha")).toBeTruthy();
    expect(screen.queryByText("words on kpqx")).toBeNull();
  });
});

const noOffers: Offers = {
  cluster: "here", kinds: ["train", "eval"], environments: [{ environment: "games:words", published: false }], trainers: [], inference: [],
  pairs: [], sandboxes: {}, presets: [], capacity: null,
};

describe("a run's controls", () => {
  const fetched = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(
    async () => new Response("{}", { headers: { "Content-Type": "application/json" } }),
  );
  beforeEach(() => { vi.stubGlobal("fetch", fetched); });
  afterEach(() => {
    cleanup();
    fetched.mockClear();
    vi.unstubAllGlobals();
  });

  const launch = (state: Launch["state"], resumes: string | null): Launch => ({
    id: "launch_1", asked: { kind: "train", name: "alpha", settings: { environment: "c:c" }, resumes }, at: 1, state, run: resumes,
    job: "run-1", backend: "ray", detail: null, updated: 1,
  });

  function shown(given: Run, launches: Launch[] = []): string[] {
    const client = newQueryClient();
    client.setQueryData(topics.launches().key, { launches, submits: true });
    render(<QueryClientProvider client={client}><RunControls run={given} /></QueryClientProvider>);
    const said = screen.queryAllByRole("button").map(button => button.textContent ?? "");
    cleanup();
    return said;
  }

  it("are shown by where the run is: pause while it goes, resume once paused or ended, stop while its launch goes", () => {
    const alpha = run("alpha", 1);
    expect(shown({ ...alpha, state: "running" })).toEqual(["pause"]);
    expect(shown({ ...alpha, state: "running", pause: true })).toEqual(["resume"]);
    expect(running({ ...alpha, state: "running", pause: true }, "here")).toBe("pausing");
    expect(shown({ ...alpha, state: "paused", pause: true })).toEqual(["resume"]);
    expect(shown({ ...alpha, state: "stopped" })).toEqual(["resume"]);
    expect(shown({ ...alpha, state: "stopped" }, [launch("running", "alpha")])).toEqual(["stop"]);  // (being resumed)
    expect(shown({ ...alpha, state: "stopped" }, [launch("stopping", "alpha")])).toEqual([]);
    expect(shown({ ...alpha, state: "finished" })).toEqual([]);
    expect(shown({ ...alpha, state: "stopped", kind: "eval", by: "train" })).toEqual([]);  // (its run plays it)
  });

  it("ask the monitor to pause the run", async () => {
    const client = newQueryClient();
    client.setQueryData(topics.launches().key, { launches: [], submits: true });
    client.setQueryData(topics.offers().key, noOffers);
    render(<QueryClientProvider client={client}><RunControls run={{ ...run("a/b", 1), state: "idle" }} /></QueryClientProvider>);
    act(() => { screen.getByRole("button", { name: "pause" }).click(); });
    await waitFor(() => expect(fetched).toHaveBeenCalledWith("api/runs/a%2Fb/pause", expect.objectContaining({ method: "POST" })));
  });

  it("show a paused run's tile as paused", () => {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system([{ ...run("alpha", 2), state: "paused", pause: true }]));
    render(<QueryClientProvider client={client}><MemoryRouter><Runs /></MemoryRouter></QueryClientProvider>);
    const tile = screen.getByText("alpha").closest("a")!;
    expect(tile.className).toContain("violet");
    expect(tile.textContent).toContain("paused");
  });
});

describe("a suite", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows every subject start by start, and compares two", () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } })));
    const client = newQueryClient();
    const checkpoints = [checkpoint("kpqxlmnoprstuvwx", "kpqx", "run_1", 3)];
    client.setQueryData(topics.system().key, { ...system([run("run_1", 1)]), checkpoints });
    client.setQueryData(topics.launches().key, { launches: [], submits: true });
    client.setQueryData(topics.offers().key, noOffers);
    const only = version(1, [["games:words", [["say-yes", 1], ["say-no", 1]]]], "words-v1");
    const evals: Evals = {
      suites: [{
        suite: "words-v1", version: only.id, number: 1, environments: ["games:words"], made: 1, starts: only.starts, versions: [only],
        subjects: [
          { subject: "eval_a", kind: "checkpoint", checkpoint: "kpqxlmnoprstuvwx", model: "tiny", version: only.id, episodes: 1, played: 2, solved: 2,
            reward: 1, results: { "1": [{ solved: true, reward: 1 }], "2": [{ solved: true, reward: 1 }] } },
          { subject: "eval_b", kind: "model", model: "org/tiny", version: only.id, episodes: 1, played: 2, solved: 1, reward: 0.5,
            results: { "1": [{ solved: true, reward: 1 }], "2": [{ solved: false, reward: 0 }] } },
        ],
      }],
      evals: [],
    };
    client.setQueryData(topics.evals().key, evals);
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Suite name="words-v1" /></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("words-v1Edit");
    expect(screen.getAllByText("say-yes · 1").length).toBe(1);  // (in the matrix; the starts compared differ only at say-no)
    expect(screen.getAllByText("say-no · 1").length).toBe(2);
    expect(screen.getByText(/At the 2 starts both played/).textContent).toMatch(/solved more at 1,\s*less at 0, and as much at 1/);
    expect(screen.getByText("+100%")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run this suite" })).toBeTruthy();
  });
});

describe("a new run's settings", () => {
  it("are read as numbers, true or false, or JSON where they look like them, and as text otherwise", async () => {
    const { typed } = await import("./pages/NewRun");
    expect(typed("3e-5")).toBe(3e-5);
    expect(typed("384")).toBe(384);
    expect(typed("true")).toBe(true);
    expect(typed("[1, 2]")).toEqual([1, 2]);
    expect(typed("rollout_lora:LoraTrainer")).toBe("rollout_lora:LoraTrainer");
  });
});

describe("a checkpoint's line", () => {
  it("is its first parents back to the one trained from the base model, oldest first", () => {
    const first = checkpoint("kpqxlmnoprstuvwx", "kpqx", "run_1", 1);
    const merged = { ...checkpoint("lmnopqrstuvwxyzk", "lmno", null, null, [first.id]), kind: "full" };
    const stacked = { ...checkpoint("mnopqrstuvwxyzkl", "mnop", "run_2", 1, [merged.id, "zzzzzzzzzzzzzzzz"]), base: merged.id };
    const byId = new Map([first, merged, stacked].map(each => [each.id, each]));
    expect(lineOf(stacked, id => (id ? byId.get(id) : undefined)).map(each => each.short)).toEqual(["kpqx", "lmno", "mnop"]);
    expect(lineOf(first, () => undefined)).toEqual([first]);
  });

  const point = (depth: number, extra: Partial<PathPoint>): PathPoint => ({
    id: `c${depth}`, short: `c${depth}`, depth, model: null, run: "run_1", name: "lora run", step: depth, kind: "lora", bookmarks: [], scores: {}, ...extra,
  });
  const score = (solved: number | null, reward: number) => ({ solved, reward, played: 4, evals: ["e"] });

  it("draws each suite's score by depth from the base model, marking where it enters a run or changes its weights", () => {
    const path: Path = {
      checkpoint: "c4",
      suites: [{ suite: "words", environments: ["games:words"] }, { suite: "maths", environments: ["games:maths"] }],
      points: [
        point(0, { id: null, short: "base", model: "tiny", run: null, name: null, step: null, kind: "model", scores: { words: score(0.25, 0.3), maths: score(null, -1) } }),
        point(1, { scores: { words: score(0.5, 0.5) } }),
        point(2, { id: "c2", run: null, name: null, kind: "full", bookmarks: ["merged"], scores: { maths: score(null, 2) } }),
        point(3, { run: "run_2", name: "stacked", scores: { words: score(0.75, 0.8) } }),
        point(4, { run: "run_2", name: "stacked" }),
      ],
    };
    const chart = pathChart(path, place => `color-${place}`);
    expect(chart.series.map(each => [each.suite, each.measure, each.color, each.points])).toEqual([
      ["words", "solved", "color-0", [[0, 0.25], [1, 0.5], [3, 0.75]]],
      ["maths", "reward", "color-1", [[0, -1], [2, 2]]],
    ]);
    expect(chart.solved).toBe(false);
    expect([...chart.labels.entries()]).toEqual([[0, "base"], [1, "c1"], [2, "merged"], [3, "c3"], [4, "c4"]]);
    expect(chart.marks).toEqual([{ x: 1, label: "lora run" }, { x: 2, label: "outside a run · full" }, { x: 3, label: "stacked · LoRA" }]);
    expect(pathChart({ ...path, suites: [{ suite: "words", environments: [] }] }).solved).toBe(true);
    expect(pathChart({ checkpoint: "c", points: [], suites: [] }).series).toEqual([]);
  });

  it("draws a suite of several environments as a line for each environment", () => {
    const both = (words: number | null, guessing: number) => ({
      ...score(null, 0), entries: { "games:words": { solved: words, reward: 1, played: 2 }, "games:guessing": { solved: null, reward: guessing, played: 3 } },
    });
    const path: Path = {
      checkpoint: "c2",
      suites: [{ suite: "mixed@1", label: "mixed", environments: ["games:words", "games:guessing"] }],
      points: [point(0, { id: null, short: "base", kind: "model", scores: { "mixed@1": both(0.5, 0.25) } }), point(1, {}), point(2, { scores: { "mixed@1": both(1, 0.75) } })],
    };
    expect(pathChart(path, place => `color-${place}`).series.map(each => [each.name, each.measure, each.points])).toEqual([
      ["mixed · words", "solved", [[0, 0.5], [2, 1]]],
      ["mixed · guessing", "reward", [[0, 0.25], [2, 0.75]]],
    ]);
  });
});

describe("a suite's columns", () => {
  it("stand by run, the base model first, then by depth", () => {
    const at = (id: string, short: string, run: string | null, depth: number, made: number): Checkpoint => ({ ...checkpoint(id, short, run, depth), depth, made });
    const checkpoints = [at("kkkkkkkkkkkkkkkk", "kkkk", "late", 1, 5), at("llllllllllllllll", "llll", "early", 2, 2),
      at("mmmmmmmmmmmmmmmm", "mmmm", "early", 1, 1), at("nnnnnnnnnnnnnnnn", "nnnn", null, 3, 3)];
    const known = knownOf(checkpoints, { early: "early run" });
    const subject = (name: string, checkpoint: string | null): Subject => ({
      subject: name, kind: checkpoint ? "checkpoint" : "model", checkpoint: checkpoint ?? undefined, model: "tiny", played: 1, solved: 1, results: {},
    });
    const subjects = [subject("e1", "kkkkkkkkkkkkkkkk"), subject("e2", "llllllllllllllll"), subject("e3", null), subject("e4", "nnnnnnnnnnnnnnnn"),
      subject("e5", "mmmmmmmmmmmmmmmm")];
    expect(columnsOf(subjects, known).map(column => [column.label, column.subjects.map(each => each.subject)])).toEqual([
      ["base model", ["e3"]], ["early run", ["e5", "e2"]], ["late", ["e1"]], ["outside a run", ["e4"]],
    ]);
  });
});

describe("a run's settings, as typed", () => {
  it("are whole numbers of 1 at least where they must be, a suite or none, else as typed", () => {
    expect(settingOf("evals.every", "2")).toEqual({ value: 2 });
    expect(settingOf("evals.every", "0")).toEqual({ error: "a whole number, 1 at least" });
    expect(settingOf("groups_per_step", "1.5")).toEqual({ error: "a whole number, 1 at least" });
    expect(settingOf("groups_per_step", "")).toEqual({ error: "a whole number, 1 at least" });
    expect(settingOf("evals.suite", "  ")).toEqual({ value: null });
    expect(settingOf("evals.episodes", "")).toEqual({ value: null });  // (the suite's own)
    expect(settingOf("evals.episodes", "0")).toEqual({ error: "a whole number, 1 at least" });
    expect(settingOf("trainer.learning_rate", "3e-5")).toEqual({ value: 3e-5 });
    expect(settingOf("trainer.truncate", "")).toEqual({ value: null });
  });

  it("ask only for what differs from what the run uses, and say why a field cannot be", () => {
    const current = { groups_per_step: 4, "evals.suite": null, "evals.every": 1, "trainer.learning_rate": 5e-5 };
    expect(wantedOf({ groups_per_step: "4", "evals.suite": "words", "evals.every": "x", "trainer.learning_rate": "0.00003" }, current)).toEqual({
      settings: { "evals.suite": "words", "trainer.learning_rate": 3e-5 }, errors: { "evals.every": "a whole number, 1 at least" },
    });
    expect(wantedOf({ "evals.suite": "" }, current)).toEqual({ settings: {}, errors: {} });
  });

  it("make a new run's evals: a suite, or none said so, and nothing until one is chosen", () => {
    expect(evalsSettings("", "2", "3")).toEqual({ settings: {}, errors: { "evals.suite": "a suite, or none" } });
    expect(evalsSettings(NO_EVALS, "2", "3")).toEqual({ settings: { "evals.suite": null }, errors: {} });
    expect(evalsSettings("words", "2", "3")).toEqual({ settings: { "evals.suite": "words", "evals.every": 2, "evals.episodes": 3 }, errors: {} });
    expect(evalsSettings("words", "1", "")).toEqual({ settings: { "evals.suite": "words", "evals.every": 1, "evals.episodes": null }, errors: {} });
    expect(evalsSettings("words", "0", "3").errors).toEqual({ "evals.every": "a whole number, 1 at least" });
  });
});


const entry = (environment: string, offset: number, starts: number, extra: Partial<SuiteEntry> = {}): SuiteEntry => ({
  environment, environment_version: "1", chosen: "rows and seeds", eval_data: null, rows: null, seeds: null, held_out: false, episodes: 1,
  thinking_tokens: null, answer_tokens: null, offset, starts, ...extra,
});

/** A version of entries, each its environment and its starts (a row and a seed each). */
const version = (number: number, entries: [string, [string, number][], Partial<SuiteEntry>?][], name = "words"): SuiteVersion => {
  const offsets = entries.map((_, place) => entries.slice(0, place).reduce((sum, [, starts]) => sum + starts.length, 0));
  return {
    id: `${name}@${number}`, number, environments: entries.map(([environment]) => environment), made: number, held_out: false,
    entries: entries.map(([environment, starts, extra], place) => entry(environment, offsets[place], starts.length, extra)),
    starts: entries.flatMap(([environment, starts], place) => starts.map(([task, seed], at) => ({
      start: String(offsets[place] + at + 1), environment, task, seed, title: task, identity: `${task}|${seed}`,
    }))),
  };
};

const versioned = (): EvalSuite => {
  const first = version(1, [["games:words", [["say-yes", 1], ["say-no", 1]]]]);
  const second = version(2, [["games:words", [["say-no", 1], ["say-maybe", 1]], { episodes: 2 }]]);
  const subject = (name: string, played: string, results: Record<string, boolean[]>): EvalSuite["subjects"][number] => ({
    subject: name, kind: "model", model: `org/${name}`, version: played, starts: 2, episodes: 1, played: Object.values(results).flat().length,
    solved: Object.values(results).flat().filter(Boolean).length,
    results: Object.fromEntries(Object.entries(results).map(([start, solved]) => [start, solved.map(each => ({ solved: each, reward: each ? 1 : 0 }))])),
  });
  return {
    suite: "words", version: "words@2", number: 2, environments: ["games:words"], made: 2, starts: second.starts, versions: [first, second],
    subjects: [subject("old", "words@1", { "1": [true], "2": [false] }), subject("new", "words@2", { "1": [true], "2": [true] }),
      subject("newer", "words@2", { "1": [false], "2": [true] })],
  };
};

describe("a suite's versions", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("stand subjects by the version each played, newest first, and every start of the versions shown once", () => {
    const suite = versioned();
    expect(versionGroups(suite, suite.subjects, ALL).map(group => [versionTag(group.version.id), group.subjects.map(each => each.subject)])).toEqual([
      ["v2", ["new", "newer"]], ["v1", ["old"]],
    ]);
    expect(versionGroups(suite, suite.subjects, "words@1").map(group => group.subjects.map(each => each.subject))).toEqual([["old"]]);
    expect(versionGroups({ ...suite, subjects: [] }, [], "words@2").map(group => group.version.id)).toEqual(["words@2"]);  // (picked, though none played it)
    const rows = gridRows(blocksOf(suite.versions!.map(each => ({ version: each, subjects: [] }))));
    expect(rows.map(row => [row.key, row.at])).toEqual([
      ["games:words|say-no|1", { "words@2": "1", "words@1": "2" }], ["games:words|say-maybe|1", { "words@2": "2" }],
      ["games:words|say-yes|1", { "words@1": "1" }],
    ]);
    expect(versionTag("words@12")).toBe("v12");
  });

  it("show where versions change in the grid, and compare subjects of one version only", () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } })));
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system([]));
    client.setQueryData(topics.launches().key, { launches: [], submits: true });
    client.setQueryData(topics.offers().key, noOffers);
    const evals: Evals = { suites: [versioned()], evals: [] };
    client.setQueryData(topics.evals().key, evals);
    const { container } = render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Suite name="words" /></MemoryRouter>
      </QueryClientProvider>,
    );
    const heads = [...container.querySelectorAll("table.evals tr.versions th")].map(each => each.textContent);
    expect(heads).toEqual(["start", "v2 · newest", "v1"]);
    expect(container.querySelectorAll("table.evals th.subject.version-start").length).toBe(2);  // (where each version begins)
    expect(container.querySelectorAll("table.evals tbody tr").length).toBe(4);  // (the totals, and three starts)
    expect(container.querySelectorAll("table.evals td.absent").length).toBe(3);  // (a start a version does not have)
    expect(screen.getByText("Compared").closest("section")?.textContent).toContain("v2");
    expect(screen.getByText(/At the 2 starts both played/)).toBeTruthy();  // (new and newer: both played v2)
    expect(screen.getByRole("option", { name: /v1 · 2 starts · 1 played/ })).toBeTruthy();
  });

  it("are made and edited from forms read as the monitor takes them", () => {
    expect(wholes("1, 2 3,5-7")).toEqual([1, 2, 3, 5, 6, 7]);
    expect(wholes("one")).toBeNull();
    expect(wholes("4-2")).toBeNull();
    const fields = { ...fieldsOf(undefined, undefined, DRAWN, "games:words"), rows: ["say-yes"], seeds: "1-3", episodes: "2", thinking: "64" };
    expect(entryBody(fields)).toEqual({
      body: { environment: "games:words", chosen: DRAWN, rows: ["say-yes"], seeds: [1, 2, 3], episodes: 2, thinking_tokens: 64, answer_tokens: null }, errors: {},
    });
    expect(entryBody({ ...fields, rows: [], seeds: "" }).errors).toEqual({ seeds: "whole numbers, as 1, 2, 3 or 1-5" });
    expect(entryBody({ ...fields, rows: [] }).body.rows).toBeNull();  // (every row)
    expect(entryBody({ ...fields, episodes: "0", answer: "x" }).errors).toEqual({ episodes: "a whole number, 1 at least", answer: "a whole number, 1 at least, or empty" });
    expect(entryBody({ ...fields, chosen: GIVEN, starts: "say-yes 1\nsay-no 4\n" }).body.starts).toEqual([{ task: "say-yes", seed: 1 }, { task: "say-no", seed: 4 }]);
    expect(entryBody({ ...fields, chosen: GIVEN, starts: "say-yes" }).errors).toEqual({ starts: "a row and a seed on each line" });
    expect(entryBody({ ...fields, chosen: "eval data", evalData: "" }).errors).toEqual({ evalData: "which eval data" });
    expect(entryBody({ ...fields, environment: " " }).errors).toEqual({ environment: "which environment" });
    const newest = versioned().versions![1];
    const edited = fieldsOf(newest.entries[0], newest);
    expect([edited.environment, edited.chosen, edited.episodes, edited.starts]).toEqual(["games:words", SAME, "2", "say-no 1\nsay-maybe 1"]);
    expect(suiteBody([{ ...edited, episodes: "3" }], 2)).toEqual({
      body: { entries: [{ environment: "games:words", chosen: SAME, episodes: 3, thinking_tokens: null, answer_tokens: null }], base: 2 }, errors: [{}], suite: null,
    });
  });

  it("are made of entries, each an environment once", () => {
    const mixed = version(1, [["games:words", [["say-yes", 1]]], ["games:guessing", [["guess-apple", 7], ["guess-river", 7]], { thinking_tokens: 10 }]], "mixed");
    const [words, guesses] = mixed.entries.map(each => fieldsOf(each, mixed));
    expect([guesses.environment, guesses.starts, guesses.thinking]).toEqual(["games:guessing", "guess-apple 7\nguess-river 7", "10"]);
    expect(mixed.entries.map(limitsText)).toEqual(["channel's own", "thinking 10 · answer channel's"]);  // (unset: the channel's)
    const made = suiteBody([words, { ...guesses, chosen: DRAWN, seeds: "x" }]);
    expect(made.errors).toEqual([{}, { seeds: "whole numbers, as 1, 2, 3 or 1-5" }]);
    expect((made.body.entries as { environment: string }[]).map(each => each.environment)).toEqual(["games:words", "games:guessing"]);
    expect(suiteBody([words, words]).suite).toBe("games:words is in two entries");
    expect(suiteBody([]).suite).toBe("an environment at least");
  });
});

describe("a suite of several environments", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  const mixed = (): EvalSuite => {
    const only = version(1, [["games:words", [["say-yes", 1]]]], "mixed");
    const both = version(2, [["games:words", [["say-yes", 1]]], ["games:guessing", [["guess-apple", 7], ["guess-river", 7]]]], "mixed");
    const subject = (name: string, played: string, results: Record<string, boolean[]>, entries?: Subject["entries"]): Subject => ({
      subject: name, kind: "model", model: `org/${name}`, version: played, episodes: 1, played: Object.values(results).flat().length,
      solved: Object.values(results).flat().filter(Boolean).length, entries,
      results: Object.fromEntries(Object.entries(results).map(([start, solved]) => [start, solved.map(each => ({ solved: each, reward: each ? 1 : 0 }))])),
    });
    return {
      suite: "mixed", version: "mixed@2", number: 2, environments: ["games:words", "games:guessing"], made: 2, starts: both.starts,
      versions: [only, both],
      subjects: [
        subject("a", "mixed@2", { "1": [true], "2": [false], "3": [true] },
          [{ environment: "games:words", played: 1, solved: 1, reward: 1 }, { environment: "games:guessing", played: 2, solved: 1, reward: 0.5 }]),
        subject("b", "mixed@2", { "1": [false], "2": [false], "3": [false] },
          [{ environment: "games:words", played: 1, solved: 0, reward: 0 }, { environment: "games:guessing", played: 2, solved: 0, reward: 0 }]),
        subject("c", "mixed@1", { "1": [true] }, [{ environment: "games:words", played: 1, solved: 1, reward: 1 }]),
      ],
    };
  };

  it("stands a column group for each environment of a version, each with its subjects' totals there", () => {
    const suite = mixed();
    const blocks = blocksOf(versionGroups(suite, suite.subjects, ALL));
    expect(blocks.map(block => [block.key, block.subjects.map(each => each.subject)])).toEqual([
      ["mixed@2#games:words", ["a", "b"]], ["mixed@2#games:guessing", ["a", "b"]], ["mixed@1", ["c"]],
    ]);
    expect(gridRows(blocks).map(row => [row.key, row.at])).toEqual([
      ["games:words|say-yes|1", { "mixed@2#games:words": "1", "mixed@1": "1" }],
      ["games:guessing|guess-apple|7", { "mixed@2#games:guessing": "2" }],
      ["games:guessing|guess-river|7", { "mixed@2#games:guessing": "3" }],
    ]);
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } })));
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system([]));
    client.setQueryData(topics.launches().key, { launches: [], submits: true });
    client.setQueryData(topics.offers().key, noOffers);
    client.setQueryData(topics.evals().key, { suites: [suite], evals: [] } satisfies Evals);
    const { container } = render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Suite name="mixed" /></MemoryRouter>
      </QueryClientProvider>,
    );
    const texts = (selector: string) => [...container.querySelectorAll(selector)].map(each => each.textContent);
    expect(texts("table.evals tr.versions th")).toEqual(["start", "v2 · newest", "v1"]);
    expect(texts("table.evals tr.entries th")).toEqual(["words", "guessing", ""]);
    expect(container.querySelectorAll("table.evals th.subject.entry-start").length).toBe(1);  // (where the second environment begins)
    expect(texts("table.evals tr.total td b")).toEqual(["1/1", "0/1", "1/2", "0/2", "1/1"]);  // (each subject at each environment)
    expect(container.querySelectorAll("table.evals td.absent").length).toBe(2 * 2 + 2 * 1 + 2);  // (a start an environment does not have)
    expect(texts("section.card h2")).toContain("Environments");
    expect(screen.queryByText("Best")).toBeNull();  // (a best across environments would weigh one's rewards against another's)
  });

  it("names environments in a word, and lists the offered first, telling apart two of a name", () => {
    expect(readable("tests.rollout_train.rollouts.games:words")).toBe("words");
    expect(readable("minecraft_team.environment:environment")).toBe("minecraft_team");
    const known = [
      { environment: "b.games:words", name: "words", versions: [], offered: false },
      { environment: "a.games:words", name: "words", versions: ["1"], offered: true },
      { environment: "c:guessing", name: "guessing", versions: [], offered: true },
    ];
    expect(pickable(known, ["d:maths"]).map(each => [each.name, each.offered])).toEqual([
      ["guessing", true], ["words (a.games)", true], ["maths", false], ["words (b.games)", false],
    ]);
  });
});
