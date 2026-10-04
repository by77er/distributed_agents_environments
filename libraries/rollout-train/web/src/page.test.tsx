import { QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Run, System, Checkpoint } from "./api/types";
import { knownOf } from "./lib/model";
import { placeOf } from "./lib/places";
import { Runs } from "./pages/Runs";

const run = (name: string, done: number): Run => ({
  run: name, name, fence: 1, wrote: 10, decided: done, open: [], next: [], channels: [], state: "ended", host: "here",
  address: null, directory: `/runs/${name}`, episodes_at: "here", reached: null, profile: null, from: null, started: 1,
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
    expect(placeOf("/checkpoints/sample")).toEqual({ page: "checkpoints", kind: "checkpoints", sample: true });
    expect(placeOf("/checkpoint/kpqx")).toEqual({ page: "checkpoints", kind: "checkpoint", id: "kpqx" });
    expect(placeOf("/system")).toEqual({ page: "statistics", kind: "statistics", section: "machines" });
    expect(placeOf("/runs/new")).toEqual({ page: "runs", kind: "launch" });
    expect(placeOf("")).toEqual({ page: "runs", kind: "runs" });
  });
});

const checkpoint = (id: string, short: string, run: string | null, step: number | null, parents: string[] = [], bookmarks: string[] = []): Checkpoint => ({
  id, short, depth: parents.length + 1, parents, base: "Qwen/Qwen3.5-9B", run, step, made: 1, metrics: {}, weights: null, state: null,
  released: null, batch: false, bookmarks,
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
