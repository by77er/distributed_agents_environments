import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Launches, Queue, System } from "./api/types";
import { QueueSection } from "./components/queue";
import { amount, barOf, ordinal, queued, shortReason, waitsFor } from "./lib/queue";
import { runPlace } from "./lib/places";
import { Run } from "./pages/Run";

const GIB = 2 ** 30;
const PENDING = "couldn't assign flavors to pod set head: insufficient unused quota for nvidia.com/gpu in flavor rollout, 1 more needed";

/** The queue as Kueue says it: alpha holds the GPU, beta CPUs and memory; gamma waits first for the GPU, delta behind it
 * for the GPU and memory. */
const queue = (): Queue => ({
  source: "kueue", order: "kueue",
  capacity: { cpu: 12, memory: 16 * GIB, gpu: 1 },
  used: { cpu: 8.5, memory: 12 * GIB, gpu: 1 },
  admitted: [
    { run: "run_a", name: "alpha", requests: { cpu: 6, memory: 9 * GIB, gpu: 1 }, since: 100, job: "run-a", workload: "rayjob-a" },
    { run: "run_b", name: "beta", requests: { cpu: 2.5, memory: 3 * GIB }, since: 110, job: "run-b", workload: "rayjob-b" },
  ],
  pending: [
    { run: "run_g", name: "gamma", requests: { cpu: 2, memory: 4 * GIB, gpu: 1 }, since: 130, position: 1, reason: PENDING,
      lacks: { gpu: 1 }, held_by: ["run_a"], workload: "rayjob-g" },
    { run: "run_d", name: "delta", requests: { cpu: 3, memory: 6 * GIB, gpu: 1 }, since: 120, position: 2, reason: PENDING,
      lacks: { gpu: 1, memory: 2 * GIB }, held_by: ["run_a", "run_b"], workload: "rayjob-d" },
  ],
});

const names: Record<string, string> = { run_a: "alpha", run_b: "beta", run_g: "gamma", run_d: "delta" };
const name = (run: string | null) => (run ? names[run] ?? run : "another job");

describe("the queue", () => {
  it("splits each resource's bar into the admitted runs' shares, in proportion, and what no run holds apart", () => {
    const cpu = barOf(queue(), "cpu")!;
    expect(cpu.segments.map(each => [each.run, each.start, each.width])).toEqual([["run_a", 0, 0.5], ["run_b", 0.5, 2.5 / 12]]);
    expect(cpu.other).toBeNull();
    const gpu = barOf(queue(), "gpu")!;
    expect(gpu.segments.map(each => [each.run, each.width])).toEqual([["run_a", 1]]);  // (beta holds none)
    const more = { ...queue(), used: { cpu: 10, memory: 12 * GIB, gpu: 1 } };
    expect(barOf(more, "cpu")!.other).toEqual({ start: 8.5 / 12, width: 1.5 / 12 });
    expect(barOf({ ...queue(), capacity: {} }, "cpu")).toBeNull();  // (no capacity known: no bar)
  });

  it("says why a run waits: what it lacks and who holds it, more than the queue holds, else Kueue's reason shortened", () => {
    const [gamma, delta] = queue().pending;
    expect(waitsFor(gamma, queue(), name)).toBe("waits for 1 GPU: in use by run alpha");
    expect(waitsFor(delta, queue(), name)).toBe("waits for 1 GPU and 2.0 GiB: in use by runs alpha and beta");
    const big = { ...gamma, requests: { gpu: 2, cpu: 1 }, lacks: { gpu: 2 } };
    expect(waitsFor(big, queue(), name)).toBe("asks for 2 GPUs; the queue holds 1 GPU");
    const behind = { ...gamma, lacks: {}, held_by: [] };
    expect(waitsFor(behind, queue(), name)).toBe("insufficient unused quota for GPU, 1 more needed");
    expect(waitsFor({ ...behind, reason: null }, queue(), name)).toBe("waits for admission");
    const ray: Queue = { ...queue(), source: "ray", capacity: {} };
    expect(waitsFor({ ...behind, reason: "waits for run/x/engine/policy/0 (1 GPU: pending creation)" }, ray, name)).toBe("no GPU free (1 wanted)");
    expect(shortReason(PENDING)).toBe("insufficient unused quota for GPU, 1 more needed");
    expect(queued(gamma, queue(), name)).toBe("1st in the queue · waits for 1 GPU: in use by run alpha");
    expect([1, 2, 3, 4, 11, 12, 13, 21, 22, 103].map(ordinal)).toEqual(["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "103rd"]);
    expect([amount("cpu", 2.5), amount("cpu", 1), amount("memory", 6 * GIB), amount("gpu", 1)]).toEqual(["2.5 CPUs", "1 CPU", "6.0 GiB", "1 GPU"]);
  });
});

describe("the queue's section", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } })));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  const system = {
    at: 200, host: "here", checkpoints: [], names: { runs: names, bookmarks: {} },
    runs: ["run_a", "run_b", "run_d", "run_g"].map(run => ({ run, name: names[run], state: "running" })),
  } as unknown as System;

  function shown(given: Queue): HTMLElement {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system);
    client.setQueryData(topics.queue().key, given);
    const { container } = render(
      <QueryClientProvider client={client}><MemoryRouter><QueueSection /></MemoryRouter></QueryClientProvider>,
    );
    return container;
  }

  it("draws a bar for each resource, each admitted run's segment sized by its share and linked to it", () => {
    const container = shown(queue());
    const bars = container.querySelectorAll(".queue-bar");
    expect([...bars].map(bar => bar.querySelector(".queue-bar-head")!.textContent)).toEqual(["CPU8.5 of 12 CPUs", "Memory12.0 GiB of 16.0 GiB", "GPU1 of 1 GPU"]);
    const cpu = [...bars[0].querySelectorAll<HTMLAnchorElement>("a.queue-segment")];
    expect(cpu.map(each => [each.getAttribute("href"), each.style.left, each.style.width])).toEqual([
      [runPlace("run_a"), "0%", "50%"], [runPlace("run_b"), "50%", "20.833%"],
    ]);
    expect(cpu.map(each => each.title)).toEqual(["alpha · 6 CPUs", "beta · 2.5 CPUs"]);
    expect(cpu[0].style.background).toBe("var(--series-1)");  // (its run's color)
    expect(bars[2].querySelectorAll("a.queue-segment").length).toBe(1);
  });

  it("lists the runs that wait in the queue's order, with what each asks for, how long it waited and why in plain words", () => {
    const given = queue();
    given.pending[1].since = Date.now() / 1000 - 7200;
    shown(given);
    const rows = within(screen.getByRole("table", { name: "waiting" })).getAllByRole("row").slice(1);
    expect(rows.map(row => within(row).getAllByRole("cell")[0].textContent)).toEqual(["1", "2"]);
    expect(rows.map(row => within(row).getByRole("link").textContent)).toEqual(["gamma", "delta"]);
    expect(within(rows[0]).getByRole("link").getAttribute("href")).toBe(runPlace("run_g"));
    expect(within(rows[1]).getAllByRole("cell").slice(2, 5).map(cell => cell.textContent)).toEqual(["3", "6.0 GiB", "1"]);
    expect(within(rows[0]).getByText("waits for 1 GPU: in use by run alpha")).toBeTruthy();
    expect(within(rows[1]).getByText("waits for 1 GPU and 2.0 GiB: in use by runs alpha and beta")).toBeTruthy();
    expect(within(rows[1]).getAllByRole("cell")[5].textContent).toBe("2.0 h");
  });

  it("says when no run holds anything and nothing waits, and nothing at all without a source", () => {
    shown({ ...queue(), used: { cpu: 0, memory: 0, gpu: 0 }, admitted: [], pending: [] });
    expect(screen.getByText("No runs hold resources.")).toBeTruthy();
    expect(screen.getByText("Nothing waits.")).toBeTruthy();
    expect(document.querySelectorAll("a.queue-segment").length).toBe(0);
    cleanup();
    const none = shown({ source: null, capacity: {}, used: {}, admitted: [], pending: [] });
    expect(none.textContent).toBe("");
  });

  it("lists what each run holds where the capacity is not known", () => {
    shown({ ...queue(), source: "ray", capacity: {}, pending: [] });
    expect(document.querySelectorAll(".queue-bar").length).toBe(0);
    expect(screen.getByText("1 GPU · 6 CPUs · 9.0 GiB")).toBeTruthy();
  });
});

describe("a run that waits in the queue", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } })));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows its place in the queue and why in its state line, before it has started", () => {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, { at: 200, host: "here", runs: [], checkpoints: [], names: { runs: names, bookmarks: {} } } as unknown as System);
    client.setQueryData(topics.queue().key, queue());
    const launches: Launches = { submits: true, launches: [{
      id: "launch_d", asked: { kind: "train", name: "delta", settings: {}, preset: null, resumes: null }, at: 120, state: "submitted",
      run: "run_d", job: "run-d", backend: "kubernetes", detail: `waits for admission by Kueue (queue runs): ${PENDING}`, updated: 150,
    }] } as unknown as Launches;
    client.setQueryData(topics.launches().key, launches);
    render(<QueryClientProvider client={client}><MemoryRouter><Run name="run_d" /></MemoryRouter></QueryClientProvider>);
    expect(screen.getByText("Run delta")).toBeTruthy();
    expect(screen.getByText("waiting · 2nd in the queue · waits for 1 GPU and 2.0 GiB: in use by runs alpha and beta")).toBeTruthy();
  });
});
