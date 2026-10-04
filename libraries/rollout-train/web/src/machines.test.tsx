import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Host, Machines as MachinesData, Measurement, PoolRole, RunnerRole, System } from "./api/types";
import { leftOf, placesOf, present, rolesOn, sandboxesOf, seriesOf, share, shortName } from "./lib/machines";
import { hostPlace, placeOf } from "./lib/places";
import { Machines } from "./pages/Machines";

const GIB = 2 ** 30;
const measured = (at: number, used: number): Measurement => ({
  at, memory: { available: (16 - used) * GIB, total: 16 * GIB }, accelerators: [{ name: "RTX", used: used * GIB, total: 16 * GIB, busy: used / 16 }],
  disk: { free: 100 * GIB, total: 500 * GIB },
});

const runner = (name: string, host: string, playing: number, places: number, alive = true): RunnerRole => ({
  name, host, alive, at: 90, run: null, places, playing, free: Math.max(places - playing, 0), claims: [], pools: [], channels: [], processes: null,
});

const pool = (name: string, kind: string, host: string, leased: number | null, size: number | null, alive = true): PoolRole => ({
  name, host, alive, at: 90, kind, runner: null, size, leased, free: size == null ? null : size - (leased ?? 0), leases: [],
});

const machines = (): MachinesData => ({
  now: 100,
  hosts: [
    { host: "gpu-1", alive: true, at: 90, machine: measured(90, 4), history: [], roles: [{ kind: "runners", name: "gpu-1/train", alive: true }] },
    { host: "gpu-2", alive: false, at: 1, machine: null, history: [], roles: [{ kind: "runners", name: "gpu-2/old", alive: false }] },
  ],
  runners: [
    { ...runner("gpu-1/train", "gpu-1", 2, 6), claims: [{ run: "train", group: 3, episode: 1, attempt: 1, at: 80, run_id: "r_1" }] },
    runner("gpu-2/old", "gpu-2", 1, 4, false),
  ],
  pools: [
    { ...pool("minecraft@gpu-1/train", "minecraft", "gpu-1", 1, 4), runner: "gpu-1/train",
      leases: [{ key: "train/3/1/1/world", kind: "minecraft", sandbox: "world", run: "train", group: 3, episode: 1, attempt: 1, run_id: "r_1", holds: true, at: 80, seconds: 600, lost: false }] },
  ],
  engines: [],
  launchers: [],
  gateways: [],
});

describe("machines", () => {
  it("say how full something is, from 0 to 1, and nothing where its size is not known", () => {
    expect([share(1, 4), share(5, 4), share(0, 0), share(2, null), share(null, 4)]).toEqual([0.25, 1, null, null, 0]);
  });

  it("count the places and sandboxes of the roles alive only", () => {
    const runners = [runner("a", "h", 2, 6), runner("b", "h", 9, 4), runner("c", "h", 1, 8, false)];
    expect(placesOf(runners)).toEqual({ places: 10, used: 6, free: 4 });
    const pools = [pool("x", "docker", "h", 1, 4), pool("y", "docker", "k", 3, 2), pool("z", "minecraft", "h", 1, null), pool("w", "docker", "h", 4, 4, false)];
    expect(sandboxesOf(pools)).toEqual([{ kind: "docker", pools: 2, size: 6, leased: 3, free: 3 }]);
  });

  it("put each role on its machine, and list the kinds present in the page's order", () => {
    const data = machines();
    const on = rolesOn(data, "gpu-1");
    expect(on.runners.map(each => each.name)).toEqual(["gpu-1/train"]);
    expect(on.pools.map(each => each.name)).toEqual(["minecraft@gpu-1/train"]);
    expect(present(on)).toEqual(["runners", "pools"]);
    expect(present(rolesOn(data, "elsewhere"))).toEqual([]);
  });

  it("call a role by its name on its machine", () => {
    expect(shortName("runners", "gpu-1/train", "gpu-1")).toBe("train");
    expect(shortName("pools", "minecraft@gpu-1/train", "gpu-1")).toBe("minecraft");
    expect(shortName("launchers", "launcher/gpu-1", "gpu-1")).toBe("");
    expect(shortName("engines", "gpu-2", "gpu-2")).toBe("");
    expect(shortName("gateways", "gateway/edge/0.0.0.0:8443", "edge")).toBe("0.0.0.0:8443");
  });

  it("say how long a lease has left before its limit", () => {
    expect(leftOf({ at: 100, seconds: 60 }, 130)).toBe(30);
    expect(leftOf({ at: 100, seconds: 60 }, 200)).toBe(0);
    expect(leftOf({ at: 100, seconds: null }, 130)).toBeNull();
  });

  it("draw a machine's memory, accelerators and disk over its beats", () => {
    const history: Host["history"] = [{ at: 1, machine: measured(1, 4) }, { at: 2, machine: { ...measured(2, 8), disk: null } }];
    const series = seriesOf(history);
    expect(series.memory).toEqual([[1, 4], [2, 8]]);
    expect(series.disk).toEqual([[1, 400]]);
    expect(series.accelerators).toEqual([{ name: "RTX", used: [[1, 4], [2, 8]], busy: [[1, 0.25], [2, 0.5]] }]);
  });

  it("are a page of their own, and each machine a place with its roles", () => {
    expect(placeOf("/machines")).toEqual({ page: "machines", kind: "machines" });
    expect(placeOf(hostPlace("gpu-1", "gpu-1/train"))).toEqual({ page: "machines", kind: "host", host: "gpu-1", role: "gpu-1/train" });
    expect(placeOf("/statistics/machines")).toEqual({ page: "machines", kind: "machines" });
  });
});

describe("the machines page", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } })));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows the roles alive by kind, with how full each is, and the gone ones apart", () => {
    const client = newQueryClient();
    const system = { at: 100, checkpoints: [], names: { runs: { train: "diamonds" }, bookmarks: {} } } as unknown as System;
    client.setQueryData(topics.system().key, system);
    client.setQueryData(topics.machines().key, machines());
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Machines /></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(screen.getByText("Runners")).toBeTruthy();
    expect(screen.getByText("Sandbox pools")).toBeTruthy();
    expect(screen.queryByText("Engine hosts")).toBeNull();
    expect(screen.getByText("2 of 6 playing · 4 free")).toBeTruthy();
    expect(screen.getByText("1 of 4 leased · 3 free")).toBeTruthy();
    expect(screen.getAllByText("diamonds").length).toBe(2);  // (its runner's claim and its pool's lease)
    expect(screen.getByText("Gone")).toBeTruthy();
    expect(screen.getByText("gpu-2/old")).toBeTruthy();
  });
});
