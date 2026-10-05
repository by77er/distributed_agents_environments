import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { DoneLine, Run, System } from "./api/types";
import { learnableText, nothingText, towardStep } from "./lib/model";
import { Runs } from "./pages/Runs";

const SAME = "every advantage is zero";

const line = (group: number, segments: number): DoneLine => ({
  group, time: 10 + group, task: "door-2", title: "door", rollout_seconds: 5, rewards: segments ? [0.375, 0.25] : [0, 0],
  solved: [false, false], durations: [40, 40], failed: 0, failures: [], segments_recorded: 80, segments,
  skipped: segments ? null : SAME, unlocked: 3, adapter: null, step: null, step_state: null, depth: null, update: null,
  segments_trained: 0, error: null, seconds: 5,
});

/** Sixteen groups done and none covered by a step: group 14 had something to train on, the others nothing; group 17
 * still plays. */
const stuck = (): Run => ({
  run: "stuck", name: "stuck", fence: 1, wrote: 10, decided: 17, channels: [], state: "running", host: "here",
  address: null, directory: "/runs/stuck", episodes_at: "here", reached: null, from: null, started: 1, starts: 1,
  written: 10, steps: [], groups_per_step: 4,
  done: Array.from({ length: 16 }, (_, index) => line(index + 1, index + 1 === 14 ? 189 : 0)),
  open: [{ number: 17, task: "open-2", title: "open", stage: "playing", decided: 9, ended: 0, count: 2, playing: [], episodes: [], step: null, error: null }],
  next: Array.from({ length: 17 }, (_, index) => index + 1),
});

const system = (runs: Run[]): System => ({
  at: 100, name: "runs", directory: null, ledger_at: "/ledger", host: "here", written: 10, runs, checkpoints: [], bookmarks: {}, runners: [],
  channels: [], ledger: { fences: {}, tables: {} }, kept: { checkpoints: 0, episodes: 0 }, names: { runs: {}, bookmarks: {} },
});

describe("toward a step", () => {
  afterEach(cleanup);

  it("counts the groups with something to train on against groups_per_step, and those with nothing apart", () => {
    const toward = towardStep(stuck());
    expect(toward.learnable).toEqual([14]);
    expect(toward.nothing).toHaveLength(15);  // (group 17 plays: neither)
    expect(learnableText(toward)).toBe("1 of 4 groups with something to train");
    expect(nothingText(toward)).toBe(`15 gave nothing to train on: ${SAME}`);
  });

  it("says nothing of skipped groups where there are none, and no total where the run says none", () => {
    const run = { ...stuck(), groups_per_step: null, done: [line(1, 8)], next: [1], open: [] };
    const toward = towardStep(run);
    expect(learnableText(toward)).toBe("1 group with something to train");
    expect(nothingText(toward)).toBe("");
    const mixed = { ...stuck(), done: [{ ...line(1, 0), skipped: "1 of 4 episodes completed" }, line(2, 0)], next: [1, 2] };
    expect(nothingText(towardStep(mixed))).toBe("2 gave nothing to train on");  // (reasons that differ: none said)
  });

  it("shows both on the run's tile", () => {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system([stuck()]));
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><Runs /></MemoryRouter>
      </QueryClientProvider>,
    );
    const tile = screen.getByText("stuck").closest("a")!;
    expect(tile.textContent).toContain("1 of 4 groups with something to train");
    expect(tile.textContent).toContain(`15 gave nothing to train on: ${SAME}`);
    expect(tile.textContent).not.toContain("toward a step");
  });
});
