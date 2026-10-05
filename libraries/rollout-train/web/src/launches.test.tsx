import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { newQueryClient, topics } from "./api/queries";
import type { Launch, Preset, Run, System } from "./api/types";
import { LaunchList } from "./components/launches";
import { changedSettings, launchState, loopChanged, settingLabel, settingValue, waitingFor } from "./lib/launches";

const launch = (state: Launch["state"], more: Partial<Launch> = {}, settings: Record<string, unknown> = {}): Launch => ({
  id: "launch_1", asked: { kind: "train", name: "alpha", settings: { environment: "games:words", ...settings }, preset: null, resumes: null },
  at: 100, state, run: "run_1", job: "run-1", backend: "ray", detail: null, updated: 160, ...more,
});

const run = (state: string, pause = false): Run => ({ run: "run_1", name: "alpha", state, pause } as unknown as Run);

describe("a launch's state", () => {
  it("is said in the run's words once its run exists, and as asked, submitted or waiting before", () => {
    expect(launchState(launch("asked")).state).toBe("asked");
    expect(launchState(launch("submitted")).state).toBe("submitted");
    const pending = launchState(launch("submitted", { detail: "Job has not started yet." }));
    expect([pending.state, pending.reason]).toEqual(["waiting", "Job has not started yet."]);
    const waits = launchState(launch("running", { detail: "waits for run/r/engine/policy/0 (1 GPU: pending creation), run/r/trainer (0.5 GPU and 1 CPU on the driver's node)" }));
    expect([waits.state, waits.reason, waits.kind]).toEqual(["waiting", "no GPU free (1.5 wanted)", "warm"]);
    expect(launchState(launch("running", { detail: "running" }), run("running")).state).toBe("running");
    expect(launchState(launch("running"), run("idle")).state).toBe("running");
    expect(launchState(launch("running"), run("paused", true)).state).toBe("paused");
    expect(launchState(launch("running"), run("lost")).state).toBe("lost");
    expect(launchState(launch("stopping")).state).toBe("stopping");
  });

  it("says finished, never ended, and failed or stopped as the launch ended", () => {
    expect(launchState(launch("ended"), run("ended")).state).toBe("finished");
    expect(launchState(launch("failed", { detail: "refused: name: taken" })).state).toBe("failed");
    expect(launchState(launch("stopped")).state).toBe("stopped");
    expect(waitingFor("waits for its Ray cluster")).toBe("its Ray cluster");
  });
});

describe("a launch's settings", () => {
  it("are said in words, and only those that differ from the defaults and the preset", () => {
    const preset: Preset = { name: "small", version: 2, id: "small@2", note: "", saved: 1, settings: { "trainer.rank": 32, "trainer.learning_rate": 5e-5 } };
    const settings = {
      environment: "games:words", start: "kpqx", groups: 100, seed: 0, "trainer.rank": 32, "trainer.learning_rate": 1e-4, "evals.suite": null,
      "evals.every": 1, "channels.policy.thinking_tokens": null, "channels.judge.model": "org/judge", "limits.spend": 2, self_judging: true,
      "trainer.colocated": false,
    };
    expect(changedSettings(settings, preset).map(each => each.text)).toEqual([
      "judge model org/judge", "no thinking budget", "no evals", "spend up to $2", "self-judging yes", "colocated no", "learning rate 1e-4",
    ]);
    expect(changedSettings({ "evals.suite": "math", "trainer.rank": 16 }).map(each => each.text)).toEqual(["evals on math", "rank 16"]);
  });

  it("format numbers and booleans, and leave the loop's defaults out", () => {
    expect([settingValue(5e-5), settingValue(0.25), settingValue(4096), settingValue(true), settingValue(null), settingValue([1, 2])])
      .toEqual(["5e-5", "0.25", "4,096", "yes", "none", "1, 2"]);
    expect([settingLabel("channels.policy.provider"), settingLabel("slots.judge"), settingLabel("trainer.tokens_per_step"), settingLabel("odd.key")])
      .toEqual(["inference", "judge slot", "tokens a step", "odd.key"]);
    expect(loopChanged({ groups: 100, groups_per_step: 4, seed: 0 })).toEqual([]);
    expect(loopChanged({ groups: 12, groups_per_step: 2, seed: 3 })).toEqual(["12 groups", "2 a step", "seed 3"]);
  });
});

describe("a launch's tile", () => {
  beforeEach(() => { vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { headers: { "Content-Type": "application/json" } }))); });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  const system = { runs: [run("running")], checkpoints: [], bookmarks: {}, names: { runs: {}, bookmarks: {} } } as unknown as System;

  function tile(given: Launch): HTMLElement {
    const client = newQueryClient();
    client.setQueryData(topics.system().key, system);
    render(<QueryClientProvider client={client}><MemoryRouter><LaunchList launches={[given]} system={system} /></MemoryRouter></QueryClientProvider>);
    return screen.getByRole("link", { name: "alpha" }).closest(".tile") as HTMLElement;
  }

  it("names its run once, as a link, with its state in the run's words and the time once", () => {
    const shown = tile(launch("running", { detail: "running" }, { groups: 12, "evals.suite": null }));
    expect(shown.querySelectorAll("b").length).toBe(1);
    expect(shown.querySelector("header")!.textContent).toBe("alphawordsrunning");
    expect(shown.textContent).toContain("12 groups");
    expect(shown.textContent).toContain("no evals");
    expect(shown.textContent).toMatch(/asked .* ago/);
    expect(shown.textContent).not.toContain("run-1");  // (the job only in a title)
  });

  it("once done, says how long it took and keeps why it failed folded", () => {
    const shown = tile(launch("failed", { detail: "Traceback: it broke" }));
    expect(shown.textContent).toMatch(/took 60 s · .* ago/);
    expect(shown.querySelector("details summary")!.textContent).toBe("why it failed");
  });
});
