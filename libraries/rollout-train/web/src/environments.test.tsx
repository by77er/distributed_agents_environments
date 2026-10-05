import { QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { newQueryClient } from "./api/queries";
import type { CheckGroup, EnvironmentRow, EnvironmentScore, EnvironmentVersion } from "./api/types";
import { ImportForm } from "./components/environments";
import { bySubject, checkFound, checkText, pickable, publishedParts, rangeText, readable, rowCounts, rowTotals, solvedShare, sourceText, stageText, versionsText } from "./lib/environments";
import { environmentPlace, launchOn, placeOf } from "./lib/places";

const ID = "3f2a9c1b2d4e".padEnd(64, "0");

const row = (key: string, groups: number, played: number, solved: number | null, said: number, held = 0, trains = true): EnvironmentRow => ({
  key, title: key, trains, groups, played, solved, said, reward: null, held,
});

const score = (run: string, checkpoint: string | null, started: number, model: string | null = null): EnvironmentScore => ({
  run, name: run, suite: "words", version: "words@1", kind: checkpoint ? "checkpoint" : "model", checkpoint, model, started, done: true,
  played: 2, solved: 1, said: 2, share: 0.5, reward: 0.5,
});

const group = (number: number, rewards: number[] | null, flagged = false): CheckGroup => ({
  group: number, task: "say-yes", episodes: 4, rewards, solved: null, failed: 0, failures: [], flagged, skipped: flagged ? "every episode scored the same" : null,
});

describe("an environment's rows", () => {
  it("say the share solved of the episodes whose results say, and nothing where none say", () => {
    expect(solvedShare(row("a", 2, 8, 6, 8))).toBe(0.75);
    expect(solvedShare(row("a", 2, 8, null, 0))).toBeNull();
    expect(solvedShare(row("a", 0, 0, 0, 0))).toBeNull();
  });

  it("add up across rows, solved only of the rows whose runs say", () => {
    const rows = [row("a", 2, 8, 6, 8), row("b", 1, 5, null, 0), row("c", 0, 0, null, 0, 2)];
    expect(rowTotals(rows)).toEqual({ groups: 3, played: 13, solved: 6, said: 8 });
    expect(rowTotals([row("b", 1, 5, null, 0)]).solved).toBeNull();
  });

  it("count those played, those held out, and those only the eval data has", () => {
    const rows = [row("a", 2, 8, 6, 8, 1), row("b", 0, 0, null, 0, 2), row("x", 0, 0, null, 0, 1, false)];
    expect(rowCounts(rows)).toEqual({ rows: 2, played: 1, held: 3, evalOnly: 1 });
  });
});

describe("an environment's evals", () => {
  it("are grouped by who played them, the one evaluated most lately first, each's evals newest first", () => {
    const grouped = bySubject([score("e1", "ckpt-a", 1), score("e2", "ckpt-b", 2), score("e3", "ckpt-a", 3), score("e4", null, 0, "tiny")]);
    expect(grouped.map(each => [each.key, each.scores.map(found => found.run)])).toEqual([
      ["ckpt-a", ["e3", "e1"]], ["ckpt-b", ["e2"]], ["model:tiny", ["e4"]],
    ]);
    expect(grouped[2]).toMatchObject({ checkpoint: null, model: "tiny" });
  });
});

describe("a check", () => {
  it("says how many of its groups have something to teach, and plainly when none has", () => {
    expect(checkFound([group(1, [1, 0, 1, 0]), group(2, [0, 0, 0, 0], true), group(3, null)])).toEqual({
      played: 2, flagged: 1, says: "1 of 2 groups have something to teach; in 1, every episode scored the same",
    });
    expect(checkFound([group(1, [0, 0], true)]).says).toBe("every episode of each of the 1 groups scored the same: training would take no step");
    expect(checkFound([group(1, null)]).says).toBe("no group played yet");
  });
});

describe("an environment's head", () => {
  it("says its reward range with open sides, and its version as it loads first", () => {
    const description = { rewards: [0, null] as [number | null, number | null], solved: true, saturated: false, duration: null, observations: null };
    expect(rangeText(description)).toBe("[0, ∞]");
    expect(versionsText({ version: "2", versions: ["1", "2"] })).toEqual(["2", "1"]);
    expect(versionsText({ version: null, versions: ["1"] })).toEqual(["1"]);
  });

  it("is at an address of its module:name, and its new run's form names it", () => {
    const name = "tests.rollout_train.rollouts.games:words";
    expect(placeOf(environmentPlace(name))).toEqual({ page: "environments", kind: "environment", environment: name });
    expect(placeOf("/environments")).toEqual({ page: "environments", kind: "environments" });
    expect(launchOn(name)).toBe("/runs/new?environment=tests.rollout_train.rollouts.games%3Awords");
  });
});

describe("a published environment", () => {
  it("is said by its name and the start of its version, and told apart from a built-in one", () => {
    expect(publishedParts(`words@${ID}`)).toEqual({ name: "words", version: ID });
    expect(publishedParts("gridworld.environment:environment")).toBeNull();
    expect(publishedParts("words@latest")).toBeNull();
    expect(readable(`words@${ID}`)).toBe("words@3f2a9c1b2d4e");
    const other = `words@${"9".repeat(64)}`;
    expect(pickable([], [`words@${ID}`, other]).map(each => each.name)).toEqual(["words@3f2a9c1b2d4e", "words@999999999999"]);
  });

  it("says where its source came from and what its check found", () => {
    expect(sourceText({ source: "https://github.com/someone/games.git", subdirectory: "environments/words" })).toBe("github.com/someone/games/environments/words");
    expect(sourceText({ source: "/srv/git/words", subdirectory: "" })).toBe("/srv/git/words");
    expect(checkText([{ check: "rows", passed: true, said: "" }, { check: "episode", passed: true, said: "", flagged: true }])).toBe("2 passed, 1 flagged");
    expect(checkText([{ check: "rows", passed: false, said: "" }, { check: "starts", passed: true, said: "" }])).toBe("1 of 2 failed");
    expect([stageText("checking"), stageText("done"), stageText(undefined)]).toEqual(["checking on Ray", "imported", "starting"]);
  });
});

describe("the import form", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  const version: EnvironmentVersion = {
    reference: `words@${ID}`, name: "words", version: ID, source: "https://example.com/words.git", ref: null, commit: "c0ffee",
    subdirectory: "envs/words", entry_point: "words:environment", check: [], imported: 1,
  };

  function answering(imported: Response) {
    const fetched = vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method === "POST") return imported;
      return new Response(JSON.stringify(path.includes("imports") ? { imports: [], importing: true } : { environments: [] }), { headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetched);
    return fetched;
  }

  function filled(onDone: (made: EnvironmentVersion) => void = () => {}) {
    render(<QueryClientProvider client={newQueryClient()}><ImportForm onDone={onDone} onCancel={() => {}} /></QueryClientProvider>);
    const [url, ref, subdirectory, entry] = screen.getAllByRole("textbox");
    fireEvent.change(url, { target: { value: " https://example.com/words.git " } });
    fireEvent.change(ref, { target: { value: "v1" } });
    fireEvent.change(subdirectory, { target: { value: "envs/words" } });
    fireEvent.change(entry, { target: { value: "" } });
    act(() => { screen.getByRole("button", { name: "Import" }).click(); });
  }

  it("asks the monitor to import what it says, and is done with the version made", async () => {
    const fetched = answering(new Response(JSON.stringify({ version, existing: false }), { headers: { "Content-Type": "application/json" } }));
    const done = vi.fn();
    filled(done);
    await waitFor(() => expect(done).toHaveBeenCalledWith(version));
    const [, init] = fetched.mock.calls.find(([path]) => path === "api/environments/import")!;
    expect(JSON.parse(String(init?.body))).toEqual({ url: "https://example.com/words.git", ref: "v1", subdirectory: "envs/words", entry_point: "" });
  });

  it("says why the monitor refused the import", async () => {
    answering(new Response(JSON.stringify({ error: "it has no envs/words/pyproject.toml" }), { status: 422, headers: { "Content-Type": "application/json" } }));
    filled();
    expect((await screen.findByRole("alert")).textContent).toBe("it has no envs/words/pyproject.toml");
  });
});
