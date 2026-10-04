import { describe, expect, it } from "vitest";
import type { CheckpointEval } from "./api/types";
import { historyOf, historySeries, scoresOf } from "./lib/history";
import { placeOf, subjectPlace } from "./lib/places";

const WORDS = "games:words", GUESSING = "games:guessing";

const entry = (environment: string, share: number | null, reward: number | null, played = 2) => ({ environment, played, solved: share == null ? null : share * played, share, reward });

const anEval = (run: string, version: string, started: number | null, entries: ReturnType<typeof entry>[] = []): CheckpointEval => ({
  suite: version.split("@")[0], version, run, name: run, kind: "checkpoint", checkpoint: "kpqx", model: null, asked_by: "by hand", by: null, by_name: null,
  step: null, episodes: 1, played: 2, expected: 2, solved: 1, share: 0.5, reward: 0.25, started, at: started, done: true, entries,
});

describe("a subject's history", () => {
  it("groups its evals by the version each played: suites by name, a suite's versions newest first, evals newest first", () => {
    const groups = historyOf([
      anEval("e1", "words@1", 10, [entry(WORDS, 0.5, 0.5)]),
      anEval("e2", "words@2", 30, [entry(WORDS, 1, 1), entry(GUESSING, 0, 0)]),
      anEval("e3", "words@1", 20, [entry(WORDS, 1, 1)]),
      anEval("e4", "arith@1", 5, [entry("games:arith", null, 0.2)]),
    ]);
    expect(groups.map(group => [group.version, group.label])).toEqual([["arith@1", "arith"], ["words@2", "words v2"], ["words@1", "words v1"]]);
    expect(groups[2].evals.map(each => each.run)).toEqual(["e3", "e1"]);
    expect(groups[1].environments).toEqual([WORDS, GUESSING]);
  });

  it("scores each eval at each environment of its group, apart, and an eval that says no entries in one column", () => {
    const [group] = historyOf([anEval("e1", "words@2", 10, [entry(GUESSING, 0, 0.1)]), anEval("e2", "words@2", 20, [entry(WORDS, 1, 1), entry(GUESSING, 0.5, 0.4)])]);
    expect(group.environments).toEqual([GUESSING, WORDS]);
    expect(scoresOf(group, group.evals[1])).toEqual([{ played: 2, share: 0, reward: 0.1 }, null]);  // (it did not play words)
    const [older] = historyOf([anEval("e0", "words@1", 5)]);
    expect(older.environments).toEqual([null]);
    expect(scoresOf(older, older.evals[0])).toEqual([{ played: 2, share: 0.5, reward: 0.25 }]);
  });

  it("draws each environment over time: the share solved where its episodes say, else the mean reward, oldest first", () => {
    const [group] = historyOf([
      anEval("e2", "words@2", 30, [entry(WORDS, 1, 1), entry(GUESSING, null, 0.7)]),
      anEval("e1", "words@2", 10, [entry(WORDS, 0.5, 0.5), entry(GUESSING, null, 0.3, 0)]),  // (played nothing of guessing)
      anEval("unstarted", "words@2", null, [entry(WORDS, 0, 0)]),
    ]);
    expect(historySeries(group)).toEqual([
      { environment: WORDS, measure: "solved", points: [[10, 0.5], [30, 1]] },
      { environment: GUESSING, measure: "reward", points: [[30, 0.7]] },
    ]);
  });

  it("has a place of its own on the evals page: a checkpoint's by id, a base model's by its whole name", () => {
    expect(placeOf(subjectPlace("checkpoint", "kpqx01"))).toEqual({ page: "evals", kind: "subject", subject: "checkpoint", id: "kpqx01" });
    expect(placeOf(subjectPlace("model", "Qwen/Qwen3-8B"))).toEqual({ page: "evals", kind: "subject", subject: "model", id: "Qwen/Qwen3-8B" });
    expect(placeOf("/evals/model/Qwen/Qwen3-8B")).toEqual({ page: "evals", kind: "subject", subject: "model", id: "Qwen/Qwen3-8B" });
    expect(placeOf("/evals/checkpoint")).toEqual({ page: "evals", kind: "suite", suite: "checkpoint" });  // (a suite of that name)
  });
});
