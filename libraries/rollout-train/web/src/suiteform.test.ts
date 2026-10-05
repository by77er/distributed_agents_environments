import { describe, expect, it } from "vitest";
import { DRAWN, fieldsOf, missingOf, suiteBody } from "./lib/suites";

describe("a new suite's form", () => {
  it("starts an entry drawing once from every row, so naming the suite and its environment is enough", () => {
    const entry = { ...fieldsOf(undefined, undefined, DRAWN, "games:words") };
    expect(entry.seeds).toBe("0");
    const { errors, suite } = suiteBody([entry]);
    expect(missingOf("math", [entry], errors, suite)).toEqual([]);
  });

  it("says what keeps it from being saved: the name, then each entry's fields", () => {
    const blank = { ...fieldsOf(undefined, undefined, DRAWN), seeds: "" };
    const { errors, suite } = suiteBody([blank]);
    expect(missingOf("", [blank], errors, suite)).toEqual(["a name", "entry 1: environment, seeds"]);
  });
});
