import { describe, expect, it } from "vitest";
import type { DoneLine } from "./api/types";
import { advantageOf, baselineOf } from "./lib/advantage";

const line = (more: Partial<DoneLine>): DoneLine => ({ rewards: [0, 0, 0, 1], segments: 8, skipped: null, ...more } as unknown as DoneLine);

describe("an episode's advantage", () => {
  it("is its reward against its group's mean where the group was trained on", () => {
    const baseline = baselineOf(line({}), [0, 0, 0, 1]);
    expect(baseline).toBe(0.25);
    expect(advantageOf(baseline, 1)).toBe(0.75);
    expect(advantageOf(baseline, 0)).toBe(-0.25);
  });

  it("is unknown where the group was skipped, trained nothing, or is still playing", () => {
    expect(baselineOf(line({ skipped: "every episode scored the same", segments: 0 }), [1, 1, 1, 1])).toBeNull();
    expect(baselineOf(line({ segments: 0 }), [0, 1])).toBeNull();
    expect(baselineOf(null, [0, 1])).toBeNull();
    expect(advantageOf(null, 1)).toBeNull();
  });
});
