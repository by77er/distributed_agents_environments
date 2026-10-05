// An episode's advantage as the page shows it: its reward against its group's mean, where the group was trained on. The
// sign is what every policy-gradient preset shares: above the mean, its tokens were made likelier.

import type { DoneLine } from "../api/types";

/** The group's mean reward, where the group was trained on (not skipped, and some of it trained); null otherwise. */
export function baselineOf(result: DoneLine | null, rewards: number[]): number | null {
  if (!result || result.skipped || !(result.segments > 0) || !rewards.length) return null;
  return rewards.reduce((sum, each) => sum + each, 0) / rewards.length;
}

/** An episode's reward against the baseline; null where either is unknown. */
export function advantageOf(baseline: number | null, reward: number | null | undefined): number | null {
  return baseline == null || reward == null ? null : reward - baseline;
}
