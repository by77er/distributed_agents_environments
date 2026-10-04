// The time on the monitor's machine, as near as the page can tell, ticking: "wrote 3 min ago" counts up between
// readings without anything being read again. Only the views that say how long ago draw on each tick.

import { useSyncExternalStore } from "react";

let skew = 0;  // seconds the monitor's clock is ahead of this browser's
let now = Date.now() / 1000;
const listeners = new Set<() => void>();

export function setServerTime(at: number): void {
  skew = at - Date.now() / 1000;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

setInterval(() => {
  now = Math.round(Date.now() / 1000 + skew);
  for (const listener of listeners) listener();
}, 1000);

/** Seconds since the epoch on the monitor's machine, to the second. */
export const useNow = (): number => useSyncExternalStore(subscribe, () => now);
