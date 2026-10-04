// What this browser remembers of the page (folds, the runs left out of the statistics, a chart's window): kept in
// localStorage, read and written through a store so every view over it draws when it changes.

import { useCallback, useSyncExternalStore } from "react";

const listeners = new Map<string, Set<() => void>>();
const cache = new Map<string, unknown>();

function read<T>(key: string, otherwise: T): T {
  if (cache.has(key)) return cache.get(key) as T;
  let value = otherwise;
  try {
    const text = localStorage.getItem(key);
    if (text != null) value = JSON.parse(text) as T;
  } catch {
    /* (a browser that keeps nothing) */
  }
  cache.set(key, value);
  return value;
}

export function write<T>(key: string, value: T): void {
  cache.set(key, value);
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* (a browser that keeps nothing) */
  }
  for (const listener of listeners.get(key) ?? []) listener();
}

export function useStored<T>(key: string, otherwise: T): [T, (value: T) => void] {
  const value = useSyncExternalStore(
    listener => {
      if (!listeners.has(key)) listeners.set(key, new Set());
      listeners.get(key)!.add(listener);
      return () => listeners.get(key)!.delete(listener);
    },
    () => read(key, otherwise),
  );
  const set = useCallback((next: T) => write(key, next), [key]);
  return [value, set];
}

/** Folds (a run, a step, a group, an episode, a lane of the versions' graph): open or closed, where the reader said so. */
export type Folds = Record<string, boolean>;
export const FOLDS = "monitor.folds";

export function useFolds(): [Folds, (key: string, open: boolean) => void, (changes: Folds) => void] {
  const [folds, setFolds] = useStored<Folds>(FOLDS, {});
  const fold = useCallback((key: string, open: boolean) => setFolds({ ...read<Folds>(FOLDS, {}), [key]: open }), [setFolds]);
  const many = useCallback((changes: Folds) => setFolds({ ...read<Folds>(FOLDS, {}), ...changes }), [setFolds]);
  return [folds, fold, many];
}
