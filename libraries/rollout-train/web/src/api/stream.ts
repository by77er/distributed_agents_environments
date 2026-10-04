// The monitor says when a topic changed (server-sent events from /api/stream); the page then reads that topic again,
// and only that one. The topics watched are those the place shown needs: moving elsewhere watches others.

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useSyncExternalStore } from "react";
import { versionHeld } from "./client";
import type { Topic } from "./queries";

export type Connection = "connecting" | "live" | "lost";

let connection: Connection = "connecting";
const listeners = new Set<() => void>();
const setConnection = (next: Connection) => {
  if (next === connection) return;
  connection = next;
  for (const listener of listeners) listener();
};

/** Whether the page hears from the monitor: connecting, live, or lost (it tries again every few seconds). */
export const useConnection = (): Connection =>
  useSyncExternalStore(
    listener => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => connection,
  );

export function useStream(watched: Topic[]): void {
  const client = useQueryClient();
  const names = watched.map(each => each.topic).sort().join("\n");
  useEffect(() => {
    const byTopic = new Map(watched.map(each => [each.topic, each]));
    const query = [...byTopic.keys()].map(topic => `topic=${encodeURIComponent(topic)}`).join("&");
    const source = new EventSource(`api/stream?${query}`);
    source.addEventListener("hello", () => setConnection("live"));
    source.addEventListener("version", event => {
      const { topic, version } = JSON.parse((event as MessageEvent<string>).data) as { topic: string; version: string };
      const each = byTopic.get(topic);
      if (!each) return;
      // (an episode's version is not an ETag: its lines are asked for after those the page has, whatever it says)
      if (!topic.startsWith("episode/") && versionHeld(each.path) === version) return;
      void client.invalidateQueries({ queryKey: each.key, exact: true });
    });
    source.onerror = () => setConnection("lost");  // (the browser tries again by itself; `hello` says it is back)
    return () => source.close();
    // (the topics, by name: a new list of the same topics keeps the stream it has)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, names]);
}
