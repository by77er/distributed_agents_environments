// The queue, as /api/queue says it: each resource's capacity split into the admitted runs' shares, and why each run that
// waits does, in words.

import type { Amounts, Queue, QueuePending, QueueResource } from "../api/types";
import { bytes } from "./format";
import { waitingFor } from "./launches";

/** The resources the queue counts, in the page's order, each with its name. */
export const RESOURCES: [QueueResource, string][] = [["cpu", "CPU"], ["memory", "Memory"], ["gpu", "GPU"]];

const number = (value: number): string => String(+value.toPrecision(3));

/** An amount of a resource in words: `2.5 CPUs`, `6.0 GiB`, `1 GPU`. */
export function amount(resource: QueueResource, value: number): string {
  if (resource === "memory") return bytes(value);
  const unit = resource === "cpu" ? "CPU" : "GPU";
  return `${number(value)} ${unit}${value === 1 ? "" : "s"}`;
}

/** An amount as a table's cell: a number of CPUs or GPUs, memory with its unit; a dash for none. */
export const cell = (resource: QueueResource, value: number | undefined): string =>
  !value ? "–" : resource === "memory" ? bytes(value) : number(value);

/** One admitted run's part of a resource's bar: how much it holds, and where its segment starts and how wide it is, as
 * shares of the bar (0 to 1). */
export interface Segment {
  run: string | null;
  name: string;
  amount: number;
  start: number;
  width: number;
}

/** A resource's bar: its capacity, what is used, each admitted run's segment in the order they were admitted, and what
 * is used beyond them (`other`: what no run the queue names holds) as a share; none where the capacity is not known, or
 * is nothing. */
export interface Bar {
  resource: QueueResource;
  capacity: number;
  used: number;
  segments: Segment[];
  other: { start: number; width: number } | null;
}

export function barOf(queue: Queue, resource: QueueResource): Bar | null {
  const capacity = queue.capacity[resource];
  if (!capacity) return null;
  const segments: Segment[] = [];
  let at = 0;
  for (const each of queue.admitted) {
    const held = each.requests[resource] ?? 0;
    if (held <= 0) continue;
    const width = Math.min(held / capacity, Math.max(1 - at, 0));
    segments.push({ run: each.run, name: each.name ?? each.run ?? "?", amount: held, start: at, width });
    at += width;
  }
  const held = queue.admitted.reduce((sum, each) => sum + (each.requests[resource] ?? 0), 0);
  const used = queue.used[resource] ?? held;
  const beyond = Math.min(Math.max(used - held, 0) / capacity, Math.max(1 - at, 0));
  return { resource, capacity, used, segments, other: beyond > 1e-6 ? { start: at, width: beyond } : null };
}

/** Names joined as a sentence says them: `a`, `a and b`, `a, b and 2 more`. */
function joined(names: string[]): string {
  if (names.length <= 2) return names.join(" and ");
  return `${names.slice(0, 2).join(", ")} and ${names.length - 2} more`;
}

/** Amounts in words, GPUs first: `1 GPU and 2.0 GiB`. */
const amounts = (given: Amounts): string =>
  joined((["gpu", "cpu", "memory"] as QueueResource[]).filter(key => given[key]).map(key => amount(key, given[key]!)));

/** Kueue's reason, shortened: without the pod set and the flavor it names, a GPU by its plain name. */
export const shortReason = (reason: string): string =>
  reason.replace(/couldn't assign flavors to pod set [^:]+: /g, "").replace(/ in flavor [^,;]+/g, "").replace(/nvidia\.com\/gpu/g, "GPU");

/** Why a run waits, in words: more than the queue holds at all; what it lacks and the runs that hold it (`name` says a run,
 * by id); else its source's reason (Kueue's, shortened; what its driver waits for, in a few words). */
export function waitsFor(pending: QueuePending, queue: Queue, name: (run: string | null) => string): string {
  const over = (Object.keys(pending.requests) as QueueResource[])
    .filter(key => queue.capacity[key] != null && (pending.requests[key] ?? 0) > queue.capacity[key]! + 1e-9);
  if (over.length) {
    return `asks for ${amounts(Object.fromEntries(over.map(key => [key, pending.requests[key]])))}; the queue holds ${amounts(Object.fromEntries(over.map(key => [key, queue.capacity[key]])))}`;
  }
  if (Object.values(pending.lacks).some(Boolean)) {
    const holders = pending.held_by.map(name);
    const by = holders.length ? `: in use by run${holders.length > 1 ? "s" : ""} ${joined(holders)}` : "";
    return `waits for ${amounts(pending.lacks)}${by}`;
  }
  if (!pending.reason) return queue.source === "kueue" ? "waits for admission" : "waits";
  return queue.source === "kueue" ? shortReason(pending.reason) : waitingFor(pending.reason);
}

/** A place in the queue in words: `1st`, `2nd`, `23rd`. */
export function ordinal(place: number): string {
  const tens = place % 100, ones = place % 10;
  const suffix = tens >= 11 && tens <= 13 ? "th" : ones === 1 ? "st" : ones === 2 ? "nd" : ones === 3 ? "rd" : "th";
  return `${place}${suffix}`;
}

/** Where a run waits in the queue, if it does. */
export const pendingOf = (queue: Queue | undefined, run: string): QueuePending | undefined =>
  queue?.pending?.find(each => each.run === run);

/** A run that waits, as its state line says it: its place in the queue, and why it waits. */
export const queued = (pending: QueuePending, queue: Queue, name: (run: string | null) => string): string =>
  `${ordinal(pending.position)} in the queue · ${waitsFor(pending, queue, name)}`;
