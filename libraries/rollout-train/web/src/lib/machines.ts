// The machines and the roles on them, as /api/machines says: what each role is called, the roles on one machine, how
// full a pool or a runner is, and a machine's measurements as series to draw.

import type { Host, Machines, PoolRole, Role, RoleKind, RunnerRole } from "../api/types";

/** The role kinds in the order the page shows them, each with its section's title and its word in the sidebar. */
export const KINDS: [RoleKind, string, string][] = [
  ["runners", "Runners", "runner"],
  ["pools", "Sandbox pools", "pool"],
  ["engines", "Engine hosts", "engines"],
  ["launchers", "Launchers", "launcher"],
  ["gateways", "Gateways", "gateway"],
];

export type Roles = Pick<Machines, RoleKind>;

/** The roles on one machine, by kind. */
export function rolesOn(machines: Roles, host: string): Roles {
  const on = <T extends Role>(roles: T[]) => roles.filter(role => role.host === host);
  return {
    runners: on(machines.runners), pools: on(machines.pools), engines: on(machines.engines),
    launchers: on(machines.launchers), gateways: on(machines.gateways),
  };
}

/** The kinds that have any role, in the page's order. */
export const present = (roles: Roles): RoleKind[] => KINDS.map(([kind]) => kind).filter(kind => roles[kind].length > 0);

/** How full something is, from 0 to 1; none where its size is not known. */
export const share = (used: number | null | undefined, size: number | null | undefined): number | null =>
  size ? Math.min(Math.max((used ?? 0) / size, 0), 1) : null;

/** What the alive runners hold together: places in all, used and free. */
export function placesOf(runners: RunnerRole[]): { places: number; used: number; free: number } {
  const alive = runners.filter(runner => runner.alive);
  const places = alive.reduce((sum, runner) => sum + runner.places, 0);
  const used = alive.reduce((sum, runner) => sum + Math.min(runner.playing, runner.places), 0);
  return { places, used, free: places - used };
}

/** What the alive pools hold together, by the kind of sandbox: size, leased and free (a pool whose size is not known
 * is left out), and how many pools. */
export function sandboxesOf(pools: PoolRole[]): { kind: string; pools: number; size: number; leased: number; free: number }[] {
  const found = new Map<string, { kind: string; pools: number; size: number; leased: number; free: number }>();
  for (const pool of pools) {
    if (!pool.alive || pool.size == null) continue;
    const each = found.get(pool.kind) ?? { kind: pool.kind, pools: 0, size: 0, leased: 0, free: 0 };
    each.pools += 1;
    each.size += pool.size;
    each.leased += Math.min(pool.leased ?? 0, pool.size);
    each.free = each.size - each.leased;
    found.set(pool.kind, each);
  }
  return [...found.values()].sort((a, b) => a.kind.localeCompare(b.kind));
}

/** What a role is called on its machine: its name without the host's (`gpu-1/train` on gpu-1 is `train`; `launcher/gpu-1`
 * on gpu-1 is nothing more than a launcher). */
export function shortName(kind: RoleKind, name: string, host: string): string {
  if (kind === "pools") return name.split("@")[0];
  const parts = name.split("/").filter(part => part !== host && part !== "launcher" && part !== "gateway" && part !== "pools");
  return parts.join("/");
}

/** Seconds a lease has left before its time limit; none where it has none. */
export const leftOf = (lease: { at: number; seconds: number | null }, now: number): number | null =>
  lease.seconds == null ? null : Math.max(lease.at + lease.seconds - now, 0);

const GIB = 2 ** 30;

interface MachineSeries {
  memory: [number, number][];
  disk: [number, number][];
  accelerators: { name: string; used: [number, number][]; busy: [number, number][] }[];
}

/** A machine's measurements over its recent beats as series: memory and disk in use (GiB), and each accelerator's
 * memory in use (GiB) and how busy (0 to 1), by when each was measured. */
export function seriesOf(history: Host["history"]): MachineSeries {
  const memory: [number, number][] = [], disk: [number, number][] = [];
  const accelerators: MachineSeries["accelerators"] = [];
  for (const { at, machine } of history) {
    if (machine.memory.total) memory.push([at, (machine.memory.total - (machine.memory.available ?? 0)) / GIB]);
    if (machine.disk) disk.push([at, (machine.disk.total - machine.disk.free) / GIB]);
    machine.accelerators.forEach((each, place) => {
      accelerators[place] ??= { name: each.name, used: [], busy: [] };
      accelerators[place].used.push([at, each.used / GIB]);
      accelerators[place].busy.push([at, each.busy]);
    });
  }
  return { memory, disk, accelerators };
}
