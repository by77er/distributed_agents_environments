// The machines in the sidebar: each machine, alive ones first, opening to the roles on it; a role opens its card on
// the machine's page.

import { useMachines } from "../api/queries";
import type { Machines, RoleKind } from "../api/types";
import { Twist } from "../components/ui";
import { KINDS, shortName } from "../lib/machines";
import { hostPlace, type Place } from "../lib/places";
import { useFolds } from "../lib/stored";
import { Node } from "./Tree";

/** How full a role is, as its row's tag: a runner's places playing, a pool's sandboxes leased, a launcher's launches. */
function fullness(machines: Machines, kind: RoleKind, name: string): string {
  if (kind === "runners") {
    const runner = machines.runners.find(each => each.name === name);
    return runner ? `${runner.playing}/${runner.places}` : "";
  }
  if (kind === "pools") {
    const pool = machines.pools.find(each => each.name === name);
    return pool?.size != null ? `${pool.leased}/${pool.size}` : "";
  }
  if (kind === "launchers") {
    const launcher = machines.launchers.find(each => each.name === name);
    return launcher ? `${launcher.playing ?? 0}/${launcher.at_once ?? 1}` : "";
  }
  return "";
}

export function MachinesTree({ place }: { place: Place }) {
  const { data: machines } = useMachines();
  const [folds, fold] = useFolds();
  if (!machines) return null;
  if (!machines.hosts.length) return <div className="empty">No heartbeat yet.</div>;
  const shown = place.kind === "host" ? place.host : null;
  const words = new Map(KINDS.map(([kind, , word]) => [kind, word]));
  const order = new Map(KINDS.map(([kind], index) => [kind, index]));
  return (
    <>
      {machines.hosts.map(host => {
        const key = `host:${host.host}`, open = folds[key] ?? (host.host === shown || machines.hosts.length === 1);
        const roles = [...host.roles].sort((a, b) => order.get(a.kind)! - order.get(b.kind)! || a.name.localeCompare(b.name));
        return (
          <div key={host.host}>
            <Node to={hostPlace(host.host)} current={place.kind === "host" && place.host === host.host && !place.role}>
              <Twist open={open} has={roles.length > 0} onToggle={() => fold(key, !open)} />
              <span className={`dot ${host.alive ? "alive" : "gone"}`} />
              <span className="name">{host.host}</span>
              <span className="tag">{roles.filter(role => role.alive).length}</span>
            </Node>
            {open && roles.length ? (
              <div className="children">
                {roles.map(role => (
                  <Node key={`${role.kind}:${role.name}`} to={hostPlace(host.host, role.name)} current={place.kind === "host" && place.role === role.name}>
                    <span className={`dot ${role.alive ? "alive" : "gone"}`} />
                    <span className="num">{words.get(role.kind)}</span>
                    <span className="name" title={role.name}>{shortName(role.kind, role.name, host.host)}</span>
                    <span className="tag">{fullness(machines, role.kind, role.name)}</span>
                  </Node>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </>
  );
}
