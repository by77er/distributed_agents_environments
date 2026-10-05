// The machines that run things, as their heartbeats and the ledger say: the queue (what the cluster gives runs, and
// who holds and waits for it), then every runner, sandbox pool, engine host and gateway alive, a section of each kind,
// with what each holds and how full it is; and one machine with everything on it.

import { Fragment } from "react";
import { useMachines } from "../api/queries";
import type { Role } from "../api/types";
import { MachineCard, RoleCards } from "../components/machines";
import { QueueSection } from "../components/queue";
import { Empty, Head, SectionTitle, Spec, Specs, Table } from "../components/ui";
import { Ago } from "../layout/runs";
import { bytes } from "../lib/format";
import { KINDS, placesOf, present, rolesOn, type Roles, sandboxesOf } from "../lib/machines";
import { hostPlace } from "../lib/places";

const titleOf = (kind: string) => KINDS.find(([each]) => each === kind)![1];

export function Machines() {
  const { data } = useMachines();
  if (!data) return <Empty>Reading the heartbeats…</Empty>;
  const hosts = new Map(data.hosts.map(host => [host.host, host]));
  const alive: Roles = {
    runners: data.runners.filter(role => role.alive), pools: data.pools.filter(role => role.alive), engines: data.engines.filter(role => role.alive),
    gateways: data.gateways.filter(role => role.alive),
  };
  const gone = KINDS.flatMap(([kind, , word]) => (data[kind] as Role[]).filter(role => !role.alive).map(role => ({ role, word })));
  const places = placesOf(data.runners);
  return (
    <>
      <Head title="Machines">
        {data.hosts.length ? (
          <Specs>
            <Spec label="machines" kind="good">{data.hosts.filter(host => host.alive).length}</Spec>
            {alive.runners.length > 1 ? <Spec label="places">{places.used} of {places.places} playing · {places.free} free</Spec> : null}
            {sandboxesOf(data.pools).filter(each => each.pools > 1).map(each => <Spec key={each.kind} label={`${each.kind} sandboxes`}>{each.leased} of {each.size} leased · {each.free} free</Spec>)}
          </Specs>
        ) : null}
      </Head>
      <QueueSection />
      {data.hosts.length ? null : <Empty>No heartbeat yet.</Empty>}
      {present(alive).map(kind => (
        <Fragment key={kind}>
          <SectionTitle id={`section-${kind}`} title={titleOf(kind)} />
          <RoleCards kind={kind} roles={alive[kind]} hosts={hosts} onHost={false} />
        </Fragment>
      ))}
      {gone.length ? (
        <>
          <SectionTitle id="section-gone" title="Gone" />
          <Table heads={[["name"], ["is"], ["machine"], ["last beat", "n"]]} keys={gone.map(({ role }) => role.name)}
            rows={gone.map(({ role, word }) => [<span className="mono">{role.name}</span>, word, role.host ?? "–", <><Ago at={role.at} /> ago</>])}
            to={gone.map(({ role }) => (role.host ? hostPlace(role.host, role.name) : null))} />
        </>
      ) : null}
    </>
  );
}

export function MachineHost({ host: name }: { host: string }) {
  const { data } = useMachines();
  if (!data) return <Empty>Reading the heartbeats…</Empty>;
  const host = data.hosts.find(each => each.host === name);
  if (!host) return <Empty>No machine called {name} beats.</Empty>;
  const roles = rolesOn(data, name);
  const machine = host.machine;
  return (
    <>
      <Head title={host.host}>
        <Specs>
          <Spec label="state" kind={host.alive ? "good" : "bad"}>{host.alive ? "beating" : "gone"}</Spec>
          <Spec label="last beat"><Ago at={host.at} /> ago</Spec>
          {machine?.memory.total ? <Spec label="memory">{bytes(machine.memory.total)}</Spec> : null}
          {machine?.accelerators.length ? <Spec label="accelerators">{machine.accelerators.map(each => each.name).join(", ")}</Spec> : null}
        </Specs>
      </Head>
      <MachineCard host={host} />
      {present(roles).map(kind => (
        <Fragment key={kind}>
          <SectionTitle id={`section-${kind}`} title={titleOf(kind)} />
          <RoleCards kind={kind} roles={roles[kind]} hosts={new Map([[host.host, host]])} onHost />
        </Fragment>
      ))}
    </>
  );
}
