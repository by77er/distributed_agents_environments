// The machines and the roles on them: a machine's meters and its measurements over the last hour, and a card for each
// runner (and a run's driver waiting for Ray), sandbox pool, engine host and gateway, with what it holds and how full it is.

import { memo, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useKnown } from "../api/queries";
import type { EngineRole, GatewayRole, Host, Measurement, PoolRole, Role, RoleChannel, RoleKind, RunnerRole } from "../api/types";
import { Ago } from "../layout/runs";
import { bytes, figure, percent, plural, span, tick } from "../lib/format";
import { leftOf, seriesOf } from "../lib/machines";
import { useNow } from "../lib/now";
import { episodePlace, groupPlace, hostPlace, runPlace } from "../lib/places";
import { LineChart, Sized } from "./charts";
import { Card, Legend, Meter, Table } from "./ui";

/** A role's card: its name with whether it beats, its machine (linked, off the machine's own page) and its last beat. */
function RoleCard({ role, title, onHost, children }: { role: Role; title: ReactNode; onHost: boolean; children: ReactNode }) {
  return (
    <div id={`role-${role.name}`}>
      <Card className={`machine${role.alive ? "" : " stale"}`}
        title={<span className="machine-title"><span className={`dot ${role.alive ? "alive" : "gone"}`} />{title}</span>}
        note={<>{role.host && !onHost ? <><Link to={hostPlace(role.host)}>{role.host}</Link> · </> : null}{role.alive ? "beat" : "gone: last beat"} <Ago at={role.at} /> ago</>}>
        {children}
      </Card>
    </div>
  );
}

/** A machine's memory, accelerators and disk now. */
function Meters({ machine }: { machine: Measurement | null | undefined }) {
  if (!machine) return null;
  return (
    <>
      {machine.memory.total ? <Meter name="memory" used={machine.memory.total - (machine.memory.available ?? 0)} total={machine.memory.total} says={`${bytes(machine.memory.available)} available of ${bytes(machine.memory.total)}`} /> : null}
      {machine.accelerators.map((each, place) => <Meter key={place} name={each.name} used={each.used} total={each.total} says={`${bytes(each.used)} of ${bytes(each.total)} · ${percent(each.busy)} busy`} />)}
      {machine.disk ? <Meter name="disk" used={machine.disk.total - machine.disk.free} total={machine.disk.total} says={`${bytes(machine.disk.free)} free of ${bytes(machine.disk.total)}`} /> : null}
    </>
  );
}

/** A machine: its meters now, and its memory, accelerators and disk over its recent beats. */
export const MachineCard = memo(function MachineCard({ host }: { host: Host }) {
  const series = seriesOf(host.history);
  const colors = series.accelerators.map((each, place) => ({ name: each.name, color: `var(--series-${place + 2})` }));
  const gib = { yTick: (value: number) => `${tick(value)} GiB`, format: (value: number) => `${value.toFixed(1)} GiB` };
  const enough = host.history.length > 1;
  return (
    <>
      <Card className={`machine${host.alive ? "" : " stale"}`} title="Now" note={<>{host.alive ? "beat" : "gone: last beat"} <Ago at={host.at} /> ago</>}>
        {host.machine ? <Meters machine={host.machine} /> : <p className="muted small">Its beats measure nothing.</p>}
      </Card>
      {enough ? (
        <div className="cols">
          {series.memory.length || series.accelerators.length ? (
            <Card title="Memory in use">
              <Sized>{width => (
                <LineChart width={width} height={160} time label="memory in use" dots={false} {...gib}
                  series={[
                    ...(series.memory.length ? [{ name: "memory", color: "var(--series-1)", points: series.memory }] : []),
                    ...series.accelerators.map((each, place) => ({ ...colors[place], points: each.used })),
                  ]} />
              )}</Sized>
              <Legend items={[...(series.memory.length ? [{ name: "memory", color: "var(--series-1)" }] : []), ...colors]} />
            </Card>
          ) : null}
          {series.accelerators.length ? (
            <Card title="Accelerators busy">
              <Sized>{width => (
                <LineChart width={width} height={160} time label="accelerators busy" dots={false} y={{ min: 0, max: 1 }} yTick={percent} format={percent}
                  series={series.accelerators.map((each, place) => ({ ...colors[place], points: each.busy }))} />
              )}</Sized>
            </Card>
          ) : null}
          {series.disk.length ? (
            <Card title="Disk in use">
              <Sized>{width => <LineChart width={width} height={160} time label="disk in use" dots={false} {...gib} series={[{ name: "disk", color: "var(--quiet)", points: series.disk }]} />}</Sized>
            </Card>
          ) : null}
        </div>
      ) : null}
    </>
  );
});

/** A channel's throughput since the beat before, as a table's cells. */
const throughput = (channel: RoleChannel) => [figure(channel.tokens_per_second), figure(channel.mean_concurrency)];

const RunnerCard = memo(function RunnerCard({ runner, host, onHost }: { runner: RunnerRole; host: Host | undefined; onHost: boolean }) {
  const known = useKnown();
  return (
    <RoleCard role={runner} title={runner.name} onHost={onHost}>
      {runner.waiting?.length ? <div className="small t-warm" title={runner.waiting.join("; ")}>waits for {runner.waiting.join(", ")}</div> : null}
      <Meter name="places" warns={false} used={Math.min(runner.playing, runner.places)} total={runner.places} says={`${runner.playing} of ${runner.places} playing · ${runner.free} free`} />
      {onHost ? null : <Meters machine={host?.machine} />}
      {runner.claims.length ? (
        <Table heads={[["run"], ["group", "n"], ["episode", "n"], ["attempt", "n"], ["since", "n"]]}
          keys={runner.claims.map(claim => `${claim.run}/${claim.group}/${claim.episode}/${claim.attempt}`)}
          rows={runner.claims.map(claim => [
            <Link to={runPlace(claim.run)} onClick={event => event.stopPropagation()}>{known.run(claim.run)}</Link>,
            <Link to={groupPlace(claim.run, claim.group)} onClick={event => event.stopPropagation()}>#{claim.group}</Link>,
            `E${claim.episode}`, String(claim.attempt), <Ago at={claim.at} />,
          ])}
          to={runner.claims.map(claim => (claim.run_id ? episodePlace(claim.run_id) : groupPlace(claim.run, claim.group)))} />
      ) : null}
      {runner.channels.length ? (
        <Table heads={[["channel"], ["serving"], ["tok/s", "n"], ["at once", "n"]]} keys={runner.channels.map(channel => channel.channel)}
          rows={runner.channels.map(channel => [channel.channel, channel.serving ? known.short(channel.serving) : channel.servers ? plural(channel.servers.length, "server") : "base model", ...throughput(channel)])} />
      ) : null}
    </RoleCard>
  );
});

const PoolCard = memo(function PoolCard({ pool, onHost }: { pool: PoolRole; onHost: boolean }) {
  const known = useKnown();
  const now = useNow();
  const says = pool.size == null ? `${pool.leases.length} leased · its size is not known` : `${pool.leased} of ${pool.size} leased · ${pool.free} free`;
  return (
    <RoleCard role={pool} title={`${pool.kind} pool`} onHost={onHost}>
      <div className="small faint mono machine-name" title={pool.name}>{pool.runner ? `in ${pool.runner}` : pool.name}</div>
      <Meter name="sandboxes" warns={false} used={pool.leased ?? pool.leases.length} total={pool.size ?? 0} says={says} />
      {pool.leases.length ? (
        <Table heads={[["episode"], ["sandbox"], ["held", "n"], ["limit", "n"], ["state"]]} keys={pool.leases.map(lease => lease.key)}
          rows={pool.leases.map(lease => {
            const left = leftOf(lease, now);
            return [
              lease.run ? <span>{known.run(lease.run)} <span className="faint">#{lease.group} E{lease.episode}{lease.attempt && lease.attempt > 1 ? ` try ${lease.attempt}` : ""}</span></span> : <span className="mono">{lease.key}</span>,
              lease.sandbox, <Ago at={lease.at} />, lease.seconds == null ? "–" : `${span(lease.seconds)}${left != null ? ` · ${span(left)} left` : ""}`,
              lease.lost ? { text: "lost", kind: "t-bad" } : lease.holds === false ? { text: "claim lapsed", kind: "t-warm" } : { text: "held", kind: "t-good" },
            ];
          })}
          to={pool.leases.map(lease => (lease.run_id ? episodePlace(lease.run_id) : lease.run && lease.group != null ? groupPlace(lease.run, lease.group) : null))} />
      ) : null}
    </RoleCard>
  );
});

const EngineCard = memo(function EngineCard({ engines, host, onHost }: { engines: EngineRole; host: Host | undefined; onHost: boolean }) {
  const known = useKnown();
  const behind = (count: number | null) => (count == null ? "–" : count === 0 ? { text: "0", kind: "t-good" } : { text: String(count), kind: "t-warm" });
  return (
    <RoleCard role={engines} title={engines.name} onHost={onHost}>
      {engines.follows ? <div className="small muted">follows <Link to={runPlace(engines.follows)}>{known.run(engines.follows)}</Link></div> : null}
      {onHost ? null : <Meters machine={host?.machine} />}
      {engines.channels.map(channel => (
        <div key={channel.channel} className="engine-channel">
          <div className="small muted">
            <b>{channel.channel}</b> · serving <span className="mono">{channel.serving ? known.short(channel.serving) : "base model"}</span>
            {channel.wanted ? <> · wants <span className="mono">{known.short(channel.wanted)}</span></> : null}
            {" · "}{figure(channel.tokens_per_second)} tok/s · {figure(channel.mean_concurrency)} at once
          </div>
          {channel.error ? <div className="small t-bad">{channel.error}</div> : null}
          <Table heads={[["engine"], ["serving"], ["behind", "n"]]} keys={channel.engines.map((each, place) => each.address ?? place)}
            rows={channel.engines.map(each => [<span className="mono">{each.address ?? "in process"}</span>, each.serving ? known.short(each.serving) : "base model", behind(each.behind)])} />
        </div>
      ))}
    </RoleCard>
  );
});

const GatewayCard = memo(function GatewayCard({ gateway, onHost }: { gateway: GatewayRole; onHost: boolean }) {
  const known = useKnown();
  return (
    <RoleCard role={gateway} title={gateway.name} onHost={onHost}>
      {gateway.listen ? <div className="small muted">listens on <span className="mono">{gateway.listen}</span></div> : null}
      {gateway.channels.length ? (
        <Table heads={[["channel"], ["serving"], ["tok/s", "n"], ["at once", "n"]]} keys={gateway.channels.map(channel => channel.channel)}
          rows={gateway.channels.map(channel => [channel.channel, channel.serving ? known.short(channel.serving) : channel.servers ? plural(channel.servers.length, "server") : "base model", ...throughput(channel)])} />
      ) : null}
    </RoleCard>
  );
});

/** Each role of one kind as its card. */
export function RoleCards({ kind, roles, hosts, onHost }: {
  kind: RoleKind;
  roles: (RunnerRole | PoolRole | EngineRole | GatewayRole)[];
  hosts: Map<string, Host>;
  onHost: boolean;
}) {
  const hostOf = (role: Role) => (role.host ? hosts.get(role.host) : undefined);
  return (
    <div className="cols">
      {roles.map(role => {
        if (kind === "runners") return <RunnerCard key={role.name} runner={role as RunnerRole} host={hostOf(role)} onHost={onHost} />;
        if (kind === "pools") return <PoolCard key={role.name} pool={role as PoolRole} onHost={onHost} />;
        if (kind === "engines") return <EngineCard key={role.name} engines={role as EngineRole} host={hostOf(role)} onHost={onHost} />;
        return <GatewayCard key={role.name} gateway={role as GatewayRole} onHost={onHost} />;
      })}
    </div>
  );
}
