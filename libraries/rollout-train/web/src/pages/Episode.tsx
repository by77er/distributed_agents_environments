// An episode: what it reported, then its rollouts, one per agent (each becomes a trajectory to train on), turn by
// turn or as a whole trajectory. New lines are added as the episode plays; what one opened, scrolled or chose stays.

import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useEpisode, useSystem } from "../api/queries";
import type { Episode as EpisodeData, Line, SampleLine } from "../api/types";
import { Avatar, Card, Empty, Head, Kpi, Kpis, SectionTitle, Spec, Specs, Table } from "../components/ui";
import { figure, tokens } from "../lib/format";
import { groupsOf, stateKind } from "../lib/model";
import { episodePlace } from "../lib/places";
import { useStored } from "../lib/stored";

export function Episode({ id, slot }: { id: string; slot: string | null }) {
  const { data: episode } = useEpisode(id);
  const { data: system } = useSystem();
  if (!episode || episode.run_id !== id) return <Empty>Reading the episode…</Empty>;
  const labels = episode.labels ?? {}, info = (episode.ended?.info ?? {}) as Record<string, unknown>;
  // An episode of a training run is of a group, whose record says its task; another says it in its labels, if at all.
  const run = labels.run ? system?.runs.find(each => each.run === labels.run) : undefined;
  const group = run && labels.group ? groupsOf(run).get(Number(labels.group)) : undefined;
  const task = group?.task ?? labels.task, title = group?.title ?? labels.title;
  return (
    <>
      <Head title={labels.episode ? `Episode ${labels.episode}` : episode.run_id.slice(-8)}>
        <Specs>
          <Spec label="state" kind={stateKind(episode.state)}>{episode.state ?? "running"}</Spec>
          {episode.ended ? <Spec label="reward" kind={episode.ended.solved ? "good" : ""}>{figure(episode.ended.reward)}</Spec> : null}
          {task ? <Spec label="task">{title ? `${task} · ${title}` : task}</Spec> : null}
          <Spec label="run">{episode.run_id}</Spec>
          {episode.source === "archive" ? <Spec label="read from" kind="warm">its kept events: replies only</Spec> : null}
        </Specs>
      </Head>
      <Kpis>
        {["reward", "turns", "duration", "ended"].filter(key => info[key] !== undefined).map(key => <Kpi key={key} label={key} value={figure(info[key])} />)}
        {episode.ended?.sampled ? <Kpi label="tokens sampled" value={tokens(episode.ended.sampled)} /> : null}
      </Kpis>
      <Reported info={info} />
      <Rollouts episode={episode} slot={slot} />
    </>
  );
}

const Reported = memo(function Reported({ info }: { info: Record<string, unknown> }) {
  if (!Object.keys(info).length) return null;
  const scalars = Object.entries(info).filter(([key, value]) => (value === null || typeof value !== "object") && !["reward", "turns", "duration", "ended"].includes(key));
  const nested = Object.entries(info).filter(([, value]) => value !== null && typeof value === "object") as [string, Record<string, unknown> | unknown[]][];
  const shown = (each: unknown) => (typeof each === "object" ? JSON.stringify(each) : figure(each));
  return (
    <Card title="What the episode reported" note="the program's result">
      <div className="cols" style={{ gap: 16 }}>
        <dl className="pairs">{scalars.map(([key, value]) => [<dt key={`k${key}`}>{key.replaceAll("_", " ")}</dt>, <dd key={`v${key}`}>{figure(value)}</dd>])}</dl>
        <div style={{ display: "grid", gap: 12 }}>
          {nested.map(([key, value]) => (
            <div key={key}>
              <div className="small muted" style={{ marginBottom: 4 }}>{key.replaceAll("_", " ")}</div>
              {Object.keys(value).length === 0 ? <span className="none">none</span> : (
                <div className="chips">
                  {Array.isArray(value) ? value.map((each, index) => <span key={index} className="chip">{typeof each === "object" ? JSON.stringify(each) : String(each)}</span>)
                    : Object.entries(value).sort(([, a], [, b]) => (typeof b === "number" ? b : 0) - (typeof a === "number" ? a : 0))
                      .map(([name, each]) => <span key={name} className="chip">{name} <b>{shown(each)}</b></span>)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
});

/** Each agent's samples, in order. An agent offered actions on its turns that summarises its memory is offered none
 * then: those samples are counted apart. An agent never offered an action (a task that is all words) has only turns. */
export function samplesOf(lines: Line[]): { slots: Map<string, SampleLine[]>; summaries: number } {
  const every = new Map<string, SampleLine[]>();
  for (const line of lines) {
    if (line.kind !== "sample") continue;
    if (!every.has(line.slot)) every.set(line.slot, []);
    every.get(line.slot)!.push(line);
  }
  const slots = new Map<string, SampleLine[]>();
  let summaries = 0;
  for (const [slot, samples] of every) {
    const acting = samples.some(sample => sample.tools.length);
    const turns = acting ? samples.filter(sample => sample.tools.length) : samples;
    summaries += samples.length - turns.length;
    slots.set(slot, turns);
  }
  return { slots: new Map([...slots].sort(([a], [b]) => a.localeCompare(b))), summaries };
}

function Rollouts({ episode, slot: asked }: { episode: EpisodeData; slot: string | null }) {
  const navigate = useNavigate();
  const { slots, summaries } = useMemo(() => samplesOf(episode.lines), [episode.lines]);
  const [chosen, setTurn] = useState(0);
  const [follow, setFollow] = useState(true);
  const [whole, setWhole] = useStored("monitor.whole", false);
  const [full, setFull] = useState(false);
  if (!slots.size) {
    return <Empty>{episode.source ? "No agent has taken a turn yet." : "Neither the feed nor the ledger has this episode's rollouts."}</Empty>;
  }
  const slot = asked && slots.has(asked) ? asked : null;
  const shown: [string, SampleLine[]][] = slot ? [[slot, slots.get(slot)!]] : [...slots];
  const turns = Math.max(...shown.map(([, samples]) => samples.length));
  const turn = Math.max(0, Math.min(follow ? turns - 1 : chosen, turns - 1));
  const move = (delta: number) => { setTurn(Math.max(0, Math.min(turns - 1, turn + delta))); setFollow(false); };
  return (
    <>
      <SectionTitle title="Rollouts" note={`${slots.size}, one per agent, each a trajectory to train on${summaries ? ` · ${summaries} memory summaries written` : ""}`} />
      <div className="segmented">
        <button type="button" className={`seg${slot ? "" : " current"}`} onClick={() => navigate(episodePlace(episode.run_id))}>All rollouts</button>
        {[...slots.keys()].map(each => (
          <button type="button" key={each} className={`seg${slot === each ? " current" : ""}`} onClick={() => navigate(episodePlace(episode.run_id, each))}><Avatar name={each} />{each}</button>
        ))}
      </div>
      <div className="scrub">
        <div className="segmented">
          <button type="button" className={`seg${whole ? "" : " current"}`} onClick={() => setWhole(false)}>Turn by turn</button>
          <button type="button" className={`seg${whole ? " current" : ""}`} onClick={() => setWhole(true)}>Whole trajectory</button>
        </div>
        {whole ? <output>{turns} turns</output> : (
          <>
            <button type="button" onClick={() => move(-1)} aria-label="previous turn">‹</button>
            <button type="button" onClick={() => move(1)} aria-label="next turn">›</button>
            <output>turn {turn + 1} of {turns}</output>
            <input type="range" min={0} max={turns - 1} value={turn} aria-label="turn" onChange={event => { setTurn(Number(event.target.value)); setFollow(false); }} />
            <label><input type="checkbox" checked={follow} onChange={event => { setFollow(event.target.checked); if (!event.target.checked) setTurn(turn); }} />follow</label>
            {episode.source === "feed" ? <label><input type="checkbox" checked={full} onChange={event => setFull(event.target.checked)} />whole context</label> : null}
          </>
        )}
      </div>
      {whole ? <Timeline shown={shown} turns={turns} episode={episode} full={full} /> : (
        <div className="turns-frame">
          <div className="turns" style={{ gridTemplateColumns: `repeat(${shown.length}, minmax(300px, 1fr))` }}>
            {shown.map(([each, samples]) => (
              <TurnCard key={each} slot={each} sample={samples[turn]} next={samples[turn + 1]} source={episode.source} state={episode.state} full={full} />
            ))}
          </div>
        </div>
      )}
      <Effects lines={episode.lines} />
    </>
  );
}

const callText = (call: SampleLine["reply"]["calls"][number]) =>
  `${call.name}(${Object.entries(call.arguments).map(([name, value]) => `${name}: ${JSON.stringify(value)}`).join(", ")})`;

/** What came back from a turn's call: the next sample's result for it, or the words a reply without a call was
 * answered in. */
function resultOf(next: SampleLine | undefined, callId: string | undefined): { text: string; error: boolean } | null {
  if (!next) return null;
  for (const message of next.messages) for (const result of message.results) if (result.id === callId) return result;
  const last = next.messages.filter(message => message.role === "user").at(-2);
  return last ? { text: last.text, error: false } : null;
}

const TurnCard = memo(function TurnCard({ slot, sample, next, source, state, full }: { slot: string; sample: SampleLine | undefined; next: SampleLine | undefined; source: EpisodeData["source"]; state: string | null; full: boolean }) {
  const pre = useRef<HTMLDivElement>(null);
  // (a turn newly shown opens at the bottom of what the agent saw: its newest observation)
  useLayoutEffect(() => {
    const element = pre.current?.querySelector("pre");
    if (element) element.scrollTop = element.scrollHeight;
  }, [sample, full]);
  const header = <header><Avatar name={slot} /><b>{slot}</b><span>{sample ? `${figure(sample.seconds)} s · ${sample.finish_reason ?? ""}` : ""}</span></header>;
  if (!sample) return <div className="turn">{header}<section><span className="none">no turn yet</span></section></div>;
  const result = resultOf(next, sample.reply.calls[0]?.id);
  return (
    <div className="turn">
      {header}
      <section className="sees" ref={pre}><h3>Sees</h3>{sample.messages.length ? <Seen sample={sample} full={full} /> : <span className="none">kept only as tokens: the feed has let this episode go</span>}</section>
      <section className="thinks"><h3>Thinks</h3>{sample.reply.reasoning ? <p>{sample.reply.reasoning.trim()}</p> : <span className="none">nothing recorded</span>}</section>
      <section>
        <h3>Does</h3>
        {sample.reply.text.trim() ? <p className="said">{sample.reply.text.trim()}</p> : null}
        {sample.reply.calls.map(call => <div key={call.id}><span className="call">{callText(call)}</span></div>)}
        {!sample.reply.calls.length && !sample.reply.text.trim() ? <span className="none">nothing</span> : null}
      </section>
      <section>
        <h3>Result</h3>
        {result ? <div className={`result${result.error ? " error" : ""}`}>{result.text}</div>
          : <span className="none">{source === "archive" ? "not kept" : state === "running" ? "pending" : "the episode ended"}</span>}
      </section>
    </div>
  );
});

/** A row of a map the agent was shown: each cell colored by what it is. */
function MapRow({ text }: { text: string }) {
  const [label, ...cells] = text.split(" ");
  return (
    <div className="row">
      {label.padStart(5)}{" "}
      {cells.map((cell, index) => {
        const kind = cell === "#" ? "solid" : cell === "." ? "air" : cell === "?" ? "unseen" : cell === "@" ? "self" : /[A-Z+]/.test(cell) ? "mate" : "thing";
        return <span key={index} className={`c-${kind}`}>{cell} </span>;
      })}
    </div>
  );
}

const Seen = memo(function Seen({ sample, full }: { sample: SampleLine; full: boolean }) {
  const messages = full ? sample.messages : sample.messages.filter(message => message.role === "user").slice(-1);
  return (
    <pre>
      {messages.map((message, index) => (
        <div key={index}>
          {full ? <div className="faint">— {message.role} —</div> : null}
          {(message.text || "").split("\n").map((line, place) => /^-?\d+( \S){6,}$/.test(line) ? <MapRow key={place} text={line} /> : <div key={place}>{line || " "}</div>)}
          {full ? message.calls.map(call => <div key={`c${call.id}`}>→ {call.name}({JSON.stringify(call.arguments)})</div>) : null}
          {full ? message.results.map(result => <div key={`r${result.id}`}>← {result.text}</div>) : null}
        </div>
      ))}
    </pre>
  );
});

/** Every turn of the rollouts shown, one row a turn and one column a rollout: what each agent did and what came back,
 * with what it saw and what it thought one click away (drawn when opened; what is open stays open). */
function Timeline({ shown, turns, episode, full }: { shown: [string, SampleLine[]][]; turns: number; episode: EpisodeData; full: boolean }) {
  const [opened, setOpened] = useState<Set<string>>(() => new Set());
  const toggle = (key: string, open: boolean) => setOpened(before => {
    const after = new Set(before);
    if (open) after.add(key); else after.delete(key);
    return after;
  });
  const cells = [];
  for (let turn = 0; turn < turns; turn += 1) {
    cells.push(<div key={`w${turn}`} className="when">{turn + 1}</div>);
    for (const [slot, samples] of shown) {
      const sample = samples[turn];
      if (!sample) { cells.push(<div key={`${slot}${turn}`} className="step"><span className="none">–</span></div>); continue; }
      cells.push(<TimelineStep key={`${slot}${turn}`} sample={sample} next={samples[turn + 1]} at={`${episode.run_id} ${slot} ${turn}`} opened={opened} toggle={toggle} full={full} />);
    }
  }
  return (
    <div className="turns-frame">
      <div className="timeline" style={{ gridTemplateColumns: `52px repeat(${shown.length}, minmax(300px, 1fr))` }}>
        <div className="when head" />
        {shown.map(([slot]) => <div key={slot} className="head"><Avatar name={slot} /><b>{slot}</b></div>)}
        {cells}
      </div>
    </div>
  );
}

const TimelineStep = memo(function TimelineStep({ sample, next, at, opened, toggle, full }: { sample: SampleLine; next: SampleLine | undefined; at: string; opened: Set<string>; toggle: (key: string, open: boolean) => void; full: boolean }) {
  const result = resultOf(next, sample.reply.calls[0]?.id);
  const fold = (key: string, label: string, fill: () => React.ReactNode) => (
    <details open={opened.has(key)} onToggle={event => { const open = (event.target as HTMLDetailsElement).open; if (open !== opened.has(key)) toggle(key, open); }}>
      <summary>{label}</summary>
      {opened.has(key) ? fill() : null}
    </details>
  );
  return (
    <div className="step">
      {sample.reply.text.trim() ? <p className="said">{sample.reply.text.trim()}</p> : null}
      {sample.reply.calls.map(call => <div key={call.id}><span className="call">{callText(call)}</span></div>)}
      {!sample.reply.calls.length && !sample.reply.text.trim() ? <span className="none">nothing</span> : null}
      {result ? <div className={`result${result.error ? " error" : ""}`}>{result.text}</div> : null}
      <div className="folds">
        {sample.reply.reasoning ? fold(`${at} thought`, "thought", () => <p className="thought">{sample.reply.reasoning.trim()}</p>) : null}
        {sample.messages.length ? fold(`${at} saw`, "saw", () => <div className="sees"><Seen sample={sample} full={full} /></div>) : null}
        <span className="faint small">{figure(sample.seconds)} s</span>
      </div>
    </div>
  );
});

/** The tool calls the program made itself (not the model's samples), newest first. */
const Effects = memo(function Effects({ lines }: { lines: Line[] }) {
  const effects = useMemo(() => {
    const requests = new Map<string, any>(), found: [any, any][] = [];
    for (const line of lines) {
      if (line.kind !== "event") continue;
      if (line.type === "effect.requested" && line.payload.kind !== "model.sample") requests.set(line.payload.effect_id, line.payload);
      if (line.type === "effect.completed" && requests.has(line.payload.effect_id)) found.push([requests.get(line.payload.effect_id), line.payload]);
    }
    return found;
  }, [lines]);
  if (!effects.length) return null;
  const rows = effects.slice(-60).reverse().map(([request, completed]) => [
    { text: String(request.payload?.tool ?? request.payload?.name ?? request.kind), kind: "key" },
    JSON.stringify(request.payload?.arguments ?? request.payload ?? {}).slice(0, 160),
    JSON.stringify(completed.payload?.structured ?? completed.payload ?? completed.status).slice(0, 240),
  ]);
  return (
    <details className="card" style={{ padding: "12px 18px" }}>
      <summary>Tool calls the program made ({effects.length}; newest first)</summary>
      <div style={{ marginTop: 10 }}><Table heads={[["tool"], ["arguments"], ["returned"]]} rows={rows} /></div>
    </details>
  );
});
