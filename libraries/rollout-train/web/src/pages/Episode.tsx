// An episode: what it reported, then its rollouts, one per agent (each becomes a trajectory to train on), turn by
// turn or as a whole trajectory. New lines are added as the episode plays; what one opened, scrolled or chose stays.

import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useEpisode, useSystem } from "../api/queries";
import type { Episode as EpisodeData, Line, PlainMessage, SampleLine } from "../api/types";
import { Avatar, Card, Empty, Head, Kpi, Kpis, SectionTitle, Spec, Specs, Table } from "../components/ui";
import { byNumber, figure, plural, tokens } from "../lib/format";
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

/** Each agent's turns, in order, its slots in their numbers' order. An agent offered actions on its turns that samples
 * with none offered (summarising its memory, say) takes no turn then: such samples are folded under its turn before
 * (or its first), in `beside`. An agent never offered an action (a task that is all words) has only turns. */
export function samplesOf(lines: Line[]): { slots: Map<string, SampleLine[]>; beside: Map<SampleLine, SampleLine[]>; folded: number } {
  const every = new Map<string, SampleLine[]>();
  for (const line of lines) {
    if (line.kind !== "sample") continue;
    if (!every.has(line.slot)) every.set(line.slot, []);
    every.get(line.slot)!.push(line);
  }
  const slots = new Map<string, SampleLine[]>(), beside = new Map<SampleLine, SampleLine[]>();
  let folded = 0;
  for (const [slot, samples] of every) {
    if (!samples.some(sample => sample.tools.length)) { slots.set(slot, samples); continue; }
    const turns: SampleLine[] = [], early: SampleLine[] = [];
    for (const sample of samples) {
      if (sample.tools.length) { turns.push(sample); continue; }
      const before = turns.at(-1);
      if (before) beside.set(before, [...(beside.get(before) ?? []), sample]); else early.push(sample);
      folded += 1;
    }
    if (early.length) beside.set(turns[0], [...early, ...(beside.get(turns[0]) ?? [])]);
    slots.set(slot, turns);
  }
  return { slots: new Map([...slots].sort(([a], [b]) => byNumber(a, b))), beside, folded };
}

function Rollouts({ episode, slot: asked }: { episode: EpisodeData; slot: string | null }) {
  const navigate = useNavigate();
  const { slots, beside, folded } = useMemo(() => samplesOf(episode.lines), [episode.lines]);
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
      <SectionTitle title="Rollouts" note={`${slots.size}, one per agent, each a trajectory to train on${folded ? ` · ${plural(folded, "sample")} offered no tools, folded under the turn before` : ""}`} />
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
      {whole ? <Timeline shown={shown} turns={turns} episode={episode} beside={beside} full={full} /> : (
        <div className="turns-frame">
          <div className="turns" style={{ gridTemplateColumns: `repeat(${shown.length}, minmax(300px, 1fr))` }}>
            {shown.map(([each, samples]) => (
              <TurnCard key={each} slot={each} sample={samples[turn]} next={samples[turn + 1]} beside={samples[turn] && beside.get(samples[turn])} source={episode.source} state={episode.state} full={full} />
            ))}
          </div>
        </div>
      )}
      <Effects lines={episode.lines} />
    </>
  );
}

type Call = SampleLine["reply"]["calls"][number];
type Came = { text: string; error: boolean };

const callText = (call: Call) =>
  `${call.name}(${Object.entries(call.arguments).map(([name, value]) => `${name}: ${JSON.stringify(value)}`).join(", ")})`;

const same = (a: PlainMessage | undefined, b: PlainMessage) => a?.role === b.role && a.text === b.text;

/** What came back from a turn: for a call, the result the next sample carries for it; for a reply without a call, the
 * words the next sample's context added after this one's own (and its reply), or, where its context was written anew
 * (summarised, say), those after its newest reply. */
export function resultOf(sample: SampleLine, next: SampleLine | undefined, call?: Call): Came | null {
  if (!next) return null;
  if (call) {
    for (const message of next.messages) for (const result of message.results) if (result.id === call.id) return result;
    return null;
  }
  const own = sample.messages.length;
  const extended = next.messages.length > own && (!own || same(next.messages[own - 1], sample.messages[own - 1]));
  const after = extended ? own : next.messages.findLastIndex(message => message.role === "assistant") + 1;
  const words = next.messages.slice(after).filter(message => message.role === "user" && message.text).map(message => message.text);
  return words.length ? { text: words.join("\n\n"), error: false } : null;
}

/** Each of a turn's calls with what came back from it, or the reply's answer where it called nothing. */
const cameOf = (sample: SampleLine, next: SampleLine | undefined): [Call | null, Came | null][] =>
  sample.reply.calls.length ? sample.reply.calls.map(call => [call, resultOf(sample, next, call)]) : [[null, resultOf(sample, next)]];

const Came = ({ came }: { came: Came }) => <div className={`result${came.error ? " error" : ""}`}>{came.text}</div>;

/** The samples an agent took after a turn with no tools offered (a summary of its memory, say), folded. */
function Beside({ samples }: { samples: SampleLine[] }) {
  return (
    <details>
      <summary>{plural(samples.length, "sample")} offered no tools</summary>
      {samples.map((each, index) => <p key={index} className="said">{each.reply.text.trim() || each.reply.reasoning.trim() || "nothing"}</p>)}
    </details>
  );
}

const TurnCard = memo(function TurnCard({ slot, sample, next, beside, source, state, full }: { slot: string; sample: SampleLine | undefined; next: SampleLine | undefined; beside: SampleLine[] | undefined; source: EpisodeData["source"]; state: string | null; full: boolean }) {
  const pre = useRef<HTMLDivElement>(null);
  // (a turn newly shown opens at the bottom of what the agent saw: its newest observation)
  useLayoutEffect(() => {
    const element = pre.current?.querySelector("pre");
    if (element) element.scrollTop = element.scrollHeight;
  }, [sample, full]);
  const header = <header><Avatar name={slot} /><b>{slot}</b><span>{sample ? `${figure(sample.seconds)} s · ${sample.finish_reason ?? ""}` : ""}</span></header>;
  if (!sample) return <div className="turn">{header}<section><span className="none">no turn yet</span></section></div>;
  const came = cameOf(sample, next), many = sample.reply.calls.length > 1;
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
        {came.some(([, each]) => each) ? came.map(([call, each], index) => (
          <div key={call?.id ?? index}>
            {many && call ? <div className="faint small">{call.name}</div> : null}
            {each ? <Came came={each} /> : <span className="none">nothing came back</span>}
          </div>
        )) : <span className="none">{source === "archive" ? "not kept" : state === "running" ? "pending" : "the episode ended"}</span>}
      </section>
      {beside ? <section><Beside samples={beside} /></section> : null}
    </div>
  );
});

const MAP = "Map of what you have seen";
/** The lines of a text that are a map's rows, each drawn with its symbols colored: those under a line that opens a map
 * as a Minecraft observation writes one (`MAP`), up to the map's key (`On the map:`) or the text's end. */
export function mapRows(text: string): boolean[] {
  let inside = false;
  return text.split("\n").map(line => {
    if (line.startsWith(MAP)) inside = true;
    else if (line.startsWith("On the map:")) inside = false;
    return inside && /^-?\d+( \S){6,}$/.test(line);
  });
}

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

/** What an agent was shown last: the messages after its newest reply (what came back from it, and anything said since),
 * the system's left out. */
export function seenOf(messages: PlainMessage[]): PlainMessage[] {
  const tail = messages.slice(messages.findLastIndex(message => message.role === "assistant") + 1).filter(message => message.role !== "system");
  return tail.length ? tail : messages.slice(-1);
}

const Seen = memo(function Seen({ sample, full }: { sample: SampleLine; full: boolean }) {
  const messages = full ? sample.messages : seenOf(sample.messages);
  return (
    <pre>
      {messages.map((message, index) => {
        const text = message.text || "", drawn = mapRows(text);
        return (
          <div key={index}>
            {full ? <div className="faint">— {message.role} —</div> : null}
            {text ? text.split("\n").map((line, place) => drawn[place] ? <MapRow key={place} text={line} /> : <div key={place}>{line || " "}</div>) : null}
            {full ? message.calls.map(call => <div key={`c${call.id}`}>→ {call.name}({JSON.stringify(call.arguments)})</div>) : null}
            {message.results.map(result => <div key={`r${result.id}`} className={result.error ? "error-text" : undefined}>← {result.text}</div>)}
          </div>
        );
      })}
    </pre>
  );
});

/** Every turn of the rollouts shown, one row a turn and one column a rollout: what each agent did and what came back,
 * with what it saw and what it thought one click away (drawn when opened; what is open stays open). */
function Timeline({ shown, turns, episode, beside, full }: { shown: [string, SampleLine[]][]; turns: number; episode: EpisodeData; beside: Map<SampleLine, SampleLine[]>; full: boolean }) {
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
      cells.push(<TimelineStep key={`${slot}${turn}`} sample={sample} next={samples[turn + 1]} beside={beside.get(sample)} at={`${episode.run_id} ${slot} ${turn}`} opened={opened} toggle={toggle} full={full} />);
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

const TimelineStep = memo(function TimelineStep({ sample, next, beside, at, opened, toggle, full }: { sample: SampleLine; next: SampleLine | undefined; beside: SampleLine[] | undefined; at: string; opened: Set<string>; toggle: (key: string, open: boolean) => void; full: boolean }) {
  const fold = (key: string, label: string, fill: () => React.ReactNode) => (
    <details open={opened.has(key)} onToggle={event => { const open = (event.target as HTMLDetailsElement).open; if (open !== opened.has(key)) toggle(key, open); }}>
      <summary>{label}</summary>
      {opened.has(key) ? fill() : null}
    </details>
  );
  const answer = sample.reply.calls.length ? null : resultOf(sample, next);
  return (
    <div className="step">
      {sample.reply.text.trim() ? <p className="said">{sample.reply.text.trim()}</p> : null}
      {sample.reply.calls.map(call => {
        const came = resultOf(sample, next, call);
        return <div key={call.id}><span className="call">{callText(call)}</span>{came ? <Came came={came} /> : null}</div>;
      })}
      {!sample.reply.calls.length && !sample.reply.text.trim() ? <span className="none">nothing</span> : null}
      {answer ? <Came came={answer} /> : null}
      <div className="folds">
        {sample.reply.reasoning ? fold(`${at} thought`, "thought", () => <p className="thought">{sample.reply.reasoning.trim()}</p>) : null}
        {sample.messages.length ? fold(`${at} saw`, "saw", () => <div className="sees"><Seen sample={sample} full={full} /></div>) : null}
        {beside ? fold(`${at} beside`, `${plural(beside.length, "sample")} offered no tools`, () => beside.map((each, index) => <p key={index} className="thought">{each.reply.text.trim() || each.reply.reasoning.trim() || "nothing"}</p>)) : null}
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
