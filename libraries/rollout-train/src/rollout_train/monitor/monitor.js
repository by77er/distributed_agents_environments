"use strict";
// The monitor's page, in three: the runs, each organised as a run is (its steps, each one update of the policy over
// the groups it covers; each group, a task, a start and a number of episodes; each episode, one run of the program,
// and its rollouts, one per agent, each of which becomes a trajectory to train on); the policies, each alone and all
// of them as a graph; and statistics across every run, with the machine, the engines and the ledger.

const h = (tag, attributes = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) node.append(child);
  return node;
};
const svg = (tag, attributes = {}, ...children) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) if (value !== undefined) node.setAttribute(key, value);
  for (const child of children.flat()) if (child) node.append(child);
  return node;
};

const state = { system: null, runs: [], group: null, episode: null, statistics: null, turn: 0, follow: true, full: false, drawn: "" };

// Words and numbers
const span = seconds => seconds == null ? "" : seconds < 90 ? `${Math.round(seconds)} s`
  : seconds < 5400 ? `${Math.round(seconds / 60)} min` : `${(seconds / 3600).toFixed(1)} h`;
const clock = at => at ? new Date(at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
const bytes = count => count >= 2 ** 30 ? `${(count / 2 ** 30).toFixed(1)} GiB` : count >= 2 ** 20 ? `${Math.round(count / 2 ** 20)} MiB`
  : `${Math.round((count ?? 0) / 1024)} KiB`;
const figure = value => value == null ? "–" : typeof value === "boolean" ? (value ? "yes" : "no")
  : Number.isInteger(value) ? value.toLocaleString() : typeof value === "number" ? Number(value).toFixed(Math.abs(value) < 1 ? 4 : 1) : String(value);
const tokens = count => count >= 1e6 ? `${(count / 1e6).toFixed(1)} M` : count >= 1e3 ? `${Math.round(count / 1e3)} k` : String(count ?? 0);
const versionOf = name => name ? `@${name.split("@").at(-1)}` : "base";
const mean = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const hue = name => [...name].reduce((sum, letter) => (sum * 31 + letter.charCodeAt(0)) % 360, 7);
// An agent's badge: its slot's number (`agent-2` → 2) in a color of its own, or, for a slot with no number, its initial.
const avatar = name => {
  const number = name.match(/(\d+)$/)?.[1];
  const color = number ? (Number(number) * 97 + 160) % 360 : hue(name);
  return h("span", { class: "avatar", style: `background:hsl(${color} 55% 46%)` }, number ?? name.slice(0, 1));
};

// Places
const go = place => { location.hash = place; };
const link = (place, attributes, ...children) => h("a", { href: place, ...attributes }, ...children);
const runPlace = run => `#/run/${encodeURIComponent(run)}`;
const groupPlace = (run, number) => `${runPlace(run)}/group/${number}`;
const stepPlace = (run, number) => `${runPlace(run)}/step/${number}`;
const episodePlace = (id, slot) => `#/episode/${encodeURIComponent(id)}${slot ? `/${encodeURIComponent(slot)}` : ""}`;
const policyPlace = name => `#/policy/${encodeURIComponent(name)}`;
// The three pages, each with its sidebar; every place is on one of them.
const PAGES = [["runs", "Runs", "#/runs"], ["policies", "Policies", "#/policies"], ["statistics", "Statistics", "#/statistics"]];
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  const at = (page, place) => ({ page, ...place });
  if (parts[0] === "run" && parts[2] === "group") return at("runs", { kind: "group", run: parts[1], number: Number(parts[3]) });
  if (parts[0] === "run" && parts[2] === "step") return at("runs", { kind: "step", run: parts[1], number: Number(parts[3]) });
  if (parts[0] === "run") return at("runs", { kind: "run", run: parts[1] });
  if (parts[0] === "episode") return at("runs", { kind: "episode", id: parts[1], slot: parts[2] || null });
  if (parts[0] === "episodes") return at("runs", { kind: "outside" });
  if (parts[0] === "policies") return at("policies", { kind: "policies", sample: parts[1] === "sample" });
  if (parts[0] === "policy") return at("policies", { kind: "policy", name: parts[1] });
  if (parts[0] === "statistics") return at("statistics", { kind: "statistics", section: parts[1] || null });
  if (parts[0] === "system") return at("statistics", { kind: "statistics", section: "machine" });  // (the machine is a section of the statistics)
  return at("runs", { kind: "runs" });
}

// Shared pieces
// A fact about the thing shown, as a datasheet gives it: a label and a value, set side by side under the title.
const spec = (key, value, kind = "") => h("span", { class: `spec ${kind}` }, h("span", { class: "k" }, key), h("span", { class: "v" }, value));
const specs = (...items) => h("div", { class: "specs" }, items.filter(Boolean));
// A state: a square in its color, then its name.
const mark = (state, text) => h("span", { class: `mark ${stateKind(state)}` }, h("i"), text ?? state);
const kpi = (label, value, note) => h("div", { class: "kpi" }, h("span", {}, label), h("b", { title: typeof value === "string" ? value : null }, value), note ? h("small", {}, note) : null);
const card = (title, note, ...body) => h("section", { class: "card" }, h("header", {}, h("h2", {}, title), note ? h("span", {}, note) : null), h("div", { class: "body" }, ...body));
const table = (heads, rows, onRow) => h("div", { class: "table" }, h("table", {},
  h("tr", {}, heads.map(([name, kind]) => h("th", kind ? { class: kind } : {}, name))),
  rows.map((row, index) => h("tr", onRow?.[index] ? { class: "link", onclick: onRow[index] } : {},
    row.map((cell, column) => h("td", { class: [heads[column][1], cell?.kind].filter(Boolean).join(" ") || null },
      cell?.node ?? cell?.text ?? (cell === undefined || cell === null ? "" : String(cell))))))));
const meter = (name, used, total, says) => {
  const share = total ? used / total : 0;
  return h("div", { class: "meter" }, h("div", {}, h("span", {}, name), h("b", {}, says)),
    h("div", { class: `track${share > 0.93 ? " bad" : share > 0.85 ? " warm" : ""}` }, h("i", { style: `width:${(100 * share).toFixed(1)}%` })));
};
const pairs = entries => h("dl", { class: "pairs" }, entries.flatMap(([key, value]) => [h("dt", {}, key), h("dd", {}, value)]));

// A group's stages, up to its result; what is trained on it is its step's (a step is shown as a thing of its own).
const STAGES = ["asked", "claimed", "played", "recorded"];
const AT = { waiting: [1, "to claim"], playing: [2, ""], ended: [3, "recording"], done: [4, ""] };
function stages(group) {
  const [at, waits] = AT[group.stage];
  const says = group.stage === "playing" ? `playing ${group.ended}/${group.count}` : waits;
  return h("div", { class: "stages" }, STAGES.map((name, index) => h("div", {
    class: index < at ? "done" : index === at ? `now${group.stage === "playing" ? " active" : ""}` : "",
  }, h("span", {}, index === at ? says : name))));
}
const outcomeOf = line => line.update
  ? { kind: "moved", text: `step ${line.step} · trained ${line.segments_trained} of ${line.segments_recorded} · moved ${Number(line.update.kl_moved ?? 0).toFixed(4)}` }
  : line.error ? { kind: "bad", text: `step ${line.step} failed: ${line.error}` }
  : line.step_state === "stepping" ? { kind: "violet", text: `in step ${line.step}, being taken` }
  : line.segments ? { kind: "warm", text: "waits for the next step" }
  : { kind: line.failed && !line.rewards.length ? "bad" : "still", text: line.skipped ?? "" };
const dotsOf = line => h("span", { class: "dots" }, [...line.rewards.map((_, index) => h("i", { class: line.solved[index] ? "solved" : "unsolved" })),
  ...Array.from({ length: line.failed }, () => h("i", { class: "failed" }))]);
// A run's groups by number (those in flight and those done with), and the step a group went into.
const groupsOf = run => new Map([...run.done.map(line => [line.group, { number: line.group, task: line.task, title: line.title, line, episodes: line.episodes ?? [] }]),
  ...run.open.map(group => [group.number, { number: group.number, task: group.task, title: group.title, open: group, episodes: group.episodes }])]);
const stepOf = (run, number) => run.steps.find(step => step.groups.includes(number) || step.skipped.includes(number));
const versionNamed = name => state.system.policies.flatMap(policy => policy.versions).find(version => version.name === name);
const range = numbers => numbers.length ? (numbers.length > 1 && numbers.at(-1) - numbers[0] === numbers.length - 1
  ? `#${numbers[0]}–${numbers.at(-1)}` : numbers.map(number => `#${number}`).join(" ")) : "no group";
const stateKind = name => ({ running: "accent", completed: "good", done: "good", failed: "bad", cancelled: "bad", playing: "accent", queued: "warm", stepping: "violet", committed: "good" })[name] ?? "";

// A group's episodes, and a place held for each one asked for that has not started (they start as there is room,
// whatever group they are of).
function asked(group) {
  const have = group.episodes.filter(each => !each.interrupted), numbers = new Set(have.map(each => String(each.episode)));
  const waiting = [];
  for (let number = 1; waiting.length < (group.count ?? 0) - have.length; number++) {
    if (!numbers.has(String(number))) waiting.push({ episode: String(number), waiting: true });
  }
  return [...group.episodes, ...waiting];
}

// A group's episodes as a strip of cells: each with its state as a bar along its top, and its reward (or, while
// it plays, how many samples its agents have taken).
const cells = episodes => h("div", { class: "cells" }, episodes.map(episode => episode.waiting
  ? h("div", { class: "cell waiting" }, h("span", {}, `E${episode.episode}`), h("b", { class: "faint" }, "–"), h("small", {}, "not started"))
  : h("div", { class: `cell ${episode.interrupted ? "" : stateKind(episode.state)}` },
    h("span", {}, `E${episode.episode ?? "?"}`), episode.outcome ? h("b", {}, figure(episode.reward)) : h("b", { class: "faint" }, `${episode.samples ?? 0}`),
    h("small", {}, episode.interrupted ? "interrupted" : episode.outcome ? (episode.solved ? "solved" : episode.outcome) : "samples"))));

// Charts
function spark(values, kind, width, height, fill) {
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" });
  if (!values.length) return drawing;
  const top = Math.max(...values, 1e-9), step = (width - 6) / Math.max(1, values.length - 1);
  const points = values.map((value, index) => [3 + index * step, height - 3 - (value / top) * (height - 10)]);
  drawing.append(svg("line", { x1: 0, x2: width, y1: height - 2.5, y2: height - 2.5, class: "s-grid" }));
  if (fill) drawing.append(svg("polygon", { class: `${kind.replace("s-", "f-")}-soft`, points: [`3,${height - 3}`, ...points.map(point => point.join(",")), `${points.at(-1)[0]},${height - 3}`].join(" ") }));
  if (values.length > 1) drawing.append(svg("polyline", { class: kind, points: points.map(point => point.join(",")).join(" ") }));
  const [x, y] = points.at(-1);
  drawing.append(svg("circle", { cx: x, cy: y, r: 3, class: kind.replace("s-", "f-") }));
  return drawing;
}

// Every group the run is done with: each episode's reward as a dot, the group's mean as a rule, and under the axis
// what was done with the group; groups stand in the order of their steps, and a rule parts one step's from the next.
// A column opens its group.
function rewardsChart(run, width) {
  // (in the order of the steps they went into, and within a step by number: a group may finish, and be trained on,
  // before one decided earlier; those toward the next step come last)
  const order = line => { const step = stepOf(run, line.group); return step ? run.steps.indexOf(step) : run.steps.length; };
  const lines = [...run.done].sort((a, b) => order(a) - order(b) || a.group - b.group), left = 32, top = 10, plot = 120, band = top + plot + 10, height = band + 28;
  const count = Math.max(lines.length, 12), column = (width - left - 6) / count;
  const most = Math.max(1, ...lines.flatMap(line => line.rewards));
  const power = 10 ** Math.floor(Math.log10(most)), ceiling = Math.ceil(most / power) * power;
  const y = value => top + plot - (value / ceiling) * plot;
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "rewards by group" });
  for (const value of [0, ceiling / 2, ceiling]) {
    drawing.append(svg("line", { x1: left, x2: width, y1: y(value), y2: y(value), class: "s-grid" }));
    drawing.append(svg("text", { x: left - 8, y: y(value) + 3, "text-anchor": "end" }, String(+value.toFixed(1))));
  }
  const every = Math.max(1, Math.ceil(34 / column)), mark = Math.max(3, Math.min(9, column - 3)), spread = Math.min(3.4, column / 6);
  lines.forEach((line, index) => {
    const x = left + (index + 0.5) * column;
    drawing.append(svg("rect", { x: x - column / 2, y: top - 4, width: column, height: height - top, class: "f-none column",
      onclick: () => go(groupPlace(run.run, line.group)) },
    svg("title", {}, `#${line.group} ${line.task} · ${line.title}\n${line.rewards.map(figure).join(" ") || "no episode"}${line.failed ? ` (${line.failed} failed)` : ""}\n${outcomeOf(line).text}`)));
    drawing.lastChild.addEventListener("click", () => go(groupPlace(run.run, line.group)));
    const step = stepOf(run, line.group), before = index ? stepOf(run, lines[index - 1].group) : step;
    if (step !== before) drawing.append(svg("line", { x1: x - column / 2, x2: x - column / 2, y1: top - 4, y2: height, class: "s-grid", "pointer-events": "none" }));
    line.rewards.forEach((value, place) => drawing.append(svg("circle", { cx: x + (place - (line.rewards.length - 1) / 2) * spread, cy: y(value),
      r: Math.min(3.2, Math.max(1.6, column / 5)), class: line.solved[place] ? "f-good" : "f-quiet", "pointer-events": "none" })));
    if (line.rewards.length) drawing.append(svg("line", { x1: x - column * 0.34, x2: x + column * 0.34, y1: y(mean(line.rewards)), y2: y(mean(line.rewards)), class: "s-ink", "pointer-events": "none" }));
    const did = line.update ? "f-accent" : line.error || (line.failed && !line.rewards.length) ? "f-bad"
      : line.step_state === "stepping" ? "f-violet" : line.segments ? "f-warm" : "f-hollow";
    drawing.append(svg("rect", { x: x - mark / 2, y: band, width: mark, height: mark, rx: 2, class: did, "pointer-events": "none" }));
    if (index % every === 0) drawing.append(svg("text", { x, y: band + 23, "text-anchor": "middle" }, String(line.group)));
  });
  return drawing;
}

function barChart(values, labels, width, height, onBar) {
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" });
  const top = Math.max(...values, 1e-9), column = width / Math.max(values.length, 12), base = height - 16;
  drawing.append(svg("line", { x1: 0, x2: width, y1: base + 0.5, y2: base + 0.5, class: "s-grid" }));
  values.forEach((value, index) => {
    const tall = Math.max(1, (value / top) * (base - 6));
    const bar = svg("rect", { x: index * column + column * 0.18, y: base - tall, width: column * 0.64, height: tall, rx: 3, class: "f-violet" },
      svg("title", {}, `${labels[index]}: ${figure(value)}`));
    if (onBar) { bar.style.cursor = "pointer"; bar.addEventListener("click", () => onBar(index)); }
    drawing.append(bar);
    if (values.length <= 24 || index % Math.ceil(values.length / 24) === 0) drawing.append(svg("text", { x: index * column + column / 2, y: height - 3, "text-anchor": "middle" }, labels[index]));
  });
  return drawing;
}

// The hierarchy, on the left. Runs and groups fold; what is folded is remembered in this browser.
const folds = (() => { try { return JSON.parse(localStorage.getItem("monitor.folds") ?? "{}"); } catch { return {}; } })();
function fold(key, open) {
  folds[key] = open;
  try { localStorage.setItem("monitor.folds", JSON.stringify(folds)); } catch { /* (a browser that keeps nothing) */ }
  drawTree();
}
const twist = (key, open, has = true) => h("button", {
  class: `twist${open ? " open" : ""}${has ? "" : " none"}`, "aria-label": open ? "collapse" : "expand", "aria-expanded": String(open),
  onclick: event => { event.stopPropagation(); fold(key, !open); },
}, svg("svg", { width: 10, height: 10, viewBox: "0 0 10 10" }, svg("path", { d: "M3 1.5 L7 5 L3 8.5", fill: "none", stroke: "currentColor", "stroke-width": 1.6 })));
const node = (place, current, ...children) => h("div", { class: `node${current ? " current" : ""}`, role: "link", tabindex: 0,
  onclick: () => go(place), onkeydown: event => { if (event.key === "Enter") go(place); } }, ...children);
// A node that says what it is, for its layout: a step, a group, an episode or a rollout.
const kindOf = (kind, element) => { element.classList.add(kind); return element; };
function episodeClass(each) {
  if (each.waiting) return "waiting";
  if (each.interrupted) return "";
  const ended = each.outcome ?? (each.state && each.state !== "running" ? each.state : null);
  return !ended ? "running" : each.solved ? "solved" : ended === "completed" ? "unsolved" : "failed";
}

// Whether a run is running (its process is there and writes), idle (there, and quiet) or ended, and on which host
// when not this one; and when it last wrote.
const running = run => `${run.state}${run.host && run.host !== state.system.host ? ` on ${run.host}` : ""}`;
const wrote = run => run.written ? `wrote ${span(Math.max(0, state.system.at - run.written))} ago` : "wrote nothing yet";
const runDot = run => h("span", { class: `dot ${run.state === "running" ? "alive" : run.state === "idle" ? "idle" : ""}`, title: running(run) });
// Where a run's episodes are read: its directory on this machine (nothing to say), the monitor on its own machine
// (said, with a link, and whether it answers), or nowhere (what is shown is the ledger's).
function elsewhere(run) {
  if (run.episodes_at === "here") return null;
  const there = run.episodes_at, address = there ? link(there, { class: "linkish", target: "_blank", rel: "noopener" }, there) : null;
  if (there && run.reached !== false) return h("div", { class: "tile rail accent notice" }, h("header", {}, h("b", {}, "Episodes from the monitor on its machine"), h("span", { class: "what" }, run.host ?? "")),
    h("p", { class: "muted small" }, "Its directory is not on this machine: its groups in flight, its episodes and its engines are asked of ", address, "; the rest is the ledger's."));
  return h("div", { class: "tile rail warm notice" }, h("header", {}, h("b", {}, "Read from the ledger alone"), h("span", { class: "what" }, run.directory ?? "")),
    h("p", { class: "muted small" }, there ? ["Its directory is not on this machine, and the monitor on its machine (", address, ") does not answer: its episodes are there."]
      : `Its directory is not on this machine${run.host ? ` (it was started on ${run.host})` : ""}, and its start names no monitor to ask: its episodes are not shown.`));
}

function drawTree() {
  const system = state.system, here = route(), tree = document.getElementById("tree");
  if (!system) return;
  document.getElementById("where").textContent = system.ledger_at;
  const nodes = here.page === "policies" ? policiesTree(here) : here.page === "statistics" ? statisticsTree(here) : runsTree(here);
  const scroll = tree.scrollTop;
  tree.replaceChildren(...nodes);
  tree.scrollTop = scroll;
}

// The runs: every run of the ledger, and under each its hierarchy: the groups toward its next step, then every step,
// newest first, each with the groups that went into it, their episodes and the episodes' rollouts.
function runsTree(here) {
  const system = state.system, showing = here.kind === "episode" ? state.episode?.labels : null;
  const nodes = [link("#/runs", { class: `label${here.kind === "runs" ? " here" : ""}` }, `Runs · ${system.runs.length}`)];
  for (const run of system.runs) {
    const runKey = `run:${run.run}`, mine = here.run === run.run || showing?.run === run.run;
    const runOpen = folds[runKey] ?? (mine || system.runs.length === 1);
    nodes.push(node(runPlace(run.run), here.kind === "run" && here.run === run.run, twist(runKey, runOpen),
      runDot(run), h("span", { class: "name" }, run.run),
      h("span", { class: "tag" }, running(run))));
    if (!runOpen) continue;
    const groups = groupsOf(run), children = [];
    const inGroup = number => (here.kind === "group" && here.run === run.run && here.number === number)
      || (showing?.run === run.run && Number(showing?.group) === number);
    const groupRows = (number, skipped) => {
      const group = groups.get(number);
      if (!group) return [];
      const key = `group:${run.run}:${number}`, open = folds[key] ?? inGroup(number);
      const tag = group.open ? group.open.stage : skipped ? "skipped" : group.line.rewards.length
        ? `${group.line.solved.filter(Boolean).length}/${group.line.rewards.length}` : "failed";
      const rows = [kindOf("group", node(groupPlace(run.run, number), here.kind === "group" && here.number === number,
        twist(key, open, group.episodes.length > 0), h("span", { class: "num" }, `#${number}`), h("span", { class: "name" }, group.task),
        group.line && !group.episodes.length ? dotsOf(group.line)
          : h("span", { class: "dots" }, (group.open ? asked(group.open) : group.episodes).map(each => h("i", { class: episodeClass(each) }))),
        h("span", { class: "tag" }, tag)))];
      const byNumber = [...group.episodes].sort((a, b) => String(a.episode).localeCompare(String(b.episode), undefined, { numeric: true }));
      if (open && byNumber.length) rows.push(h("div", { class: "children" }, byNumber.flatMap(each => {
        const episodeKey = `episode:${each.run_id}`, slots = each.slots ?? [];
        const episodeOpen = folds[episodeKey] ?? (here.kind === "episode" && here.id === each.run_id && Boolean(here.slot));
        const row = kindOf("episode", node(episodePlace(each.run_id), here.kind === "episode" && here.id === each.run_id && !here.slot,
          twist(episodeKey, episodeOpen, slots.length > 0), h("span", { class: "num" }, `E${each.episode ?? "?"}`),
          h("span", { class: "dots" }, h("i", { class: episodeClass(each) })),
          h("span", { class: "name" }, each.interrupted ? "interrupted" : each.outcome ?? (each.state === "running" ? `${each.samples ?? 0} samples` : each.state)),
          h("span", { class: "tag" }, each.outcome || each.state === "completed" ? figure(each.reward) : "")));
        if (!episodeOpen || !slots.length) return [row];
        return [row, h("div", { class: "children" }, slots.map(slot => kindOf("rollout", node(episodePlace(each.run_id, slot),
          here.kind === "episode" && here.id === each.run_id && here.slot === slot, avatar(slot), h("span", { class: "name" }, `rollout ${slot}`)))))];
      })));
      return rows;
    };
    // The groups the next step will cover, then every step, newest first, each with the groups that went into it.
    if (run.next.length) {
      const key = `next:${run.run}`, open = folds[key] ?? true;
      children.push(kindOf("step", node(runPlace(run.run), false, twist(key, open), h("span", { class: "num" }, "next"), h("span", { class: "name" }, "toward a step"),
        h("span", { class: "tag" }, `${run.next.length} group${run.next.length === 1 ? "" : "s"}`))));
      if (open) children.push(h("div", { class: "children" }, [...run.next].reverse().flatMap(number => groupRows(number, false))));
    }
    const newest = run.steps.at(-1)?.step;
    for (const step of [...run.steps].reverse()) {
      const key = `step:${run.run}:${step.step}`, members = [...step.groups, ...step.skipped];
      const open = folds[key] ?? (step.step === newest || members.some(inGroup) || (here.kind === "step" && here.run === run.run && here.number === step.step));
      children.push(kindOf("step", node(stepPlace(run.run, step.step), here.kind === "step" && here.run === run.run && here.number === step.step,
        twist(key, open, members.length > 0), h("span", { class: "num" }, `S${step.step}`),
        h("span", { class: "name" }, step.state === "committed" ? versionOf(step.makes) : `${versionOf(step.makes)} ${step.state}`),
        h("span", { class: "tag" }, range(step.groups)))));
      if (open) children.push(h("div", { class: "children" }, [...step.groups].reverse().flatMap(number => groupRows(number, false)),
        [...step.skipped].reverse().flatMap(number => groupRows(number, true))));
    }
    nodes.push(h("div", { class: "children" }, children));
  }
  const others = state.runs.filter(run => !run.labels.run);
  nodes.push(h("div", { class: "label" }, "Outside a run"));
  nodes.push(node("#/episodes", here.kind === "outside", h("span", { class: "name" }, "Episodes outside a run"), h("span", { class: "tag" }, String(others.length))));
  return nodes;
}

// The policies: the graph of them all, with or without the sample fixture, then each policy.
function policiesTree(here) {
  const system = state.system;
  return [link(policiesPlace(false), { class: `label${here.kind === "policies" && !here.sample ? " here" : ""}` }, `Policies · ${system.policies.length}`),
    ...system.policies.map(policy => node(policyPlace(policy.policy), here.kind === "policy" && here.name === policy.policy,
      h("span", { class: "name" }, policy.policy), h("span", { class: "tag" }, `${versionOf(policy.head)} · ${policy.versions.length}`))),
    system.policies.length ? null : h("div", { class: "empty" }, "No policy yet."),
    h("div", { class: "label" }, "Sample"),
    node(policiesPlace(true), here.kind === "policies" && here.sample, h("span", { class: "name" }, "Sample fixture"), h("span", { class: "chip sample" }, "sample"))].filter(Boolean);
}

// The statistics: its sections, and the runs drawn (each in its color; a click leaves it out or takes it back).
const SECTIONS = [["outcomes", "Outcomes"], ["rows", "Rows"], ["steps", "Steps"], ["pace", "Pace"], ["queue", "Queue"], ["inference", "Inference"], ["machine", "Machine"]];
const sectionName = key => SECTIONS.find(([each]) => each === key)[1];
function statisticsTree(here) {
  const runs = state.system.runs;
  return [h("div", { class: "label" }, "Sections"),
    ...SECTIONS.map(([key, name]) => node(`#/statistics/${key}`, here.section === key, h("span", { class: "name" }, name))),
    h("div", { class: "label" }, `Runs · ${runs.filter(run => !hidden.has(run.run)).length} of ${runs.length}`),
    ...runs.map(run => h("div", { class: `node${hidden.has(run.run) ? " off" : ""}`, role: "checkbox", tabindex: 0, "aria-checked": String(!hidden.has(run.run)),
      title: hidden.has(run.run) ? "draw this run" : "leave this run out", onclick: () => hide(run.run), onkeydown: event => { if (event.key === "Enter" || event.key === " ") hide(run.run); } },
    h("span", { class: "swatch", style: `background:${runColor(run.run)}` }), h("span", { class: "name" }, run.run), h("span", { class: "tag" }, running(run))))];
}
const hidden = new Set((() => { try { return JSON.parse(localStorage.getItem("monitor.hidden") ?? "[]"); } catch { return []; } })());
function hide(run) {
  if (hidden.has(run)) hidden.delete(run); else hidden.add(run);
  try { localStorage.setItem("monitor.hidden", JSON.stringify([...hidden])); } catch { /* (a browser that keeps nothing) */ }
  redraw();
}
// A run's color: the categorical slots in order, by the run's place among every run (so it keeps its color whatever
// is drawn); a run past the eighth is drawn in gray.
const runColor = name => {
  const place = (state.system?.runs ?? []).map(run => run.run).sort().indexOf(name);
  return place >= 0 && place < 8 ? `var(--series-${place + 1})` : "var(--faint)";
};

function drawBar(crumbs) {
  const system = state.system, here = route();
  const counts = { runs: system?.runs.length, policies: system?.policies.length, statistics: null };
  document.getElementById("pages").replaceChildren(...PAGES.map(([page, name, place]) => link(place, { class: here.page === page ? "current" : null, "aria-current": here.page === page ? "page" : null },
    name, counts[page] != null ? h("span", { class: "count" }, String(counts[page])) : null)));
  document.getElementById("crumbs").replaceChildren(...crumbs.flatMap((crumb, index) => [
    ...(index ? [h("span", { class: "sep" }, "/")] : []), index === crumbs.length - 1 ? h("b", {}, crumb[0]) : link(crumb[1], {}, crumb[0])]));
  const live = document.getElementById("live");
  if (!system) { live.replaceChildren("connecting…"); return; }
  const busy = system.runs.filter(run => run.state === "running").length;
  live.replaceChildren(h("span", { class: `dot ${busy ? "alive" : ""}` }), `${busy} of ${system.runs.length} run${system.runs.length === 1 ? "" : "s"} running`,
    system.written ? ` · wrote ${span(Math.max(0, system.at - system.written))} ago` : "");
}

// The training run
function drawRun(name) {
  const system = state.system, run = system.runs.find(each => each.run === name);
  if (!run) return [h("div", { class: "empty" }, `There is no run ${name}.`)];
  // (the policy its steps make, or its start names; its engines, as its feed has them)
  const trains = run.steps.findLast(step => step.makes)?.makes?.split("@")[0] ?? run.policy;
  const policy = system.policies.find(each => each.policy === trains);
  const channel = run.channels.find(each => each.adapter) ?? run.channels[0];
  const trained = run.done.filter(line => line.update).length, last = run.done.at(-1);
  const committed = run.steps.filter(step => step.state === "committed").length;
  const throughput = channel?.throughput.at(-1);
  const width = Math.max(300, document.getElementById("main").clientWidth - 100);
  const from = run.steps[0]?.parent, current = run.steps.findLast(step => step.state === "committed")?.makes;
  const head = h("div", { class: "head" }, h("h1", {}, `Run ${run.run}`),
    specs(spec("state", `${running(run)} · ${wrote(run)}`, run.state === "running" ? "good" : run.state === "idle" ? "warm" : ""), policy ? spec("trains", policy.policy, "violet") : null,
      spec("from", from ?? (run.steps.length ? "the base model" : "–")), current ? spec("now", current, "violet") : null, channel?.adapter ? spec("serving", channel.adapter, "accent") : null,
      spec("fence", run.fence ?? "–"), spec("directory", run.directory ?? "–"), run.starts > 1 ? spec("started", `${run.starts} times`) : null));
  const groups = groupsOf(run), stepping = run.steps.filter(step => step.state === "stepping");
  const waiting = run.next.filter(number => groups.get(number)?.line);
  const member = number => {
    const line = groups.get(number)?.line;
    return link(groupPlace(run.run, number), { class: "member" }, h("b", {}, `#${number}`), h("span", { class: "what" }, groups.get(number)?.task ?? ""),
      line ? dotsOf(line) : null, h("span", { class: "faint" }, line?.rewards.length ? line.rewards.map(figure).join(" ") : ""));
  };
  const flight = run.open.length || stepping.length || waiting.length ? h("div", { class: "tiles" },
    stepping.map(step => link(stepPlace(run.run, step.step), { class: "tile rail violet" },
      h("header", {}, h("b", {}, `Step ${step.step}`), h("span", { class: "what" }, `→ ${versionOf(step.makes)} · ${step.segments ?? "?"} segments`),
        mark("stepping", `${span(system.at - step.decided)}`)),
      h("div", { class: "members" }, step.groups.map(member)))),
    waiting.length ? h("div", { class: "tile rail warm" },
      h("header", {}, h("b", {}, "Toward the next step"), h("span", { class: "what" }, `${waiting.length} recorded, waiting for a step`)),
      h("div", { class: "members" }, waiting.map(member))) : null,
    run.open.map(group => link(groupPlace(run.run, group.number), { class: "tile" },
      h("header", {}, h("b", {}, `#${group.number}`), h("span", { class: "what" }, `${group.task} · ${group.title ?? ""}`),
        h("span", { class: "faint small" }, group.decided ? span(system.at - group.decided) : "")),
      stages(group),
      group.count ? cells(asked(group)) : null)))
    : h("div", { class: "empty" }, "Nothing is in flight.");
  const recent = [...run.steps].reverse().slice(0, 10);
  const tasks = new Map();
  for (const line of run.done) {
    const task = tasks.get(line.task) ?? { title: line.title, groups: 0, trained: 0 };
    task.groups += 1; task.trained += line.update ? 1 : 0; task.last = line; tasks.set(line.task, task);
  }
  const played = [...tasks].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));
  // From what it started, where it is now, and how its groups went, early and late.
  const solvedOf = lines => lines.flatMap(line => line.solved), share = outcomes => outcomes.length ? `${Math.round(100 * outcomes.filter(Boolean).length / outcomes.length)}%` : "–";
  const outcomes = solvedOf(run.done), half = Math.ceil(run.done.length / 2);
  const all = run.done.filter(line => line.solved.length && line.solved.every(Boolean)).length, none = run.done.filter(line => line.solved.length && !line.solved.some(Boolean)).length;
  const kpis = h("div", { class: "kpis" },
    kpi("Started from", from ? versionOf(from) : run.steps.length ? "base" : "–", from ?? (run.steps.length ? "the base model" : "no step yet")),
    kpi("Now", current ? versionOf(current) : "–", policy ? `${policy.versions.length} versions kept` : "no step committed"),
    kpi("Steps", `${run.steps.length}`, `${committed} committed · ${run.next.length} waiting`),
    kpi("Groups done", `${run.done.length}`, `${trained} trained on, of ${run.decided} decided`),
    kpi("Groups solved", `${all} · ${run.done.length - all - none} · ${none}`, "all · some · none solved"),
    kpi("Episodes", `${outcomes.length}`, `${outcomes.filter(Boolean).length} solved · ${share(outcomes)}`),
    kpi("Solved, early → late", run.done.length > 1 ? `${share(solvedOf(run.done.slice(0, half)))} → ${share(solvedOf(run.done.slice(half)))}` : "–", run.done.length > 1 ? `groups 1–${half}, then the ${run.done.length - half} after` : ""),
    kpi("Mean reward", figure(mean(run.done.flatMap(line => line.rewards))), "over every episode done"),
    kpi("Rows unlocked", last ? `${last.unlocked}` : "–", "of the catalog"),
    kpi("Inference", throughput ? `${figure(throughput.tokens_per_second)} tok/s` : "–", throughput ? `${figure(throughput.mean_concurrency)} requests at once` : "no measurement yet"));
  return [head, elsewhere(run), kpis,
    h("div", { class: "section-title" }, h("h2", {}, "In flight"),
      h("span", {}, [`${run.open.length} groups playing`, stepping.length ? `step ${stepping.map(step => step.step).join(", ")} being taken` : null].filter(Boolean).join(" · "))), flight,
    card("Rewards by group", "each dot an episode; rewards are each task's own",
      run.done.length ? [rewardsChart(run, width), h("div", { class: "legend" },
        h("span", {}, h("i", { style: "background:var(--good)" }), "solved"), h("span", {}, h("i", { style: "background:var(--faint)" }), "not solved"),
        h("span", {}, h("i", { class: "rule" }), "mean"), h("span", {}, h("i", { class: "square", style: "background:var(--accent)" }), "trained on"),
        h("span", {}, h("i", { class: "square", style: "background:var(--violet)" }), "in the step being taken"),
        h("span", {}, h("i", { class: "square", style: "background:var(--warm)" }), "waits for a step"),
        h("span", {}, h("i", { class: "square hollow" }), "skipped"), h("span", {}, h("i", { class: "square", style: "background:var(--bad)" }), "no episode, or the step failed"))]
        : h("div", { class: "empty" }, "No group is done with yet.")),
    h("div", { class: "cols" },
      card("Steps", "newest first", table([["step"], ["made"], ["groups"], ["solved"], ["segments", "n"], ["moved", "n"], ["took", "n"]],
        recent.map(step => {
          const version = versionNamed(step.makes), lines = step.groups.map(number => groups.get(number)?.line).filter(Boolean);
          const solved = lines.flatMap(line => line.solved);
          return [{ text: `S${step.step}`, kind: "key" }, step.state === "committed" ? versionOf(step.makes) : { text: step.state, kind: stateKind(step.state) },
            { node: h("span", {}, range(step.groups), step.skipped.length ? h("span", { class: "faint" }, ` + ${step.skipped.length} skipped`) : null) },
            solved.length ? `${solved.filter(Boolean).length}/${solved.length}` : "–", figure(step.segments),
            version?.metrics.kl_moved?.toFixed(4) ?? "–", span(version?.metrics.update_seconds ?? version?.metrics.seconds)];
        }),
        recent.map(step => () => go(stepPlace(run.run, step.step))))),
      card("Tasks played", `${played.length} rows`, table([["task"], ["groups", "n"], ["trained", "n"], ["last rewards"], ["solved", "n"]],
        played.map(([key, task]) => [{ text: key, kind: "key" }, task.groups, task.trained, task.last.rewards.map(figure).join(" ") || "–",
          `${task.last.solved.filter(Boolean).length}/${task.last.rewards.length}`]),
        played.map(([, task]) => () => go(groupPlace(run.run, task.last.group))))))];
}

// A step: one update of the policy, over the groups it covers
function drawStep(here) {
  const run = state.system.runs.find(each => each.run === here.run), step = run?.steps.find(each => each.step === here.number);
  if (!step) return [h("div", { class: "empty" }, `There is no step ${here.number}.`)];
  const version = versionNamed(step.makes), metrics = version?.metrics, groups = groupsOf(run);
  const lines = step.groups.map(number => groups.get(number)?.line).filter(Boolean), solved = lines.flatMap(line => line.solved);
  const head = h("div", { class: "head" }, h("h1", {}, `Step ${step.step}`),
    specs(spec("makes", step.makes ? link(policyPlace(step.makes.split("@")[0]), {}, step.makes) : "–", "violet"),
      spec("from", step.parent ?? "the base model"), spec("state", step.state, stateKind(step.state)), spec("decided", clock(step.decided) || "–")));
  const kpis = h("div", { class: "kpis" },
    kpi("Groups", `${step.groups.length}`, `${range(step.groups)}${step.skipped.length ? ` · ${step.skipped.length} gave nothing to train on` : ""}`),
    kpi("Solved", solved.length ? `${solved.filter(Boolean).length} of ${solved.length}` : "–", "episodes of its groups"),
    kpi("Segments", figure(step.segments), "trained on"),
    kpi("Moved", metrics?.kl_moved != null ? metrics.kl_moved.toFixed(4) : "–", "KL from its parent"),
    kpi("Took", metrics ? span(metrics.update_seconds ?? metrics.seconds) : step.state === "stepping" ? span(state.system.at - step.decided) : "–",
      step.state === "stepping" ? "so far" : ""));
  const tiles = h("div", { class: "tiles" }, [...step.groups, ...step.skipped].map(number => {
    const group = groups.get(number), line = group?.line, skipped = step.skipped.includes(number);
    return link(groupPlace(run.run, number), { class: `tile rail ${skipped ? "" : line?.rewards.length ? "good" : "bad"}` },
      h("header", {}, h("b", {}, `#${number}`), h("span", { class: "what" }, `${group?.task ?? ""} · ${group?.title ?? ""}`)),
      h("div", { class: "big" }, line?.rewards.length ? figure(mean(line.rewards)) : h("span", { class: "faint" }, "–")),
      h("div", { class: "facts" }, line ? dotsOf(line) : null, line?.rewards.length ? h("span", {}, line.rewards.map(figure).join(" ")) : null,
        skipped ? h("span", {}, line?.skipped ?? "nothing to train on") : line ? h("span", {}, h("b", {}, line.segments ?? "–"), " segments") : null));
  }));
  const what = card("The update", version ? `${version.name}, from ${version.parent ?? "the base model"}` : step.state === "failed" ? "the step failed" : "being taken",
    metrics ? pairs([...["kl_moved", "kl_floor", "loss", "clip_fraction", "mean_mismatch", "mean_weight", "truncated_fraction", "optimizer_steps", "tokens", "longest_segment_tokens", "peak_gpu_gib"]
      .filter(key => metrics[key] !== undefined).map(key => [key.replaceAll("_", " "), figure(metrics[key])]), ["took", span(metrics.update_seconds ?? metrics.seconds)]])
      : step.error ? h("p", { class: "error-text" }, step.error) : h("p", { class: "muted" }, "The trainer is working on it."));
  return [head, kpis, h("div", { class: "section-title" }, h("h2", {}, "Groups"), h("span", {}, "what went into the step")), tiles, what];
}

// A group
function drawGroup(here) {
  const group = state.group;
  if (!group || group.run !== here.run || group.number !== here.number) return [h("div", { class: "empty" }, "Reading the group…")];
  const outcome = group.outcome, version = group.version, result = group.result;
  const rewards = result?.rewards ?? group.episodes.filter(each => each.outcome === "completed").map(each => each.reward);
  const head = h("div", { class: "head" }, h("h1", {}, `Group #${group.number}`),
    specs(spec("task", group.task), spec("row", group.title ?? ""), spec("stage", group.stage, stateKind(group.stage)),
      outcome ? spec("outcome", outcomeOf(outcome).text.split(" · ")[0], outcome.update ? "good" : outcome.error ? "bad" : "") : null));
  const kpis = h("div", { class: "kpis" },
    kpi("Mean reward", rewards.length ? figure(mean(rewards)) : "–", rewards.length ? rewards.map(figure).join("  ") : "no episode has ended"),
    kpi("Solved", result ? `${result.solved.filter(Boolean).length} of ${result.solved.length}` : `${group.episodes.filter(each => each.solved).length} of ${group.count ?? "?"}`, result?.failed ? `${result.failed} failed` : ""),
    kpi("Played for", result ? span(result.rollout_seconds) : group.decided ? span(state.system.at - group.decided) : "–", group.decided ? `decided ${clock(group.decided)}` : ""),
    kpi("To train on", result ? `${result.segments}` : "–", result ? `of ${result.segments_recorded} segments` : ""),
    (() => {
      const run = state.system.runs.find(each => each.run === group.run), step = run && stepOf(run, group.number);
      const kept = step && !step.groups.includes(group.number);
      return kpi("Step", step ? link(stepPlace(group.run, step.step), {}, `S${step.step} → ${versionOf(step.makes)}`) : "–",
        step ? (kept ? "decided after it; nothing of it trained" : `over ${range(step.groups)}`) : run?.next.includes(group.number) ? "toward the next step" : "");
    })());
  const episodes = group.episodes.length || group.count ? h("div", { class: "tiles" }, asked(group).map(episode => {
    if (episode.waiting) return h("div", { class: "tile rail waiting" },
      h("header", {}, h("b", {}, `Episode ${episode.episode}`), h("span", { class: "what" }, ""), mark("", "not started")),
      h("div", { class: "big" }, h("span", { class: "faint" }, "–")),
      h("div", { class: "facts" }, h("span", {}, "waits for room: at most so many episodes run at once")));
    const info = episode.info ?? {};
    return link(episodePlace(episode.run_id), { class: `tile rail ${episode.interrupted ? "" : stateKind(episode.state)}` },
      h("header", {}, h("b", {}, `Episode ${episode.episode ?? "?"}`), h("span", { class: "what" }, ""),
        mark(episode.interrupted ? "" : episode.state ?? "running", episode.interrupted ? "interrupted" : episode.state ?? "running")),
      h("div", { class: "big" }, episode.outcome ? figure(episode.reward) : h("span", { class: "faint" }, "…")),
      h("div", { class: "facts" }, info.turns != null ? h("span", {}, h("b", {}, info.turns), " turns") : episode.samples != null ? h("span", {}, h("b", {}, episode.samples), " samples") : null,
        info.duration != null ? h("span", {}, h("b", {}, figure(info.duration)), " game min") : null,
        info.ended ? h("span", {}, "ended by ", h("b", {}, info.ended)) : null,
        episode.sampled ? h("span", {}, h("b", {}, tokens(episode.sampled)), " tokens") : null,
        episode.solved ? h("span", { class: "moved" }, "solved") : null),
      episode.detail ? h("div", { class: "error-text" }, episode.detail.slice(0, 240)) : null);
  })) : h("div", { class: "empty" }, "No episode has started.");
  const metrics = version?.metrics ?? outcome?.update;
  const what = card("What was done", outcome ? outcomeOf(outcome).text : "nothing yet",
    metrics ? [pairs([
      ["version", version ? link(policyPlace(version.name.split("@")[0]), {}, h("b", {}, version.name)) : outcome?.adapter ?? "–"],
      ["from", version?.parent ?? group.step?.parent ?? "the base model"],
      ...["kl_moved", "kl_floor", "loss", "clip_fraction", "mean_mismatch", "mean_weight", "truncated_fraction", "optimizer_steps", "tokens", "longest_segment_tokens", "peak_gpu_gib"]
        .filter(key => metrics[key] !== undefined).map(key => [key.replaceAll("_", " "), figure(metrics[key])]),
      ["took", span(metrics.update_seconds ?? metrics.seconds)]])]
      : group.step ? pairs([["makes", group.step.makes ?? "–"], ["from", group.step.parent ?? "the base model"], ["segments", figure(group.step.segments)],
        ["decided", group.step.decided ? `${span(state.system.at - group.step.decided)} ago` : "–"]])
      : result?.skipped ? h("p", { class: "muted" }, result.skipped)
      : result ? h("p", { class: "muted" }, "Recorded; it waits in the queue until enough groups are there for a step.")
      : h("p", { class: "muted" }, "The group is still being played."));
  const start = card("Start", "what every episode of the group was given", pairs(Object.entries(group.parameters ?? {}).map(([key, value]) => [key, figure(value)])));
  const failures = result?.failures?.length ? card("Why episodes failed", `${result.failures.length}`, result.failures.map(failure => h("p", { class: "error-text" }, failure))) : null;
  const run = state.system.runs.find(each => each.run === group.run);
  return [head, run ? elsewhere({ ...run, episodes_at: group.episodes_at }) : null, kpis, group.stage !== "done" ? card("Stage", "", stages(group)) : null,
    h("div", { class: "section-title" }, h("h2", {}, "Episodes"), h("span", {}, group.count ? `${group.count} asked for · ${group.playing.length ? `${group.playing.map(claim => claim.runner).filter((name, index, names) => names.indexOf(name) === index).join(", ")} playing` : `${group.ended} ended`}` : "")), episodes,
    h("div", { class: "cols" }, what, start), failures];
}

// An episode: what it reported, then its rollouts, one per agent (each becomes a trajectory to train on)
function drawEpisode(here) {
  const episode = state.episode;
  if (!episode || episode.run_id !== here.id) return [h("div", { class: "empty" }, "Reading the episode…")];
  const labels = episode.labels ?? {}, info = episode.ended?.info ?? {};
  const slots = new Map();
  let summaries = 0;
  for (const line of episode.lines) {
    if (line.kind !== "sample") continue;
    if (!line.tools.length) { summaries += 1; continue; }  // (an agent summarising its memory: no action is offered)
    if (!slots.has(line.slot)) slots.set(line.slot, []);
    slots.get(line.slot).push(line);
  }
  const ordered = new Map([...slots].sort(([a], [b]) => a.localeCompare(b)));
  // An episode of a training run is of a group, whose record says its task; another says it in its labels, if at all.
  const run = labels.run ? state.system.runs.find(each => each.run === labels.run) : null;
  const group = run && labels.group ? groupsOf(run).get(Number(labels.group)) : null;
  const task = group?.task ?? labels.task, title = group?.title ?? labels.title;
  const head = h("div", { class: "head" }, h("h1", {}, labels.episode ? `Episode ${labels.episode}` : episode.run_id.slice(-8)),
    specs(spec("state", episode.state ?? "running", stateKind(episode.state)),
      episode.ended ? spec("reward", figure(episode.ended.reward), episode.ended.solved ? "good" : "") : null,
      task ? spec("task", title ? `${task} · ${title}` : task) : null, spec("run", episode.run_id),
      episode.source === "archive" ? spec("read from", "its kept events: replies only", "warm") : null));
  const scalars = Object.entries(info).filter(([, value]) => value === null || typeof value !== "object");
  const nested = Object.entries(info).filter(([, value]) => value !== null && typeof value === "object");
  const kpis = h("div", { class: "kpis" }, ["reward", "turns", "duration", "ended"].filter(key => info[key] !== undefined).map(key => kpi(key, figure(info[key]))),
    episode.ended?.sampled ? kpi("tokens sampled", tokens(episode.ended.sampled)) : null);
  const result = Object.keys(info).length ? card("What the episode reported", "the program's result", h("div", { class: "cols", style: "gap:16px" },
    pairs(scalars.filter(([key]) => !["reward", "turns", "duration", "ended"].includes(key)).map(([key, value]) => [key.replaceAll("_", " "), figure(value)])),
    h("div", { style: "display:grid;gap:12px" }, nested.map(([key, value]) => h("div", {}, h("div", { class: "small muted", style: "margin-bottom:4px" }, key.replaceAll("_", " ")),
      Object.keys(value).length === 0 ? h("span", { class: "none" }, "none") : h("div", { class: "chips" }, Array.isArray(value) ? value.map(each => h("span", { class: "chip" }, typeof each === "object" ? JSON.stringify(each) : String(each)))
        : Object.entries(value).sort(([, a], [, b]) => (typeof b === "number" ? b : 0) - (typeof a === "number" ? a : 0))
          .map(([name, each]) => h("span", { class: "chip" }, `${name} `, h("b", {}, typeof each === "object" ? JSON.stringify(each) : figure(each))))))))))
    : null;
  if (!ordered.size) return [head, kpis, result, h("div", { class: "empty" }, episode.source ? "No agent has taken a turn yet." : "Neither the feed nor the job's log has this episode's rollouts.")];
  const slot = here.slot && ordered.has(here.slot) ? here.slot : null;
  const shown = slot ? [[slot, ordered.get(slot)]] : [...ordered];
  const turns = Math.max(...shown.map(([, samples]) => samples.length));
  if (state.follow) state.turn = turns - 1;
  state.turn = Math.max(0, Math.min(state.turn, turns - 1));
  const tabs = h("div", { class: "segmented" }, h("button", { class: `seg${slot ? "" : " current"}`, onclick: () => go(episodePlace(here.id)) }, "All rollouts"),
    [...ordered.keys()].map(each => h("button", { class: `seg${slot === each ? " current" : ""}`, onclick: () => go(episodePlace(here.id, each)) }, avatar(each), each)));
  const move = delta => () => { state.turn = Math.max(0, Math.min(turns - 1, state.turn + delta)); state.follow = false; redraw(); };
  const modes = h("div", { class: "segmented" },
    h("button", { class: `seg${state.whole ? "" : " current"}`, onclick: () => { state.whole = false; redraw(); } }, "Turn by turn"),
    h("button", { class: `seg${state.whole ? " current" : ""}`, onclick: () => { state.whole = true; redraw(); } }, "Whole trajectory"));
  const scrub = h("div", { class: "scrub" }, modes, state.whole ? h("output", {}, `${turns} turns`) : [
    h("button", { onclick: move(-1), "aria-label": "previous turn" }, "‹"), h("button", { onclick: move(1), "aria-label": "next turn" }, "›"),
    h("output", {}, `turn ${state.turn + 1} of ${turns}`),
    h("input", { type: "range", min: 0, max: turns - 1, value: state.turn, "aria-label": "turn", oninput: event => { state.turn = Number(event.target.value); state.follow = false; redraw(); } }),
    h("label", {}, h("input", { type: "checkbox", checked: state.follow, onchange: event => { state.follow = event.target.checked; redraw(); } }), "follow"),
    episode.source === "feed" ? h("label", {}, h("input", { type: "checkbox", checked: state.full, onchange: event => { state.full = event.target.checked; redraw(); } }), "whole context") : null]);
  const cards = state.whole ? timeline(shown, turns, episode) : h("div", { class: "turns-frame" }, h("div", { class: "turns", style: `grid-template-columns: repeat(${shown.length}, minmax(300px, 1fr))` },
    shown.map(([each, samples]) => turnCard(each, samples, state.turn, episode))));
  const effects = otherEffects(episode.lines);
  return [head, kpis, result, h("div", { class: "section-title" }, h("h2", {}, "Rollouts"), h("span", {}, `${ordered.size}, one per agent, each a trajectory to train on${summaries ? ` · ${summaries} memory summaries written` : ""}`)), tabs, scrub, cards, effects];
}

// Every turn of the rollouts shown, one row a turn and one column a rollout: what each agent did and what came
// back, with what it saw and what it thought one click away (drawn when opened; what is open stays open).
function timeline(shown, turns, episode) {
  const opened = state.opened ??= new Set();
  const folded = (key, label, fill) => {
    const details = h("details", { open: opened.has(key) }, h("summary", {}, label));
    const filled = () => { if (details.open && details.childElementCount === 1) details.append(fill()); };
    details.addEventListener("toggle", () => { if (details.open) opened.add(key); else opened.delete(key); filled(); });
    filled();
    return details;
  };
  const grid = h("div", { class: "timeline", style: `grid-template-columns: 52px repeat(${shown.length}, minmax(300px, 1fr))` },
    h("div", { class: "when head" }), shown.map(([slot]) => h("div", { class: "head" }, avatar(slot), h("b", {}, slot))));
  for (let turn = 0; turn < turns; turn += 1) {
    grid.append(h("div", { class: "when" }, String(turn + 1)));
    for (const [slot, samples] of shown) {
      const sample = samples[turn];
      if (!sample) { grid.append(h("div", { class: "step" }, h("span", { class: "none" }, "–"))); continue; }
      const result = resultOf(samples, turn, sample.reply.calls[0]?.id);
      const key = `${episode.run_id} ${slot} ${turn}`;
      grid.append(h("div", { class: "step" },
        sample.reply.text.trim() ? h("p", { class: "said" }, sample.reply.text.trim()) : null,
        sample.reply.calls.map(call => h("div", {}, h("span", { class: "call" }, `${call.name}(${Object.entries(call.arguments).map(([name, value]) => `${name}: ${JSON.stringify(value)}`).join(", ")})`))),
        !sample.reply.calls.length && !sample.reply.text.trim() ? h("span", { class: "none" }, "nothing") : null,
        result ? h("div", { class: `result${result.error ? " error" : ""}` }, result.text) : null,
        h("div", { class: "folds" },
          sample.reply.reasoning ? folded(`${key} thought`, "thought", () => h("p", { class: "thought" }, sample.reply.reasoning.trim())) : null,
          sample.messages.length ? folded(`${key} saw`, "saw", () => h("div", { class: "sees" }, seen(sample))) : null,
          h("span", { class: "faint small" }, `${figure(sample.seconds)} s`))));
    }
  }
  return h("div", { class: "turns-frame" }, grid);
}

function turnCard(slot, samples, turn, episode) {
  const sample = samples[turn];
  const header = h("header", {}, avatar(slot), h("b", {}, slot), h("span", {}, sample ? `${figure(sample.seconds)} s · ${sample.finish_reason ?? ""}` : ""));
  if (!sample) return h("div", { class: "turn", "data-slot": slot }, header, h("section", {}, h("span", { class: "none" }, "no turn yet")));
  const sees = h("section", { class: "sees" }, h("h3", {}, "Sees"), sample.messages.length ? seen(sample) : h("span", { class: "none" }, "kept only as tokens: the feed has let this episode go"));
  const thinks = h("section", { class: "thinks" }, h("h3", {}, "Thinks"), sample.reply.reasoning ? h("p", {}, sample.reply.reasoning.trim()) : h("span", { class: "none" }, "nothing recorded"));
  const does = h("section", {}, h("h3", {}, "Does"), sample.reply.text.trim() ? h("p", { class: "said" }, sample.reply.text.trim()) : null,
    sample.reply.calls.map(call => h("div", {}, h("span", { class: "call" }, `${call.name}(${Object.entries(call.arguments).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join(", ")})`))),
    !sample.reply.calls.length && !sample.reply.text.trim() ? h("span", { class: "none" }, "nothing") : null);
  const result = resultOf(samples, turn, sample.reply.calls[0]?.id);
  const answered = h("section", {}, h("h3", {}, "Result"), result ? h("div", { class: `result${result.error ? " error" : ""}` }, result.text)
    : h("span", { class: "none" }, episode.source === "archive" ? "not kept" : episode.state === "running" ? "pending" : "the episode ended"));
  return h("div", { class: "turn", "data-slot": slot }, header, sees, thinks, does, answered);
}

function resultOf(samples, index, callId) {
  const next = samples[index + 1];
  if (!next) return null;
  for (const message of next.messages) for (const result of message.results) if (result.id === callId) return result;
  const last = next.messages.filter(message => message.role === "user").at(-2);  // (a reply without a call is answered in words)
  return last ? { text: last.text, error: false } : null;
}

function mapRow(text) {
  const row = h("div", { class: "row" }), [label, ...cells] = text.split(" ");
  row.append(label.padStart(5) + " ");
  for (const cell of cells) {
    const kind = cell === "#" ? "solid" : cell === "." ? "air" : cell === "?" ? "unseen" : cell === "@" ? "self" : /[A-Z+]/.test(cell) ? "mate" : "thing";
    row.append(h("span", { class: `c-${kind}` }, cell + " "));
  }
  return row;
}

function seen(sample) {
  const pre = h("pre");
  const messages = state.full ? sample.messages : sample.messages.filter(message => message.role === "user").slice(-1);
  for (const message of messages) {
    if (state.full) pre.append(h("div", { class: "faint" }, `— ${message.role} —`));
    for (const line of (message.text || "").split("\n")) pre.append(/^-?\d+( \S){6,}$/.test(line) ? mapRow(line) : h("div", {}, line || " "));
    if (state.full) for (const call of message.calls) pre.append(h("div", {}, `→ ${call.name}(${JSON.stringify(call.arguments)})`));
    if (state.full) for (const result of message.results) pre.append(h("div", {}, `← ${result.text}`));
  }
  return pre;
}

function otherEffects(lines) {
  const requests = new Map(), effects = [];
  for (const line of lines) {
    if (line.kind !== "event") continue;
    if (line.type === "effect.requested" && line.payload.kind !== "model.sample") requests.set(line.payload.effect_id, line.payload);
    if (line.type === "effect.completed" && requests.has(line.payload.effect_id)) effects.push([requests.get(line.payload.effect_id), line.payload]);
  }
  if (!effects.length) return null;
  const rows = effects.slice(-60).reverse().map(([request, completed]) => [
    { text: String(request.payload?.tool ?? request.payload?.name ?? request.kind), kind: "key" },
    JSON.stringify(request.payload?.arguments ?? request.payload ?? {}).slice(0, 160),
    JSON.stringify(completed.payload?.structured ?? completed.payload ?? completed.status).slice(0, 240)]);
  return h("details", { class: "card", style: "padding:12px 18px" }, h("summary", {}, `Tool calls the program made (${effects.length}; newest first)`),
    h("div", { style: "margin-top:10px" }, table([["tool"], ["arguments"], ["returned"]], rows)));
}

// A policy
function drawPolicy(name) {
  const system = state.system, policy = system.policies.find(each => each.policy === name);
  if (!policy) return [h("div", { class: "empty" }, `There is no policy ${name}.`)];
  const versions = policy.versions, width = Math.max(300, document.getElementById("main").clientWidth - 100);
  const madeBy = new Map(system.runs.flatMap(run => run.steps.filter(step => step.makes).map(step => [step.makes, [run.run, step.step]])));
  const size = versions.reduce((sum, version) => sum + (version.weights?.bytes ?? 0) + (version.state?.bytes ?? 0), 0);
  const serving = system.channels.find(channel => channel.adapter?.startsWith(`${name}@`));
  const newest = [...versions].reverse();
  return [h("div", { class: "head" }, h("h1", {}, `Policy ${policy.policy}`),
    specs(serving ? spec("served on", serving.channel, "accent") : null, spec("fence", policy.fence ?? "–"),
      spec("a version is", "the commit of one step: its weights and the trainer's state, as files in the blob store"))),
  h("div", { class: "kpis" }, kpi("Head", versionOf(policy.head), policy.head ?? ""), kpi("Versions", `${versions.length}`),
    kpi("Kept", bytes(size), "weights and trainer state"),
    kpi("Last step moved", versions.length ? figure(versions.at(-1).metrics.kl_moved) : "–", "KL from its parent")),
  card("How far each step moved the policy", "KL between a version and its parent", barChart(versions.map(version => version.metrics.kl_moved ?? 0),
    versions.map(version => versionOf(version.name)), width, 150, index => { const made = madeBy.get(versions[index].name); if (made) go(stepPlace(...made)); })),
  card("Versions", "newest first", table([["version"], ["from"], ["made"], ["by"], ["segments", "n"], ["steps", "n"], ["moved", "n"], ["loss", "n"], ["took", "n"], ["state"], ["size", "n"]],
    newest.map(version => [{ text: versionOf(version.name), kind: "key" }, versionOf(version.parent), clock(version.made),
      madeBy.has(version.name) ? `step ${madeBy.get(version.name)[1]}` : "–", figure(version.metrics.segments), figure(version.metrics.optimizer_steps),
      version.metrics.kl_moved?.toFixed(4), version.metrics.loss?.toFixed(3), span(version.metrics.update_seconds ?? version.metrics.seconds),
      version.state ? "kept" : version.released ? { text: "released", kind: "still" } : "–", bytes((version.weights?.bytes ?? 0) + (version.state?.bytes ?? 0))]),
    newest.map(version => madeBy.has(version.name) ? () => go(stepPlace(...madeBy.get(version.name))) : null)))];
}

// Every policy, as a graph: a lane for each policy with its versions from the left (folded to the ones that matter
// until it is opened), and between lanes the forks and the distillations; beside it the trainers with their queues,
// the inference workers with what each serves, and evaluations. What no run writes yet comes from the sample
// fixture, when it is asked for, and is marked so.
const policiesPlace = sample => `#/policies${sample ? "/sample" : ""}`;
const LANE = 78, COLUMN = 54, PAD = 34;
const short = name => name ? String(name).split("/").at(-1) : "–";
const modeKind = mode => mode === "on-policy" ? "violet" : mode === "off-policy" ? "warm" : "accent";
const lifeKind = state => ({ serving: "good", "rolling out": "warm", resharding: "violet", resharded: "accent" })[state] ?? "";
const sampleChip = () => h("span", { class: "chip sample", title: "from the sample fixture: no run writes this yet" }, "sample");
const ago = (lineage, at) => at ? span(Math.max(0, lineage.now - at)) : "–";

// What each version is, wherever it is drawn: its policy's line, the run that made it, its evaluations.
function indexOf(lineage) {
  const versions = new Map(), runs = new Map(lineage.runs.map(run => [run.run, run])), scores = new Map();
  for (const policy of lineage.policies) for (const version of policy.versions) versions.set(version.name, { ...version, policy: policy.policy });
  for (const suite of lineage.evaluations) for (const subject of suite.subjects) {
    if (subject.version && !scores.has(subject.version)) scores.set(subject.version, { suite: suite.suite, starts: suite.starts.length, ...subject });
  }
  return { versions, runs, scores };
}

// The lanes, in order: under each policy, those that start from it (by a fork, or a distillation's start); versions
// that this ledger does not have, but that something here starts from, in a lane of their own at the top.
function lanesOf(lineage, index) {
  const policies = new Map(lineage.policies.map(policy => [policy.policy, policy]));
  const source = policy => {
    const first = policy.versions[0], run = first?.by ? index.runs.get(first.by.run) : null;
    const from = run?.kind === "distill" ? run.from ?? run.teachers[0] : first?.parent;
    if (!from) return null;
    const owner = from.split("@")[0];
    return owner === policy.policy ? null : policies.has(owner) ? owner : "outside";
  };
  const children = new Map();
  for (const policy of lineage.policies) {
    const parent = source(policy);
    children.set(parent, [...(children.get(parent) ?? []), policy]);
  }
  const made = policy => policy.versions[0]?.made ?? 0;
  const lanes = [];
  const visit = (key, depth) => {
    for (const policy of [...(children.get(key) ?? [])].sort((a, b) => made(a) - made(b))) {
      lanes.push({ key: policy.policy, policy, depth });
      visit(policy.policy, depth + 1);
    }
  };
  if (lineage.outside.length) { lanes.push({ key: "outside", outside: lineage.outside, depth: 0 }); visit("outside", 1); }
  visit(null, 0);
  return lanes;
}

// A lane's items, left to right: versions, the distillations that made some of them (each before the first version
// it made), and, while the lane is folded, a gap for each stretch of versions that nothing points at.
function itemsOf(lane, open, anchors, lineage, index) {
  if (lane.outside) return lane.outside.map(name => ({ kind: "version", name, outside: true }));
  const items = [], versions = lane.policy.versions, distills = lineage.runs.filter(run => run.kind === "distill" && run.policy === lane.key);
  const before = new Map(distills.map(run => [run.versions[0] ?? null, run]));
  let hidden = [];
  const flush = () => { if (hidden.length) items.push({ kind: "gap", count: hidden.length, names: hidden }); hidden = []; };
  versions.forEach((version, place) => {
    const run = before.get(version.name);
    const by = version.by?.run, next = versions[place + 1], last = versions[place - 1];
    const shown = open || run || place === 0 || place === versions.length - 1 || anchors.has(version.name)
      || by !== last?.by?.run || by !== next?.by?.run || !["written", "superseded"].includes(version.life.state);
    if (run) { flush(); items.push({ kind: "distill", run }); }
    if (shown) { flush(); items.push({ kind: "version", name: version.name }); } else hidden.push(version.name);
  });
  flush();
  for (const run of distills) if (!run.versions.length) items.push({ kind: "distill", run });
  return items;
}

const itemId = item => item.kind === "version" ? `v:${item.name}` : item.kind === "distill" ? `d:${item.run.run}` : `g:${item.names[0]}`;

function lineageGraph(lineage, index, lanes) {
  const anchors = new Set(lineage.edges.flatMap(edge => [edge.from, edge.kind === "fork" ? edge.to : null]).filter(Boolean));
  for (const name of index.scores.keys()) anchors.add(name);
  const laid = lanes.map(lane => ({ ...lane, open: Boolean(folds[`lane:${lane.key}`]), items: [] }));
  for (const lane of laid) lane.items = itemsOf(lane, lane.open, anchors, lineage, index);
  // Columns: every item stands right of what it comes from, in its lane and across lanes (longest path).
  const before = new Map(), where = new Map();
  laid.forEach((lane, row) => lane.items.forEach((item, place) => {
    where.set(itemId(item), { item, row });
    before.set(itemId(item), place ? [itemId(lane.items[place - 1])] : []);
  }));
  for (const edge of lineage.edges) {
    const to = edge.kind === "fork" ? `v:${edge.to}` : `d:${edge.to}`;
    if (where.has(to) && where.has(`v:${edge.from}`)) before.get(to).push(`v:${edge.from}`);
  }
  const column = new Map(), visiting = new Set();
  const columnOf = id => {
    if (column.has(id)) return column.get(id);
    if (visiting.has(id)) return 0;  // (a cycle cannot happen: a version is made after what it comes from)
    visiting.add(id);
    const found = Math.max(0, ...before.get(id).map(each => columnOf(each) + 1));
    visiting.delete(id);
    column.set(id, found);
    return found;
  };
  for (const id of where.keys()) columnOf(id);
  const columns = Math.max(0, ...column.values()) + 1;
  const room = (document.getElementById("main")?.clientWidth ?? 1200) - 64 - 252;  // (the bands reach across the frame)
  const width = Math.max(room, PAD * 2 + (columns - 1) * COLUMN + 60), height = laid.length * LANE;
  const x = id => PAD + column.get(id) * COLUMN, y = row => row * LANE + LANE / 2 + 8;
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "policies as a graph", class: "dag-drawing" });
  laid.forEach((lane, row) => drawing.append(svg("rect", { x: 0, y: row * LANE, width, height: LANE,
    class: `lane-band${row % 2 ? " odd" : ""}${lane.policy?.sample ? " sample" : ""}` })));
  const kindOfVersion = name => {
    const version = index.versions.get(name), run = version?.by ? index.runs.get(version.by.run) : null;
    return run?.kind === "distill" ? modeKind(run.mode) : "accent";
  };
  // Along each lane: a line from item to item, in the color of what made the later one.
  laid.forEach((lane, row) => lane.items.forEach((item, place) => {
    if (!place) return;
    const from = lane.items[place - 1], kind = item.kind === "gap" || from.kind === "gap" ? "quiet dash"
      : item.kind === "distill" ? `${modeKind(item.run.mode)} dash` : lane.outside ? "quiet dash" : kindOfVersion(item.name);
    drawing.append(svg("line", { x1: x(itemId(from)), x2: x(itemId(item)), y1: y(row), y2: y(row), class: kind.split(" ").map(each => each === "dash" ? "dash" : `s-${each}`).join(" ") }));
  }));
  // Between lanes: forks, and each distillation's teachers and start.
  for (const edge of lineage.edges) {
    const from = where.get(`v:${edge.from}`), to = where.get(edge.kind === "fork" ? `v:${edge.to}` : `d:${edge.to}`);
    if (!from || !to) continue;
    const x1 = x(`v:${edge.from}`), y1 = y(from.row), x2 = x(itemId(to.item)) - (to.item.kind === "distill" ? 9 : 7), y2 = y(to.row);
    const middle = x1 + Math.max(18, (x2 - x1) * 0.55);
    const kind = edge.kind === "fork" ? "s-quiet" : `s-${modeKind(edge.mode)}${edge.kind === "start" ? " dash" : ""}`;
    drawing.append(svg("path", { d: `M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2} ${y2}`, class: `edge ${kind}` },
      svg("title", {}, edge.kind === "fork" ? `${edge.to} forks from ${edge.from}` : edge.kind === "teach" ? `${edge.from} teaches (${edge.mode})` : `the student starts from ${edge.from}`)));
  }
  // The items, over the lines; and above each lane, where a run's stretch of it begins.
  laid.forEach((lane, row) => {
    let previous = null;
    lane.items.forEach(item => {
      const id = itemId(item), cx = x(id), cy = y(row);
      if (item.kind === "gap") {
        drawing.append(svg("g", { class: "gap", onclick: () => { folds[`lane:${lane.key}`] = true; keepFolds(); redraw(); } },
          svg("rect", { x: cx - 15, y: cy - 9, width: 30, height: 18, rx: 9 }), svg("text", { x: cx, y: cy + 3.5, "text-anchor": "middle" }, `+${item.count}`),
          svg("title", {}, `${item.count} more versions: ${item.names[0]} to ${item.names.at(-1)} (open the lane)`)));
        return;
      }
      if (item.kind === "distill") {
        const run = item.run, kind = modeKind(run.mode);
        drawing.append(svg("g", { class: "distill" }, svg("path", { d: `M ${cx - 9} ${cy} L ${cx} ${cy - 9} L ${cx + 9} ${cy} L ${cx} ${cy + 9} Z`, class: `f-${kind}` }),
          svg("title", {}, `${run.run}: distil ${run.teachers.join(" + ")} into ${run.policy}${run.from ? `, from ${run.from}` : ""}\n${run.mode}: trains on ${(run.data.sampled_by ?? []).join(", ")}'s samples · objective ${run.objective}`)));
        drawing.append(svg("text", { x: cx, y: cy + 23, "text-anchor": "middle", class: `t-${kind}` }, run.mode));
        drawing.append(svg("text", { x: cx - 9, y: cy - 26, class: "stretch" }, run.run));
        previous = run.run;
        return;
      }
      const version = index.versions.get(item.name);
      if (!version) {  // (a version of a policy this ledger does not have)
        drawing.append(svg("circle", { cx, cy, r: 6, class: "dot-outside" }, svg("title", {}, `${item.name}: not in this ledger`)));
        drawing.append(svg("text", { x: cx + 10, y: cy + 3.5 }, item.name));
        return;
      }
      const run = version.by?.run;
      if (run !== previous && run && !(index.runs.get(run)?.kind === "distill")) drawing.append(svg("text", { x: cx - 6, y: cy - 26, class: "stretch" }, run));
      previous = run;
      const life = version.life, workers = Object.entries(life.workers).filter(([, span]) => span.until == null).map(([worker]) => worker);
      const score = index.scores.get(item.name);
      const real = state.system.runs.find(each => each.run === run && !version.sample);
      const group = svg("g", { class: `version${real || !lane.policy.sample ? " link" : ""}`, onclick: () => {
        if (real && version.by.step) go(stepPlace(run, version.by.step)); else if (!lane.policy.sample) go(policyPlace(lane.key));
      } });
      if (["serving", "rolling out", "resharding"].includes(life.state)) group.append(svg("circle", { cx, cy, r: 10.5, class: `ring ring-${lifeKind(life.state)}` }));
      group.append(svg("circle", { cx, cy, r: 6, class: version.kept ? `dot-${kindOfVersion(item.name)}` : "dot-released" }));
      group.append(svg("text", { x: cx, y: cy + 22, "text-anchor": "middle", class: "v" }, `@${version.number}`));
      if (score) group.append(svg("text", { x: cx, y: cy - 13, "text-anchor": "middle", class: `score t-${score.solved / Math.max(1, score.played) >= 0.6 ? "good" : score.solved / Math.max(1, score.played) >= 0.35 ? "warm" : "bad"}` },
        `${score.solved}/${score.played}${score.played < score.starts ? "…" : ""}`));
      group.append(svg("title", {}, [`${version.name}${lane.policy.sample ? " (sample)" : ""}`,
        `made ${clock(version.made)}${run ? ` by ${run}${version.by.step ? ` step ${version.by.step}` : ""}` : ""} from ${version.parent ?? "the base model"}`,
        `${life.state}${workers.length ? ` on ${workers.join(", ")}` : ""}${life.waiting ? ` · ${life.waiting} requests waiting` : ""}${life.latest_of ? ` · ${life.latest_of}'s latest` : ""}`,
        version.metrics.kl_moved != null ? `moved ${version.metrics.kl_moved.toFixed(4)} from its parent` : null,
        version.kept ? "its weights are kept" : "released: its weights are gone, its record stays",
        score ? `${score.suite}: solved ${score.solved} of ${score.played}${score.played < score.starts ? ` (${score.starts - score.played} starts to play)` : ""}` : null].filter(Boolean).join("\n")));
      drawing.append(group);
    });
  });
  // On the left, a label for each lane: it folds and unfolds the lane.
  const labels = h("div", { class: "dag-labels" }, laid.map(lane => {
    const toggle = () => { folds[`lane:${lane.key}`] = !lane.open; keepFolds(); redraw(); };
    if (lane.outside) return h("div", { class: "lane-label" }, h("span", {}), h("b", { class: "muted" }, "Outside this ledger"),
      h("small", {}, `${lane.outside.length} version${lane.outside.length === 1 ? "" : "s"} something here starts from`));
    const policy = lane.policy, first = policy.versions[0], run = first?.by ? index.runs.get(first.by.run) : null;
    const says = run?.kind === "distill" ? `distilled from ${run.teachers.join(" + ")}` : policy.fork ? `fork of ${policy.fork}` : "from the base model";
    return h("div", { class: "lane-label", style: `padding-left:${4 + Math.min(lane.depth, 3) * 10}px`, onclick: toggle, title: lane.open ? "fold the lane" : "show every version" },
      h("button", { class: `twist${lane.open ? " open" : ""}`, "aria-label": lane.open ? "collapse" : "expand", "aria-expanded": String(lane.open), onclick: event => { event.stopPropagation(); toggle(); } },
        svg("svg", { width: 10, height: 10, viewBox: "0 0 10 10" }, svg("path", { d: "M3 1.5 L7 5 L3 8.5", fill: "none", stroke: "currentColor", "stroke-width": 1.6 }))),
      h("b", {}, policy.sample ? policy.policy : link(policyPlace(policy.policy), { onclick: event => event.stopPropagation() }, policy.policy)),
      h("small", {}, `${policy.versions.length} versions · ${says}`),
      h("span", { class: "lane-tags" }, h("span", { class: "chip" }, policy.definition?.weights === "full" ? "full" : "LoRA"), policy.sample ? sampleChip() : null));
  }));
  return h("div", { class: "dag" }, labels, h("div", { class: "dag-frame" }, drawing));
}
const keepFolds = () => { try { localStorage.setItem("monitor.folds", JSON.stringify(folds)); } catch { /* (a browser that keeps nothing) */ } };

// A count over time, drawn as steps up to now: a trainer's queue (what waits, over what is being taken), and the
// groups of a run that wait toward a step.
function queueChart(depth, groups, now, width, height) {
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, role: "img", preserveAspectRatio: "none", class: "queue-chart" });
  const times = [...depth.map(point => point[0]), ...groups.map(point => point[0])];
  if (!times.length) return drawing;
  const start = Math.min(...times), top = Math.max(1, ...depth.map(point => point[1] + point[2]), ...groups.map(point => point[1]));
  const left = 18, base = height - 14, x = at => left + (at - start) / Math.max(1, now - start) * (width - left - 4), y = value => base - value / top * (base - 6);
  const stepped = (points, value) => { let path = ""; points.forEach((point, place) => {
    const next = points[place + 1]?.[0] ?? now; path += `${place ? " L" : "M"} ${x(point[0])} ${y(value(point))} L ${x(next)} ${y(value(point))}`; }); return path; };
  drawing.append(svg("line", { x1: left, x2: width, y1: base + 0.5, y2: base + 0.5, class: "s-grid" }));
  drawing.append(svg("text", { x: left - 5, y: y(top) + 3, "text-anchor": "end" }, String(top)), svg("text", { x: left - 5, y: base + 3, "text-anchor": "end" }, "0"));
  drawing.append(svg("text", { x: left, y: height - 2 }, `${span(now - start)} ago`), svg("text", { x: width - 4, y: height - 2, "text-anchor": "end" }, "now"));
  if (depth.length) {
    drawing.append(svg("path", { d: `${stepped(depth, point => point[2])} L ${x(now)} ${base} L ${x(depth[0][0])} ${base} Z`, class: "f-violet-soft" }));
    drawing.append(svg("path", { d: stepped(depth, point => point[2]), class: "s-violet" }));
    drawing.append(svg("path", { d: stepped(depth, point => point[1] + point[2]), class: "s-warm" }));
  }
  if (groups.length) drawing.append(svg("path", { d: stepped(groups, point => point[1]), class: "s-accent dash" }));
  return drawing;
}

function trainerTile(trainer, lineage) {
  const taking = trainer.queue.filter(entry => entry.state === "taking"), queued = trainer.queue.filter(entry => entry.state === "queued");
  const done = trainer.queue.filter(entry => entry.state === "made" || entry.state === "failed").slice(-3).reverse();
  const waited = trainer.queue.filter(entry => entry.began && entry.queued).map(entry => entry.began - entry.queued);
  const row = entry => h("div", { class: "member" }, h("b", {}, `S${entry.step}`), h("span", { class: "what" }, entry.run),
    mark(entry.state === "taking" ? "stepping" : entry.state === "queued" ? "queued" : entry.state === "made" ? "committed" : "failed",
      entry.state === "taking" ? `taking, ${ago(lineage, entry.began)}` : entry.state === "queued" ? `queued ${ago(lineage, entry.queued)}` : entry.state),
    h("span", { class: "faint" }, `→ ${entry.makes}`));
  return h("div", { class: `tile rail ${taking.length ? "violet" : queued.length ? "warm" : ""}` },
    h("header", {}, h("b", {}, trainer.trainer), h("span", { class: "what" }, `${trainer.weights === "full" ? "full weights" : "LoRA"}${trainer.base ? ` on ${short(trainer.base)}` : ""}`),
      trainer.sample ? sampleChip() : null),
    h("div", { class: "facts" },
      h("span", {}, trainer.weights === "full" ? "dedicated to " : trainer.implicit ? "trains " : "any LoRA of its base: ", h("b", {}, trainer.policies.join(", "))),
      trainer.colocated ? h("span", {}, "shares the engines' accelerator: they sleep while it steps") : trainer.where ? h("span", {}, trainer.where) : null,
      trainer.implicit ? h("span", {}, "not registered: the run's own, from its profile") : null),
    h("div", { class: "cells four" },
      h("div", { class: `cell ${taking.length ? "violet" : ""}` }, h("span", {}, "taking"), h("b", {}, String(taking.length)), h("small", {}, taking[0] ? `${taking[0].makes}` : "idle")),
      h("div", { class: `cell ${queued.length ? "warm" : "waiting"}` }, h("span", {}, "queued"), h("b", {}, String(queued.length)), h("small", {}, queued.length ? `oldest ${ago(lineage, queued[0].queued)}` : "none")),
      h("div", { class: "cell" }, h("span", {}, "waited"), h("b", {}, waited.length && !trainer.implicit ? span(mean(waited)) : "–"),
        h("small", {}, trainer.implicit ? "not recorded" : "mean, queued to taken")),
      h("div", { class: "cell good" }, h("span", {}, "made"), h("b", {}, String(trainer.queue.filter(entry => entry.state === "made").length)), h("small", {}, "versions"))),
    h("div", {}, queueChart(trainer.depth, trainer.groups ?? [], lineage.now, 420, 92),
      h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--violet)" }), "being taken"), h("span", {}, h("i", { style: "background:var(--warm)" }), "with those queued"),
        trainer.groups?.length ? h("span", {}, h("i", { class: "rule", style: "background:var(--accent)" }), "groups waiting toward a step") : null)),
    h("div", { class: "members queue" }, [...taking, ...queued, ...done].map(row)));
}

function workerTile(worker, lineage) {
  const holds = worker.holds?.policy ? `${worker.holds.policy}, full weights` : worker.holds?.base ? `${worker.serving.length} of ${worker.adapters ?? "?"} adapter slots` : "the run's engines";
  const waiting = lineage.routing.waiting ?? {};
  return h("div", { class: `tile rail ${worker.serving.length ? "good" : ""}` },
    h("header", {}, h("b", {}, worker.worker), h("span", { class: "what" }, holds), worker.share ? mark(worker.share === "evaluations" ? "queued" : "", worker.share) : null, worker.sample ? sampleChip() : null),
    worker.machine ? h("div", { class: "facts" }, h("span", {}, worker.machine), h("span", {}, worker.accelerators), worker.holds?.base ? h("span", {}, short(worker.holds.base)) : null) : null,
    h("div", { class: "chips" }, worker.serving.length ? worker.serving.map(name => h("span", { class: "chip", title: `${waiting[name] ?? 0} requests waiting for ${name}` }, name,
      waiting[name] ? h("b", { class: "waits" }, ` ${waiting[name]} waiting`) : null)) : h("span", { class: "none" }, "serves nothing")));
}

// A version's way to the engines, as stages: written, resharded (full weights only), rolling out, serving.
function way(version) {
  const life = version.life, at = { written: 0, resharding: 1, resharded: 1, "rolling out": 2, serving: 3, superseded: 4 }[life.state];
  const names = ["written", life.reshard ? "resharded" : "no reshard", "rolling", "serving"];
  return h("div", { class: "stages way" }, names.map((name, place) => h("div", {
    class: [place < at || (place === at && life.state === "resharded") ? "done" : place === at ? `now${life.state === "resharding" || life.state === "rolling out" ? " active" : ""}` : "",
      place === 1 && !life.reshard ? "skipped" : ""].join(" "),
  }, h("span", {}, place === at && life.state === "resharding" ? "resharding" : name))));
}

function drawPolicies(here) {
  const lineage = state.lineage;
  if (!lineage || state.lineageSample !== here.sample) return [h("div", { class: "empty" }, "Reading the policies…")];
  const index = indexOf(lineage), lanes = lanesOf(lineage, index);
  const versions = lineage.policies.reduce((sum, policy) => sum + policy.versions.length, 0);
  const toggle = h("div", { class: "segmented" },
    link(policiesPlace(false), { class: `seg${here.sample ? "" : " current"}` }, "The ledger"),
    link(policiesPlace(true), { class: `seg${here.sample ? " current" : ""}` }, "With the sample fixture"));
  const head = h("div", { class: "head" }, h("h1", {}, "Policies"),
    h("div", { class: "sub" }, "Each policy a line of versions; forks and distillations between them. Below: what trains them, what serves them, and how they play a fixed suite."),
    specs(spec("policies", String(lineage.policies.length)), spec("versions", String(versions)), spec("runs", String(lineage.runs.length)),
      spec("distillations", String(lineage.runs.filter(run => run.kind === "distill").length), "warm"), spec("trainers", String(lineage.trainers.length), "violet"),
      spec("workers", String(lineage.workers.length), "accent"), spec("suites", String(lineage.evaluations.length))), toggle);
  const notice = here.sample ? h("div", { class: "tile rail warm notice" }, h("header", {}, h("b", {}, "Sample fixture"), sampleChip()),
    h("p", { class: "muted small" }, "Everything marked sample comes from rollout_train/monitor/sample-lineage.json: the tables proposed in docs/research/policy-dag.md (a run's plan, trainers and their queues, resharding, inference workers and their loads, the router's waiting requests, evaluation suites). No run writes them yet. The rest is the ledger's, and the runs' feeds'."))
    : null;
  const opened = lanes.filter(lane => folds[`lane:${lane.key}`]).length;
  const all = open => () => { for (const lane of lanes) folds[`lane:${lane.key}`] = open; keepFolds(); redraw(); };
  const graph = h("section", { class: "card" }, h("header", {}, h("h2", {}, "Lineage"),
    h("span", {}, `a lane per policy, its versions from the left; ${opened ? `${opened} open` : "folded to the versions something points at"} · `,
      h("button", { class: "linkish", onclick: all(true) }, "open all"), " · ", h("button", { class: "linkish", onclick: all(false) }, "fold all"))),
    lanes.length ? lineageGraph(lineage, index, lanes) : h("div", { class: "empty" }, "The ledger has no policy yet."),
    h("div", { class: "legend dag-legend" },
      h("span", {}, h("i", { style: "background:var(--accent)" }), "trained by a run"), h("span", {}, h("i", { style: "background:var(--warm)" }), "distilled off policy"),
      h("span", {}, h("i", { style: "background:var(--violet)" }), "distilled on policy"), h("span", {}, h("i", { class: "rule", style: "background:var(--quiet)" }), "fork"),
      h("span", {}, h("i", { class: "hollow" }), "released (weights let go)"), h("span", {}, h("i", { class: "ring-good" }), "serving"),
      h("span", {}, h("i", { class: "ring-warm" }), "rolling out"), h("span", {}, h("i", { class: "ring-violet" }), "resharding"),
      lineage.evaluations.length ? h("span", {}, h("b", { class: "t-good" }, "9/16"), " solved of the suite played") : null));
  // Training: each trainer, its queue over time and now.
  const training = [h("div", { class: "section-title" }, h("h2", {}, "Trainers"), h("span", {}, "finished groups collect into a step; a step waits in its trainer's queue")),
    h("div", { class: "tiles wide-tiles" }, lineage.trainers.map(trainer => trainerTile(trainer, lineage)))];
  // Serving: every version on its way to the engines or served, the workers, the requests waiting.
  const order = new Map(lanes.map((lane, place) => [lane.key, place]));
  const moving = lineage.policies.flatMap(policy => policy.versions.filter(version => !["superseded", "written"].includes(version.life.state)
    || (version.life.state === "written" && version.name === policy.head)).map(version => ({ ...version, policy })))
    .sort((a, b) => (order.get(a.policy.policy) ?? 0) - (order.get(b.policy.policy) ?? 0) || b.number - a.number);
  const waiting = lineage.routing.waiting ?? {}, totalWaiting = Object.values(waiting).reduce((sum, count) => sum + count, 0);
  const serving = [h("div", { class: "section-title" }, h("h2", {}, "Serving"), h("span", {}, "a request names an exact version, or a run's latest; the router sends it to a worker that has it")),
    h("div", { class: "kpis" },
      kpi("Requests waiting", lineage.routing.history.length ? String(totalWaiting) : "–", lineage.routing.history.length ? "by the version they name" : "the router notes none"),
      kpi("Serving", String(moving.filter(version => version.life.state === "serving").length), "versions on some worker"),
      kpi("Rolling out", String(moving.filter(version => version.life.state === "rolling out").length), "a run's latest, not yet on every worker"),
      kpi("Resharding", String(moving.filter(version => ["resharding", "resharded"].includes(version.life.state)).length), "full weights, for the engines' layout"),
      kpi("Workers", String(lineage.workers.length), `${lineage.workers.filter(worker => worker.registered).length} registered`)),
    card("On their way to the engines, and served", "newest first in each policy",
      table([["version"], ["made by"], ["way"], ["workers"], ["waiting", "n"], ["latest of"], ["made", "n"]], moving.map(version => {
        const workers = Object.entries(version.life.workers);
        return [{ node: h("span", {}, h("b", { class: "mono" }, version.name), version.policy.sample ? " " : null, version.policy.sample ? sampleChip() : null) },
          version.by ? `${version.by.run}${version.by.step ? ` S${version.by.step}` : ""}` : "–", { node: way(version) },
          { node: h("div", { class: "chips" }, workers.length ? workers.map(([worker, served]) => h("span", { class: `chip${served.until == null ? " on" : " off"}`,
            title: served.until == null ? `serving since ${clock(served.since)}` : `served ${clock(served.since)} to ${clock(served.until)}` }, worker))
            : h("span", { class: "none" }, version.life.state === "resharding" ? "files being rewritten" : "on no worker")) },
          version.life.waiting ? { text: String(version.life.waiting), kind: "bad" } : "0", version.life.latest_of ?? "–", ago(lineage, version.made)];
      }))),
    h("div", { class: "tiles" }, lineage.workers.map(worker => workerTile(worker, lineage)))];
  // Runs, and what each was set up to do.
  const runs = card("Runs and distillations", "what made each stretch of a line", table([["run"], ["kind"], ["policy"], ["from"], ["teachers"], ["trains on"], ["objective"], ["versions"], ["latest"]],
    lineage.runs.map(run => [{ node: h("span", {}, h("b", { class: "mono" }, run.run), run.sample ? " " : null, run.sample ? sampleChip() : null) },
      run.kind === "distill" ? { node: mark(modeKind(run.mode) === "violet" ? "stepping" : "queued", `distil, ${run.mode}`) } : "train",
      run.policy, run.from ?? "–", run.teachers.join(", ") || "–",
      run.kind === "distill" ? `${(run.data.sampled_by ?? []).join(", ")}'s samples${run.data.runs ? ` from ${run.data.runs.join(", ")}` : ""}${run.data.episodes ? ` (${run.data.episodes})` : ""}` : "its own groups",
      run.objective ?? "policy gradient", run.versions.length ? `${versionOf(run.versions[0])}–${versionOf(run.versions.at(-1))}` : "–", run.latest ?? "–"]),
    lineage.runs.map(run => state.system.runs.some(each => each.run === run.run) && !run.sample ? () => go(runPlace(run.run)) : null)));
  // Evaluations: each suite, a column for each subject that played it.
  const suites = lineage.evaluations.map(suite => {
    const subjects = [...suite.subjects].sort((a, b) => {
      const place = subject => subject.version ? (order.get(subject.version.split("@")[0]) ?? 99) * 1000 + Number(subject.version.split("@")[1]) : 1e9;
      return place(a) - place(b);
    });
    const header = h("tr", {}, h("th", {}, "start"), subjects.map(subject => h("th", { class: "subject" },
      h("div", {}, subject.version ?? subject.model ?? subject.subject), h("small", {}, subject.kind === "model" ? short(subject.subject.split(".").at(-1)) : subject.asked_by === "by hand" ? "" : "scheduled"))));
    const total = h("tr", { class: "total" }, h("td", {}, "solved"), subjects.map(subject => h("td", { class: "n" },
      h("b", {}, `${subject.solved}/${subject.played}`), subject.played < suite.starts.length ? h("small", { class: "faint" }, ` of ${suite.starts.length}`) : null,
      h("div", { class: "track" }, h("i", { style: `width:${(100 * subject.solved / Math.max(1, suite.starts.length)).toFixed(1)}%` })))));
    const rows = suite.starts.map(start => h("tr", {}, h("td", { class: "key", title: start.title ?? "" }, `${start.task} · ${start.seed}`),
      subjects.map(subject => {
        const played = subject.results[start.start] ?? [];
        return h("td", { class: "cell-result" }, played.length ? played.map(each => h("i", { class: each.solved ? "solved" : "unsolved", title: `${subject.subject} on ${start.task} seed ${start.seed}: reward ${figure(each.reward)}${each.solved ? ", solved" : ""}` }))
          : h("i", { class: "unplayed", title: "not played yet" }));
      })));
    return h("section", { class: "card" }, h("header", {}, h("h2", {}, `Evaluation · ${suite.suite}`), h("span", {}, `${suite.starts.length} fixed starts (row and seed), played by ${subjects.length} subjects`, suite.sample ? " " : null, suite.sample ? sampleChip() : null)),
      h("div", { class: "body" }, h("div", { class: "table" }, h("table", { class: "evals" }, header, total, rows)),
        h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--good)" }), "solved"), h("span", {}, h("i", { style: "background:var(--line-strong)" }), "not solved"),
          h("span", {}, h("i", { class: "hollow" }), "not played yet"))));
  });
  return [head, notice, graph, ...training, ...serving, runs,
    h("div", { class: "section-title" }, h("h2", {}, "Evaluations"), h("span", {}, lineage.evaluations.length ? "a fixed suite of starts, played by versions and by other models" : "")),
    ...(suites.length ? suites : [h("div", { class: "empty" }, "No evaluation suite is in the ledger.")])];
}

// Every run of the ledger: running ones first, each with how it is going.
function drawRuns() {
  const system = state.system, others = state.runs.filter(run => !run.labels.run);
  const tiles = system.runs.map(run => {
    const recent = run.done.slice(-12), solved = recent.flatMap(line => line.solved);
    const committed = run.steps.filter(step => step.state === "committed").length;
    const head = run.steps.findLast(step => step.state === "committed")?.makes;
    const trains = head?.split("@")[0] ?? run.policy;
    return link(runPlace(run.run), { class: `tile rail ${run.state === "running" ? "good" : run.state === "idle" ? "warm" : ""}` },
      h("header", {}, runDot(run), h("b", {}, run.run), h("span", { class: "what" }, trains ? `trains ${trains}` : ""), h("span", { class: "faint small" }, running(run))),
      h("div", { class: "cells four" },
        h("div", { class: `cell ${run.open.length ? "accent" : "waiting"}` }, h("span", {}, "in flight"), h("b", {}, String(run.open.length)), h("small", {}, `${run.next.length} toward a step`)),
        h("div", { class: "cell" }, h("span", {}, "groups done"), h("b", {}, String(run.done.length)), h("small", {}, `of ${run.decided} decided`)),
        h("div", { class: "cell violet" }, h("span", {}, "steps"), h("b", {}, String(committed)), h("small", {}, head ? versionOf(head) : "none yet")),
        h("div", { class: `cell ${solved.length ? "good" : ""}` }, h("span", {}, "solved"), h("b", {}, solved.length ? `${Math.round(100 * solved.filter(Boolean).length / solved.length)}%` : "–"),
          h("small", {}, `of the last ${recent.length} groups`))),
      run.done.length > 1 ? h("div", {}, spark(run.done.map(line => mean(line.rewards) ?? 0), "s-accent", 420, 40, true),
        h("div", { class: "small muted" }, "each group's mean reward, in order")) : null,
      h("div", { class: "facts" }, h("span", {}, wrote(run)), run.host ? h("span", {}, "on ", h("b", {}, run.host)) : null,
        run.episodes_at === "here" ? null : h("span", { class: run.reached === false || !run.episodes_at ? "t-warm" : "" }, run.episodes_at ? `episodes on ${run.episodes_at}` : "ledger only")));
  });
  const states = ["running", "idle", "ended"].map(name => [name, system.runs.filter(run => run.state === name).length]).filter(([, count]) => count);
  return [h("div", { class: "head" }, h("h1", {}, "Runs"),
    h("div", { class: "sub" }, "The runs in the ledger, running ones first. A run opens its steps, groups, episodes and rollouts."),
    specs(spec("ledger", system.ledger_at), ...states.map(([name, count]) => spec(name, String(count), name === "running" ? "good" : name === "idle" ? "warm" : "")),
      spec("this host", system.host))),
  system.runs.length ? h("div", { class: "tiles wide-tiles" }, tiles) : h("div", { class: "empty" }, "The ledger has no run yet."),
  others.length ? card("Episodes outside a run", `${others.length} in the feeds`, h("p", { class: "muted small", style: "margin:0" },
    link("#/episodes", { class: "linkish" }, "Evaluations, tests and programs run by hand"), " that no training run asked for.")) : null];
}

// Charts across runs, drawn to scale: round ticks on one axis each way, a line or column for each run in its own
// color, and under the pointer a rule with every series' value there.
const DAY = 86400;
const TIME_STEPS = [60, 300, 900, 1800, 3600, 7200, 10800, 21600, 43200, DAY, 2 * DAY, 7 * DAY];
const day = at => new Date(at * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
const tick = value => Math.abs(value) >= 1e6 ? `${+(value / 1e6).toPrecision(3)}M` : Math.abs(value) >= 1e4 ? `${+(value / 1e3).toPrecision(3)}k` : String(+value.toPrecision(3));
const percent = value => `${Math.round(100 * value)}%`;
const tickSpan = seconds => seconds < 120 ? `${Math.round(seconds)} s` : seconds < 7200 ? `${+(seconds / 60).toPrecision(2)} min` : `${+(seconds / 3600).toPrecision(2)} h`;

// A scale from the values to round ends, with its ticks: `zero` takes it down (or up) to 0; `min` and `max` fix an end.
function scale(values, { zero = true, min, max, count = 4 } = {}) {
  const finite = values.filter(Number.isFinite);
  let low = min ?? Math.min(...finite, ...(zero ? [0] : [])), high = max ?? Math.max(...finite, ...(zero ? [0] : []));
  if (!finite.length && min == null && max == null) [low, high] = [0, 1];
  if (!(high > low)) { const pad = Math.abs(high) * 0.1 || 1; if (min == null) low -= zero && low === 0 ? 0 : pad; if (max == null) high += pad; }
  const raw = (high - low) / count, power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(each => each * power).find(each => each >= raw * 0.999);
  if (min == null) low = Math.floor(low / step + 1e-9) * step;
  if (max == null) high = Math.ceil(high / step - 1e-9) * step;
  const ticks = [];
  for (let value = Math.ceil(low / step - 1e-9) * step; value <= high + step * 1e-6; value += step) ticks.push(+value.toPrecision(12));
  return { low, high, ticks, step };
}
// Ticks on local clock times (or days), about `count` of them.
function timeTicks(low, high, count) {
  const step = TIME_STEPS.find(each => (high - low) / each <= count) ?? 7 * DAY, offset = new Date(low * 1000).getTimezoneOffset() * 60;
  const ticks = [];
  for (let at = Math.ceil((low - offset) / step) * step + offset; at <= high; at += step) ticks.push([at, step >= DAY || (at - offset) % DAY === 0 ? day(at) : clock(at)]);
  return ticks;
}
const plain = value => figure(value);

// A line for each series ({name, color, points: [[x, y]], until?, scatter?}) over x (a group's place, a step, or a
// time with `time`), on one y scale (`y`: as `scale` takes it). `stepped` holds each value until the next (a count);
// `rules` are reference lines ({value, label}).
function lineChart({ series, width, height = 210, time = false, y = {}, stepped = false, rules = [], label, format = plain, xFormat = plain, yTick, dots, gap = Infinity }) {
  const left = 50, right = 14, top = 12, bottom = 26;
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": label, class: "chart" });
  const shown = series.filter(each => each.points.length);
  const points = shown.flatMap(each => [...each.points, ...(each.scatter ?? [])]).filter(point => Number.isFinite(point[1]));
  if (!points.length) {
    drawing.append(svg("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, "nothing to draw yet"));
    return drawing;
  }
  let x0 = Math.min(...points.map(point => point[0])), x1 = Math.max(...points.map(point => point[0]), ...shown.map(each => each.until ?? -Infinity));
  if (!(x1 > x0)) { x0 -= time ? 1800 : 1; x1 += time ? 1800 : 1; }
  const ys = scale([...points.map(point => point[1]), ...rules.map(rule => rule.value)], y);
  const X = value => left + (value - x0) / (x1 - x0) * (width - left - right), Y = value => top + (ys.high - value) / (ys.high - ys.low) * (height - top - bottom);
  const decimals = Math.max(0, -Math.floor(Math.log10(ys.step) + 1e-9));  // (as many as the ticks' step needs)
  const labelOf = yTick ?? (value => Math.abs(value) >= 1e4 ? tick(value) : value.toFixed(decimals));
  for (const value of ys.ticks) {
    drawing.append(svg("line", { x1: left, x2: width - right, y1: Y(value), y2: Y(value), class: "s-grid" }));
    drawing.append(svg("text", { x: left - 7, y: Y(value) + 3.5, "text-anchor": "end" }, labelOf(value)));
  }
  const across = Math.max(2, Math.floor((width - left - right) / (time ? 96 : 64)));
  const xTicks = time ? timeTicks(x0, x1, across) : scale([x0, x1], { zero: false, min: x0, max: x1, count: across }).ticks.filter(Number.isInteger).map(value => [value, xFormat(value)]);
  for (const [value, text] of xTicks) {
    drawing.append(svg("line", { x1: X(value), x2: X(value), y1: height - bottom, y2: height - bottom + 4, class: "s-grid" }));
    drawing.append(svg("text", { x: X(value), y: height - bottom + 15, "text-anchor": "middle" }, text));
  }
  drawing.append(svg("line", { x1: left, x2: width - right, y1: height - bottom + 0.5, y2: height - bottom + 0.5, class: "s-axis" }));
  for (const rule of rules) {
    drawing.append(svg("line", { x1: left, x2: width - right, y1: Y(rule.value), y2: Y(rule.value), class: "s-rule" }));
    drawing.append(svg("text", { x: width - right - 2, y: Y(rule.value) - 4, "text-anchor": "end" }, rule.label));
  }
  for (const each of shown) {
    for (const [x, value] of each.scatter ?? []) drawing.append(svg("circle", { cx: X(x), cy: Y(value), r: 2.2, style: `fill:${each.color}`, class: "scatter" }));
    const line = each.points.filter(point => Number.isFinite(point[1]));
    if (!line.length) continue;
    let path = `M ${X(line[0][0])} ${Y(line[0][1])}`;
    line.slice(1).forEach(([x, value], place) => {  // (a line breaks where measurements stopped for longer than `gap`)
      path += stepped ? ` H ${X(x)} V ${Y(value)}` : `${x - line[place][0] > gap ? " M" : " L"} ${X(x)} ${Y(value)}`;
    });
    if (stepped && each.until != null && each.until > line.at(-1)[0]) path += ` H ${X(each.until)}`;
    drawing.append(svg("path", { d: path, class: "series", style: `stroke:${each.color}` }));
    if (dots ?? line.length <= 40) for (const [x, value] of line) drawing.append(svg("circle", { cx: X(x), cy: Y(value), r: 3, class: "dot-series", style: `fill:${each.color}` }));
    else drawing.append(svg("circle", { cx: X(line.at(-1)[0]), cy: Y(line.at(-1)[1]), r: 3, class: "dot-series", style: `fill:${each.color}` }));
  }
  hover(drawing, { width, height, left, right, top, bottom, x0, x1, X, Y, time, format, xFormat, stepped,
    series: shown.map(each => ({ ...each, points: each.points.filter(point => Number.isFinite(point[1])) })) });
  return drawing;
}

// Under the pointer: a rule at the nearest x any series has, and a label with each series' value there (a stepped
// series: its value then).
function hover(drawing, { width, height, left, right, top, bottom, x0, x1, X, Y, time, format, xFormat, stepped, series }) {
  const layer = svg("g", { class: "hover", "pointer-events": "none" });
  const xs = [...new Set(series.flatMap(each => each.points.map(point => point[0])))].sort((a, b) => a - b);
  const valueAt = (each, x) => {
    if (stepped) { const before = each.points.filter(point => point[0] <= x).at(-1); return before && (each.until == null || x <= each.until || x === before[0]) ? before[1] : null; }
    return each.points.find(point => point[0] === x)?.[1] ?? null;
  };
  const area = svg("rect", { x: left, y: top, width: width - left - right, height: height - top - bottom, class: "f-none" });
  area.addEventListener("mousemove", event => {
    const box = drawing.getBoundingClientRect(), at = x0 + ((event.clientX - box.left) * width / box.width - left) / (width - left - right) * (x1 - x0);
    const x = xs.reduce((best, each) => Math.abs(each - at) < Math.abs(best - at) ? each : best, xs[0]);
    const rows = series.map(each => [each, valueAt(each, x)]).filter(([, value]) => value != null);
    layer.replaceChildren(svg("line", { x1: X(x), x2: X(x), y1: top, y2: height - bottom, class: "s-cross" }));
    for (const [each, value] of rows) layer.append(svg("circle", { cx: X(x), cy: Y(value), r: 4, class: "dot-series", style: `fill:${each.color}` }));
    const lines = [[null, time ? `${day(x)} ${clock(x)}` : xFormat(x)], ...rows.map(([each, value]) => [each, `${each.name}  ${format(value)}`])];
    const wide = Math.max(...lines.map(([, text]) => text.length)) * 6.3 + 26, tall = lines.length * 15 + 8;
    const tipX = X(x) + 10 + wide > width - right ? X(x) - 10 - wide : X(x) + 10, tipY = top + 2;
    layer.append(svg("rect", { x: tipX, y: tipY, width: wide, height: tall, rx: 3, class: "tip" }));
    lines.forEach(([each, text], place) => {
      if (each) layer.append(svg("rect", { x: tipX + 8, y: tipY + 9 + place * 15, width: 8, height: 8, rx: 1, style: `fill:${each.color}` }));
      layer.append(svg("text", { x: tipX + (each ? 21 : 8), y: tipY + 16 + place * 15, class: each ? "tip-text" : "tip-head" }, text));
    });
  });
  area.addEventListener("mouseleave", () => layer.replaceChildren());
  drawing.append(layer, area);
}

// Columns over time, one for each span of `step` seconds, stacked by series (a value per series in each), on one
// scale; each column says what it holds when pointed at.
function columnChart({ spans, series, width, height = 190, label, format = plain }) {
  const left = 50, right = 14, top = 12, bottom = 26;
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": label, class: "chart" });
  if (!spans.length) {
    drawing.append(svg("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, "nothing to draw yet"));
    return drawing;
  }
  const x0 = spans[0].from, x1 = spans.at(-1).to;
  const ys = scale(spans.map(each => each.values.reduce((sum, value) => sum + value, 0)), {});
  const X = value => left + (value - x0) / (x1 - x0) * (width - left - right), Y = value => top + (ys.high - value) / (ys.high - ys.low) * (height - top - bottom);
  for (const value of ys.ticks) {
    drawing.append(svg("line", { x1: left, x2: width - right, y1: Y(value), y2: Y(value), class: "s-grid" }));
    drawing.append(svg("text", { x: left - 7, y: Y(value) + 3.5, "text-anchor": "end" }, tick(value)));
  }
  for (const [value, text] of timeTicks(x0, x1, Math.max(2, Math.floor((width - left - right) / 96)))) {
    drawing.append(svg("line", { x1: X(value), x2: X(value), y1: height - bottom, y2: height - bottom + 4, class: "s-grid" }));
    drawing.append(svg("text", { x: X(value), y: height - bottom + 15, "text-anchor": "middle" }, text));
  }
  for (const each of spans) {
    const x = X(each.from) + 1, wide = Math.max(1, X(each.to) - X(each.from) - 2);
    let base = 0;
    const column = svg("g", { class: "column-stack" });
    each.values.forEach((value, place) => {
      if (!value) return;
      const y1 = Y(base), y2 = Y(base + value);
      column.append(svg("rect", { x, y: y2, width: wide, height: Math.max(1, y1 - y2 - (base ? 1 : 0)), style: `fill:${series[place].color}` }));
      base += value;
    });
    column.append(svg("rect", { x: X(each.from), y: top, width: X(each.to) - X(each.from), height: height - top - bottom, class: "f-none" },
      svg("title", {}, [`${day(each.from)} ${clock(each.from)} to ${clock(each.to)}`, ...series.map((one, place) => each.values[place] ? `${one.name}: ${format(each.values[place])}` : null).filter(Boolean)].join("\n"))));
    drawing.append(column);
  }
  drawing.append(svg("line", { x1: left, x2: width - right, y1: height - bottom + 0.5, y2: height - bottom + 0.5, class: "s-axis" }));
  return drawing;
}

// A legend for two series or more (one needs none: the title names it).
const legendOf = series => series.length > 1 ? h("div", { class: "legend" }, series.map(each => h("span", {}, h("i", { style: `background:${each.color}` }), each.name))) : null;
// A count of things that happened at times, in spans of a round length (about `count` of them between `from` and `to`),
// for each series.
function spansOf(times, from, to, count = 48) {
  const step = TIME_STEPS.find(each => (to - from) / each <= count) ?? 7 * DAY, offset = new Date(from * 1000).getTimezoneOffset() * 60;
  const start = Math.floor((from - offset) / step) * step + offset, spans = [];
  for (let at = start; at <= to; at += step) spans.push({ from: at, to: at + step, values: times.map(() => 0) });
  times.forEach((list, place) => { for (const at of list) { const bucket = spans[Math.floor((at - start) / step)]; if (bucket) bucket.values[place] += 1; } });
  return { spans, step };
}
const solvedShare = groups => {
  const solved = groups.reduce((sum, group) => sum + group.solved.filter(Boolean).length, 0), played = groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0);
  return played ? solved / played : null;
};
const share = (value, kind = "") => ({ node: value == null ? h("span", { class: "faint" }, "–")
  : h("span", { class: "share" }, h("span", { class: `track ${kind}` }, h("i", { style: `width:${(100 * value).toFixed(1)}%` })), h("b", {}, percent(value))) });
const STEP_FIGURES = [["kl_moved", "KL moved", "from the version before"], ["kl_floor", "KL floor", "the update's noise floor"], ["clip_fraction", "Clip fraction", "of tokens clipped"],
  ["mean_mismatch", "Mean mismatch", "sampler against trainer"], ["mean_weight", "Mean weight", "the off-policy correction"], ["truncated_fraction", "Truncated fraction", "of weights truncated"],
  ["loss", "Loss", "the objective"], ["seconds", "Step time", "the update, start to end"], ["start_seconds", "Start time", "before the first optimizer step"]];

function drawStatistics(here) {
  const figures = state.statistics, system = state.system;
  if (!figures) return [h("div", { class: "empty" }, "Reading the statistics…")];
  const main = document.getElementById("main"), frame = Math.min(1400, main.clientWidth - 64);
  const whole = Math.max(280, frame - 38), half = frame >= 980 ? Math.floor((frame - 22) / 2 - 38) : whole;
  const runs = figures.runs.filter(run => !hidden.has(run.run)).sort((a, b) => a.run.localeCompare(b.run));
  const colored = run => ({ name: run.run, color: runColor(run.run) });
  const life = new Map(system.runs.map(run => [run.run, run]));
  const done = run => run.groups.filter(group => group.time != null).sort((a, b) => a.time - b.time);
  const every = runs.flatMap(done);
  const title = (key, name, note) => h("div", { class: "section-title", id: `section-${key}` }, h("h2", {}, name), h("span", {}, note));
  if (!runs.length) return [h("div", { class: "head" }, h("h1", {}, "Statistics")), h("div", { class: "empty" }, figures.runs.length ? "Every run is left out: choose some in the sidebar." : "The ledger has no run yet."), ...machine(half)];

  // Totals
  const episodes = every.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), failed = every.reduce((sum, group) => sum + group.failed, 0);
  const lastDay = every.filter(group => group.time > figures.now - DAY);
  const steps = runs.flatMap(run => run.steps);
  const kpis = h("div", { class: "kpis" },
    kpi("Runs", String(runs.length), `${runs.filter(run => life.get(run.run)?.state === "running").length} running`),
    kpi("Groups done", every.length.toLocaleString(), `${runs.reduce((sum, run) => sum + run.groups.filter(group => group.time == null).length, 0)} in flight`),
    kpi("Episodes", episodes.toLocaleString(), `${failed} failed`),
    kpi("Solved", every.length ? percent(solvedShare(every)) : "–", "of the episodes played"),
    kpi("Steps", String(steps.filter(step => step.state === "committed").length), `${steps.filter(step => step.state === "failed").length} failed`),
    kpi("Last day", `${lastDay.length} groups`, `${lastDay.reduce((sum, group) => sum + group.rewards.length + group.failed, 0)} episodes`));
  const head = h("div", { class: "head" }, h("h1", {}, "Statistics"),
    h("div", { class: "sub" }, "Each run in its own color, read from the ledger (and from a run's feed for its engines). Leave runs out in the sidebar."));

  // Outcomes over groups: a rolling solve rate and mean reward, a line for each run.
  const stretch = Number(stored("monitor.window", 8));
  const rolling = (groups, value) => groups.map((group, place) => {
    const span = groups.slice(Math.max(0, place + 1 - stretch), place + 1);
    return [place + 1, value(span)];
  });
  const meanReward = groups => mean(groups.flatMap(group => group.rewards));
  const windows = h("div", { class: "segmented" }, [4, 8, 16, 32].map(size => h("button", { class: `seg${size === stretch ? " current" : ""}`,
    onclick: () => { keep("monitor.window", size); redraw(); } }, `${size} groups`)));
  const solveSeries = runs.map(run => ({ ...colored(run), points: rolling(done(run), solvedShare), scatter: done(run).map((group, place) => [place + 1, solvedShare([group])]) }));
  const rewardSeries = runs.map(run => ({ ...colored(run), points: rolling(done(run), meanReward), scatter: done(run).map((group, place) => [place + 1, mean(group.rewards)]) }));
  const outcomes = [title("outcomes", sectionName("outcomes"), `groups in the order their results were written; the line the mean of the last ${stretch}, a dot each group`),
    windows, h("div", { class: "cols" },
      card("Solve rate", "the share of a group's episodes that solved its row", lineChart({ series: solveSeries, width: half, label: "solve rate over groups", y: { min: 0, max: 1 }, yTick: percent, format: percent, xFormat: value => `group ${value}`, dots: false }), legendOf(runs.map(colored))),
      card("Mean reward", "of the episodes that completed; each row's rewards are its own", lineChart({ series: rewardSeries, width: half, label: "mean reward over groups", format: value => figure(value), xFormat: value => `group ${value}`, dots: false }), legendOf(runs.map(colored))))];

  // Results by row
  const rows = new Map();
  for (const run of runs) for (const group of done(run)) {
    const row = rows.get(group.task) ?? { task: group.task, title: group.title, runs: new Set(), groups: [], best: null };
    row.runs.add(run.run); row.groups.push({ ...group, run: run.run });
    for (const reward of group.rewards) row.best = row.best == null ? reward : Math.max(row.best, reward);
    rows.set(group.task, row);
  }
  const byRow = [...rows.values()].sort((a, b) => String(a.task).localeCompare(String(b.task), undefined, { numeric: true }));
  const rowTable = card("Rows", `${byRow.length} rows; the last rewards are of the row's newest group`,
    table([["row"], ["title"], ...(runs.length > 1 ? [["runs", "n"]] : []), ["groups", "n"], ["episodes", "n"], ["solved"], ["mean", "n"], ["best", "n"], ["last rewards"], ["last", "n"]],
      byRow.map(row => {
        const last = row.groups.at(-1);
        return [{ text: row.task ?? "–", kind: "key" }, { node: h("span", { class: "muted" }, row.title ?? "") }, ...(runs.length > 1 ? [row.runs.size] : []), row.groups.length,
          row.groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), share(solvedShare(row.groups), "good"),
          figure(meanReward(row.groups)), figure(row.best), { node: h("span", {}, dotsOf(last), " ", h("span", { class: "faint small" }, last.rewards.map(figure).join(" "))) },
          `${span(figures.now - last.time)} ago`];
      }),
      byRow.map(row => { const last = row.groups.at(-1); return () => go(groupPlace(last.run, last.group)); })));
  const named = every.filter(group => group.names != null);
  const sizes = [...new Set(named.map(group => group.names))].sort((a, b) => a - b);
  const bySize = named.length ? card("By the names a start gives", "groups whose start lists names, by how many",
    table([["names", "n"], ["groups", "n"], ["episodes", "n"], ["solved"], ["mean reward", "n"]], sizes.map(size => {
      const groups = named.filter(group => group.names === size);
      return [size, groups.length, groups.reduce((sum, group) => sum + group.rewards.length + group.failed, 0), share(solvedShare(groups), "good"), figure(meanReward(groups))];
    }))) : null;

  // Steps: each statistic of the update over the steps, a line for each run.
  const present = STEP_FIGURES.filter(([key]) => steps.some(step => step.metrics[key] != null || (key === "seconds" && step.metrics.update_seconds != null)));
  const small = Math.max(260, Math.floor((frame - 14 * (Math.max(1, Math.floor((frame + 14) / 334)) - 1)) / Math.max(1, Math.floor((frame + 14) / 334)) - 34));
  const valueOf = (step, key) => key === "seconds" ? step.metrics.update_seconds ?? step.metrics.seconds : step.metrics[key];
  const multiples = h("div", { class: "multiples" }, present.map(([key, name, note]) => {
    const seconds = key.endsWith("seconds");
    return h("section", { class: "card small-card" }, h("header", {}, h("h2", {}, name), h("span", {}, note)), h("div", { class: "body" },
      lineChart({ series: runs.map(run => ({ ...colored(run), points: run.steps.filter(step => valueOf(step, key) != null).map(step => [step.step, valueOf(step, key)]) })),
        width: small, height: 150, label: `${name} over steps`, y: { zero: seconds || ["clip_fraction", "truncated_fraction", "kl_moved"].includes(key) },
        yTick: seconds ? tickSpan : undefined, format: seconds ? span : value => Number(value).toPrecision(4), xFormat: value => `step ${value}` })));
  }));

  // Pace: episodes and groups an hour, and what was done with each group.
  const from = Math.min(...every.map(group => group.time)), to = Math.max(...every.map(group => group.time)) + 1;
  const paced = count => spansOf(runs.map(run => done(run).flatMap(group => Array.from({ length: count(group) }, () => group.time))), from, to);
  const episodeSpans = every.length ? paced(group => group.rewards.length + group.failed) : { spans: [], step: 3600 };
  const groupSpans = every.length ? paced(() => 1) : { spans: [], step: 3600 };
  const perHour = (spans, step) => spans.map(each => ({ ...each, values: each.values.map(value => value * 3600 / step) }));
  const kinds = [["trained", "trained on", "var(--accent)"], ["stepping", "in a step being taken", "var(--violet)"], ["waiting", "waiting for a step", "var(--warm)"],
    ["skipped", "nothing to train on", "var(--line-strong)"], ["failed", "no episode, or its step failed", "var(--bad)"], ["flight", "in flight", "var(--faint)"]];
  const kindOf = group => group.time == null ? "flight" : group.trained === "committed" ? "trained" : group.trained === "failed" || !group.rewards.length ? "failed"
    : group.trained === "stepping" ? "stepping" : group.segments ? "waiting" : "skipped";
  const fates = h("div", { class: "fates" }, runs.map(run => {
    const counts = new Map(kinds.map(([key]) => [key, 0]));
    for (const group of run.groups) counts.set(kindOf(group), counts.get(kindOf(group)) + 1);
    return h("div", { class: "fate" }, h("span", { class: "fate-name" }, h("i", { class: "swatch", style: `background:${runColor(run.run)}` }), run.run),
      h("div", { class: "stack" }, kinds.filter(([key]) => counts.get(key)).map(([key, name, color]) => h("i", { style: `flex:${counts.get(key)};background:${color}`, title: `${name}: ${counts.get(key)}` }))),
      h("span", { class: "fate-count" }, kinds.filter(([key]) => counts.get(key)).map(([key]) => `${counts.get(key)} ${key}`).join(" · ")));
  }), h("div", { class: "legend" }, kinds.map(([, name, color]) => h("span", {}, h("i", { style: `background:${color}` }), name))));
  const reasons = new Map();
  for (const run of runs) for (const group of done(run)) if (group.skipped) reasons.set(group.skipped, (reasons.get(group.skipped) ?? 0) + 1);
  const pace = [title("pace", "Pace", "counted when each group's result was written"),
    h("div", { class: "cols" },
      card("Episodes an hour", `in spans of ${tickSpan(episodeSpans.step)}`, columnChart({ spans: perHour(episodeSpans.spans, episodeSpans.step), series: runs.map(colored), width: half, label: "episodes an hour", format: value => `${figure(+value.toFixed(1))} an hour` }), legendOf(runs.map(colored))),
      card("Groups an hour", `in spans of ${tickSpan(groupSpans.step)}`, columnChart({ spans: perHour(groupSpans.spans, groupSpans.step), series: runs.map(colored), width: half, label: "groups an hour", format: value => `${figure(+value.toFixed(2))} an hour` }), legendOf(runs.map(colored)))),
    h("div", { class: "cols" },
      card("What was done with each group", "every group decided, by run", fates),
      card("Why groups gave nothing to train on", `${[...reasons.values()].reduce((sum, count) => sum + count, 0)} groups skipped`,
        reasons.size ? table([["reason"], ["groups", "n"]], [...reasons].sort((a, b) => b[1] - a[1]).map(([reason, count]) => [reason, count])) : h("div", { class: "empty" }, "None skipped.")))];

  // In flight and waiting, over time.
  const until = run => life.get(run.run)?.state === "running" ? figures.now : run.wrote;
  const queue = [title("queue", sectionName("queue"), "how many groups were being played, and how many waited for a step, over time"),
    h("div", { class: "cols" },
      card("Groups in flight", "decided, their result not written", lineChart({ series: runs.map(run => ({ ...colored(run), points: run.flight.map(point => [point[0], point[1]]), until: until(run) })), width: half, time: true, stepped: true, label: "groups in flight", format: String, dots: false }), legendOf(runs.map(colored))),
      card("Groups waiting for a step", "recorded with something to train on, no step begun over them", lineChart({ series: runs.map(run => ({ ...colored(run), points: run.flight.map(point => [point[0], point[2]]), until: until(run) })), width: half, time: true, stepped: true, label: "groups waiting for a step", format: String, dots: false }), legendOf(runs.map(colored))))];

  // Inference, from each run's feed.
  const measured = runs.filter(run => run.inference.length);
  const channelSeries = (place, scaleBy) => measured.flatMap(run => run.inference.map(channel => ({ name: run.inference.length > 1 ? `${run.run} · ${channel.channel}` : run.run, color: runColor(run.run),
    points: channel.points.map(point => [point[0], scaleBy(point[place])]) })));
  const quiet = 10 * 60 * Math.max(1, ...measured.flatMap(run => run.inference.map(channel => channel.every)));  // (engines idle that long: no line across)
  const inference = [title("inference", "Inference", "each run's engines, as its feed has them: a measurement a minute while they are busy"),
    measured.length ? h("div", { class: "cols" },
      card("Tokens a second", "generated, across requests", lineChart({ series: channelSeries(1, value => value), width: half, time: true, label: "tokens a second", format: value => `${figure(value)} tok/s`, dots: false, gap: quiet }), legendOf(channelSeries(1, value => value))),
      card("Requests at once", "on average over each minute", lineChart({ series: channelSeries(2, value => value), width: half, time: true, label: "requests at once", format: value => figure(value), dots: false, gap: quiet }), legendOf(channelSeries(2, value => value))))
      : h("div", { class: "empty" }, "No feed of these runs has a measurement of its engines (a run's feed is read where its directory is on this machine).")];

  return [head, kpis, ...outcomes, title("rows", sectionName("rows"), "every row the runs drawn have played"), rowTable, bySize,
    title("steps", "Steps", "the trainer's statistics, a point for each step"), present.length ? multiples : h("div", { class: "empty" }, "No step has made a version yet."),
    ...pace, ...queue, ...inference, ...machine(half)];
}
const stored = (key, otherwise) => { try { return localStorage.getItem(key) ?? otherwise; } catch { return otherwise; } };
const keep = (key, value) => { try { localStorage.setItem(key, String(value)); } catch { /* (a browser that keeps nothing) */ } };

// The machine this monitor is on (memory, accelerators and disk, now and over the last hours), the engines, the
// jobs, the fences and the tables.
function machine(half) {
  const system = state.system, now = system.machine.now, history = system.machine.history;
  const gib = value => value / 2 ** 30;
  const accelerators = now.accelerators.map((each, place) => ({ name: each.name, color: `var(--series-${place + 1})`, place }));
  const memory = card("Memory in use", now.memory.total ? `of ${bytes(now.memory.total)}` : "not measured", lineChart({ width: half, time: true, label: "memory in use", dots: false,
    series: [{ name: "in use", color: "var(--series-1)", points: history.filter(each => each.memory.total).map(each => [each.at, gib(each.memory.total - each.memory.available)]) }],
    rules: now.memory.total ? [{ value: gib(now.memory.total), label: "total" }] : [], yTick: value => `${tick(value)} GiB`, format: value => `${value.toFixed(1)} GiB` }));
  const gpu = accelerators.length ? card("Accelerator memory in use", accelerators.map(each => each.name).join(", "), lineChart({ width: half, time: true, label: "accelerator memory in use", dots: false,
    series: accelerators.map(each => ({ ...each, points: history.filter(one => one.accelerators[each.place]).map(one => [one.at, gib(one.accelerators[each.place].used)]) })),
    rules: [{ value: gib(now.accelerators[0].total), label: "total" }], yTick: value => `${tick(value)} GiB`, format: value => `${value.toFixed(1)} GiB` }), legendOf(accelerators)) : null;
  const busy = accelerators.length ? card("Accelerators busy", "the share of time a kernel ran", lineChart({ width: half, time: true, label: "accelerators busy", dots: false, y: { min: 0, max: 1 }, yTick: percent, format: percent,
    series: accelerators.map(each => ({ ...each, points: history.filter(one => one.accelerators[each.place]).map(one => [one.at, one.accelerators[each.place].busy]) })) }), legendOf(accelerators)) : null;
  const meters = card("Machine", `where this monitor runs: ${system.host}`,
    now.memory.total ? meter("memory", now.memory.total - now.memory.available, now.memory.total, `${bytes(now.memory.available)} available of ${bytes(now.memory.total)}`) : null,
    now.accelerators.map(each => meter(each.name, each.used, each.total, `${bytes(each.used)} of ${bytes(each.total)} · ${Math.round(100 * each.busy)}% busy`)),
    now.disk ? meter("disk", now.disk.total - now.disk.free, now.disk.total, `${bytes(now.disk.free)} free`) : null,
    h("p", { class: "small muted", style: "margin:4px 0 0" }, `Measured every 15 s while this monitor runs; the newest ${history.length} cover ${span(now.at - (history[0]?.at ?? now.at))}.`));
  const channels = system.channels.map(channel => {
    const latest = channel.throughput.at(-1);
    return card(`Channel ${channel.channel}`, `${channel.directory.split("/").at(-1)} · ${channel.adapter ? `serving ${channel.adapter} since ${clock(channel.published)}` : "serving the base model"}`,
      latest ? [h("div", { class: "kpis", style: "margin-bottom:12px" }, kpi("tokens a second", figure(latest.tokens_per_second)),
        kpi("requests at once", figure(latest.mean_concurrency)), kpi("each", `${figure(latest.tokens_per_second_per_stream)} tok/s`))]
        : h("p", { class: "muted" }, "No request has been measured yet."));
  });
  const runners = system.runners.map(runner => card(`Runner ${runner.runner}`, runner.last ? `last claimed ${span(Math.max(0, system.at - runner.last))} ago` : "has claimed nothing",
    [h("div", { class: "kpis", style: "margin-bottom:12px" }, kpi("playing", String(runner.playing.length)), kpi("claims", String(runner.claims)), kpi("fence", String(runner.fence ?? "–"))),
      runner.playing.length ? table([["run"], ["group", "n"], ["episode", "n"], ["attempt", "n"], ["since", "n"]],
        runner.playing.map(claim => [claim.run, `#${claim.group}`, `E${claim.episode}`, String(claim.attempt), claim.at ? `${span(Math.max(0, system.at - claim.at))}` : "–"])) : null]));
  const ledger = card("Ledger", `${system.ledger_at} · ${bytes(system.kept.versions)} of versions and ${bytes(system.kept.episodes)} of episodes kept`,
    table([["scope"], ["fence", "n"]], Object.entries(system.ledger.fences)), h("div", { style: "height:14px" }),
    table([["table"], ["records", "n"]], Object.entries(system.ledger.tables)));
  return [h("div", { class: "section-title", id: "section-machine" }, h("h2", {}, "Machine"), h("span", {}, "the machine this monitor runs on, the engines it can read, the runners in the ledger, and the ledger")),
    h("div", { class: "cols" }, memory, gpu, busy, meters), h("div", { class: "cols" }, ...channels, ...runners, ledger)];
}

function drawOthers() {
  const others = state.runs.filter(run => !run.labels.run);
  return [h("div", { class: "head" }, h("h1", {}, "Episodes outside a run"), h("div", { class: "sub" }, "Episodes in the feed that no training run asked for: evaluations, tests, programs run by hand.")),
    others.length ? h("div", { class: "tiles" }, others.map(run => link(episodePlace(run.run_id), { class: `tile rail ${stateKind(run.state)}` },
      h("header", {}, h("b", {}, run.labels.title ?? run.labels.task ?? run.run_id.slice(-8)), h("span", { class: "what" }, ""), mark(run.state)),
      h("div", { class: "big" }, Object.values(run.rewards).length ? figure(Object.values(run.rewards)[0]) : h("span", { class: "faint" }, "…")),
      h("div", { class: "facts" }, h("span", {}, h("b", {}, run.samples), " samples"), h("span", {}, h("b", {}, run.slots.length), " slots"), h("span", {}, clock(run.started))))))
      : h("div", { class: "empty" }, "None.")];
}

// Drawing and reading
const stepCrumb = (name, number) => {
  const run = state.system?.runs.find(each => each.run === name), step = run && stepOf(run, number);
  return step ? [[`Step ${step.step}`, stepPlace(name, step.step)]] : [];
};
function redraw() {
  const here = route(), system = state.system, main = document.getElementById("main");
  drawTree();
  let crumbs = [[PAGES.find(([page]) => page === here.page)[1], PAGES.find(([page]) => page === here.page)[2]]], content;
  if (!system) content = [h("div", { class: "empty" }, "Reading the run…")];
  else if (here.kind === "runs") content = drawRuns();
  else if (here.kind === "run") { crumbs.push([`Run ${here.run}`, runPlace(here.run)]); content = drawRun(here.run); }
  else if (here.kind === "step") { crumbs.push([`Run ${here.run}`, runPlace(here.run)], [`Step ${here.number}`, ""]); content = drawStep(here); }
  else if (here.kind === "group") { crumbs.push([`Run ${here.run}`, runPlace(here.run)], ...stepCrumb(here.run, here.number), [`Group #${here.number}`, ""]); content = drawGroup(here); }
  else if (here.kind === "episode") {
    const labels = state.episode?.labels ?? {};
    if (labels.run && labels.group) crumbs.push([`Run ${labels.run}`, runPlace(labels.run)], ...stepCrumb(labels.run, Number(labels.group)),
      [`Group #${Number(labels.group)}`, groupPlace(labels.run, Number(labels.group))]);
    else crumbs.push(["Episodes outside a run", "#/episodes"]);
    crumbs.push([labels.episode ? `Episode ${labels.episode}` : "Episode", here.slot ? episodePlace(here.id) : ""]);
    if (here.slot) crumbs.push([`Rollout ${here.slot}`, ""]);
    content = drawEpisode(here);
  } else if (here.kind === "outside") { crumbs.push(["Episodes outside a run", ""]); content = drawOthers(); }
  else if (here.kind === "policy") { crumbs.push([`Policy ${here.name}`, ""]); content = drawPolicy(here.name); }
  else if (here.kind === "policies") { if (here.sample) crumbs.push(["Sample fixture", ""]); content = drawPolicies(here); }
  else content = drawStatistics(here);
  drawBar(crumbs);
  const scroll = main.scrollTop, across = main.querySelector(".dag-frame")?.scrollLeft;
  const kept = new Map([...main.querySelectorAll(".turn")].map(each => [each.dataset.slot, each.querySelector(".sees pre")?.scrollTop]));
  const showing = `${location.hash} ${state.turn} ${state.full}`;
  main.replaceChildren(h("div", { class: `page${here.kind === "episode" ? " wide" : ""}` }, content));  // (an episode's rollouts take the whole width)
  main.scrollTop = scroll;
  if (across) { const frame = main.querySelector(".dag-frame"); if (frame) frame.scrollLeft = across; }
  for (const each of main.querySelectorAll(".turn")) {
    const pre = each.querySelector(".sees pre");
    if (pre) pre.scrollTop = showing === state.showing ? kept.get(each.dataset.slot) ?? pre.scrollHeight : pre.scrollHeight;
  }
  state.showing = showing;
  const section = state.scrollTo && document.getElementById(`section-${state.scrollTo}`);
  if (section) { section.scrollIntoView({ block: "start" }); state.scrollTo = null; }
}

async function read(path) {
  const answer = await fetch(path);
  if (!answer.ok) throw new Error(`${path}: ${answer.status}`);
  return answer.json();
}

async function pull() {
  if (state.pulling) return;  // (one read at a time: two would both ask for the same new lines)
  state.pulling = true;
  try {
    const [system, runs] = await Promise.all([read("api/system"), read("api/runs")]);
    state.system = system; state.runs = runs;
    const here = route();
    if (here.kind === "group") state.group = await read(`api/groups/${encodeURIComponent(here.run)}/${here.number}`);
    if (here.kind === "statistics") state.statistics = await read("api/statistics");
    if (here.kind === "policies") { state.lineage = await read(`api/policies${here.sample ? "?sample=1" : ""}`); state.lineageSample = here.sample; }
    if (here.kind === "episode") {
      const known = state.episode?.run_id === here.id ? state.episode : null;
      const more = await read(`api/episodes/${encodeURIComponent(here.id)}?after=${known ? known.lines.length : 0}`);
      if (known && more.source === known.source) { known.lines.push(...more.lines); Object.assign(known, { ...more, lines: known.lines }); }
      else state.episode = more;
    }
    const drawn = JSON.stringify([location.hash, system.at > (state.drawnAt ?? 0) + 10 ? system.at : state.drawnAt, system.runs, system.policies,
      system.channels.map(channel => channel.throughput.length), here.kind === "statistics" ? [system.machine.now, { ...state.statistics, now: 0 }] : 0, state.group, state.episode?.lines.length, state.episode?.state,
      here.kind === "policies" ? { ...state.lineage, now: 0 } : 0]);
    if (drawn !== state.drawn) { state.drawn = drawn; state.drawnAt = system.at; redraw(); }
  } catch (error) {
    document.getElementById("live").replaceChildren(h("span", { class: "dot gone" }), "cannot reach the monitor");
  } finally {
    state.pulling = false;
  }
}

addEventListener("hashchange", () => {
  const here = route();
  if (here.kind !== "episode" || here.id !== state.episode?.run_id) state.follow = true;  // (another of the same episode's rollouts keeps its turn)
  state.scrollTo = here.section ?? null;
  state.drawn = ""; document.body.classList.remove("open"); redraw(); pull();
});
addEventListener("resize", () => redraw());
document.getElementById("menu").onclick = () => document.body.classList.toggle("open");
state.scrollTo = route().section ?? null;
pull();
setInterval(pull, 2500);
