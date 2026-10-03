"use strict";
// The monitor's page, organised as a run is: the training run; its steps (each one update of the policy, over the
// groups it covers); each group (a task, a start, a number of episodes); each episode (one run of the program) and
// its rollouts, one per agent, each of which becomes a trajectory to train on; and beside them the policy and the
// machine.

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

const state = { system: null, runs: [], group: null, episode: null, turn: 0, follow: true, full: false, drawn: "" };

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
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (parts[0] === "run" && parts[2] === "group") return { kind: "group", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run" && parts[2] === "step") return { kind: "step", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run") return { kind: "run", run: parts[1] };
  if (parts[0] === "episode") return { kind: "episode", id: parts[1], slot: parts[2] || null };
  if (parts[0] === "policy") return { kind: "policy", name: parts[1] };
  if (parts[0] === "system") return { kind: "system" };
  if (parts[0] === "episodes") return { kind: "outside" };
  const first = state.system?.runs[0];
  return first ? { kind: "run", run: first.run } : state.system ? { kind: "outside" } : { kind: "loading" };
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
const STAGES = ["decided", "asked", "played", "recorded"];
const AT = { decided: [1, "to ask"], waiting: [2, "to start"], playing: [2, ""], ended: [3, "recording"], done: [4, ""] };
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

function drawTree() {
  const system = state.system, here = route(), tree = document.getElementById("tree");
  if (!system) return;
  document.getElementById("where").textContent = system.directory;
  const showing = here.kind === "episode" ? state.episode?.labels : null;
  const nodes = [];
  for (const run of system.runs) {
    const runKey = `run:${run.run}`, runOpen = folds[runKey] ?? true;
    nodes.push(h("div", { class: "label" }, "Training run"));
    nodes.push(node(runPlace(run.run), here.kind === "run" && here.run === run.run, twist(runKey, runOpen),
      h("span", { class: `dot ${system.processes?.alive ? "alive" : ""}` }), h("span", { class: "name" }, run.run),
      h("span", { class: "tag" }, `${run.steps.length} steps`)));
    if (!runOpen) continue;
    const groups = groupsOf(run), children = [];
    const inGroup = number => (here.kind === "group" && here.run === run.run && here.number === number)
      || (showing?.job === run.run && Number(showing?.group) === number);
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
  if (system.policies.length) nodes.push(h("div", { class: "label" }, "Policies"));
  for (const policy of system.policies) nodes.push(node(policyPlace(policy.policy), here.kind === "policy" && here.name === policy.policy,
    h("span", { class: "name" }, policy.policy), h("span", { class: "tag" }, versionOf(policy.head))));
  nodes.push(h("div", { class: "label" }, "Around it"));
  nodes.push(node("#/system", here.kind === "system", h("span", { class: "name" }, "Machine, engines and ledger")));
  const others = state.runs.filter(run => !run.labels.job);
  if (others.length) nodes.push(node("#/episodes", here.kind === "outside", h("span", { class: "name" }, "Episodes outside a run"), h("span", { class: "tag" }, String(others.length))));
  const scroll = tree.scrollTop;
  tree.replaceChildren(...nodes);
  tree.scrollTop = scroll;
}

function drawBar(crumbs) {
  const system = state.system;
  document.getElementById("crumbs").replaceChildren(...crumbs.flatMap((crumb, index) => [
    ...(index ? [h("span", { class: "sep" }, "/")] : []), index === crumbs.length - 1 ? h("b", {}, crumb[0]) : link(crumb[1], {}, crumb[0])]));
  const live = document.getElementById("live");
  if (!system) { live.replaceChildren("connecting…"); return; }
  const alive = system.processes?.alive;
  live.replaceChildren(h("span", { class: `dot ${alive ? "alive" : system.processes ? "gone" : ""}` }),
    system.processes ? (alive ? "running" : "not running") : "", system.written ? ` · wrote ${span(Math.max(0, system.at - system.written))} ago` : "");
}

// The training run
function drawRun(name) {
  const system = state.system, run = system.runs.find(each => each.run === name);
  if (!run) return [h("div", { class: "empty" }, `There is no run ${name}.`)];
  const policy = system.policies[0], channel = system.channels.find(each => each.adapter) ?? system.channels[0];
  const trained = run.done.filter(line => line.update).length, last = run.done.at(-1);
  const committed = run.steps.filter(step => step.state === "committed").length;
  const throughput = channel?.throughput.at(-1);
  const width = Math.max(300, document.getElementById("main").clientWidth - 100);
  const head = h("div", { class: "head" }, h("h1", {}, `Run ${run.run}`),
    specs(policy ? spec("trains", policy.policy, "violet") : null, channel?.adapter ? spec("serving", channel.adapter, "accent") : null,
      spec("fence", run.fence ?? "–"), spec("directory", system.directory)));
  const kpis = h("div", { class: "kpis" },
    kpi("Steps", `${run.steps.length}`, `${committed} committed · ${run.next.length} groups toward the next`),
    kpi("Groups done", `${run.done.length}`, `${trained} trained on, of ${run.decided} decided`),
    kpi("Rows unlocked", last ? `${last.unlocked}` : "–", "of the catalog"),
    kpi("Policy head", policy ? versionOf(policy.head) : "–", policy ? `${policy.versions.length} versions` : ""),
    kpi("Inference", throughput ? `${figure(throughput.tokens_per_second)} tok/s` : "–", throughput ? `${figure(throughput.mean_concurrency)} requests at once` : "no measurement yet"),
    kpi("Episodes ended", `${system.jobs.find(job => job.job === run.run)?.episodes ?? 0}`, `${tokens(system.jobs.find(job => job.job === run.run)?.sampled)} tokens sampled`));
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
  return [head, kpis,
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
  return [head, kpis, group.stage !== "done" ? card("Stage", "", stages(group)) : null,
    h("div", { class: "section-title" }, h("h2", {}, "Episodes"), h("span", {}, group.ticket ? `${group.count ?? "?"} asked for under ticket ${group.ticket}` : "")), episodes,
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
  const run = labels.job ? state.system.runs.find(each => each.run === labels.job) : null;
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
  const size = versions.reduce((sum, version) => sum + version.weights.bytes + (version.state?.bytes ?? 0), 0);
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
      version.state ? "kept" : version.released ? { text: "released", kind: "still" } : "–", bytes(version.weights.bytes + (version.state?.bytes ?? 0))]),
    newest.map(version => madeBy.has(version.name) ? () => go(stepPlace(...madeBy.get(version.name))) : null)))];
}

// The machine, the engines, the jobs and the ledger
function drawSystem() {
  const system = state.system, machine = system.machine.now, history = system.machine.history;
  const width = Math.min(560, Math.max(260, document.getElementById("main").clientWidth / 2 - 90));
  const machineCard = card("Machine", "where this monitor runs",
    machine.memory.total ? meter("memory", machine.memory.total - machine.memory.available, machine.memory.total, `${bytes(machine.memory.available)} available of ${bytes(machine.memory.total)}`) : null,
    machine.accelerators.map(each => meter(each.name, each.used, each.total, `${bytes(each.used)} of ${bytes(each.total)} · ${Math.round(100 * each.busy)}% busy`)),
    machine.disk ? meter("disk", machine.disk.total - machine.disk.free, machine.disk.total, `${bytes(machine.disk.free)} free`) : null,
    history.length > 1 && machine.memory.total ? [spark(history.map(each => each.memory.available ?? 0), "s-warm", width, 56, true),
      h("div", { class: "small muted" }, `Memory available over the last ${span(machine.at - history[0].at)}.`)] : null,
    system.processes ? h("p", { class: "small muted", style: "margin:12px 0 0" }, `Process ${system.processes.owner} ${system.processes.alive ? "running" : "gone"}`,
      system.processes.started.map(each => ` · ${each.name} ${each.pid} ${each.alive ? "running" : "gone"}`)) : null);
  const channels = system.channels.map(channel => {
    const latest = channel.throughput.at(-1);
    return card(`Channel ${channel.channel}`, channel.adapter ? `serving ${channel.adapter} since ${clock(channel.published)}` : "serving the base model",
      latest ? [h("div", { class: "kpis", style: "margin-bottom:12px" }, kpi("tokens a second", figure(latest.tokens_per_second)),
        kpi("requests at once", figure(latest.mean_concurrency)), kpi("each", `${figure(latest.tokens_per_second_per_stream)} tok/s`)),
      spark(channel.throughput.map(each => each.tokens_per_second), "s-accent", width, 70, true),
      h("div", { class: "small muted" }, `The engines' last ${channel.throughput.length} busy minutes.`)] : h("p", { class: "muted" }, "No request has been measured yet."));
  });
  const jobs = system.jobs.map(job => card(`Job ${job.job}`, `${job.tickets} tickets`, h("div", { class: "kpis" },
    kpi("episodes in its log", `${job.episodes}`, Object.entries(job.outcomes).map(([outcome, count]) => `${count} ${outcome}`).join(", ")),
    kpi("acknowledged", `${job.acknowledged} of ${job.last}`), kpi("tokens sampled", tokens(job.sampled)))));
  const ledger = card("Ledger", `${bytes(system.kept.versions)} of versions and ${bytes(system.kept.episodes)} of episodes kept`,
    table([["scope"], ["fence", "n"]], Object.entries(system.ledger.fences)), h("div", { style: "height:14px" }),
    table([["table"], ["records", "n"]], Object.entries(system.ledger.tables)));
  return [h("div", { class: "head" }, h("h1", {}, "Machine, engines and ledger"), h("div", { class: "sub" }, system.directory)),
    h("div", { class: "cols" }, machineCard, ...channels, ...jobs, ledger)];
}

function drawOthers() {
  const others = state.runs.filter(run => !run.labels.job);
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
  let crumbs = [[system?.directory?.split("/").at(-1) ?? "…", "#/"]], content;
  if (!system) content = [h("div", { class: "empty" }, "Reading the run…")];
  else if (here.kind === "run") { crumbs.push([`Run ${here.run}`, runPlace(here.run)]); content = drawRun(here.run); }
  else if (here.kind === "step") { crumbs.push([`Run ${here.run}`, runPlace(here.run)], [`Step ${here.number}`, ""]); content = drawStep(here); }
  else if (here.kind === "group") { crumbs.push([`Run ${here.run}`, runPlace(here.run)], ...stepCrumb(here.run, here.number), [`Group #${here.number}`, ""]); content = drawGroup(here); }
  else if (here.kind === "episode") {
    const labels = state.episode?.labels ?? {};
    if (labels.job && labels.group) crumbs.push([`Run ${labels.job}`, runPlace(labels.job)], ...stepCrumb(labels.job, Number(labels.group)),
      [`Group #${Number(labels.group)}`, groupPlace(labels.job, Number(labels.group))]);
    else crumbs.push(["Episodes outside a run", "#/episodes"]);
    crumbs.push([labels.episode ? `Episode ${labels.episode}` : "Episode", here.slot ? episodePlace(here.id) : ""]);
    if (here.slot) crumbs.push([`Rollout ${here.slot}`, ""]);
    content = drawEpisode(here);
  } else if (here.kind === "policy") { crumbs.push([`Policy ${here.name}`, ""]); content = drawPolicy(here.name); }
  else if (here.kind === "system") { crumbs.push(["Machine, engines and ledger", ""]); content = drawSystem(); }
  else content = drawOthers();
  drawBar(crumbs);
  const scroll = main.scrollTop;
  const kept = new Map([...main.querySelectorAll(".turn")].map(each => [each.dataset.slot, each.querySelector(".sees pre")?.scrollTop]));
  const showing = `${location.hash} ${state.turn} ${state.full}`;
  main.replaceChildren(h("div", { class: `page${here.kind === "episode" ? " wide" : ""}` }, content));  // (an episode's rollouts take the whole width)
  main.scrollTop = scroll;
  for (const each of main.querySelectorAll(".turn")) {
    const pre = each.querySelector(".sees pre");
    if (pre) pre.scrollTop = showing === state.showing ? kept.get(each.dataset.slot) ?? pre.scrollHeight : pre.scrollHeight;
  }
  state.showing = showing;
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
    if (here.kind === "episode") {
      const known = state.episode?.run_id === here.id ? state.episode : null;
      const more = await read(`api/episodes/${encodeURIComponent(here.id)}?after=${known ? known.lines.length : 0}`);
      if (known && more.source === known.source) { known.lines.push(...more.lines); Object.assign(known, { ...more, lines: known.lines }); }
      else state.episode = more;
    }
    const drawn = JSON.stringify([location.hash, system.at > (state.drawnAt ?? 0) + 10 ? system.at : state.drawnAt, system.runs, system.policies,
      system.channels.map(channel => channel.throughput.length), here.kind === "system" ? system.machine.now : 0, state.group, state.episode?.lines.length, state.episode?.state]);
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
  state.drawn = ""; document.body.classList.remove("open"); redraw(); pull();
});
addEventListener("resize", () => redraw());
document.getElementById("menu").onclick = () => document.body.classList.toggle("open");
pull();
setInterval(pull, 2500);
