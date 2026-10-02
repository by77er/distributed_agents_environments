"use strict";
// The monitor's page, organised as a run is: the training run; its groups (a task, a start, a number of rollouts
// and a step); each rollout (one episode of the program, which produces a trajectory per agent); and beside them
// the policy and the machine.

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

const state = { system: null, runs: [], group: null, episode: null, slot: null, turn: 0, follow: true, full: false, drawn: "" };

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
const avatar = name => h("span", { class: "avatar", style: `background:hsl(${hue(name)} 55% 46%)` }, name.slice(0, 1));

// Places
const go = place => { location.hash = place; };
const link = (place, attributes, ...children) => h("a", { href: place, ...attributes }, ...children);
const runPlace = run => `#/run/${encodeURIComponent(run)}`;
const groupPlace = (run, number) => `${runPlace(run)}/group/${number}`;
const rolloutPlace = id => `#/rollout/${encodeURIComponent(id)}`;
const policyPlace = name => `#/policy/${encodeURIComponent(name)}`;
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (parts[0] === "run" && parts[2] === "group") return { kind: "group", run: parts[1], number: Number(parts[3]) };
  if (parts[0] === "run") return { kind: "run", run: parts[1] };
  if (parts[0] === "rollout") return { kind: "rollout", id: parts[1] };
  if (parts[0] === "policy") return { kind: "policy", name: parts[1] };
  if (parts[0] === "system") return { kind: "system" };
  if (parts[0] === "rollouts") return { kind: "outside" };
  const first = state.system?.runs[0];
  return first ? { kind: "run", run: first.run } : state.system ? { kind: "outside" } : { kind: "loading" };
}

// Shared pieces
const pill = (text, kind = "", dot) => h("span", { class: `pill ${kind}` }, dot ? h("span", { class: `dot ${dot}` }) : null, text);
const kpi = (label, value, note) => h("div", { class: "kpi" }, h("span", {}, label), h("b", { title: String(value) }, value), note ? h("small", {}, note) : null);
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

const STAGES = ["decided", "asked for", "played", "stepped", "version made", "outcome written"];
const AT = { decided: [1, "to be asked for"], waiting: [2, "queued"], playing: [2, ""], ended: [3, "awaiting its step"], stepping: [3, ""], made: [5, "writing outcome"], done: [6, ""] };
function stages(group) {
  const [at, waits] = AT[group.stage];
  const says = group.stage === "playing" ? `playing ${group.ended}/${group.count}`
    : group.stage === "stepping" ? `stepping → ${versionOf(group.step?.makes)}` : waits;
  return h("div", { class: "stages" }, STAGES.map((name, index) => h("div", {
    class: index < at ? "done" : index === at ? `now${["playing", "stepping"].includes(group.stage) ? " active" : ""}` : "",
  }, h("span", {}, index === at ? says : name))));
}
const outcomeOf = line => line.update
  ? { kind: "moved", text: `trained ${line.sequences_trained} of ${line.sequences_recorded} · ${line.update.optimizer_steps ?? "?"} steps · moved ${Number(line.update.kl_moved ?? 0).toFixed(4)}` }
  : line.error ? { kind: "bad", text: `step failed: ${line.error}` }
  : { kind: line.failed && !line.rewards.length ? "bad" : "still", text: line.skipped ?? "" };
const dotsOf = line => h("span", { class: "dots" }, [...line.rewards.map((_, index) => h("i", { class: line.solved[index] ? "solved" : "unsolved" })),
  ...Array.from({ length: line.failed }, () => h("i", { class: "failed" }))]);
const stateKind = name => ({ running: "accent", completed: "good", done: "good", failed: "bad", cancelled: "bad", playing: "accent", stepping: "violet", made: "violet" })[name] ?? "";

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

// Every group the run is done with: each rollout's reward as a dot, the group's mean as a rule, and under the axis
// what was done with the group. A column opens its group.
function rewardsChart(run, width) {
  const lines = run.iterations, left = 32, top = 10, plot = 120, band = top + plot + 10, height = band + 28;
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
      onclick: () => go(groupPlace(run.run, line.iteration)) },
    svg("title", {}, `#${line.iteration} ${line.task} · ${line.title}\n${line.rewards.map(figure).join(" ") || "no rollout"}${line.failed ? ` (${line.failed} failed)` : ""}\n${outcomeOf(line).text}`)));
    drawing.lastChild.addEventListener("click", () => go(groupPlace(run.run, line.iteration)));
    line.rewards.forEach((value, place) => drawing.append(svg("circle", { cx: x + (place - (line.rewards.length - 1) / 2) * spread, cy: y(value),
      r: Math.min(3.2, Math.max(1.6, column / 5)), class: line.solved[place] ? "f-good" : "f-quiet", "pointer-events": "none" })));
    if (line.rewards.length) drawing.append(svg("line", { x1: x - column * 0.34, x2: x + column * 0.34, y1: y(mean(line.rewards)), y2: y(mean(line.rewards)), class: "s-ink", "pointer-events": "none" }));
    const did = line.update ? "f-accent" : line.error || (line.failed && !line.rewards.length) ? "f-bad" : "f-hollow";
    drawing.append(svg("rect", { x: x - mark / 2, y: band, width: mark, height: mark, rx: 2, class: did, "pointer-events": "none" }));
    if (line.iteration % every === 0) drawing.append(svg("text", { x, y: band + 23, "text-anchor": "middle" }, String(line.iteration)));
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

// The hierarchy, on the left
function drawTree() {
  const system = state.system, here = route(), tree = document.getElementById("tree");
  if (!system) return;
  document.getElementById("where").textContent = system.directory;
  const nodes = [];
  const node = (place, current, ...children) => link(place, { class: `node${current ? " current" : ""}` }, ...children);
  for (const run of system.runs) {
    const running = system.processes?.alive;
    nodes.push(h("div", { class: "label" }, "Training run"));
    nodes.push(node(runPlace(run.run), here.kind === "run" && here.run === run.run, h("span", { class: `dot ${running ? "alive" : ""}` }),
      h("span", { class: "name" }, run.run), h("span", { class: "tag" }, `${run.iterations.length} done`)));
    const children = [];
    for (const group of run.open) children.push(node(groupPlace(run.run, group.number), here.kind === "group" && here.number === group.number,
      h("span", { class: "num" }, `#${group.number}`), h("span", { class: "name" }, group.task),
      h("span", { class: "dots" }, group.episodes.map(each => h("i", { class: each.outcome ? (each.solved ? "solved" : each.outcome === "completed" ? "unsolved" : "failed") : "running" }))),
      h("span", { class: "tag" }, group.stage)));
    for (const line of [...run.iterations].reverse()) children.push(node(groupPlace(run.run, line.iteration), here.kind === "group" && here.number === line.iteration,
      h("span", { class: "num" }, `#${line.iteration}`), h("span", { class: "name" }, line.task), dotsOf(line),
      h("span", { class: "tag" }, line.adapter ? versionOf(line.adapter) : line.error ? "failed" : "–")));
    nodes.push(h("div", { class: "children" }, children));
  }
  if (system.policies.length) nodes.push(h("div", { class: "label" }, "Policies"));
  for (const policy of system.policies) nodes.push(node(policyPlace(policy.policy), here.kind === "policy" && here.name === policy.policy,
    h("span", { class: "name" }, policy.policy), h("span", { class: "tag" }, versionOf(policy.head))));
  nodes.push(h("div", { class: "label" }, "Around it"));
  nodes.push(node("#/system", here.kind === "system", h("span", { class: "name" }, "Machine, engines and ledger")));
  const others = state.runs.filter(run => !run.labels.job);
  if (others.length) nodes.push(node("#/rollouts", here.kind === "outside", h("span", { class: "name" }, "Rollouts outside a run"), h("span", { class: "tag" }, String(others.length))));
  tree.replaceChildren(...nodes);
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
  const trained = run.iterations.filter(line => line.update).length, last = run.iterations.at(-1);
  const throughput = channel?.throughput.at(-1);
  const width = Math.max(300, document.getElementById("main").clientWidth - 100);
  const head = h("div", { class: "head" }, h("h1", {}, `Run ${run.run}`),
    h("div", { class: "pills" }, policy ? pill(h("span", {}, "trains ", h("b", {}, policy.policy)), "violet") : null,
      channel?.adapter ? pill(h("span", {}, "serving ", h("b", {}, channel.adapter)), "accent") : null, pill(`fence ${run.fence ?? "–"}`)),
    h("div", { class: "sub" }, system.directory));
  const kpis = h("div", { class: "kpis" },
    kpi("Groups done", `${run.iterations.length}`, `${run.decided} decided`),
    kpi("Trained on", `${trained}`, `${run.iterations.length - trained} skipped or failed`),
    kpi("Rows unlocked", last ? `${last.unlocked}` : "–", "of the catalog"),
    kpi("Policy head", policy ? versionOf(policy.head) : "–", policy ? `${policy.versions.length} versions` : ""),
    kpi("Inference", throughput ? `${figure(throughput.tokens_per_second)} tok/s` : "–", throughput ? `${figure(throughput.mean_concurrency)} requests at once` : "no measurement yet"),
    kpi("Rollouts ended", `${system.jobs.find(job => job.job === run.run)?.episodes ?? 0}`, `${tokens(system.jobs.find(job => job.job === run.run)?.sampled)} tokens sampled`));
  const flight = run.open.length ? h("div", { class: "tiles" }, run.open.map(group => link(groupPlace(run.run, group.number), { class: "tile" },
    h("header", {}, h("b", {}, `#${group.number}`), h("span", { class: "what" }, `${group.task} · ${group.title ?? ""}`),
      h("span", { class: "faint small" }, group.decided ? span(system.at - group.decided) : "")),
    stages(group),
    group.episodes.length ? h("div", { class: "pips" }, group.episodes.map(episode => h("span", { class: "pip" },
      h("span", { class: `dot ${episode.interrupted ? "" : episode.state}` }), `${episode.episode ?? "?"}`,
      episode.outcome ? h("b", {}, figure(episode.reward)) : h("span", { class: "faint" }, `${episode.samples ?? 0} turns`)))) : null)))
    : h("div", { class: "empty" }, "No group is in flight.");
  const recent = [...run.iterations].reverse().slice(0, 10);
  const tasks = new Map();
  for (const line of run.iterations) {
    const task = tasks.get(line.task) ?? { title: line.title, groups: 0, trained: 0 };
    task.groups += 1; task.trained += line.update ? 1 : 0; task.last = line; tasks.set(line.task, task);
  }
  const played = [...tasks].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));
  return [head, kpis,
    h("div", { class: "section-title" }, h("h2", {}, "In flight"), h("span", {}, `${run.open.length} groups`)), flight,
    card("Rewards by group", "each dot a rollout; rewards are each task's own",
      run.iterations.length ? [rewardsChart(run, width), h("div", { class: "legend" },
        h("span", {}, h("i", { style: "background:var(--good)" }), "solved"), h("span", {}, h("i", { style: "background:var(--faint)" }), "not solved"),
        h("span", {}, h("i", { class: "rule" }), "mean"), h("span", {}, h("i", { class: "square", style: "background:var(--accent)" }), "trained on"),
        h("span", {}, h("i", { class: "square hollow" }), "skipped"), h("span", {}, h("i", { class: "square", style: "background:var(--bad)" }), "no rollout, or the step failed"))]
        : h("div", { class: "empty" }, "No group is done with yet.")),
    h("div", { class: "cols" },
      card("Latest groups", "newest first", table([["group"], ["task"], ["rewards"], ["what was done"], ["took", "n"]],
        recent.map(line => [{ text: `#${line.iteration}`, kind: "key" }, line.task, line.rewards.map(figure).join(" ") || `${line.failed} failed`, outcomeOf(line), span(line.seconds)]),
        recent.map(line => () => go(groupPlace(run.run, line.iteration))))),
      card("Tasks played", `${played.length} rows`, table([["task"], ["groups", "n"], ["trained", "n"], ["last rewards"], ["solved", "n"]],
        played.map(([key, task]) => [{ text: key, kind: "key" }, task.groups, task.trained, task.last.rewards.map(figure).join(" ") || "–",
          `${task.last.solved.filter(Boolean).length}/${task.last.rewards.length}`]),
        played.map(([, task]) => () => go(groupPlace(run.run, task.last.iteration))))))];
}

// A group
function drawGroup(here) {
  const group = state.group;
  if (!group || group.run !== here.run || group.number !== here.number) return [h("div", { class: "empty" }, "Reading the group…")];
  const outcome = group.outcome, version = group.version;
  const rewards = outcome?.rewards ?? group.episodes.filter(each => each.outcome === "completed").map(each => each.reward);
  const head = h("div", { class: "head" }, h("h1", {}, `Group #${group.number}`),
    h("div", { class: "pills" }, pill(group.stage, stateKind(group.stage)), outcome ? pill(outcomeOf(outcome).text.split(" · ")[0], outcome.update ? "good" : outcome.error ? "bad" : "") : null),
    h("div", { class: "sub" }, `${group.task} · ${group.title ?? ""}`));
  const kpis = h("div", { class: "kpis" },
    kpi("Mean reward", rewards.length ? figure(mean(rewards)) : "–", rewards.length ? rewards.map(figure).join("  ") : "no rollout has ended"),
    kpi("Solved", outcome ? `${outcome.solved.filter(Boolean).length} of ${outcome.solved.length}` : `${group.episodes.filter(each => each.solved).length} of ${group.count ?? "?"}`, outcome?.failed ? `${outcome.failed} failed` : ""),
    kpi("Played for", outcome ? span(outcome.rollout_seconds) : group.decided ? span(state.system.at - group.decided) : "–", group.decided ? `decided ${clock(group.decided)}` : ""),
    kpi("Trained on", outcome?.update ? `${outcome.sequences_trained}` : "–", outcome ? `of ${outcome.sequences_recorded} sequences` : group.step ? `${group.step.sequences ?? "?"} sequences, stepping` : ""),
    kpi("Made", version ? versionOf(version.name) : group.step ? `${versionOf(group.step.makes)}…` : "–", group.step?.parent ? `from ${versionOf(group.step.parent)}` : ""));
  const episodes = group.episodes.length ? h("div", { class: "tiles" }, group.episodes.map(episode => {
    const info = episode.info ?? {};
    return link(rolloutPlace(episode.run_id), { class: "tile" },
      h("header", {}, h("b", {}, `Rollout ${episode.episode ?? "?"}`), h("span", { class: "what" }, ""),
        pill(episode.interrupted ? "interrupted" : episode.state ?? "running", stateKind(episode.state), episode.interrupted ? "" : episode.state)),
      h("div", { class: "big" }, episode.outcome ? figure(episode.reward) : h("span", { class: "faint" }, "…")),
      h("div", { class: "facts" }, info.turns != null ? h("span", {}, h("b", {}, info.turns), " turns") : episode.samples != null ? h("span", {}, h("b", {}, episode.samples), " samples") : null,
        info.duration != null ? h("span", {}, h("b", {}, figure(info.duration)), " game min") : null,
        info.ended ? h("span", {}, "ended by ", h("b", {}, info.ended)) : null,
        episode.sampled ? h("span", {}, h("b", {}, tokens(episode.sampled)), " tokens") : null,
        episode.solved ? h("span", { class: "moved" }, "solved") : null),
      episode.detail ? h("div", { class: "error-text" }, episode.detail.slice(0, 240)) : null);
  })) : h("div", { class: "empty" }, "No rollout has started.");
  const metrics = version?.metrics ?? outcome?.update;
  const what = card("What was done", outcome ? outcomeOf(outcome).text : group.step ? "a step is being taken" : "nothing yet",
    metrics ? [pairs([
      ["version", version ? link(policyPlace(version.name.split("@")[0]), {}, h("b", {}, version.name)) : outcome?.adapter ?? "–"],
      ["from", version?.parent ?? group.step?.parent ?? "the base model"],
      ...["kl_moved", "kl_floor", "loss", "clip_fraction", "mean_mismatch", "optimizer_steps", "tokens", "longest_sequence_tokens", "peak_gpu_gib"]
        .filter(key => metrics[key] !== undefined).map(key => [key.replaceAll("_", " "), figure(metrics[key])]),
      ["took", span(metrics.update_seconds ?? metrics.seconds)]])]
      : group.step ? pairs([["makes", group.step.makes ?? "–"], ["from", group.step.parent ?? "the base model"], ["sequences", figure(group.step.sequences)],
        ["decided", group.step.decided ? `${span(state.system.at - group.step.decided)} ago` : "–"]])
      : outcome?.skipped ? h("p", { class: "muted" }, outcome.skipped) : h("p", { class: "muted" }, "The group is still being played."));
  const start = card("Start", "what every rollout of the group was given", pairs(Object.entries(group.parameters ?? {}).map(([key, value]) => [key, figure(value)])));
  const failures = outcome?.failures?.length ? card("Why rollouts failed", `${outcome.failures.length}`, outcome.failures.map(failure => h("p", { class: "error-text" }, failure))) : null;
  return [head, kpis, group.stage !== "done" ? card("Stage", "", stages(group)) : null,
    h("div", { class: "section-title" }, h("h2", {}, "Rollouts"), h("span", {}, group.ticket ? `${group.count ?? "?"} asked for under ticket ${group.ticket}` : "")), episodes,
    h("div", { class: "cols" }, what, start), failures];
}

// A rollout: what its episode reported, then its trajectories (one per agent)
function drawRollout(here) {
  const episode = state.episode;
  if (!episode || episode.run_id !== here.id) return [h("div", { class: "empty" }, "Reading the rollout…")];
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
  const head = h("div", { class: "head" }, h("h1", {}, labels.episode ? `Rollout ${labels.episode}` : episode.run_id.slice(-8)),
    h("div", { class: "pills" }, pill(episode.state ?? "running", stateKind(episode.state), episode.state),
      episode.ended ? pill(h("span", {}, "reward ", h("b", {}, figure(episode.ended.reward))), episode.ended.solved ? "good" : "") : null,
      episode.source === "archive" ? pill("replayed from its kept events: replies only", "warm") : null),
    h("div", { class: "sub" }, [labels.task, labels.title, episode.run_id].filter(Boolean).join(" · ")));
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
  if (!ordered.size) return [head, kpis, result, h("div", { class: "empty" }, episode.source ? "No agent has taken a turn yet." : "Neither the feed nor the job's log has this rollout's trajectories.")];
  if (state.slot && !ordered.has(state.slot)) state.slot = null;
  const shown = state.slot ? [[state.slot, ordered.get(state.slot)]] : [...ordered];
  const turns = Math.max(...shown.map(([, samples]) => samples.length));
  if (state.follow) state.turn = turns - 1;
  state.turn = Math.max(0, Math.min(state.turn, turns - 1));
  const tabs = h("div", { class: "agents" }, h("button", { class: `agent-tab all${state.slot ? "" : " current"}`, onclick: () => { state.slot = null; redraw(); } }, "All trajectories"),
    [...ordered.keys()].map(slot => h("button", { class: `agent-tab${state.slot === slot ? " current" : ""}`, onclick: () => { state.slot = slot; redraw(); } }, avatar(slot), slot)));
  const move = delta => () => { state.turn = Math.max(0, Math.min(turns - 1, state.turn + delta)); state.follow = false; redraw(); };
  const scrub = h("div", { class: "scrub" }, h("button", { onclick: move(-1), "aria-label": "previous turn" }, "‹"), h("button", { onclick: move(1), "aria-label": "next turn" }, "›"),
    h("output", {}, `turn ${state.turn + 1} of ${turns}`),
    h("input", { type: "range", min: 0, max: turns - 1, value: state.turn, "aria-label": "turn", oninput: event => { state.turn = Number(event.target.value); state.follow = false; redraw(); } }),
    h("label", {}, h("input", { type: "checkbox", checked: state.follow, onchange: event => { state.follow = event.target.checked; redraw(); } }), "follow"),
    episode.source === "feed" ? h("label", {}, h("input", { type: "checkbox", checked: state.full, onchange: event => { state.full = event.target.checked; redraw(); } }), "whole context") : null);
  const cards = h("div", { class: "turns" }, shown.map(([slot, samples]) => turnCard(slot, samples, state.turn, episode)));
  const effects = otherEffects(episode.lines);
  return [head, kpis, result, h("div", { class: "section-title" }, h("h2", {}, "Trajectories"), h("span", {}, `${ordered.size}, one per agent${summaries ? ` · ${summaries} memory summaries written` : ""}`)), tabs, scrub, cards, effects];
}

function turnCard(slot, samples, turn, episode) {
  const sample = samples[turn];
  const header = h("header", {}, avatar(slot), h("b", {}, slot), h("span", {}, sample ? `${figure(sample.seconds)} s · ${sample.finish_reason ?? ""}` : ""));
  if (!sample) return h("div", { class: "turn", "data-slot": slot }, header, h("section", {}, h("span", { class: "none" }, "no turn yet")));
  const sees = h("section", { class: "sees" }, h("h3", {}, "Sees"), sample.messages.length ? seen(sample) : h("span", { class: "none" }, "kept only as tokens: the feed has let this rollout go"));
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
  const madeBy = new Map(system.runs.flatMap(run => run.iterations.filter(line => line.adapter).map(line => [line.adapter, [run.run, line.iteration]])));
  const size = versions.reduce((sum, version) => sum + version.weights.bytes + (version.state?.bytes ?? 0), 0);
  const serving = system.channels.find(channel => channel.adapter?.startsWith(`${name}@`));
  const newest = [...versions].reverse();
  return [h("div", { class: "head" }, h("h1", {}, `Policy ${policy.policy}`),
    h("div", { class: "pills" }, serving ? pill(h("span", {}, "served on ", h("b", {}, serving.channel)), "accent") : null, pill(`fence ${policy.fence ?? "–"}`)),
    h("div", { class: "sub" }, "Each version is the commit of one step: its weights and the trainer's state are files in the blob store.")),
  h("div", { class: "kpis" }, kpi("Head", versionOf(policy.head), policy.head ?? ""), kpi("Versions", `${versions.length}`),
    kpi("Kept", bytes(size), "weights and trainer state"),
    kpi("Last step moved", versions.length ? figure(versions.at(-1).metrics.kl_moved) : "–", "KL from its parent")),
  card("How far each step moved the policy", "KL between a version and its parent", barChart(versions.map(version => version.metrics.kl_moved ?? 0),
    versions.map(version => versionOf(version.name)), width, 150, index => { const made = madeBy.get(versions[index].name); if (made) go(groupPlace(...made)); })),
  card("Versions", "newest first", table([["version"], ["from"], ["made"], ["by"], ["sequences", "n"], ["steps", "n"], ["moved", "n"], ["loss", "n"], ["took", "n"], ["size", "n"]],
    newest.map(version => [{ text: versionOf(version.name), kind: "key" }, versionOf(version.parent), clock(version.made),
      madeBy.has(version.name) ? `group #${madeBy.get(version.name)[1]}` : "–", figure(version.metrics.sequences), figure(version.metrics.optimizer_steps),
      version.metrics.kl_moved?.toFixed(4), version.metrics.loss?.toFixed(3), span(version.metrics.update_seconds ?? version.metrics.seconds),
      bytes(version.weights.bytes + (version.state?.bytes ?? 0))]),
    newest.map(version => madeBy.has(version.name) ? () => go(groupPlace(...madeBy.get(version.name))) : null)))];
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
    kpi("rollouts in its log", `${job.episodes}`, Object.entries(job.outcomes).map(([outcome, count]) => `${count} ${outcome}`).join(", ")),
    kpi("acknowledged", `${job.acknowledged} of ${job.last}`), kpi("tokens sampled", tokens(job.sampled)))));
  const ledger = card("Ledger", `${bytes(system.kept.versions)} of versions and ${bytes(system.kept.episodes)} of rollouts kept`,
    table([["scope"], ["fence", "n"]], Object.entries(system.ledger.fences)), h("div", { style: "height:14px" }),
    table([["table"], ["records", "n"]], Object.entries(system.ledger.tables)));
  return [h("div", { class: "head" }, h("h1", {}, "Machine, engines and ledger"), h("div", { class: "sub" }, system.directory)),
    h("div", { class: "cols" }, machineCard, ...channels, ...jobs, ledger)];
}

function drawOthers() {
  const others = state.runs.filter(run => !run.labels.job);
  return [h("div", { class: "head" }, h("h1", {}, "Rollouts outside a run"), h("div", { class: "sub" }, "Rollouts in the feed that no training run asked for: evaluations, tests, programs run by hand.")),
    others.length ? h("div", { class: "tiles" }, others.map(run => link(rolloutPlace(run.run_id), { class: "tile" },
      h("header", {}, h("b", {}, run.labels.title ?? run.labels.task ?? run.run_id.slice(-8)), h("span", { class: "what" }, ""), pill(run.state, stateKind(run.state), run.state)),
      h("div", { class: "big" }, Object.values(run.rewards).length ? figure(Object.values(run.rewards)[0]) : h("span", { class: "faint" }, "…")),
      h("div", { class: "facts" }, h("span", {}, h("b", {}, run.samples), " samples"), h("span", {}, h("b", {}, run.slots.length), " slots"), h("span", {}, clock(run.started))))))
      : h("div", { class: "empty" }, "None.")];
}

// Drawing and reading
function redraw() {
  const here = route(), system = state.system, main = document.getElementById("main");
  drawTree();
  let crumbs = [[system?.directory?.split("/").at(-1) ?? "…", "#/"]], content;
  if (!system) content = [h("div", { class: "empty" }, "Reading the run…")];
  else if (here.kind === "run") { crumbs.push([`Run ${here.run}`, runPlace(here.run)]); content = drawRun(here.run); }
  else if (here.kind === "group") { crumbs.push([`Run ${here.run}`, runPlace(here.run)], [`Group #${here.number}`, ""]); content = drawGroup(here); }
  else if (here.kind === "rollout") {
    const labels = state.episode?.labels ?? {};
    if (labels.job && labels.iteration) crumbs.push([`Run ${labels.job}`, runPlace(labels.job)], [`Group #${labels.iteration}`, groupPlace(labels.job, labels.iteration)]);
    else crumbs.push(["Rollouts outside a run", "#/rollouts"]);
    crumbs.push([labels.episode ? `Rollout ${labels.episode}` : "Rollout", ""]);
    content = drawRollout(here);
  } else if (here.kind === "policy") { crumbs.push([`Policy ${here.name}`, ""]); content = drawPolicy(here.name); }
  else if (here.kind === "system") { crumbs.push(["Machine, engines and ledger", ""]); content = drawSystem(); }
  else content = drawOthers();
  drawBar(crumbs);
  const scroll = main.scrollTop;
  const kept = new Map([...main.querySelectorAll(".turn")].map(each => [each.dataset.slot, each.querySelector(".sees pre")?.scrollTop]));
  const showing = `${location.hash} ${state.turn} ${state.slot} ${state.full}`;
  main.replaceChildren(h("div", { class: "page" }, content));
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
  try {
    const [system, runs] = await Promise.all([read("api/system"), read("api/runs")]);
    state.system = system; state.runs = runs;
    const here = route();
    if (here.kind === "group") state.group = await read(`api/groups/${encodeURIComponent(here.run)}/${here.number}`);
    if (here.kind === "rollout") {
      const known = state.episode?.run_id === here.id ? state.episode : null;
      const more = await read(`api/rollouts/${encodeURIComponent(here.id)}?after=${known ? known.lines.length : 0}`);
      if (known && more.source === known.source) { known.lines.push(...more.lines); Object.assign(known, { ...more, lines: known.lines }); }
      else state.episode = more;
    }
    const drawn = JSON.stringify([location.hash, system.at > (state.drawnAt ?? 0) + 10 ? system.at : state.drawnAt, system.runs, system.policies,
      system.channels.map(channel => channel.throughput.length), here.kind === "system" ? system.machine.now : 0, state.group, state.episode?.lines.length, state.episode?.state]);
    if (drawn !== state.drawn) { state.drawn = drawn; state.drawnAt = system.at; redraw(); }
  } catch (error) {
    document.getElementById("live").replaceChildren(h("span", { class: "dot gone" }), "cannot reach the monitor");
  }
}

addEventListener("hashchange", () => { state.follow = true; state.slot = null; state.drawn = ""; document.body.classList.remove("open"); redraw(); pull(); });
addEventListener("resize", () => redraw());
document.getElementById("menu").onclick = () => document.body.classList.toggle("open");
pull();
setInterval(pull, 2500);
