// Evals from a checkpoint's point of view: every eval it had, and each suite's score along its line from the base model
// (each version's apart: two versions' scores do not compare).

import { Link } from "react-router-dom";
import { useCheckpointEvals, useEvals, useKnown, usePath } from "../api/queries";
import type { CheckpointEval } from "../api/types";
import { Ago } from "../layout/runs";
import { clock, figure, percent } from "../lib/format";
import { evalPlace, runPlace, subjectPlace, suitePlace } from "../lib/places";
import { readable } from "../lib/environments";
import { pathChart } from "../lib/scores";
import { versionsOf, versionTag } from "../lib/suites";
import { LineChart, Sized } from "./charts";
import { Card, Legend, Table } from "./ui";

/** Each suite's score at every checkpoint on a checkpoint's line, from the base model; nothing where none was evaluated. */
export function PathCard({ checkpoint }: { checkpoint: string }) {
  const { data: path } = usePath(checkpoint);
  if (!path?.points.length) return null;
  const chart = pathChart(path);
  if (!chart.series.length) return null;
  const deepest = Math.max(...path.points.map(point => point.depth));
  return (
    <Card title="Scores along its line">
      <Sized>{width => (
        <LineChart series={chart.series} width={width} height={240} y={{ zero: false }} marks={chart.marks} domain={[0, deepest]} dots
          label="each suite's score at each checkpoint on the line" xFormat={depth => chart.labels.get(depth) ?? String(depth)}
          format={chart.solved ? percent : figure} yTick={chart.solved ? percent : undefined} />
      )}</Sized>
      <Legend items={chart.series.map(each => ({ name: `${each.suite}${chart.solved ? "" : each.measure === "solved" ? " (solved)" : " (mean reward)"}`, color: each.color }))} />
    </Card>
  );
}

/** Each environment's share solved (or mean reward) of an eval of several, one a line. */
function EntryFigures({ entries, solved = false }: { entries: NonNullable<CheckpointEval["entries"]>; solved?: boolean }) {
  return (
    <span className="stacked">
      {entries.map(each => (
        <span key={each.environment ?? ""} title={each.environment ?? ""}>
          <small className="faint">{readable(each.environment)}</small> {solved ? (each.share == null ? "–" : percent(each.share)) : figure(each.reward)}
        </span>
      ))}
    </span>
  );
}

/** Every eval a checkpoint had, by hand or by its run's schedule, newest first: one opens the eval; `history` opens
 * them all on the evals page. */
export function CheckpointEvalsCard({ checkpoint }: { checkpoint: string }) {
  const { data } = useCheckpointEvals(checkpoint);
  const { data: suites } = useEvals();
  const known = useKnown();
  const several = new Set((suites?.suites ?? []).filter(each => versionsOf(each).length > 1).map(each => each.suite));
  const evals = data?.evals ?? [];
  return (
    <Card title="Evals" note={evals.length ? <Link to={subjectPlace("checkpoint", checkpoint)} className="linkish">history</Link> : null}>
      {evals.length ? (
        <Table
          heads={[["suite"], ["solved", "n"], ["mean reward", "n"], ["episodes", "n"], ["asked by"], ["started"], ["eval"]]}
          keys={evals.map(each => each.run)}
          rows={evals.map(each => [
            <><Link to={suitePlace(each.suite)} className="linkish" onClick={event => event.stopPropagation()}>{each.suite}</Link>{several.has(each.suite) ? <span className="tag-version">{versionTag(each.version)}</span> : null}</>,
            (each.entries?.length ?? 0) > 1 ? <EntryFigures entries={each.entries!} solved /> : each.share == null ? "–" : <span title={`${each.solved} of ${each.played}`}>{percent(each.share)}</span>,
            (each.entries?.length ?? 0) > 1 ? <EntryFigures entries={each.entries!} /> : figure(each.reward),
            <span title={`${each.episodes} of each start`}>{each.played === each.expected ? each.played : `${each.played} of ${each.expected}`}</span>,
            each.asked_by === "schedule" && each.by ? (
              <Link to={runPlace(each.by)} className="linkish" onClick={event => event.stopPropagation()}>{known.run(each.by)}{each.step != null ? ` · S${each.step}` : ""}</Link>
            ) : "by hand",
            each.started ? <span title={clock(each.started)}><Ago at={each.started} /> ago</span> : "–",
            <b title={`id: ${each.run}`}>{each.name}</b>,
          ])}
          to={evals.map(each => evalPlace(each.run))}
        />
      ) : <p className="muted">None yet.</p>}
    </Card>
  );
}
