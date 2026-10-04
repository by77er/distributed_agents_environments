// How a checkpoint is said wherever it appears: by where it came from (the run that made it and its step) and the
// shortest start of its id, with the bookmarks that name it (and `full` for all of a model's weights); the whole id and
// its line are under the pointer.

import { Link } from "react-router-dom";
import { useKnown } from "../api/queries";
import { checkpointPlace } from "../lib/places";

export const Marks = ({ names }: { names: string[] }) => (
  <>{names.map(name => <span key={name} className="chip bookmark" title={`the bookmark ${name}`}>{name}</span>)}</>
);

/** A checkpoint, as `RUN · S31 kpqx` and its bookmarks, opening its page; none is the base model. `bare` leaves out where
 * it came from (where that is said beside it already). */
export function CheckpointTag({ id, base, bare = false, link = true }: { id: string | null | undefined; base?: string | null; bare?: boolean; link?: boolean }) {
  const known = useKnown();
  if (!id) return <span className="checkpoint-tag" title={base ?? undefined}>{base ? `the base model ${base.split("/").at(-1)}` : "the base model"}</span>;
  const inside = known.checkpoint(id) !== undefined;
  const body = (
    <>
      {bare || !inside ? null : <span className="origin">{known.origin(id)}</span>}
      <b className="mono">{known.short(id)}</b>
    </>
  );
  return (
    <span className="checkpoint-tag">
      {link && inside ? <Link to={checkpointPlace(id)} title={known.title(id)}>{body}</Link> : <span title={known.title(id)}>{body}</span>}
      <WeightsChip id={id} />
      <Marks names={known.bookmarks(id)} />
    </span>
  );
}

/** What a checkpoint's weights are, where they are not an adapter over a model: `full` weights, or an adapter `over full`
 * weights (a checkpoint of its own). */
function WeightsChip({ id }: { id: string | null | undefined }) {
  const known = useKnown();
  const checkpoint = known.checkpoint(id);
  if (checkpoint?.kind === "full") return <span className="chip" title="all of a model's weights">full</span>;
  const under = known.checkpoint(checkpoint?.base);
  return under ? <span className="chip" title={`a LoRA adapter over ${known.base(checkpoint?.base)}`}>over full</span> : null;
}

/** What a checkpoint's weights build on: a checkpoint this ledger has (as a tag), else the model's name (its last part,
 * with `short`). */
export function BaseName({ base, short = false }: { base: string | null | undefined; short?: boolean }) {
  const known = useKnown();
  if (base && known.checkpoint(base)) return <CheckpointTag id={base} />;
  return <span title={base ?? undefined}>{base ? (short ? base.split("/").at(-1) : base) : "the base model"}</span>;
}
