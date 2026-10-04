// Picking an environment: a select of those the system knows, each by its name, with its `module:name` under the select.

import { useEnvironments } from "../api/queries";
import { pickable } from "../lib/environments";

interface PickerProps {
  value: string;
  onChange: (environment: string) => void;
  /** For a launch: those no launcher alive offers cannot be picked. */
  launching?: boolean;
  /** Environments to list besides those the system knows (the one picked, say). */
  also?: (string | null | undefined)[];
  /** Environments not to list (those another entry has). */
  without?: string[];
  label?: string;
}

/** A select of environments: offered by a launcher alive first, then those none offers (marked so). */
export function EnvironmentPicker({ value, onChange, launching = false, also = [], without = [], label = "environment" }: PickerProps) {
  const { data: known } = useEnvironments();
  const listed = pickable(known ?? [], [...also, value]).filter(each => each.environment === value || !without.includes(each.environment));
  const offered = listed.filter(each => each.offered), others = listed.filter(each => !each.offered);
  const option = (each: (typeof listed)[number], disabled = false) => (
    <option key={each.environment} value={each.environment} disabled={disabled} title={each.environment}>{each.name}</option>
  );
  return (
    <>
      <select value={value} onChange={event => onChange(event.target.value)} aria-label={label} required>
        {value ? null : <option value="" disabled>{known ? "choose" : "reading…"}</option>}
        {offered.length && others.length ? <optgroup label="offered">{offered.map(each => option(each))}</optgroup> : offered.map(each => option(each))}
        {others.length ? <optgroup label="no launcher offers">{others.map(each => option(each, launching))}</optgroup> : null}
      </select>
      {value ? <small className="mono" title={value}>{value}</small> : null}
    </>
  );
}
