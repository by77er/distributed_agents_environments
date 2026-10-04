# Contracts

Code: `rollout.contracts` · See [API reference](../../../guide/reference.md#rolloutcontracts)

Types that cross two or more layers are defined once, in `rollout.contracts`. Their fields are in the
[API reference](../../../guide/reference.md#rolloutcontracts). These pages say what the fields do not: what an
identifier means, what a digest covers, which events a run records and when, and what a model endpoint guarantees.

| Page | Covers | Crosses |
|---|---|---|
| [identifiers](identifiers.md) | the identifiers with a structure, and who makes them | everything |
| [canonical content](canonical-content.md) | messages, blocks, tool specifications and results, digests | harness, runners, model endpoints, tool sets, the recorder |
| [effects](effects.md) | the operations that leave a run's code, their identity and how they complete | runners, model endpoints, tool sets, environment services |
| [run events](run-events.md) | the record of what happened in a run | runners, episode runners, hooks, clients |
| [model endpoint](model-endpoint.md) | what serves a model slot | runners, the recorder, direct adapters |

## What every contract type has in common

Every contract type is a [`ContractModel`](../../../guide/reference.md#contractmodel):

- **Immutable.** A value is never modified; code builds a new one. Sequence fields accept any sequence and are
  stored as tuples.
- **Unknown fields are kept.** A component that reads a record and writes it again keeps the fields it does not
  know, so a record written by other code passes through unchanged.
- **Digests are over canonical JSON**, never over a wire encoding ([digests](canonical-content.md#digests)).
- **Closed catalogs.** The kinds of effect, the types of run event and the classes of failure are enumerations. A
  new one is a change to `rollout.contracts`.
