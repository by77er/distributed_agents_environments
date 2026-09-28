# Contracts

Status: **Proposed**

Types that cross two or more boundaries are defined here, once. Component docs reference them; they never
redefine them.

| Contract | Crosses |
|---|---|
| [identifiers](identifiers.md) | everything |
| [canonical-content](canonical-content.md) | harness, runners, model endpoint, tool bindings, recorder, trajectories |
| [run-events](run-events.md) | runners, trajectory assembler, client event streams |
| [effects](effects.md) | runners, model endpoint, tool bindings, environment layer |
| [model-endpoint](model-endpoint.md) | runners ↔ recorder / direct adapters |

## Evolution rules

Persisted data outlives code. These rules apply to every contract here and every persisted record elsewhere.

1. **Every persisted record carries `schema_version`.** Readers dispatch on it.
2. **Events are forever.** A reader (in particular a durable runner replaying history) MUST handle every version of every event
   type ever committed, for as long as runs containing them can be replayed or exported.
3. **Additive within a major version.** New optional fields and new enum values only. Readers MUST ignore
   unknown fields and MUST treat unknown enum values as a defined fallback (usually "unknown/other").
4. **Unknown fields are preserved** by any component that reads and re-writes a record.
5. **Breaking changes** require a new major version, an ADR, and a period in which writers emit the old version
   while readers accept both.
6. **Hashes are over canonical JSON** ([RFC 8785 JCS](https://www.rfc-editor.org/rfc/rfc8785)) of the fields the
   hash is declared to cover — never over a wire encoding.
7. **Sizes are bounded.** Any content block larger than 64 KiB MUST be moved to object storage and referenced by
   `BlobReference`. Event payloads MUST NOT exceed 256 KiB.
