# ADR 0056: Preserve planner compute-work provenance

## Status

Accepted for M6.2 review remediation.

## Decision

Every `colibri-lite-planner-request-v1` document must contain a positive
`compute_work.gflop_per_token` value and a non-empty
`compute_work.source`. `clr-cli plan` accepts these through
`--compute-gflop-per-token` and `--compute-work-source`, then copies both into
the emitted result request.

`context_tokens` remains an explicit planner input and must be greater than
zero at both the JSON-schema and CLI boundaries.

## Consequences

The caller-supplied model-work term that determines analytical tok/s is now
auditable and sufficient for replay alongside the hashed profile documents.
Calls without compute-work provenance, or with a zero context, fail before any
planner result is emitted. This retains the existing `analytical-v1` model; it
does not promote its prediction as a measurement or retune it.
