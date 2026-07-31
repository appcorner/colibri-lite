# ADR 0050: Planner contract boundaries

## Status

Accepted for M6.2-01.

## Decision

The placement planner uses one versioned JSON-schema document with two
top-level document types: a request and a result. Requests cite hardware and
model profiles by immutable profile ID and SHA-256 document hash, then add
explicit workload and RAM/VRAM budgets. Results embed the request and report
candidate resources, analytical estimate provenance, quality risk, rejections,
and a ranking.

An available analytical estimate must cite measured input records. Missing
inputs remain explicit as an incomplete estimate; they are never represented
as zero performance. Resource rejections carry both calculated requirement and
budget limit. The contract introduces no backend, precision format, GPU path,
or Rust public API.

## Consequences

M6.2-02 can implement a cost model against a stable document boundary without
hard-coded machine results. M6.2-03 through M6.2-05 can add candidate
enumeration, admission, and deterministic ranking without changing profile
provenance. Cross-field validation such as ranking only existing feasible
plans remains implementation work for M6.2-05, because JSON Schema alone
cannot express it portably.
