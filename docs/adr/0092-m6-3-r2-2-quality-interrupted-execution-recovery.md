# ADR 0092: M6.3-R2.2 quality interrupted-execution recovery

- Status: Accepted
- Date: 2026-09-04
- Milestone: M6.3-R2.2b

## Context

The first frozen R2.2b held-out quality execution started from commit `912b15a`
with release binary SHA-256
`00fac75370364ec6592f5255af9ca3ea4dd8757ee429e08732715debbd0f3dbb`.
The process later disappeared with no active session, no quality TSV, no result
JSON, and no partial quality evidence available for inspection.

No R2.2 quality result was observed. This is an interrupted execution, not a
quality PASS or quality failure.

## Decision

A single recovery execution is allowed using the exact same frozen binary,
quality harness, reference, sentinel artifacts, host, environment inputs, and
quality thresholds. No code, artifact, quantization, or gate may change before
that recovery execution.

The R2.2 performance no-retry rule remains unchanged and applies only after a
valid quality PASS authorizes R2.2c official performance samples.
