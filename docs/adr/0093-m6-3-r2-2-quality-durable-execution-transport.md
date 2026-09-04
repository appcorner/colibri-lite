# ADR 0093: M6.3-R2.2 quality durable execution transport

- Status: Accepted
- Date: 2026-09-04
- Milestone: M6.3-R2.2b

## Context

ADR 0092 authorized one recovery execution after the original quality process
was interrupted without producing any quality result. That recovery execution
was also interrupted by the command transport: its process disappeared and no
quality TSV, result JSON, partial evidence, panic output, or gate verdict was
persisted.

Therefore neither execution is a valid quality observation. R2.2b remains
undecided, and R2.2c performance remains blocked.

## Decision

A transport-only correction is allowed before another quality execution. The
frozen release binary, harness, F32 reference, sentinel artifacts, host,
environment inputs, and all quality thresholds MUST remain byte-identical to
ADR 0092. No model/runtime code or gate may change.
The corrected execution MUST run through a detached worker that persists
stdout, stderr, and the process exit code to files outside the connector
session. The worker may be launched only after a preflight re-verifies every
frozen input identity and confirms that the quality output path is absent.

A connector disconnect is no longer sufficient to classify the execution as
interrupted. The persisted worker exit code and quality output decide whether
the execution completed. If the worker itself disappears without an exit-code
record and without quality evidence, R2.2b remains undecided and a further
protocol decision is required; no automatic retry is allowed.

This transport correction does not authorize R2.2c. Only a valid frozen quality
PASS may authorize the 108 official performance samples.

## Consequences

The new runner is execution infrastructure only. It does not alter arithmetic,
routing, generation, artifacts, thresholds, or quality assertions. The frozen
binary SHA-256 remains
`00fac75370364ec6592f5255af9ca3ea4dd8757ee429e08732715debbd0f3dbb`.
